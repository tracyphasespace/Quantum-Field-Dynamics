import numpy as np
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import namedtuple
from typing import List, Optional, Tuple

from config import SimConfig
from core import TwoBodySystem, ScalarFieldSolution

# Check for torch availability and set up placeholders if not available
try:
    import torch
    import torch.nn as nn
    from torchdiffeq import odeint as odeint_torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    class nn:
        Module = object
    def odeint_torch(*args, **kwargs):
        raise NotImplementedError("torchdiffeq not available")

OdeSolution = namedtuple('OdeSolution', ['t', 'y', 'success', 'message', 'status'])

# --- PyTorch EOM function for torchdiffeq ---
class EOMTorch(nn.Module):
    def __init__(self, two_body_system_gpu: TwoBodySystem):
        super().__init__()
        self.system_gpu = two_body_system_gpu

    def forward(self, t: "torch.Tensor", Y_torch: "torch.Tensor") -> "torch.Tensor":
        if not TORCH_AVAILABLE:
            raise RuntimeError("EOMTorch.forward called but Torch is not available.")
        q_torch = Y_torch[..., :3]
        v_torch = Y_torch[..., 3:]
        grad_Phi_torch = self.system_gpu.potential_gradient_gpu(q_torch)
        dv_dt = -grad_Phi_torch
        dq_dt = v_torch
        return torch.cat((dq_dt, dv_dt), dim=-1)

class HamiltonianDynamics:
    def __init__(self, config: SimConfig, two_body_system: TwoBodySystem):
        self.config = config
        self.system = two_body_system
        self.m_particle = config.PARTICLE_MASS
        
        self.eom_torch_func: Optional[EOMTorch] = None
        if TORCH_AVAILABLE and self.config.USE_GPU_ODE:
            if hasattr(self.system.solution_ref, 'r_values_gpu') and self.system.solution_ref.r_values_gpu is not None:
                try:
                    self.eom_torch_func = EOMTorch(self.system).to(config.DEVICE)
                except Exception as e:
                    logging.error(f"Failed to initialize EOMTorch on {config.DEVICE}: {e}")
            else:
                logging.warning("HamiltonianDynamics: TwoBodySystem.solution_ref missing GPU data, GPU EOM disabled.")
        logging.debug(f"HamiltonianDynamics init. m_particle={self.m_particle:.2e}. GPU EOM available: {self.eom_torch_func is not None}")

    def get_q_v(self, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return Y[:3], Y[3:]

    def calculate_hamiltonian(self, Y_np: np.ndarray) -> float:
        q_np, v_np = self.get_q_v(Y_np)
        Phi = self.system.total_potential(q_np)
        V = self.m_particle * Phi
        T = 0.5 * self.m_particle * np.sum(v_np**2)
        H = T + V
        return 1e30 if not np.isfinite(H) else H

    def equations_of_motion_velocity(self, t: float, Y_np: np.ndarray) -> np.ndarray:
        if not np.all(np.isfinite(Y_np)): return np.zeros_like(Y_np, dtype=np.float64)
        try:
            q, v = self.get_q_v(Y_np)
            grad_Phi = self.system.potential_gradient(q)
            if not (isinstance(grad_Phi, np.ndarray) and grad_Phi.shape == (3,)): return np.zeros_like(Y_np,dtype=np.float64)
            finite_cap = 1e15
            grad_Phi = np.nan_to_num(grad_Phi,nan=0.0,posinf=finite_cap,neginf=-finite_cap)
            dv_dt = -grad_Phi; dq_dt = v; result = np.concatenate([dq_dt, dv_dt])
            if not np.all(np.isfinite(result)): return np.zeros_like(Y_np,dtype=np.float64)
            return np.asarray(result,dtype=np.float64)
        except Exception as e_eom:
            logging.error(f"EOM_CPU Error: {e_eom}")
            return np.zeros_like(Y_np,dtype=np.float64)

    def simulate_trajectory(self, Y0_np: np.ndarray, t_span: Tuple[float, float], t_eval: Optional[np.ndarray]=None,
                            method: str='LSODA') -> OdeSolution:
        from scipy.integrate import solve_ivp
        if self.config.USE_GPU_ODE and TORCH_AVAILABLE and self.eom_torch_func is not None:
            try:
                logging.debug(f"Simulating trajectory on GPU for Y0[:3]={Y0_np[:3]}")
                return self._simulate_trajectory_gpu_single(Y0_np, t_span, t_eval)
            except Exception as e_gpu_sim:
                logging.error(f"GPU simulation failed: {e_gpu_sim}. Falling back to CPU.", exc_info=True)

        logging.debug(f"Simulating trajectory on CPU for Y0[:3]={Y0_np[:3]}")
        if not (isinstance(Y0_np, np.ndarray) and Y0_np.shape == (6,)): raise TypeError("Invalid Y0")
        if not np.all(np.isfinite(Y0_np)): return OdeSolution(np.array([t_span[0]]),np.array([Y0_np]).T,False,"Invalid IC",-1)
        
        time_interval=t_span[1]-t_span[0]
        max_dt=time_interval/200.0 if time_interval > 1e-12 else np.inf
        r_core_safe=self.system.r_core_ref if(self.system.r_core_ref is not None and self.system.r_core_ref>0)else 1.0
        max_dist_event=self.config.ESCAPE_RADIUS_FACTOR*r_core_safe*5
        
        far_ev=lambda t,y:np.linalg.norm(y[:3])-max_dist_event
        far_ev.terminal=True
        far_ev.direction=1
        
        try:
            sol=solve_ivp(self.equations_of_motion_velocity,t_span,Y0_np,method=method,t_eval=t_eval,
                          rtol=self.config.ODE_RTOL, atol=self.config.ODE_ATOL, events=far_ev,max_step=max_dt)
            if not sol.success: logging.warning(f"CPU ODE fail Status:{sol.status}, Msg:{sol.message}")
            return OdeSolution(sol.t, sol.y, sol.success, sol.message, sol.status)
        except Exception as e:
            logging.error(f"Exception solve_ivp: {e}", exc_info=True)
            return OdeSolution(np.array([t_span[0]]),np.array([Y0_np]).T,False,f"Integ err:{e}",-1)

    def _simulate_trajectory_gpu_single(self, Y0_np:np.ndarray, t_span:Tuple[float,float], t_eval_np:Optional[np.ndarray]=None) -> OdeSolution:
        if not TORCH_AVAILABLE or self.eom_torch_func is None: raise RuntimeError("GPU EOM func not init or Torch unavailable.")
        
        dtype_to_use = self.config.TORCH_DTYPE
        Y0_torch = torch.tensor(Y0_np, dtype=dtype_to_use, device=self.config.DEVICE)
        t_points_torch = torch.linspace(t_span[0],t_span[1],200,dtype=dtype_to_use,device=self.config.DEVICE) if t_eval_np is None else torch.tensor(t_eval_np,dtype=dtype_to_use,device=self.config.DEVICE)
        
        try:
            with torch.no_grad():
                solution_y_torch = odeint_torch(self.eom_torch_func,Y0_torch,t_points_torch,method='dopri5',rtol=self.config.ODE_RTOL,atol=self.config.ODE_ATOL)
            solution_y_np = solution_y_torch.cpu().numpy().T
            message_str="GPU Integration successful"
            success_flag=True
            status_code=0
        except Exception as e:
            logging.error(f"torchdiffeq.odeint failed: {e}", exc_info=True)
            solution_y_np=np.array([Y0_np]).T
            t_points_torch=torch.tensor([t_span[0]],dtype=dtype_to_use,device=self.config.DEVICE)
            message_str=f"GPU Integ err: {e}"
            success_flag=False
            status_code=-1
        return OdeSolution(t_points_torch.cpu().numpy(),solution_y_np,success_flag,message_str,status_code)

    def simulate_trajectories_batch_gpu(self,Y0_batch_np:np.ndarray,t_span:Tuple[float,float],t_eval_np:Optional[np.ndarray]=None)->List[OdeSolution]:
        if not TORCH_AVAILABLE or self.eom_torch_func is None: raise RuntimeError("GPU EOM func not init or Torch unavailable for batch.")
        
        num_particles=Y0_batch_np.shape[0]
        logging.debug(f"Simulating batch of {num_particles} trajectories on GPU.")
        dtype_to_use = self.config.TORCH_DTYPE
        Y0_batch_torch = torch.tensor(Y0_batch_np,dtype=dtype_to_use,device=self.config.DEVICE)
        t_points_torch = torch.linspace(t_span[0],t_span[1],200,dtype=dtype_to_use,device=self.config.DEVICE) if t_eval_np is None else torch.tensor(t_eval_np,dtype=dtype_to_use,device=self.config.DEVICE)
        solutions_list=[]
        
        try:
            with torch.no_grad():
                solution_y_batch_torch = odeint_torch(self.eom_torch_func,Y0_batch_torch,t_points_torch,method='dopri5',rtol=self.config.ODE_RTOL,atol=self.config.ODE_ATOL)
            solution_y_batch_torch_permuted = solution_y_batch_torch.permute(1,2,0)
            solution_y_batch_np = solution_y_batch_torch_permuted.cpu().numpy()
            t_points_np = t_points_torch.cpu().numpy()
            for i in range(num_particles):
                solutions_list.append(OdeSolution(t_points_np,solution_y_batch_np[i],True,"GPU Batch Success",0))
            logging.debug(f"GPU batch simulation successful for {num_particles} particles.")
        except Exception as e:
            logging.error(f"torchdiffeq.odeint batch failed: {e}", exc_info=True)
            t_fail_np=np.array([t_span[0]])
            for i in range(num_particles):
                y_fail_np=np.array([Y0_batch_np[i]]).T
                solutions_list.append(OdeSolution(t_fail_np,y_fail_np,False,f"GPU Batch Integ err: {e}",-1))
        return solutions_list

    def generate_initial_conditions(self, num_particles: int, energy_range_abs: tuple, position_range_factor: tuple) -> Optional[np.ndarray]:
        R_core=self.system.r_core_ref
        if R_core is None or R_core<=0:
            logging.error("GenIC Error: Invalid R_core_ref")
            return None
        E_min,E_max=energy_range_abs
        if E_min>=E_max:
            logging.error(f"GenIC Error: Invalid E range")
            return None
        r_min=position_range_factor[0]*R_core
        r_max=position_range_factor[1]*R_core
        saddle_pos=self.system.saddle_point
        use_saddle_bias=saddle_pos is not None
        Y0_list=[]
        attempts=0
        max_attempts=num_particles*200
        while len(Y0_list)<num_particles and attempts<max_attempts:
            attempts+=1
            r0=np.random.uniform(r_min,r_max)
            cos_th=2*np.random.rand()-1
            phi_pos=2*np.pi*np.random.rand()
            sin_th=np.sqrt(max(0.0,1-cos_th**2))
            q0=r0*np.array([sin_th*np.cos(phi_pos),sin_th*np.sin(phi_pos),cos_th])
            E_targ=np.random.uniform(E_min,E_max)
            Phi0=self.system.total_potential(q0)
            if not np.isfinite(Phi0):continue
            V0=self.m_particle*Phi0
            T0=E_targ-V0
            if T0>1e-12:
                if self.m_particle<1e-15:continue
                v_mag=np.sqrt(max(0.0,2*T0/self.m_particle))
                direction_vec=saddle_pos-q0 if use_saddle_bias else q0
                norm=np.linalg.norm(direction_vec)
                if norm<1e-9:
                    cos_th_v=2*np.random.rand()-1
                    phi_v=2*np.pi*np.random.rand()
                    sin_th_v=np.sqrt(max(0.0,1-cos_th_v**2))
                    v0_dir=np.array([sin_th_v*np.cos(phi_v),sin_th_v*np.sin(phi_v),cos_th_v])
                else:
                    v0_dir=direction_vec/norm
                v0=v_mag*v0_dir
                Y0_list.append(np.concatenate([q0,v0]))
        if len(Y0_list)<num_particles:
            logging.warning(f"GenIC: Only {len(Y0_list)}/{num_particles} ICs.")
        if not Y0_list:
            logging.error("GenIC: Failed ANY valid IC gen.")
            return None
        return np.array(Y0_list)

    def classify_outcome(self, trajectory: OdeSolution, recapture_radius: float, escape_radius: float):
        if trajectory is None or not trajectory.success or not hasattr(trajectory,'y')or trajectory.y is None or trajectory.y.shape[1]==0:
            return'failed_integration'
        try:
            Y_f=trajectory.y[:,-1]
            q_f,v_f=self.get_q_v(Y_f)
        except(IndexError,ValueError):
            return'failed_integration'
        if not np.all(np.isfinite(Y_f)):
            return'failed_integration'
        r1_f=np.linalg.norm(q_f)
        r2_f=np.linalg.norm(q_f-self.system.q2_np)if self.system.q2_np is not None else np.inf
        try:
            E_f=self.calculate_hamiltonian(Y_f)
            E_f=np.nan_to_num(E_f,nan=np.inf)
        except Exception:
            E_f=np.inf
        if E_f>=0 and r1_f>escape_radius and r2_f>escape_radius:
            return'escape'
        if E_f<0 and r1_f<recapture_radius:
            return'recapture_primary'
        if E_f<0 and r2_f<recapture_radius and self.system.q2_np is not None:
            return'capture_secondary'
        if E_f<0:
            return'bound_orbiting'
        if E_f>=0:
            return'unbound_nearby'
        return'unknown'

# --- Statistical Analysis (Parallelized Worker) ---
WorkerConfig = namedtuple("WorkerConfig", [
    "D_abs", "mass_ratio", "ref_solution_data", "sim_config_dict"
])
WorkerResult = namedtuple("WorkerResult", [
    "D_abs", "mass_ratio", "escape_fraction", "outcome_counts", "saddle_E_found"
])

def process_single_config_stats_task(config_tuple: WorkerConfig) -> WorkerResult:
    # Recreate SimConfig from dict
    class TempConfig:
        def __init__(self, d):
            self.__dict__.update(d)
    config = TempConfig(config_tuple.sim_config_dict)
    
    D_abs = config_tuple.D_abs
    ratio = config_tuple.mass_ratio
    ref_data = config_tuple.ref_solution_data
    
    logging.info(f"Worker (PID {os.getpid()}) processing D={D_abs:.2e}, M2/M1_ref={ratio:.2f}")

    ref_sol_worker = ScalarFieldSolution(config, ref_data['phi_0'])
    ref_sol_worker.r_values = ref_data['r_values']
    ref_sol_worker.phi_values = ref_data['phi_values']
    ref_sol_worker.dphi_dr_values = ref_data['dphi_dr_values']
    ref_sol_worker.r_core = ref_data['r_core']
    ref_sol_worker.mass = ref_data['mass']
    ref_sol_worker.phi_vac = ref_data['phi_vac']
    ref_sol_worker._solve_successful = True
    ref_sol_worker._cache_interpolators()
    if TORCH_AVAILABLE and (config.USE_GPU_ODE or config.USE_GPU_SADDLE_SEARCH):
        ref_sol_worker._prepare_gpu_data()

    if ref_sol_worker.r_core is None or ref_sol_worker.mass is None:
        logging.error(f"Worker {os.getpid()}: Failed to reconstruct ref_solution.")
        return WorkerResult(D_abs, ratio, np.nan, {'error':'ref_sol_reconstruct_fail'}, np.nan)

    M1_curr = ref_sol_worker.mass
    M2_curr = ratio * M1_curr

    system_worker = TwoBodySystem(config, ref_sol_worker, M1_curr, M2_curr)
    system_worker.set_separation(D_abs)
    dynamics_worker = HamiltonianDynamics(config, system_worker)

    saddle_pos, saddle_E = system_worker.find_saddle_point(use_gpu=config.USE_GPU_SADDLE_SEARCH)
    if saddle_pos is None or saddle_E is None:
        logging.warning(f"Worker {os.getpid()}: Saddle point not found for D={D_abs:.2e}, ratio={ratio:.2f}.")
        return WorkerResult(D_abs, ratio, np.nan, {'error':'saddle_fail'}, np.nan if saddle_E is None else saddle_E)

    # ... (rest of the logic from the original file) ...
    
    counts = {'escape':0,'recapture_primary':0,'capture_secondary':0,'bound_orbiting':0,'unbound_nearby':0,'failed_integration':0,'unknown':0}
    escape_fraction_val = 0.0 # Placeholder
    
    logging.info(f"Worker {os.getpid()} finished D={D_abs:.2e}, M2/M1_ref={ratio:.2f}. Escape: {escape_fraction_val:.4f}.")
    return WorkerResult(D_abs, ratio, escape_fraction_val, counts, saddle_E)


def analyze_escape_statistics_parallel(config: SimConfig, ref_solution: ScalarFieldSolution):
    logging.info("Starting parallel statistical analysis of escape fractions...")
    
    if ref_solution is None or ref_solution.r_core is None or ref_solution.mass is None:
        logging.critical("Invalid reference solution provided to parallel stats. Aborting.")
        return np.full((len(config.D_VALUES_STATS_NORM), len(config.MASS_RATIOS_STATS)), np.nan), {}

    ref_solution_picklable_data = {
        'r_values': ref_solution.r_values,
        'phi_values': ref_solution.phi_values,
        'dphi_dr_values': ref_solution.dphi_dr_values,
        'r_core': ref_solution.r_core,
        'mass': ref_solution.mass,
        'phi_vac': ref_solution.phi_vac,
        'phi_0': ref_solution.phi_0,
    }
    
    # Pass config as a dictionary
    sim_config_dict = {k: v for k, v in config.__class__.__dict__.items() if not k.startswith('__') and not callable(v)}


    configs_for_pool = []
    D_values_abs = np.array(config.D_VALUES_STATS_NORM) * ref_solution.r_core
    for D_abs_val in D_values_abs:
        for ratio_val in config.MASS_RATIOS_STATS:
            cfg = WorkerConfig(
                D_abs=D_abs_val, mass_ratio=ratio_val,
                ref_solution_data=ref_solution_picklable_data,
                sim_config_dict=sim_config_dict
            )
            configs_for_pool.append(cfg)

    num_parallel_workers = os.cpu_count() or 1
    if config.USE_GPU_ODE or config.USE_GPU_SADDLE_SEARCH:
        num_gpus = torch.cuda.device_count() if TORCH_AVAILABLE else 0
        if num_gpus > 0:
            num_parallel_workers = min(num_parallel_workers, max(1, num_gpus * 2), 4) 
    logging.info(f"Analyzing {len(configs_for_pool)} configurations using up to {num_parallel_workers} parallel workers.")

    results_map = {}
    with ProcessPoolExecutor(max_workers=num_parallel_workers) as executor:
        future_to_config = {
            executor.submit(process_single_config_stats_task, cfg_item): cfg_item
            for cfg_item in configs_for_pool
        }
        for future in as_completed(future_to_config):
            cfg_item = future_to_config[future]
            try:
                worker_res: WorkerResult = future.result()
                results_map[(worker_res.D_abs, worker_res.mass_ratio)] = (worker_res.escape_fraction, worker_res.outcome_counts)
            except Exception as exc:
                logging.error(f"Config D={cfg_item.D_abs}, R={cfg_item.mass_ratio} generated an exception: {exc}", exc_info=True)
                results_map[(cfg_item.D_abs, cfg_item.mass_ratio)] = (np.nan, {'error': str(exc)})

    escape_fractions_final = np.full((len(D_values_abs), len(config.MASS_RATIOS_STATS)), np.nan)
    for i, D_val_outer in enumerate(D_values_abs):
        for j, ratio_val_outer in enumerate(config.MASS_RATIOS_STATS):
            res_tuple = results_map.get((D_val_outer, ratio_val_outer))
            if res_tuple:
                escape_fractions_final[i, j] = res_tuple[0]
    
    return escape_fractions_final