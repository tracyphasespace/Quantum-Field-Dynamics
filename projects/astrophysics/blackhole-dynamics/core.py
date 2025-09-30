import numpy as np
import torch
from scipy.integrate import solve_ivp, trapezoid
from scipy.interpolate import interp1d
import logging
from typing import Callable, Optional, Union

from config import SimConfig

# Check for torch availability and set up placeholders if not available
try:
    import torch
    from torchdiffeq import odeint as odeint_torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    class nn_Placeholder:
        Module = object
    nn = nn_Placeholder()
    def odeint_torch_placeholder(*args, **kwargs):
        raise NotImplementedError("torchdiffeq not available")
    odeint_torch = odeint_torch_placeholder

class ScalarFieldSolution:
    R_MIN_ODE_CLS = 1e-3
    R_MAX_ODE_CLS = 1e3

    def __init__(self, config: SimConfig, phi_0: float):
        self.config = config
        self.alpha_1 = config.ALPHA_1
        self.alpha_2 = config.ALPHA_2
        self.phi_vac = config.PHI_VAC
        self.phi_0 = phi_0
        self.r_min_param = self.R_MIN_ODE_CLS
        self.r_max_param = self.R_MAX_ODE_CLS

        K = (2.0 * self.alpha_2 / self.alpha_1) * self.phi_vac**2 if self.alpha_1 != 0 else 0.0
        self.R_scale = 1.0 / np.sqrt(K) if K > 1e-12 else 1.0
        self.Phi_scale = self.phi_vac if self.phi_vac > 0.0 else 1.0
        logging.debug(f"ScalarFieldSolution init: R_scale={self.R_scale:.3e}, Phi_scale={self.Phi_scale:.3e}")

        self._solve_successful = False
        self.r_core: Optional[float] = None
        self.r_values: Optional[np.ndarray] = None
        self.phi_values: Optional[np.ndarray] = None
        self.dphi_dr_values: Optional[np.ndarray] = None
        self.mass: Optional[float] = None

        self._phi_interp: Optional[Callable] = None
        self._dphi_dr_interp: Optional[Callable] = None

        self.r_values_gpu: Optional["torch.Tensor"] = None
        self.phi_values_gpu: Optional["torch.Tensor"] = None
        self.dphi_dr_values_gpu: Optional["torch.Tensor"] = None

    def solve(self, r_min_param: float = None, r_max: float = None):
        r_min = r_min_param or self.r_min_param
        r_max = r_max or self.r_max_param
        R_scale, Phi_scale = self.R_scale, self.Phi_scale
        α1, α2, φ_vac, φ0 = self.alpha_1, self.alpha_2, self.phi_vac, self.phi_0
        φ0̂ = φ0 / Phi_scale
        eff_phys = - (2.0 * α2 / α1) * φ0 * (φ0**2 - φ_vac**2) if α1 != 0 else 0.0
        C_phys = eff_phys / 6.0
        Ĉ = C_phys * (R_scale**2 / Phi_scale)
        r_init_phys = max(r_min, 1e-3)
        r̂_init = r_init_phys / R_scale
        r̂_max  = r_max / R_scale
        if r̂_init >= r̂_max:
            logging.error("Invalid dimensionless interval")
            return
        y0̂ = [φ0̂ + Ĉ * r̂_init**2, 2.0 * Ĉ * r̂_init]
        sol = None
        for method in ('Radau','BDF','LSODA'):
            try:
                tmp = solve_ivp(fun=self._field_equation_nd, t_span=(r̂_init, r̂_max), y0=y0̂,
                                method=method, rtol=1e-9, atol=1e-11, dense_output=True)
                if tmp.success:
                    sol = tmp
                    break
            except Exception as e:
                logging.warning(f"{method} exception: {e}")
        if sol is None:
            logging.error("All dimensionless solvers failed.")
            return
        r̂_sol, φ̂_sol, dφ̂_dr_sol = sol.t, sol.y[0], sol.y[1]
        r_phys=r̂_sol*R_scale
        φ_phys=φ̂_sol*Phi_scale
        dφ_dr_phys=dφ̂_dr_sol*(Phi_scale/R_scale)
        r_series=np.logspace(np.log10(r_min),np.log10(r_init_phys),50,endpoint=False)
        φ_series=φ0+C_phys*r_series**2
        dφ_dr_series=2.0*C_phys*r_series
        self.r_values=np.concatenate((r_series,r_phys))
        self.phi_values=np.concatenate((φ_series,φ_phys))
        self.dphi_dr_values=np.concatenate((dφ_dr_series,dφ_dr_phys))
        idx=np.argsort(self.r_values)
        self.r_values,self.phi_values,self.dphi_dr_values = self.r_values[idx],self.phi_values[idx],self.dphi_dr_values[idx]
        self._solve_successful=True
        self.r_core=self._calculate_r_core()
        self._validate_and_process_solution()

    def _field_equation_nd(self, r̂: float, ŷ: np.ndarray):
        φ̂,dφ̂=ŷ
        force=-φ̂*(φ̂**2-1.0)
        if abs(r̂)<1e-15:
            return[0.0,force/3.0]
        d2=force-(2.0/r̂)*dφ̂
        if not np.isfinite(dφ̂)or not np.isfinite(d2):
            return[0.0,0.0]
        return[dφ̂,d2]

    def _validate_and_process_solution(self):
        if(not self._solve_successful or self.r_values is None or self.phi_values is None or self.dphi_dr_values is None):
            return
        if not(len(self.r_values)==len(self.phi_values)==len(self.dphi_dr_values)):
            self._solve_successful=False
            return
        if self.r_core is None:
            self._solve_successful=False
        if self._solve_successful:
            self.mass=self.compute_mass()
            self._cache_interpolators()
            if TORCH_AVAILABLE:
                self._prepare_gpu_data()

    def _calculate_r_core(self)->Union[float,None]:
        if(self.r_values is None or self.phi_values is None or len(self.r_values)<2):
            return None
        try:
            φ_half=self.phi_vac+0.5*(self.phi_0-self.phi_vac)
            finite=np.isfinite(self.phi_values)
            φs=self.phi_values[finite]
            rs=self.r_values[finite]
            if len(φs)<2:
                raise ValueError("Not enough finite φ.")
            i=np.argmin(np.abs(φs-φ_half))
            r_core=float(rs[i])
            if not(np.isfinite(r_core)and r_core>1e-7):
                raise ValueError(f"Bad r_core={r_core}")
            return r_core
        except Exception as e:
            α2s=self.alpha_2 if abs(self.alpha_2)>1e-9 else 1e-9
            φvs=self.phi_vac if abs(self.phi_vac)>1e-9 else 1.0
            est=np.sqrt(abs(self.alpha_1/(2*α2s)))/φvs if α2s!=0 and self.alpha_1!=0 else 1e-7
            return max(est,1e-7)

    def _cache_interpolators(self):
        if(self.r_values is None or self.phi_values is None or len(self.r_values)<2):
            return
        r_u,idx=np.unique(self.r_values,return_index=True)
        φ_u=self.phi_values[idx]
        dφ_u=self.dphi_dr_values[idx]
        kind='cubic'if len(r_u)>=4 else 'linear'
        try:
            self._phi_interp=interp1d(r_u,φ_u,kind=kind,bounds_error=False,fill_value=self.phi_vac)
            self._dphi_dr_interp=interp1d(r_u,dφ_u,kind=kind,bounds_error=False,fill_value=0.0)
        except ValueError as e:
            logging.error(f"Interpolator build failed: {e}")

    def _prepare_gpu_data(self):
        if not TORCH_AVAILABLE:
            return
        if self.r_values is None or self.phi_values is None or self.dphi_dr_values is None:
            return
        try:
            r_u, idx = np.unique(self.r_values, return_index=True)
            phi_u = self.phi_values[idx]
            dphi_dr_u = self.dphi_dr_values[idx]

            self.r_values_gpu = torch.tensor(r_u, dtype=self.config.TORCH_DTYPE, device=self.config.DEVICE)
            self.phi_values_gpu = torch.tensor(phi_u, dtype=self.config.TORCH_DTYPE, device=self.config.DEVICE)
            self.dphi_dr_values_gpu = torch.tensor(dphi_dr_u, dtype=self.config.TORCH_DTYPE, device=self.config.DEVICE)
            logging.debug(f"Prepared GPU data for ScalarFieldSolution (phi_0={self.phi_0:.2f}) on {self.config.DEVICE}")
        except Exception as e:
            logging.error(f"Failed to prepare GPU data: {e}")
            self.r_values_gpu = None
            self.phi_values_gpu = None
            self.dphi_dr_values_gpu = None

    def energy_density(self, r, φ, dφ_dr):
        r_arr, φ_arr, dφ_arr = map(np.asarray, (r, φ, dφ_dr))
        V = 0.5*self.alpha_2*(φ_arr**2 - self.phi_vac**2)**2
        G = 0.5*self.alpha_1*dφ_arr**2
        return G + V

    def compute_mass(self):
        if (self.r_values is None or len(self.r_values)<2):
            return None
        try:
            ρ = self.energy_density(self.r_values,self.phi_values,self.dphi_dr_values)
            ρ = np.nan_to_num(ρ,nan=0.0,posinf=0.0,neginf=0.0)
            integrand = 4*np.pi*self.r_values**2 * ρ
            M = trapezoid(integrand, self.r_values)
            return max(M, 0.0)
        except Exception as e:
            logging.error(f"Mass calculation failed: {e}")
            return None

    def potential(self, r_in: np.ndarray) -> np.ndarray:
        if self._phi_interp is None:
            return np.full_like(np.asarray(r_in), np.nan, dtype=float)
        try:
            phi_vals = self._phi_interp(np.asarray(r_in, dtype=float))
            return self.config.K_M * np.nan_to_num(phi_vals, nan=self.phi_vac)
        except:
            return np.full_like(np.asarray(r_in), np.nan, dtype=float)

    def gradient(self, r_in: np.ndarray) -> np.ndarray:
        if self._dphi_dr_interp is None:
            return np.full_like(np.asarray(r_in), np.nan, dtype=float)
        try:
            dphi_vals = self._dphi_dr_interp(np.asarray(r_in, dtype=float))
            return self.config.K_M * np.nan_to_num(dphi_vals, nan=0.0)
        except:
            return np.full_like(np.asarray(r_in), np.nan, dtype=float)

    def _interpolate_gpu(self, x_query_gpu: "torch.Tensor", x_table_gpu: "torch.Tensor", y_table_gpu: "torch.Tensor", fill_value_left, fill_value_right) -> "torch.Tensor":
        if not TORCH_AVAILABLE:
            raise RuntimeError("Torch not available for _interpolate_gpu")
        if x_table_gpu is None or y_table_gpu is None or x_table_gpu.numel() < 2:
            dtype_to_use = self.config.TORCH_DTYPE if self.config.TORCH_DTYPE is not None else torch.float64
            return torch.full_like(x_query_gpu, fill_value_left if fill_value_left == fill_value_right else float('nan'), device=self.config.DEVICE, dtype=dtype_to_use)

        upper_idx = torch.searchsorted(x_table_gpu, x_query_gpu, right=True)
        left_oob = x_query_gpu < x_table_gpu[0]
        right_oob = x_query_gpu > x_table_gpu[-1]
        idx1 = torch.clamp(upper_idx, 1, x_table_gpu.shape[0] - 1)
        idx0 = idx1 - 1
        x0, x1 = x_table_gpu[idx0], x_table_gpu[idx1]
        y0, y1 = y_table_gpu[idx0], y_table_gpu[idx1]
        denom = x1 - x0
        denom = torch.where(denom < 1e-9, torch.ones_like(denom), denom)
        t = (x_query_gpu - x0) / denom
        y_interp = y0 + t * (y1 - y0)
        y_interp = torch.where(left_oob, torch.full_like(y_interp, fill_value_left), y_interp)
        y_interp = torch.where(right_oob, torch.full_like(y_interp, fill_value_right), y_interp)
        return y_interp

    def potential_gpu(self, r_query_gpu: "torch.Tensor") -> "torch.Tensor":
        if not TORCH_AVAILABLE:
            raise RuntimeError("Torch not available for potential_gpu")
        if self.r_values_gpu is None or self.phi_values_gpu is None:
            logging.warning("GPU data for potential not available. Returning NaNs.")
            dtype_to_use = self.config.TORCH_DTYPE if self.config.TORCH_DTYPE is not None else torch.float64
            return torch.full_like(r_query_gpu, float('nan'), device=self.config.DEVICE, dtype=dtype_to_use)
        
        phi_vals_gpu = self._interpolate_gpu(r_query_gpu, self.r_values_gpu, self.phi_values_gpu,
                                             fill_value_left=self.phi_vac, fill_value_right=self.phi_vac)
        return self.config.K_M * phi_vals_gpu

    def gradient_gpu(self, r_query_gpu: "torch.Tensor") -> "torch.Tensor":
        if not TORCH_AVAILABLE:
            raise RuntimeError("Torch not available for gradient_gpu")
        if self.r_values_gpu is None or self.dphi_dr_values_gpu is None:
            logging.warning("GPU data for gradient not available. Returning NaNs.")
            dtype_to_use = self.config.TORCH_DTYPE if self.config.TORCH_DTYPE is not None else torch.float64
            return torch.full_like(r_query_gpu, float('nan'), device=self.config.DEVICE, dtype=dtype_to_use)

        dphi_vals_gpu = self._interpolate_gpu(r_query_gpu, self.r_values_gpu, self.dphi_dr_values_gpu,
                                              fill_value_left=0.0, fill_value_right=0.0)
        return self.config.K_M * dphi_vals_gpu

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_phi_interp'] = None
        state['_dphi_dr_interp'] = None
        state['r_values_gpu'] = None 
        state['phi_values_gpu'] = None
        state['dphi_dr_values_gpu'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.r_values is not None and hasattr(self, '_solve_successful') and self._solve_successful:
            self._cache_interpolators()
            if TORCH_AVAILABLE:
                self._prepare_gpu_data()

class TwoBodySystem:
    def __init__(self, config: SimConfig, solution_ref: ScalarFieldSolution, M1: float, M2: float):
        self.config = config
        if not isinstance(solution_ref, ScalarFieldSolution) or solution_ref.mass is None or solution_ref.mass <= 0:
            raise ValueError("Reference solution must be a valid ScalarFieldSolution with positive mass.")
        self.solution_ref = solution_ref
        self.M_ref = solution_ref.mass
        self.r_core_ref = solution_ref.r_core if solution_ref.r_core is not None and solution_ref.r_core > 0 else 1.0
        
        self.M1, self.M2 = M1, M2
        self.scale1 = M1 / self.M_ref if self.M_ref != 0 else 0.0
        self.scale2 = M2 / self.M_ref if self.M_ref != 0 else 0.0
        
        self.D: Union[float, None] = None
        self.q2_np: Union[np.ndarray, None] = None
        self.q2_torch: Optional["torch.Tensor"] = None

        self.saddle_point: Union[np.ndarray, None] = None
        self.saddle_energy: Union[float, None] = None
        logging.debug(f"TwoBodySystem: M1={M1:.2e}, M2={M2:.2e} (RefM={self.M_ref:.2e}, RefRc={self.r_core_ref:.2e})")

    def set_separation(self, D_val: Union[float, None]):
        if D_val is None:
            self.D, self.q2_np, self.q2_torch = None, None, None
            return
        self.D = float(abs(D_val)) if D_val !=0 else 1e-9
        self.q2_np = np.array([self.D, 0.0, 0.0])
        if TORCH_AVAILABLE:
            self.q2_torch = torch.tensor(self.q2_np, dtype=self.config.TORCH_DTYPE, device=self.config.DEVICE)

    def update_masses(self, M1: float, M2: float):
        self.M1, self.M2 = M1, M2
        self.scale1 = M1 / self.M_ref if self.M_ref != 0 else 0.0
        self.scale2 = M2 / self.M_ref if self.M_ref != 0 else 0.0
        self.saddle_point, self.saddle_energy = None, None

    def total_potential(self, q_np: np.ndarray) -> float:
        q_arr=np.asarray(q_np)
        q_arr=np.nan_to_num(q_arr)
        epsilon=1e-9
        r1=np.linalg.norm(q_arr)
        r1_safe=max(r1,epsilon)
        V1=self.solution_ref.potential(r1_safe)
        V2=0.0
        if self.q2_np is not None:
            r2=np.linalg.norm(q_arr-self.q2_np)
            r2_safe=max(r2,epsilon)
            V2=self.solution_ref.potential(r2_safe)
        if np.isnan(V1)or np.isnan(V2):
            return 1e20
        return float(self.scale1*V1+self.scale2*V2)

    def potential_gradient(self, q_np: np.ndarray) -> np.ndarray:
        q_arr=np.asarray(q_np)
        q_arr=np.nan_to_num(q_arr)
        epsilon=1e-9
        r1=np.linalg.norm(q_arr)
        grad1=np.zeros(3)
        if r1>=epsilon:
            dV1_dr=self.solution_ref.gradient(r1)
            if not np.isnan(dV1_dr):
                grad1=self.scale1*dV1_dr*(q_arr/r1)
        grad2=np.zeros(3)
        if self.q2_np is not None:
            vec_q_q2=q_arr-self.q2_np
            r2=np.linalg.norm(vec_q_q2)
            if r2>=epsilon:
                dV2_dr=self.solution_ref.gradient(r2)
                if not np.isnan(dV2_dr):
                    grad2=self.scale2*dV2_dr*(vec_q_q2/r2)
        return np.nan_to_num(grad1+grad2,nan=0.0,posinf=1e20,neginf=-1e20)

    def total_potential_gpu(self, q_gpu: "torch.Tensor") -> "torch.Tensor":
        if not TORCH_AVAILABLE:
            raise RuntimeError("Torch not available for total_potential_gpu")
        
        is_batched = q_gpu.ndim == 2
        current_q_gpu = q_gpu if is_batched else q_gpu.unsqueeze(0)
        dtype_to_use = self.config.TORCH_DTYPE if self.config.TORCH_DTYPE is not None else torch.float64
        epsilon_gpu = torch.tensor(1e-9, dtype=dtype_to_use, device=self.config.DEVICE)

        r1_gpu = torch.linalg.norm(current_q_gpu, dim=1)
        r1_safe_gpu = torch.maximum(r1_gpu, epsilon_gpu)
        V1_gpu = self.solution_ref.potential_gpu(r1_safe_gpu)

        V2_gpu = torch.zeros_like(V1_gpu, device=self.config.DEVICE)
        if self.q2_torch is not None:
            r2_gpu = torch.linalg.norm(current_q_gpu - self.q2_torch.unsqueeze(0), dim=1)
            r2_safe_gpu = torch.maximum(r2_gpu, epsilon_gpu)
            V2_gpu = self.solution_ref.potential_gpu(r2_safe_gpu)
        
        fill_V1 = torch.tensor(self.solution_ref.phi_vac * self.config.K_M, device=self.config.DEVICE, dtype=dtype_to_use)
        fill_V2 = torch.tensor(self.solution_ref.phi_vac * self.config.K_M, device=self.config.DEVICE, dtype=dtype_to_use)

        V1_gpu = torch.where(torch.isnan(V1_gpu), fill_V1, V1_gpu)
        V2_gpu = torch.where(torch.isnan(V2_gpu), fill_V2, V2_gpu)

        total_V_gpu = self.scale1 * V1_gpu + self.scale2 * V2_gpu
        
        penalty_val = torch.tensor(1e20, dtype=dtype_to_use, device=self.config.DEVICE)
        total_V_gpu = torch.where(torch.isnan(total_V_gpu), penalty_val, total_V_gpu)

        return total_V_gpu if is_batched else total_V_gpu.squeeze(0)

    def potential_gradient_gpu(self, q_gpu: "torch.Tensor") -> "torch.Tensor":
        if not TORCH_AVAILABLE:
            raise RuntimeError("Torch not available for potential_gradient_gpu")

        is_batched = q_gpu.ndim == 2
        current_q_gpu = q_gpu if is_batched else q_gpu.unsqueeze(0)
        dtype_to_use = self.config.TORCH_DTYPE if self.config.TORCH_DTYPE is not None else torch.float64
        epsilon_gpu = torch.tensor(1e-9, dtype=dtype_to_use, device=self.config.DEVICE)

        r1_gpu = torch.linalg.norm(current_q_gpu, dim=1)
        grad1_gpu = torch.zeros_like(current_q_gpu, device=self.config.DEVICE)
        valid_r1_mask = r1_gpu >= epsilon_gpu
        if torch.any(valid_r1_mask):
            r1_safe_gpu_masked = r1_gpu[valid_r1_mask]
            q_masked_r1 = current_q_gpu[valid_r1_mask]
            dV1_dr_gpu_masked = self.solution_ref.gradient_gpu(r1_safe_gpu_masked)
            dV1_dr_gpu_masked = torch.nan_to_num(dV1_dr_gpu_masked, nan=0.0)
            term1_masked = self.scale1 * dV1_dr_gpu_masked.unsqueeze(1) * (q_masked_r1 / r1_safe_gpu_masked.unsqueeze(1))
            grad1_gpu[valid_r1_mask] = term1_masked

        grad2_gpu = torch.zeros_like(current_q_gpu, device=self.config.DEVICE)
        if self.q2_torch is not None:
            vec_q_q2_gpu = current_q_gpu - self.q2_torch.unsqueeze(0)
            r2_gpu = torch.linalg.norm(vec_q_q2_gpu, dim=1)
            valid_r2_mask = r2_gpu >= epsilon_gpu
            if torch.any(valid_r2_mask):
                r2_safe_gpu_masked = r2_gpu[valid_r2_mask]
                vec_q_q2_masked = vec_q_q2_gpu[valid_r2_mask]
                dV2_dr_gpu_masked = self.solution_ref.gradient_gpu(r2_safe_gpu_masked)
                dV2_dr_gpu_masked = torch.nan_to_num(dV2_dr_gpu_masked, nan=0.0)
                term2_masked = self.scale2 * dV2_dr_gpu_masked.unsqueeze(1) * (vec_q_q2_masked / r2_safe_gpu_masked.unsqueeze(1))
                grad2_gpu[valid_r2_mask] = term2_masked
        
        total_grad_gpu = grad1_gpu + grad2_gpu
        total_grad_gpu = torch.nan_to_num(total_grad_gpu, nan=0.0, posinf=1e20, neginf=-1e20)
        return total_grad_gpu if is_batched else total_grad_gpu.squeeze(0)

    def find_saddle_point(self, num_grid_points=500, fit_fraction=0.05, use_gpu=False):
        if self.D is None or self.D <= 1e-9:
            return None, None
        if self.M1 <=0 or self.M2 <= 0:
            logging.warning("Non-positive mass for saddle search.")
        x_grid_min = 0.001 * self.D
        x_grid_max = 0.999 * self.D
        if x_grid_min >= x_grid_max:
            x_grid_min = 0.0
            x_grid_max = self.D

        if use_gpu and TORCH_AVAILABLE and self.solution_ref.r_values_gpu is not None:
            logging.debug(f"Searching L1 saddle for D={self.D:.3e} using GPU grid...")
            dtype_to_use = self.config.TORCH_DTYPE if self.config.TORCH_DTYPE is not None else torch.float64
            xs_torch = torch.linspace(x_grid_min, x_grid_max, num_grid_points, dtype=dtype_to_use, device=self.config.DEVICE)
            q_grid_torch = torch.zeros((num_grid_points, 3), dtype=dtype_to_use, device=self.config.DEVICE)
            q_grid_torch[:, 0] = xs_torch
            Vtot_grid_torch = self.total_potential_gpu(q_grid_torch)
            Vtot_grid_cpu = Vtot_grid_torch.cpu().numpy()
            xs_cpu = xs_torch.cpu().numpy()
            if not np.all(np.isfinite(Vtot_grid_cpu)):
                logging.error(f"find_saddle_point_gpu: Non-finite Vtot_grid D={self.D:.3e}.")
                return None,None
            idx_max = np.nanargmax(Vtot_grid_cpu)
            x0_coarse, V0_coarse = xs_cpu[idx_max], Vtot_grid_cpu[idx_max]
        else:
            logging.debug(f"Searching L1 saddle for D={self.D:.3e} using CPU grid...")
            xs_cpu = np.linspace(x_grid_min, x_grid_max, num_grid_points)
            Vtot_grid_cpu = np.array([self.total_potential(np.array([x,0.0,0.0])) for x in xs_cpu])
            if not np.all(np.isfinite(Vtot_grid_cpu)):
                logging.error(f"find_saddle_point_cpu: Non-finite Vtot_grid D={self.D:.3e}.")
                return None,None
            idx_max = np.nanargmax(Vtot_grid_cpu)
            x0_coarse, V0_coarse = xs_cpu[idx_max], Vtot_grid_cpu[idx_max]
        
        fit_delta = self.D * fit_fraction / 2.0
        fit_x_min = max(x_grid_min, x0_coarse - fit_delta)
        fit_x_max = min(x_grid_max, x0_coarse + fit_delta)
        x_saddle, E_saddle = x0_coarse, V0_coarse
        if fit_x_min < fit_x_max:
            num_fit_points = 100
            x_fit_cpu = np.linspace(fit_x_min, fit_x_max, num_fit_points)
            if use_gpu and TORCH_AVAILABLE and self.solution_ref.r_values_gpu is not None:
                dtype_to_use_fit = self.config.TORCH_DTYPE if self.config.TORCH_DTYPE is not None else torch.float64
                x_fit_torch = torch.tensor(x_fit_cpu, dtype=dtype_to_use_fit, device=self.config.DEVICE)
                q_fit_torch = torch.zeros((num_fit_points, 3), dtype=dtype_to_use_fit, device=self.config.DEVICE)
                q_fit_torch[:, 0] = x_fit_torch
                V_fit_torch = self.total_potential_gpu(q_fit_torch)
                V_fit_cpu = V_fit_torch.cpu().numpy()
            else:
                V_fit_cpu = np.array([self.total_potential(np.array([x,0.0,0.0])) for x in x_fit_cpu])
            if np.all(np.isfinite(V_fit_cpu)):
                try:
                    coeffs = np.polyfit(x_fit_cpu, V_fit_cpu, 2)
                    a,b,_ = coeffs
                    if abs(a) > 1e-12:
                        x_saddle_fitted = -b/(2*a)
                        if x_grid_min < x_saddle_fitted < x_grid_max:
                            x_saddle = x_saddle_fitted
                            E_saddle = np.polyval(coeffs, x_saddle)
                except (np.linalg.LinAlgError, ValueError) as e_fit:
                    logging.warning(f"Polyfit failed: {e_fit}. Using coarse.")
        self.saddle_point = np.array([x_saddle,0.0,0.0])
        self.saddle_energy = float(E_saddle)
        logging.debug(f"Found L1 saddle: {self.saddle_point}, E={self.saddle_energy:.4e} (method: {'GPU' if use_gpu and TORCH_AVAILABLE else 'CPU'})")
        return self.saddle_point, self.saddle_energy

    def analyze_saddle_vs_separation(self, D_values:np.ndarray, use_gpu_search=False):
        s_pos,s_E=[],[]
        for D_val in np.asarray(D_values):
            self.set_separation(D_val)
            pos,energy=self.find_saddle_point(use_gpu=use_gpu_search)
            s_pos.append(pos if pos is not None else np.full(3,np.nan))
            s_E.append(energy if energy is not None else np.nan)
        return np.array(s_pos),np.array(s_E)

    def _compute_hessian(self, q_critical:np.ndarray, h:float=1e-5)->np.ndarray:
        q_c=np.asarray(q_critical)
        hess=np.zeros((3,3))
        for i in range(3):
            q_p=q_c.copy()
            q_p[i]+=h
            q_m=q_c.copy()
            q_m[i]-=h
            g_p=self.potential_gradient(q_p)
            g_m=self.potential_gradient(q_m)
            if np.any(np.isnan(g_p))or np.any(np.isnan(g_m)):
                return np.zeros((3,3))
            hess[i,:]=(g_p-g_m)/(2*h)
        return(hess+hess.T)/2.0

    def _check_hessian_for_saddle(self,hessian:np.ndarray,q_critical:np.ndarray,expected_neg_eigs:int=1)->bool:
        try:
            evals=np.linalg.eigvals(hessian)
        except np.linalg.LinAlgError:
            return False
        tol=1e-9
        n_neg=np.sum(evals<-tol)
        n_pos=np.sum(evals>tol)
        return(n_neg==expected_neg_eigs and n_pos==(3-expected_neg_eigs))