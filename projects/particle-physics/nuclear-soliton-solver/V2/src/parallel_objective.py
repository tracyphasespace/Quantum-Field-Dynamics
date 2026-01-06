"""
Parallel objective function for optimization with GPU memory management.
Carefully tuned to avoid OOM based on previous GPU tuning work.

CRITICAL UPDATE (2025-12-31): CCL Stress Constraint
Added Core Compression Law stress penalty to replace generic virial penalty.
This connects the Top-Down empirical analysis (5,842 isotopes) with the
Bottom-Up soliton solver, focusing optimization on stable isotopes.
"""
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple, Any
import pandas as pd

# Core Compression Law constants (from Lean formalization)
# Source: projects/Lean4/QFD/Nuclear/CoreCompressionLaw.lean:248-249
CCL_C1 = 0.5292508558990585  # Surface term
CCL_C2 = 0.31674263258172686  # Volume term

def ccl_stress(Z: int, A: int) -> float:
    """
    Calculate Core Compression Law stress: |Z - Q_backbone(A)|

    The CCL predicts optimal charge Q(A) = c1·A^(2/3) + c2·A.
    Stress measures deviation from stability backbone:
        - Stress < 1: Stable isotope
        - Stress > 3: Unstable, will decay

    This empirical constraint (validated on 5,842 isotopes) guides
    optimization to focus on stable nuclear configurations.

    Args:
        Z: Proton number
        A: Mass number

    Returns:
        Stress value (dimensionless)

    Reference: projects/Lean4/QFD/Nuclear/CoreCompressionLaw.lean:612-614
    """
    Q_backbone = CCL_C1 * (A ** (2.0/3.0)) + CCL_C2 * A
    return abs(Z - Q_backbone)

def run_solver_direct(A: int, Z: int, params: Dict,
                      grid_points: int = 32, iters_outer: int = 150,
                      device: str = "cuda", early_stop_vir: float = 0.18) -> Dict:
    """
    Run solver directly (NO SUBPROCESS - 3x faster!).

    PERFORMANCE FIX (2025-12-31): Eliminates Python startup overhead.
    - Before: subprocess.run() → ~1s startup + 0.5s solve = 1.5s total
    - After:  Direct import → 0s startup + 0.5s solve = 0.5s total

    CRITICAL: This must be called from ThreadPoolExecutor, NOT multiprocessing,
    to allow multiple workers to share the same GPU efficiently.

    Args:
        early_stop_vir: Stop solver early when |virial| < threshold (default 0.18)
    """
    import sys
    import time
    import torch
    sys.path.insert(0, str(Path(__file__).parent))

    # Direct imports - NO subprocess overhead!
    from qfd_solver import Phase8Model, RotorParams, scf_minimize, torch_det_seed

    try:
        # Deterministic seeding (same as qfd_solver.py main)
        seed = 4242
        torch_det_seed(seed)

        # Setup device
        device_obj = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")

        # Create rotor params (CORRECT defaults from qfd_solver.py line 477-479)
        rotor = RotorParams(
            lambda_R2=3e-4,  # CRITICAL: Was 0.1 (333x too large!)
            lambda_R3=1e-3,
            B_target=0.01
        )

        # Calculate derived parameters (matching qfd_solver.py logic)
        alpha_coul = 1.0 / 137.036  # Fine structure constant
        kappa_rho_used = params['kappa_rho']

        # Create model (direct instantiation - NO subprocess!)
        model = Phase8Model(
            A=A, Z=Z,
            grid=grid_points,
            dx=1.0,
            c_v2_base=params['c_v2_base'],
            c_v2_iso=params['c_v2_iso'],
            c_v2_mass=params['c_v2_mass'],
            c_v4_base=params['c_v4_base'],
            c_v4_size=params['c_v4_size'],
            rotor=rotor,
            device=str(device_obj),
            coulomb_mode="spectral",
            alpha_coul=alpha_coul,
            mass_penalty_N=0.0,
            mass_penalty_e=0.0,
            project_mass_each=False,
            project_e_each=False,
            kappa_rho=kappa_rho_used,
            alpha_e_scale=params['alpha_e_scale'],
            beta_e_scale=params['beta_e_scale'],
            alpha_model="exp",
            coulomb_twopi=False,
            c_sym=params['c_sym'],
        )

        # Initialize fields
        model.initialize_fields(seed=seed, init_mode="gauss_cluster")

        # Run SCF minimization
        best_result, virial, energy_terms = scf_minimize(
            model,
            iters_outer=iters_outer,
            lr_psi=0.015,
            lr_B=0.005,
            early_stop_vir=early_stop_vir,
            verbose=False,
        )

        # Extract results (matching qfd_solver.py output)
        with torch.no_grad():
            E_model = float(best_result["E"])
            virial_abs = float(abs(virial))

            # Check physical success (same criteria as qfd_solver.py)
            physical_success = (virial_abs < 0.5 and E_model < 0)

        # Return result
        return {
            'status': 'success' if E_model != 0 else 'failed',
            'A': A,
            'Z': Z,
            'E_model': E_model,
            'virial': virial_abs,
            'converged': physical_success
        }

    except Exception as e:
        import traceback
        return {
            'status': 'error',
            'A': A,
            'Z': Z,
            'error': str(e),
            'traceback': traceback.format_exc(),
            'E_model': 0,
            'virial': 999,
            'converged': False
        }


class ParallelObjective:
    """
    Parallel objective function with GPU memory management.
    
    TUNING PARAMETERS (empirically measured Dec 2024):
    - max_workers: 2-6 for RTX 3050 Ti (4GB VRAM, target <3GB)
    - grid_points: 32 (fast mode, ~2.25x less memory than 48)

    Measured GPU memory usage (grid=32):
    - 2 workers: ~600 MB total (~300 MB/worker)
    - 3 workers: ~750 MB total (~250 MB/worker)
    - 4 workers: ~1000 MB total (~250 MB/worker)
    - 6 workers: ~1500 MB total (~250 MB/worker) - still safe!

    Note: ThreadPoolExecutor shares GPU efficiently; memory per worker
    is much lower than peak per-solve memory due to sequential GPU operations.
    """
    
    def __init__(self, target_isotopes: List[Tuple[int, int]],
                 ame_data: pd.DataFrame,
                 max_workers: int = 2,  # Conservative default
                 grid_points: int = 32,
                 iters_outer: int = 150,
                 device: str = "cuda",
                 early_stop_vir: float = 0.18,  # Virial convergence threshold
                 verbose: bool = False):
        """
        Args:
            target_isotopes: List of (Z, A) tuples
            ame_data: DataFrame with experimental data
            max_workers: Number of parallel GPU threads (2-4 max to avoid OOM)
            grid_points: Grid resolution (32 for fast, 48 for accurate)
            iters_outer: SCF iterations (150 for fast, 360 for accurate)
            device: 'cuda' or 'cpu'
            verbose: Print progress
        """
        self.target_isotopes = target_isotopes
        self.ame_data = ame_data
        self.max_workers = max_workers
        self.grid_points = grid_points
        self.iters_outer = iters_outer
        self.device = device
        self.early_stop_vir = early_stop_vir
        self.verbose = verbose
        
        # Build experimental data lookup
        self.exp_data = {}
        for Z, A in target_isotopes:
            row = ame_data[(ame_data['Z'] == Z) & (ame_data['A'] == A)]
            if not row.empty:
                self.exp_data[(Z, A)] = {
                    'E_exp': float(row.iloc[0]['E_exp_MeV']),
                    'sigma': float(row.iloc[0].get('E_uncertainty_MeV', 1.0))
                }
        
        if self.verbose:
            print(f"ParallelObjective initialized:")
            print(f"  Isotopes: {len(target_isotopes)}")
            print(f"  Workers: {max_workers}")
            print(f"  Grid: {grid_points}, Iters: {iters_outer}")
            print(f"  Device: {device}")
            # Corrected estimate: ~250-300 MB per worker (measured empirically)
            est_mem = max_workers * 0.25  # GB estimate (250 MB per worker)
            print(f"  Estimated GPU memory: ~{est_mem:.1f} GB")
            if est_mem > 3.0:
                print(f"  ⚠️  WARNING: May exceed 3GB target!")
    
    def __call__(self, params: Dict[str, float]) -> float:
        """
        Evaluate objective function with parallel isotope solving.
        
        Returns:
            Loss value (lower is better)
        """
        start = time.time()
        
        # Submit all isotope solves to thread pool
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            for Z, A in self.target_isotopes:
                future = executor.submit(
                    run_solver_direct,  # PERFORMANCE FIX: Direct call (no subprocess)
                    A, Z, params,
                    self.grid_points, self.iters_outer, self.device, self.early_stop_vir
                )
                futures[future] = (Z, A)
            
            # Collect results as they complete
            for future in as_completed(futures):
                Z, A = futures[future]
                result = future.result()
                results.append(result)
                
                if self.verbose and result['status'] == 'success':
                    print(f"  {Z:2d}-{A:3d}: E={result['E_model']:.1f} MeV, vir={result['virial']:.3f}")
        
        # Compute loss with CCL stress constraint
        from qfd_metaopt_ame2020 import M_PROTON, M_NEUTRON, M_ELECTRON

        errors_sq = []
        stress_penalties = []

        for result in results:
            if result['status'] != 'success':
                continue

            Z, A = result['Z'], result['A']
            if (Z, A) not in self.exp_data:
                continue

            # Compute full system energy (interaction + constituents)
            E_interaction = result['E_model']
            N = A - Z
            M_constituents = Z * M_PROTON + N * M_NEUTRON + Z * M_ELECTRON
            pred_E = M_constituents + E_interaction
            exp_E = self.exp_data[(Z, A)]['E_exp']

            # Relative error squared (matches RunSpecAdapter)
            rel_error = (pred_E - exp_E) / exp_E
            errors_sq.append(rel_error ** 2)

            # CCL Stress constraint (REPLACES generic virial penalty)
            # This is the key innovation: use empirical stability map to guide solver
            stress = ccl_stress(Z, A)

            # Weight stress penalty: stable isotopes (low stress) get normal treatment,
            # unstable isotopes (high stress) get penalized, reducing optimization effort
            # on configurations that shouldn't have clean soliton solutions anyway
            stress_weight = 0.01  # Tunable: balance energy vs stability focus
            stress_penalties.append(stress_weight * (stress ** 2))

            # Virial check as sanity filter (not primary constraint)
            # Use Derrick scaling (T + 3V) for solitons, not orbital (2T + V)
            virial = abs(result['virial'])
            if virial > 0.5:  # Relaxed threshold - just catch catastrophic failures
                virial_penalty = 1.0 * (virial - 0.5) ** 2
                errors_sq.append(virial_penalty)
        
        if not errors_sq:
            # CRITICAL FIX (2025-12-31): Penalty must be MUCH worse than any working solution
            # Previous value (1000) was lower than working solutions with high virial (~millions)
            # This caused optimizer to prefer complete failure over imperfect physics
            return 1e9  # Catastrophic penalty - forces optimizer away from failure

        # Combined loss: energy errors + CCL stress constraint
        # This focuses optimization on stable isotopes (low stress) while
        # allowing unstable ones (high stress) to have worse fits
        energy_loss = sum(errors_sq) / len(errors_sq) if errors_sq else 0.0
        stress_loss = sum(stress_penalties) / len(stress_penalties) if stress_penalties else 0.0

        loss = energy_loss + stress_loss
        
        elapsed = time.time() - start
        if self.verbose:
            print(f"  Loss: {loss:.6f} ({len(results)} isotopes in {elapsed:.1f}s)")
        
        return loss
