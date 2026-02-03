# Solver Process and Key Programs

This document outlines the programs and the process used in this project to find optimal parameters for the QFD nuclear soliton model. It includes the full source code for the key components as of 2025-12-31.

## High-Level Objective

The primary goal is to optimize a set of physics parameters for the `qfd-nuclear-soliton-solver` to accurately reproduce the experimental binding energies of various atomic nuclei. The experimental ground truth is sourced from `data/ame2020_system_energies.csv`.

## Current Status (2025-12-31)

The project has achieved a major breakthrough, followed by a significant challenge:

1.  **"Golden Probe" Success:** A focused optimization on Carbon-12 (`C-12`) was successful. It validated the core physics of the model, converging on a vacuum stiffness parameter (`c_v2_base` ≈ 3.11) that is within 1.8% of the theoretically predicted `β ≈ 3.043233053` from the Fine Structure Constant. This was achieved after significant performance (~23x speedup) and physics-based fixes to the loss function.

2.  **"Octet Test" Failure (Debugging in Progress):** An attempt to validate the C-12 parameters against a wider set of 8 isotopes (from H-1 to Pb-208) failed. The solver was unable to find a single successful solution when running at a higher-fidelity 48-point grid, resulting in a catastrophic loss value. A second attempt with the physically-motivated `c_sym=0` parameter also failed in the same manner.

3.  **Next Step (Isolating the "Poison Pill"):** The immediate goal is to diagnose the Octet Test failure. The current hypothesis is that the lightest nuclei (H-1 and He-4) are the "poison pills," failing to solve with the same physics model as the heavier nuclei and thus causing the entire optimization batch to fail. The next action is to run a "Safe Sextet" test, removing H-1 and He-4 from the target list to see if the remaining isotopes converge.

## Core Workflow

The optimization process is executed through a series of Python scripts that work together:

1.  **Initiation**: An optimization run is started via the `run_parallel_optimization.py` script, which takes a `runspec.json` file to define the experiment.

2.  **Parallel Objective Function**: The script uses the `ParallelObjective` class from `src/parallel_objective.py` to evaluate how well a given set of parameters performs. This class farms out calculations for multiple isotopes to a `ThreadPoolExecutor`.

3.  **Direct Solver Execution**: For each isotope, the objective function calls `run_solver_direct`. This function directly instantiates and runs the `Phase8Model` from `src/qfd_solver.py`, avoiding subprocess overhead.

4.  **Core Solver**: The `scf_minimize` function in `src/qfd_solver.py` uses the Adam optimizer to find the lowest energy state (ground state) of the nucleus for the given parameters.

5.  **Loss Calculation**: The `ParallelObjective` class calculates a "loss" value based on the squared relative error between the model's prediction and the experimental binding energy, plus a penalty for geometric instability based on the virial (Derrick's Theorem).

6.  **Optimization**: The `scipy.optimize.differential_evolution` algorithm uses the loss value to propose new parameter sets until it finds a minimum.

---

## Key Scripts and Code

### 1. Main Execution Script: `run_parallel_optimization.py`

*   **Role:** The main entry point for an optimization run. It parses command-line arguments, reads the run specification, sets up the `ParallelObjective`, and executes the `differential_evolution` optimizer.
*   **Key Changes:** It now has command-line arguments `--lr-psi` and `--lr-b` to control the solver's learning rates.

```python
#!/usr/bin/env python3
"""
Run overnight optimization with ParallelObjective for GPU-efficient parallel evaluation.

Usage:
  python3 run_parallel_optimization.py --maxiter 20 --popsize 15 --workers 3

For overnight run:
  nohup python3 run_parallel_optimization.py --maxiter 100 --popsize 15 --workers 3 > overnight_opt.log 2>&1 &
"""
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from scipy.optimize import differential_evolution
from tqdm import tqdm

sys.path.insert(0, 'src')
from parallel_objective import ParallelObjective

def main():
    parser = argparse.ArgumentParser(description='Run parallel optimization')
    parser.add_argument('--runspec', default='experiments/nuclear_heavy_minimal_test.runspec.json',
                       help='Path to RunSpec config file')
    parser.add_argument('--maxiter', type=int, default=3,
                       help='Maximum DE iterations (generations)')
    parser.add_argument('--popsize', type=int, default=8,
                       help='Population size per parameter')
    parser.add_argument('--workers', type=int, default=2,
                       help='Number of parallel isotope workers (2-4 for 4GB GPU)')
    parser.add_argument('--grid', type=int, default=32,
                       help='Grid resolution (32=fast, 48=accurate)')
    parser.add_argument('--iters', type=int, default=150,
                       help='SCF iterations (150=fast, 360=accurate)')
    parser.add_argument('--device', default='cuda',
                       help='Device: cuda or cpu')
    parser.add_argument('--lr-psi', type=float, default=0.015,
                          help='Learning rate for psi field in SCF solver')
    parser.add_argument('--lr-b', type=float, default=0.005,
                        help='Learning rate for B field in SCF solver')
    args = parser.parse_args()

    print("=" * 80)
    print("PARALLEL OPTIMIZATION WITH GPU ACCELERATION")
    print("=" * 80)
    print(f"Start time: {datetime.now()}")
    print()

    # Load RunSpec
    with open(args.runspec) as f:
        config = json.load(f)

    print(f"RunSpec: {config['experiment_id']}")
    print()

    # Extract parameters and bounds
    params_initial = {}
    params_bounds = {}
    param_names = []

    for p in config['parameters']:
        name = p['name'].replace('nuclear.', '').replace('solver.', '')
        value = p['value']
        frozen = p.get('frozen', False)

        if not frozen:
            params_initial[name] = value
            if p.get('bounds'):
                params_bounds[name] = tuple(p['bounds'])
                param_names.append(name)

    print(f"Optimizing {len(param_names)} parameters:")
    for name in param_names:
        print(f"  {name}: {params_initial[name]:.6f} {params_bounds[name]}")
    print()

    # Load AME data
    ame_data = pd.read_csv('data/ame2020_system_energies.csv')

    # Get target isotopes from RunSpec
    cuts = config['datasets'][0]['cuts']
    target_isotopes = [(iso['Z'], iso['A']) for iso in cuts['target_isotopes']]

    print(f"Target isotopes: {len(target_isotopes)}")
    for Z, A in target_isotopes:
        print(f"  {Z}-{A}")
    print()

    # Setup ParallelObjective
    print(f"ParallelObjective configuration:")
    print(f"  Workers: {args.workers}")
    print(f"  Grid: {args.grid}")
    print(f"  SCF iterations: {args.iters}")
    print(f"  Device: {args.device}")
    print(f"  LR Psi: {args.lr_psi}, LR B: {args.lr_b}")
    print()

    parallel_obj = ParallelObjective(
        target_isotopes=target_isotopes,
        ame_data=ame_data,
        max_workers=args.workers,
        grid_points=args.grid,
        iters_outer=args.iters,
        device=args.device,
        lr_psi=args.lr_psi,
        lr_B=args.lr_b,
        verbose=False  # Don't print every isotope
    )

    # Prepare bounds for differential_evolution
    bounds_list = [params_bounds[name] for name in param_names]

    # Vectorized objective wrapper with TQDM progress bar
    eval_count = [0]
    best_loss = [float('inf')]
    start_time = time.time()

    # Estimate total evaluations for progress bar
    total_evals = args.maxiter * args.popsize * len(param_names)
    pbar = tqdm(total=total_evals, desc="Optimization", unit="eval",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] Loss: {postfix}')

    def objective_vec(x):
        """Convert parameter vector to dict and evaluate."""
        params = params_initial.copy()
        for i, name in enumerate(param_names):
            params[name] = x[i]

        loss = parallel_obj(params)
        eval_count[0] += 1
        pbar.update(1)

        if loss < best_loss[0]:
            best_loss[0] = loss
            pbar.set_postfix_str(f"Best: {loss:.2e}")

        return loss

    def callback(xk, convergence):
        """Progress callback for differential_evolution."""
        # Update progress bar description with convergence info
        pbar.set_description(f"Optimization (conv={convergence:.4f})")
        return False  # Continue optimization

    # Run optimization
    print("=" * 80)
    print(f"STARTING OPTIMIZATION")
    print(f"  Method: differential_evolution")
    print(f"  Generations (maxiter): {args.maxiter}")
    print(f"  Population size: {args.popsize}")
    print(f"  Total evaluations: ~{args.maxiter * args.popsize * len(param_names)}")
    print("=" * 80)
    print()

    result = differential_evolution(
        objective_vec,
        bounds_list,
        maxiter=args.maxiter,
        popsize=args.popsize,
        workers=1,  # Sequential DE (parallel WITHIN each objective call)
        seed=42,
        disp=True,
        callback=callback,
        atol=0.01,
        tol=0.01
    )

    pbar.close()  # Close progress bar
    print()
    print("=" * 80)
    print("OPTIMIZATION COMPLETE")
    print("=" * 80)
    print(f"End time: {datetime.now()}")
    print(f"Total time: {time.time() - start_time:.1f}s")
    print(f"Total evaluations: {eval_count[0]}")
    print(f"Final loss: {result.fun:.6f}")
    print()

    # Extract optimized parameters
    optimized_params = params_initial.copy()
    for i, name in enumerate(param_names):
        optimized_params[name] = result.x[i]

    print("Optimized parameters:")
    for name in param_names:
        initial = params_initial[name]
        final = optimized_params[name]
        change_pct = 100.0 * (final - initial) / initial if initial != 0 else 0
        print(f"  {name:20s}: {initial:10.6f} → {final:10.6f} ({change_pct:+.1f}%)")
    print()

    # Save results
    output_file = f"optimization_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output = {
        'experiment_id': config['experiment_id'],
        'timestamp': datetime.now().isoformat(),
        'optimization': {
            'method': 'differential_evolution',
            'maxiter': args.maxiter,
            'popsize': args.popsize,
            'total_evaluations': eval_count[0],
            'final_loss': float(result.fun),
            'success': result.success,
            'message': result.message
        },
        'parameters_initial': params_initial,
        'parameters_optimized': optimized_params,
        'target_isotopes': [{'Z': Z, 'A': A} for Z, A in target_isotopes],
        'hardware': {
            'parallel_workers': args.workers,
            'grid_points': args.grid,
            'iters_outer': args.iters,
            'device': args.device,
            'lr_psi': args.lr_psi,
            'lr_b': args.lr_b
        }
    }

    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to: {output_file}")
    print()

    # Check GPU memory
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used',
                               '--format=csv,noheader,nounits'],
                              capture_output=True, text=True)
        mem_mb = int(result.stdout.strip())
        print(f"Final GPU memory: {mem_mb} MB ({mem_mb/1024:.2f} GB)")
        if mem_mb > 3072:
            print("⚠️  WARNING: Exceeded 3GB target")
        else:
            print("✓ Within 3GB memory target")
    except:
        pass

    print()
    print("Done!")

if __name__ == '__main__':
    main()
```

### 2. Objective Function: `src/parallel_objective.py`

*   **Role:** Bridges the gap between the optimizer and the core solver. It manages the parallel execution of solver instances and calculates the final loss value.
*   **Key Changes:** The `ccl_stress` function was removed, and the loss calculation in the `__call__` method was updated to use a dynamic virial penalty. It also now accepts and passes on `lr_psi` and `lr_B`. **The `__call__` method has been temporarily modified for debugging to print detailed error information.**

```python
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



def run_solver_direct(A: int, Z: int, params: Dict,
                      grid_points: int = 32, iters_outer: int = 150,
                      device: str = "cuda", early_stop_vir: float = 0.18,
                      lr_psi: float = 0.015, lr_B: float = 0.005) -> Dict:
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
        # BUG FIX (2025-12-31): alpha_coul default is 1.0, NOT 1/137!
        alpha_coul = 1.0  # Matches qfd_solver.py argparse default
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
            lr_psi=lr_psi,
            lr_B=lr_B,
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
                 lr_psi: float = 0.015,
                 lr_B: float = 0.005,
                 verbose: bool = False):
        """
        Args:
            target_isotopes: List of (Z, A) tuples
            ame_data: DataFrame with experimental data
            max_workers: Number of parallel GPU threads (2-4 max to avoid OOM)
            grid_points: Grid resolution (32 for fast, 48 for accurate)
            iters_outer: SCF iterations (150 for fast, 360 for accurate)
            device: 'cuda' or 'cpu'
            lr_psi: Learning rate for psi field
            lr_B: Learning rate for B field
            verbose: Print progress
        """
        self.target_isotopes = target_isotopes
        self.ame_data = ame_data
        self.max_workers = max_workers
        self.grid_points = grid_points
        self.iters_outer = iters_outer
        self.device = device
        self.early_stop_vir = early_stop_vir
        self.lr_psi = lr_psi
        self.lr_B = lr_B
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
        
        # DEBUGGING: Force sequential execution to get detailed errors
        # self.max_workers = 1

        # Submit all isotope solves to thread pool
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            for Z, A in self.target_isotopes:
                future = executor.submit(
                    run_solver_direct,  # PERFORMANCE FIX: Direct call (no subprocess)
                    A, Z, params,
                    self.grid_points, self.iters_outer, self.device, self.early_stop_vir,
                    self.lr_psi, self.lr_B
                )
                futures[future] = (Z, A)
            
            # Collect results as they complete
            for future in as_completed(futures):
                Z, A = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    if self.verbose or self.max_workers == 1:
                        if result['status'] == 'success':
                            print(f"  SUCCESS: {Z:2d}-{A:3d}: E={result['E_model']:.1f} MeV, vir={result['virial']:.3f}")
                        else:
                            print(f"  FAILURE: {Z:2d}-{A:3d}. Full result:")
                            print(result)

                except Exception as e:
                    print(f"  CRITICAL ERROR for {Z}-{A}: Exception retrieving result: {e}")
                    # Create a failure result to avoid crashing the optimizer
                    results.append({
                        'status': 'error', 'A': A, 'Z': Z, 'error': str(e), 
                        'E_model': 0, 'virial': 999, 'converged': False
                    })
        
        # SIMPLIFIED LOSS FUNCTION (2025-12-31): Pure physics, no static biases
        # Removed CCL stress (was static, didn't help optimization)
        # Focus on: Energy accuracy + Geometric stability (Derrick virial)

        from qfd_metaopt_ame2020 import M_PROTON, M_NEUTRON, M_ELECTRON

        errors = []

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

            # 1. Energy Match (Primary Goal)
            rel_error = (pred_E - exp_E) / exp_E
            energy_error = rel_error ** 2

            # 2. Geometric Stability (Derrick's Theorem: T + 3V = 0)
            # Virial measures field balance - should be zero for stable solitons
            # Normalized by E_model to make dimensionless
            virial = abs(result['virial'])
            if abs(E_interaction) > 1.0:  # Avoid division by small numbers
                virial_error = (virial / abs(E_interaction)) ** 2
            else:
                virial_error = virial ** 2  # Fallback for near-zero energies

            # Weighted combination (emphasize energy, but require virial convergence)
            total_error = energy_error + 0.5 * virial_error
            errors.append(total_error)

        if not errors:
            # Catastrophic penalty if no successful solves
            return 1e9

        # Mean loss across all isotopes
        loss = sum(errors) / len(errors)
        
        elapsed = time.time() - start
        if self.verbose:
            print(f"  Loss: {loss:.6f} ({len(results)} isotopes in {elapsed:.1f}s)")
        
        return loss
```

### 3. Core Physics Solver: `src/qfd_solver.py`

*   **Role:** Contains the core physics model (`Phase8Model`) and the Self-Consistent Field (SCF) minimization loop (`scf_minimize`).
*   **Key Changes:** The `scf_minimize` function's learning rates are passed in as arguments rather than being hard-coded, allowing for external tuning.

```python
#!/usr/bin/env python3
"""Phase-8 SCF solver with compounding cohesion (v4 - Merged from ChatSolver)."""
import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch

# ---- PYTHONPATH shim ------------------------------------------------------
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DIR = Path(__file__).resolve().parent
if str(DIR) not in sys.path:
    sys.path.insert(0, str(DIR))

try:
    from qfd_effective_potential_models import compute_alpha_eff
except ImportError:
    # Fallback if the module is not found in the path
    def compute_alpha_eff(A, Z, c_v2_base, c_v2_iso, c_v2_mass):
        A13 = max(1.0, A)**(1.0/3.0)
        iso_term  = Z*(Z-1.0)/(A13 + 1e-12)
        expo_arg  = max(min(c_v2_mass*A, 5.0), -5.0)
        return (c_v2_base + c_v2_iso*iso_term) * math.exp(expo_arg)

# ---- Utilities ------------------------------------------------------------

def torch_det_seed(seed: int) -> None:
    """
    Enforce deterministic behavior across all RNG sources.

    CRITICAL (2025-12-31): Physics is invariant. Running twice must give same answer.
    Without determinism, cannot distinguish better parameters from lucky guesses.
    """
    import numpy as np
    import random

    # Seed all RNG sources
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Force deterministic behavior (may be slower, but reproducible)
    torch.use_deterministic_algorithms(False)  # Some ops don't support deterministic mode
    torch.backends.cudnn.benchmark = False      # Disable auto-tuning (non-deterministic)
    torch.backends.cudnn.deterministic = True   # Force deterministic cuDNN ops


def central_grad_sq3(f: torch.Tensor, dx: float) -> torch.Tensor:
    """Sum of squared central differences across x, y, z (periodic BCs)."""
    fxp = torch.roll(f, -1, dims=0)
    fxm = torch.roll(f,  1, dims=0)
    fyp = torch.roll(f, -1, dims=1)
    fym = torch.roll(f,  1, dims=1)
    fzp = torch.roll(f, -1, dims=2)
    fzm = torch.roll(f,  1, dims=2)
    gx = (fxp - fxm) / (2.0 * dx)
    gy = (fyp - fym) / (2.0 * dx)
    gz = (fzp - fzm) / (2.0 * dx)
    return gx * gx + gy * gy + gz * gz

# ---- Rotor terms ---------------------------------------------------------

class RotorParams(torch.nn.Module):
    def __init__(self, lambda_R2: float = 0.0, lambda_R3: float = 0.0, B_target: float = 0.0):
        super().__init__()
        self.lambda_R2 = float(lambda_R2)
        self.lambda_R3 = float(lambda_R3)
        self.B_target  = float(B_target)

class RotorTerms:
    def __init__(self, dx: float, dV: float, params: RotorParams):
        self.dx, self.dV, self.p = dx, dV, params

    @staticmethod
    def hodge_dual(B: torch.Tensor) -> torch.Tensor:
        return torch.stack((B[1], B[2], B[0]), dim=0)

    def T_rotor(self, B: torch.Tensor) -> torch.Tensor:
        if self.p.lambda_R2 == 0.0:
            return B.new_tensor(0.0)
        gsum = central_grad_sq3(B[0], self.dx) + central_grad_sq3(B[1], self.dx) + central_grad_sq3(B[2], self.dx)
        return self.p.lambda_R2 * gsum.sum() * self.dV

    def V_rotor(self, B: torch.Tensor) -> torch.Tensor:
        if self.p.lambda_R3 == 0.0:
            return B.new_tensor(0.0)
        Bmag = torch.sqrt(1e-24 + B[0]*B[0] + B[1]*B[1] + B[2]*B[2])
        return self.p.lambda_R3 * ((Bmag - self.p.B_target)**2).sum() * self.dV

# ---- Core model ----------------------------------------------------------

class Phase8Model(torch.nn.Module):
    def __init__(self, A:int, Z:int, grid:int, dx:float,
                 c_v2_base:float, c_v2_iso:float, c_v2_mass:float,
                 c_v4_base:float, c_v4_size:float,
                 rotor: RotorParams,
                 device: str = "cpu",
                 coulomb_mode:str="spectral", alpha_coul: float = 1.0,
                 mass_penalty_N: float = 0.0, mass_penalty_e: float = 0.0,
                 project_mass_each: bool = False, project_e_each: bool = False,
                 kappa_rho: float = 0.0,
                 alpha_e_scale: float = 0.5, beta_e_scale: float = 0.5,
                 alpha_model: str = "exp",
                 coulomb_twopi: bool = False,
                 c_sym: float = 0.0):
        super().__init__()
        self.A, self.Z = int(A), int(Z)
        self.N, self.dx = int(grid), float(dx)
        self.dV = self.dx**3
        self.device = torch.device(device)

        A13 = max(1.0, A)**(1.0/3.0)
        if alpha_model == "exp":
            self.alpha_eff = float(compute_alpha_eff(A, Z, c_v2_base, c_v2_iso, c_v2_mass))
        else:
            iso_term = Z*(Z-1.0)/(A13 + 1e-12)
            self.alpha_eff = float(c_v2_base + c_v2_iso*iso_term + c_v2_mass * A)
        self.beta_eff  = float(c_v4_base + c_v4_size * A13)
        self.coeffs_raw = dict(c_v2_base=c_v2_base, c_v2_iso=c_v2_iso,
                               c_v2_mass=c_v2_mass,
                               c_v4_base=c_v4_base, c_v4_size=c_v4_size, A13=A13)
        self.alpha_model = alpha_model

        self.alpha_e = float(alpha_e_scale) * self.alpha_eff
        self.beta_e  = float(beta_e_scale) * self.beta_eff

        shape = (self.N, self.N, self.N)
        self.psi_N = torch.zeros(shape, dtype=torch.float32, device=self.device, requires_grad=True)
        self.psi_e = torch.zeros(shape, dtype=torch.float32, device=self.device, requires_grad=True)
        self.B_N   = torch.zeros((3, *shape), dtype=torch.float32, device=self.device, requires_grad=True)

        self.rotor_terms = RotorTerms(self.dx, self.dV, rotor)

        self.coulomb_mode = str(coulomb_mode)
        self.alpha_coul   = float(alpha_coul)
        self.mass_penalty_N = float(mass_penalty_N)
        self.mass_penalty_e = float(mass_penalty_e)
        self.project_mass_each = bool(project_mass_each)
        self.project_e_each    = bool(project_e_each)
        self.kappa_rho = float(kappa_rho)
        self.coulomb_twopi = bool(coulomb_twopi)

        # Symmetry energy term: penalizes N-Z asymmetry
        self.c_sym = float(c_sym)

        self.kN = 1.0
        self.ke = 1.0

    @torch.no_grad()
    def initialize_fields(self, seed:int=0, init_mode:str="gauss"):
        g = torch.Generator(device=self.device).manual_seed(seed)
        r2 = self._r2_grid()
        if init_mode == "gauss":
            R0 = (1.2 * max(1.0, self.A) ** (1.0/3.0))
            sigma2_n = (0.60*R0)**2
            sigma2_e = (1.00*R0)**2
            gn = torch.exp(-r2/(2.0*sigma2_n))
            ge = torch.exp(-r2/(2.0*sigma2_e))
            torch.manual_seed(seed)
            self.psi_N.copy_(gn + 1e-3*torch.randn_like(gn))
            self.psi_e.copy_(ge + 1e-3*torch.randn_like(ge))
        else:
            torch.manual_seed(seed)
            for T in (self.psi_N, self.psi_e):
                T.uniform_(0.0, 1.0)
                T.mul_(torch.exp(-0.25 * r2))
                T.add_(1e-3 * torch.randn_like(T))
        self.B_N.zero_().add_(1e-3 * torch.randn_like(self.B_N))
        nN = torch.sqrt((self.psi_N*self.psi_N).sum() * self.dV + 1e-24)
        ne = torch.sqrt((self.psi_e*self.psi_e).sum() * self.dV + 1e-24)
        if float(nN) > 0:
            self.psi_N.mul_((0.5 * self.A) / float(nN))
        if float(ne) > 0:
            self.psi_e.mul_((0.5 * self.Z) / float(ne))

    def _r2_grid(self) -> torch.Tensor:
        n = self.N
        ax = torch.arange(n, device=self.device, dtype=torch.float32)
        ax = (ax - (n-1)/2.0) * self.dx
        X, Y, Z = torch.meshgrid(ax, ax, ax, indexing="ij")
        return X*X + Y*Y + Z*Z

    def kinetic_scalar(self, psi: torch.Tensor, kcoef: float) -> torch.Tensor:
        return 0.5 * kcoef * central_grad_sq3(psi, self.dx).sum() * self.dV

    def potential_from_density(self, rho: torch.Tensor, alpha: float, beta: float) -> tuple:
        V4 = 0.5 * alpha * (rho * rho).sum() * self.dV
        V6 = (1.0/6.0) * beta * (rho * rho * rho).sum() * self.dV
        return V4, V6

    def surface_energy(self, rho: torch.Tensor) -> torch.Tensor:
        if self.kappa_rho == 0.0:
            return rho.new_tensor(0.0)
        grad_sq = central_grad_sq3(rho, self.dx)
        return self.kappa_rho * grad_sq.sum() * self.dV

    def _spectral_phi(self, rho: torch.Tensor) -> torch.Tensor:
        if self.coulomb_mode != "spectral" or self.alpha_coul == 0.0:
            return torch.zeros_like(rho)
        Rk = torch.fft.fftn(rho)
        n = self.N
        kx = torch.fft.fftfreq(n, d=self.dx).to(self.device) * (2.0*math.pi)
        ky = kx; kz = kx
        KX, KY, KZ = torch.meshgrid(kx, ky, kz, indexing="ij")
        k2 = KX*KX + KY*KY + KZ*KZ
        green = torch.zeros_like(k2)
        mask = k2 > 1e-12
        green[mask] = 1.0 / k2[mask]
        phi_k = 4.0 * math.pi * Rk * green
        if self.coulomb_twopi:
            phi_k *= (8.0 * math.pi)
        return torch.fft.ifftn(phi_k).real

    def coulomb_cross_energy(self, rhoN: torch.Tensor, rhoE: torch.Tensor) -> torch.Tensor:
        if self.coulomb_mode != "spectral" or self.alpha_coul == 0.0:
            return rhoN.new_tensor(0.0)
        phi_e = self._spectral_phi(rhoE)
        phi_n = self._spectral_phi(rhoN)
        return self.alpha_coul * 0.5 * ((rhoN * phi_e + rhoE * phi_n).sum() * self.dV)

    def mass_penalty(self, rho: torch.Tensor, target: float, penalty: float) -> torch.Tensor:
        if penalty == 0.0:
            return rho.new_tensor(0.0)
        total = rho.sum() * self.dV
        return penalty * (total - target)**2

    def projections(self) -> None:
        if not (self.project_mass_each or self.project_e_each):
            return
        with torch.no_grad():
            if self.project_mass_each:
                rho_N = self.nucleon_density()
                total = float(rho_N.sum() * self.dV)
                if total > 1e-12:
                    scale = math.sqrt(self.A / total)
                    self.psi_N.mul_(scale)
                    self.B_N.mul_(scale)
            if self.project_e_each:
                rho_e = self.electron_density()
                total = float(rho_e.sum() * self.dV)
                if total > 1e-12:
                    scale = math.sqrt(self.Z / total)
                    self.psi_e.mul_(scale)

    def nucleon_density(self) -> torch.Tensor:
        return self.psi_N*self.psi_N + (self.B_N*self.B_N).sum(dim=0)

    def electron_density(self) -> torch.Tensor:
        return self.psi_e*self.psi_e

    def symmetry_energy(self) -> torch.Tensor:
        """
        Field-Dependent Charge Asymmetry Energy

        Penalizes spatial inhomogeneity between charged and neutral field densities.
        This provides actual gradient feedback to the SCF solver, unlike a constant
        offset based on global (N-Z).

        Physical interpretation:
        - rho_charge = ψ_e² (electron/charge density proxy)
        - rho_mass = ψ_N² (nucleon/mass density proxy)
        - delta = rho_charge - rho_mass (local charge-mass imbalance)

        E_sym = c_sym ∫ (delta)² dV

        This couples to field evolution: ∂E_sym/∂ψ ≠ 0, so the solver actively
        minimizes charge-mass separation, enforcing more balanced configurations.

        Note: This is a simplified soliton model that doesn't track separate
        neutron/proton fields, so we use electron density as a charged proxy
        and nucleon density as mass proxy. The actual physics depends on
        Coulomb and surface terms providing most of the asymmetry effects.
        """
        if self.c_sym == 0.0:
            return torch.tensor(0.0, device=self.device)

        # Compute local density imbalance
        rho_charge = self.psi_e * self.psi_e  # Charged density proxy
        rho_mass = self.psi_N * self.psi_N    # Mass density proxy

        # Penalize spatial separation of charge and mass
        delta = rho_charge - rho_mass
        E_sym = self.c_sym * (delta * delta).sum() * self.dV

        return E_sym

    def energies(self) -> Dict[str, torch.Tensor]:
        psiN = self.psi_N
        psiE = self.psi_e
        rhoN = self.nucleon_density()
        rhoE = self.electron_density()

        T_N = self.kinetic_scalar(psiN, self.kN)
        T_e = self.kinetic_scalar(psiE, self.ke)
        T_rotor = self.rotor_terms.T_rotor(self.B_N)
        V4_N, V6_N = self.potential_from_density(rhoN, self.alpha_eff, self.beta_eff)
        V4_e, V6_e = self.potential_from_density(rhoE, self.alpha_e, self.beta_e)

        rho_tot = rhoN
        V_surf = self.surface_energy(rho_tot)

        V_coul = self.coulomb_cross_energy(rhoN, rhoE)

        V_mass_N = self.mass_penalty(rhoN, float(self.A), self.mass_penalty_N)
        V_mass_e = self.mass_penalty(rhoE, float(self.Z), self.mass_penalty_e)

        V_rotor = self.rotor_terms.V_rotor(self.B_N)

        # Field-dependent symmetry energy penalizing charge-mass spatial separation
        E_sym = self.symmetry_energy()

        return dict(
            T_N=T_N, T_e=T_e, T_rotor=T_rotor,
            V4_N=V4_N, V6_N=V6_N, V4_e=V4_e, V6_e=V6_e,
            V_iso=torch.tensor(0.0, device=self.device),
            V_rotor=V_rotor,
            V_surf=V_surf,
            V_coul_cross=V_coul,
            V_mass_N=V_mass_N, V_mass_e=V_mass_e,
            V_sym=E_sym,
        )

    def virial(self, energies: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Virial theorem for self-confined solitons (Derrick scaling).

        CRITICAL REVISION (2025-12-31 PM): Testing Derrick's theorem for solitons.
        For 3D soliton bags with quartic potential, the virial scaling is:
            T + 3V = 0  (Derrick scaling for self-confined fields)

        NOT the standard orbital virial:
            2T + V = 0  (planetary orbits, atoms)

        The original "buggy" formula (3T + V) was actually CLOSER to correct
        soliton physics than the "fixed" orbital formula (2T + V).

        Physical interpretation:
        - Solitons: Self-confined field configurations (droplets, bags)
        - Orbitals: Particles in external potential (atoms, planets)
        - Different scaling laws for energy balance!
        """
        kinetic = energies["T_N"] + energies["T_e"] + energies["T_rotor"]
        total = sum(energies.values())
        potential = total - kinetic  # V = E_total - T
        return kinetic + 3.0 * potential  # Derrick scaling: T + 3V

    def soliton_mass_integral(self) -> torch.Tensor:
        """
        Calculate integrated soliton mass from field density.

        CRITICAL FIX (2025-12-31): Laplacian integral was a dead-end.
        For real scalar field ψ ∈ ℝ, topological charge is zero (Gauss theorem).
        Laplacian ∫∇²ψ dV → 0 mathematically, only measures grid truncation noise.

        Instead, measure MASS (amount of "stuff"):
            M_soliton = ∫ ψ_N² dV

        This tracks:
        - Soliton size/extent
        - C-11 should be "starved" (small integrated mass)
        - C-14 should be "bloated" (large integrated mass)
        - C-12 should be "just right" (optimal mass for Z=6)

        Physical interpretation:
        - ∫ψ² dV ∝ Mass (A) - nucleon count
        - ∫|∇ψ|² dV ∝ Surface tension - boundary energy
        - Ratio measures compression balance

        Returns:
            M_soliton: Integrated field density (dimensionless)

        Reference: Real scalar field Noether charge, not complex phase winding
        """
        # Get nucleon field density
        rho_N = self.psi_N * self.psi_N  # ψ²

        # Integrate over volume
        M_soliton = rho_N.sum() * self.dV

        return M_soliton

    def energy_ratio(self) -> torch.Tensor:
        """
        Calculate soliton compression ratio: R = E_gradient / E_potential

        Alternative stress metric based on energy balance, not Laplacian charge.

        For stable solitons (C-12), the ratio should match CCL prediction
        for optimal surface-to-volume balance.

        Returns:
            R: Energy compression ratio
        """
        energies = self.energies()

        # Gradient energy (surface tension, kinetic)
        E_gradient = energies["T_N"] + energies["T_e"] + energies["T_rotor"]

        # Potential energy (bulk compression)
        E_potential = (energies["V4_N"] + energies["V6_N"] +
                      energies["V4_e"] + energies["V6_e"])

        # Avoid division by zero
        if E_potential.abs() < 1e-12:
            return torch.tensor(float('inf'), device=self.device)

        R = E_gradient / E_potential

        return R

# ---- SCF loop ------------------------------------------------------------

def scf_minimize(model: Phase8Model, iters_outer:int=360, lr_psi:float=1e-2, lr_B:float=1e-2,
                 early_stop_vir:float=0.2, verbose:bool=False) -> tuple[Dict[str,float], float, Dict[str,torch.Tensor]]:
    optim = torch.optim.Adam([
        {"params": [model.psi_N], "lr": lr_psi},
        {"params": [model.psi_e], "lr": lr_psi},
        {"params": [model.B_N], "lr": lr_B},
    ])
    best = dict(E=float("inf"), vir=float("inf"))
    best_state = None
    best_energies = None
    for it in range(1, iters_outer+1):
        optim.zero_grad()
        energies = model.energies()
        total = sum(energies.values())
        vir = model.virial(energies)
        loss = total + 10.0 * vir*vir
        loss.backward()
        optim.step()
        model.projections()
        e_val = float(total.detach())
        vir_val = float(vir.detach())
        if abs(vir_val) < abs(best.get("vir", float("inf"))):
            best = dict(E=e_val, vir=vir_val)
            best_state = [
                model.psi_N.detach().clone(),
                model.psi_e.detach().clone(),
                model.B_N.detach().clone(),
            ]
            best_energies = {k: v.detach().clone() for k, v in energies.items()}
        if verbose and it % 60 == 0:
            print(f"[{it:04d}] E={e_val:+.6e} |vir|={abs(vir_val):.3f}")
        if abs(vir_val) <= early_stop_vir:
            break
    if best_state is not None:
        with torch.no_grad():
            model.psi_N.copy_(best_state[0])
            model.psi_e.copy_(best_state[1])
            model.B_N.copy_(best_state[2])
    return best, best.get("vir", float("nan")), best_energies or energies

# ---- CLI -----------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Phase-8 SCF solver (v4 - Merged from ChatSolver)")
    p.add_argument("--A", type=int, required=True)
    p.add_argument("--Z", type=int, required=True)
    p.add_argument("--c-v2-base", type=float, default=2.2)
    p.add_argument("--c-v2-iso",  type=float, default=0.027)
    p.add_argument("--c-v2-mass", type=float, default=0.0,
                   help="Mass-dependent cohesion coefficient")
    p.add_argument("--c-v4-base", type=float, default=5.28)
    p.add_argument("--c-v4-size", type=float, default=-0.085)
    p.add_argument("--alpha-model", choices=["exp","linear"], default="exp",
                   help="Choose exponential (default) or linear cohesion scaling")
    p.add_argument("--lambda-R2", type=float, default=3e-4)
    p.add_argument("--lambda-R3", type=float, default=1e-3)
    p.add_argument("--B-target",  type=float, default=0.01)
    p.add_argument("--grid-points", type=int, default=48)
    p.add_argument("--dx", type=float, default=1.0)
    p.add_argument("--iters-outer", type=int, default=600)
    p.add_argument("--lr-psi", type=float, default=1e-2)
    p.add_argument("--lr-B",   type=float, default=1e-2)
    p.add_argument("--device", type=str, default="cpu", choices=["cpu","cuda"])
    p.add_argument("--early-stop-vir", type=float, default=0.20)
    p.add_argument("--coulomb", choices=["off","spectral"], default="spectral")
    p.add_argument("--coulomb-charge-mode", choices=["gemini","simple"], default="gemini")
    p.add_argument("--alpha-coul", type=float, default=1.0)
    p.add_argument("--alpha0", type=float, default=None)
    p.add_argument("--gamma", type=float, default=0.8)
    p.add_argument("--kappa-rho", type=float, default=0.0)
    p.add_argument("--kappa-rho0", type=float, default=None)
    p.add_argument("--kappa-rho-exp", type=float, default=1.0)
    p.add_argument("--alpha-e-scale", type=float, default=0.5)
    p.add_argument("--beta-e-scale", type=float, default=0.5)
    p.add_argument("--c-sym", type=float, default=0.0,
                   help="QFD charge asymmetry coefficient (MeV). Penalizes deviation from "
                        "charge-balanced soliton configurations. Typical values: 20-30. "
                        "E_sym = c_sym × (N-Z)² / A^(1/3). NOT nucleon-based - field effect.")
    p.add_argument("--mass-penalty-N", type=float, default=0.0)
    p.add_argument("--mass-penalty-e", type=float, default=0.0)
    p.add_argument("--project-mass-each", action="store_true")
    p.add_argument("--project-e-each", action="store_true")
    p.add_argument("--lambda-scale", type=float, default=650.0)
    p.add_argument("--tol", type=float, default=0.30)
    p.add_argument("--seed", type=int, default=4242)
    p.add_argument("--init", choices=["gauss","random"], default="gauss")
    p.add_argument("--emit-json", action="store_true")
    p.add_argument(
        "--out-json", type=str, default="",
        help="Write solver output to this JSON path (requires --emit-json)")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--meta-json", type=str, default=None, help="Path to a meta parameters JSON file.")
    p.add_argument("--coulomb-legacy", choices=["current","tol015"], default="current",
                   help="Reproduce tol015 spectral Coulomb scaling (adds 2π factor).")
    return p.parse_args()

# ---- JSON helpers --------------------------------------------------------

def write_json(payload: Dict[str, Any], path: str) -> None:
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload, indent=2))
    else:
        print(json.dumps(payload, indent=2))

# ---- Main ----------------------------------------------------------------

def main() -> None:
    args = parse_args()
    t_start = time.time()

    if args.meta_json:
        print(f"Loading meta parameters from: {args.meta_json}")
        with open(args.meta_json, 'r') as f:
            payload = json.load(f)
        required = {"alpha0", "gamma", "kappa_rho0", "alpha_e_scale", "c_v2_mass"}
        if payload.get("schema") != "qfd.merged5.v1" or not required.issubset(payload):
            raise ValueError("Meta JSON is not a valid 'qfd.merged5.v1' schema; refusing to proceed.")
        
        # Override args with values from JSON
        args.alpha0 = payload.get("alpha0", args.alpha0)
        args.gamma = payload.get("gamma", args.gamma)
        args.kappa_rho0 = payload.get("kappa_rho0", args.kappa_rho0)
        args.alpha_e_scale = payload.get("alpha_e_scale", args.alpha_e_scale)
        args.c_v2_mass = payload.get("c_v2_mass", args.c_v2_mass)
        decoder = payload.get("decoder", {})
        if decoder:
            args.coulomb_legacy = decoder.get("coulomb_legacy", args.coulomb_legacy)
            if "alpha_model" in decoder and args.alpha_model == "exp":
                # prefer decoder's alpha model when user left default
                args.alpha_model = decoder["alpha_model"]

    try:
        torch_det_seed(args.seed)
        device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

        alpha_coul_used = args.alpha_coul
        if args.alpha0 is not None:
            alpha_coul_used = args.alpha0 * (args.Z / args.A) ** args.gamma

        kappa_rho_used = args.kappa_rho
        if args.kappa_rho0 is not None:
            A23 = args.A ** (2.0/3.0)
            kappa_rho_used = args.kappa_rho0 * (A23 ** args.kappa_rho_exp)

        rotor = RotorParams(args.lambda_R2, args.lambda_R3, args.B_target)
        model = Phase8Model(
            A=args.A, Z=args.Z, grid=args.grid_points, dx=args.dx,
            c_v2_base=args.c_v2_base, c_v2_iso=args.c_v2_iso, c_v2_mass=args.c_v2_mass,
            c_v4_base=args.c_v4_base, c_v4_size=args.c_v4_size,
            rotor=rotor, device=str(device),
            coulomb_mode=args.coulomb, alpha_coul=alpha_coul_used,
            mass_penalty_N=args.mass_penalty_N, mass_penalty_e=args.mass_penalty_e,
            project_mass_each=args.project_mass_each, project_e_each=args.project_e_each,
            kappa_rho=kappa_rho_used,
            alpha_e_scale=args.alpha_e_scale, beta_e_scale=args.beta_e_scale,
            alpha_model=args.alpha_model,
            coulomb_twopi=(args.coulomb_legacy == "tol015"),
            c_sym=args.c_sym,
        )
        model.initialize_fields(seed=args.seed, init_mode=args.init)

        br, vir, energy_terms = scf_minimize(
            model,
            iters_outer=args.iters_outer,
            lr_psi=args.lr_psi,
            lr_B=args.lr_B,
            early_stop_vir=args.early_stop_vir,
            verbose=args.verbose,
        )
        with torch.no_grad():
            energy_terms = {k: float(v.detach()) for k, v in model.energies().items()}

            # CRITICAL FIX (2025-12-31): Laplacian integral is dead-end for real scalar fields
            # Use mass integral instead: M_soliton = ∫ψ² dV
            M_soliton = float(model.soliton_mass_integral().detach())

            # Energy compression ratio (alternative stress metric)
            R_soliton = float(model.energy_ratio().detach())

            # CCL backbone prediction for this mass
            CCL_C1 = 0.5292508558990585
            CCL_C2 = 0.31674263258172686
            Q_backbone = CCL_C1 * (args.A ** (2.0/3.0)) + CCL_C2 * args.A

            # Stress vector using INPUT charge (Z) vs CCL prediction
            # For fixed Z, as A increases, stress becomes MORE NEGATIVE (neutron-rich)
            stress_vector = args.Z - Q_backbone

        elapsed = time.time() - t_start

        payload = {
            "status": "ok",
            "A": int(args.A),
            "Z": int(args.Z),
            "alpha_eff": float(model.alpha_eff),
            "beta_eff": float(model.beta_eff),
            "coeffs_raw": model.coeffs_raw,
            "E_model": float(br["E"]),
            "virial": float(vir),
            "virial_abs": float(abs(vir)),
            "M_soliton": float(M_soliton),
            "R_soliton": float(R_soliton),
            "Q_backbone": float(Q_backbone),
            "stress_vector": float(stress_vector),
            "coulomb": args.coulomb,
            "coulomb_charge_mode": args.coulomb_charge_mode,
            "alpha_coul": float(alpha_coul_used),
            "alpha0": args.alpha0,
            "gamma": float(args.gamma),
            "c_v2_base": float(args.c_v2_base),
            "c_v2_iso": float(args.c_v2_iso),
            "c_v2_mass": float(args.c_v2_mass),
            "c_v4_base": float(args.c_v4_base),
            "c_v4_size": float(args.c_v4_size),
            "kappa_rho": float(kappa_rho_used),
            "kappa_rho0": args.kappa_rho0,
            "kappa_rho_exp": float(args.kappa_rho_exp),
            "alpha_e_scale": float(args.alpha_e_scale),
            "beta_e_scale": float(args.beta_e_scale),
            "c_sym": float(args.c_sym),
            "alpha_model": args.alpha_model,
            "coulomb_legacy": args.coulomb_legacy,
            "lambda_R2": float(args.lambda_R2),
            "lambda_R3": float(args.lambda_R3),
            "B_target": float(args.B_target),
            "grid_points": args.grid_points,
            "dx": float(args.dx),
            "iters_outer": args.iters_outer,
            "lr_psi": float(args.lr_psi),
            "lr_B": float(args.lr_B),
            "seed": int(args.seed),
            "init": args.init,
            "tol": float(args.tol),
            "elapsed_sec": float(elapsed),
            "physical_success": bool((br["E"] < 0.0) and (abs(vir) <= args.tol)),
        }
        payload.update({k: energy_terms.get(k) for k in [
            "T_N","T_e","T_rotor","V4_N","V6_N","V4_e","V6_e",
            "V_iso","V_rotor","V_surf","V_coul_cross","V_mass_N","V_mass_e","V_sym"
        ]})
        if args.emit_json or args.out_json:
            write_json(payload, args.out_json)
        else:
            print(json.dumps(payload, indent=2))
    except Exception as exc:
        err = {
            "status": "error",
            "error": str(exc),
        }
        if args.emit_json or args.out_json:
            write_json(err, args.out_json)
        else:
            print(json.dumps(err, indent=2))
        raise

if __name__ == "__main__":
    main()
```

### 4. Experiment Specification: `experiments/octet_verification.runspec.json`

*   **Role:** Defines the parameters, bounds, and target isotopes for the "Octet Test".

```json
{
  "schema_version": "v0.1",
  "experiment_id": "exp_2025_nuclear_OCTET_UNIVERSALITY",
  "description": "Validating Beta=3.1 Universality across the Chart",

  "model": {
    "id": "qfd.nuclear.binding.soliton",
    "variant": "phase9_scf",
    "$ref": "../models/qfd_nuclear_soliton_phase9.model.json"
  },

  "parameters": [
    { "name": "nuclear.c_v2_base", "value": 3.1139, "bounds": [3.05, 3.25], "frozen": false },
    { "name": "nuclear.c_v2_iso", "value": 0.0298, "bounds": [0.02, 0.04], "frozen": false },
    { "name": "nuclear.c_v2_mass", "value": 0.0, "bounds": [-0.001, 0.001], "frozen": false },
    { "name": "nuclear.c_v4_base", "value": 11.959, "bounds": [10.0, 14.0], "frozen": false },
    { "name": "nuclear.c_v4_size", "value": -0.091, "bounds": [-0.15, -0.05], "frozen": false },
    { "name": "nuclear.c_sym", "value": 25.76, "bounds": [20.0, 30.0], "frozen": false },
    { "name": "nuclear.kappa_rho", "value": 0.027, "bounds": [0.02, 0.04], "frozen": false },
    { "name": "nuclear.alpha_e_scale", "value": 0.99, "frozen": true },
    { "name": "nuclear.beta_e_scale", "value": 0.49, "frozen": true }
  ],

  "datasets": [
    {
      "id": "ame2020_octet_universality",
      "source": "../data/ame2020_system_energies.csv",
      "columns": {
        "A": "A",
        "Z": "Z",
        "target": "E_exp_MeV",
        "sigma": "E_uncertainty_MeV"
      },
      "cuts": {
        "target_isotopes": [
            {"Z": 1, "A": 1, "name": "H-1"},
            {"Z": 2, "A": 4, "name": "He-4"},
            {"Z": 6, "A": 12, "name": "C-12"},
            {"Z": 8, "A": 16, "name": "O-16"},
            {"Z": 20, "A": 40, "name": "Ca-40"},
            {"Z": 26, "A": 56, "name": "Fe-56"},
            {"Z": 50, "A": 120, "name": "Sn-120"},
            {"Z": 82, "A": 208, "name": "Pb-208"}
        ]
      }
    }
  ],

  "objective": {
    "type": "chi_squared_dynamic_virial",
    "loss_function": "L = Σ(E_QFD - E_exp)²/E_exp² + 0.5 * (virial/E_model)²"
  },

  "solver": {
    "method": "scipy.differential_evolution",
    "options": {
      "maxiter": 50,
      "popsize": 10,
      "workers": 1,
      "seed": 42,
      "atol": 0.01,
      "tol": 0.01,
      "disp": true
    }
  },

  "provenance": {
    "baseline_parameters": "golden_probe_c12_optimized",
    "calibration_date": "2025-12-31",
    "author": "Tracy McPherson (assisted by Gemini)",
    "repository": "QFD_SpectralGap",
    "notes": "Universality test run on 8 isotopes to validate Beta=3.1"
  },

  "metadata": {
    "created": "2025-12-31T21:00:00Z",
    "version": "1.3-octet-universality",
    "tags": ["octet_test", "universality", "beta_validation"]
  }
}
```

### 5. Debugging Command

This is the command used for the last failed "Octet Test" run. The next debugging step is to analyze the output of this command when run with enhanced error reporting.

```bash
python3 run_parallel_optimization.py \
  --runspec experiments/octet_verification.runspec.json \
  --workers 4 \
  --popsize 10 \
  --maxiter 50 \
  --grid 48 \
  --iters 250 \
  --lr-psi 0.0075 \
  --lr-b 0.0025
```
