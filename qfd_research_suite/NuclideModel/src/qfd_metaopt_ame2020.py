#!/usr/bin/env python3
"""
QFD Meta-Optimizer: Calibrate Field Parameters Against AME2020 Experimental Data

Optimizes QFD soliton field parameters to match experimental binding energies
from the Atomic Mass Evaluation 2020 (AME2020) dataset.

Target: Minimize L = Σ |BE/A_QFD - BE/A_exp|² across calibration isotope set

Parameters to optimize:
- c_v2_base, c_v2_iso, c_v2_mass (cohesion field coupling)
- c_v4_base, c_v4_size (higher-order field terms)
- alpha_e_scale, beta_e_scale (charged field scaling)
- c_sym (NEW - charge asymmetry energy coefficient)
- kappa_rho (density-dependent coupling)

Fixed parameters (for now):
- Rotor terms: lambda_R2, lambda_R3, B_target
- Grid: 48 points, dx=1.0
- Solver: 200 iterations

Methodology:
1. Select diverse calibration set (30-50 isotopes)
2. Run QFD solver for each isotope with trial parameters
3. Compute BE/A from QFD E_model
4. Calculate loss vs AME2020 experimental BE/A
5. Use optimizer (CMA-ES or Optuna) to find best parameters
"""

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Physical constants
M_PROTON = 938.272  # MeV/c²
M_NEUTRON = 939.565  # MeV/c²
M_ELECTRON = 0.511  # MeV/c²

def load_ame2020_data(csv_path: str) -> pd.DataFrame:
    """Load AME2020 experimental binding energies."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} isotopes from AME2020")
    return df

def select_calibration_set(df: pd.DataFrame, n_isotopes: int = 40) -> pd.DataFrame:
    """
    Select physics-driven calibration set targeting:
    - Magic numbers (Z or N = 2, 8, 20, 28, 50, 82, 126)
    - Doubly magic nuclei (both Z and N magic)
    - Valley of stability backbone
    - Charge asymmetry test cases (e.g., Fe-56 vs Fe-45)
    - Representative elements across periodic table
    """

    # PRIORITY 1: Doubly magic nuclei (shell closures)
    doubly_magic = [
        (2, 4),     # He-4 (Z=2, N=2)
        (8, 16),    # O-16 (Z=8, N=8)
        (20, 40),   # Ca-40 (Z=20, N=20)
        (20, 48),   # Ca-48 (Z=20, N=28) neutron-rich
        (28, 56),   # Ni-56 (Z=28, N=28) - if stable
        (50, 100),  # Sn-100 (Z=50, N=50) - if available
        (82, 208),  # Pb-208 (Z=82, N=126) heaviest stable doubly magic
    ]

    # PRIORITY 2: Single magic (one of Z or N is magic)
    single_magic = [
        (6, 12),    # C-12 (common, light)
        (10, 20),   # Ne-20 (Z=10, N=10)
        (14, 28),   # Si-28 (Z=14, N=14)
        (26, 56),   # Fe-56 (stability peak, N=30 near magic 28)
        (28, 58),   # Ni-58 (Z=28 magic)
        (28, 62),   # Ni-62 (Z=28 magic, highest BE/A)
        (28, 64),   # Ni-64 (Z=28 magic, neutron-rich)
        (50, 120),  # Sn-120 (Z=50 magic)
        (82, 206),  # Pb-206 (Z=82 magic)
        (82, 207),  # Pb-207 (Z=82 magic)
    ]

    # PRIORITY 3: Charge asymmetry tests (N-Z variations)
    charge_asymmetry = [
        (26, 54),   # Fe-54 (charge-rich, N-Z=-2)
        (26, 57),   # Fe-57 (near symmetric, N-Z=+5)
        (26, 58),   # Fe-58 (charge-poor, N-Z=+6)
    ]

    # PRIORITY 4: Light elements (test weak binding)
    light_elements = [
        (1, 2),     # H-2 (deuterium)
        (1, 3),     # H-3 (tritium) if available
        (3, 6),     # Li-6
        (3, 7),     # Li-7
        (4, 9),     # Be-9
        (5, 11),    # B-11
    ]

    # PRIORITY 5: Medium-heavy (transition region)
    medium_heavy = [
        (29, 63),   # Cu-63
        (29, 65),   # Cu-65
        (30, 64),   # Zn-64
        (47, 107),  # Ag-107
        (47, 109),  # Ag-109
    ]

    # PRIORITY 6: Heavy (test compounding saturation)
    heavy = [
        (79, 197),  # Au-197
        (80, 200),  # Hg-200
        (92, 238),  # U-238 (if needed for A>200 test)
    ]

    calibration_set = []

    # Add isotopes in priority order
    all_targets = doubly_magic + single_magic + charge_asymmetry + light_elements + medium_heavy + heavy

    for Z, A in all_targets:
        row = df[(df['Z'] == Z) & (df['A'] == A)]
        if not row.empty:
            calibration_set.append(row.iloc[0])
            if len(calibration_set) >= n_isotopes:
                break

    cal_df = pd.DataFrame(calibration_set).drop_duplicates(subset=['Z', 'A'])

    print(f"\nCalibration set: {len(cal_df)} isotopes (physics-driven)")
    print(f"  Doubly magic: {sum(1 for Z, A in doubly_magic if not df[(df['Z']==Z) & (df['A']==A)].empty)}")
    print(f"  Single magic: {sum(1 for Z, A in single_magic if not df[(df['Z']==Z) & (df['A']==A)].empty)}")
    print(f"  A range: {cal_df['A'].min()} to {cal_df['A'].max()}")
    print(f"  Z range: {cal_df['Z'].min()} to {cal_df['Z'].max()}")

    return cal_df.sort_values('A')

# DELETED: compute_binding_energy_per_A()
#
# DO NOT USE "BINDING ENERGY" - This is flat-earth nuclear physics!
# QFD compares TOTAL SYSTEM ENERGIES directly:
#   E_model (QFD prediction) vs E_exp (AME2020 measurement)
#
# Loss = (E_model - E_exp)² / E_exp²

def run_qfd_solver(A: int, Z: int, params: Dict, verbose: bool = False, retry: bool = True,
                   fast_mode: bool = True) -> Dict:
    """
    Run QFD solver with given parameters.

    Args:
        fast_mode: If True, use reduced grid/iterations for search (32/150).
                   If False, use full resolution for verification (48/360).
    """
    import os
    import signal
    from pathlib import Path

    # Fix entrypoint: use correct solver path
    solver_path = Path(__file__).resolve().parent / "qfd_solver.py"

    # Fast search defaults or full verification
    grid_points = "32" if fast_mode else "48"
    iters_outer = "150" if fast_mode else "360"
    timeout_sec = 90 if fast_mode else 180

    cmd = [
        "python3", str(solver_path),
        "--A", str(A),
        "--Z", str(Z),
        "--alpha-model", "exp",  # Force exponential compounding decoder
        "--coulomb", "spectral",  # Spectral Coulomb method
        "--c-v2-base", str(params['c_v2_base']),
        "--c-v2-iso", str(params['c_v2_iso']),
        "--c-v2-mass", str(params['c_v2_mass']),
        "--c-v4-base", str(params['c_v4_base']),
        "--c-v4-size", str(params['c_v4_size']),
        "--alpha-e-scale", str(params['alpha_e_scale']),
        "--beta-e-scale", str(params['beta_e_scale']),
        "--c-sym", str(params['c_sym']),
        "--kappa-rho", str(params['kappa_rho']),
        "--grid-points", grid_points,
        "--iters-outer", iters_outer,
        "--emit-json",
    ]
    
    if verbose:
        print(f"  Running {Z=}, {A=}...", end=" ", flush=True)

    # Try with configured timeout, kill process group on timeout
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            preexec_fn=os.setsid  # Create new process group for clean killing
        )
        data = json.loads(result.stdout)

        if data.get("status") == "ok":
            if verbose:
                print(f"✓ E={data['E_model']:.2f} MeV, vir={abs(data.get('virial_abs', 0)):.3f}")
            return data
        else:
            if verbose:
                print(f"✗ {data.get('error', 'Unknown')}")
            return None

    except subprocess.TimeoutExpired as e:
        # Kill entire process group on timeout
        try:
            os.killpg(os.getpgid(e.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass  # Already dead

        # Retry with even fewer iterations if in fast mode and retry enabled
        if retry and fast_mode:
            if verbose:
                print(f"⚠ Timeout, retrying with 100 iters...", end=" ")
            cmd_retry = cmd[:-4] + ["--grid-points", "24", "--iters-outer", "100", "--emit-json"]
            try:
                result = subprocess.run(
                    cmd_retry,
                    capture_output=True,
                    text=True,
                    timeout=60,
                    preexec_fn=os.setsid
                )
                data = json.loads(result.stdout)
                if data.get("status") == "ok":
                    if verbose:
                        print(f"✓ E={data['E_model']:.2f} MeV")
                    return data
            except Exception:
                pass

        if verbose:
            print(f"✗ Timeout")
        return None

    except json.JSONDecodeError as e:
        if verbose:
            print(f"✗ JSONDecodeError")
            print(f"    stdout: {result.stdout[:200]}")
        return None

    except Exception as e:
        if verbose:
            print(f"✗ {type(e).__name__}: {e}")
        return None

def evaluate_parameters(params: Dict, calibration_df: pd.DataFrame, verbose: bool = False) -> Tuple[float, Dict]:
    """
    Evaluate QFD field parameters against calibration set.

    CORRECT QFD PHYSICS:
    - E_model = QFD interaction energy (negative for bound states)
    - E_total_QFD = M_constituents + E_interaction
    - M_constituents = Z×M_proton + N×M_neutron + Z×M_electron (rest masses)
    - Compare: E_total_QFD vs E_exp (AME2020 total mass-energy)
    - Loss = Σ (E_total_QFD - E_exp)² / E_exp²  (relative squared error)

    DO NOT use "binding energy" - that's nucleon-counting physics, not QFD!

    Returns:
        loss: Mean squared relative error in total energy
        metrics: Dictionary of diagnostic metrics
    """

    if verbose:
        print(f"\nEvaluating parameters:")
        print(f"  c_v2_base={params['c_v2_base']:.4f}, c_sym={params['c_sym']:.2f}")

    errors = []
    results = []

    # Track best loss for early pruning
    best_so_far = getattr(evaluate_parameters, "_best", float("inf"))
    running_sum = 0.0
    count = 0

    for idx, (_, row) in enumerate(calibration_df.iterrows()):
        A, Z = int(row['A']), int(row['Z'])
        E_exp_MeV = float(row['E_exp_MeV'])  # TOTAL experimental mass-energy

        data = run_qfd_solver(A, Z, params, verbose=verbose)

        if data and data.get('physical_success'):
            E_interaction = data['E_model']  # QFD interaction energy (negative for bound states)

            # Add rest masses to get total system energy
            N = A - Z
            M_constituents = Z * M_PROTON + N * M_NEUTRON + Z * M_ELECTRON
            E_total_QFD = M_constituents + E_interaction

            # Compare total energies
            rel_error = (E_total_QFD - E_exp_MeV) / E_exp_MeV

            # Add virial hinge penalty (prevent "good numbers / bad physics")
            V0 = 0.18  # Virial threshold
            rho_V = 4.0  # Virial penalty weight
            virial_abs = abs(float(data.get('virial_abs', data.get('virial', 0.0))))
            virial_penalty = max(0.0, virial_abs - V0) ** 2

            # Combined loss: energy error + virial hinge
            total_error = rel_error ** 2 + rho_V * virial_penalty
            errors.append(total_error)

            results.append({
                'A': A,
                'Z': Z,
                'E_exp_MeV': E_exp_MeV,
                'E_interaction': E_interaction,
                'E_total_QFD': E_total_QFD,
                'rel_error': rel_error,
                'virial_abs': virial_abs,
                'virial_penalty': virial_penalty,
            })
        else:
            # Penalize failed convergence heavily
            errors.append(10.0)  # Large penalty

        # Early pruning: stop if running mean already 15% worse than best
        running_sum += errors[-1]
        count += 1
        if count >= 5 and running_sum / count > best_so_far * 1.15:
            if verbose:
                print(f"  [Pruned after {count} isotopes: running_mean={running_sum/count:.3f} > best={best_so_far:.3f}]")
            # Return partial loss
            return running_sum / count, {
                'loss': running_sum / count,
                'n_success': len([r for r in results if r]),
                'n_total': len(calibration_df),
                'pruned': True
            }

    loss = np.mean(errors) if errors else 1e6

    # Remember best loss for next trial
    evaluate_parameters._best = min(best_so_far, loss)

    metrics = {
        'loss': loss,
        'n_success': len(results),
        'n_total': len(calibration_df),
        'results': results,
    }

    if verbose and results:
        print(f"\n  Success: {len(results)}/{len(calibration_df)}")
        print(f"  Loss (MSE relative): {loss:.6f}")
        print(f"  Mean |rel_error|: {np.mean([abs(r['rel_error']) for r in results]):.4f}")

    return loss, metrics

def main():
    parser = argparse.ArgumentParser("QFD Meta-Optimizer (AME2020)")
    parser.add_argument("--ame-csv", type=str, default="../data/ame2020_system_energies.csv")
    parser.add_argument("--n-calibration", type=int, default=40)
    parser.add_argument("--test-run", action="store_true", help="Quick test with Trial 32 params")
    args = parser.parse_args()
    
    print("="*70)
    print("QFD Meta-Optimizer: AME2020 Calibration")
    print("="*70)
    print()
    
    # Load AME2020 data
    ame_df = load_ame2020_data(args.ame_csv)
    
    # Select calibration set
    cal_df = select_calibration_set(ame_df, n_isotopes=args.n_calibration)
    
    if args.test_run:
        print("\n" + "="*70)
        print("TEST RUN: Evaluating Trial 32 parameters")
        print("="*70)
        
        trial32_params = {
            'c_v2_base': 2.201711,
            'c_v2_iso': 0.027035,
            'c_v2_mass': -0.000205,
            'c_v4_base': 5.282364,
            'c_v4_size': -0.085018,
            'alpha_e_scale': 1.007419,
            'beta_e_scale': 0.504312,
            'c_sym': 25.0,  # Add symmetry energy
            'kappa_rho': 0.029816,
        }
        
        loss, metrics = evaluate_parameters(trial32_params, cal_df, verbose=True)
        
        print("\n" + "="*70)
        print(f"Trial 32 + c_sym=25.0: Loss = {loss:.6f}")
        print("="*70)
        
        # Save results
        output_file = Path("trial32_ame2020_test.json")
        with open(output_file, 'w') as f:
            json.dump({
                'params': trial32_params,
                'loss': loss,
                'metrics': {k: v for k, v in metrics.items() if k != 'results'},
                'results': metrics['results'],
            }, f, indent=2)
        print(f"\nResults saved to: {output_file}")
    
    else:
        print("\n" + "="*70)
        print("FULL META-OPTIMIZATION")
        print("="*70)
        print()

        # Use scipy differential_evolution
        print("Using scipy.optimize.differential_evolution")
        print()

        from scipy.optimize import differential_evolution

        def objective(x):
            params = {
                'c_v2_base': x[0],
                'c_v2_iso': x[1],
                'c_v2_mass': x[2],
                'c_v4_base': x[3],
                'c_v4_size': x[4],
                'alpha_e_scale': x[5],
                'beta_e_scale': x[6],
                'c_sym': x[7],
                'kappa_rho': x[8],
            }
            loss, _ = evaluate_parameters(params, cal_df, verbose=False)
            print(f"  Trial: loss={loss:.4f}, c_v2_base={x[0]:.3f}, c_sym={x[7]:.1f}")
            return loss

        # TRIAL 32 BASELINE (known good parameters)
        trial32 = {
            'c_v2_base': 2.201711,
            'c_v2_iso': 0.027035,
            'c_v2_mass': -0.000205,
            'c_v4_base': 5.282364,
            'c_v4_size': -0.085018,
            'alpha_e_scale': 1.007419,
            'beta_e_scale': 0.504312,
            'c_sym': 25.0,
            'kappa_rho': 0.029816,
        }

        # LOCAL REFINEMENT: ±10% around Trial 32 (not global search!)
        bounds = [
            (trial32['c_v2_base'] * 0.9, trial32['c_v2_base'] * 1.1),      # c_v2_base
            (trial32['c_v2_iso'] * 0.8, trial32['c_v2_iso'] * 1.2),        # c_v2_iso
            (trial32['c_v2_mass'] * 2.0, trial32['c_v2_mass'] * 0.0),      # c_v2_mass (keep near 0)
            (trial32['c_v4_base'] * 0.95, trial32['c_v4_base'] * 1.05),    # c_v4_base
            (trial32['c_v4_size'] * 1.2, trial32['c_v4_size'] * 0.8),      # c_v4_size
            (trial32['alpha_e_scale'] * 0.95, trial32['alpha_e_scale'] * 1.05),  # alpha_e_scale
            (trial32['beta_e_scale'] * 0.9, trial32['beta_e_scale'] * 1.1),      # beta_e_scale
            (trial32['c_sym'] * 0.9, trial32['c_sym'] * 1.1),              # c_sym
            (trial32['kappa_rho'] * 0.8, trial32['kappa_rho'] * 1.2),      # kappa_rho
        ]

        print("Starting LOCAL REFINEMENT around Trial 32...")
        print(f"Calibration set: {len(cal_df)} isotopes")
        print(f"Parameter space: 9 dimensions (±10% bounds)")
        print(f"Baseline (Trial 32): c_v2_base={trial32['c_v2_base']:.4f}, c_sym={trial32['c_sym']:.1f}")
        print(f"Virial hinge: V0={0.18}, penalty weight={4.0}")
        print()

        # Seed population with Trial 32
        trial32_array = [
            trial32['c_v2_base'],
            trial32['c_v2_iso'],
            trial32['c_v2_mass'],
            trial32['c_v4_base'],
            trial32['c_v4_size'],
            trial32['alpha_e_scale'],
            trial32['beta_e_scale'],
            trial32['c_sym'],
            trial32['kappa_rho'],
        ]

        result = differential_evolution(
            objective,
            bounds,
            maxiter=25,
            popsize=10,
            workers=1,  # Serial (parallel causes pickling issues with nested function)
            seed=42,
            x0=trial32_array,  # Seed with Trial 32!
            atol=0.01,  # Tighter tolerance for local refinement
            tol=0.01,
            disp=True,
        )

        best_params = {
            'c_v2_base': result.x[0],
            'c_v2_iso': result.x[1],
            'c_v2_mass': result.x[2],
            'c_v4_base': result.x[3],
            'c_v4_size': result.x[4],
            'alpha_e_scale': result.x[5],
            'beta_e_scale': result.x[6],
            'c_sym': result.x[7],
            'kappa_rho': result.x[8],
        }

        print("\n" + "="*70)
        print("OPTIMIZATION COMPLETE")
        print("="*70)
        print(f"Best loss: {result.fun:.6f}")
        print("\nOptimized parameters:")
        for key, val in best_params.items():
            print(f"  {key}: {val:.6f}")

        # Evaluate on full calibration set with verbose output
        print("\n" + "="*70)
        print("FINAL EVALUATION")
        print("="*70)
        loss, metrics = evaluate_parameters(best_params, cal_df, verbose=True)

        # Save results
        output_file = Path("qfd_metaopt_best.json")
        with open(output_file, 'w') as f:
            json.dump({
                'params': best_params,
                'loss': loss,
                'metrics': {k: v for k, v in metrics.items() if k != 'results'},
                'results': metrics['results'],
                'optimization': {
                    'method': 'scipy.differential_evolution',
                    'n_iterations': result.nit,
                    'n_evaluations': result.nfev,
                }
            }, f, indent=2)
        print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()
