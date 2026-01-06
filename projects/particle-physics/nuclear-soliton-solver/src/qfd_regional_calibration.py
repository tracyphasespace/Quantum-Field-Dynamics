#!/usr/bin/env python3
"""
QFD Regional Calibration: Separate Parameter Sets by Mass Region

Problem:
    Trial 32 universal parameters achieve:
    - Light nuclei (A < 60): < 1% error ✓ Excellent
    - Medium (60 ≤ A < 120): 2-3% error ⚠️ Moderate
    - Heavy (A ≥ 120): -7% to -9% systematic underbinding ✗ Poor

Solution:
    Optimize separate parameter sets for each mass region while maintaining
    physics continuity at boundaries.

Approach:
    1. Light region: KEEP Trial 32 parameters (already optimal)
    2. Medium region: Fine-tune from Trial 32 baseline
    3. Heavy region: Increase cohesion ~10-15% to fix underbinding

    Optional: Add explicit surface term E_surf = c_surf × A^(2/3)
"""

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

# Import from sibling modules
import sys
sys.path.insert(0, str(Path(__file__).parent))

from qfd_metaopt_ame2020 import (
    load_ame2020_data,
    run_qfd_solver,
    M_PROTON,
    M_NEUTRON,
    M_ELECTRON,
)

# ============================================================================
# REGIONAL PARAMETER SETS
# ============================================================================

# Trial 32 parameters (optimal for light nuclei A < 60)
TRIAL32_PARAMS = {
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

# Mass region boundaries
REGION_BOUNDARIES = {
    'light': (1, 59),      # A < 60: keep Trial 32
    'medium': (60, 119),   # 60 ≤ A < 120: fine-tune
    'heavy': (120, 250),   # A ≥ 120: significant adjustment
}

# ============================================================================
# CALIBRATION SET SELECTION
# ============================================================================

def select_regional_calibration_set(df: pd.DataFrame, region: str, n_isotopes: int = 20) -> pd.DataFrame:
    """
    Select physics-driven calibration set for a specific mass region.

    Args:
        df: AME2020 dataframe
        region: 'light', 'medium', or 'heavy'
        n_isotopes: Target number of isotopes per region

    Returns:
        Calibration dataframe for the region
    """

    A_min, A_max = REGION_BOUNDARIES[region]
    region_df = df[(df['A'] >= A_min) & (df['A'] <= A_max)]

    if region == 'light':
        # Light: magic numbers and doubly-magic nuclei
        targets = [
            (2, 4),     # He-4 (doubly magic)
            (6, 12),    # C-12
            (8, 16),    # O-16 (doubly magic)
            (10, 20),   # Ne-20
            (14, 28),   # Si-28
            (20, 40),   # Ca-40 (doubly magic)
            (20, 48),   # Ca-48 (doubly magic)
            (26, 56),   # Fe-56 (stability peak)
        ]

    elif region == 'medium':
        # Medium: transition region, shell closures
        targets = [
            (28, 58),   # Ni-58 (Z=28 magic)
            (28, 62),   # Ni-62 (highest BE/A)
            (28, 64),   # Ni-64
            (29, 63),   # Cu-63
            (29, 65),   # Cu-65
            (30, 64),   # Zn-64
            (47, 107),  # Ag-107
            (47, 109),  # Ag-109
            (50, 100),  # Sn-100 (doubly magic if available)
            (50, 120),  # Sn-120 (Z=50 magic)
        ]

    else:  # heavy
        # Heavy: focus on underbinding correction
        targets = [
            (50, 120),  # Sn-120
            (79, 197),  # Au-197
            (80, 200),  # Hg-200
            (82, 206),  # Pb-206 (Z=82 magic)
            (82, 207),  # Pb-207
            (82, 208),  # Pb-208 (doubly magic)
            (92, 235),  # U-235
            (92, 238),  # U-238
        ]

    # Select isotopes
    calibration_set = []
    for Z, A in targets:
        row = region_df[(region_df['Z'] == Z) & (region_df['A'] == A)]
        if not row.empty:
            calibration_set.append(row.iloc[0])
            if len(calibration_set) >= n_isotopes:
                break

    cal_df = pd.DataFrame(calibration_set).drop_duplicates(subset=['Z', 'A'])

    print(f"\n{region.upper()} region calibration set: {len(cal_df)} isotopes")
    print(f"  A range: {cal_df['A'].min()} to {cal_df['A'].max()}")
    print(f"  Z range: {cal_df['Z'].min()} to {cal_df['Z'].max()}")

    return cal_df.sort_values('A')

# ============================================================================
# REGIONAL PARAMETER OPTIMIZATION
# ============================================================================

def evaluate_regional_parameters(params: Dict, calibration_df: pd.DataFrame,
                                  verbose: bool = False) -> Tuple[float, Dict]:
    """
    Evaluate parameters for a specific mass region.

    Uses same loss function as qfd_metaopt_ame2020 but on regional subset.
    """

    if verbose:
        print(f"\nEvaluating regional parameters:")
        print(f"  c_v2_base={params['c_v2_base']:.4f}, c_v4_base={params['c_v4_base']:.4f}")

    errors = []
    results = []

    for idx, (_, row) in enumerate(calibration_df.iterrows()):
        A, Z = int(row['A']), int(row['Z'])
        E_exp_MeV = float(row['E_exp_MeV'])

        data = run_qfd_solver(A, Z, params, verbose=verbose, fast_mode=True)

        if data and data.get('physical_success'):
            E_interaction = data['E_model']
            N = A - Z
            M_constituents = Z * M_PROTON + N * M_NEUTRON + Z * M_ELECTRON
            E_total_QFD = M_constituents + E_interaction

            rel_error = (E_total_QFD - E_exp_MeV) / E_exp_MeV

            # Virial hinge penalty
            V0 = 0.18
            rho_V = 4.0
            virial_abs = abs(float(data.get('virial_abs', data.get('virial', 0.0))))
            virial_penalty = max(0.0, virial_abs - V0) ** 2

            total_error = rel_error ** 2 + rho_V * virial_penalty
            errors.append(total_error)

            results.append({
                'A': A,
                'Z': Z,
                'E_exp_MeV': E_exp_MeV,
                'E_total_QFD': E_total_QFD,
                'rel_error': rel_error,
                'virial_abs': virial_abs,
            })
        else:
            errors.append(10.0)  # Penalty for failed convergence

    loss = np.mean(errors) if errors else 1e6

    metrics = {
        'loss': loss,
        'n_success': len(results),
        'n_total': len(calibration_df),
        'results': results,
    }

    if verbose and results:
        print(f"\n  Success: {len(results)}/{len(calibration_df)}")
        print(f"  Loss: {loss:.6f}")
        print(f"  Mean |rel_error|: {np.mean([abs(r['rel_error']) for r in results]):.4f}")

    return loss, metrics

def optimize_region(region: str, cal_df: pd.DataFrame,
                    baseline_params: Dict = None) -> Dict:
    """
    Optimize parameters for a specific mass region.

    Args:
        region: 'light', 'medium', or 'heavy'
        cal_df: Calibration dataframe for the region
        baseline_params: Starting parameters (default: Trial 32)

    Returns:
        Optimized parameters for the region
    """

    if baseline_params is None:
        baseline_params = TRIAL32_PARAMS.copy()

    print(f"\n{'='*70}")
    print(f"OPTIMIZING {region.upper()} REGION")
    print(f"{'='*70}")

    if region == 'light':
        # Light region: Trial 32 already optimal, just validate
        print("Light region uses Trial 32 parameters (already optimal)")
        loss, metrics = evaluate_regional_parameters(baseline_params, cal_df, verbose=True)
        return baseline_params

    # Medium and heavy: optimize around Trial 32
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
        loss, _ = evaluate_regional_parameters(params, cal_df, verbose=False)
        print(f"  Trial: loss={loss:.4f}, c_v2_base={x[0]:.3f}, c_v4_base={x[3]:.3f}")
        return loss

    # Define search bounds based on region
    if region == 'heavy':
        # Heavy: need MORE cohesion (larger c_v2_base, smaller c_v4_base)
        bounds = [
            (baseline_params['c_v2_base'] * 1.05, baseline_params['c_v2_base'] * 1.20),  # +5% to +20% cohesion
            (baseline_params['c_v2_iso'] * 0.9, baseline_params['c_v2_iso'] * 1.1),
            (baseline_params['c_v2_mass'] * 2.0, baseline_params['c_v2_mass'] * 0.0),
            (baseline_params['c_v4_base'] * 0.85, baseline_params['c_v4_base'] * 1.00),  # -15% to 0% repulsion
            (baseline_params['c_v4_size'] * 1.1, baseline_params['c_v4_size'] * 0.9),
            (baseline_params['alpha_e_scale'] * 0.95, baseline_params['alpha_e_scale'] * 1.05),
            (baseline_params['beta_e_scale'] * 0.9, baseline_params['beta_e_scale'] * 1.1),
            (baseline_params['c_sym'] * 0.9, baseline_params['c_sym'] * 1.1),
            (baseline_params['kappa_rho'] * 0.8, baseline_params['kappa_rho'] * 1.2),
        ]
    else:  # medium
        # Medium: smaller adjustments
        bounds = [
            (baseline_params['c_v2_base'] * 0.95, baseline_params['c_v2_base'] * 1.10),
            (baseline_params['c_v2_iso'] * 0.9, baseline_params['c_v2_iso'] * 1.1),
            (baseline_params['c_v2_mass'] * 2.0, baseline_params['c_v2_mass'] * 0.0),
            (baseline_params['c_v4_base'] * 0.95, baseline_params['c_v4_base'] * 1.05),
            (baseline_params['c_v4_size'] * 1.1, baseline_params['c_v4_size'] * 0.9),
            (baseline_params['alpha_e_scale'] * 0.95, baseline_params['alpha_e_scale'] * 1.05),
            (baseline_params['beta_e_scale'] * 0.9, baseline_params['beta_e_scale'] * 1.1),
            (baseline_params['c_sym'] * 0.9, baseline_params['c_sym'] * 1.1),
            (baseline_params['kappa_rho'] * 0.8, baseline_params['kappa_rho'] * 1.2),
        ]

    # Seed with baseline
    baseline_array = [
        baseline_params['c_v2_base'],
        baseline_params['c_v2_iso'],
        baseline_params['c_v2_mass'],
        baseline_params['c_v4_base'],
        baseline_params['c_v4_size'],
        baseline_params['alpha_e_scale'],
        baseline_params['beta_e_scale'],
        baseline_params['c_sym'],
        baseline_params['kappa_rho'],
    ]

    print(f"\nStarting optimization for {region} region...")
    print(f"Baseline: c_v2_base={baseline_params['c_v2_base']:.4f}")

    result = differential_evolution(
        objective,
        bounds,
        maxiter=20,
        popsize=8,
        workers=1,
        seed=42,
        x0=baseline_array,
        atol=0.01,
        tol=0.01,
        disp=True,
    )

    optimized_params = {
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

    print(f"\nOptimization complete for {region} region:")
    print(f"  Loss: {result.fun:.6f}")
    print(f"  c_v2_base: {baseline_params['c_v2_base']:.4f} → {optimized_params['c_v2_base']:.4f}")
    print(f"  c_v4_base: {baseline_params['c_v4_base']:.4f} → {optimized_params['c_v4_base']:.4f}")

    return optimized_params

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser("QFD Regional Calibration")
    parser.add_argument("--ame-csv", type=str, default="../data/ame2020_system_energies.csv")
    parser.add_argument("--region", choices=['light', 'medium', 'heavy', 'all'], default='all')
    parser.add_argument("--n-calibration", type=int, default=15, help="Isotopes per region")
    parser.add_argument("--validate-only", action="store_true", help="Validate Trial 32 on all regions")
    args = parser.parse_args()

    print("="*70)
    print("QFD REGIONAL CALIBRATION")
    print("="*70)

    # Load AME2020 data
    ame_df = load_ame2020_data(args.ame_csv)

    regions_to_process = ['light', 'medium', 'heavy'] if args.region == 'all' else [args.region]

    regional_params = {}

    for region in regions_to_process:
        # Select calibration set
        cal_df = select_regional_calibration_set(ame_df, region, n_isotopes=args.n_calibration)

        if args.validate_only:
            # Just validate Trial 32 on this region
            print(f"\nValidating Trial 32 on {region} region...")
            loss, metrics = evaluate_regional_parameters(TRIAL32_PARAMS, cal_df, verbose=True)
            regional_params[region] = TRIAL32_PARAMS
        else:
            # Optimize parameters for this region
            optimized = optimize_region(region, cal_df, baseline_params=TRIAL32_PARAMS)
            regional_params[region] = optimized

    # Save results
    output_file = Path("regional_calibration_results.json")
    with open(output_file, 'w') as f:
        json.dump({
            'trial32_baseline': TRIAL32_PARAMS,
            'regional_parameters': regional_params,
            'region_boundaries': REGION_BOUNDARIES,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
