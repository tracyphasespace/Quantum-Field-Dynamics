#!/usr/bin/env python3
"""
Stage 1: Measure deviations on new SNe with fixed physics parameters.

Given physics parameters from Stage 0, this measures how well new SNe fit
by computing Δχ² = χ²_fixed - χ²_optimized.

Large Δχ² indicates the SN doesn't follow the physics model (likely BBH-dominated).
"""

import argparse
import json
import numpy as np
from pathlib import Path
import sys
from typing import Dict, Tuple

# Add core directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))
sys.path.insert(0, str(Path(__file__).parent))
from v15_data import LightcurveLoader
from stage1_optimize import optimize_single_sn


def chi2_with_fixed_physics(lc_data, global_params_fixed, persn_params_stage1):
    """
    Compute chi² with fixed global physics and Stage 1 per-SN parameters.

    This gives the "baseline" chi² from Stage 1 where physics was free.
    """
    import jax.numpy as jnp
    from v15_model import chi2_single_sn_jax

    L_peak = 1.5e43  # Fixed canonical value

    # Prepare photometry array
    photometry = jnp.array(np.column_stack([
        lc_data.mjd,
        lc_data.wavelength_nm,
        lc_data.flux_jy,
        lc_data.flux_err_jy
    ]))

    persn_tuple = tuple(float(x) for x in persn_params_stage1)

    chi2 = chi2_single_sn_jax(
        global_params=global_params_fixed,
        persn_params=persn_tuple,
        L_peak=L_peak,
        photometry=photometry,
        z_obs=float(lc_data.z)
    )

    return float(chi2)


def optimize_with_fixed_physics(lc_data, snid, global_params_fixed):
    """
    Re-optimize per-SN parameters with fixed global physics.

    Returns optimized chi² and parameters.
    """
    result = optimize_single_sn(
        snid=str(snid),
        lc_data=lc_data,
        global_params=global_params_fixed,
        max_iters=500,
        tol=1e-6,
        verbose=False,
        use_studentt=False
    )

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage1-results', required=True, help='Stage 1 results directory')
    parser.add_argument('--lightcurves', required=True, help='Lightcurve data file')
    parser.add_argument('--physics-params', required=True, help='JSON file with fixed physics parameters (k_J, eta_prime, xi)')
    parser.add_argument('--exclude-snids', help='JSON file with SNIDs to exclude (e.g., Stage 0 training set)')
    parser.add_argument('--n-test', type=int, default=500, help='Number of SNe to test')
    parser.add_argument('--out', required=True, help='Output directory for deviation results')

    args = parser.parse_args()

    print("="*80)
    print("STAGE 1: Measure Deviations with Fixed Physics Parameters")
    print("="*80)

    # Load fixed physics parameters
    print(f"\nLoading fixed physics parameters from {args.physics_params}...")
    with open(args.physics_params) as f:
        physics_data = json.load(f)

    k_J = physics_data['k_J']
    eta_prime = physics_data['eta_prime']
    xi = physics_data['xi']
    global_params_fixed = (k_J, eta_prime, xi)

    print(f"Fixed physics parameters:")
    print(f"  k_J = {k_J:.6f}")
    print(f"  η'  = {eta_prime:.6f}")
    print(f"  ξ   = {xi:.6f}")

    # Load excluded SNIDs
    excluded_snids = set()
    if args.exclude_snids:
        print(f"\nLoading excluded SNIDs from {args.exclude_snids}...")
        with open(args.exclude_snids) as f:
            exclude_data = json.load(f)
            excluded_snids = set(exclude_data['snids'])
        print(f"Excluding {len(excluded_snids)} SNe from testing")

    # Load lightcurves
    print(f"\nLoading lightcurves from {args.lightcurves}...")
    loader = LightcurveLoader(Path(args.lightcurves))
    all_lcs = loader.load()
    print(f"Loaded {len(all_lcs)} lightcurves")

    # Get list of SNe to test
    stage1_path = Path(args.stage1_results)
    test_sne = []

    for snid_dir in sorted(stage1_path.iterdir()):
        if not snid_dir.is_dir():
            continue

        snid = int(snid_dir.name)

        # Skip excluded SNe
        if snid in excluded_snids:
            continue

        snid_str = str(snid)
        if snid_str not in all_lcs:
            continue

        metrics_file = snid_dir / "metrics.json"
        persn_file = snid_dir / "persn_best.npy"

        if not (metrics_file.exists() and persn_file.exists()):
            continue

        lc = all_lcs[snid_str]

        # Basic quality cuts
        if len(lc.mjd) < 5:
            continue

        try:
            with open(metrics_file) as f:
                metrics = json.load(f)

            persn_best = np.load(persn_file)

            # Skip obviously bad fits from Stage 1
            if abs(metrics['chi2']) > 2000:
                continue

            test_sne.append({
                'snid': snid,
                'lc': lc,
                'persn_stage1': persn_best,
                'chi2_stage1': abs(metrics['chi2'])
            })

            if len(test_sne) >= args.n_test:
                break

        except Exception as e:
            print(f"Warning: Failed to load SN {snid}: {e}")
            continue

    print(f"\nSelected {len(test_sne)} SNe for deviation testing")

    # Measure deviations
    print("\nMeasuring deviations...")
    results = []
    n_failed = 0

    for i, sn_data in enumerate(test_sne):
        if (i+1) % 50 == 0:
            print(f"  Progress: {i+1}/{len(test_sne)} SNe")

        snid = sn_data['snid']
        lc = sn_data['lc']

        # Compute chi² with Stage 1 parameters and fixed physics
        try:
            chi2_fixed = chi2_with_fixed_physics(
                lc,
                global_params_fixed,
                sn_data['persn_stage1']
            )
        except Exception as e:
            print(f"  Warning: Failed to compute fixed chi² for SN {snid}: {e}")
            n_failed += 1
            continue

        # Re-optimize per-SN parameters with fixed physics
        try:
            opt_result = optimize_with_fixed_physics(lc, snid, global_params_fixed)

            if not opt_result['ok'] or np.isnan(opt_result['chi2']) or opt_result['chi2'] > 1e10:
                n_failed += 1
                continue

            chi2_optimized = opt_result['chi2']

        except Exception as e:
            print(f"  Warning: Failed to optimize SN {snid}: {e}")
            n_failed += 1
            continue

        # Compute deviation
        delta_chi2 = chi2_fixed - chi2_optimized
        n_obs = len(lc.mjd)
        delta_chi2_per_obs = delta_chi2 / n_obs if n_obs > 0 else 0

        results.append({
            'snid': snid,
            'z': lc.z,
            'n_obs': n_obs,
            'chi2_stage1': sn_data['chi2_stage1'],
            'chi2_fixed': chi2_fixed,
            'chi2_optimized': chi2_optimized,
            'delta_chi2': delta_chi2,
            'delta_chi2_per_obs': delta_chi2_per_obs,
            'reduced_chi2_optimized': chi2_optimized / (n_obs - 4) if n_obs > 4 else np.nan
        })

    print(f"\nSuccessfully analyzed {len(results)} SNe")
    print(f"Failed: {n_failed} SNe")

    # Analyze deviations
    if len(results) > 0:
        delta_chi2_vals = [r['delta_chi2'] for r in results]
        delta_per_obs_vals = [r['delta_chi2_per_obs'] for r in results]
        reduced_chi2_vals = [r['reduced_chi2_optimized'] for r in results if not np.isnan(r['reduced_chi2_optimized'])]

        print(f"\nDeviation statistics:")
        print(f"  Mean Δχ²: {np.mean(delta_chi2_vals):.2f}")
        print(f"  Median Δχ²: {np.median(delta_chi2_vals):.2f}")
        print(f"  Std Δχ²: {np.std(delta_chi2_vals):.2f}")
        print(f"  Mean Δχ²/N_obs: {np.mean(delta_per_obs_vals):.3f}")
        print(f"  Median Δχ²/N_obs: {np.median(delta_per_obs_vals):.3f}")

        print(f"\nFit quality (with fixed physics):")
        print(f"  Mean reduced χ²: {np.mean(reduced_chi2_vals):.3f}")
        print(f"  Median reduced χ²: {np.median(reduced_chi2_vals):.3f}")

        # Identify outliers: large Δχ² suggests doesn't fit physics model
        # Use criterion: Δχ²/N_obs > 2 (i.e., physics constraint costs 2χ² per observation)
        outliers = [r for r in results if r['delta_chi2_per_obs'] > 2.0]
        print(f"\nOutliers (Δχ²/N_obs > 2.0): {len(outliers)}/{len(results)} = {100*len(outliers)/len(results):.1f}%")

        # Show distribution
        bins = [0, 0.5, 1.0, 2.0, 5.0, 10.0, np.inf]
        hist, _ = np.histogram(delta_per_obs_vals, bins=bins)
        print(f"\nΔχ²/N_obs distribution:")
        for i in range(len(bins)-1):
            print(f"  {bins[i]:.1f} - {bins[i+1]:.1f}: {hist[i]} SNe ({100*hist[i]/len(results):.1f}%)")

    # Save results
    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)

    results_file = out_path / "deviations.json"
    with open(results_file, 'w') as f:
        json.dump({
            'physics_params': {
                'k_J': k_J,
                'eta_prime': eta_prime,
                'xi': xi
            },
            'n_tested': len(results),
            'n_failed': n_failed,
            'results': results,
            'statistics': {
                'mean_delta_chi2': float(np.mean(delta_chi2_vals)) if len(results) > 0 else None,
                'median_delta_chi2': float(np.median(delta_chi2_vals)) if len(results) > 0 else None,
                'mean_delta_chi2_per_obs': float(np.mean(delta_per_obs_vals)) if len(results) > 0 else None,
                'median_reduced_chi2': float(np.median(reduced_chi2_vals)) if len(reduced_chi2_vals) > 0 else None
            }
        }, f, indent=2)

    print(f"\nSaved results to {results_file}")

    # Save list of outlier SNIDs for exclusion
    if len(results) > 0:
        outlier_snids = [r['snid'] for r in results if r['delta_chi2_per_obs'] > 2.0]
        outlier_file = out_path / "outlier_snids.json"
        with open(outlier_file, 'w') as f:
            json.dump({
                'snids': outlier_snids,
                'n_outliers': len(outlier_snids),
                'criterion': 'delta_chi2_per_obs > 2.0'
            }, f, indent=2)
        print(f"Saved outlier SNIDs to {outlier_file}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
