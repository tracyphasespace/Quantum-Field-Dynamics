#!/usr/bin/env python3
"""
Evaluate full QFD lightcurve model against all ~118k observations.

This computes the full physics forward model for each observation, not just the
Stage 2 regression on ln_A values.
"""

import argparse
import json
import numpy as np
from pathlib import Path
import sys
import time

# JAX
import jax
import jax.numpy as jnp

# Add core directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))
from v15_data import LightcurveLoader
from v15_model import chi2_single_sn_jax

# L_peak fixed value from Stage 1
L_PEAK_CANONICAL = 1.5e43  # erg/s


def load_sn_data_with_params(stage1_dir, lightcurves_dict, quality_cut=2000):
    """
    Load SNe with both lightcurve data and Stage 1 fitted parameters.

    Returns:
        list of dicts with keys: 'snid', 'lc', 'persn_params', 'z_obs'
    """
    stage1_path = Path(stage1_dir)
    sn_data = []
    excluded = 0

    for snid_str, lc in lightcurves_dict.items():
        snid = int(snid_str)
        snid_dir = stage1_path / str(snid)
        metrics_file = snid_dir / "metrics.json"
        persn_file = snid_dir / "persn_best.npy"

        if not (metrics_file.exists() and persn_file.exists()):
            continue

        try:
            # Load metrics
            with open(metrics_file) as f:
                metrics = json.load(f)

            chi2 = abs(metrics['chi2'])

            # Quality cut
            if chi2 >= quality_cut:
                excluded += 1
                continue

            # Load per-SN parameters: [t0, A_plasma, beta, ln_A]
            persn_best = np.load(persn_file)

            # Quality cuts on parameters
            ln_A = persn_best[3]
            if ln_A >= 28 or ln_A <= -28:
                excluded += 1
                continue

            A_plasma, beta = persn_best[1], persn_best[2]
            if A_plasma <= 0.001 or A_plasma >= 0.999:
                excluded += 1
                continue
            if beta <= 0.001 or beta >= 3.999:
                excluded += 1
                continue

            iters = metrics.get('iters', 0)
            if iters < 1:
                excluded += 1
                continue

            # Prepare photometry array: [N_obs, 4] with [mjd, wavelength, flux, flux_err]
            photometry = np.column_stack([
                lc.mjd,
                lc.wavelength_nm,
                lc.flux_jy,
                lc.flux_err_jy
            ])

            sn_data.append({
                'snid': snid,
                'photometry': photometry,
                'persn_params': persn_best,  # [t0, A_plasma, beta, ln_A]
                'z_obs': lc.z,
                'n_obs': len(lc.mjd)
            })

        except Exception as e:
            print(f"Warning: Failed to load SN {snid}: {e}")
            continue

    print(f"  Loaded {len(sn_data)} SNe with full lightcurve data")
    print(f"  Excluded {excluded} SNe")
    print(f"  Total observations: {sum(sn['n_obs'] for sn in sn_data)}")

    return sn_data


def evaluate_global_params(sn_data, k_J, eta_prime, xi, param_name):
    """
    Evaluate fit quality for given global parameters against all lightcurves.

    This is the REAL forward problem - computing full physics model for each observation!
    """
    print(f"\n{'='*80}")
    print(f"{param_name}")
    print(f"{'='*80}")
    print(f"Global parameters:")
    print(f"  k_J = {k_J:.6f}")
    print(f"  η′  = {eta_prime:.6f}")
    print(f"  ξ   = {xi:.6f}")

    global_params = (k_J, eta_prime, xi)

    print(f"\nComputing chi-squared for {len(sn_data)} SNe...")
    print(f"(This uses full QFD physics model for each observation)")

    start_time = time.time()

    chi2_list = []
    n_obs_list = []

    # Process each SN
    for i, sn in enumerate(sn_data):
        if (i+1) % 500 == 0:
            elapsed = time.time() - start_time
            rate = (i+1) / elapsed
            remaining = (len(sn_data) - (i+1)) / rate
            print(f"  Progress: {i+1}/{len(sn_data)} SNe ({rate:.1f} SNe/s, {remaining:.0f}s remaining)")

        # Convert to JAX arrays
        photometry_jax = jnp.array(sn['photometry'])
        persn_params = tuple(float(x) for x in sn['persn_params'])

        # Compute chi-squared using full QFD model
        chi2 = chi2_single_sn_jax(
            global_params=global_params,
            persn_params=persn_params,
            L_peak=L_PEAK_CANONICAL,
            photometry=photometry_jax,
            z_obs=float(sn['z_obs'])
        )

        chi2_list.append(float(chi2))
        n_obs_list.append(sn['n_obs'])

    elapsed = time.time() - start_time

    # Compute statistics
    total_chi2 = np.sum(chi2_list)
    total_obs = np.sum(n_obs_list)
    n_params = 3  # Global parameters
    dof = total_obs - len(sn_data) * 4 - n_params  # Subtract per-SN params + global params
    reduced_chi2 = total_chi2 / dof

    mean_chi2_per_sn = np.mean(chi2_list)
    median_chi2_per_sn = np.median(chi2_list)

    print(f"\nResults:")
    print(f"  Time: {elapsed:.1f} seconds ({len(sn_data)/elapsed:.1f} SNe/sec)")
    print(f"  Total observations: {total_obs}")
    print(f"  Total chi²: {total_chi2:.1f}")
    print(f"  Degrees of freedom: {dof}")
    print(f"  Reduced chi²: {reduced_chi2:.4f}")
    print(f"  Mean chi² per SN: {mean_chi2_per_sn:.2f}")
    print(f"  Median chi² per SN: {median_chi2_per_sn:.2f}")

    return {
        'total_chi2': total_chi2,
        'dof': dof,
        'reduced_chi2': reduced_chi2,
        'mean_chi2_per_sn': mean_chi2_per_sn,
        'median_chi2_per_sn': median_chi2_per_sn,
        'n_sne': len(sn_data),
        'n_obs': total_obs,
        'time_seconds': elapsed
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage1-results', required=True)
    parser.add_argument('--lightcurves', required=True)
    parser.add_argument('--quality-cut', type=float, default=2000)

    args = parser.parse_args()

    print("="*80)
    print("FULL LIGHTCURVE EVALUATION: ALL ~118k OBSERVATIONS")
    print("="*80)
    print(f"Stage 1 results: {args.stage1_results}")
    print(f"Quality cut: chi2 < {args.quality_cut}")

    # Load lightcurves
    print("\nLoading lightcurves...")
    loader = LightcurveLoader(Path(args.lightcurves))
    all_lcs = loader.load()
    print(f"  Loaded {len(all_lcs)} lightcurves")

    # Load SNe with Stage 1 parameters
    print("\nLoading Stage 1 results and lightcurve data...")
    sn_data = load_sn_data_with_params(args.stage1_results, all_lcs, args.quality_cut)

    # Define parameter sets to test
    param_sets = [
        {
            'name': 'FIXED_TEST Results (200 samples, no priors)',
            'k_J': 5.381956298963422,
            'eta_prime': -1.900259394073368,
            'xi': -3.8146692019334054
        },
        {
            'name': 'Recovery Golden Values',
            'k_J': 10.770038588319618,
            'eta_prime': -7.987900510670775,
            'xi': -6.907618767280434
        }
    ]

    # Evaluate each parameter set
    results = []
    for params in param_sets:
        result = evaluate_global_params(
            sn_data,
            params['k_J'],
            params['eta_prime'],
            params['xi'],
            params['name']
        )
        result['params'] = params
        results.append(result)

    # Compare
    print(f"\n{'='*80}")
    print("COMPARISON")
    print(f"{'='*80}")
    print(f"\n{'Parameter Set':<50} {'Reduced χ²':<15} {'Mean χ²/SN':<15}")
    print("-" * 80)
    for i, params in enumerate(param_sets):
        print(f"{params['name']:<50} {results[i]['reduced_chi2']:<15.4f} {results[i]['mean_chi2_per_sn']:<15.2f}")

    best_idx = np.argmin([r['reduced_chi2'] for r in results])
    print(f"\nBest fit: {param_sets[best_idx]['name']}")
    print(f"  Reduced χ² = {results[best_idx]['reduced_chi2']:.4f}")
    print(f"  This is the FULL physics model evaluation!")

    return 0


if __name__ == '__main__':
    sys.exit(main())
