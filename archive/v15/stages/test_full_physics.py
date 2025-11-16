#!/usr/bin/env python3
"""
Test full QFD physics model by refitting 50 SNe from scratch.

Uses Stage 2 global parameters and refits per-SN parameters to see actual chi².
"""

import argparse
import json
import numpy as np
from pathlib import Path
import sys

# Add core directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))
sys.path.insert(0, str(Path(__file__).parent))
from v15_data import LightcurveLoader
from stage1_optimize import optimize_single_sn

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lightcurves', required=True)
    parser.add_argument('--n-sne', type=int, default=50)

    args = parser.parse_args()

    print("="*80)
    print("FULL PHYSICS TEST: Refit SNe with Stage 2 Global Parameters")
    print("="*80)

    # Load lightcurves
    print(f"\nLoading lightcurves from {args.lightcurves}...")
    loader = LightcurveLoader(Path(args.lightcurves))
    all_lcs = loader.load()
    print(f"Loaded {len(all_lcs)} lightcurves")

    # Get first N SNe with enough observations
    test_sne = []
    for snid_str, lc in all_lcs.items():
        if len(lc.mjd) >= 5:  # At least 5 observations
            test_sne.append((int(snid_str), lc))
            if len(test_sne) >= args.n_sne:
                break

    print(f"\nSelected {len(test_sne)} SNe for testing")

    # Global parameters from Stage 2 FIXED_TEST
    global_params_fixed = (5.381956298963422, -1.900259394073368, -3.8146692019334054)
    global_params_recovery = (10.770038588319618, -7.987900510670775, -6.907618767280434)

    L_peak = 1.5e43

    # Test with both parameter sets
    for param_name, global_params in [
        ("FIXED_TEST (k_J=5.38)", global_params_fixed),
        ("Recovery (k_J=10.77)", global_params_recovery)
    ]:
        print(f"\n{'='*80}")
        print(f"Testing: {param_name}")
        print(f"{'='*80}")
        print(f"Global params: k_J={global_params[0]:.3f}, η'={global_params[1]:.3f}, ξ={global_params[2]:.3f}")

        chi2_list = []
        reduced_chi2_list = []
        n_obs_total = 0
        n_failed = 0

        for snid, lc in test_sne:
            # Fit per-SN parameters with these global parameters
            result = optimize_single_sn(
                snid=str(snid),
                lc_data=lc,
                global_params=global_params,
                max_iters=500,
                tol=1e-6,
                verbose=False,
                use_studentt=False
            )

            chi2 = result['chi2']
            if np.isnan(chi2) or chi2 > 1e10 or not result['ok']:
                n_failed += 1
                continue

            n_obs = len(lc.mjd)
            dof = n_obs - 4  # 4 per-SN parameters
            reduced_chi2 = chi2 / dof if dof > 0 else np.nan

            chi2_list.append(chi2)
            reduced_chi2_list.append(reduced_chi2)
            n_obs_total += n_obs

        # Statistics
        if len(chi2_list) > 0:
            total_chi2 = np.sum(chi2_list)
            mean_chi2 = np.mean(chi2_list)
            median_chi2 = np.median(chi2_list)
            mean_reduced = np.mean(reduced_chi2_list)
            median_reduced = np.median(reduced_chi2_list)

            print(f"\nResults:")
            print(f"  Successfully fitted: {len(chi2_list)}/{len(test_sne)} SNe")
            print(f"  Failed fits: {n_failed}")
            print(f"  Total observations: {n_obs_total}")
            print(f"  Total chi²: {total_chi2:.1f}")
            print(f"  Mean chi² per SN: {mean_chi2:.2f}")
            print(f"  Median chi² per SN: {median_chi2:.2f}")
            print(f"  Mean reduced chi²: {mean_reduced:.3f}")
            print(f"  Median reduced chi²: {median_reduced:.3f}")

            # Show distribution
            bins = [0, 10, 25, 50, 100, 500, 1000, 10000, np.inf]
            hist, _ = np.histogram(chi2_list, bins=bins)
            print(f"\nChi² distribution:")
            for i in range(len(bins)-1):
                print(f"  {bins[i]:.0f} - {bins[i+1]:.0f}: {hist[i]} SNe")
        else:
            print(f"\nERROR: All fits failed!")

    return 0

if __name__ == '__main__':
    sys.exit(main())
