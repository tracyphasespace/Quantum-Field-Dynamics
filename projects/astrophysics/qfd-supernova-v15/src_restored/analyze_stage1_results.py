#!/usr/bin/env python3
"""
Analyze Stage 1 results properly
"""
import json
import numpy as np
from pathlib import Path

def main():
    stage1_dir = Path("results/v15_stage1_production")

    all_results = []
    quality_results = []

    for sn_dir in stage1_dir.iterdir():
        if not sn_dir.is_dir():
            continue

        snid = sn_dir.name
        metrics_file = sn_dir / "metrics.json"
        persn_file = sn_dir / "persn_best.npy"

        if not metrics_file.exists():
            continue

        try:
            with open(metrics_file) as f:
                metrics = json.load(f)

            # Load persn_best if available
            if persn_file.exists():
                persn_best = np.load(persn_file)
            else:
                persn_best = None

            # Compute chi2/obs (need to get n_obs from somewhere)
            # For now, estimate from chi2 and logL
            chi2 = metrics['chi2']

            result = {
                'snid': snid,
                'chi2': chi2,
                'iters': metrics['iters'],
                'grad_norm': metrics['grad_norm'],
                'persn_best': persn_best
            }

            all_results.append(result)

            # Quality filter: chi2 < 2000 (very permissive for now)
            if chi2 < 2000 and metrics['iters'] >= 5:
                quality_results.append(result)

        except Exception as e:
            print(f"Warning: Failed to load {snid}: {e}")
            continue

    print(f"Total SNe: {len(all_results)}")
    print(f"Quality SNe (chi2 < 2000, iters >= 5): {len(quality_results)}")
    print()

    if all_results:
        chi2_vals = [r['chi2'] for r in all_results if r['chi2'] < 1e6]
        print(f"Chi2 statistics (excluding outliers > 1e6):")
        print(f"  Median: {np.median(chi2_vals):.2f}")
        print(f"  Mean: {np.mean(chi2_vals):.2f}")
        print(f"  Q1-Q3: {np.percentile(chi2_vals, 25):.2f} - {np.percentile(chi2_vals, 75):.2f}")
        print()

        # Iteration statistics
        iters_vals = [r['iters'] for r in all_results]
        print(f"Iteration statistics:")
        print(f"  Median: {np.median(iters_vals):.0f}")
        print(f"  Mean: {np.mean(iters_vals):.1f}")
        print()

        # Distribution
        excellent = sum(1 for r in all_results if r['chi2'] < 20)
        good = sum(1 for r in all_results if 20 <= r['chi2'] < 100)
        medium = sum(1 for r in all_results if 100 <= r['chi2'] < 1000)
        poor = sum(1 for r in all_results if r['chi2'] >= 1000)

        print(f"Quality distribution:")
        print(f"  Excellent (chi2 < 20): {excellent} ({excellent/len(all_results)*100:.1f}%)")
        print(f"  Good (20-100): {good} ({good/len(all_results)*100:.1f}%)")
        print(f"  Medium (100-1000): {medium} ({medium/len(all_results)*100:.1f}%)")
        print(f"  Poor (>1000): {poor} ({poor/len(all_results)*100:.1f}%)")
        print()

    if len(quality_results) >= 50:
        print("✅ READY FOR STAGE 2!")
        print(f"   {len(quality_results)} quality SNe available")
        return 0
    else:
        print("⚠️  NOT ENOUGH QUALITY SNe FOR STAGE 2")
        print(f"   Only {len(quality_results)} quality SNe, need at least 50")
        return 1

if __name__ == '__main__':
    import sys
    sys.exit(main())
