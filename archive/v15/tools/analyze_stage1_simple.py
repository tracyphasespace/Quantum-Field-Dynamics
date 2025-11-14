#!/usr/bin/env python3
"""Simple Stage 1 Results Analysis"""
import json
import numpy as np
from pathlib import Path
from collections import Counter

def analyze_stage1_results(stage1_dir):
    """Analyze Stage 1 results and print summary statistics."""
    stage1_path = Path(stage1_dir)

    # Get all SN directories
    sn_dirs = [d for d in stage1_path.iterdir() if d.is_dir()]
    total_sne = len(sn_dirs)

    print(f"\n{'='*70}")
    print(f"STAGE 1 RESULTS ANALYSIS: {stage1_dir}")
    print(f"{'='*70}\n")

    # Count status types
    statuses = []
    chi2_values = []
    iters_list = []
    ok_flags = []

    for sn_dir in sn_dirs:
        status_file = sn_dir / "status.txt"
        metrics_file = sn_dir / "metrics.json"

        if status_file.exists():
            with open(status_file, 'r') as f:
                status = f.read().strip()
                statuses.append(status)

        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
                chi2_values.append(metrics.get('chi2', np.nan))
                iters_list.append(metrics.get('iters', np.nan))
                ok_flags.append(metrics.get('ok', False))

    # Status summary
    status_counts = Counter(statuses)
    print(f"Total SNe: {total_sne}")
    print(f"\nStatus Breakdown:")
    for status, count in status_counts.most_common():
        pct = 100 * count / total_sne
        print(f"  {status:30s}: {count:5d} ({pct:5.1f}%)")

    # Chi-squared statistics
    chi2_array = np.array([c for c in chi2_values if np.isfinite(c)])
    if len(chi2_array) > 0:
        print(f"\nChi-squared Statistics (finite values only):")
        print(f"  Count: {len(chi2_array)}")
        print(f"  Min:   {chi2_array.min():.2f}")
        print(f"  25%:   {np.percentile(chi2_array, 25):.2f}")
        print(f"  Median: {np.median(chi2_array):.2f}")
        print(f"  75%:   {np.percentile(chi2_array, 75):.2f}")
        print(f"  Max:   {chi2_array.max():.2e}")

    # Iteration statistics
    iters_array = np.array([i for i in iters_list if np.isfinite(i)])
    if len(iters_array) > 0:
        print(f"\nOptimizer Iterations:")
        print(f"  Min:    {iters_array.min():.0f}")
        print(f"  Median: {np.median(iters_array):.0f}")
        print(f"  Max:    {iters_array.max():.0f}")

    # OK flag summary
    num_ok = sum(ok_flags)
    num_not_ok = len(ok_flags) - num_ok
    print(f"\nOptimizer Success (ok=True):")
    print(f"  Success (ok=True):  {num_ok:5d} ({100*num_ok/len(ok_flags):.1f}%)")
    print(f"  Failed  (ok=False): {num_not_ok:5d} ({100*num_not_ok/len(ok_flags):.1f}%)")

    # Chi2 distribution by bins
    reasonable_chi2 = chi2_array[chi2_array < 1e6]  # Exclude pathological cases
    if len(reasonable_chi2) > 0:
        print(f"\nChi-squared Distribution (chi2 < 1e6 only, N={len(reasonable_chi2)}):")
        bins = [0, 100, 200, 500, 1000, 5000, 1e6]
        for i in range(len(bins)-1):
            low, high = bins[i], bins[i+1]
            count = np.sum((reasonable_chi2 >= low) & (reasonable_chi2 < high))
            pct = 100 * count / len(reasonable_chi2)
            print(f"  [{low:6.0f}, {high:6.0f}): {count:5d} ({pct:5.1f}%)")

    print(f"\n{'='*70}\n")

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        stage1_dir = sys.argv[1]
    else:
        stage1_dir = "results/v15_production/stage1"

    analyze_stage1_results(stage1_dir)
