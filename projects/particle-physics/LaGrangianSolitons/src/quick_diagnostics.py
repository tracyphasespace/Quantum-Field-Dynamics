#!/usr/bin/env python3
"""
Quick diagnostic analysis of harmonic scores.

Previews Experiment 2 (stability selector) by comparing
epsilon distributions for stable vs unstable nuclides.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats


def load_scores(scores_file):
    """Load harmonic scores."""
    df = pd.read_parquet(scores_file)
    print(f"Loaded {len(df)} scored nuclides")
    return df


def analyze_stability(df):
    """
    Compare epsilon distributions for stable vs unstable nuclides.

    Preview of Experiment 2.
    """
    print("\n" + "="*80)
    print("STABILITY SELECTOR DIAGNOSTIC (Experiment 2 Preview)")
    print("="*80)

    # Split by stability
    stable = df[df['is_stable']]
    unstable = df[~df['is_stable']]

    print(f"\nDataset split:")
    print(f"  Stable:   {len(stable):4d} nuclides")
    print(f"  Unstable: {len(unstable):4d} nuclides")

    # Epsilon statistics
    eps_stable = stable['epsilon_best'].values
    eps_unstable = unstable['epsilon_best'].values

    print(f"\nEpsilon statistics:")
    print(f"  Stable:   mean={eps_stable.mean():.4f}, median={np.median(eps_stable):.4f}, std={eps_stable.std():.4f}")
    print(f"  Unstable: mean={eps_unstable.mean():.4f}, median={np.median(eps_unstable):.4f}, std={eps_unstable.std():.4f}")

    # Effect sizes
    mean_diff = eps_stable.mean() - eps_unstable.mean()
    median_diff = np.median(eps_stable) - np.median(eps_unstable)

    print(f"\nEffect sizes:")
    print(f"  Mean difference:   {mean_diff:.4f} (stable - unstable)")
    print(f"  Median difference: {median_diff:.4f}")

    if mean_diff < 0:
        print("  ✓ Stable nuclides have LOWER epsilon (as predicted)")
    else:
        print("  ✗ Stable nuclides have HIGHER epsilon (unexpected!)")

    # KS test
    ks_stat, ks_pvalue = stats.ks_2samp(eps_stable, eps_unstable)
    print(f"\nKolmogorov-Smirnov test:")
    print(f"  D = {ks_stat:.4f}")
    print(f"  p-value = {ks_pvalue:.2e}")

    if ks_pvalue < 0.001:
        print("  ✓ Distributions are SIGNIFICANTLY different (p < 0.001)")
    elif ks_pvalue < 0.05:
        print("  ✓ Distributions are significantly different (p < 0.05)")
    else:
        print("  ✗ Distributions are NOT significantly different")

    # Category breakdown
    print(f"\nCategory breakdown:")
    for category in ['harmonic', 'near_harmonic', 'dissonant']:
        n_stable = (stable['category'] == category).sum()
        n_unstable = (unstable['category'] == category).sum()
        pct_stable = 100 * n_stable / len(stable)
        pct_unstable = 100 * n_unstable / len(unstable)

        print(f"  {category:15s}:")
        print(f"    Stable:   {n_stable:3d} ({pct_stable:5.1f}%)")
        print(f"    Unstable: {n_unstable:4d} ({pct_unstable:5.1f}%)")


def analyze_families(df):
    """Analyze family distribution."""
    print("\n" + "="*80)
    print("FAMILY DISTRIBUTION")
    print("="*80)

    stable = df[df['is_stable']]
    unstable = df[~df['is_stable']]

    for family in ['A', 'B', 'C']:
        n_stable = (stable['best_family'] == family).sum()
        n_unstable = (unstable['best_family'] == family).sum()
        pct_stable = 100 * n_stable / len(stable)
        pct_unstable = 100 * n_unstable / len(unstable)

        print(f"\nFamily {family}:")
        print(f"  Stable:   {n_stable:3d} ({pct_stable:5.1f}%)")
        print(f"  Unstable: {n_unstable:4d} ({pct_unstable:5.1f}%)")


def analyze_by_decay_mode(df):
    """Analyze epsilon by decay mode."""
    print("\n" + "="*80)
    print("EPSILON BY DECAY MODE (Experiment 3 Preview)")
    print("="*80)

    unstable = df[~df['is_stable']]

    # Group by decay mode
    for mode in ['alpha', 'beta_minus', 'beta_plus', 'EC', 'proton', 'neutron', 'fission']:
        subset = unstable[unstable['dominant_mode'] == mode]
        if len(subset) == 0:
            continue

        eps = subset['epsilon_best'].values
        print(f"\n{mode:15s} (n={len(subset):4d}):")
        print(f"  Mean ε:   {eps.mean():.4f}")
        print(f"  Median ε: {np.median(eps):.4f}")
        print(f"  Std ε:    {eps.std():.4f}")

        # Category breakdown
        n_harmonic = (subset['category'] == 'harmonic').sum()
        n_near = (subset['category'] == 'near_harmonic').sum()
        n_diss = (subset['category'] == 'dissonant').sum()
        print(f"  Harmonic: {n_harmonic:3d} ({100*n_harmonic/len(subset):5.1f}%)")
        print(f"  Near:     {n_near:3d} ({100*n_near/len(subset):5.1f}%)")
        print(f"  Dissonant:{n_diss:4d} ({100*n_diss/len(subset):5.1f}%)")


def analyze_extremes(df):
    """Identify nuclides with extreme epsilon values."""
    print("\n" + "="*80)
    print("EXTREME EPSILON VALUES")
    print("="*80)

    # Lowest epsilon (most harmonic)
    print("\nMost harmonic (lowest ε):")
    df_sorted = df.sort_values('epsilon_best').head(10)
    for idx, row in df_sorted.iterrows():
        print(f"  {row['element']:6s} (A={row['A']:3d}, Z={row['Z']:3d}): "
              f"ε={row['epsilon_best']:.4f}, family={row['best_family']}, "
              f"stable={row['is_stable']}")

    # Highest epsilon (most dissonant)
    print("\nMost dissonant (highest ε):")
    df_sorted = df.sort_values('epsilon_best', ascending=False).head(10)
    for idx, row in df_sorted.iterrows():
        print(f"  {row['element']:6s} (A={row['A']:3d}, Z={row['Z']:3d}): "
              f"ε={row['epsilon_best']:.4f}, family={row['best_family']}, "
              f"stable={row['is_stable']}")


def main():
    """Run diagnostics."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Quick diagnostic analysis of harmonic scores'
    )
    parser.add_argument(
        '--scores',
        default='data/derived/harmonic_scores.parquet',
        help='Path to harmonic_scores.parquet'
    )

    args = parser.parse_args()

    # Load data
    df = load_scores(args.scores)

    # Run analyses
    analyze_stability(df)
    analyze_families(df)
    analyze_by_decay_mode(df)
    analyze_extremes(df)

    print("\n" + "="*80)
    print("DIAGNOSTIC COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
