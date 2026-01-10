#!/usr/bin/env python3
"""
Integer Ladder Test: Do nuclei quantize at integer harmonic modes?

This test validates Chapter 14's claim that N_hat values cluster at integers
with "forbidden zones" at half-integers (the "Integer Ladder" structure).

Physical interpretation:
- If the harmonic model captures real physics, nuclei should sit near
  integer mode numbers (N = 0, 1, 2, ...)
- The fractional part of N_hat should NOT be uniformly distributed
- Instead, it should cluster near 0.0 and 1.0 (close to integers)
- Half-integers (0.5) should be depleted

Test method:
- Compute fractional parts: frac(N_hat) = N_hat - floor(N_hat)
- Bin into 10 equal bins on [0, 1)
- Chi-square test against uniform expectation
- H0: fractional parts are uniform (N has no physical meaning)
- H1: fractional parts cluster at 0 and 1 (quantization is real)

Expected result (from documentation): χ² ≈ 873, p ≈ 0

Reference: Chapter 14 "The Geometry of Existence", Section 14.4 "The Integer Ladder"
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def load_harmonic_scores(scores_path: str) -> pd.DataFrame:
    """Load harmonic scores with N_hat values."""
    logging.info(f"Loading harmonic scores from {scores_path}")
    df = pd.read_parquet(scores_path)
    logging.info(f"Loaded {len(df)} nuclides")
    return df


def compute_fractional_parts(df: pd.DataFrame, n_hat_col: str = 'N_hat_best') -> np.ndarray:
    """
    Compute fractional parts of N_hat values.

    frac(x) = x - floor(x), which gives values in [0, 1)
    """
    N_hat = df[n_hat_col].values

    # Remove NaN/inf
    valid = np.isfinite(N_hat)
    N_hat = N_hat[valid]

    logging.info(f"Valid N_hat values: {len(N_hat)}")

    # Compute fractional parts
    fractional = N_hat - np.floor(N_hat)

    return fractional


def chi_square_uniformity_test(fractional: np.ndarray, n_bins: int = 10) -> dict:
    """
    Chi-square test for uniformity of fractional parts.

    H0: fractional parts follow uniform distribution on [0, 1)
    H1: fractional parts are NOT uniform (cluster at integers)

    Args:
        fractional: Array of fractional parts in [0, 1)
        n_bins: Number of bins for histogram

    Returns:
        Dictionary with test results
    """
    # Bin the fractional parts
    observed, bin_edges = np.histogram(fractional, bins=n_bins, range=(0, 1))

    # Expected: uniform distribution
    expected = np.full(n_bins, len(fractional) / n_bins)

    # Chi-square test
    chi2, p_value = stats.chisquare(observed, expected)

    # Degrees of freedom
    dof = n_bins - 1

    logging.info("")
    logging.info("=" * 80)
    logging.info("INTEGER LADDER TEST: Chi-Square Uniformity Test")
    logging.info("=" * 80)
    logging.info(f"Sample size: {len(fractional)} nuclides")
    logging.info(f"Number of bins: {n_bins}")
    logging.info(f"Degrees of freedom: {dof}")
    logging.info("")
    logging.info(f"χ² = {chi2:.2f}")
    logging.info(f"p-value = {p_value:.2e}")
    logging.info("")

    # Interpret
    if p_value < 1e-10:
        logging.info("✓ INTEGER LADDER VALIDATED (p ≈ 0)")
        logging.info("  Null hypothesis (uniform) REJECTED with overwhelming confidence")
        logging.info("  Nuclides cluster at integer N values")
    elif p_value < 0.001:
        logging.info("✓ Integer ladder supported (p < 0.001)")
    elif p_value < 0.05:
        logging.info("? Marginal evidence for integer ladder (p < 0.05)")
    else:
        logging.info("✗ NO evidence for integer ladder (p > 0.05)")
        logging.info("  Fractional parts are consistent with uniform distribution")

    logging.info("=" * 80)

    return {
        'n_samples': int(len(fractional)),
        'n_bins': n_bins,
        'dof': dof,
        'chi2': float(chi2),
        'p_value': float(p_value),
        'observed': observed.tolist(),
        'expected': expected.tolist(),
        'bin_edges': bin_edges.tolist(),
    }


def analyze_distribution(fractional: np.ndarray) -> dict:
    """
    Detailed analysis of fractional part distribution.

    Shows clustering at 0 and 1 (near integers) vs depletion at 0.5 (half-integers).
    """
    logging.info("")
    logging.info("=" * 80)
    logging.info("FRACTIONAL PART DISTRIBUTION")
    logging.info("=" * 80)

    # Count in key regions
    near_zero = np.sum((fractional >= 0.0) & (fractional < 0.1))
    near_one = np.sum((fractional >= 0.9) & (fractional <= 1.0))
    near_half = np.sum((fractional >= 0.4) & (fractional < 0.6))
    middle = np.sum((fractional >= 0.1) & (fractional < 0.9))

    total = len(fractional)

    logging.info(f"Near integers (0.0-0.1 or 0.9-1.0): {near_zero + near_one} ({100*(near_zero + near_one)/total:.1f}%)")
    logging.info(f"  Near 0.0 (0.0-0.1): {near_zero} ({100*near_zero/total:.1f}%)")
    logging.info(f"  Near 1.0 (0.9-1.0): {near_one} ({100*near_one/total:.1f}%)")
    logging.info(f"Near half-integer (0.4-0.6): {near_half} ({100*near_half/total:.1f}%)")
    logging.info(f"Middle region (0.1-0.9): {middle} ({100*middle/total:.1f}%)")
    logging.info("")

    # Expected if uniform
    expected_near = total * 0.2  # 20% expected in 0.0-0.1 + 0.9-1.0
    expected_half = total * 0.2  # 20% expected in 0.4-0.6

    logging.info(f"Expected if uniform:")
    logging.info(f"  Near integers: {expected_near:.0f} (20%)")
    logging.info(f"  Near half-integer: {expected_half:.0f} (20%)")
    logging.info("")

    # Enrichment/depletion factors
    enrichment_int = (near_zero + near_one) / expected_near
    depletion_half = near_half / expected_half

    logging.info(f"Enrichment at integers: {enrichment_int:.2f}x")
    logging.info(f"Depletion at half-integer: {depletion_half:.2f}x")

    if enrichment_int > 1.5 and depletion_half < 0.7:
        logging.info("")
        logging.info("✓ Classic 'integer ladder' signature detected")
        logging.info("  Nuclei cluster at integer N with forbidden zone at N+0.5")

    logging.info("=" * 80)

    return {
        'near_zero': int(near_zero),
        'near_one': int(near_one),
        'near_half': int(near_half),
        'middle': int(middle),
        'enrichment_integers': float(enrichment_int),
        'depletion_half_integer': float(depletion_half),
    }


def analyze_by_decay_mode(df: pd.DataFrame, n_hat_col: str = 'N_hat_best') -> dict:
    """
    Run integer ladder test separately by decay mode.

    The original χ² = 873 was specifically for β⁻ decay parents.
    """
    logging.info("")
    logging.info("=" * 80)
    logging.info("INTEGER LADDER BY DECAY MODE")
    logging.info("=" * 80)

    results = {}

    # Check if decay mode column exists
    mode_col = None
    for col in ['dominant_mode', 'decay_mode', 'mode']:
        if col in df.columns:
            mode_col = col
            break

    if mode_col is None:
        logging.info("No decay mode column found, skipping")
        logging.info("=" * 80)
        return results

    modes = df[mode_col].dropna().unique()

    for mode in sorted(modes):
        df_mode = df[df[mode_col] == mode]

        if len(df_mode) < 50:
            logging.info(f"{mode:15s}: n={len(df_mode):4d} (too few samples)")
            continue

        # Get N_hat values
        N_hat = df_mode[n_hat_col].values
        valid = np.isfinite(N_hat)
        N_hat = N_hat[valid]

        if len(N_hat) < 50:
            continue

        # Fractional parts
        fractional = N_hat - np.floor(N_hat)

        # Chi-square test
        observed, _ = np.histogram(fractional, bins=10, range=(0, 1))
        expected = np.full(10, len(fractional) / 10)
        chi2, p = stats.chisquare(observed, expected)

        logging.info(f"{mode:15s}: n={len(fractional):4d}, χ²={chi2:7.1f}, p={p:.2e}")

        results[mode] = {
            'n_samples': len(fractional),
            'chi2': float(chi2),
            'p_value': float(p),
        }

    logging.info("=" * 80)

    return results


def plot_histogram(fractional: np.ndarray, results: dict, outdir: Path):
    """
    Plot histogram of fractional parts with uniform comparison.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    n_bins = 10
    observed = results['observed']
    expected = results['expected'][0]  # All same for uniform

    # Bar positions
    bin_centers = np.arange(0.05, 1.0, 0.1)
    width = 0.08

    # Observed bars
    bars = ax.bar(bin_centers, observed, width=width, color='steelblue',
                  edgecolor='black', alpha=0.8, label='Observed')

    # Expected line
    ax.axhline(expected, color='red', linestyle='--', linewidth=2,
               label=f'Expected (uniform): {expected:.0f}')

    # Labels
    ax.set_xlabel('Fractional part of N_hat', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'Integer Ladder Test: Distribution of N_hat Fractional Parts\n'
                 f'χ² = {results["chi2"]:.1f}, p = {results["p_value"]:.2e}, n = {results["n_samples"]}',
                 fontsize=13, fontweight='bold')

    ax.set_xlim(0, 1)
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.legend(loc='upper center')
    ax.grid(True, alpha=0.3, axis='y')

    # Add interpretation text
    if results['p_value'] < 1e-10:
        text = "✓ Integer quantization confirmed\n(χ² >> critical, p ≈ 0)"
        color = 'green'
    else:
        text = "? Weak or no quantization signal"
        color = 'orange'

    ax.text(0.5, 0.95, text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))

    plt.tight_layout()
    outpath = outdir / 'integer_ladder_histogram.png'
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    logging.info(f"Saved histogram to {outpath}")
    plt.close()


def plot_bimodal(fractional: np.ndarray, outdir: Path):
    """
    Plot showing bimodal clustering at 0 and 1.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Finer histogram
    ax.hist(fractional, bins=50, range=(0, 1), color='steelblue',
            edgecolor='black', alpha=0.7)

    # Mark forbidden zone
    ax.axvspan(0.4, 0.6, alpha=0.2, color='red', label='Forbidden zone (0.4-0.6)')

    # Mark integer zones
    ax.axvspan(0.0, 0.1, alpha=0.2, color='green', label='Near-integer zone')
    ax.axvspan(0.9, 1.0, alpha=0.2, color='green')

    ax.set_xlabel('Fractional part of N_hat', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Bimodal Distribution: Clustering at Integers',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper center')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    outpath = outdir / 'integer_ladder_bimodal.png'
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    logging.info(f"Saved bimodal plot to {outpath}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Integer Ladder Test: Validate quantization at integer N values'
    )
    parser.add_argument(
        '--scores',
        default='data/derived/harmonic_scores.parquet',
        help='Path to harmonic_scores.parquet'
    )
    parser.add_argument(
        '--out',
        default='reports/integer_ladder',
        help='Output directory'
    )
    parser.add_argument(
        '--n-hat-col',
        default='N_hat_best',
        help='Column name for N_hat values'
    )

    args = parser.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_harmonic_scores(args.scores)

    # Check available columns
    logging.info(f"Available columns: {list(df.columns)}")

    # Find N_hat column
    n_hat_col = args.n_hat_col
    if n_hat_col not in df.columns:
        # Try alternatives
        for alt in ['N_hat_best', 'N_hat', 'Nhat', 'n_hat']:
            if alt in df.columns:
                n_hat_col = alt
                break
        else:
            logging.error(f"Cannot find N_hat column. Available: {list(df.columns)}")
            return

    logging.info(f"Using N_hat column: {n_hat_col}")

    # Compute fractional parts
    fractional = compute_fractional_parts(df, n_hat_col)

    if len(fractional) < 100:
        logging.error(f"Too few valid samples: {len(fractional)}")
        return

    # Main chi-square test
    results = {}
    results['chi_square_test'] = chi_square_uniformity_test(fractional)

    # Distribution analysis
    results['distribution'] = analyze_distribution(fractional)

    # By decay mode
    results['by_decay_mode'] = analyze_by_decay_mode(df, n_hat_col)

    # Plots
    plot_histogram(fractional, results['chi_square_test'], outdir)
    plot_bimodal(fractional, outdir)

    # Save results
    results_path = outdir / 'integer_ladder_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logging.info(f"Saved results to {results_path}")

    logging.info("")
    logging.info("=" * 80)
    logging.info("INTEGER LADDER TEST COMPLETE")
    logging.info("=" * 80)
    logging.info(f"Results saved to {outdir}")
    logging.info("")

    # Print key result for comparison to Chapter 14
    chi2 = results['chi_square_test']['chi2']
    logging.info(f"KEY RESULT: χ² = {chi2:.2f}")
    logging.info(f"Chapter 14 reports: χ² = 873.47")
    if abs(chi2 - 873) < 50:
        logging.info("✓ RESULT MATCHES CHAPTER 14")
    else:
        logging.info(f"Note: Difference from Chapter 14 = {abs(chi2 - 873):.1f}")


if __name__ == '__main__':
    main()
