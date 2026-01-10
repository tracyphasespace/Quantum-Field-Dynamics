#!/usr/bin/env python3
"""
Tacoma Narrows Test: Does ε anti-correlate with half-life?

This test validates the reinterpretation that low ε (harmonic) predicts
INSTABILITY (short half-life), not existence, analogous to the Tacoma
Narrows Bridge resonance-driven collapse.

Physical interpretation:
- Low ε → resonant coupling → enhanced decay → short half-life
- High ε → off-resonance → damped decay → long half-life
- Stable nuclides → high ε (anti-resonant, like modern bridges)

Expected: Positive correlation between ε and log10(half_life)
If passes: Model predicts instability ✓
If fails: Model has no predictive power ✗
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def load_scored_nuclides(scores_path: str) -> pd.DataFrame:
    """Load scored nuclides with half-life data."""
    logging.info(f"Loading scored nuclides from {scores_path}")
    df = pd.read_parquet(scores_path)

    logging.info(f"Loaded {len(df)} nuclides")
    logging.info(f"  Stable: {df['is_stable'].sum()}")
    logging.info(f"  Unstable: {(~df['is_stable']).sum()}")

    # Check half-life data availability
    has_halflife = df['half_life_s'].notna() & ~np.isinf(df['half_life_s'])
    logging.info(f"  With finite half-life: {has_halflife.sum()}")

    return df

def filter_unstable_with_halflife(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to unstable nuclides with known finite half-lives."""
    # Unstable only
    unstable = df[~df['is_stable']].copy()
    logging.info(f"Unstable nuclides: {len(unstable)}")

    # With finite half-life
    has_halflife = unstable['half_life_s'].notna() & ~np.isinf(unstable['half_life_s'])
    df_filtered = unstable[has_halflife].copy()
    logging.info(f"With finite half-life: {len(df_filtered)}")

    # With epsilon score
    has_epsilon = df_filtered['epsilon_best'].notna()
    df_filtered = df_filtered[has_epsilon].copy()
    logging.info(f"With epsilon score: {len(df_filtered)}")

    # Compute log10(half_life)
    df_filtered['log10_halflife'] = np.log10(df_filtered['half_life_s'])

    return df_filtered

def compute_correlation(df: pd.DataFrame) -> dict:
    """Compute Spearman correlation between ε and log10(half-life)."""
    eps = df['epsilon_best'].values
    log_t = df['log10_halflife'].values

    # Spearman (rank correlation, robust to outliers)
    r_spearman, p_spearman = stats.spearmanr(eps, log_t)

    # Pearson (linear correlation, for comparison)
    r_pearson, p_pearson = stats.pearsonr(eps, log_t)

    # Kendall tau (alternative rank correlation)
    tau, p_tau = stats.kendalltau(eps, log_t)

    logging.info("")
    logging.info("="*80)
    logging.info("TACOMA NARROWS TEST: ε vs Half-Life Correlation")
    logging.info("="*80)
    logging.info(f"Sample size: {len(df)} unstable nuclides with finite half-life")
    logging.info("")
    logging.info(f"Spearman correlation: r = {r_spearman:+.4f}, p = {p_spearman:.2e}")
    logging.info(f"Pearson correlation:  r = {r_pearson:+.4f}, p = {p_pearson:.2e}")
    logging.info(f"Kendall tau:          τ = {tau:+.4f}, p = {p_tau:.2e}")
    logging.info("")

    # Interpret
    if r_spearman > 0 and p_spearman < 0.001:
        logging.info("✓ TACOMA NARROWS MODEL VALIDATED")
        logging.info("  Higher ε → Longer half-life")
        logging.info("  Model predicts INSTABILITY (low ε → rapid decay)")
    elif r_spearman < 0 and p_spearman < 0.001:
        logging.info("✗ TACOMA NARROWS MODEL CONTRADICTED")
        logging.info("  Higher ε → Shorter half-life (wrong direction!)")
    else:
        logging.info("? TACOMA NARROWS MODEL FAILS")
        logging.info("  No significant correlation (p > 0.001)")
        logging.info("  Harmonic model has no predictive power for half-life")

    logging.info("="*80)

    return {
        'n_samples': len(df),
        'spearman_r': float(r_spearman),
        'spearman_p': float(p_spearman),
        'pearson_r': float(r_pearson),
        'pearson_p': float(p_pearson),
        'kendall_tau': float(tau),
        'kendall_p': float(p_tau),
    }

def analyze_by_decay_mode(df: pd.DataFrame) -> dict:
    """Analyze correlation separately by decay mode."""
    logging.info("")
    logging.info("="*80)
    logging.info("CORRELATION BY DECAY MODE")
    logging.info("="*80)

    results = {}

    # Check if decay mode column exists
    if 'dominant_mode' not in df.columns:
        logging.info("No decay mode column found, skipping analysis")
        logging.info("="*80)
        return results

    # Get unique decay modes
    decay_modes = df['dominant_mode'].unique()

    for mode in decay_modes:
        if pd.isna(mode):
            continue

        df_mode = df[df['dominant_mode'] == mode]

        if len(df_mode) < 10:
            logging.info(f"{mode}: n={len(df_mode)} (too few samples)")
            continue

        eps = df_mode['epsilon_best'].values
        log_t = df_mode['log10_halflife'].values

        r, p = stats.spearmanr(eps, log_t)

        logging.info(f"{mode:10s}: n={len(df_mode):5d}, r={r:+.3f}, p={p:.2e}")

        results[mode] = {
            'n_samples': len(df_mode),
            'spearman_r': float(r),
            'spearman_p': float(p),
        }

    logging.info("="*80)

    return results

def analyze_by_mass_region(df: pd.DataFrame) -> dict:
    """Analyze correlation separately by mass region."""
    logging.info("")
    logging.info("="*80)
    logging.info("CORRELATION BY MASS REGION")
    logging.info("="*80)

    results = {}

    # Define mass regions
    regions = [
        ('light', 0, 60),
        ('medium', 60, 150),
        ('heavy', 150, 1000),
    ]

    for name, A_min, A_max in regions:
        df_region = df[(df['A'] >= A_min) & (df['A'] < A_max)]

        if len(df_region) < 10:
            logging.info(f"{name:10s}: n={len(df_region)} (too few samples)")
            continue

        eps = df_region['epsilon_best'].values
        log_t = df_region['log10_halflife'].values

        r, p = stats.spearmanr(eps, log_t)

        logging.info(f"{name:10s}: A∈[{A_min:3d},{A_max:3d}), n={len(df_region):5d}, r={r:+.3f}, p={p:.2e}")

        results[name] = {
            'A_min': A_min,
            'A_max': A_max,
            'n_samples': len(df_region),
            'spearman_r': float(r),
            'spearman_p': float(p),
        }

    logging.info("="*80)

    return results

def analyze_by_family(df: pd.DataFrame) -> dict:
    """Analyze correlation separately by harmonic family."""
    logging.info("")
    logging.info("="*80)
    logging.info("CORRELATION BY HARMONIC FAMILY")
    logging.info("="*80)

    results = {}

    # Check if family column exists
    if 'best_family' not in df.columns:
        logging.info("No family column found, skipping analysis")
        logging.info("="*80)
        return results

    families = df['best_family'].unique()

    for family in sorted(families):
        df_family = df[df['best_family'] == family]

        if len(df_family) < 10:
            logging.info(f"{family}: n={len(df_family)} (too few samples)")
            continue

        eps = df_family['epsilon_best'].values
        log_t = df_family['log10_halflife'].values

        r, p = stats.spearmanr(eps, log_t)

        logging.info(f"{family}: n={len(df_family):5d}, r={r:+.3f}, p={p:.2e}")

        results[family] = {
            'n_samples': len(df_family),
            'spearman_r': float(r),
            'spearman_p': float(p),
        }

    logging.info("="*80)

    return results

def plot_correlation(df: pd.DataFrame, outdir: Path):
    """Plot ε vs log10(half-life) with correlation."""
    eps = df['epsilon_best'].values
    log_t = df['log10_halflife'].values

    r, p = stats.spearmanr(eps, log_t)

    fig, ax = plt.subplots(figsize=(10, 7))

    # Scatter plot
    ax.scatter(eps, log_t, alpha=0.3, s=1, color='black')

    # Linear fit (for visualization)
    slope, intercept = np.polyfit(eps, log_t, 1)
    eps_fit = np.linspace(eps.min(), eps.max(), 100)
    log_t_fit = slope * eps_fit + intercept
    ax.plot(eps_fit, log_t_fit, 'r--', linewidth=1.5, alpha=0.7,
            label=f'Linear fit: slope={slope:.2f}')

    ax.set_xlabel('Epsilon (dissonance)', fontsize=12)
    ax.set_ylabel('log₁₀(Half-life [s])', fontsize=12)
    ax.set_title(f'Tacoma Narrows Test: ε vs Half-Life\n' +
                 f'Spearman r = {r:+.4f}, p = {p:.2e}, n = {len(df)}',
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Add interpretation
    if r > 0 and p < 0.001:
        interpretation = "✓ Model predicts instability\n(higher ε → longer t₁/₂)"
        color = 'green'
    elif r < 0 and p < 0.001:
        interpretation = "✗ Reversed correlation\n(higher ε → shorter t₁/₂)"
        color = 'red'
    else:
        interpretation = "? No significant correlation\n(p > 0.001)"
        color = 'orange'

    ax.text(0.05, 0.95, interpretation,
            transform=ax.transAxes,
            fontsize=11, fontweight='bold',
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))

    plt.tight_layout()
    outpath = outdir / 'tacoma_narrows_correlation.png'
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    logging.info(f"Saved correlation plot to {outpath}")
    plt.close()

def plot_by_decay_mode(df: pd.DataFrame, outdir: Path):
    """Plot ε vs log10(half-life) colored by decay mode."""
    # Check if decay mode column exists
    if 'dominant_mode' not in df.columns:
        logging.info("No decay mode column, skipping mode plot")
        return

    decay_modes = df['dominant_mode'].unique()

    # Filter to modes with enough samples
    mode_counts = df['dominant_mode'].value_counts()
    major_modes = mode_counts[mode_counts >= 50].index.tolist()

    fig, ax = plt.subplots(figsize=(12, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, len(major_modes)))

    for i, mode in enumerate(major_modes):
        df_mode = df[df['dominant_mode'] == mode]
        eps = df_mode['epsilon_best'].values
        log_t = df_mode['log10_halflife'].values

        r, p = stats.spearmanr(eps, log_t)

        ax.scatter(eps, log_t, alpha=0.5, s=10, color=colors[i],
                   label=f'{mode} (n={len(df_mode)}, r={r:+.2f})')

    ax.set_xlabel('Epsilon (dissonance)', fontsize=12)
    ax.set_ylabel('log₁₀(Half-life [s])', fontsize=12)
    ax.set_title('Tacoma Narrows Test by Decay Mode', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9)

    plt.tight_layout()
    outpath = outdir / 'tacoma_narrows_by_mode.png'
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    logging.info(f"Saved decay mode plot to {outpath}")
    plt.close()

def plot_by_mass_region(df: pd.DataFrame, outdir: Path):
    """Plot ε vs log10(half-life) by mass region."""
    regions = [
        ('light', 0, 60, 'blue'),
        ('medium', 60, 150, 'green'),
        ('heavy', 150, 1000, 'red'),
    ]

    fig, ax = plt.subplots(figsize=(12, 8))

    for name, A_min, A_max, color in regions:
        df_region = df[(df['A'] >= A_min) & (df['A'] < A_max)]

        if len(df_region) < 10:
            continue

        eps = df_region['epsilon_best'].values
        log_t = df_region['log10_halflife'].values

        r, p = stats.spearmanr(eps, log_t)

        ax.scatter(eps, log_t, alpha=0.4, s=10, color=color,
                   label=f'{name} A∈[{A_min},{A_max}) (n={len(df_region)}, r={r:+.2f})')

    ax.set_xlabel('Epsilon (dissonance)', fontsize=12)
    ax.set_ylabel('log₁₀(Half-life [s])', fontsize=12)
    ax.set_title('Tacoma Narrows Test by Mass Region', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)

    plt.tight_layout()
    outpath = outdir / 'tacoma_narrows_by_mass.png'
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    logging.info(f"Saved mass region plot to {outpath}")
    plt.close()

def check_magic_numbers(df: pd.DataFrame) -> dict:
    """Check if doubly-magic nuclides have higher ε (anti-resonant)."""
    logging.info("")
    logging.info("="*80)
    logging.info("MAGIC NUMBER TEST: Anti-Resonance Hypothesis")
    logging.info("="*80)

    magic_Z = [2, 8, 20, 28, 50, 82]
    magic_N = [2, 8, 20, 28, 50, 82, 126]

    # Flag doubly-magic
    df['is_doubly_magic'] = df['Z'].isin(magic_Z) & df['N'].isin(magic_N)

    n_magic = df['is_doubly_magic'].sum()
    n_normal = (~df['is_doubly_magic']).sum()

    logging.info(f"Doubly-magic nuclides: {n_magic}")
    logging.info(f"Normal nuclides: {n_normal}")

    if n_magic < 3:
        logging.info("Too few doubly-magic nuclides in unstable set")
        logging.info("="*80)
        return {}

    eps_magic = df[df['is_doubly_magic']]['epsilon_best'].values
    eps_normal = df[~df['is_doubly_magic']]['epsilon_best'].values

    mean_magic = np.mean(eps_magic)
    mean_normal = np.mean(eps_normal)
    diff = mean_magic - mean_normal

    # KS test
    ks_stat, ks_p = stats.ks_2samp(eps_magic, eps_normal)

    # t-test (for comparison)
    t_stat, t_p = stats.ttest_ind(eps_magic, eps_normal)

    logging.info(f"")
    logging.info(f"Mean ε (doubly-magic): {mean_magic:.4f}")
    logging.info(f"Mean ε (normal):       {mean_normal:.4f}")
    logging.info(f"Difference:            {diff:+.4f}")
    logging.info(f"")
    logging.info(f"KS test: D = {ks_stat:.3f}, p = {ks_p:.2e}")
    logging.info(f"t-test:  t = {t_stat:.3f}, p = {t_p:.2e}")
    logging.info("")

    if diff > 0 and (ks_p < 0.05 or t_p < 0.05):
        logging.info("✓ Magic numbers have HIGHER ε (anti-resonant, as predicted)")
    elif diff < 0 and (ks_p < 0.05 or t_p < 0.05):
        logging.info("✗ Magic numbers have LOWER ε (contradicts anti-resonance)")
    else:
        logging.info("? No significant difference (p > 0.05)")

    logging.info("="*80)

    return {
        'n_magic': int(n_magic),
        'n_normal': int(n_normal),
        'mean_magic': float(mean_magic),
        'mean_normal': float(mean_normal),
        'difference': float(diff),
        'ks_stat': float(ks_stat),
        'ks_p': float(ks_p),
        't_stat': float(t_stat),
        't_p': float(t_p),
    }

def main():
    parser = argparse.ArgumentParser(description='Tacoma Narrows Test: ε vs Half-Life Correlation')
    parser.add_argument('--scores', required=True, help='Scored nuclides parquet file')
    parser.add_argument('--out', required=True, help='Output directory')
    args = parser.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load data
    df_all = load_scored_nuclides(args.scores)

    # Filter to unstable with finite half-life
    df = filter_unstable_with_halflife(df_all)

    if len(df) < 10:
        logging.error("Too few samples for correlation analysis")
        return

    # Compute overall correlation
    results = {}
    results['overall'] = compute_correlation(df)

    # Analyze by decay mode
    results['by_decay_mode'] = analyze_by_decay_mode(df)

    # Analyze by mass region
    results['by_mass_region'] = analyze_by_mass_region(df)

    # Analyze by harmonic family
    results['by_family'] = analyze_by_family(df)

    # Magic number test
    results['magic_numbers'] = check_magic_numbers(df)

    # Plot
    plot_correlation(df, outdir)
    plot_by_decay_mode(df, outdir)
    plot_by_mass_region(df, outdir)

    # Save results
    results_path = outdir / 'tacoma_narrows_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logging.info(f"Saved results to {results_path}")

    logging.info("")
    logging.info("="*80)
    logging.info("TACOMA NARROWS TEST COMPLETE")
    logging.info("="*80)
    logging.info(f"Results saved to {outdir}")
    logging.info("")

if __name__ == '__main__':
    main()
