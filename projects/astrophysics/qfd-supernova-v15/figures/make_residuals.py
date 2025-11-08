#!/usr/bin/env python3
"""
Figure 3: Residual Diagnostics (MNRAS single column)

Generates a 3-panel diagnostic figure:
(a) Residuals vs z with running median and 16-84% band
(b) Residuals vs nuisance parameter (if available)
(c) Q-Q plot or histogram of standardized residuals

Canvas: 244 pt × 216 pt (single column, vertical stack)
Style: Monochrome-friendly
"""

import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from mnras_style import (setup_mnras_style, create_figure_single_column,
                         save_figure_with_provenance, add_panel_label,
                         LINE_STYLES)

def load_data(stage3_dir):
    """Load Stage 3 residuals data."""
    hubble_file = Path(stage3_dir) / "hubble_data.csv"
    if not hubble_file.exists():
        hubble_file = Path(stage3_dir) / "stage3_results.csv"

    df = pd.read_csv(hubble_file)
    return df

def running_median_band(x, y, window_frac=0.2, percentiles=[16, 50, 84]):
    """
    Compute running median and percentile bands.

    Args:
        x: independent variable (e.g., redshift)
        y: dependent variable (e.g., residuals)
        window_frac: fraction of data points in sliding window
        percentiles: percentiles to compute [low, median, high]

    Returns:
        x_smooth, y_median, y_low, y_high
    """
    # Sort by x
    idx = np.argsort(x)
    x_sorted = x[idx]
    y_sorted = y[idx]

    n = len(x)
    window_size = max(int(n * window_frac), 10)

    x_smooth = []
    y_median = []
    y_low = []
    y_high = []

    for i in range(n):
        # Window centered on i
        start = max(0, i - window_size // 2)
        end = min(n, i + window_size // 2)

        window_y = y_sorted[start:end]

        if len(window_y) > 5:
            p_low, p_med, p_high = np.percentile(window_y, percentiles)
            x_smooth.append(x_sorted[i])
            y_median.append(p_med)
            y_low.append(p_low)
            y_high.append(p_high)

    return (np.array(x_smooth), np.array(y_median),
            np.array(y_low), np.array(y_high))

def main():
    setup_mnras_style()

    # Configuration
    stage3_dir = "../results/v15_production/stage3"
    output_file = "figure_residuals.pdf"

    # Load data
    print("Loading residual data...")
    df = load_data(stage3_dir)

    # Extract residuals
    z = df['z'].values
    if 'residual_qfd' in df.columns:
        residuals = df['residual_qfd'].values
    elif 'residual_alpha' in df.columns:
        K = 2.5 / np.log(10)
        residuals = -K * df['residual_alpha'].values
    else:
        raise ValueError("No residual column found in data")

    # Standardize residuals
    sigma = np.std(residuals)
    residuals_std = residuals / sigma

    print(f"Residual statistics:")
    print(f"  Mean: {np.mean(residuals):.4f} mag")
    print(f"  Std: {sigma:.4f} mag")
    print(f"  Median: {np.median(residuals):.4f} mag")

    # Compute running statistics
    z_smooth, res_med, res_low, res_high = running_median_band(z, residuals)

    # Anderson-Darling test
    ad_stat, ad_crit, ad_sig = stats.anderson(residuals_std)
    ks_stat, ks_p = stats.kstest(residuals_std, 'norm')

    print(f"Anderson-Darling statistic: {ad_stat:.3f}")
    print(f"Kolmogorov-Smirnov p-value: {ks_p:.4f}")

    # Create figure (3 panels, vertical stack)
    fig = create_figure_single_column(aspect_ratio=0.8)  # Taller
    gs = fig.add_gridspec(3, 1, hspace=0.3)

    # Panel (a): Residuals vs z
    ax1 = fig.add_subplot(gs[0])

    # Plot all residuals (small, transparent)
    ax1.plot(z, residuals, **LINE_STYLES['data'],
             alpha=0.4, markersize=2, zorder=1)

    # Running median and band
    ax1.plot(z_smooth, res_med, linestyle='-', linewidth=1.0,
             color='black', label='Running median', zorder=3)
    ax1.fill_between(z_smooth, res_low, res_high,
                     color='gray', alpha=0.3, label='16-84%', zorder=2)

    # Zero line
    ax1.axhline(0, color='black', linestyle='--', linewidth=0.6, zorder=2)

    ax1.set_xlabel('Redshift z', fontsize=7.0)
    ax1.set_ylabel('Residual Δμ (mag)', fontsize=7.0)
    ax1.legend(loc='best', fontsize=6.0)
    ax1.grid(alpha=0.2, linewidth=0.3)
    add_panel_label(ax1, '(a)', loc='top-left', fontsize=7)

    # Panel (b): Residuals vs nuisance (if available)
    ax2 = fig.add_subplot(gs[1])

    # Try to find nuisance parameter
    if 'x1' in df.columns:  # SALT2 stretch
        nuisance = df['x1'].values
        nuisance_label = 'SALT2 stretch x₁'
    elif 'c' in df.columns:  # SALT2 color
        nuisance = df['c'].values
        nuisance_label = 'SALT2 color c'
    elif 'alpha_obs' in df.columns:  # QFD alpha
        nuisance = df['alpha_obs'].values
        nuisance_label = 'α_obs'
    else:
        # Fallback: plot vs magnitude
        if 'mu_obs' in df.columns:
            nuisance = df['mu_obs'].values
            nuisance_label = 'μ_obs (mag)'
        else:
            nuisance = z  # Last resort
            nuisance_label = 'Redshift z'

    ax2.plot(nuisance, residuals, **LINE_STYLES['data'],
             alpha=0.4, markersize=2)

    # Add smoothed trend (LOWESS-style binning)
    nbins = 20
    nuisance_sorted = np.sort(nuisance)
    bin_edges = np.percentile(nuisance_sorted, np.linspace(0, 100, nbins+1))
    bin_centers = []
    bin_means = []

    for i in range(nbins):
        mask = (nuisance >= bin_edges[i]) & (nuisance < bin_edges[i+1])
        if mask.sum() > 5:
            bin_centers.append(np.median(nuisance[mask]))
            bin_means.append(np.median(residuals[mask]))

    if bin_centers:
        ax2.plot(bin_centers, bin_means, linestyle='-', linewidth=0.8,
                 color='black', marker='s', markersize=3,
                 label='Binned trend')

    ax2.axhline(0, color='black', linestyle='--', linewidth=0.6)

    ax2.set_xlabel(nuisance_label, fontsize=7.0)
    ax2.set_ylabel('Residual Δμ (mag)', fontsize=7.0)
    if bin_centers:
        ax2.legend(loc='best', fontsize=6.0)
    ax2.grid(alpha=0.2, linewidth=0.3)
    add_panel_label(ax2, '(b)', loc='top-left', fontsize=7)

    # Panel (c): Q-Q plot
    ax3 = fig.add_subplot(gs[2])

    # Compute theoretical quantiles
    (osm, osr), (slope, intercept, r) = stats.probplot(residuals_std,
                                                         dist='norm')

    ax3.plot(osm, osr, **LINE_STYLES['data'], markersize=2.5, alpha=0.7)

    # Reference line
    ax3.plot(osm, slope * osm + intercept, linestyle='-',
             linewidth=0.8, color='black', label='Normal reference')

    # Confidence envelope (rough approximation)
    n = len(residuals_std)
    se = 1.36 / np.sqrt(n)  # Approximate SE
    ax3.plot(osm, slope * osm + intercept + 1.96*se, linestyle='--',
             linewidth=0.6, color='gray', alpha=0.7)
    ax3.plot(osm, slope * osm + intercept - 1.96*se, linestyle='--',
             linewidth=0.6, color='gray', alpha=0.7)

    ax3.set_xlabel('Theoretical quantiles', fontsize=7.0)
    ax3.set_ylabel('Sample quantiles', fontsize=7.0)
    ax3.legend(loc='best', fontsize=6.0)
    ax3.grid(alpha=0.2, linewidth=0.3)

    # Add test statistics as text
    text = f"KS p = {ks_p:.3f}\nAD = {ad_stat:.2f}"
    ax3.text(0.05, 0.95, text, transform=ax3.transAxes,
             fontsize=6.0, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='black', linewidth=0.5, alpha=0.8))

    add_panel_label(ax3, '(c)', loc='top-right', fontsize=7)

    # Adjust layout
    plt.tight_layout()

    # Save with provenance
    provenance = {
        'stage3_dir': str(stage3_dir),
        'n_sne': len(df),
        'residual_mean': float(np.mean(residuals)),
        'residual_std': float(sigma),
        'residual_median': float(np.median(residuals)),
        'anderson_darling_stat': float(ad_stat),
        'ks_p_value': float(ks_p),
        'nuisance_parameter': nuisance_label,
    }

    save_figure_with_provenance(fig, output_file, provenance)

if __name__ == '__main__':
    main()
