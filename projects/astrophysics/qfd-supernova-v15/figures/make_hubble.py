#!/usr/bin/env python3
"""
Figure 1: Hubble Diagram & Residuals (MNRAS single column)

Generates a two-panel stacked plot:
(a) Hubble diagram: μ_obs vs z with QFD model and ΛCDM reference
(b) Residuals: Δμ vs z with binned statistics

Canvas: 244 pt × 216 pt (single column)
Style: Monochrome-friendly, distinguishable in grayscale
"""

import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from mnras_style import (setup_mnras_style, create_figure_single_column,
                         save_figure_with_provenance, add_panel_label,
                         equal_count_bins, LINE_STYLES, compute_data_hash)

def load_data(stage3_dir):
    """Load Stage 3 Hubble diagram data."""
    hubble_file = Path(stage3_dir) / "hubble_data.csv"
    if not hubble_file.exists():
        hubble_file = Path(stage3_dir) / "stage3_results.csv"

    df = pd.read_csv(hubble_file)
    return df

def load_best_fit(stage2_dir):
    """Load best-fit parameters from Stage 2."""
    best_fit_file = Path(stage2_dir) / "best_fit.json"

    with open(best_fit_file) as f:
        best_fit = json.load(f)

    return best_fit

def alpha_pred(z, k_J, eta_prime, xi):
    """QFD alpha prediction."""
    phi1 = np.log1p(z)
    phi2 = z
    phi3 = z / (1.0 + z)
    return -(k_J * phi1 + eta_prime * phi2 + xi * phi3)

def mu_qfd(z, k_J, eta_prime, xi, alpha_obs=None):
    """
    QFD distance modulus.

    If alpha_obs is provided, use it; otherwise compute from model.
    """
    # Simplified: mu_th - K * alpha_pred
    # For plotting, we use a reference distance modulus
    mu_flat = 5 * np.log10(3000 * z) + 25  # Flat reference

    if alpha_obs is not None:
        K = 2.5 / np.log(10)
        return mu_flat - K * alpha_obs
    else:
        alpha_model = alpha_pred(z, k_J, eta_prime, xi)
        K = 2.5 / np.log(10)
        return mu_flat - K * alpha_model

def mu_lcdm(z, H0=70, Om=0.3):
    """ΛCDM distance modulus for reference."""
    from scipy.integrate import quad

    def E(zp):
        return np.sqrt(Om * (1 + zp)**3 + (1 - Om))

    # Luminosity distance
    c_km_s = 299792.458
    D_H = c_km_s / H0  # Hubble distance in Mpc

    z_arr = np.atleast_1d(z)
    D_L = []

    for zi in z_arr:
        if zi > 0:
            integral, _ = quad(lambda zp: 1/E(zp), 0, zi)
            D_L.append((1 + zi) * D_H * integral)
        else:
            D_L.append(0)

    D_L = np.array(D_L)
    mu = 5 * np.log10(D_L) + 25

    return mu if z.ndim > 0 else mu[0]

def main():
    setup_mnras_style()

    # Configuration
    stage3_dir = "../results/v15_production/stage3"
    stage2_dir = "../results/v15_production/stage2"
    output_file = "figure_hubble.pdf"

    # Load data
    print("Loading data...")
    df = load_data(stage3_dir)
    best_fit = load_best_fit(stage2_dir)

    # Extract parameters
    k_J = best_fit['k_J']
    eta_prime = best_fit['eta_prime']
    xi = best_fit['xi']

    print(f"Best-fit parameters:")
    print(f"  k_J = {k_J:.3f}")
    print(f"  η' = {eta_prime:.6f}")
    print(f"  ξ = {xi:.3f}")

    # Extract data
    z = df['z'].values
    mu_obs = df['mu_obs'].values

    # Compute models
    z_model = np.linspace(0.01, z.max() * 1.05, 200)
    mu_qfd_model = mu_qfd(z_model, k_J, eta_prime, xi)
    mu_lcdm_model = mu_lcdm(z_model)

    # Compute residuals
    mu_qfd_data = mu_qfd(z, k_J, eta_prime, xi)
    residuals = mu_obs - mu_qfd_data

    # Bin data for cleaner plot
    z_bin, mu_bin, mu_err = equal_count_bins(z, mu_obs, nbins=30)

    # Create figure
    fig = create_figure_single_column(aspect_ratio=1.0)  # Square-ish
    gs = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.05)

    # Panel (a): Hubble diagram
    ax1 = fig.add_subplot(gs[0])

    # Plot binned data
    ax1.errorbar(z_bin, mu_bin, yerr=mu_err,
                 **LINE_STYLES['data'], label='Data (binned)', zorder=3)

    # Plot models
    ax1.plot(z_model, mu_qfd_model, **LINE_STYLES['qfd'], zorder=2)
    ax1.plot(z_model, mu_lcdm_model, **LINE_STYLES['lcdm'], zorder=1)

    ax1.set_ylabel('Distance modulus μ (mag)', fontsize=7.5)
    ax1.set_xticklabels([])  # Hide x-tick labels (shared with panel b)
    ax1.legend(loc='lower right', fontsize=6.5, framealpha=0.9)
    ax1.grid(alpha=0.2)
    add_panel_label(ax1, '(a)', loc='top-left')

    # Panel (b): Residuals
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    # Plot individual residuals (all points with transparency)
    ax2.plot(z, residuals, **LINE_STYLES['data'], alpha=0.5, zorder=1)

    # Binned residuals
    z_bin_res, res_bin, res_err = equal_count_bins(z, residuals, nbins=25)
    ax2.errorbar(z_bin_res, res_bin, yerr=res_err,
                 fmt='s', markersize=3, markerfacecolor='gray',
                 markeredgecolor='black', markeredgewidth=0.4,
                 elinewidth=0.6, capsize=1.5, label='Binned', zorder=3)

    # Zero line
    ax2.axhline(0, color='black', linestyle='-', linewidth=0.6, zorder=1)

    # 1σ band (estimate from data)
    sigma = np.std(residuals)
    ax2.axhspan(-sigma, sigma, color='gray', alpha=0.15, zorder=0)

    ax2.set_xlabel('Redshift z', fontsize=7.5)
    ax2.set_ylabel('Residual Δμ (mag)', fontsize=7.5)
    ax2.legend(loc='best', fontsize=6.5)
    ax2.grid(alpha=0.2)
    add_panel_label(ax2, '(b)', loc='top-left')

    # Set x-limits
    ax1.set_xlim(0, z.max() * 1.05)
    ax2.set_xlim(0, z.max() * 1.05)

    # Adjust layout
    plt.tight_layout()

    # Save with provenance
    provenance = {
        'stage3_dir': str(stage3_dir),
        'stage2_dir': str(stage2_dir),
        'n_sne': len(df),
        'z_range': [float(z.min()), float(z.max())],
        'best_fit': {
            'k_J': float(k_J),
            'eta_prime': float(eta_prime),
            'xi': float(xi)
        },
        'rms_residual': float(np.std(residuals)),
        'nbins': 30,
    }

    save_figure_with_provenance(fig, output_file, provenance)

if __name__ == '__main__':
    main()
