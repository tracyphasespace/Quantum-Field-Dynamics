#!/usr/bin/env python3
"""
Figure 2: Basis Functions & Correlations (MNRAS single column)

Generates a 2×2 grid:
(a) Basis functions b_k(z) with distinct line styles
(b) Finite-difference derivatives b'_k(z)
(c) Correlation matrix ρ with annotations
(d) Identifiability metrics (condition number, max correlation)

Canvas: 244 pt × 208 pt (single column)
Style: Monochrome-friendly, no color dependency
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
                         LINE_STYLES)

def phi1(z):
    """Basis function 1: ln(1+z)"""
    return np.log1p(z)

def phi2(z):
    """Basis function 2: z"""
    return z

def phi3(z):
    """Basis function 3: z/(1+z)"""
    return z / (1.0 + z)

def load_data(stage3_dir):
    """Load redshift data."""
    hubble_file = Path(stage3_dir) / "hubble_data.csv"
    if not hubble_file.exists():
        hubble_file = Path(stage3_dir) / "stage3_results.csv"

    df = pd.read_csv(hubble_file)
    return df['z'].values

def load_posterior_corr(stage2_dir):
    """
    Load posterior correlation matrix if available.
    Otherwise compute from basis design matrix.
    """
    # Try to load from samples
    samples_file = Path(stage2_dir) / "samples.json"

    if samples_file.exists():
        with open(samples_file) as f:
            samples = json.load(f)

        # Check if we have correlation matrix
        if 'correlation_matrix' in samples:
            return np.array(samples['correlation_matrix'])

    # If not available, return None (will compute from basis)
    return None

def main():
    setup_mnras_style()

    # Configuration
    stage3_dir = "../results/v15_production/stage3"
    stage2_dir = "../results/v15_production/stage2"
    output_file = "figure_basis_inference.pdf"

    # Load data
    print("Loading redshift data...")
    z = load_data(stage3_dir)

    # Sort for plotting
    z_sorted = np.sort(z)

    # Compute basis functions
    Phi1 = phi1(z_sorted)
    Phi2 = phi2(z_sorted)
    Phi3 = phi3(z_sorted)

    # Stack into design matrix
    Phi = np.column_stack([Phi1, Phi2, Phi3])

    # Compute correlation matrix
    corr = np.corrcoef(Phi, rowvar=False)

    # Try to load posterior correlation (preferred)
    posterior_corr = load_posterior_corr(stage2_dir)
    if posterior_corr is not None:
        print("Using posterior correlation matrix from MCMC")
        corr = posterior_corr
    else:
        print("Using basis design matrix correlation")

    # Compute condition number
    XT_X = Phi.T @ Phi
    cond = np.linalg.cond(XT_X)

    # Max absolute off-diagonal correlation
    max_corr = np.max(np.abs(corr[np.triu_indices_from(corr, k=1)]))

    print(f"Condition number κ: {cond:.2e}")
    print(f"Max |ρ|: {max_corr:.4f}")

    # Create figure (2×2 grid) - 25% taller for subtitles
    # aspect_ratio = width/height, so SMALLER = TALLER
    fig = create_figure_single_column(aspect_ratio=0.88)
    gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.55)

    # Panel (a): Basis functions
    ax1 = fig.add_subplot(gs[0, 0])

    ax1.plot(z_sorted, Phi1, linestyle='-', linewidth=0.8, color='#1f77b4',
             label='φ₁ = ln(1+z)')
    ax1.plot(z_sorted, Phi2, linestyle='--', linewidth=0.8, color='#ff7f0e',
             label='φ₂ = z')
    ax1.plot(z_sorted, Phi3, linestyle=':', linewidth=1.0, color='#2ca02c',
             label='φ₃ = z/(1+z)')

    ax1.set_xlabel('Redshift z', fontsize=7.0)
    ax1.set_ylabel('Basis value', fontsize=7.0)
    ax1.legend(loc='upper left', fontsize=6.0, framealpha=0.9)
    ax1.grid(alpha=0.2, linewidth=0.3)
    # Subtitle below the panel
    ax1.text(0.5, -0.32, '(a) Basis Functions', transform=ax1.transAxes,
             fontsize=6.5, verticalalignment='top', horizontalalignment='center')

    # Panel (b): Derivatives
    ax2 = fig.add_subplot(gs[0, 1])

    # Finite differences
    dz = np.diff(z_sorted)
    dPhi1 = np.diff(Phi1) / dz
    dPhi2 = np.diff(Phi2) / dz
    dPhi3 = np.diff(Phi3) / dz
    z_mid = (z_sorted[:-1] + z_sorted[1:]) / 2

    ax2.plot(z_mid, dPhi1, linestyle='-', linewidth=0.8, color='#1f77b4',
             label="dφ₁/dz")
    ax2.plot(z_mid, dPhi2, linestyle='--', linewidth=0.8, color='#ff7f0e',
             label="dφ₂/dz")
    ax2.plot(z_mid, dPhi3, linestyle=':', linewidth=1.0, color='#2ca02c',
             label="dφ₃/dz")

    ax2.set_xlabel('Redshift z', fontsize=7.0)
    ax2.set_ylabel('Finite difference dφ/dz', fontsize=7.0)
    ax2.legend(loc='upper right', fontsize=6.0, framealpha=0.9)
    ax2.grid(alpha=0.2, linewidth=0.3)
    # Subtitle below the panel
    ax2.text(0.5, -0.32, '(b) Derivatives', transform=ax2.transAxes,
             fontsize=6.5, verticalalignment='top', horizontalalignment='center')

    # Panel (c): Correlation matrix
    ax3 = fig.add_subplot(gs[1, 0])

    # Plot as grayscale heatmap
    im = ax3.imshow(corr, vmin=-1, vmax=1, cmap='RdBu_r',
                    interpolation='none', aspect='auto')

    # Annotations
    for i in range(3):
        for j in range(3):
            text = ax3.text(j, i, f'{corr[i, j]:.2f}',
                           ha="center", va="center",
                           color="black" if abs(corr[i, j]) < 0.5 else "white",
                           fontsize=6.5)

    ax3.set_xticks([0, 1, 2])
    ax3.set_yticks([0, 1, 2])
    ax3.set_xticklabels(['φ₁', 'φ₂', 'φ₃'], fontsize=7.0)
    ax3.set_yticklabels(['φ₁', 'φ₂', 'φ₃'], fontsize=7.0)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    cbar.set_label('Correlation ρ', fontsize=7.0)
    cbar.ax.tick_params(labelsize=6.0)

    # Subtitle below the panel
    ax3.text(0.5, -0.20, '(c) Correlation Matrix', transform=ax3.transAxes,
             fontsize=6.5, verticalalignment='top', horizontalalignment='center')

    # Panel (d): Identifiability box
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')  # No axes for text box

    # Create text box with metrics
    text_lines = [
        "Identifiability Metrics",
        "",
        f"Condition number:",
        f"  κ = {cond:.2e}",
        "",
        f"Max|ρ|off-diagonal:",
        f"  {max_corr:.4f}",
        "",
        "Status:",
    ]

    if cond > 1e4:
        text_lines.append("  ⚠ High collinearity")
        text_lines.append("  Consider orthogonalization")
    elif cond > 100:
        text_lines.append("  ⚠ Moderate conditioning")
    else:
        text_lines.append("  ✓ Well-conditioned")

    text = "\n".join(text_lines)

    # Plain text without gray box, centered
    ax4.text(0.5, 0.55, text, transform=ax4.transAxes,
             fontsize=6.5, verticalalignment='center', horizontalalignment='center',
             family='monospace')

    # Subtitle below the panel (no duplicate label needed)
    ax4.text(0.5, -0.20, '(d) Identifiability Metrics', transform=ax4.transAxes,
             fontsize=6.5, verticalalignment='top', horizontalalignment='center')

    # Adjust layout
    plt.tight_layout()

    # Add separator lines above (a) and (b) subtitles
    # Get the position of the subtitles in figure coordinates
    # Line at y position just above the subtitle text
    fig.add_artist(plt.Line2D([0.12, 0.48], [0.455, 0.455], transform=fig.transFigure,
                               color='black', linewidth=0.5))
    fig.add_artist(plt.Line2D([0.52, 0.88], [0.455, 0.455], transform=fig.transFigure,
                               color='black', linewidth=0.5))

    # Save with provenance
    provenance = {
        'stage3_dir': str(stage3_dir),
        'stage2_dir': str(stage2_dir),
        'n_redshifts': len(z),
        'z_range': [float(z.min()), float(z.max())],
        'condition_number': float(cond),
        'max_correlation': float(max_corr),
        'correlation_matrix': corr.tolist(),
    }

    save_figure_with_provenance(fig, output_file, provenance)

if __name__ == '__main__':
    main()
