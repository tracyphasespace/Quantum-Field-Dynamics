#!/usr/bin/env python3
"""
Figure 4: Parameter Posteriors / Corner Plot (MNRAS double column)

Generates corner plot of key QFD parameters:
- k_J (cosmological drag)
- η' (plasma veil evolution)
- ξ (FDR/saturation)
- Optional: σ_α (intrinsic scatter)
- Optional: ν (Student-t degrees of freedom)

Canvas: 508 pt × 320-360 pt (double column)
Style: Monochrome contours, no color fill
"""

import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from scipy.ndimage import gaussian_filter

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from mnras_style import (setup_mnras_style, create_figure_double_column,
                         save_figure_with_provenance, add_panel_label)

def load_samples(stage2_dir):
    """Load MCMC samples from .npy files (most reliable format)."""
    stage2_path = Path(stage2_dir)

    # Load from individual .npy files (preferred - most reliable)
    k_J_file = stage2_path / "k_J_samples.npy"
    if k_J_file.exists():
        samples = {
            'k_J': np.load(stage2_path / "k_J_samples.npy"),
            'eta_prime': np.load(stage2_path / "eta_prime_samples.npy"),
            'xi': np.load(stage2_path / "xi_samples.npy"),
        }

        # Optional parameters
        sigma_file = stage2_path / "sigma_alpha_samples.npy"
        if sigma_file.exists():
            samples['sigma_alpha'] = np.load(sigma_file)

        nu_file = stage2_path / "nu_samples.npy"
        if nu_file.exists():
            samples['nu'] = np.load(nu_file)

        return samples

    # Fallback: Try JSON format
    samples_file = stage2_path / "samples.json"
    if samples_file.exists():
        with open(samples_file) as f:
            data = json.load(f)

        # Extract arrays
        if 'k_J' in data and isinstance(data['k_J'], list):
            samples = {
                'k_J': np.array(data['k_J']),
                'eta_prime': np.array(data['eta_prime']),
                'xi': np.array(data['xi']),
            }

            if 'sigma_alpha' in data:
                samples['sigma_alpha'] = np.array(data['sigma_alpha'])
            if 'nu' in data:
                samples['nu'] = np.array(data['nu'])

            return samples

    # Try NumPy format
    npz_file = stage2_path / "samples.npz"
    if npz_file.exists():
        data = np.load(npz_file)
        return {key: data[key] for key in data.files}

    raise FileNotFoundError(f"No samples found in {stage2_dir}")

def compute_2d_contours(x, y, levels=[0.68, 0.95]):
    """
    Compute 2D density contours for corner plot.

    Args:
        x, y: sample arrays
        levels: confidence levels (e.g., [0.68, 0.95])

    Returns:
        X, Y, Z, levels_sorted
    """
    # Create 2D histogram
    H, xedges, yedges = np.histogram2d(x, y, bins=30)

    # Smooth
    H = gaussian_filter(H, sigma=1.0)

    # Normalize
    H = H / H.sum()

    # Sort density values
    H_sorted = np.sort(H.flatten())[::-1]
    H_cumsum = np.cumsum(H_sorted)

    # Find contour levels
    contour_levels = []
    for level in levels:
        idx = np.searchsorted(H_cumsum, level)
        if idx < len(H_sorted):
            contour_levels.append(H_sorted[idx])

    # Create meshgrid for contour
    X, Y = np.meshgrid(xedges[:-1], yedges[:-1])

    return X, Y, H.T, contour_levels

def plot_1d_hist(ax, samples, label, color='#1f77b4'):
    """Plot 1D histogram with median and percentiles."""
    ax.hist(samples, bins=30, density=True, histtype='step',
            color=color, linewidth=1.0)

    # Add median and percentiles
    median = np.median(samples)
    p16, p84 = np.percentile(samples, [16, 84])

    ymax = ax.get_ylim()[1]
    ax.axvline(median, color=color, linestyle='-', linewidth=1.0, ymax=0.8)
    ax.axvline(p16, color=color, linestyle='--', linewidth=0.6, alpha=0.7, ymax=0.6)
    ax.axvline(p84, color=color, linestyle='--', linewidth=0.6, alpha=0.7, ymax=0.6)

    # Add text
    text = f"{median:.3f}$^{{+{p84-median:.3f}}}_{{-{median-p16:.3f}}}$"
    ax.text(0.95, 0.95, text, transform=ax.transAxes,
            fontsize=6.0, ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                     edgecolor='none', alpha=0.7))

    ax.set_yticks([])
    ax.set_ylabel('')

def plot_2d_contour(ax, x, y, xlabel, ylabel):
    """Plot 2D contour with color levels."""
    X, Y, Z, levels = compute_2d_contours(x, y, levels=[0.68, 0.95])

    # Plot contours (solid for 68%, dashed for 95%)
    # Use colorful contours: blue for 68%, orange for 95%
    cs1 = ax.contour(X, Y, Z, levels=[levels[0]], colors='#1f77b4',
                     linewidths=1.0, linestyles='-')
    if len(levels) > 1:
        cs2 = ax.contour(X, Y, Z, levels=[levels[1]], colors='#ff7f0e',
                         linewidths=0.8, linestyles='--')

    ax.set_xlabel(xlabel, fontsize=7.0)
    ax.set_ylabel(ylabel, fontsize=7.0)

def main():
    setup_mnras_style()

    # Configuration
    stage2_dir = "../results/v15_production/stage2"
    output_file = "figure_corner.pdf"

    # Load samples
    print("Loading MCMC samples...")
    samples = load_samples(stage2_dir)

    # Select parameters to plot (max 5 for readability)
    param_names = ['k_J', 'eta_prime', 'xi']
    param_labels = ['$k_J$', "$\\eta'$", '$\\xi$']

    # Add optional parameters if available
    if 'sigma_alpha' in samples:
        param_names.append('sigma_alpha')
        param_labels.append('$\\sigma_\\alpha$')

    if 'nu' in samples and len(param_names) < 5:
        param_names.append('nu')
        param_labels.append('$\\nu$')

    n_params = len(param_names)

    print(f"Plotting {n_params} parameters: {param_names}")

    # Create figure (aspect_ratio ~1.0 for square corner plot)
    fig = create_figure_double_column(aspect_ratio=1.0)

    # Create grid
    gs = fig.add_gridspec(n_params, n_params, hspace=0.05, wspace=0.05)

    # Plot corner
    for i in range(n_params):
        for j in range(n_params):
            if i == j:
                # Diagonal: 1D histogram
                ax = fig.add_subplot(gs[i, j])
                plot_1d_hist(ax, samples[param_names[i]],
                            param_labels[i])

                if i < n_params - 1:
                    ax.set_xticklabels([])
                else:
                    ax.set_xlabel(param_labels[i], fontsize=7.0)

            elif i > j:
                # Lower triangle: 2D contour
                ax = fig.add_subplot(gs[i, j])
                plot_2d_contour(ax, samples[param_names[j]],
                               samples[param_names[i]],
                               param_labels[j] if i == n_params-1 else '',
                               param_labels[i] if j == 0 else '')

                if i < n_params - 1:
                    ax.set_xticklabels([])
                if j > 0:
                    ax.set_yticklabels([])

            else:
                # Upper triangle: hide
                ax = fig.add_subplot(gs[i, j])
                ax.axis('off')

    # Add title
    fig.suptitle('Parameter Posteriors', fontsize=8, fontweight='bold', y=0.98)

    # Adjust layout
    plt.tight_layout()

    # Compute summary statistics
    summary = {}
    for pname in param_names:
        data = samples[pname]
        summary[pname] = {
            'median': float(np.median(data)),
            'mean': float(np.mean(data)),
            'std': float(np.std(data)),
            'p16': float(np.percentile(data, 16)),
            'p84': float(np.percentile(data, 84)),
        }

    # Save with provenance
    provenance = {
        'stage2_dir': str(stage2_dir),
        'n_samples': int(len(samples[param_names[0]])),
        'parameters': param_names,
        'summary': summary,
    }

    save_figure_with_provenance(fig, output_file, provenance)

    # Print summary
    print("\nParameter Summary (median ± std):")
    for pname in param_names:
        s = summary[pname]
        print(f"  {pname:12s}: {s['median']:.4f} ± {s['std']:.4f}")

if __name__ == '__main__':
    main()
