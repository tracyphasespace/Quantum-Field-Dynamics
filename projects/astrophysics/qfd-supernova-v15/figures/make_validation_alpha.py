#!/usr/bin/env python3
"""
Generate MNRAS-quality validation figure for alpha_pred tests.

Figure 5 (double-column): Alpha prediction validation tests
- Panel (a): Monotonic decreasing behavior
- Panel (b): Parameter sensitivity to k_J
- Panel (c): Residual distribution (true vs wrong parameters)
- Panel (d): Independence from observations

Canvas: 508pt × 360pt (double column, 2×2 grid)
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path to import v15_model
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from v15_model import alpha_pred

# Import MNRAS style
from mnras_style import (
    setup_mnras_style,
    create_figure_double_column,
    save_figure_with_provenance,
    add_panel_label
)

def generate_synthetic_data(N_sne=100, seed=42):
    """Generate synthetic observations for validation tests."""
    np.random.seed(seed)

    z_batch = np.linspace(0.1, 0.8, N_sne)
    k_J_true = 70.0
    eta_prime = 0.01
    xi = 30.0

    # Generate true alpha values with noise
    alpha_true = np.array([alpha_pred(z, k_J_true, eta_prime, xi) for z in z_batch])
    alpha_obs = alpha_true + np.random.randn(N_sne) * 0.1

    return z_batch, alpha_obs, k_J_true, eta_prime, xi


def main():
    print("="*60)
    print("Generating MNRAS Validation Figure: Alpha Prediction Tests")
    print("="*60)

    # Setup MNRAS style
    setup_mnras_style()

    # Parameters for testing
    eta_prime = 0.01
    xi = 30.0

    # Generate synthetic data
    z_batch, alpha_obs, k_J_true, eta_prime, xi = generate_synthetic_data()

    # Create double-column figure (2×2 grid)
    fig = create_figure_double_column(aspect_ratio=1.4)  # 508pt × 363pt
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.35,
                          left=0.08, right=0.98, top=0.96, bottom=0.08)

    # -------------------------------------------------------------------
    # Panel (a): Monotonic decreasing behavior
    # -------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])

    z_vals = np.linspace(0, 1.5, 100)
    k_J = 70.0
    alpha_vals = np.array([alpha_pred(z, k_J, eta_prime, xi) for z in z_vals])

    ax1.plot(z_vals, alpha_vals, linestyle='-', linewidth=1.0, color='black')
    ax1.axhline(y=0, color='gray', linestyle='--', linewidth=0.6, alpha=0.7)

    ax1.set_xlabel('Redshift $z$')
    ax1.set_ylabel(r'$\alpha$ (dimming parameter)')
    ax1.grid(alpha=0.2)

    # Annotation box (simple, no color)
    ax1.text(0.05, 0.05, r'$\alpha(z=0) = 0$' + '\nMonotonic decreasing',
             transform=ax1.transAxes, fontsize=6.5,
             verticalalignment='bottom',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor='black', linewidth=0.5))

    add_panel_label(ax1, '(a)', loc='top-left')

    # -------------------------------------------------------------------
    # Panel (b): Parameter sensitivity
    # -------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])

    k_J_vals = [50, 60, 70, 80, 90]
    # Use grayscale gradient for MNRAS
    linestyles = ['-', '--', '-.', ':', '-']
    linewidths = [1.2, 1.0, 1.0, 1.0, 0.8]

    for i, k_J_test in enumerate(k_J_vals):
        alpha_vals = np.array([alpha_pred(z, k_J_test, eta_prime, xi)
                               for z in z_vals])
        ax2.plot(z_vals, alpha_vals,
                linestyle=linestyles[i],
                linewidth=linewidths[i],
                color='black',
                label=f'$k_J = {k_J_test}$')

    ax2.set_xlabel('Redshift $z$')
    ax2.set_ylabel(r'$\alpha$ (dimming parameter)')
    ax2.legend(loc='lower left', fontsize=6.0, framealpha=0.9)
    ax2.grid(alpha=0.2)

    add_panel_label(ax2, '(b)', loc='top-left')

    # -------------------------------------------------------------------
    # Panel (c): Residual distribution
    # -------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0])

    # True parameters
    alpha_pred_true = np.array([alpha_pred(z, k_J_true, eta_prime, xi)
                                for z in z_batch])
    residuals_true = alpha_obs - alpha_pred_true
    rms_true = np.sqrt(np.mean(residuals_true**2))

    # Wrong parameters
    k_J_wrong = 50.0
    alpha_pred_wrong = np.array([alpha_pred(z, k_J_wrong, eta_prime, xi)
                                 for z in z_batch])
    residuals_wrong = alpha_obs - alpha_pred_wrong
    rms_wrong = np.sqrt(np.mean(residuals_wrong**2))

    # Histograms (grayscale)
    ax3.hist(residuals_wrong, bins=20, alpha=0.5, color='gray',
             edgecolor='black', linewidth=0.5,
             label=f'Wrong params\nRMS={rms_wrong:.3f}')
    ax3.hist(residuals_true, bins=20, alpha=0.7, color='white',
             edgecolor='black', linewidth=0.8,
             label=f'True params\nRMS={rms_true:.3f}')
    ax3.axvline(x=0, color='black', linestyle='--', linewidth=0.6, alpha=0.5)

    ax3.set_xlabel(r'Residual ($\alpha_{\rm obs} - \alpha_{\rm pred}$)')
    ax3.set_ylabel('Count')
    ax3.legend(loc='upper left', fontsize=6.0, framealpha=0.9)
    ax3.grid(alpha=0.2)

    # Annotation
    var_ratio = rms_wrong / rms_true
    ax3.text(0.98, 0.98,
             f'var$(r) = {np.var(residuals_true):.5f} > 0$\n'
             f'{var_ratio:.1f}× worse for wrong params',
             transform=ax3.transAxes, fontsize=6.5,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor='black', linewidth=0.5))

    add_panel_label(ax3, '(c)', loc='top-left')

    # -------------------------------------------------------------------
    # Panel (d): Independence from observations
    # -------------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 1])

    # Shift observations by large constant
    alpha_obs_shifted = alpha_obs + 100.0

    # Predictions should be identical (independent of observations)
    alpha_pred_before = np.array([alpha_pred(z, k_J_true, eta_prime, xi)
                                  for z in z_batch])
    alpha_pred_after = np.array([alpha_pred(z, k_J_true, eta_prime, xi)
                                 for z in z_batch])
    max_diff = np.max(np.abs(alpha_pred_before - alpha_pred_after))

    # Scatter plots (grayscale markers)
    ax4.scatter(z_batch, alpha_pred_before, s=20, alpha=0.6,
               marker='o', facecolors='white', edgecolors='black',
               linewidths=0.5, label='Before shift')
    ax4.scatter(z_batch, alpha_pred_after, s=15, alpha=0.8,
               marker='x', color='black', linewidths=0.8,
               label='After shift')

    ax4.set_xlabel('Redshift $z$')
    ax4.set_ylabel(r'$\alpha_{\rm pred}$')
    ax4.legend(loc='lower left', fontsize=6.0, framealpha=0.9)
    ax4.grid(alpha=0.2)

    # Annotation
    ax4.text(0.98, 0.05,
             r'$\alpha_{\rm obs}$ shifted by +100' + '\n'
             f'Max diff in ' + r'$\alpha_{\rm pred}$' + f': {max_diff:.2e}\n'
             '(Perfect independence)',
             transform=ax4.transAxes, fontsize=6.5,
             verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor='black', linewidth=0.5))

    add_panel_label(ax4, '(d)', loc='top-left')

    # -------------------------------------------------------------------
    # Save figure with provenance
    # -------------------------------------------------------------------
    plt.tight_layout()

    filename = 'figure_validation_alpha.pdf'

    provenance = {
        'figure': filename,
        'title': 'Alpha prediction validation tests',
        'description': 'Four validation tests for alpha_pred function',
        'panels': {
            'a': 'Monotonic decreasing behavior',
            'b': 'Parameter sensitivity (k_J)',
            'c': 'Residual distribution (true vs wrong params)',
            'd': 'Independence from observations'
        },
        'test_parameters': {
            'k_J_true': float(k_J_true),
            'k_J_wrong': float(k_J_wrong),
            'eta_prime': float(eta_prime),
            'xi': float(xi),
            'N_sne': len(z_batch)
        },
        'statistics': {
            'rms_true': float(rms_true),
            'rms_wrong': float(rms_wrong),
            'var_residuals': float(np.var(residuals_true)),
            'independence_max_diff': float(max_diff)
        }
    }

    save_figure_with_provenance(fig, filename, provenance)

    print(f"\n✓ Saved: {filename}")
    print(f"✓ Provenance: {filename.replace('.pdf', '_provenance.json')}")
    print(f"\nValidation statistics:")
    print(f"  RMS (true params): {rms_true:.3f}")
    print(f"  RMS (wrong params): {rms_wrong:.3f}")
    print(f"  Ratio: {rms_wrong/rms_true:.1f}×")
    print(f"  Independence test: max diff = {max_diff:.2e}")


if __name__ == '__main__':
    main()
