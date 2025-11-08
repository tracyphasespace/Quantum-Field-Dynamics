#!/usr/bin/env python3
"""
Generate MNRAS comparison figure: Time Dilation vs Thermal Distribution

Figure 6 (single-column): Morphological similarity between cosmological
time dilation and thermal distributions.

Panel (a): Supernova light curves at different redshifts (z=1 to z=10)
          showing cosmological time dilation effects
Panel (b): Blackbody radiation curves for temperatures T(z=1) to T(z=10)
          showing similar morphology

Canvas: 244pt × 320pt (single column, 2 panels stacked)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from mnras_style import (
    setup_mnras_style,
    create_figure_single_column,
    save_figure_with_provenance,
    add_panel_label
)


def supernova_light_curve(t, t_peak=20, width=10, amplitude=1.0):
    """
    Model supernova light curve (simplified Gaussian-like rise + exponential decay).

    Args:
        t: Time array (days)
        t_peak: Time of maximum light (days)
        width: Width parameter
        amplitude: Peak amplitude
    """
    # Rise phase (Gaussian)
    rise = np.exp(-0.5 * ((t - t_peak) / (width * 0.5))**2)

    # Decay phase (exponential)
    decay = np.exp(-(t - t_peak) / (width * 1.5))

    # Combine: rise until peak, then decay
    lc = np.where(t < t_peak, rise, rise[np.argmax(rise)] * decay / decay[0])

    return amplitude * lc


def planck_distribution(wavelength, T):
    """
    Planck blackbody distribution (normalized).

    Args:
        wavelength: Wavelength array (arbitrary units)
        T: Temperature (arbitrary units, higher T = peak at shorter wavelength)

    Returns:
        Normalized spectral radiance
    """
    # Simplified Wien approximation (good for peak region)
    # B(λ,T) ∝ λ^-5 * exp(-hc/λkT)
    # Using arbitrary units: B ∝ λ^-5 * exp(-const/λT)

    const = 50.0  # Arbitrary constant for scaling

    with np.errstate(over='ignore', invalid='ignore'):
        B = (wavelength**-5) * np.exp(-const / (wavelength * T))

    # Normalize
    B = np.nan_to_num(B)
    if np.max(B) > 0:
        B = B / np.max(B)

    return B


def main():
    print("="*60)
    print("Generating MNRAS Comparison Figure:")
    print("Time Dilation vs Thermal Distribution")
    print("="*60)

    # Setup MNRAS style
    setup_mnras_style()

    # Create single-column figure (2 panels stacked)
    fig = create_figure_single_column(aspect_ratio=0.76)  # 244pt × 320pt
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.35,  # More spacing
                          left=0.14, right=0.96, top=0.94, bottom=0.08)  # Adjust top

    # Color scheme: distinct colors + line styles for monochrome-friendliness
    n_curves = 10

    # Distinct colors (colorblind-friendly palette)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    # Line styles for distinction in grayscale
    linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--']
    linewidths = [1.2, 1.0, 1.0, 1.2, 1.0, 1.0, 1.0, 1.2, 1.0, 1.0]

    # -------------------------------------------------------------------
    # Panel (a): Supernova Light Curves with Time Dilation
    # -------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0])

    # Time array (rest frame) - increased points for smoother curves
    t_rest = np.linspace(0, 200, 2000)  # Match the scale to show full curves

    # Generate light curves for z=1 to z=10
    redshifts = np.arange(1, 11)

    for i, z in enumerate(redshifts):
        # Time dilation: observed time = (1+z) * rest-frame time
        time_dilation_factor = 1 + z

        # Observed time axis
        t_obs = t_rest * time_dilation_factor

        # Light curve (normalized flux vs observed time)
        flux = supernova_light_curve(t_rest, t_peak=20, width=10, amplitude=1.0)

        # Plot with color + line styles (NO markers for clean curves)
        ax1.plot(t_obs, flux,
                linestyle=linestyles[i],
                linewidth=linewidths[i],
                color=colors[i],
                label=f'$z={z}$',
                alpha=0.9)

    # Set matching scale: max observed time for z=10 is ~2200 days
    max_time = 200 * (1 + 10)  # 2200 days

    ax1.set_xlabel('Time (days)', fontsize=9)
    ax1.set_ylabel('Normalized Flux', fontsize=9)
    ax1.set_xlim(0, max_time)  # 0-2200 to show all curves fully
    ax1.set_ylim(0, 1.0)
    ax1.tick_params(axis='both', which='major', labelsize=8)
    ax1.grid(alpha=0.2)

    # Legend (compact, 2 columns)
    ax1.legend(loc='upper right', fontsize=6.5, ncol=2,
              framealpha=0.9, columnspacing=1.0, handlelength=1.5)

    add_panel_label(ax1, '(a)', loc='top-left')

    # Panel title (moved up to avoid overlap)
    ax1.text(0.5, 1.05, 'Supernova Light Curves ($z=1$ to $10$)',
            transform=ax1.transAxes, ha='center', va='bottom',
            fontsize=9, fontweight='bold')

    # -------------------------------------------------------------------
    # Panel (b): Planck/Wien Thermal Distribution
    # -------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[1])

    # Wavelength array (arbitrary units) - match panel (a) scale
    max_time = 200 * (1 + 10)  # 2200 days (same as panel a)
    wavelength = np.linspace(0.1, max_time, 2000)  # 0-2200 to match panel (a)

    # Temperature range (arbitrary units, decreasing T shifts peak right)
    # Use inverse mapping so visual similarity is maximized
    temperatures = np.linspace(1.0, 0.1, n_curves)

    for i, T in enumerate(temperatures):
        # Planck distribution
        B = planck_distribution(wavelength, T)

        # Plot with same colors/styles as panel (a) for visual consistency
        # Label as T(z=1), T(z=2), etc. to match redshift parameterization
        ax2.plot(wavelength, B,
                linestyle=linestyles[i],
                linewidth=linewidths[i],
                color=colors[i],
                label=f'$T(z={i+1})$',
                alpha=0.9)

    ax2.set_xlabel(r'Wavelength (scaled to days)', fontsize=9)
    ax2.set_ylabel('Normalized Radiance', fontsize=9)
    ax2.set_xlim(0, max_time)   # Match panel (a): 0-2200 for visual similarity
    ax2.set_ylim(0, 1.0)
    ax2.tick_params(axis='both', which='major', labelsize=8)
    ax2.grid(alpha=0.2)

    # Legend
    ax2.legend(loc='upper right', fontsize=6.5, ncol=2,
              framealpha=0.9, columnspacing=1.0, handlelength=1.5)

    add_panel_label(ax2, '(b)', loc='top-left')

    # Panel title (moved up to avoid overlap)
    ax2.text(0.5, 1.05, 'Blackbody Radiation Curves',
            transform=ax2.transAxes, ha='center', va='bottom',
            fontsize=9, fontweight='bold')

    # -------------------------------------------------------------------
    # Save figure with provenance
    # -------------------------------------------------------------------
    plt.tight_layout()

    filename = 'figure_time_dilation_thermal_comparison.pdf'

    provenance = {
        'figure': filename,
        'title': 'Morphological comparison: Time dilation vs blackbody radiation',
        'description': 'Demonstrates similar curve shapes between cosmological time dilation and thermal blackbody radiation',
        'panels': {
            'a': 'Supernova light curves at redshifts z=1 to z=10 showing time dilation',
            'b': 'Blackbody radiation curves for temperatures T(z=1) to T(z=10)'
        },
        'key_insight': 'Both phenomena produce similar morphological curve families, suggesting thermal interpretation of cosmological effects',
        'parameters': {
            'redshift_range': [1, 10],
            'n_curves': n_curves,
            'time_peak_rest': 20,
            'time_width': 10
        }
    }

    save_figure_with_provenance(fig, filename, provenance)

    print(f"\n✓ Saved: {filename}")
    print(f"✓ Provenance: {filename.replace('.pdf', '_provenance.json')}")
    print(f"\nFigure comparison:")
    print(f"  Panel (a): {n_curves} supernova light curves (z=1 to {n_curves})")
    print(f"  Panel (b): {n_curves} thermal distributions")
    print(f"  Key: Morphological similarity highlights potential deep connection")


if __name__ == '__main__':
    main()
