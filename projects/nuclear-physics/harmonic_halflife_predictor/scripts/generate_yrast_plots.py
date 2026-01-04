#!/usr/bin/env python3
"""
Generate Yrast Spectroscopy Plots for Harmonic Nuclear Model

Creates publication-quality figures showing nuclear spectroscopy
using harmonic mode N as analog to angular momentum J.

Generates:
  - yrast_spectral_analysis.png: 4-panel comprehensive analysis
  - yrast_comparison.png: Traditional vs. harmonic comparison

Usage:
    python generate_yrast_plots.py

Output:
    Saves figures to ../figures/ directory
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Add scripts directory to path for imports
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

from nucleus_classifier import classify_nucleus

# Element symbols for common elements
ELEMENT_SYMBOLS = {
    1: 'H', 2: 'He', 6: 'C', 8: 'O', 26: 'Fe', 28: 'Ni',
    50: 'Sn', 82: 'Pb', 92: 'U'
}


def load_data():
    """Load AME2020 nuclear data."""
    data_path = script_dir.parent / 'data' / 'ame2020_system_energies.csv'

    if not data_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {data_path}\n"
            "Run scripts/download_ame2020.py first."
        )

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} nuclei from AME2020")

    # Classify all nuclei
    classifications = []
    for _, row in df.iterrows():
        A = int(row['A'])
        Z = int(row['Z'])
        N_mode, family = classify_nucleus(A, Z)
        classifications.append({
            'A': A,
            'Z': Z,
            'N': N_mode,
            'family': family,
            'BE_per_A': row['BE_per_A_MeV']
        })

    df_classified = pd.DataFrame(classifications)
    df_classified = df_classified[df_classified['N'].notna()]  # Remove unclassified

    print(f"Classified {len(df_classified)} nuclei into families A, B, C")
    return df_classified


def plot_yrast_spectral_analysis(df):
    """
    Create 4-panel yrast spectral analysis figure.

    Panels:
      A: Yrast diagram for Sn isotopes (Z=50)
      B: Energy level diagram for A=100 isobars
      C: Band structure for multiple mass numbers
      D: Mode occupation histogram
    """
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    # Color scheme for families
    colors = {'A': '#2E86AB', 'B': '#A23B72', 'C': '#F18F01'}

    # Panel A: Yrast diagram for Sn (Z=50)
    ax1 = fig.add_subplot(gs[0, 0])
    Z_target = 50
    sn_isotopes = df[df['Z'] == Z_target].sort_values('N')

    if len(sn_isotopes) > 0:
        for family in ['A', 'B', 'C']:
            family_data = sn_isotopes[sn_isotopes['family'] == family]
            if len(family_data) > 0:
                ax1.plot(family_data['N'], family_data['BE_per_A'],
                        'o-', color=colors[family], label=f'Family {family}',
                        markersize=8, linewidth=2, alpha=0.8)

        ax1.set_xlabel('N (Harmonic Mode)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('BE/A (MeV/nucleon)', fontsize=12, fontweight='bold')
        ax1.set_title(f'A) Yrast Diagram: Sn Isotopes (Z={Z_target})',
                     fontsize=13, fontweight='bold', loc='left')
        ax1.legend(frameon=True, fancybox=True, shadow=True)
        ax1.grid(True, alpha=0.3, linestyle='--')

    # Panel B: Energy level diagram for A=100
    ax2 = fig.add_subplot(gs[0, 1])
    A_target = 100
    isobars = df[df['A'] == A_target]

    if len(isobars) > 0:
        for _, iso in isobars.iterrows():
            family = iso['family']
            N = iso['N']
            E = iso['BE_per_A']

            # Draw horizontal line for energy level
            ax2.hlines(E, N-0.3, N+0.3, colors=colors[family], linewidth=4, alpha=0.7)

            # Add element symbol if known
            Z = int(iso['Z'])
            if Z in ELEMENT_SYMBOLS:
                ax2.text(N+0.35, E, ELEMENT_SYMBOLS[Z],
                        fontsize=9, va='center')

        ax2.set_xlabel('N (Harmonic Mode)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('BE/A (MeV/nucleon)', fontsize=12, fontweight='bold')
        ax2.set_title(f'B) Energy Levels: A={A_target} Isobars',
                     fontsize=13, fontweight='bold', loc='left')
        ax2.grid(True, alpha=0.3, linestyle='--')

    # Panel C: Band structure for multiple mass numbers
    ax3 = fig.add_subplot(gs[1, 0])

    # Select representative mass numbers
    mass_numbers = [80, 120, 160, 200, 240]

    for A in mass_numbers:
        subset = df[(df['A'] == A) & (df['family'] == 'A')]  # Family A only
        subset = subset.sort_values('N')

        if len(subset) > 0:
            ax3.plot(subset['N'], subset['BE_per_A'],
                    'o-', label=f'A={A}', markersize=6, linewidth=2, alpha=0.7)

    ax3.set_xlabel('N (Harmonic Mode)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('BE/A (MeV/nucleon)', fontsize=12, fontweight='bold')
    ax3.set_title('C) Band Structure (Family A)',
                 fontsize=13, fontweight='bold', loc='left')
    ax3.legend(frameon=True, fancybox=True, shadow=True, ncol=2)
    ax3.grid(True, alpha=0.3, linestyle='--')

    # Panel D: Mode occupation histogram
    ax4 = fig.add_subplot(gs[1, 1])

    for family in ['A', 'B', 'C']:
        family_data = df[df['family'] == family]
        if len(family_data) > 0:
            counts = family_data['N'].value_counts().sort_index()
            ax4.bar(counts.index + (0.25 if family == 'B' else (-0.25 if family == 'A' else 0)),
                   counts.values, width=0.25, color=colors[family],
                   label=f'Family {family}', alpha=0.7)

    ax4.set_xlabel('N (Harmonic Mode)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Number of Nuclei', fontsize=12, fontweight='bold')
    ax4.set_title('D) Mode Occupation Statistics',
                 fontsize=13, fontweight='bold', loc='left')
    ax4.legend(frameon=True, fancybox=True, shadow=True)
    ax4.grid(True, alpha=0.3, linestyle='--', axis='y')

    plt.suptitle('Harmonic Nuclear Spectroscopy: Yrast and Band Structure',
                fontsize=16, fontweight='bold', y=0.995)

    return fig


def plot_yrast_comparison(df):
    """
    Create 2-panel comparison: Traditional vs. Harmonic yrast concept.

    Panels:
      Left: Traditional yrast concept (schematic)
      Right: Harmonic yrast lines (actual data)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: Traditional yrast concept (schematic)
    J_vals = np.arange(0, 21, 2)
    E_yrast = 0.02 * J_vals * (J_vals + 1)  # Rotational formula

    # Add some excited bands
    E_beta = E_yrast + 1.5
    E_gamma = E_yrast + 2.8

    ax1.plot(J_vals, E_yrast, 'o-', color='#2E86AB', linewidth=3,
            markersize=8, label='Yrast (ground) band')
    ax1.plot(J_vals, E_beta, 's--', color='#A23B72', linewidth=2,
            markersize=6, alpha=0.7, label='β-band (excited)')
    ax1.plot(J_vals, E_gamma, '^--', color='#F18F01', linewidth=2,
            markersize=6, alpha=0.7, label='γ-band (excited)')

    ax1.set_xlabel('J (Angular Momentum)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Energy (MeV)', fontsize=12, fontweight='bold')
    ax1.set_title('Traditional Yrast Concept\n(E vs. J for rotation)',
                 fontsize=13, fontweight='bold')
    ax1.legend(frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.text(0.02, 0.98, 'Schematic', transform=ax1.transAxes,
            fontsize=10, va='top', style='italic',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Panel 2: Harmonic yrast lines (actual data)
    colors = {'A': '#2E86AB', 'B': '#A23B72', 'C': '#F18F01'}

    # Plot yrast lines for a few selected elements
    selected_Z = [28, 50, 82]  # Ni, Sn, Pb (magic numbers)

    for Z in selected_Z:
        element_data = df[df['Z'] == Z].sort_values('N')

        if len(element_data) > 0:
            symbol = ELEMENT_SYMBOLS.get(Z, f'Z={Z}')

            # Plot by family
            for family in ['A', 'B', 'C']:
                family_data = element_data[element_data['family'] == family]
                if len(family_data) > 0:
                    label = f'{symbol} (Family {family})' if family == 'A' else None
                    ax2.plot(family_data['N'], family_data['BE_per_A'],
                            'o-', color=colors[family], linewidth=2,
                            markersize=6, alpha=0.7, label=label)

    ax2.set_xlabel('N (Harmonic Mode)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('BE/A (MeV/nucleon)', fontsize=12, fontweight='bold')
    ax2.set_title('Harmonic Yrast Lines\n(Experimental data)',
                 fontsize=13, fontweight='bold')
    ax2.legend(frameon=True, fancybox=True, shadow=True)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.text(0.02, 0.98, 'AME2020 Data', transform=ax2.transAxes,
            fontsize=10, va='top', style='italic',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.suptitle('Comparison: Traditional vs. Harmonic Spectroscopy',
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()

    return fig


def main():
    """Generate all yrast plots."""
    print("=" * 60)
    print("Harmonic Nuclear Spectroscopy - Yrast Plot Generator")
    print("=" * 60)

    # Load and classify data
    df = load_data()

    # Create output directory
    output_dir = script_dir.parent / 'figures'
    output_dir.mkdir(exist_ok=True)

    # Generate Figure 1: Yrast spectral analysis (4 panels)
    print("\nGenerating yrast_spectral_analysis.png...")
    fig1 = plot_yrast_spectral_analysis(df)
    output_path1 = output_dir / 'yrast_spectral_analysis.png'
    fig1.savefig(output_path1, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path1}")
    plt.close(fig1)

    # Generate Figure 2: Yrast comparison (2 panels)
    print("\nGenerating yrast_comparison.png...")
    fig2 = plot_yrast_comparison(df)
    output_path2 = output_dir / 'yrast_comparison.png'
    fig2.savefig(output_path2, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path2}")
    plt.close(fig2)

    print("\n" + "=" * 60)
    print("✓ All yrast plots generated successfully!")
    print("=" * 60)

    # Summary
    print("\nGenerated figures:")
    print(f"  1. {output_path1.name} - 4-panel spectral analysis")
    print(f"  2. {output_path2.name} - Traditional vs. harmonic comparison")
    print(f"\nOutput directory: {output_dir}")


if __name__ == '__main__':
    main()
