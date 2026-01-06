#!/usr/bin/env python3
"""
Generate manuscript figures for V22 Lepton Analysis paper.

These are the main figures for the manuscript body:
- Figure 1: Golden Loop schematic
- Figure 2: Hill Vortex density profile and streamlines
- Figure 3: Mass spectrum relative error
- Figure 4: Scaling law (U vs mass)
- Figure 5: Cross-sector β consistency

Usage:
    python create_manuscript_figures.py

Outputs to: ./manuscript_figures/
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib import patches
from pathlib import Path

# Publication quality settings
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

RESULTS_DIR = Path("validation_tests/results")
OUTPUT_DIR = Path("manuscript_figures")
OUTPUT_DIR.mkdir(exist_ok=True)

def load_json(filename):
    """Load JSON result file."""
    with open(RESULTS_DIR / filename, 'r') as f:
        return json.load(f)

def create_figure1_golden_loop():
    """
    Figure 1: Schematic of the "Golden Loop" hypothesis.

    Flow: α → β → lepton masses (e, μ, τ)
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Box style
    box_style = dict(boxstyle='round,pad=0.3', facecolor='lightblue',
                     edgecolor='black', linewidth=2)
    arrow_style = dict(arrowstyle='->', lw=2.5, color='darkred')

    # Fine Structure Constant (top left)
    ax.text(1.5, 4.5, r'$\alpha = \frac{1}{137.036}$', fontsize=14, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFE5B4', edgecolor='black', linewidth=2))
    ax.text(1.5, 3.8, 'Fine Structure\nConstant', fontsize=9, ha='center', va='top')

    # Arrow to Beta
    ax.annotate('', xy=(3.8, 4.5), xytext=(2.5, 4.5), arrowprops=arrow_style)
    ax.text(3.15, 4.8, 'conjectured\nrelation', fontsize=8, ha='center', style='italic')

    # Vacuum Stiffness (center)
    ax.text(5, 4.5, r'$\beta \approx 3.058$', fontsize=14, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#B4E5FF', edgecolor='black', linewidth=2))
    ax.text(5, 3.8, 'Vacuum Stiffness', fontsize=9, ha='center', va='top')

    # Arrow down to leptons
    ax.annotate('', xy=(5, 2.8), xytext=(5, 3.6), arrowprops=arrow_style)
    ax.text(5.5, 3.2, 'geometric\nresonances', fontsize=8, ha='left', style='italic')

    # Lepton Masses (bottom)
    leptons = ['e', 'μ', 'τ']
    masses = ['0.511 MeV', '105.66 MeV', '1776.9 MeV']
    x_positions = [2.5, 5, 7.5]
    colors = ['#90EE90', '#FFD700', '#FF6B6B']

    for i, (lep, mass, x, color) in enumerate(zip(leptons, masses, x_positions, colors)):
        # Lepton box
        ax.text(x, 1.8, lep, fontsize=16, ha='center', va='center', weight='bold',
                bbox=dict(boxstyle='circle,pad=0.3', facecolor=color, edgecolor='black', linewidth=2))
        ax.text(x, 1.0, mass, fontsize=8, ha='center', va='top')

        # Arrow from beta to each lepton
        if i == 1:
            ax.annotate('', xy=(x, 2.3), xytext=(5, 2.8), arrowprops=arrow_style)
        else:
            ax.annotate('', xy=(x, 2.3), xytext=(5, 2.8),
                       arrowprops=dict(arrowstyle='->', lw=2.5, color='darkred',
                                     connectionstyle="arc3,rad=0.3"))

    # Cross-sector validation (right side)
    ax.text(8.5, 4.5, 'β from:', fontsize=10, ha='center', weight='bold')
    sectors = ['Nuclear\n3.1 ± 0.1', 'CMB\n3.0-3.2', 'Leptons\n3.058 ± 0.012']
    for i, sector in enumerate(sectors):
        y = 4.0 - i * 0.8
        ax.text(8.5, y, sector, fontsize=8, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='wheat', alpha=0.7))

    # Title
    ax.text(5, 5.5, 'The "Golden Loop": From α to Lepton Masses',
            fontsize=14, ha='center', weight='bold')

    plt.tight_layout()
    for fmt in ['pdf', 'png']:
        plt.savefig(OUTPUT_DIR / f'figure1_golden_loop.{fmt}', dpi=300)

    print(f"✓ Figure 1 (Golden Loop) saved to {OUTPUT_DIR}/figure1_golden_loop.[pdf|png]")
    plt.close()

def create_figure2_hill_vortex():
    """
    Figure 2: Hill Vortex density profile and streamlines.

    Shows the electron solution with R, U parameters.
    """
    data = load_json("three_leptons_beta_from_alpha.json")
    electron = data['results'][0]

    R = electron['R']
    U = electron['U']
    amplitude = electron['amplitude']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Panel A: Density Profile
    r = np.linspace(0, 2*R, 500)

    # Simplified density perturbation model (from HillVortex.lean)
    def density_perturbation(r_val, R, amp):
        if r_val < R:
            return -amp * (1 - (r_val/R)**2)
        else:
            return 0

    rho_vac = 1.0  # Normalized vacuum density
    rho = []
    for r_val in r:
        delta_rho = density_perturbation(r_val, R, amplitude)
        rho.append(rho_vac + delta_rho)

    ax1.plot(r, rho, 'b-', linewidth=2, label='Total density')
    ax1.axhline(y=rho_vac, color='gray', linestyle='--', alpha=0.5, label='Vacuum floor')
    ax1.axvline(x=R, color='red', linestyle='--', alpha=0.7, label=f'R = {R:.3f}')
    ax1.fill_between(r, 0, rho, where=(r < R), alpha=0.2, color='blue', label='Vortex core')

    ax1.set_xlabel('Radius r')
    ax1.set_ylabel('Density ρ (normalized)')
    ax1.set_title('(a) Density Profile (Electron)', fontsize=11)
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.2)

    # Panel B: Streamlines (simplified visualization)
    # Create a grid in cylindrical coordinates (r, z)
    r_grid = np.linspace(0, 2*R, 100)
    theta_vals = np.linspace(0, np.pi, 50)

    # Stream function ψ(r, θ) for Hill vortex (from Lean definition)
    def stream_function(r_val, theta, R, U):
        sin_sq = np.sin(theta)**2
        if r_val < R:
            # Internal: rotational flow
            return -(3*U/(2*R**2)) * (R**2 - r_val**2) * r_val**2 * sin_sq
        else:
            # External: potential flow
            return (U/2) * (r_val**2 - R**3/r_val) * sin_sq

    # Create contour plot
    R_mesh, Theta_mesh = np.meshgrid(r_grid, theta_vals)
    Psi = np.zeros_like(R_mesh)

    for i in range(len(theta_vals)):
        for j in range(len(r_grid)):
            Psi[i, j] = stream_function(R_mesh[i, j], Theta_mesh[i, j], R, U)

    # Convert to Cartesian for plotting
    X = R_mesh * np.sin(Theta_mesh)
    Y = R_mesh * np.cos(Theta_mesh)

    levels = np.linspace(Psi.min(), Psi.max(), 20)
    contour = ax2.contour(X, Y, Psi, levels=levels, colors='blue', linewidths=1, alpha=0.6)

    # Add vortex boundary
    circle = Circle((0, R), R, fill=False, edgecolor='red', linewidth=2, linestyle='--', label=f'R = {R:.3f}')
    ax2.add_patch(circle)

    # Add circulation arrow
    arrow = FancyArrowPatch((0.3*R, R), (0.5*R, R),
                           arrowstyle='->', mutation_scale=20,
                           color='darkred', linewidth=2)
    ax2.add_patch(arrow)
    ax2.text(0.4*R, R + 0.15*R, f'U = {U:.4f}', fontsize=9, ha='center')

    ax2.set_xlabel('x (transverse)')
    ax2.set_ylabel('z (axis)')
    ax2.set_title('(b) Streamlines (Cross-section)', fontsize=11)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.set_xlim(-0.5*R, R)
    ax2.set_ylim(0, 2*R)

    plt.tight_layout()
    for fmt in ['pdf', 'png']:
        plt.savefig(OUTPUT_DIR / f'figure2_hill_vortex.{fmt}', dpi=300)

    print(f"✓ Figure 2 (Hill Vortex) saved to {OUTPUT_DIR}/figure2_hill_vortex.[pdf|png]")
    plt.close()

def create_figure3_mass_spectrum_error():
    """
    Figure 3: Mass spectrum relative error.

    Shows residuals for e, μ, τ compared to CODATA values.
    """
    data = load_json("three_leptons_beta_from_alpha.json")
    results = data['results']

    particles = [r['particle'] for r in results]
    residuals = [abs(r['residual']) for r in results]

    # CODATA mass values (for reference)
    codata_masses = {
        'electron': 1.0,
        'muon': 206.7682826,
        'tau': 3477.228
    }

    relative_errors = []
    for r in results:
        target = codata_masses[r['particle']]
        rel_err = abs(r['residual']) / target
        relative_errors.append(rel_err)

    fig, ax = plt.subplots(figsize=(7, 5))

    colors = ['#90EE90', '#FFD700', '#FF6B6B']
    x = np.arange(len(particles))

    bars = ax.bar(x, relative_errors, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    ax.set_ylabel('Relative Error |residual| / mass', fontsize=12)
    ax.set_xlabel('Lepton Generation', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(['Electron (e)', 'Muon (μ)', 'Tau (τ)'], fontsize=11)
    ax.set_yscale('log')
    ax.set_ylim(1e-11, 1e-6)
    ax.grid(True, alpha=0.3, which='both', axis='y')
    ax.set_title('Lepton Mass Spectrum: Relative Error vs CODATA 2018', fontsize=13, weight='bold')

    # Add residual values as text
    for i, (bar, rel_err, abs_res) in enumerate(zip(bars, relative_errors, residuals)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height * 3,
                f'{rel_err:.1e}\n({abs_res:.1e})',
                ha='center', va='bottom', fontsize=8)

    # Add reference line for 10^-7
    ax.axhline(y=1e-7, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
               label=r'$10^{-7}$ precision threshold')
    ax.legend(loc='upper right', fontsize=10)

    # Add text box with β value
    textstr = r'$\beta = 3.058$ (from $\alpha$)' + '\nSingle parameter fit'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    for fmt in ['pdf', 'png']:
        plt.savefig(OUTPUT_DIR / f'figure3_mass_spectrum.{fmt}', dpi=300)

    print(f"✓ Figure 3 (Mass Spectrum) saved to {OUTPUT_DIR}/figure3_mass_spectrum.[pdf|png]")
    plt.close()

def create_figure4_scaling_law():
    """
    Figure 4: Scaling law U vs mass.

    Shows approximate U ∝ √m relationship.
    """
    data = load_json("three_leptons_beta_from_alpha.json")
    results = data['results']

    particles = [r['particle'] for r in results]
    masses = [r['target_mass'] for r in results]
    U_vals = [r['U'] for r in results]

    # Also get sqrt(masses) for scaling
    sqrt_masses = np.sqrt(masses)

    fig, ax = plt.subplots(figsize=(7, 5))

    colors = ['#90EE90', '#FFD700', '#FF6B6B']

    # Scatter plot
    for i, (m, u, name, color) in enumerate(zip(masses, U_vals, particles, colors)):
        ax.scatter(m, u, s=200, color=color, edgecolors='black',
                  linewidth=2, zorder=3, alpha=0.8, label=name.capitalize())

    # Fit power law: U = a * m^b
    coeffs = np.polyfit(np.log(masses), np.log(U_vals), 1)
    power = coeffs[0]
    prefactor = np.exp(coeffs[1])

    m_fit = np.logspace(np.log10(min(masses)*0.8), np.log10(max(masses)*1.2), 100)
    U_fit = prefactor * m_fit**power

    ax.plot(m_fit, U_fit, '--', color='darkblue', linewidth=2.5, alpha=0.7,
            label=f'Fit: U ∝ m^{power:.3f}')

    # Add sqrt(m) reference for comparison
    U_sqrt_ref = U_vals[0] * np.sqrt(m_fit / masses[0])
    ax.plot(m_fit, U_sqrt_ref, ':', color='gray', linewidth=2, alpha=0.5,
            label=r'U ∝ $\sqrt{m}$ reference')

    ax.set_xlabel('Mass Ratio (relative to electron)', fontsize=12)
    ax.set_ylabel('Circulation Velocity U', fontsize=12)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('Scaling Law: Circulation Velocity vs Mass', fontsize=13, weight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, which='both')

    # Add text box with interpretation
    textstr = f'Power law exponent: {power:.3f}\n' + r'(Close to $\sqrt{m}$ = 0.500)'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.95, 0.05, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)

    plt.tight_layout()
    for fmt in ['pdf', 'png']:
        plt.savefig(OUTPUT_DIR / f'figure4_scaling_law.{fmt}', dpi=300)

    print(f"✓ Figure 4 (Scaling Law) saved to {OUTPUT_DIR}/figure4_scaling_law.[pdf|png]")
    plt.close()

def create_figure5_cross_sector_beta():
    """
    Figure 5: Cross-sector β consistency.

    Shows β values from particle, nuclear, and cosmology sectors.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # Data from different sectors
    sectors = ['Lepton\nMasses\n(this work)', 'Nuclear\nStability', 'CMB\nMorphology']
    beta_values = [3.058, 3.1, 3.1]
    uncertainties = [0.012, 0.1, 0.15]  # Approximate uncertainties
    colors = ['#4A90E2', '#E94B3C', '#50C878']

    y_positions = np.arange(len(sectors))

    # Horizontal error bars
    for i, (y, beta, err, color, sector) in enumerate(zip(y_positions, beta_values, uncertainties, colors, sectors)):
        ax.errorbar(beta, y, xerr=err, fmt='o', markersize=12,
                   color=color, ecolor=color, capsize=8, capthick=2,
                   linewidth=2.5, alpha=0.8, label=sector)

        # Add text with value
        ax.text(beta + err + 0.05, y, f'{beta:.3f} ± {err:.3f}',
               fontsize=10, va='center', ha='left')

    # Add shaded consistency region
    beta_mean = np.mean(beta_values)
    beta_std = np.std(beta_values)
    ax.axvspan(beta_mean - beta_std, beta_mean + beta_std,
              alpha=0.2, color='yellow', label=f'Mean ± σ\n({beta_mean:.2f} ± {beta_std:.2f})')

    ax.set_xlabel(r'Vacuum Stiffness Parameter $\beta$', fontsize=12)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(sectors, fontsize=11)
    ax.set_xlim(2.7, 3.4)
    ax.set_title('Cross-Sector Consistency: β from Independent Observables',
                fontsize=13, weight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)

    # Add scale annotation
    ax.text(0.02, 0.98, '40 orders of magnitude\nin physical scale',
           transform=ax.transAxes, fontsize=9, style='italic',
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    plt.tight_layout()
    for fmt in ['pdf', 'png']:
        plt.savefig(OUTPUT_DIR / f'figure5_cross_sector.{fmt}', dpi=300)

    print(f"✓ Figure 5 (Cross-Sector β) saved to {OUTPUT_DIR}/figure5_cross_sector.[pdf|png]")
    plt.close()

def create_all_manuscript_figures():
    """Generate all manuscript figures."""
    print("\nGenerating manuscript figures for V22 Lepton Analysis paper...")
    print(f"Reading data from: {RESULTS_DIR}")
    print(f"Saving figures to: {OUTPUT_DIR}\n")

    create_figure1_golden_loop()
    create_figure2_hill_vortex()
    create_figure3_mass_spectrum_error()
    create_figure4_scaling_law()
    create_figure5_cross_sector_beta()

    print(f"\n✓ All manuscript figures generated successfully!")
    print(f"\nOutput directory: {OUTPUT_DIR.absolute()}")
    print("\nFigures created:")
    print("  • Figure 1: Golden Loop schematic")
    print("  • Figure 2: Hill Vortex density profile and streamlines")
    print("  • Figure 3: Mass spectrum relative error")
    print("  • Figure 4: Scaling law (U vs mass)")
    print("  • Figure 5: Cross-sector β consistency")
    print("\nFormats: PDF (vector) + PNG (raster, 300 dpi)")

if __name__ == "__main__":
    create_all_manuscript_figures()
