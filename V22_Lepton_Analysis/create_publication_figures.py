#!/usr/bin/env python3
"""
Generate publication-quality figures for lepton Hill vortex manuscript.

Creates Figures 1-5 from validation test JSON results:
- Figure 1: Main result - lepton mass precision
- Figure 2: Grid convergence test
- Figure 3: Multi-start robustness
- Figure 4: Profile sensitivity
- Figure 5: Parameter scaling law

Usage:
    python create_publication_figures.py

Outputs to: ./publication_figures/
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Publication quality settings
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.figsize': (7, 5),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'text.usetex': False,  # Set True if LaTeX installed
})

RESULTS_DIR = Path("validation_tests/results")
OUTPUT_DIR = Path("publication_figures")
OUTPUT_DIR.mkdir(exist_ok=True)

def load_json(filename):
    """Load JSON result file."""
    with open(RESULTS_DIR / filename, 'r') as f:
        return json.load(f)

def create_figure1_main_result():
    """
    Figure 1: Lepton mass precision - core result of the paper.
    Bar chart showing target vs achieved masses with log-scale residuals.
    """
    data = load_json("three_leptons_beta_from_alpha.json")
    results = data['results']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    # Panel A: Mass ratios (target vs achieved)
    particles = [r['particle'] for r in results]
    targets = [r['target_mass'] for r in results]
    achieved = [r['E_total'] for r in results]

    x = np.arange(len(particles))
    width = 0.35

    ax1.bar(x - width/2, targets, width, label='Target', alpha=0.8, color='#2E86AB')
    ax1.bar(x + width/2, achieved, width, label='Achieved', alpha=0.8, color='#A23B72')

    ax1.set_ylabel('Mass Ratio (relative to electron)')
    ax1.set_xlabel('Lepton')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['Electron', 'Muon', 'Tau'])
    ax1.set_yscale('log')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3, which='both')
    ax1.set_title(r'(a) Target vs Achieved Mass Ratios ($\beta = 3.058$)', fontsize=10)

    # Panel B: Residuals (log scale)
    residuals = [abs(r['residual']) for r in results]
    colors = ['#06A77D', '#D4B483', '#F18F01']

    bars = ax2.bar(particles, residuals, color=colors, alpha=0.8)
    ax2.set_ylabel('Absolute Residual')
    ax2.set_xlabel('Lepton')
    ax2.set_yscale('log')
    ax2.set_ylim(1e-11, 1e-6)
    ax2.grid(True, alpha=0.3, which='both', axis='y')
    ax2.set_title('(b) Optimization Residuals', fontsize=10)
    ax2.set_xticklabels(['Electron', 'Muon', 'Tau'])

    # Add residual values as text
    for i, (bar, res, particle) in enumerate(zip(bars, residuals, particles)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height * 2,
                f'{res:.1e}',
                ha='center', va='bottom', fontsize=8)

    plt.tight_layout()

    # Save in multiple formats
    for fmt in ['pdf', 'png']:
        plt.savefig(OUTPUT_DIR / f'figure1_main_result.{fmt}', dpi=300)

    print(f"✓ Figure 1 saved to {OUTPUT_DIR}/figure1_main_result.[pdf|png]")
    plt.close()

def create_figure2_grid_convergence():
    """
    Figure 2: Grid convergence test.
    Shows parameter drift vs grid resolution.
    """
    data = load_json("grid_convergence_results.json")
    grids = data['grid_resolutions']
    analysis = data['convergence_analysis']['convergence_data']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    # Panel A: Parameter drift
    grid_sizes = [g['num_r'] * g['num_theta'] for g in grids]
    drift_R = [a['drift_R_percent'] for a in analysis]
    drift_U = [a['drift_U_percent'] for a in analysis]
    drift_amp = [a['drift_amplitude_percent'] for a in analysis]

    ax1.plot(grid_sizes, drift_R, 'o-', label='R (radius)', color='#2E86AB', linewidth=2, markersize=6)
    ax1.plot(grid_sizes, drift_U, 's-', label='U (velocity)', color='#A23B72', linewidth=2, markersize=6)
    ax1.plot(grid_sizes, drift_amp, '^-', label='Amplitude', color='#F18F01', linewidth=2, markersize=6)

    ax1.set_xlabel('Grid Points (num_r × num_theta)')
    ax1.set_ylabel('Parameter Drift (%)')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3, which='both')
    ax1.set_title('(a) Parameter Convergence', fontsize=10)

    # Add 1% threshold line
    ax1.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, linewidth=1, label='1% threshold')

    # Panel B: Energy drift
    drift_E = [a['drift_E_total_percent'] for a in analysis]

    ax2.plot(grid_sizes, drift_E, 'o-', color='#06A77D', linewidth=2, markersize=6)
    ax2.set_xlabel('Grid Points (num_r × num_theta)')
    ax2.set_ylabel('Energy Drift (%)')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.set_title('(b) Energy Convergence', fontsize=10)

    # Add text showing max drift
    max_drift = max(drift_R[:-1] + drift_U[:-1] + drift_amp[:-1])
    ax1.text(0.05, 0.95, f'Max drift: {max_drift:.2f}%\n(< 1% criterion)',
             transform=ax1.transAxes, fontsize=8,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    for fmt in ['pdf', 'png']:
        plt.savefig(OUTPUT_DIR / f'figure2_grid_convergence.{fmt}', dpi=300)

    print(f"✓ Figure 2 saved to {OUTPUT_DIR}/figure2_grid_convergence.[pdf|png]")
    plt.close()

def create_figure3_multistart():
    """
    Figure 3: Multi-start robustness test.
    2D scatter plot in (R, U) parameter space showing convergence to single solution.
    """
    data = load_json("multistart_robustness_results.json")
    solutions = data['solutions']

    # Extract converged solutions
    converged = [s for s in solutions if s['converged']]
    R_vals = [s['R'] for s in converged]
    U_vals = [s['U'] for s in converged]
    residuals = [abs(s['residual']) for s in converged]

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    # Panel A: Parameter space scatter
    scatter = ax1.scatter(R_vals, U_vals, c=np.log10(residuals),
                         s=50, alpha=0.6, cmap='viridis', edgecolors='black', linewidth=0.5)

    ax1.set_xlabel('R (vortex radius)')
    ax1.set_ylabel('U (circulation velocity)')
    ax1.set_title(f'(a) {len(converged)}/{len(solutions)} Runs Converged', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax1, label='log₁₀(residual)')

    # Add statistics box
    R_mean = np.mean(R_vals)
    R_std = np.std(R_vals)
    U_mean = np.mean(U_vals)
    U_std = np.std(U_vals)

    stats_text = f'R: {R_mean:.3f} ± {R_std:.3f}\nU: {U_mean:.4f} ± {U_std:.4f}'
    ax1.text(0.05, 0.95, stats_text, transform=ax1.transAxes,
             fontsize=8, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Panel B: Residual distribution
    ax2.hist(np.log10(residuals), bins=20, color='#06A77D', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('log₁₀(residual)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('(b) Residual Distribution', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add median line
    median_log_res = np.median(np.log10(residuals))
    ax2.axvline(median_log_res, color='red', linestyle='--', linewidth=2,
                label=f'Median: {10**median_log_res:.1e}')
    ax2.legend()

    plt.tight_layout()

    for fmt in ['pdf', 'png']:
        plt.savefig(OUTPUT_DIR / f'figure3_multistart_robustness.{fmt}', dpi=300)

    print(f"✓ Figure 3 saved to {OUTPUT_DIR}/figure3_multistart_robustness.[pdf|png]")
    plt.close()

def create_figure4_profile_sensitivity():
    """
    Figure 4: Profile sensitivity test.
    Bar chart comparing results across 4 different velocity profile forms.
    """
    data = load_json("profile_sensitivity_results.json")
    results = data['results']

    # Extract data
    profile_names = [r['profile'] for r in results]
    residuals = [abs(r['residual']) for r in results]
    R_vals = [r['R'] for r in results]
    U_vals = [r['U'] for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))

    colors = ['#2E86AB', '#A23B72', '#F18F01', '#06A77D']
    x = np.arange(len(profile_names))

    # Panel A: Residuals
    bars = axes[0].bar(x, residuals, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    axes[0].set_ylabel('Absolute Residual')
    axes[0].set_xlabel('Velocity Profile')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(profile_names, rotation=45, ha='right', fontsize=8)
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3, which='both', axis='y')
    axes[0].set_title('(a) Residuals Across Profiles', fontsize=10)

    # Panel B: R values
    axes[1].bar(x, R_vals, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    axes[1].set_ylabel('R (vortex radius)')
    axes[1].set_xlabel('Velocity Profile')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(profile_names, rotation=45, ha='right', fontsize=8)
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].set_title('(b) Optimized Radius', fontsize=10)

    # Panel C: U values
    axes[2].bar(x, U_vals, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    axes[2].set_ylabel('U (circulation velocity)')
    axes[2].set_xlabel('Velocity Profile')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(profile_names, rotation=45, ha='right', fontsize=8)
    axes[2].grid(True, alpha=0.3, axis='y')
    axes[2].set_title('(c) Optimized Velocity', fontsize=10)

    # Add statistics
    R_std = np.std(R_vals)
    U_std = np.std(U_vals)
    res_range = max(residuals) / min(residuals)

    stats_text = f'R std: {R_std:.4f}\nU std: {U_std:.4f}\nRes range: {res_range:.1f}×'
    axes[0].text(0.05, 0.95, stats_text, transform=axes[0].transAxes,
                fontsize=7, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    for fmt in ['pdf', 'png']:
        plt.savefig(OUTPUT_DIR / f'figure4_profile_sensitivity.{fmt}', dpi=300)

    print(f"✓ Figure 4 saved to {OUTPUT_DIR}/figure4_profile_sensitivity.[pdf|png]")
    plt.close()

def create_figure5_scaling_law():
    """
    Figure 5: Parameter scaling law.
    U vs sqrt(m) showing systematic relationship and deviations.
    """
    data = load_json("three_leptons_beta_from_alpha.json")
    results = data['results']

    particles = [r['particle'] for r in results]
    masses = [r['target_mass'] for r in results]
    U_vals = [r['U'] for r in results]
    R_vals = [r['R'] for r in results]

    sqrt_masses = np.sqrt(masses)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    # Panel A: U vs sqrt(m)
    ax1.scatter(sqrt_masses, U_vals, s=100, color='#2E86AB', edgecolors='black',
                linewidth=1, zorder=3, alpha=0.8)

    # Fit linear trend
    coeffs = np.polyfit(sqrt_masses, U_vals, 1)
    fit_line = np.poly1d(coeffs)
    x_fit = np.linspace(0, max(sqrt_masses)*1.1, 100)
    ax1.plot(x_fit, fit_line(x_fit), '--', color='red', linewidth=2,
             label=f'Linear fit: U = {coeffs[0]:.4f}√m + {coeffs[1]:.4f}')

    # Add particle labels
    for i, (x, y, name) in enumerate(zip(sqrt_masses, U_vals, particles)):
        ax1.annotate(name.capitalize(), (x, y), xytext=(5, 5),
                    textcoords='offset points', fontsize=8)

    ax1.set_xlabel('√(mass ratio)')
    ax1.set_ylabel('U (circulation velocity)')
    ax1.set_title('(a) Circulation Velocity Scaling', fontsize=10)
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Panel B: Deviations from perfect scaling
    # Perfect scaling would have constant U/sqrt(m)
    normalized_U = np.array(U_vals) / np.array(sqrt_masses)
    mean_normalized = np.mean(normalized_U)
    deviations = (normalized_U - mean_normalized) / mean_normalized * 100

    colors = ['#06A77D', '#D4B483', '#F18F01']
    bars = ax2.bar(particles, deviations, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax2.set_ylabel('Deviation from Mean Scaling (%)')
    ax2.set_xlabel('Lepton')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.set_xticklabels(['Electron', 'Muon', 'Tau'])
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_title('(b) Deviations from Perfect U ∝ √m', fontsize=10)

    # Add deviation values
    for bar, dev in zip(bars, deviations):
        height = bar.get_height()
        y_pos = height + 0.5 if height > 0 else height - 0.5
        ax2.text(bar.get_x() + bar.get_width()/2., y_pos,
                f'{dev:+.1f}%', ha='center', va='bottom' if height > 0 else 'top',
                fontsize=8)

    plt.tight_layout()

    for fmt in ['pdf', 'png']:
        plt.savefig(OUTPUT_DIR / f'figure5_scaling_law.{fmt}', dpi=300)

    print(f"✓ Figure 5 saved to {OUTPUT_DIR}/figure5_scaling_law.[pdf|png]")
    plt.close()

def create_all_figures():
    """Generate all publication figures."""
    print("\nGenerating publication figures for lepton Hill vortex manuscript...")
    print(f"Reading data from: {RESULTS_DIR}")
    print(f"Saving figures to: {OUTPUT_DIR}\n")

    create_figure1_main_result()
    create_figure2_grid_convergence()
    create_figure3_multistart()
    create_figure4_profile_sensitivity()
    create_figure5_scaling_law()

    print(f"\n✓ All figures generated successfully!")
    print(f"\nOutput directory: {OUTPUT_DIR.absolute()}")
    print("\nFigures created:")
    print("  • Figure 1: Main result (mass precision)")
    print("  • Figure 2: Grid convergence")
    print("  • Figure 3: Multi-start robustness")
    print("  • Figure 4: Profile sensitivity")
    print("  • Figure 5: Parameter scaling law")
    print("\nFormats: PDF (vector) + PNG (raster, 300 dpi)")

if __name__ == "__main__":
    create_all_figures()
