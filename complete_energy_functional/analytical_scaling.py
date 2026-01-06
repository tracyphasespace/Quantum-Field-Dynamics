#!/usr/bin/env python3
"""
Analytical Scaling Analysis: Map (β, ξ) degeneracy structure

Goal: Understand relationship between β_V22 ≈ 3.15 and β_theory = 3.058
      when gradient term ξ is included.

Approach:
1. Use Hill vortex profile (no solver needed - analytical)
2. Compute E(β, ξ) on grid
3. Find contours of constant mass
4. Check if β=3.058 line intersects m_electron target
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import json

# Add local imports
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import functionals as func
import solvers as solv


# Physical constants (normalized units for now)
M_ELECTRON = 0.511  # MeV
M_MUON = 105.658
M_TAU = 1776.86


def compute_energy_grid(beta_range, xi_range, R=1.0, U=0.5, A=1.0, r_max=10.0):
    """
    Compute energy E(β, ξ) on grid using Hill vortex profile.

    Returns
    -------
    beta_grid, xi_grid : 2D arrays
        Mesh grid of parameter values
    E_grid : 2D array
        Energy at each (β, ξ) point
    """
    # Create grid
    beta_vals = np.linspace(*beta_range, 50)
    xi_vals = np.linspace(*xi_range, 50)
    beta_grid, xi_grid = np.meshgrid(beta_vals, xi_vals)

    # Fixed density profile (Hill vortex)
    r = np.linspace(0, r_max * R, 500)
    ρ = solv.hill_vortex_profile(r, R, U, A)

    # Compute energy at each grid point
    E_grid = np.zeros_like(beta_grid)

    for i in range(len(xi_vals)):
        for j in range(len(beta_vals)):
            β = beta_grid[i, j]
            ξ = xi_grid[i, j]

            # Compute energy
            E_total, _, _ = func.gradient_energy_functional(ρ, r, ξ, β)
            E_grid[i, j] = E_total

    return beta_grid, xi_grid, E_grid, r, ρ


def find_degeneracy_lines(beta_grid, xi_grid, E_grid, E_targets):
    """
    Find lines in (β, ξ) space where E = constant (mass).

    For each target energy (mass), find contour line.
    """
    from matplotlib.contour import QuadContourSet

    results = {}

    for name, E_target in E_targets.items():
        # Find closest match in grid
        idx = np.argmin(np.abs(E_grid - E_target))
        idx_2d = np.unravel_index(idx, E_grid.shape)

        best_beta = beta_grid[idx_2d]
        best_xi = xi_grid[idx_2d]
        best_E = E_grid[idx_2d]

        results[name] = {
            'E_target': E_target,
            'E_closest': best_E,
            'beta_closest': best_beta,
            'xi_closest': best_xi,
            'error': abs(best_E - E_target) / E_target
        }

    return results


def plot_energy_landscape(beta_grid, xi_grid, E_grid, beta_theory=3.058, beta_v22=3.15):
    """
    Visualize E(β, ξ) landscape with key lines marked.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: Energy surface
    ax = axes[0]
    contour = ax.contourf(beta_grid, xi_grid, E_grid, levels=20, cmap='viridis')
    plt.colorbar(contour, ax=ax, label='Energy E')

    # Mark key β values
    ax.axvline(beta_theory, color='r', ls='--', lw=2, label=f'β theory = {beta_theory}')
    ax.axvline(beta_v22, color='orange', ls='--', lw=2, label=f'β V22 = {beta_v22}')

    # Mark ξ=0 line (V22 limit)
    ax.axhline(0, color='cyan', ls='--', alpha=0.5, label='ξ=0 (V22 limit)')

    ax.set_xlabel('β (vacuum stiffness)')
    ax.set_ylabel('ξ (gradient stiffness)')
    ax.set_title('Energy Landscape E(β, ξ)')
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel 2: Gradient vs Compression ratio
    ax = axes[1]

    # Compute ratio at each point
    ratio_grid = np.zeros_like(E_grid)
    for i in range(E_grid.shape[0]):
        for j in range(E_grid.shape[1]):
            # Approximate from total energy
            # E_total = E_grad + E_comp
            # E_grad / E_comp = ?
            # Placeholder - need actual decomposition
            ratio_grid[i, j] = xi_grid[i, j] / (beta_grid[i, j] + 1e-6)

    contour2 = ax.contourf(beta_grid, xi_grid, ratio_grid, levels=20, cmap='plasma')
    plt.colorbar(contour2, ax=ax, label='ξ/β ratio')

    ax.axvline(beta_theory, color='r', ls='--', lw=2, label=f'β theory = {beta_theory}')
    ax.axvline(beta_v22, color='orange', ls='--', lw=2, label=f'β V22 = {beta_v22}')
    ax.set_xlabel('β (vacuum stiffness)')
    ax.set_ylabel('ξ (gradient stiffness)')
    ax.set_title('Parameter Ratio ξ/β')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig


def analyze_beta_lines(beta_grid, xi_grid, E_grid, beta_values=[3.058, 3.15]):
    """
    Extract E(ξ) along fixed β lines.

    Check: For β=3.058, what ξ gives E ~ m_electron?
    """
    results = {}

    for β_target in beta_values:
        # Find closest β in grid
        j = np.argmin(np.abs(beta_grid[0, :] - β_target))
        β_actual = beta_grid[0, j]

        # Extract E vs ξ along this line
        xi_vals = xi_grid[:, j]
        E_vals = E_grid[:, j]

        results[f'beta_{β_target:.3f}'] = {
            'beta_actual': β_actual,
            'xi_vals': xi_vals.tolist(),
            'E_vals': E_vals.tolist()
        }

    return results


def main():
    """
    Run complete analytical scaling analysis.
    """
    print("="*70)
    print("ANALYTICAL SCALING ANALYSIS")
    print("="*70)
    print()
    print("Computing E(β, ξ) on grid...")
    print()

    # Parameter ranges
    beta_range = (2.5, 3.5)  # Around V22 and theory values
    xi_range = (0.0, 2.0)     # Expect order unity

    # Compute grid
    beta_grid, xi_grid, E_grid, r, ρ = compute_energy_grid(
        beta_range, xi_range,
        R=1.0, U=0.5, A=1.0
    )

    print(f"✓ Grid computed: {beta_grid.shape}")
    print(f"  β range: [{beta_range[0]:.2f}, {beta_range[1]:.2f}]")
    print(f"  ξ range: [{xi_range[0]:.2f}, {xi_range[1]:.2f}]")
    print(f"  E range: [{E_grid.min():.3f}, {E_grid.max():.3f}]")
    print()

    # Target energies (normalized to match grid units - placeholder)
    # TODO: Proper unit conversion
    E_targets = {
        'V22_baseline': E_grid[0, np.argmin(np.abs(beta_grid[0, :] - 3.15))],  # E at (β=3.15, ξ=0)
        'theory': E_grid[0, np.argmin(np.abs(beta_grid[0, :] - 3.058))]        # E at (β=3.058, ξ=0)
    }

    print("Target energies:")
    for name, E in E_targets.items():
        print(f"  {name:15s}: E = {E:.4f}")
    print()

    # Analyze β lines
    print("Extracting E(ξ) along fixed β lines...")
    beta_lines = analyze_beta_lines(beta_grid, xi_grid, E_grid, [3.058, 3.15])

    # Check β=3.058 line
    beta_3058_data = beta_lines['beta_3.058']
    xi_vals = np.array(beta_3058_data['xi_vals'])
    E_vals = np.array(beta_3058_data['E_vals'])

    # Find ξ where E matches V22 baseline
    E_v22_baseline = E_targets['V22_baseline']
    idx_match = np.argmin(np.abs(E_vals - E_v22_baseline))
    xi_match = xi_vals[idx_match]
    E_match = E_vals[idx_match]

    print("CRITICAL RESULT:")
    print("="*70)
    print(f"For β = 3.058 (Golden Loop):")
    print(f"  To match E_V22 = {E_v22_baseline:.4f}, need ξ ≈ {xi_match:.3f}")
    print(f"  Achieved: E = {E_match:.4f}")
    print(f"  Error: {abs(E_match - E_v22_baseline)/E_v22_baseline * 100:.2f}%")
    print()

    if 0.5 <= xi_match <= 2.0:
        print("✓ SUCCESS: ξ is order unity (physically reasonable!)")
        print()
        print("INTERPRETATION:")
        print("  - V22 used β≈3.15 with NO gradient term")
        print("  - Full model uses β=3.058 WITH ξ≈{:.1f}".format(xi_match))
        print("  - Both give same energy (mass)")
        print("  - Gradient term RESOLVES β offset!")
    else:
        print(f"⚠ WARNING: ξ={xi_match:.3f} is outside expected range")
        print("  May indicate scaling issue or unit mismatch")

    print("="*70)
    print()

    # Plot landscape
    print("Generating visualization...")
    fig = plot_energy_landscape(beta_grid, xi_grid, E_grid, beta_theory=3.058, beta_v22=3.15)
    fig.savefig('results/analytical_scaling_landscape.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: results/analytical_scaling_landscape.png")
    print()

    # Plot β=3.058 line specifically
    fig2, ax = plt.subplots(figsize=(10, 6))
    ax.plot(xi_vals, E_vals, 'b-', lw=2, label='E(ξ) at β=3.058')
    ax.axhline(E_v22_baseline, color='orange', ls='--', lw=2, label=f'E_V22 = {E_v22_baseline:.3f}')
    ax.axvline(xi_match, color='r', ls='--', lw=2, alpha=0.5, label=f'ξ = {xi_match:.3f}')
    ax.scatter([xi_match], [E_match], s=100, c='red', marker='*', zorder=5, label='Match point')

    ax.set_xlabel('ξ (gradient stiffness)', fontsize=12)
    ax.set_ylabel('Energy E', fontsize=12)
    ax.set_title('Energy vs ξ at β = 3.058 (Golden Loop)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # Add annotation
    ax.annotate(
        f'ξ ≈ {xi_match:.2f}\n(Order unity!)',
        xy=(xi_match, E_match),
        xytext=(xi_match + 0.3, E_match - 0.2),
        fontsize=11,
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
        arrowprops=dict(arrowstyle='->', lw=1.5)
    )

    plt.tight_layout()
    fig2.savefig('results/beta_3058_line.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: results/beta_3058_line.png")
    print()

    # Save numerical results
    results = {
        'beta_theory': 3.058,
        'beta_v22': 3.15,
        'xi_match': float(xi_match),
        'E_v22_baseline': float(E_v22_baseline),
        'E_match': float(E_match),
        'match_error_percent': float(abs(E_match - E_v22_baseline)/E_v22_baseline * 100),
        'beta_lines': beta_lines,
        'grid_info': {
            'beta_range': list(beta_range),
            'xi_range': list(xi_range),
            'E_min': float(E_grid.min()),
            'E_max': float(E_grid.max())
        }
    }

    with open('results/analytical_scaling_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("✓ Saved: results/analytical_scaling_results.json")
    print()

    print("="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print()
    print("KEY FINDING:")
    print(f"  β = 3.058 + ξ ≈ {xi_match:.2f}  ←→  β = 3.15 + ξ = 0")
    print()
    print("Next step: Run MCMC to confirm ξ posterior peaks at {:.2f}".format(xi_match))
    print("="*70)


if __name__ == '__main__':
    import os
    os.makedirs('results', exist_ok=True)
    main()
