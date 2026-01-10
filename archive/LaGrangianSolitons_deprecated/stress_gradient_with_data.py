#!/usr/bin/env python3
"""
STRESS GRADIENT MANIFOLD WITH REAL NUCLEAR DATA
================================================================================
Calculate ∇|N| to show decay directions on the geometric stress manifold
Overlay real stable and unstable isotope data

Terminology: "Neutron Core (NC)" = the frozen core region (high A-Z)
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import FancyArrowPatch
import matplotlib.patches as mpatches

# 15-Path Model Parameters
c1_0 = 0.970454
c2_0 = 0.234920
c3_0 = -1.928732
dc1 = -0.021538
dc2 = 0.001730
dc3 = -0.540530

def calculate_N_continuous(A, Z):
    """Calculate continuous geometric coordinate N(A,Z)"""
    if A < 1:
        return 0
    A_term = A**(2/3)
    Z_0 = c1_0 * A_term + c2_0 * A + c3_0
    dZ = dc1 * A_term + dc2 * A + dc3
    if abs(dZ) < 1e-10:
        return 0
    return (Z - Z_0) / dZ

def calculate_stress_gradient(A, Z, delta=1.0):
    """
    Calculate stress gradient ∇|N| = (∂|N|/∂A, ∂|N|/∂Z)
    Uses finite differences for numerical gradient
    """
    N_center = calculate_N_continuous(A, Z)

    # Partial derivatives using centered differences
    N_A_plus = calculate_N_continuous(A + delta, Z)
    N_A_minus = calculate_N_continuous(A - delta, Z)
    dN_dA = (N_A_plus - N_A_minus) / (2 * delta)

    N_Z_plus = calculate_N_continuous(A, Z + delta)
    N_Z_minus = calculate_N_continuous(A, Z - delta)
    dN_dZ = (N_Z_plus - N_Z_minus) / (2 * delta)

    # Gradient of |N| (stress)
    # d|N|/dA = sign(N) * dN/dA
    # d|N|/dZ = sign(N) * dN/dZ
    sign_N = np.sign(N_center)

    grad_stress_A = sign_N * dN_dA if abs(N_center) > 0.01 else 0
    grad_stress_Z = sign_N * dN_dZ if abs(N_center) > 0.01 else 0

    return grad_stress_A, grad_stress_Z

# Load real nuclear data (stable isotopes from our previous work)
with open('qfd_optimized_suite.py', 'r') as f:
    content = f.read()
start = content.find('test_nuclides = [')
end = content.find(']', start) + 1
stable_nuclei = eval(content[start:end].replace('test_nuclides = ', ''))

# Extract A, Z for stable nuclei
stable_A = np.array([A for _, Z, A in stable_nuclei])
stable_Z = np.array([Z for _, Z, A in stable_nuclei])
stable_N_coord = np.array([calculate_N_continuous(A, Z) for _, Z, A in stable_nuclei])
stable_stress = np.abs(stable_N_coord)

# Load unstable isotopes (from earlier test)
unstable_isotopes = [
    ("H-3", 1, 3), ("C-14", 6, 14), ("Na-22", 11, 22), ("P-32", 15, 32),
    ("S-35", 16, 35), ("K-40", 19, 40), ("Ca-45", 20, 45), ("Fe-55", 26, 55),
    ("Co-60", 27, 60), ("Sr-90", 38, 90), ("I-131", 53, 131), ("Cs-137", 55, 137),
    ("Pm-147", 61, 147), ("Ra-226", 88, 226), ("Th-232", 90, 232),
    ("U-235", 92, 235), ("U-238", 92, 238), ("Pu-239", 94, 239),
]

unstable_A = np.array([A for _, Z, A in unstable_isotopes])
unstable_Z = np.array([Z for _, Z, A in unstable_isotopes])
unstable_N_coord = np.array([calculate_N_continuous(A, Z) for _, Z, A in unstable_isotopes])
unstable_stress = np.abs(unstable_N_coord)

print("="*80)
print("STRESS GRADIENT MANIFOLD WITH REAL NUCLEAR DATA")
print("="*80)
print()
print(f"Loaded {len(stable_nuclei)} stable isotopes")
print(f"Loaded {len(unstable_isotopes)} unstable isotopes")
print()

# Create fine grid for stress field
A_fine = np.linspace(1, 260, 2000)
Z_fine = np.linspace(1, 110, 1000)
AA, ZZ = np.meshgrid(A_fine, Z_fine)

# Calculate N field
A_term = AA**(2/3)
Z_0 = c1_0 * A_term + c2_0 * AA + c3_0
dZ_field = dc1 * A_term + dc2 * AA + dc3
dZ_field[np.abs(dZ_field) < 1e-9] = 1e-9
N_field = (ZZ - Z_0) / dZ_field

# Stress field
Stress_field = np.abs(N_field)

# Mask extreme values
Stress_field_masked = np.ma.masked_where(Stress_field > 6, Stress_field)

# Calculate gradient on coarser grid (for arrows)
A_coarse = np.arange(10, 250, 15)
Z_coarse = np.arange(5, 105, 8)

grad_A_vals = []
grad_Z_vals = []
A_arrow = []
Z_arrow = []
stress_arrow = []

for A in A_coarse:
    for Z in Z_coarse:
        if Z < A:  # Physical constraint
            grad_A, grad_Z = calculate_stress_gradient(A, Z)
            stress = abs(calculate_N_continuous(A, Z))

            # Only show arrows in moderate stress regions
            if 0.5 < stress < 4.0:
                grad_A_vals.append(grad_A)
                grad_Z_vals.append(grad_Z)
                A_arrow.append(A)
                Z_arrow.append(Z)
                stress_arrow.append(stress)

grad_A_vals = np.array(grad_A_vals)
grad_Z_vals = np.array(grad_Z_vals)
A_arrow = np.array(A_arrow)
Z_arrow = np.array(Z_arrow)
stress_arrow = np.array(stress_arrow)

# Normalize gradient for visualization
grad_magnitude = np.sqrt(grad_A_vals**2 + grad_Z_vals**2)
grad_magnitude[grad_magnitude < 1e-6] = 1e-6
grad_A_norm = grad_A_vals / grad_magnitude
grad_Z_norm = grad_Z_vals / grad_magnitude

print(f"Calculated {len(A_arrow)} gradient vectors")
print()

# ============================================================================
# CREATE FIGURE
# ============================================================================

fig = plt.figure(figsize=(20, 16))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)

# Panel A: Stress Field with Real Data
ax1 = fig.add_subplot(gs[0, :])

# Plot stress field
im1 = ax1.imshow(Stress_field_masked, origin='lower', extent=[1, 260, 1, 110],
                 cmap='magma_r', vmin=0, vmax=4.5, aspect='auto', alpha=0.8)

# Contour lines at integer N
ax1.contour(AA, ZZ, N_field, levels=np.arange(-4, 5, 1.0),
            colors='white', linewidths=1.0, alpha=0.4, linestyles='--')

# Overlay stable nuclei
scatter_stable = ax1.scatter(stable_A, stable_Z, c=stable_stress,
                            cmap='viridis', s=25, edgecolors='white',
                            linewidths=0.5, vmin=0, vmax=4,
                            marker='o', label='Stable Isotopes', zorder=10)

# Overlay unstable nuclei
ax1.scatter(unstable_A, unstable_Z, c=unstable_stress,
           cmap='Reds', s=80, edgecolors='yellow',
           linewidths=1.5, marker='s', vmin=0, vmax=4,
           label='Unstable Isotopes', zorder=11)

ax1.set_xlabel('Mass Number A (Soliton Mass)', fontsize=13, fontweight='bold')
ax1.set_ylabel('Charge Z (Topological Winding)', fontsize=13, fontweight='bold')
ax1.set_title('(A) Geometric Stress Manifold with Real Nuclear Data',
              fontsize=15, fontweight='bold', pad=15)

# Colorbar
cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.03, pad=0.01)
cbar1.set_label('Geometric Stress |N|', fontsize=12, fontweight='bold')

# Legend
ax1.legend(loc='upper left', fontsize=11, framealpha=0.9)

# Annotations
ax1.text(0.02, 0.95, 'Dark Valley = Low Stress (Stable Region)\nBright Slopes = High Stress (Drip Lines)',
         transform=ax1.transAxes, fontsize=11, va='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax1.text(0.98, 0.05, 'Neutron Core (NC) →',
         transform=ax1.transAxes, fontsize=11, ha='right', va='bottom',
         color='white', fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='black', alpha=0.6))

ax1.set_xlim(0, 260)
ax1.set_ylim(0, 110)
ax1.grid(alpha=0.2)

# Panel B: Stress Gradient Vectors (Decay Direction)
ax2 = fig.add_subplot(gs[1, 0])

# Plot stress field (lighter)
im2 = ax2.imshow(Stress_field_masked, origin='lower', extent=[1, 260, 1, 110],
                 cmap='gray_r', vmin=0, vmax=4.5, aspect='auto', alpha=0.3)

# Plot gradient vectors (pointing DOWNHILL toward lower stress)
# Negative gradient = direction of stress decrease
arrow_scale = 8.0
quiver = ax2.quiver(A_arrow, Z_arrow, -grad_A_norm, -grad_Z_norm,
                   stress_arrow, cmap='plasma', scale=30, width=0.003,
                   headwidth=4, headlength=5, alpha=0.7, clim=[0.5, 4])

# Overlay stable nuclei
ax2.scatter(stable_A, stable_Z, c='lime', s=15, alpha=0.6,
           edgecolors='darkgreen', linewidths=0.3, label='Stable')

# Overlay unstable nuclei
ax2.scatter(unstable_A, unstable_Z, c='red', s=60, marker='s',
           edgecolors='yellow', linewidths=1, label='Unstable', zorder=10)

ax2.set_xlabel('Mass Number A', fontsize=12, fontweight='bold')
ax2.set_ylabel('Charge Z', fontsize=12, fontweight='bold')
ax2.set_title('(B) Stress Gradient ∇|N| (Decay Direction)',
              fontsize=14, fontweight='bold', pad=10)

cbar2 = fig.colorbar(quiver, ax=ax2, fraction=0.03, pad=0.01)
cbar2.set_label('Local Stress |N|', fontsize=11)

ax2.legend(loc='upper left', fontsize=10)
ax2.text(0.02, 0.95, 'Arrows point toward LOWER stress\n(Predicted decay direction)',
         transform=ax2.transAxes, fontsize=10, va='top',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

ax2.set_xlim(0, 260)
ax2.set_ylim(0, 110)
ax2.grid(alpha=0.3)

# Panel C: N-coordinate distribution
ax3 = fig.add_subplot(gs[1, 1])

# Histogram of N-coordinates
bins = np.linspace(-4, 4, 50)
ax3.hist(stable_N_coord, bins=bins, color='green', alpha=0.6,
         label=f'Stable (n={len(stable_nuclei)})', edgecolor='black', linewidth=1)
ax3.hist(unstable_N_coord, bins=bins, color='red', alpha=0.6,
         label=f'Unstable (n={len(unstable_isotopes)})', edgecolor='black', linewidth=1)

# Mark integer paths
for N_int in range(-4, 5):
    ax3.axvline(N_int, color='blue', linestyle='--', alpha=0.4, linewidth=1)

# Mark ground state
ax3.axvline(0, color='gold', linestyle='-', linewidth=3, alpha=0.7, label='Ground State (N=0)')

# Shaded stability regions
ax3.axvspan(-2.5, 2.5, alpha=0.1, color='green', label='Stable Region (|N|<2.5)')
ax3.axvspan(-4, -3.5, alpha=0.1, color='red')
ax3.axvspan(3.5, 4, alpha=0.1, color='red', label='Drip Lines (|N|>3.5)')

ax3.set_xlabel('Geometric Coordinate N', fontsize=12, fontweight='bold')
ax3.set_ylabel('Number of Isotopes', fontsize=12, fontweight='bold')
ax3.set_title('(C) Distribution of Isotopes in Geometric Coordinates',
              fontsize=14, fontweight='bold', pad=10)
ax3.legend(loc='upper right', fontsize=9)
ax3.grid(alpha=0.3)

plt.suptitle('GEOMETRIC STRESS MANIFOLD: Continuous Field Theory of Nuclear Stability\n' +
             'Neutron Core (NC) = Frozen Core Region | Stress = |N| = Deviation from Ground State',
             fontsize=16, fontweight='bold', y=0.995)

plt.savefig('stress_gradient_manifold_with_data.png', dpi=200, bbox_inches='tight')
plt.savefig('stress_gradient_manifold_with_data.pdf', bbox_inches='tight')

print("="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print()
print("Figures saved:")
print("  - stress_gradient_manifold_with_data.png (200 DPI)")
print("  - stress_gradient_manifold_with_data.pdf (vector)")
print()

# Analyze stress statistics
print("STRESS STATISTICS:")
print("-"*80)
print(f"Stable isotopes:")
print(f"  Mean stress: {np.mean(stable_stress):.3f}")
print(f"  Median stress: {np.median(stable_stress):.3f}")
print(f"  Max stress: {np.max(stable_stress):.3f}")
print(f"  % with |N| < 2.5: {100*np.sum(stable_stress < 2.5)/len(stable_stress):.1f}%")
print()
print(f"Unstable isotopes:")
print(f"  Mean stress: {np.mean(unstable_stress):.3f}")
print(f"  Median stress: {np.median(unstable_stress):.3f}")
print(f"  Max stress: {np.max(unstable_stress):.3f}")
print(f"  % with |N| < 2.5: {100*np.sum(unstable_stress < 2.5)/len(unstable_stress):.1f}%")
print()

print("="*80)
print("INTERPRETATION")
print("="*80)
print()
print("The stress gradient ∇|N| points DOWNHILL toward lower stress.")
print("Unstable isotopes should decay in the direction opposite to ∇|N|.")
print()
print("Neutron Core (NC) interpretation:")
print("  - High A-Z ratio → Large NC → Positive N (core-dominated)")
print("  - Low A-Z ratio → Small NC → Negative N (envelope-dominated)")
print("  - Balanced A-Z → Minimal NC → N ≈ 0 (ground state)")
print()
print("="*80)
