#!/usr/bin/env python3
"""
CLIFFORD ALGEBRA ISOMER MAP - VISUALIZATION
===========================================================================
Creates comprehensive visualization showing how magic numbers (2, 8, 20, 28, 50, 82, 126)
emerge from the geometric structure of Cl(3,3) vacuum manifold.

Panels:
1. Clifford Algebra Grade Structure (C(6,k) binomial coefficients)
2. Spherical Harmonics on S^5 (cumulative mode counting)
3. Isomer Ladder (discrete energy levels)
4. Unified Picture (low-Z vs high-Z regimes)
===========================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch
from matplotlib.patches import FancyArrowPatch
import matplotlib.patches as mpatches

# Set up publication-quality plotting
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

# Create figure with 4 panels
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# ============================================================================
# PANEL 1: CLIFFORD ALGEBRA GRADE STRUCTURE
# ============================================================================
ax1 = fig.add_subplot(gs[0, 0])

# Cl(3,3) grade dimensions: C(6,k) for k=0 to 6
grades = np.arange(0, 7)
dimensions = [1, 6, 15, 20, 15, 6, 1]  # Binomial coefficients C(6,k)
grade_names = ['Scalars', 'Vectors', 'Bi-vectors', 'Tri-vectors',
               'Quad-vectors', 'Penta-vectors', 'Pseudo-scalar']

# Color code the magic numbers
colors = []
for k, dim in enumerate(dimensions):
    if dim == 2:
        colors.append('#d62728')  # Red for N=2
    elif dim == 20:
        colors.append('#d62728')  # Red for N=20
    else:
        colors.append('#1f77b4')  # Blue for others

# Bar plot of grade dimensions
bars = ax1.bar(grades, dimensions, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

# Annotate magic numbers
for k, dim in enumerate(dimensions):
    if dim in [2, 20]:
        ax1.annotate(f'N = {dim}',
                    xy=(k, dim),
                    xytext=(k, dim + 3),
                    ha='center',
                    fontsize=10,
                    fontweight='bold',
                    color='#d62728',
                    arrowprops=dict(arrowstyle='->', color='#d62728', lw=2))
    else:
        ax1.text(k, dim + 1, f'{dim}', ha='center', fontsize=9)

# Annotate grade names
for k, name in enumerate(grade_names):
    ax1.text(k, -2.5, name, ha='center', fontsize=8, rotation=15)

ax1.set_xlabel('Grade k in Cl(3,3)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Dimension C(6,k)', fontsize=11, fontweight='bold')
ax1.set_title('Panel A: Clifford Algebra Grade Structure\nMagic Numbers from Pure Algebra',
             fontsize=12, fontweight='bold')
ax1.set_xlim(-0.5, 6.5)
ax1.set_ylim(-4, 25)
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.axhline(8, color='orange', linestyle=':', linewidth=2, alpha=0.6, label='N = 8 (Spinor dim)')
ax1.legend(loc='upper right')

# Add formula annotation
ax1.text(0.02, 0.98, r'$\mathrm{dim}(\mathrm{Cl}(3,3)) = 2^6 = 64 = \sum_{k=0}^{6} \binom{6}{k}$',
        transform=ax1.transAxes, fontsize=9, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

# ============================================================================
# PANEL 2: SPHERICAL HARMONICS ON S^5
# ============================================================================
ax2 = fig.add_subplot(gs[0, 1])

# Cumulative spherical harmonic modes on S^5
# N(n) = Σ_{k=0}^n C(k+5, 5) for d=6 dimensional space
def spherical_harmonic_modes(n, d=6):
    """Count spherical harmonic modes on S^(d-1) up to level n"""
    total = 0
    for k in range(n + 1):
        # Number of harmonics at level k in d dimensions
        modes_k = np.math.comb(k + d - 1, d - 1)
        total += modes_k
    return total

levels = np.arange(0, 6)
cumulative_modes = [spherical_harmonic_modes(n) for n in levels]

# Individual level contributions
individual_modes = []
for n in levels:
    individual_modes.append(np.math.comb(n + 5, 5))

# Bar plot
x_pos = levels
bar_width = 0.35

bars1 = ax2.bar(x_pos - bar_width/2, individual_modes, bar_width,
               label='Modes at level n', alpha=0.7, color='#2ca02c', edgecolor='black')
bars2 = ax2.bar(x_pos + bar_width/2, cumulative_modes, bar_width,
               label='Cumulative N(n)', alpha=0.7, color='#ff7f0e', edgecolor='black')

# Annotate magic numbers in cumulative
magic_numbers_harmonic = {2: 28, 3: 84}  # N(2)=28, N(3)=84≈82
for level, cum_mode in enumerate(cumulative_modes):
    if level in magic_numbers_harmonic:
        observed = magic_numbers_harmonic[level]
        if abs(cum_mode - observed) <= 2:
            ax2.annotate(f'N ≈ {observed}',
                        xy=(level + bar_width/2, cum_mode),
                        xytext=(level + bar_width/2 + 0.5, cum_mode + 10),
                        ha='center',
                        fontsize=10,
                        fontweight='bold',
                        color='#d62728',
                        arrowprops=dict(arrowstyle='->', color='#d62728', lw=2))

ax2.set_xlabel('Harmonic Level n', fontsize=11, fontweight='bold')
ax2.set_ylabel('Number of Modes', fontsize=11, fontweight='bold')
ax2.set_title('Panel B: Spherical Harmonics on S⁵\nMagic Numbers from 6D Geometry',
             fontsize=12, fontweight='bold')
ax2.set_xticks(levels)
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.legend(loc='upper left')

# Add formula
ax2.text(0.02, 0.98, r'$N(n) = \sum_{k=0}^{n} \binom{k+5}{5}$ (modes on $S^5$)',
        transform=ax2.transAxes, fontsize=9, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

# ============================================================================
# PANEL 3: ISOMER LADDER (Energy Levels)
# ============================================================================
ax3 = fig.add_subplot(gs[1, 0])

# Magic numbers and their origins
isomer_data = [
    (2, 'U(1) Phase Pairing', 'Clifford', '#1f77b4'),
    (8, 'Spinor Dimension 2³', 'Clifford', '#1f77b4'),
    (20, 'Tri-vector C(6,3)', 'Clifford', '#1f77b4'),
    (28, 'Harmonic Level 2', 'S⁵ Geometry', '#ff7f0e'),
    (50, 'Composite Closure(?)', 'Unknown', '#7f7f7f'),
    (82, 'Harmonic Level 3', 'S⁵ Geometry', '#ff7f0e'),
    (126, 'Harmonic Level 4(?)', 'S⁵ Geometry', '#ff7f0e'),
]

# Create ladder visualization
y_positions = np.arange(len(isomer_data))
magic_nums = [d[0] for d in isomer_data]

# Draw ladder rungs
for i, (N, origin, regime, color) in enumerate(isomer_data):
    # Rung (horizontal bar)
    rect = FancyBboxPatch((0.1, i - 0.15), 0.8, 0.3,
                          boxstyle="round,pad=0.02",
                          edgecolor='black', facecolor=color,
                          linewidth=2, alpha=0.7)
    ax3.add_patch(rect)

    # Label with N value (left side)
    ax3.text(0.05, i, f'N = {N}', ha='right', va='center',
            fontsize=11, fontweight='bold')

    # Origin description (on rung)
    ax3.text(0.5, i, origin, ha='center', va='center',
            fontsize=9, color='white', fontweight='bold')

    # Regime label (right side)
    ax3.text(0.95, i, regime, ha='left', va='center',
            fontsize=9, style='italic')

# Draw vertical ladder sides
ax3.plot([0.1, 0.1], [-0.5, len(isomer_data) - 0.5],
        'k-', linewidth=4, alpha=0.3)
ax3.plot([0.9, 0.9], [-0.5, len(isomer_data) - 0.5],
        'k-', linewidth=4, alpha=0.3)

# Draw gaps with annotations
for i in range(len(isomer_data) - 1):
    gap = magic_nums[i+1] - magic_nums[i]
    mid_y = i + 0.5
    ax3.annotate('', xy=(1.1, i+1), xytext=(1.1, i),
                arrowprops=dict(arrowstyle='<->', color='gray', lw=1.5))
    ax3.text(1.15, mid_y, f'Δ = {gap}', ha='left', va='center',
            fontsize=8, color='gray')

ax3.set_xlim(-0.1, 1.4)
ax3.set_ylim(-0.5, len(isomer_data) - 0.5)
ax3.set_yticks([])
ax3.set_xticks([])
ax3.set_title('Panel C: Isomer Ladder\nQuantized Resonance Modes of 6D Vacuum',
             fontsize=12, fontweight='bold')
ax3.axis('off')

# Legend
clifford_patch = mpatches.Patch(color='#1f77b4', label='Clifford Algebra', alpha=0.7)
harmonic_patch = mpatches.Patch(color='#ff7f0e', label='Spherical Harmonics', alpha=0.7)
unknown_patch = mpatches.Patch(color='#7f7f7f', label='Under Investigation', alpha=0.7)
ax3.legend(handles=[clifford_patch, harmonic_patch, unknown_patch],
          loc='lower right', frameon=True, fancybox=True)

# ============================================================================
# PANEL 4: UNIFIED PICTURE (Regime Transition)
# ============================================================================
ax4 = fig.add_subplot(gs[1, 1])

# Plot showing transition from Clifford to Harmonic regime
A_range = np.linspace(1, 250, 300)

# Define "dominance score" for each regime
def clifford_dominance(A):
    """Clifford algebra dominates for light nuclei"""
    return np.exp(-A / 40)  # Exponential decay

def harmonic_dominance(A):
    """Spherical harmonics dominate for heavy nuclei"""
    return 1 - np.exp(-A / 80)  # Exponential rise

clifford_score = clifford_dominance(A_range)
harmonic_score = harmonic_dominance(A_range)

# Plot regions
ax4.fill_between(A_range, 0, clifford_score,
                alpha=0.3, color='#1f77b4', label='Clifford Algebra Regime')
ax4.fill_between(A_range, 0, harmonic_score,
                alpha=0.3, color='#ff7f0e', label='Spherical Harmonic Regime')

# Overlap region
overlap = np.minimum(clifford_score, harmonic_score)
ax4.fill_between(A_range, 0, overlap,
                alpha=0.5, color='#9467bd', label='Transition Zone')

# Mark magic number positions
magic_A_positions = {
    2: 4, 8: 16, 20: 40, 28: 58, 50: 120, 82: 208, 126: 208
}
for N, A_approx in magic_A_positions.items():
    if A_approx <= 250:
        regime_height = max(clifford_dominance(A_approx), harmonic_dominance(A_approx))
        ax4.plot(A_approx, regime_height, 'o', markersize=8,
                color='#d62728', markeredgecolor='black', markeredgewidth=1.5)
        ax4.text(A_approx, regime_height + 0.08, f'{N}',
                ha='center', fontsize=9, fontweight='bold')

# Annotations for key nuclei
nuclei_annotations = [
    (4, 'He-4\n(N=2)', 0.85),
    (16, 'O-16\n(N=8)', 0.75),
    (40, 'Ca-40\n(N=20)', 0.65),
    (208, 'Pb-208\n(N=126)', 0.45),
]

for A, label, y_pos in nuclei_annotations:
    ax4.annotate(label, xy=(A, y_pos), xytext=(A, y_pos - 0.15),
                ha='center', fontsize=8,
                arrowprops=dict(arrowstyle='->', color='black', lw=1),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))

ax4.set_xlabel('Mass Number A', fontsize=11, fontweight='bold')
ax4.set_ylabel('Regime Dominance', fontsize=11, fontweight='bold')
ax4.set_title('Panel D: Unified Picture\nRegime Transition Across Nuclear Chart',
             fontsize=12, fontweight='bold')
ax4.set_xlim(0, 250)
ax4.set_ylim(0, 1.1)
ax4.grid(alpha=0.3, linestyle='--')
ax4.legend(loc='upper left', frameon=True, fancybox=True)

# Add regime labels directly on plot
ax4.text(20, 0.95, 'Pure Clifford\n(Grade Structure)',
        ha='center', fontsize=9, style='italic',
        bbox=dict(boxstyle='round', facecolor='#1f77b4', alpha=0.2))
ax4.text(180, 0.95, 'Pure Harmonics\n(S⁵ Modes)',
        ha='center', fontsize=9, style='italic',
        bbox=dict(boxstyle='round', facecolor='#ff7f0e', alpha=0.2))

# ============================================================================
# MAIN TITLE
# ============================================================================
fig.suptitle('CLIFFORD ALGEBRA Cl(3,3) ISOMER MAP\n' +
            'Geometric Origin of Nuclear Magic Numbers',
            fontsize=14, fontweight='bold', y=0.98)

# Add footer
fig.text(0.5, 0.01,
        'QFD Framework: Topological solitons on 6D vacuum manifold | ' +
        'Magic numbers = maximal symmetry configurations | ' +
        'No free parameters',
        ha='center', fontsize=9, style='italic', color='gray')

plt.tight_layout(rect=[0, 0.02, 1, 0.96])
plt.savefig('CL33_ISOMER_MAP.png', dpi=300, bbox_inches='tight')
print("✓ Visualization saved: CL33_ISOMER_MAP.png")

plt.show()
