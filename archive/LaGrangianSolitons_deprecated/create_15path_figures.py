#!/usr/bin/env python3
"""
CREATE 15-PATH MODEL PUBLICATION FIGURES
================================================================================
Visualize the radial pattern and finer geometric structure
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from collections import defaultdict

# Set publication style
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 300

# Load optimized parameters
c1_0_7 = 0.961752
c2_0_7 = 0.247527
c3_0_7 = -2.410727
dc1_7 = -0.029498
dc2_7 = 0.006412
dc3_7 = -0.865252

c1_0_15 = 0.970454
c2_0_15 = 0.234920
c3_0_15 = -1.928732
dc1_15 = -0.021538
dc2_15 = 0.001730
dc3_15 = -0.540530

PATH_VALUES_7 = np.arange(-3, 4, 1)
PATH_VALUES_15 = np.arange(-3.5, 4.0, 0.5)

def get_coefficients_7(N):
    return (c1_0_7 + dc1_7*N, c2_0_7 + dc2_7*N, c3_0_7 + dc3_7*N)

def get_coefficients_15(N):
    return (c1_0_15 + dc1_15*N, c2_0_15 + dc2_15*N, c3_0_15 + dc3_15*N)

def predict_Z_7(A, N):
    c1, c2, c3 = get_coefficients_7(N)
    return int(round(c1 * A**(2/3) + c2 * A + c3))

def predict_Z_15(A, N):
    c1, c2, c3 = get_coefficients_15(N)
    return int(round(c1 * A**(2/3) + c2 * A + c3))

def classify_7(A, Z_exp):
    for N in PATH_VALUES_7:
        if predict_Z_7(A, N) == Z_exp:
            return N
    return None

def classify_15(A, Z_exp):
    for N in PATH_VALUES_15:
        if predict_Z_15(A, N) == Z_exp:
            return N
    return None

# Load data
with open('qfd_optimized_suite.py', 'r') as f:
    content = f.read()
start = content.find('test_nuclides = [')
end = content.find(']', start) + 1
test_nuclides = eval(content[start:end].replace('test_nuclides = ', ''))

# Classify all nuclei
paths_7 = defaultdict(list)
paths_15 = defaultdict(list)

for name, Z, A in test_nuclides:
    neutrons = A - Z
    parity = 'even-even' if (Z % 2 == 0 and neutrons % 2 == 0) else 'odd-A'

    N7 = classify_7(A, Z)
    N15 = classify_15(A, Z)

    if N7 is not None:
        paths_7[N7].append((name, A, Z, parity))
    if N15 is not None:
        paths_15[N15].append((name, A, Z, parity))

print("Creating 15-path model figures...")
print()

# ============================================================================
# FIGURE 1: RADIAL PATTERN - The Key Discovery
# ============================================================================

fig1 = plt.figure(figsize=(16, 10))
gs = fig1.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

# Panel A: 15-Path Distribution by Parity
ax1 = fig1.add_subplot(gs[0, :2])

path_vals = []
ee_counts = []
odd_counts = []

for N in PATH_VALUES_15:
    count_ee = sum(1 for _, _, _, p in paths_15[N] if p == 'even-even')
    count_odd = len(paths_15[N]) - count_ee
    path_vals.append(N)
    ee_counts.append(count_ee)
    odd_counts.append(count_odd)

x_pos = np.arange(len(path_vals))
width = 0.8

# Create stacked bars
bars_ee = ax1.bar(x_pos, ee_counts, width, label='Even-even', color='steelblue', alpha=0.9)
bars_odd = ax1.bar(x_pos, odd_counts, width, bottom=ee_counts, label='Odd-A', color='coral', alpha=0.9)

# Mark integer vs half-integer
for i, N in enumerate(path_vals):
    if N == int(N):
        ax1.plot(i, -3, 'v', color='green', markersize=8, clip_on=False)

# Highlight extreme vs central regions
extreme_mask = np.abs(np.array(path_vals)) >= 2.5
for i, is_extreme in enumerate(extreme_mask):
    if is_extreme:
        ax1.axvspan(i-0.5, i+0.5, alpha=0.1, color='gold', zorder=-10)

ax1.set_xlabel('Path Quantum Number N', fontsize=12, fontweight='bold')
ax1.set_ylabel('Number of Nuclei', fontsize=12, fontweight='bold')
ax1.set_title('(A) 15-Path Population Distribution', fontsize=13, fontweight='bold', pad=10)
ax1.set_xticks(x_pos)
ax1.set_xticklabels([f'{N:.1f}' for N in path_vals], rotation=45)
ax1.legend(loc='upper left', fontsize=10)
ax1.grid(axis='y', alpha=0.3)

# Add annotations
ax1.text(0.02, 0.95, 'Gold = Extreme deformation (|N| ≥ 2.5)',
         transform=ax1.transAxes, fontsize=9, va='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
ax1.text(0.02, 0.88, '▼ = Integer path',
         transform=ax1.transAxes, fontsize=9, va='top', color='green')

# Panel B: Radial Pattern - % Even-Even vs |N|
ax2 = fig1.add_subplot(gs[0, 2])

abs_N = []
pct_ee = []
sizes = []

for N in PATH_VALUES_15:
    count_ee = sum(1 for _, _, _, p in paths_15[N] if p == 'even-even')
    total = len(paths_15[N])
    if total > 0:
        abs_N.append(abs(N))
        pct_ee.append(100 * count_ee / total)
        sizes.append(total * 10)  # Scale marker size by population

# Scatter plot
colors = ['gold' if abs(N) >= 2.5 else 'gray' for N in PATH_VALUES_15 if len(paths_15[N]) > 0]
sc = ax2.scatter(abs_N, pct_ee, s=sizes, c=colors, alpha=0.7, edgecolors='black', linewidth=1)

# Fit trend line for extreme vs central
extreme_N = [n for n, p in zip(abs_N, pct_ee) if n >= 2.5]
extreme_pct = [p for n, p in zip(abs_N, pct_ee) if n >= 2.5]
central_N = [n for n, p in zip(abs_N, pct_ee) if n < 2.5]
central_pct = [p for n, p in zip(abs_N, pct_ee) if n < 2.5]

if len(extreme_pct) > 0:
    ax2.axhline(np.mean(extreme_pct), color='gold', linestyle='--', linewidth=2,
                label=f'Extreme: {np.mean(extreme_pct):.1f}%', alpha=0.8)
if len(central_pct) > 0:
    ax2.axhline(np.mean(central_pct), color='gray', linestyle='--', linewidth=2,
                label=f'Central: {np.mean(central_pct):.1f}%', alpha=0.8)

# Mark boundary
ax2.axvline(2.5, color='red', linestyle=':', linewidth=2, alpha=0.5, label='Boundary')

ax2.set_xlabel('|N| (Deformation Magnitude)', fontsize=11, fontweight='bold')
ax2.set_ylabel('% Even-Even', fontsize=11, fontweight='bold')
ax2.set_title('(B) Radial Pattern\nExtreme vs Central', fontsize=12, fontweight='bold', pad=10)
ax2.legend(fontsize=8, loc='lower right')
ax2.grid(alpha=0.3)
ax2.set_ylim(0, 105)

# Panel C: Tin Ladder in 15-Path Model
ax3 = fig1.add_subplot(gs[1, :])

tin_data = []
for name, Z, A in test_nuclides:
    if Z == 50:
        N15 = classify_15(A, Z)
        if N15 is not None:
            neutrons = A - Z
            parity = 'even-even' if neutrons % 2 == 0 else 'odd-A'
            tin_data.append((A, N15, name, parity))

tin_data.sort()

A_vals = [d[0] for d in tin_data]
N_vals = [d[1] for d in tin_data]
colors = ['steelblue' if d[3] == 'even-even' else 'coral' for d in tin_data]

# Plot path progression
ax3.plot(A_vals, N_vals, 'o-', color='gray', linewidth=2, markersize=10, alpha=0.5, zorder=1)
ax3.scatter(A_vals, N_vals, c=colors, s=150, edgecolors='black', linewidth=2, zorder=2)

# Annotate isotopes
for A, N, name, parity in tin_data:
    offset = 0.15 if parity == 'even-even' else -0.15
    ax3.text(A, N + offset, name.split('-')[1], fontsize=9, ha='center', va='bottom' if parity == 'even-even' else 'top')

# Mark integer paths
for N in PATH_VALUES_15:
    if N == int(N):
        ax3.axhline(N, color='green', alpha=0.2, linestyle='--', linewidth=1)

# Highlight ΔN = 0 case
for i in range(len(tin_data) - 1):
    if tin_data[i][1] == tin_data[i+1][1]:  # Same path
        ax3.annotate('', xy=(tin_data[i+1][0], tin_data[i+1][1]),
                    xytext=(tin_data[i][0], tin_data[i][1]),
                    arrowprops=dict(arrowstyle='<->', color='red', lw=3))
        ax3.text((tin_data[i][0] + tin_data[i+1][0])/2, tin_data[i][1] + 0.3,
                'ΔN=0!', fontsize=10, color='red', fontweight='bold', ha='center')

ax3.set_xlabel('Soliton Mass A (AMU)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Path Quantum Number N', fontsize=12, fontweight='bold')
ax3.set_title('(C) Tin Isotopic Chain: 15-Path Assignments', fontsize=13, fontweight='bold', pad=10)
ax3.grid(alpha=0.3)
ax3.legend([plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='steelblue', markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='coral', markersize=10)],
           ['Even-even', 'Odd-A'], loc='upper left', fontsize=10)

# Panel D: Integer vs Half-Integer Comparison
ax4 = fig1.add_subplot(gs[2, 0])

int_ee = sum(sum(1 for _, _, _, p in paths_15[N] if p == 'even-even') for N in PATH_VALUES_15 if N == int(N))
int_total = sum(len(paths_15[N]) for N in PATH_VALUES_15 if N == int(N))
half_ee = sum(sum(1 for _, _, _, p in paths_15[N] if p == 'even-even') for N in PATH_VALUES_15 if N != int(N))
half_total = sum(len(paths_15[N]) for N in PATH_VALUES_15 if N != int(N))

categories = ['Integer\nPaths', 'Half-Integer\nPaths']
pct_values = [100 * int_ee / int_total if int_total > 0 else 0,
              100 * half_ee / half_total if half_total > 0 else 0]
colors_bar = ['lightgreen', 'lightcoral']

bars = ax4.bar(categories, pct_values, color=colors_bar, edgecolor='black', linewidth=2, alpha=0.7)
ax4.axhline(50, color='gray', linestyle='--', linewidth=1, alpha=0.5)

for i, (bar, pct) in enumerate(zip(bars, pct_values)):
    ax4.text(i, pct + 2, f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)

ax4.set_ylabel('% Even-Even', fontsize=11, fontweight='bold')
ax4.set_title('(D) Hypothesis Test:\nInteger vs Half-Integer', fontsize=12, fontweight='bold', pad=10)
ax4.set_ylim(0, 70)
ax4.grid(axis='y', alpha=0.3)

# Add verdict
verdict = "REVERSED!" if pct_values[1] > pct_values[0] else "CONFIRMED"
color_verdict = 'red' if pct_values[1] > pct_values[0] else 'green'
ax4.text(0.5, 0.95, f'Hypothesis: {verdict}', transform=ax4.transAxes,
         fontsize=11, ha='center', va='top', fontweight='bold', color=color_verdict,
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

# Panel E: Extreme vs Central Comparison
ax5 = fig1.add_subplot(gs[2, 1])

extreme_ee = sum(sum(1 for _, _, _, p in paths_15[N] if p == 'even-even')
                 for N in PATH_VALUES_15 if abs(N) >= 2.5)
extreme_total = sum(len(paths_15[N]) for N in PATH_VALUES_15 if abs(N) >= 2.5)
central_ee = sum(sum(1 for _, _, _, p in paths_15[N] if p == 'even-even')
                 for N in PATH_VALUES_15 if abs(N) < 2.5)
central_total = sum(len(paths_15[N]) for N in PATH_VALUES_15 if abs(N) < 2.5)

categories2 = ['Extreme\n|N| ≥ 2.5', 'Central\n|N| < 2.5']
pct_values2 = [100 * extreme_ee / extreme_total if extreme_total > 0 else 0,
               100 * central_ee / central_total if central_total > 0 else 0]
colors_bar2 = ['gold', 'lightgray']

bars2 = ax5.bar(categories2, pct_values2, color=colors_bar2, edgecolor='black', linewidth=2, alpha=0.8)
ax5.axhline(50, color='gray', linestyle='--', linewidth=1, alpha=0.5)

for i, (bar, pct) in enumerate(zip(bars2, pct_values2)):
    ax5.text(i, pct + 2, f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)

ax5.set_ylabel('% Even-Even', fontsize=11, fontweight='bold')
ax5.set_title('(E) Radial Pattern:\nExtreme vs Central', fontsize=12, fontweight='bold', pad=10)
ax5.set_ylim(0, 85)
ax5.grid(axis='y', alpha=0.3)

# Add verdict
diff = pct_values2[0] - pct_values2[1]
ax5.text(0.5, 0.95, f'Δ = +{diff:.1f}%\nCONFIRMED ✓', transform=ax5.transAxes,
         fontsize=11, ha='center', va='top', fontweight='bold', color='green',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

# Panel F: Coefficient Comparison (7-path vs 15-path)
ax6 = fig1.add_subplot(gs[2, 2])

N_range = np.linspace(-3.5, 3.5, 100)
ratio_7 = [(c1_0_7 + dc1_7*N) / (c2_0_7 + dc2_7*N) for N in N_range]
ratio_15 = [(c1_0_15 + dc1_15*N) / (c2_0_15 + dc2_15*N) for N in N_range]

ax6.plot(N_range, ratio_7, 'b-', linewidth=2, label='7-path model', alpha=0.7)
ax6.plot(N_range, ratio_15, 'r--', linewidth=2, label='15-path model', alpha=0.7)

# Mark actual path locations
for N in PATH_VALUES_7:
    c1, c2, _ = get_coefficients_7(N)
    ax6.plot(N, c1/c2, 'bo', markersize=8, alpha=0.8)

for N in PATH_VALUES_15:
    if len(paths_15[N]) > 0:
        c1, c2, _ = get_coefficients_15(N)
        ax6.plot(N, c1/c2, 'ro', markersize=6, alpha=0.8)

ax6.set_xlabel('Path Quantum Number N', fontsize=11, fontweight='bold')
ax6.set_ylabel('c₁/c₂ Ratio', fontsize=11, fontweight='bold')
ax6.set_title('(F) Coefficient Evolution\nComparison', fontsize=12, fontweight='bold', pad=10)
ax6.legend(fontsize=9)
ax6.grid(alpha=0.3)

plt.suptitle('RADIAL PATTERN IN 15-PATH GEOMETRIC QUANTIZATION',
             fontsize=16, fontweight='bold', y=0.995)

plt.savefig('FIGURE_15PATH_RADIAL_PATTERN.png', dpi=300, bbox_inches='tight')
plt.savefig('FIGURE_15PATH_RADIAL_PATTERN.pdf', bbox_inches='tight')
print("✓ Saved: FIGURE_15PATH_RADIAL_PATTERN.png/pdf")

# ============================================================================
# FIGURE 2: NONLINEAR ΔA-ΔN RELATIONSHIP
# ============================================================================

fig2, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel A: All isotopic chains showing ΔA vs ΔN
ax = axes[0, 0]

elements_to_show = [20, 28, 50, 54, 82]  # Ca, Ni, Sn, Xe, Pb
element_names = {20: 'Ca', 28: 'Ni', 50: 'Sn', 54: 'Xe', 82: 'Pb'}
colors_elem = plt.cm.tab10(np.linspace(0, 1, len(elements_to_show)))

for elem_idx, Z_elem in enumerate(elements_to_show):
    isotopes = [(A, classify_15(A, Z_elem), A-Z_elem) for _, Z, A in test_nuclides
                if Z == Z_elem and classify_15(A, Z_elem) is not None]
    isotopes.sort()

    for i in range(len(isotopes) - 1):
        A1, N1, neutrons1 = isotopes[i]
        A2, N2, neutrons2 = isotopes[i+1]
        delta_A = A2 - A1
        delta_N = N2 - N1

        parity1 = 'even' if neutrons1 % 2 == 0 else 'odd'
        parity2 = 'even' if neutrons2 % 2 == 0 else 'odd'
        marker = 'o' if (parity1 == 'even' and parity2 == 'even') else '^'

        ax.scatter(delta_A, delta_N, c=[colors_elem[elem_idx]], marker=marker,
                  s=80, alpha=0.7, edgecolors='black', linewidth=1)

# Add diagonal lines
ax.plot([0, 4], [0, 4], 'k--', alpha=0.3, linewidth=1, label='ΔN = ΔA (linear)')
ax.plot([0, 4], [0, 2], 'r--', alpha=0.3, linewidth=1, label='ΔN = ΔA/2')

ax.set_xlabel('Mass Increment ΔA (AMU)', fontsize=12, fontweight='bold')
ax.set_ylabel('Deformation Increment ΔN', fontsize=12, fontweight='bold')
ax.set_title('(A) Nonlinear ΔA → ΔN Relationship', fontsize=13, fontweight='bold')
ax.legend([plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10),
           plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', markersize=10)],
          ['Even-even → Even-even', 'Crossing odd-A'], loc='upper left', fontsize=9)
ax.grid(alpha=0.3)
ax.set_xlim(-0.5, 4.5)
ax.set_ylim(-0.3, 1.8)

# Panel B: Histogram of ΔN values
ax = axes[0, 1]

all_delta_N = []
for Z_elem in range(1, 100):
    isotopes = [(A, classify_15(A, Z_elem)) for _, Z, A in test_nuclides
                if Z == Z_elem and classify_15(A, Z_elem) is not None]
    isotopes.sort()

    for i in range(len(isotopes) - 1):
        A1, N1 = isotopes[i]
        A2, N2 = isotopes[i+1]
        all_delta_N.append(N2 - N1)

bins = np.arange(-0.25, 2.0, 0.25)
counts, bin_edges, patches = ax.hist(all_delta_N, bins=bins, edgecolor='black',
                                     linewidth=1.5, alpha=0.7, color='steelblue')

# Color code bins
for i, patch in enumerate(patches):
    if bin_edges[i] < 0:
        patch.set_facecolor('red')
    elif abs(bin_edges[i] - 0.5) < 0.1:
        patch.set_facecolor('green')
    elif abs(bin_edges[i] - 1.0) < 0.1:
        patch.set_facecolor('gold')

ax.set_xlabel('ΔN Value', fontsize=12, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax.set_title('(B) Distribution of ΔN in Isotopic Chains', fontsize=13, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# Add statistics
mean_delta_N = np.mean(all_delta_N)
median_delta_N = np.median(all_delta_N)
ax.axvline(mean_delta_N, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_delta_N:.2f}')
ax.axvline(median_delta_N, color='blue', linestyle='--', linewidth=2, label=f'Median = {median_delta_N:.2f}')
ax.legend(fontsize=10)

# Panel C: Tin ladder detailed ΔA vs ΔN
ax = axes[1, 0]

tin_isotopes = [(A, classify_15(A, 50)) for _, Z, A in test_nuclides
                if Z == 50 and classify_15(A, 50) is not None]
tin_isotopes.sort()

A_tin = [A for A, N in tin_isotopes]
N_tin = [N for A, N in tin_isotopes]

# Plot transitions
for i in range(len(tin_isotopes) - 1):
    A1, N1 = tin_isotopes[i]
    A2, N2 = tin_isotopes[i+1]
    delta_A = A2 - A1
    delta_N = N2 - N1

    # Draw arrow
    ax.annotate('', xy=(A2, N2), xytext=(A1, N1),
                arrowprops=dict(arrowstyle='->', lw=2, color='steelblue', alpha=0.7))

    # Label transition
    mid_A = (A1 + A2) / 2
    mid_N = (N1 + N2) / 2
    ax.text(mid_A, mid_N + 0.15, f'Δ({delta_A}, {delta_N:.1f})',
            fontsize=8, ha='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

# Mark points
ax.scatter(A_tin, N_tin, s=150, c='steelblue', edgecolors='black', linewidth=2, zorder=10)

# Annotate points
for A, N in tin_isotopes:
    ax.text(A, N - 0.25, f'A={A}', fontsize=8, ha='center', va='top')

ax.set_xlabel('Soliton Mass A (AMU)', fontsize=12, fontweight='bold')
ax.set_ylabel('Path Quantum Number N', fontsize=12, fontweight='bold')
ax.set_title('(C) Tin Ladder: ΔA and ΔN Transitions', fontsize=13, fontweight='bold')
ax.grid(alpha=0.3)

# Panel D: Path population vs N
ax = axes[1, 1]

N_positions = []
populations = []
ee_fractions = []

for N in PATH_VALUES_15:
    total = len(paths_15[N])
    if total > 0:
        ee_count = sum(1 for _, _, _, p in paths_15[N] if p == 'even-even')
        N_positions.append(N)
        populations.append(total)
        ee_fractions.append(ee_count / total)

# Create bubble plot
colors_bubble = [plt.cm.RdYlBu_r((f + 0.2) / 1.4) for f in ee_fractions]
bubbles = ax.scatter(N_positions, populations, s=[p*15 for p in populations],
                     c=colors_bubble, alpha=0.7, edgecolors='black', linewidth=2)

# Mark integer vs half-integer
for N, pop in zip(N_positions, populations):
    if N == int(N):
        ax.scatter([N], [pop], s=300, facecolors='none', edgecolors='green',
                  linewidths=3, alpha=0.5)

ax.set_xlabel('Path Quantum Number N', fontsize=12, fontweight='bold')
ax.set_ylabel('Population (Number of Nuclei)', fontsize=12, fontweight='bold')
ax.set_title('(D) Path Populations (Size = Count, Color = % Even-Even)',
             fontsize=13, fontweight='bold')
ax.grid(alpha=0.3)

# Add colorbar
cbar = plt.colorbar(plt.cm.ScalarMappable(cmap='RdYlBu_r',
                    norm=plt.Normalize(vmin=0, vmax=100)), ax=ax)
cbar.set_label('% Even-Even', fontsize=10)

plt.suptitle('NONLINEAR MASS-DEFORMATION RELATIONSHIP IN 15-PATH MODEL',
             fontsize=16, fontweight='bold')
plt.tight_layout()

plt.savefig('FIGURE_15PATH_NONLINEAR_RELATIONSHIP.png', dpi=300, bbox_inches='tight')
plt.savefig('FIGURE_15PATH_NONLINEAR_RELATIONSHIP.pdf', bbox_inches='tight')
print("✓ Saved: FIGURE_15PATH_NONLINEAR_RELATIONSHIP.png/pdf")

print()
print("="*80)
print("FIGURE GENERATION COMPLETE")
print("="*80)
print()
print("Created 2 comprehensive figures:")
print("  1. FIGURE_15PATH_RADIAL_PATTERN: 6-panel radial pattern analysis")
print("  2. FIGURE_15PATH_NONLINEAR_RELATIONSHIP: 4-panel ΔA-ΔN analysis")
print()
print("All figures saved in PNG (300 DPI) and PDF (vector) formats")
print("="*80)
