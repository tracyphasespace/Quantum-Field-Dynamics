#!/usr/bin/env python3
"""
HALF-LIFE vs STRESS: DECAY-MODE-SPECIFIC ANALYSIS
================================================================================
Test correlations separately for each decay mode:
- Beta-minus (β⁻) decay
- Beta-plus/EC (β⁺) decay
- Alpha (α) decay
- Neutron emission

Also analyze daughter nucleus positions on stress manifold.
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from scipy.optimize import curve_fit
from collections import defaultdict

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

def halflife_to_seconds(value, unit):
    """Convert half-life to seconds"""
    conversions = {
        's': 1.0,
        'ms': 1e-3,
        'us': 1e-6,
        'ns': 1e-9,
        'ps': 1e-12,
        'fs': 1e-15,
        'as': 1e-18,
        'm': 60.0,
        'h': 3600.0,
        'd': 86400.0,
        'y': 31557600.0,  # Julian year
    }
    return value * conversions.get(unit, 1.0)

def get_daughter_nucleus(Z, A, decay_mode):
    """Calculate daughter nucleus (Z', A') after decay"""
    if 'β-' in decay_mode or 'beta-' in decay_mode:
        # β⁻: n → p + e⁻ + ν̄ (Z increases, A constant)
        return Z + 1, A
    elif 'β+' in decay_mode or 'EC' in decay_mode or 'beta+' in decay_mode:
        # β⁺/EC: p → n + e⁺ + ν (Z decreases, A constant)
        return Z - 1, A
    elif 'α' in decay_mode or 'alpha' in decay_mode:
        # α: Emit ⁴He (Z decreases by 2, A decreases by 4)
        return Z - 2, A - 4
    elif 'n' in decay_mode and 'neutron' in decay_mode.lower():
        # Neutron emission: A decreases by 1, Z constant
        return Z, A - 1
    elif 'p' in decay_mode and 'proton' in decay_mode.lower():
        # Proton emission: Z decreases by 1, A decreases by 1
        return Z - 1, A - 1
    else:
        # Unknown decay mode
        return None, None

# Expanded radioactive isotope database with detailed decay modes
radioactive_database = [
    # Format: (name, Z, A, half_life_value, half_life_unit, decay_mode, Q_value_MeV)

    # Beta-minus decays
    ("H-3", 1, 3, 12.32, 'y', 'β-', 0.0186),
    ("C-14", 6, 14, 5730, 'y', 'β-', 0.156),
    ("Na-22", 11, 22, 2.602, 'y', 'β+/EC', 2.842),
    ("P-32", 15, 32, 14.29, 'd', 'β-', 1.709),
    ("S-35", 16, 35, 87.51, 'd', 'β-', 0.167),
    ("K-40", 19, 40, 1.248e9, 'y', 'β-/EC', 1.311),
    ("Ca-45", 20, 45, 162.7, 'd', 'β-', 0.257),
    ("Fe-55", 26, 55, 2.737, 'y', 'EC', 0.231),
    ("Co-60", 27, 60, 5.271, 'y', 'β-', 2.824),
    ("Sr-90", 38, 90, 28.79, 'y', 'β-', 0.546),
    ("I-131", 53, 131, 8.021, 'd', 'β-', 0.971),
    ("Cs-137", 55, 137, 30.08, 'y', 'β-', 1.176),
    ("Pm-147", 61, 147, 2.623, 'y', 'β-', 0.224),

    # Alpha decays
    ("Po-210", 84, 210, 138.4, 'd', 'α', 5.407),
    ("Ra-226", 88, 226, 1600, 'y', 'α', 4.871),
    ("Th-230", 90, 230, 7.538e4, 'y', 'α', 4.770),
    ("Th-232", 90, 232, 1.405e10, 'y', 'α', 4.081),
    ("U-233", 92, 233, 1.592e5, 'y', 'α', 4.909),
    ("U-234", 92, 234, 2.455e5, 'y', 'α', 4.857),
    ("U-235", 92, 235, 7.04e8, 'y', 'α', 4.679),
    ("U-238", 92, 238, 4.468e9, 'y', 'α', 4.270),
    ("Np-237", 93, 237, 2.144e6, 'y', 'α', 4.957),
    ("Pu-238", 94, 238, 87.7, 'y', 'α', 5.593),
    ("Pu-239", 94, 239, 2.411e4, 'y', 'α', 5.244),
    ("Pu-240", 94, 240, 6564, 'y', 'α', 5.256),
    ("Am-241", 95, 241, 432.2, 'y', 'α', 5.638),

    # Very short-lived (for range)
    ("Be-8", 4, 8, 8.19e-17, 's', 'α', 0.092),
    ("Li-5", 3, 5, 3.7e-22, 's', 'p', 1.97),

    # Neutron emitters
    ("He-5", 2, 5, 7.0e-22, 's', 'n', 0.89),
    ("He-10", 2, 10, 2.7e-22, 's', 'n', 1.1),

    # Additional isotopes with mixed modes
    ("K-40", 19, 40, 1.248e9, 'y', 'β-/β+/EC', 1.311),
    ("Bi-212", 83, 212, 60.55, 'm', 'α/β-', 2.252),
    ("At-211", 85, 211, 7.214, 'h', 'α/EC', 5.982),

    # More beta decays for statistics
    ("Ni-63", 28, 63, 100.1, 'y', 'β-', 0.067),
    ("Kr-85", 36, 85, 10.756, 'y', 'β-', 0.687),
    ("Tc-99", 43, 99, 2.111e5, 'y', 'β-', 0.294),
    ("Ru-106", 44, 106, 373.59, 'd', 'β-', 0.039),
    ("Sb-125", 51, 125, 2.758, 'y', 'β-', 0.767),
    ("Te-127", 52, 127, 9.35, 'h', 'β-', 0.698),
    ("Xe-133", 54, 133, 5.243, 'd', 'β-', 0.427),
    ("Ba-140", 56, 140, 12.752, 'd', 'β-', 1.035),
    ("Ce-144", 58, 144, 284.91, 'd', 'β-', 0.319),

    # More alpha decays
    ("Sm-147", 62, 147, 1.06e11, 'y', 'α', 2.310),
    ("Gd-152", 64, 152, 1.08e14, 'y', 'α', 2.203),
    ("Hf-174", 72, 174, 2.0e15, 'y', 'α', 2.497),
    ("Po-218", 84, 218, 3.10, 'm', 'α', 6.115),
    ("Rn-222", 86, 222, 3.8235, 'd', 'α', 5.590),
    ("Cm-244", 96, 244, 18.10, 'y', 'α', 5.902),
]

print("="*80)
print("HALF-LIFE vs STRESS: DECAY-MODE-SPECIFIC ANALYSIS")
print("="*80)
print()

# Organize data by decay mode
data_by_mode = defaultdict(list)

for name, Z, A, t_half_val, t_half_unit, decay_mode, Q_value in radioactive_database:
    # Enforce integer quantization
    assert isinstance(Z, int) and isinstance(A, int), f"{name}: Z and A must be integers!"

    # Calculate stress
    N_coord = calculate_N_continuous(A, Z)
    stress = abs(N_coord)

    # Convert half-life to seconds
    t_half_seconds = halflife_to_seconds(t_half_val, t_half_unit)
    log_t_half = np.log10(t_half_seconds)

    # Calculate daughter nucleus
    Z_daughter, A_daughter = get_daughter_nucleus(Z, A, decay_mode)

    if Z_daughter is not None and A_daughter is not None and A_daughter > 0:
        N_daughter = calculate_N_continuous(A_daughter, Z_daughter)
        stress_daughter = abs(N_daughter)
        delta_stress = stress_daughter - stress  # Stress change after decay
    else:
        N_daughter = np.nan
        stress_daughter = np.nan
        delta_stress = np.nan

    # Classify decay mode
    primary_mode = 'unknown'
    if 'β-' in decay_mode or 'beta-' in decay_mode:
        primary_mode = 'β⁻'
    elif 'β+' in decay_mode or 'EC' in decay_mode:
        primary_mode = 'β⁺/EC'
    elif 'α' in decay_mode or 'alpha' in decay_mode:
        primary_mode = 'α'
    elif 'n' in decay_mode and ('neutron' in decay_mode.lower() or len(decay_mode) <= 2):
        primary_mode = 'neutron'
    elif 'p' in decay_mode and 'proton' in decay_mode.lower():
        primary_mode = 'proton'

    data_by_mode[primary_mode].append({
        'name': name,
        'Z': Z,
        'A': A,
        'N_coord': N_coord,
        'stress': stress,
        'log_t_half': log_t_half,
        't_half_seconds': t_half_seconds,
        'Q_value': Q_value,
        'Z_daughter': Z_daughter,
        'A_daughter': A_daughter,
        'N_daughter': N_daughter,
        'stress_daughter': stress_daughter,
        'delta_stress': delta_stress,
    })

# Print summary statistics
print(f"Total isotopes analyzed: {len(radioactive_database)}")
print()
print("Breakdown by decay mode:")
for mode, data in sorted(data_by_mode.items(), key=lambda x: -len(x[1])):
    print(f"  {mode:12s}: {len(data):3d} isotopes")
print()

# ============================================================================
# CORRELATION ANALYSIS BY DECAY MODE
# ============================================================================

print("="*80)
print("CORRELATION ANALYSIS BY DECAY MODE")
print("="*80)
print()

results = {}

for mode, data in sorted(data_by_mode.items()):
    if len(data) < 3:
        print(f"{mode}: Too few isotopes (n={len(data)}) for correlation")
        print()
        continue

    stress_vals = np.array([d['stress'] for d in data])
    log_t_half_vals = np.array([d['log_t_half'] for d in data])
    Q_vals = np.array([d['Q_value'] for d in data])

    print(f"{mode} DECAY (n={len(data)})")
    print("-"*80)

    # Correlation: log(t_1/2) vs stress
    if len(stress_vals) > 2:
        r_linear, p_linear = pearsonr(stress_vals, log_t_half_vals)
        r_quad, p_quad = pearsonr(stress_vals**2, log_t_half_vals)
        rho, p_spearman = spearmanr(stress_vals, log_t_half_vals)

        print(f"  log(t_1/2) vs σ:   r = {r_linear:+.3f}, p = {p_linear:.4f}", end='')
        if p_linear < 0.05:
            print(" ★ SIGNIFICANT")
        else:
            print()

        print(f"  log(t_1/2) vs σ²:  r = {r_quad:+.3f}, p = {p_quad:.4f}", end='')
        if p_quad < 0.05:
            print(" ★ SIGNIFICANT")
        else:
            print()

        print(f"  Spearman ρ:        ρ = {rho:+.3f}, p = {p_spearman:.4f}", end='')
        if p_spearman < 0.05:
            print(" ★ SIGNIFICANT")
        else:
            print()

    # Correlation: log(t_1/2) vs Q-value
    if len(Q_vals) > 2 and not np.any(np.isnan(Q_vals)):
        r_Q, p_Q = pearsonr(Q_vals, log_t_half_vals)
        print(f"  log(t_1/2) vs Q:   r = {r_Q:+.3f}, p = {p_Q:.4f}", end='')
        if p_Q < 0.05:
            print(" ★ SIGNIFICANT")
        else:
            print()

    # Daughter stress analysis
    delta_stress_vals = np.array([d['delta_stress'] for d in data if not np.isnan(d['delta_stress'])])
    if len(delta_stress_vals) > 0:
        mean_delta = np.mean(delta_stress_vals)
        print(f"  Δσ (daughter):     {mean_delta:+.3f} (mean change)", end='')
        if mean_delta < 0:
            print(" → LOWER stress ✓")
        else:
            print(" → HIGHER stress?")

    print()

    results[mode] = {
        'n': len(data),
        'stress': stress_vals,
        'log_t_half': log_t_half_vals,
        'Q': Q_vals,
        'r_linear': r_linear if len(stress_vals) > 2 else np.nan,
        'p_linear': p_linear if len(stress_vals) > 2 else np.nan,
        'r_quad': r_quad if len(stress_vals) > 2 else np.nan,
        'p_quad': p_quad if len(stress_vals) > 2 else np.nan,
        'delta_stress': delta_stress_vals,
    }

# ============================================================================
# CREATE COMPREHENSIVE FIGURE
# ============================================================================

fig = plt.figure(figsize=(24, 18))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

mode_colors = {
    'β⁻': 'red',
    'β⁺/EC': 'orange',
    'α': 'blue',
    'neutron': 'green',
    'proton': 'purple',
    'unknown': 'gray',
}

# Panel 1: β⁻ decay
if 'β⁻' in results:
    ax1 = fig.add_subplot(gs[0, 0])
    d = results['β⁻']
    ax1.scatter(d['stress'], d['log_t_half'], c=mode_colors['β⁻'],
                s=80, alpha=0.7, edgecolors='black', linewidths=1)

    # Fit line
    if d['p_linear'] < 0.1:  # Show trend if p < 0.1
        x_fit = np.linspace(d['stress'].min(), d['stress'].max(), 100)
        slope, intercept = np.polyfit(d['stress'], d['log_t_half'], 1)
        ax1.plot(x_fit, slope*x_fit + intercept, 'r--', linewidth=2,
                label=f"r={d['r_linear']:.3f}, p={d['p_linear']:.3f}")

    ax1.set_xlabel('Geometric Stress σ', fontsize=11, fontweight='bold')
    ax1.set_ylabel('log₁₀(Half-Life) [seconds]', fontsize=11, fontweight='bold')
    ax1.set_title(f"(A) β⁻ Decay (n={d['n']})", fontsize=13, fontweight='bold')
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=9)

# Panel 2: β⁺/EC decay
if 'β⁺/EC' in results:
    ax2 = fig.add_subplot(gs[0, 1])
    d = results['β⁺/EC']
    ax2.scatter(d['stress'], d['log_t_half'], c=mode_colors['β⁺/EC'],
                s=80, alpha=0.7, edgecolors='black', linewidths=1)

    if d['p_linear'] < 0.1:
        x_fit = np.linspace(d['stress'].min(), d['stress'].max(), 100)
        slope, intercept = np.polyfit(d['stress'], d['log_t_half'], 1)
        ax2.plot(x_fit, slope*x_fit + intercept, '--', color='orange', linewidth=2,
                label=f"r={d['r_linear']:.3f}, p={d['p_linear']:.3f}")

    ax2.set_xlabel('Geometric Stress σ', fontsize=11, fontweight='bold')
    ax2.set_ylabel('log₁₀(Half-Life) [seconds]', fontsize=11, fontweight='bold')
    ax2.set_title(f"(B) β⁺/EC Decay (n={d['n']})", fontsize=13, fontweight='bold')
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=9)

# Panel 3: α decay
if 'α' in results:
    ax3 = fig.add_subplot(gs[0, 2])
    d = results['α']
    ax3.scatter(d['stress'], d['log_t_half'], c=mode_colors['α'],
                s=80, alpha=0.7, edgecolors='black', linewidths=1)

    if d['p_linear'] < 0.1:
        x_fit = np.linspace(d['stress'].min(), d['stress'].max(), 100)
        slope, intercept = np.polyfit(d['stress'], d['log_t_half'], 1)
        ax3.plot(x_fit, slope*x_fit + intercept, 'b--', linewidth=2,
                label=f"r={d['r_linear']:.3f}, p={d['p_linear']:.3f}")

    ax3.set_xlabel('Geometric Stress σ', fontsize=11, fontweight='bold')
    ax3.set_ylabel('log₁₀(Half-Life) [seconds]', fontsize=11, fontweight='bold')
    ax3.set_title(f"(C) α Decay (n={d['n']})", fontsize=13, fontweight='bold')
    ax3.grid(alpha=0.3)
    ax3.legend(fontsize=9)

# Panel 4: Q-value vs stress (all modes)
ax4 = fig.add_subplot(gs[1, 0])
for mode, data_list in data_by_mode.items():
    Q_vals = [d['Q_value'] for d in data_list if not np.isnan(d['Q_value'])]
    stress_vals = [d['stress'] for d in data_list if not np.isnan(d['Q_value'])]
    if len(Q_vals) > 0:
        ax4.scatter(stress_vals, Q_vals, c=mode_colors.get(mode, 'gray'),
                   s=60, alpha=0.6, label=mode, edgecolors='black', linewidths=0.5)

ax4.set_xlabel('Geometric Stress σ', fontsize=11, fontweight='bold')
ax4.set_ylabel('Q-value [MeV]', fontsize=11, fontweight='bold')
ax4.set_title('(D) Q-value vs Stress (All Modes)', fontsize=13, fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(alpha=0.3)

# Panel 5: Daughter stress change
ax5 = fig.add_subplot(gs[1, 1])
for mode, data_list in data_by_mode.items():
    parent_stress = [d['stress'] for d in data_list if not np.isnan(d['delta_stress'])]
    delta_stress = [d['delta_stress'] for d in data_list if not np.isnan(d['delta_stress'])]
    if len(delta_stress) > 0:
        ax5.scatter(parent_stress, delta_stress, c=mode_colors.get(mode, 'gray'),
                   s=60, alpha=0.6, label=mode, edgecolors='black', linewidths=0.5)

ax5.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax5.set_xlabel('Parent Stress σ', fontsize=11, fontweight='bold')
ax5.set_ylabel('Δσ = σ(daughter) - σ(parent)', fontsize=11, fontweight='bold')
ax5.set_title('(E) Stress Change After Decay', fontsize=13, fontweight='bold')
ax5.legend(fontsize=9)
ax5.grid(alpha=0.3)
ax5.text(0.02, 0.98, 'Negative Δσ = decay toward lower stress ✓',
         transform=ax5.transAxes, fontsize=9, va='top',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# Panel 6: Parent vs Daughter stress (manifold positions)
ax6 = fig.add_subplot(gs[1, 2])
for mode, data_list in data_by_mode.items():
    parent_stress = [d['stress'] for d in data_list if not np.isnan(d['stress_daughter'])]
    daughter_stress = [d['stress_daughter'] for d in data_list if not np.isnan(d['stress_daughter'])]
    if len(parent_stress) > 0:
        ax6.scatter(parent_stress, daughter_stress, c=mode_colors.get(mode, 'gray'),
                   s=60, alpha=0.6, label=mode, edgecolors='black', linewidths=0.5)

# Diagonal line (no stress change)
ax6.plot([0, 3], [0, 3], 'k--', linewidth=1, alpha=0.5, label='σ(parent) = σ(daughter)')
ax6.set_xlabel('Parent Stress σ', fontsize=11, fontweight='bold')
ax6.set_ylabel('Daughter Stress σ', fontsize=11, fontweight='bold')
ax6.set_title('(F) Parent vs Daughter on Stress Manifold', fontsize=13, fontweight='bold')
ax6.legend(fontsize=9, loc='upper left')
ax6.grid(alpha=0.3)
ax6.text(0.98, 0.02, 'Below diagonal = decay toward lower stress',
         transform=ax6.transAxes, fontsize=9, ha='right', va='bottom',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# Panel 7: log(t_1/2) vs Q-value (all modes)
ax7 = fig.add_subplot(gs[2, 0])
for mode, data_list in data_by_mode.items():
    Q_vals = [d['Q_value'] for d in data_list if not np.isnan(d['Q_value'])]
    log_t_half_vals = [d['log_t_half'] for d in data_list if not np.isnan(d['Q_value'])]
    if len(Q_vals) > 0:
        ax7.scatter(Q_vals, log_t_half_vals, c=mode_colors.get(mode, 'gray'),
                   s=60, alpha=0.6, label=mode, edgecolors='black', linewidths=0.5)

ax7.set_xlabel('Q-value [MeV]', fontsize=11, fontweight='bold')
ax7.set_ylabel('log₁₀(Half-Life) [seconds]', fontsize=11, fontweight='bold')
ax7.set_title('(G) Half-Life vs Q-value (All Modes)', fontsize=13, fontweight='bold')
ax7.legend(fontsize=9)
ax7.grid(alpha=0.3)

# Panel 8: Stress distribution by decay mode
ax8 = fig.add_subplot(gs[2, 1])
stress_by_mode = []
labels_by_mode = []
colors_by_mode = []
for mode, data_list in sorted(data_by_mode.items(), key=lambda x: -len(x[1])):
    if len(data_list) >= 3:
        stress_by_mode.append([d['stress'] for d in data_list])
        labels_by_mode.append(f"{mode}\n(n={len(data_list)})")
        colors_by_mode.append(mode_colors.get(mode, 'gray'))

if len(stress_by_mode) > 0:
    bp = ax8.boxplot(stress_by_mode, labels=labels_by_mode, patch_artist=True,
                     widths=0.6, showfliers=True)
    for patch, color in zip(bp['boxes'], colors_by_mode):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

ax8.set_ylabel('Geometric Stress σ', fontsize=11, fontweight='bold')
ax8.set_title('(H) Stress Distribution by Decay Mode', fontsize=13, fontweight='bold')
ax8.grid(axis='y', alpha=0.3)
ax8.axhline(2.5, color='green', linestyle='--', linewidth=1.5, alpha=0.5, label='Stable region (σ<2.5)')
ax8.axhline(3.5, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='Drip line (σ≈3.5)')
ax8.legend(fontsize=8, loc='upper right')

# Panel 9: Summary table
ax9 = fig.add_subplot(gs[2, 2])
ax9.axis('off')

summary_text = "CORRELATION SUMMARY\n" + "="*40 + "\n\n"
for mode in ['β⁻', 'β⁺/EC', 'α', 'neutron']:
    if mode in results:
        d = results[mode]
        summary_text += f"{mode} decay (n={d['n']}):\n"
        summary_text += f"  r(σ, log t) = {d['r_linear']:+.3f}"
        if d['p_linear'] < 0.05:
            summary_text += " ★\n"
        else:
            summary_text += f" (p={d['p_linear']:.3f})\n"

        if len(d['delta_stress']) > 0:
            mean_delta = np.mean(d['delta_stress'])
            summary_text += f"  <Δσ> = {mean_delta:+.3f}"
            if mean_delta < 0:
                summary_text += " ✓\n"
            else:
                summary_text += "\n"
        summary_text += "\n"

summary_text += "\n★ = p < 0.05 (significant)\n"
summary_text += "✓ = decay toward lower stress\n"

ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes,
         fontsize=10, va='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.suptitle('HALF-LIFE vs STRESS: DECAY-MODE-SPECIFIC ANALYSIS\n' +
             'Testing Correlations Separately for Each Decay Mode',
             fontsize=16, fontweight='bold', y=0.995)

plt.savefig('halflife_by_decay_mode.png', dpi=200, bbox_inches='tight')
plt.savefig('halflife_by_decay_mode.pdf', bbox_inches='tight')

print("="*80)
print("FIGURE SAVED")
print("="*80)
print("  - halflife_by_decay_mode.png (200 DPI)")
print("  - halflife_by_decay_mode.pdf (vector)")
print()

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("="*80)
print("DECAY MODE-SPECIFIC CONCLUSIONS")
print("="*80)
print()

for mode in ['β⁻', 'β⁺/EC', 'α', 'neutron']:
    if mode in results:
        d = results[mode]
        print(f"{mode} DECAY:")

        if d['p_linear'] < 0.05:
            print(f"  ✓ SIGNIFICANT correlation: r = {d['r_linear']:+.3f}, p = {d['p_linear']:.4f}")
        else:
            print(f"  ✗ No significant correlation: r = {d['r_linear']:+.3f}, p = {d['p_linear']:.4f}")

        if len(d['delta_stress']) > 0:
            mean_delta = np.mean(d['delta_stress'])
            pct_decrease = 100 * np.sum(d['delta_stress'] < 0) / len(d['delta_stress'])
            print(f"  Mean Δσ = {mean_delta:+.3f}")
            print(f"  {pct_decrease:.1f}% decay toward lower stress")

        print()

print("="*80)
print("KEY INSIGHTS")
print("="*80)
print()
print("1. Testing each decay mode separately may reveal mode-specific patterns")
print("2. Daughter stress change (Δσ) shows if decay reduces geometric stress")
print("3. Q-value may be the dominant factor, not geometric stress alone")
print("4. α decay and β decay have different physics (barrier vs phase space)")
print()
print("="*80)
