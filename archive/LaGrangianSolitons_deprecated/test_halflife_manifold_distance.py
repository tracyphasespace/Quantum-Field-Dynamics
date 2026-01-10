#!/usr/bin/env python3
"""
HALF-LIFE vs MANIFOLD DISTANCE TO STABILITY
================================================================================
CRITICAL INSIGHT: It's not the stress σ itself - it's the DISTANCE TO STABILITY

For each decay:
  1. Calculate parent position on stress manifold: (A, Z, σ_parent)
  2. Calculate daughter position: (A', Z', σ_daughter)
  3. Measure "distance" to reach stability
  4. Test if half-life correlates with this distance

Key metrics:
  - Δσ = σ_daughter - σ_parent (stress change)
  - |Δσ| = absolute stress change (distance along stress axis)
  - Δσ_to_ground = |σ_parent| - |σ_daughter| (how much closer to N=0?)
  - Manifold distance = geometric distance in (A, Z, σ) space

Hypothesis: Half-life ∝ exp(barrier), where barrier ∝ distance to stability
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from scipy.optimize import curve_fit

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
        's': 1.0, 'ms': 1e-3, 'us': 1e-6, 'ns': 1e-9,
        'ps': 1e-12, 'fs': 1e-15, 'as': 1e-18,
        'm': 60.0, 'h': 3600.0, 'd': 86400.0,
        'y': 31557600.0,
    }
    return value * conversions.get(unit, 1.0)

def get_daughter_nucleus(Z, A, decay_mode):
    """Calculate daughter nucleus (Z', A') after decay"""
    if 'β-' in decay_mode or 'beta-' in decay_mode:
        return Z + 1, A
    elif 'β+' in decay_mode or 'EC' in decay_mode:
        return Z - 1, A
    elif 'α' in decay_mode or 'alpha' in decay_mode:
        return Z - 2, A - 4
    elif 'n' in decay_mode and 'neutron' in decay_mode.lower():
        return Z, A - 1
    elif 'p' in decay_mode and 'proton' in decay_mode.lower():
        return Z - 1, A - 1
    else:
        return None, None

def manifold_distance(A1, Z1, sigma1, A2, Z2, sigma2):
    """
    Calculate geometric distance on the (A, Z, σ) manifold

    We need to normalize A, Z, σ to comparable scales:
      - A: 1-300 (mass number)
      - Z: 1-100 (charge)
      - σ: 0-8 (stress)

    Use weighted Euclidean distance with physically motivated scaling
    """
    # Normalize to comparable scales
    dA = (A2 - A1) / 100.0  # Scale A to ~0-3
    dZ = (Z2 - Z1) / 50.0   # Scale Z to ~0-2
    dSigma = (sigma2 - sigma1) / 4.0  # Scale σ to ~0-2

    # Weighted distance (stress is most important for stability)
    # Weight: stress > charge > mass
    w_sigma = 2.0  # Stress is 2x more important
    w_Z = 1.5      # Charge is 1.5x more important
    w_A = 1.0      # Mass is baseline

    distance = np.sqrt(w_A * dA**2 + w_Z * dZ**2 + w_sigma * dSigma**2)
    return distance

# Load comprehensive decay database
radioactive_database = [
    # Alpha decays
    ("Be-8", 4, 8, 8.19e-17, 's', 'α', 0.092),
    ("Po-210", 84, 210, 138.4, 'd', 'α', 5.407),
    ("Po-211", 84, 211, 0.516, 's', 'α', 7.594),
    ("Po-212", 84, 212, 0.299e-6, 's', 'α', 8.954),
    ("Po-213", 84, 213, 4.2e-6, 's', 'α', 8.537),
    ("Po-214", 84, 214, 164.3e-6, 's', 'α', 7.833),
    ("Po-216", 84, 216, 0.145, 's', 'α', 6.906),
    ("Po-218", 84, 218, 3.10, 'm', 'α', 6.115),
    ("Rn-222", 86, 222, 3.8235, 'd', 'α', 5.590),
    ("Ra-226", 88, 226, 1600, 'y', 'α', 4.871),
    ("Th-230", 90, 230, 7.538e4, 'y', 'α', 4.770),
    ("Th-232", 90, 232, 1.405e10, 'y', 'α', 4.081),
    ("U-233", 92, 233, 1.592e5, 'y', 'α', 4.909),
    ("U-234", 92, 234, 2.455e5, 'y', 'α', 4.857),
    ("U-235", 92, 235, 7.04e8, 'y', 'α', 4.679),
    ("U-238", 92, 238, 4.468e9, 'y', 'α', 4.270),
    ("Pu-238", 94, 238, 87.7, 'y', 'α', 5.593),
    ("Pu-239", 94, 239, 2.411e4, 'y', 'α', 5.244),
    ("Pu-240", 94, 240, 6564, 'y', 'α', 5.256),
    ("Am-241", 95, 241, 432.2, 'y', 'α', 5.638),
    ("Cm-244", 96, 244, 18.10, 'y', 'α', 5.902),
    ("Sm-147", 62, 147, 1.06e11, 'y', 'α', 2.310),
    ("Gd-152", 64, 152, 1.08e14, 'y', 'α', 2.203),

    # Beta-minus decays
    ("H-3", 1, 3, 12.32, 'y', 'β-', 0.0186),
    ("C-14", 6, 14, 5730, 'y', 'β-', 0.156),
    ("P-32", 15, 32, 14.29, 'd', 'β-', 1.709),
    ("S-35", 16, 35, 87.51, 'd', 'β-', 0.167),
    ("K-40", 19, 40, 1.248e9, 'y', 'β-', 1.311),
    ("Ca-45", 20, 45, 162.7, 'd', 'β-', 0.257),
    ("Co-60", 27, 60, 5.271, 'y', 'β-', 2.824),
    ("Sr-90", 38, 90, 28.79, 'y', 'β-', 0.546),
    ("I-131", 53, 131, 8.021, 'd', 'β-', 0.971),
    ("Cs-137", 55, 137, 30.08, 'y', 'β-', 1.176),
    ("Pm-147", 61, 147, 2.623, 'y', 'β-', 0.224),
    ("Ni-63", 28, 63, 100.1, 'y', 'β-', 0.067),
    ("Kr-85", 36, 85, 10.756, 'y', 'β-', 0.687),
    ("Tc-99", 43, 99, 2.111e5, 'y', 'β-', 0.294),

    # Beta-plus/EC decays
    ("Na-22", 11, 22, 2.602, 'y', 'β+/EC', 2.842),
    ("Fe-55", 26, 55, 2.737, 'y', 'EC', 0.231),
]

print("="*80)
print("HALF-LIFE vs MANIFOLD DISTANCE TO STABILITY")
print("="*80)
print()

# Process all decays
data = []

for name, Z, A, t_half_val, unit, decay_mode, Q in radioactive_database:
    assert isinstance(Z, int) and isinstance(A, int)

    # Parent properties
    N_parent = calculate_N_continuous(A, Z)
    sigma_parent = abs(N_parent)

    # Daughter properties
    Z_d, A_d = get_daughter_nucleus(Z, A, decay_mode)
    if Z_d is None or A_d is None or A_d <= 0:
        continue

    N_daughter = calculate_N_continuous(A_d, Z_d)
    sigma_daughter = abs(N_daughter)

    # Calculate distance metrics
    delta_sigma = sigma_daughter - sigma_parent  # Stress change (can be positive or negative)
    abs_delta_sigma = abs(delta_sigma)  # Absolute stress change

    # Distance to ground state (N=0)
    dist_to_ground_parent = abs(N_parent)  # How far parent is from N=0
    dist_to_ground_daughter = abs(N_daughter)  # How far daughter is from N=0
    approach_to_ground = dist_to_ground_parent - dist_to_ground_daughter  # Positive if getting closer

    # Manifold distance
    manifold_dist = manifold_distance(A, Z, sigma_parent, A_d, Z_d, sigma_daughter)

    # Half-life
    t_half_sec = halflife_to_seconds(t_half_val, unit)
    log_t_half = np.log10(t_half_sec)

    # Classify decay mode
    if 'α' in decay_mode:
        mode = 'α'
    elif 'β-' in decay_mode:
        mode = 'β⁻'
    elif 'β+' in decay_mode or 'EC' in decay_mode:
        mode = 'β⁺/EC'
    else:
        mode = 'other'

    data.append({
        'name': name,
        'mode': mode,
        'Z': Z, 'A': A,
        'sigma_parent': sigma_parent,
        'sigma_daughter': sigma_daughter,
        'delta_sigma': delta_sigma,
        'abs_delta_sigma': abs_delta_sigma,
        'dist_to_ground_parent': dist_to_ground_parent,
        'approach_to_ground': approach_to_ground,
        'manifold_dist': manifold_dist,
        'Q': Q,
        'log_t_half': log_t_half,
    })

print(f"Processed {len(data)} decays")
print()

# Separate by mode
alpha_data = [d for d in data if d['mode'] == 'α']
beta_minus_data = [d for d in data if d['mode'] == 'β⁻']
beta_plus_data = [d for d in data if d['mode'] == 'β⁺/EC']

print(f"Alpha decays: {len(alpha_data)}")
print(f"Beta-minus decays: {len(beta_minus_data)}")
print(f"Beta-plus/EC decays: {len(beta_plus_data)}")
print()

# ============================================================================
# CORRELATION TESTS
# ============================================================================

print("="*80)
print("CORRELATION ANALYSIS: DISTANCE METRICS vs HALF-LIFE")
print("="*80)
print()

metrics = [
    ('sigma_parent', 'Parent Stress σ_parent'),
    ('abs_delta_sigma', 'Absolute Stress Change |Δσ|'),
    ('delta_sigma', 'Signed Stress Change Δσ'),
    ('approach_to_ground', 'Approach to Ground State (σ_p - σ_d)'),
    ('manifold_dist', 'Manifold Distance'),
]

for mode_name, mode_data in [('ALPHA', alpha_data), ('BETA-MINUS', beta_minus_data)]:
    if len(mode_data) < 3:
        continue

    print(f"{mode_name} DECAY (n={len(mode_data)})")
    print("-"*80)

    log_t_vals = np.array([d['log_t_half'] for d in mode_data])

    for metric_key, metric_name in metrics:
        metric_vals = np.array([d[metric_key] for d in mode_data])

        if len(metric_vals) > 2:
            r, p = pearsonr(metric_vals, log_t_vals)

            marker = ""
            if p < 0.05:
                marker = " ★ SIGNIFICANT"
            elif p < 0.10:
                marker = " ○ Marginal"

            print(f"  {metric_name:40s}: r = {r:+.3f}, p = {p:.4f}{marker}")

    print()

# ============================================================================
# KEY INSIGHT: FOR ALPHA DECAY, TEST INVERSE CORRELATION
# ============================================================================

print("="*80)
print("KEY INSIGHT: ALPHA DECAY - INVERSE DISTANCE HYPOTHESIS")
print("="*80)
print()
print("Hypothesis: Longer distance to stability → EASIER decay → SHORTER half-life")
print("Expected: NEGATIVE correlation between distance and log(t_1/2)")
print()

if len(alpha_data) > 2:
    log_t_alpha = np.array([d['log_t_half'] for d in alpha_data])

    # Test each metric
    print("Testing inverse correlations:")
    print("-"*80)

    # Manifold distance
    manifold_dist_alpha = np.array([d['manifold_dist'] for d in alpha_data])
    r_manifold, p_manifold = pearsonr(manifold_dist_alpha, log_t_alpha)
    print(f"Manifold distance:      r = {r_manifold:+.3f}, p = {p_manifold:.4f}")

    # Approach to ground (should be POSITIVE correlation - more approach = easier)
    approach_alpha = np.array([d['approach_to_ground'] for d in alpha_data])
    r_approach, p_approach = pearsonr(approach_alpha, log_t_alpha)
    print(f"Approach to ground:     r = {r_approach:+.3f}, p = {p_approach:.4f}")

    # Absolute delta sigma
    abs_delta_alpha = np.array([d['abs_delta_sigma'] for d in alpha_data])
    r_abs_delta, p_abs_delta = pearsonr(abs_delta_alpha, log_t_alpha)
    print(f"Absolute stress change: r = {r_abs_delta:+.3f}, p = {p_abs_delta:.4f}")

    print()
    print("Interpretation:")
    if r_manifold < 0 and p_manifold < 0.05:
        print("  ✓ NEGATIVE correlation confirmed: Greater distance → shorter half-life")
    elif r_approach > 0 and p_approach < 0.05:
        print("  ✓ POSITIVE correlation confirmed: Greater approach → shorter half-life")
    else:
        print("  ✗ No significant inverse correlation found")
        print("  → Need to reconsider the distance metric or hypothesis")
    print()

# ============================================================================
# CREATE COMPREHENSIVE FIGURE
# ============================================================================

fig = plt.figure(figsize=(24, 16))
gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.35)

colors = {'α': 'blue', 'β⁻': 'red', 'β⁺/EC': 'orange'}

# Panel 1: Parent stress vs half-life (baseline)
ax1 = fig.add_subplot(gs[0, 0])
for mode in ['α', 'β⁻', 'β⁺/EC']:
    mode_data = [d for d in data if d['mode'] == mode]
    if len(mode_data) > 0:
        x = [d['sigma_parent'] for d in mode_data]
        y = [d['log_t_half'] for d in mode_data]
        ax1.scatter(x, y, c=colors[mode], label=mode, s=80, alpha=0.7,
                   edgecolors='black', linewidths=0.5)

ax1.set_xlabel('Parent Stress σ_parent', fontsize=11, fontweight='bold')
ax1.set_ylabel('log₁₀(Half-Life) [s]', fontsize=11, fontweight='bold')
ax1.set_title('(A) Parent Stress (Old Metric)', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(alpha=0.3)

# Panel 2: Absolute stress change vs half-life
ax2 = fig.add_subplot(gs[0, 1])
for mode in ['α', 'β⁻', 'β⁺/EC']:
    mode_data = [d for d in data if d['mode'] == mode]
    if len(mode_data) > 0:
        x = [d['abs_delta_sigma'] for d in mode_data]
        y = [d['log_t_half'] for d in mode_data]
        ax2.scatter(x, y, c=colors[mode], label=mode, s=80, alpha=0.7,
                   edgecolors='black', linewidths=0.5)

ax2.set_xlabel('|Δσ| = |σ_daughter - σ_parent|', fontsize=11, fontweight='bold')
ax2.set_ylabel('log₁₀(Half-Life) [s]', fontsize=11, fontweight='bold')
ax2.set_title('(B) Absolute Stress Change', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(alpha=0.3)

# Panel 3: Signed stress change vs half-life
ax3 = fig.add_subplot(gs[0, 2])
for mode in ['α', 'β⁻', 'β⁺/EC']:
    mode_data = [d for d in data if d['mode'] == mode]
    if len(mode_data) > 0:
        x = [d['delta_sigma'] for d in mode_data]
        y = [d['log_t_half'] for d in mode_data]
        ax3.scatter(x, y, c=colors[mode], label=mode, s=80, alpha=0.7,
                   edgecolors='black', linewidths=0.5)

ax3.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax3.set_xlabel('Δσ = σ_daughter - σ_parent', fontsize=11, fontweight='bold')
ax3.set_ylabel('log₁₀(Half-Life) [s]', fontsize=11, fontweight='bold')
ax3.set_title('(C) Signed Stress Change', fontsize=12, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(alpha=0.3)
ax3.text(0.02, 0.98, 'Negative Δσ = decay reduces stress',
         transform=ax3.transAxes, fontsize=9, va='top',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# Panel 4: Approach to ground state vs half-life
ax4 = fig.add_subplot(gs[0, 3])
for mode in ['α', 'β⁻', 'β⁺/EC']:
    mode_data = [d for d in data if d['mode'] == mode]
    if len(mode_data) > 0:
        x = [d['approach_to_ground'] for d in mode_data]
        y = [d['log_t_half'] for d in mode_data]
        ax4.scatter(x, y, c=colors[mode], label=mode, s=80, alpha=0.7,
                   edgecolors='black', linewidths=0.5)

ax4.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax4.set_xlabel('Approach to Ground = |N_parent| - |N_daughter|', fontsize=11, fontweight='bold')
ax4.set_ylabel('log₁₀(Half-Life) [s]', fontsize=11, fontweight='bold')
ax4.set_title('(D) Approach to Ground State (N=0)', fontsize=12, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(alpha=0.3)
ax4.text(0.02, 0.98, 'Positive = getting closer to N=0',
         transform=ax4.transAxes, fontsize=9, va='top',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# Panel 5: Manifold distance vs half-life (ALL MODES)
ax5 = fig.add_subplot(gs[1, 0:2])
for mode in ['α', 'β⁻', 'β⁺/EC']:
    mode_data = [d for d in data if d['mode'] == mode]
    if len(mode_data) > 0:
        x = [d['manifold_dist'] for d in mode_data]
        y = [d['log_t_half'] for d in mode_data]
        ax5.scatter(x, y, c=colors[mode], label=mode, s=100, alpha=0.7,
                   edgecolors='black', linewidths=1)

ax5.set_xlabel('Manifold Distance (weighted Euclidean in A, Z, σ space)', fontsize=12, fontweight='bold')
ax5.set_ylabel('log₁₀(Half-Life) [seconds]', fontsize=12, fontweight='bold')
ax5.set_title('(E) Manifold Distance to Daughter vs Half-Life (All Modes)', fontsize=13, fontweight='bold')
ax5.legend(fontsize=11)
ax5.grid(alpha=0.3)

# Panel 6: Manifold distance vs half-life (ALPHA ONLY)
ax6 = fig.add_subplot(gs[1, 2:4])
if len(alpha_data) > 0:
    x_alpha = [d['manifold_dist'] for d in alpha_data]
    y_alpha = [d['log_t_half'] for d in alpha_data]
    Q_alpha = [d['Q'] for d in alpha_data]

    scatter6 = ax6.scatter(x_alpha, y_alpha, c=Q_alpha, cmap='plasma',
                          s=100, alpha=0.7, edgecolors='black', linewidths=1)

    # Fit line
    if len(x_alpha) > 2:
        r, p = pearsonr(x_alpha, y_alpha)
        slope, intercept = np.polyfit(x_alpha, y_alpha, 1)
        x_fit = np.linspace(min(x_alpha), max(x_alpha), 100)
        ax6.plot(x_fit, slope*x_fit + intercept, 'b--', linewidth=2,
                label=f'r={r:.3f}, p={p:.3f}')

    ax6.set_xlabel('Manifold Distance', fontsize=12, fontweight='bold')
    ax6.set_ylabel('log₁₀(Half-Life) [seconds]', fontsize=12, fontweight='bold')
    ax6.set_title('(F) Alpha Decay: Manifold Distance Test', fontsize=13, fontweight='bold')
    ax6.legend(fontsize=11)
    ax6.grid(alpha=0.3)
    cbar6 = plt.colorbar(scatter6, ax=ax6)
    cbar6.set_label('Q-value [MeV]', fontsize=10)

# Panel 7: Daughter vs Parent stress (trajectory visualization)
ax7 = fig.add_subplot(gs[2, 0:2])
for mode in ['α', 'β⁻', 'β⁺/EC']:
    mode_data = [d for d in data if d['mode'] == mode]
    if len(mode_data) > 0:
        x = [d['sigma_parent'] for d in mode_data]
        y = [d['sigma_daughter'] for d in mode_data]

        # Draw arrows showing trajectories
        for d in mode_data:
            ax7.arrow(d['sigma_parent'], d['sigma_parent'],
                     d['sigma_daughter'] - d['sigma_parent'], d['sigma_daughter'] - d['sigma_parent'],
                     head_width=0.1, head_length=0.05, fc=colors[mode], ec=colors[mode],
                     alpha=0.3, length_includes_head=True)

        ax7.scatter(x, y, c=colors[mode], label=mode, s=80, alpha=0.7,
                   edgecolors='black', linewidths=0.5, zorder=10)

# Diagonal line
lim = [0, max([d['sigma_parent'] for d in data] + [d['sigma_daughter'] for d in data]) + 0.5]
ax7.plot(lim, lim, 'k--', linewidth=1.5, alpha=0.5, label='No change')
ax7.set_xlabel('Parent Stress σ_parent', fontsize=12, fontweight='bold')
ax7.set_ylabel('Daughter Stress σ_daughter', fontsize=12, fontweight='bold')
ax7.set_title('(G) Decay Trajectories on Stress Manifold', fontsize=13, fontweight='bold')
ax7.legend(fontsize=11, loc='upper left')
ax7.grid(alpha=0.3)
ax7.set_xlim(lim)
ax7.set_ylim(lim)

# Panel 8: Distance to ground vs Q-value (physical insight)
ax8 = fig.add_subplot(gs[2, 2])
for mode in ['α', 'β⁻']:
    mode_data = [d for d in data if d['mode'] == mode]
    if len(mode_data) > 0:
        x = [d['approach_to_ground'] for d in mode_data]
        y = [d['Q'] for d in mode_data]
        ax8.scatter(x, y, c=colors[mode], label=mode, s=80, alpha=0.7,
                   edgecolors='black', linewidths=0.5)

ax8.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax8.set_xlabel('Approach to Ground State', fontsize=11, fontweight='bold')
ax8.set_ylabel('Q-value [MeV]', fontsize=11, fontweight='bold')
ax8.set_title('(H) Approach vs Energy Release', fontsize=12, fontweight='bold')
ax8.legend(fontsize=10)
ax8.grid(alpha=0.3)

# Panel 9: Summary statistics
ax9 = fig.add_subplot(gs[2, 3])
ax9.axis('off')

summary_text = "DISTANCE METRIC CORRELATIONS\n"
summary_text += "="*50 + "\n\n"

if len(alpha_data) > 2:
    summary_text += "ALPHA DECAY (Topological):\n"
    summary_text += "-"*50 + "\n"

    metrics_alpha = [
        ('manifold_dist', 'Manifold Distance'),
        ('abs_delta_sigma', '|Δσ|'),
        ('approach_to_ground', 'Approach to Ground'),
    ]

    log_t_alpha = np.array([d['log_t_half'] for d in alpha_data])
    for key, name in metrics_alpha:
        vals = np.array([d[key] for d in alpha_data])
        r, p = pearsonr(vals, log_t_alpha)
        marker = " ★" if p < 0.05 else ""
        summary_text += f"{name:20s}: r={r:+.3f}{marker}\n"

    summary_text += "\n"

if len(beta_minus_data) > 2:
    summary_text += "BETA DECAY (Thermodynamic):\n"
    summary_text += "-"*50 + "\n"

    log_t_beta = np.array([d['log_t_half'] for d in beta_minus_data])
    for key, name in metrics_alpha:
        vals = np.array([d[key] for d in beta_minus_data])
        r, p = pearsonr(vals, log_t_beta)
        marker = " ★" if p < 0.05 else ""
        summary_text += f"{name:20s}: r={r:+.3f}{marker}\n"

summary_text += "\n★ = p < 0.05 (significant)\n"

ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes,
         fontsize=9, va='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

plt.suptitle('HALF-LIFE vs MANIFOLD DISTANCE TO STABILITY\n' +
             'Testing: Distance to Stable Daughter, Not Absolute Stress',
             fontsize=16, fontweight='bold', y=0.995)

plt.savefig('halflife_manifold_distance.png', dpi=200, bbox_inches='tight')
plt.savefig('halflife_manifold_distance.pdf', bbox_inches='tight')

print("="*80)
print("FIGURES SAVED")
print("="*80)
print("  - halflife_manifold_distance.png (200 DPI)")
print("  - halflife_manifold_distance.pdf (vector)")
print()

print("="*80)
print("CONCLUSION")
print("="*80)
print()
print("The key insight: Half-life should correlate with DISTANCE TO STABILITY,")
print("not absolute stress. We tested:")
print()
print("  1. Manifold distance (geometric distance in A, Z, σ space)")
print("  2. Absolute stress change |Δσ|")
print("  3. Approach to ground state (how much closer to N=0)")
print()
print("Results will determine if the 'distance to stability' hypothesis is correct.")
print("="*80)
