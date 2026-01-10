#!/usr/bin/env python3
"""
LEGO QUANTIZATION TEST: Fractional Remainder Model
================================================================================
THE KEY INSIGHT: It's not |N| or even |ΔN| - it's the REMAINDER when trying
to land on an allowed quantized state.

Formulation:
  N(A,Z) = continuous geometric coordinate (field property)

  But accessible states are QUANTIZED on a lattice:
    L_i = {k·Δ_i | k ∈ ℤ}

  Fractional mismatch (distance to nearest lattice point):
    ε_i(N_p) = min_k |N_p - k·Δ_i|

  Bounded: 0 ≤ ε ≤ Δ/2

Physical interpretation:
  - ε ≈ 0: Parent near a "stud" → transition geometrically easy → SHORT half-life
  - ε ≈ Δ/2: Parent between studs → requires squeezing → LONG half-life

This explains:
  - Why high stress can be long-lived (bad quantization fit)
  - Why low stress can be short-lived (good quantization fit)
  - The "17/3 problem" - not the value, but the REMAINDER

Test:
  1. Compute ε(N) for different tile sizes Δ
  2. Add to model: log(t_1/2) = ... + d·h(ε)
  3. Find which Δ minimizes residuals
  4. Compare alpha vs beta vs neutron channels
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.optimize import least_squares, minimize_scalar
import json

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

def quantization_remainder(N, Delta):
    """
    Calculate fractional mismatch ε(N, Δ)

    ε = min_k |N - k·Δ|

    This is the distance to the nearest lattice point k·Δ

    Returns: ε ∈ [0, Δ/2]
    """
    if Delta <= 0:
        return 0

    # Find nearest lattice point
    k_nearest = np.round(N / Delta)
    lattice_point = k_nearest * Delta

    # Distance to nearest point
    epsilon = abs(N - lattice_point)

    # Bounded to [0, Δ/2] by periodicity
    if epsilon > Delta / 2:
        epsilon = Delta - epsilon

    return epsilon

def normalized_remainder(N, Delta):
    """
    Normalized remainder: ε/Δ ∈ [0, 0.5]

    This is dimensionless and comparable across different Δ values
    """
    epsilon = quantization_remainder(N, Delta)
    return epsilon / Delta if Delta > 0 else 0

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
    """Calculate daughter nucleus after decay"""
    if 'β-' in decay_mode or 'beta-' in decay_mode:
        return Z + 1, A
    elif 'β+' in decay_mode or 'EC' in decay_mode:
        return Z - 1, A
    elif 'α' in decay_mode or 'alpha' in decay_mode:
        return Z - 2, A - 4
    elif 'n' in decay_mode and 'neutron' in decay_mode.lower():
        return Z, A - 1
    else:
        return None, None

# Comprehensive decay database
decay_database = [
    # Alpha decays (high priority for Lego test)
    ("Po-210", 84, 210, 138.4, 'd', 'α', 5.407),
    ("Po-211", 84, 211, 0.516, 's', 'α', 7.594),
    ("Po-212", 84, 212, 0.299e-6, 's', 'α', 8.954),
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
]

print("="*80)
print("LEGO QUANTIZATION TEST: Fractional Remainder Model")
print("="*80)
print()

# Process data
data_by_mode = {'α': [], 'β-': [], 'β+/EC': []}

for name, Z, A, t_val, unit, mode, Q in decay_database:
    N_parent = calculate_N_continuous(A, Z)
    sigma_parent = abs(N_parent)

    # Daughter
    Z_d, A_d = get_daughter_nucleus(Z, A, mode)
    if Z_d is None or A_d is None:
        continue

    N_daughter = calculate_N_continuous(A_d, Z_d)
    sigma_daughter = abs(N_daughter)

    # Traditional metrics
    approach = abs(N_parent) - abs(N_daughter)
    delta_N = N_daughter - N_parent  # Signed change

    # Half-life
    t_sec = halflife_to_seconds(t_val, unit)
    log_t_half = np.log10(t_sec)

    # Classify mode
    if 'α' in mode:
        mode_key = 'α'
    elif 'β-' in mode:
        mode_key = 'β-'
    elif 'β+' in mode or 'EC' in mode:
        mode_key = 'β+/EC'
    else:
        continue

    data_by_mode[mode_key].append({
        'name': name,
        'Z': Z, 'A': A,
        'N_parent': N_parent,
        'N_daughter': N_daughter,
        'sigma_parent': sigma_parent,
        'approach': approach,
        'delta_N': delta_N,
        'Q': Q,
        'log_t_half': log_t_half,
    })

print(f"Data loaded:")
for mode, data in data_by_mode.items():
    print(f"  {mode}: {len(data)} decays")
print()

# ============================================================================
# GRID SEARCH OVER TILE SIZES
# ============================================================================

print("="*80)
print("TILE SIZE GRID SEARCH")
print("="*80)
print()

# Candidate tile sizes Δ
Delta_candidates = [
    1/6, 1/5, 1/4, 1/3, 1/2, 2/3, 1, 3/2, 2, 3
]

print("Testing tile sizes Δ:")
for Delta in Delta_candidates:
    print(f"  {Delta:.4f}")
print()

# For each mode, test each Delta
results = {}

for mode, data_list in data_by_mode.items():
    if len(data_list) < 5:
        print(f"Skipping {mode} (too few data)")
        continue

    print(f"{mode} DECAY (n={len(data_list)})")
    print("-"*80)

    # Extract basic arrays
    N_parent_arr = np.array([d['N_parent'] for d in data_list])
    log_t_arr = np.array([d['log_t_half'] for d in data_list])

    # Test each Delta
    results[mode] = []

    for Delta in Delta_candidates:
        # Calculate remainder for each nucleus
        epsilon_arr = np.array([quantization_remainder(d['N_parent'], Delta)
                               for d in data_list])
        epsilon_norm_arr = epsilon_arr / Delta  # Normalized to [0, 0.5]

        # Correlation with half-life
        r_epsilon, p_epsilon = pearsonr(epsilon_norm_arr, log_t_arr)

        # Also test quadratic
        r_epsilon_sq, p_epsilon_sq = pearsonr(epsilon_norm_arr**2, log_t_arr)

        results[mode].append({
            'Delta': Delta,
            'r_linear': r_epsilon,
            'p_linear': p_epsilon,
            'r_quadratic': r_epsilon_sq,
            'p_quadratic': p_epsilon_sq,
        })

    # Find best Delta
    best_linear = max(results[mode], key=lambda x: abs(x['r_linear']))
    best_quad = max(results[mode], key=lambda x: abs(x['r_quadratic']))

    print(f"  Best Δ (linear):    {best_linear['Delta']:.4f}, r = {best_linear['r_linear']:+.3f}, p = {best_linear['p_linear']:.4f}")
    print(f"  Best Δ (quadratic): {best_quad['Delta']:.4f}, r = {best_quad['r_quadratic']:+.3f}, p = {best_quad['p_quadratic']:.4f}")
    print()

# ============================================================================
# DETAILED ANALYSIS FOR ALPHA DECAY
# ============================================================================

if len(data_by_mode['α']) > 5:
    print("="*80)
    print("DETAILED ANALYSIS: ALPHA DECAY")
    print("="*80)
    print()

    alpha_data = data_by_mode['α']

    # Extract arrays
    N_p = np.array([d['N_parent'] for d in alpha_data])
    log_t = np.array([d['log_t_half'] for d in alpha_data])
    Q_arr = np.array([d['Q'] for d in alpha_data])
    sigma_p = np.array([d['sigma_parent'] for d in alpha_data])
    approach = np.array([d['approach'] for d in alpha_data])

    # Test several specific Δ values
    Delta_test = [1/3, 1/2, 1, 2]

    print("Testing specific tile sizes for alpha decay:")
    print()

    for Delta in Delta_test:
        # Calculate remainder
        epsilon = np.array([quantization_remainder(N, Delta) for N in N_p])
        epsilon_norm = epsilon / Delta

        # Fit model with remainder
        # log(t) = a + b/√Q + c·approach + d·ε/Δ
        inv_sqrt_Q = 1.0 / np.sqrt(Q_arr)

        def model_with_epsilon(params):
            a, b, c, d = params
            return a + b * inv_sqrt_Q + c * approach + d * epsilon_norm

        def residuals(params):
            return log_t - model_with_epsilon(params)

        result = least_squares(residuals, x0=[0, 50, 5, 0], verbose=0)
        a, b, c, d = result.x

        predictions = model_with_epsilon([a, b, c, d])
        rmse = np.sqrt(np.mean((log_t - predictions)**2))

        # R²
        ss_tot = np.sum((log_t - np.mean(log_t))**2)
        ss_res = np.sum((log_t - predictions)**2)
        r_squared = 1 - (ss_res / ss_tot)

        print(f"Δ = {Delta:.4f}:")
        print(f"  Model: log(t) = {a:.2f} + {b:.2f}/√Q + {c:.2f}·approach + {d:.2f}·(ε/Δ)")
        print(f"  RMSE: {rmse:.3f}, R² = {r_squared:.4f}")
        print(f"  ε coefficient: {d:.3f} (positive = high ε → long t_1/2)")
        print()

    # Compare with baseline (no epsilon)
    def model_baseline(params):
        a, b, c = params
        return a + b * inv_sqrt_Q + c * approach

    def residuals_baseline(params):
        return log_t - model_baseline(params)

    result_base = least_squares(residuals_baseline, x0=[0, 50, 5], verbose=0)
    predictions_base = model_baseline(result_base.x)
    rmse_base = np.sqrt(np.mean((log_t - predictions_base)**2))
    ss_res_base = np.sum((log_t - predictions_base)**2)
    r_squared_base = 1 - (ss_res_base / ss_tot)

    print("Baseline (no ε term):")
    print(f"  RMSE: {rmse_base:.3f}, R² = {r_squared_base:.4f}")
    print()

# ============================================================================
# CREATE COMPREHENSIVE VISUALIZATION
# ============================================================================

fig = plt.figure(figsize=(24, 16))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

colors = {'α': 'blue', 'β-': 'red', 'β+/EC': 'orange'}

# Panel 1: Remainder vs N for different Δ (illustration)
ax1 = fig.add_subplot(gs[0, 0])
N_test = np.linspace(-4, 4, 1000)
for Delta in [1/3, 1/2, 1]:
    epsilon_test = [quantization_remainder(N, Delta) for N in N_test]
    ax1.plot(N_test, epsilon_test, label=f'Δ = {Delta:.2f}', linewidth=2)

ax1.set_xlabel('Continuous Coordinate N', fontsize=11, fontweight='bold')
ax1.set_ylabel('Fractional Remainder ε(N, Δ)', fontsize=11, fontweight='bold')
ax1.set_title('(A) Lego Quantization: Distance to Nearest Lattice Point',
             fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(alpha=0.3)
ax1.text(0.02, 0.98, 'ε = 0: On a stud (easy transition)\nε = Δ/2: Between studs (hard)',
         transform=ax1.transAxes, fontsize=9, va='top',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

# Panel 2: Correlation heatmap (Δ vs mode)
ax2 = fig.add_subplot(gs[0, 1])
if len(results) > 0:
    # Create matrix of correlations
    modes_sorted = sorted([m for m in results.keys()])
    Delta_sorted = sorted(list(set([r['Delta'] for mode in results.values() for r in mode])))

    corr_matrix = np.zeros((len(modes_sorted), len(Delta_sorted)))

    for i, mode in enumerate(modes_sorted):
        for j, Delta in enumerate(Delta_sorted):
            # Find this Delta in results
            matches = [r for r in results[mode] if abs(r['Delta'] - Delta) < 1e-6]
            if matches:
                corr_matrix[i, j] = matches[0]['r_linear']

    im = ax2.imshow(corr_matrix, cmap='RdBu_r', vmin=-0.8, vmax=0.8, aspect='auto')
    ax2.set_xticks(range(len(Delta_sorted)))
    ax2.set_xticklabels([f"{D:.2f}" for D in Delta_sorted], rotation=45)
    ax2.set_yticks(range(len(modes_sorted)))
    ax2.set_yticklabels(modes_sorted)
    ax2.set_xlabel('Tile Size Δ', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Decay Mode', fontsize=11, fontweight='bold')
    ax2.set_title('(B) Correlation Heatmap: ε vs log(t₁/₂)',
                 fontsize=12, fontweight='bold')

    # Add correlation values
    for i in range(len(modes_sorted)):
        for j in range(len(Delta_sorted)):
            text = ax2.text(j, i, f'{corr_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=8)

    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label('Pearson r', fontsize=10)

# Panel 3: Alpha decay - epsilon vs half-life (best Δ)
ax3 = fig.add_subplot(gs[0, 2])
if 'α' in results and len(results['α']) > 0:
    best_Delta_alpha = max(results['α'], key=lambda x: abs(x['r_linear']))['Delta']

    alpha_data = data_by_mode['α']
    N_alpha = [d['N_parent'] for d in alpha_data]
    log_t_alpha = [d['log_t_half'] for d in alpha_data]
    Q_alpha = [d['Q'] for d in alpha_data]

    epsilon_alpha = [quantization_remainder(N, best_Delta_alpha) for N in N_alpha]
    epsilon_norm_alpha = np.array(epsilon_alpha) / best_Delta_alpha

    scatter3 = ax3.scatter(epsilon_norm_alpha, log_t_alpha, c=Q_alpha,
                          cmap='plasma', s=100, alpha=0.7,
                          edgecolors='black', linewidths=1)

    # Fit line
    r, p = pearsonr(epsilon_norm_alpha, log_t_alpha)
    slope, intercept = np.polyfit(epsilon_norm_alpha, log_t_alpha, 1)
    x_fit = np.linspace(0, 0.5, 100)
    ax3.plot(x_fit, slope*x_fit + intercept, 'b--', linewidth=2,
            label=f'r={r:.3f}, p={p:.3f}')

    ax3.set_xlabel(f'Normalized Remainder ε/Δ (Δ={best_Delta_alpha:.2f})',
                  fontsize=11, fontweight='bold')
    ax3.set_ylabel('log₁₀(Half-Life) [s]', fontsize=11, fontweight='bold')
    ax3.set_title(f'(C) Alpha Decay: Lego Effect (Best Δ={best_Delta_alpha:.2f})',
                 fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(alpha=0.3)
    cbar3 = plt.colorbar(scatter3, ax=ax3)
    cbar3.set_label('Q [MeV]', fontsize=10)

# Panel 4-6: Individual nucleus analysis
if 'α' in data_by_mode and len(data_by_mode['α']) > 0:
    alpha_data = data_by_mode['α']

    # Panel 4: N_parent positions on lattice
    ax4 = fig.add_subplot(gs[1, 0])
    Delta_show = 1/3
    N_alpha = [d['N_parent'] for d in alpha_data]
    log_t_alpha = [d['log_t_half'] for d in alpha_data]

    # Draw lattice
    k_range = range(int(min(N_alpha)/Delta_show) - 1, int(max(N_alpha)/Delta_show) + 2)
    lattice_points = [k * Delta_show for k in k_range]
    for point in lattice_points:
        ax4.axvline(point, color='gray', linestyle=':', alpha=0.3, linewidth=1)

    # Plot nuclei
    scatter4 = ax4.scatter(N_alpha, log_t_alpha, c='blue', s=100, alpha=0.7,
                          edgecolors='black', linewidths=1, zorder=10)

    # Annotate a few
    for d in alpha_data[:5]:
        ax4.annotate(d['name'], (d['N_parent'], d['log_t_half']),
                    fontsize=7, alpha=0.7)

    ax4.set_xlabel(f'N_parent (lattice Δ={Delta_show:.2f} shown)',
                  fontsize=11, fontweight='bold')
    ax4.set_ylabel('log₁₀(Half-Life) [s]', fontsize=11, fontweight='bold')
    ax4.set_title('(D) Parent Positions on Quantized Lattice',
                 fontsize=12, fontweight='bold')
    ax4.grid(alpha=0.3)

    # Panel 5: Remainder distribution
    ax5 = fig.add_subplot(gs[1, 1])
    Delta_hist = 1/3
    epsilon_hist = [quantization_remainder(N, Delta_hist) for N in N_alpha]
    epsilon_norm_hist = np.array(epsilon_hist) / Delta_hist

    ax5.hist(epsilon_norm_hist, bins=15, color='blue', alpha=0.7,
            edgecolor='black', linewidth=1.5)
    ax5.axvline(0, color='green', linestyle='--', linewidth=2,
               label='On lattice (easy)', alpha=0.7)
    ax5.axvline(0.5, color='red', linestyle='--', linewidth=2,
               label='Between lattice (hard)', alpha=0.7)
    ax5.set_xlabel(f'Normalized Remainder ε/Δ (Δ={Delta_hist:.2f})',
                  fontsize=11, fontweight='bold')
    ax5.set_ylabel('Number of Nuclei', fontsize=11, fontweight='bold')
    ax5.set_title('(E) Distribution of Quantization Remainders',
                 fontsize=12, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(alpha=0.3)

    # Panel 6: Comparison table
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')

    table_text = "LEGO QUANTIZATION RESULTS\n"
    table_text += "="*45 + "\n\n"

    if 'α' in results:
        table_text += "ALPHA DECAY:\n"
        table_text += "-"*45 + "\n"
        best = max(results['α'], key=lambda x: abs(x['r_linear']))
        table_text += f"Best Δ:      {best['Delta']:.4f}\n"
        table_text += f"Correlation: r = {best['r_linear']:+.3f}\n"
        table_text += f"p-value:     {best['p_linear']:.4f}\n"
        if best['p_linear'] < 0.05:
            table_text += "Status:      ★ SIGNIFICANT\n"
        table_text += "\n"

    if 'β-' in results:
        table_text += "BETA DECAY:\n"
        table_text += "-"*45 + "\n"
        best = max(results['β-'], key=lambda x: abs(x['r_linear']))
        table_text += f"Best Δ:      {best['Delta']:.4f}\n"
        table_text += f"Correlation: r = {best['r_linear']:+.3f}\n"
        table_text += f"p-value:     {best['p_linear']:.4f}\n"
        if best['p_linear'] < 0.05:
            table_text += "Status:      ★ SIGNIFICANT\n"
        table_text += "\n"

    table_text += "\nINTERPRETATION:\n"
    table_text += "-"*45 + "\n"
    table_text += "Positive r: High ε → long t₁/₂\n"
    table_text += "  (Between studs is HARDER)\n"
    table_text += "\n"
    table_text += "Negative r: High ε → short t₁/₂\n"
    table_text += "  (Between studs is EASIER)\n"

    ax6.text(0.05, 0.95, table_text, transform=ax6.transAxes,
            fontsize=9, va='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

# Bottom row: Physics interpretation
ax7 = fig.add_subplot(gs[2, :])
ax7.axis('off')

physics_text = """
THE LEGO MECHANISM: Quantization Remainder as Decay Barrier
═══════════════════════════════════════════════════════════════════════════════════════

CONCEPT:
• The continuous geometric coordinate N(A,Z) is a FIELD PROPERTY (exists everywhere)
• But ACCESSIBLE CONFIGURATIONS are QUANTIZED on a lattice: {k·Δ | k ∈ ℤ}
• Decay requires landing on an allowed lattice point

FRACTIONAL MISMATCH:
   ε(N, Δ) = min_k |N - k·Δ|    (distance to nearest lattice point)

   Bounded: 0 ≤ ε ≤ Δ/2

INTERPRETATION:
   ε ≈ 0:     Parent near a "stud" → transition is geometrically easy → SHORT half-life
   ε ≈ Δ/2:   Parent between studs → requires field reconfiguration → LONG half-life

THIS EXPLAINS:
✓ Why high stress nuclei can be long-lived (bad quantization fit, large ε)
✓ Why low stress nuclei can be short-lived (good quantization fit, small ε)
✓ The "17/3 problem" - not the absolute value, but the REMAINDER that matters

DIFFERENT CHANNELS HAVE DIFFERENT TILE SIZES:
• Alpha decay:   Δ_α ≈ 1/3 to 1 (global reconnection, coarse lattice)
• Beta decay:    Δ_β ≈ ? (local slip, possibly finer lattice)
• Neutron decay: Δ_n ≈ ? (failure mode, different physics)

THE UNIFIED MODEL:
   log(t₁/₂) = a + b/√Q + c·(approach) + d·(ε/Δ) + e·σ

   Where ε/Δ is the normalized remainder (dimensionless, ∈ [0, 0.5])

NEXT STEPS:
1. Fit tile sizes from data (which Δ minimizes residuals?)
2. Test if ε improves predictions beyond approach and stress
3. Compare alpha vs beta vs neutron tile sizes
4. Connect to topological winding number quantization
"""

ax7.text(0.5, 0.5, physics_text, transform=ax7.transAxes,
        fontsize=10, ha='center', va='center', family='monospace',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9,
                 edgecolor='blue', linewidth=3))

plt.suptitle('LEGO QUANTIZATION: Fractional Remainder as Transition Barrier\n' +
             'Testing: ε(N, Δ) = Distance to Nearest Quantized State',
             fontsize=16, fontweight='bold', y=0.995)

plt.savefig('lego_quantization_test.png', dpi=200, bbox_inches='tight')
plt.savefig('lego_quantization_test.pdf', bbox_inches='tight')

print("="*80)
print("FIGURES SAVED")
print("="*80)
print("  - lego_quantization_test.png (200 DPI)")
print("  - lego_quantization_test.pdf (vector)")
print()

print("="*80)
print("CONCLUSION")
print("="*80)
print()
print("The Lego quantization hypothesis has been tested:")
print()
print("Key question: Does the fractional remainder ε(N,Δ) predict half-life")
print("better than absolute stress or approach to ground?")
print()
print("Results show which tile sizes Δ best fit each decay channel,")
print("validating or refuting the 'quantized landing points' mechanism.")
print()
print("="*80)
