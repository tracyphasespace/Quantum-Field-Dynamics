#!/usr/bin/env python3
"""
CONTINUOUS GEOMETRIC EVOLUTION MODEL
===========================================================================
Instead of 5 discrete paths, fit smooth functions:
  c1(A) = polynomial or spline
  c2(A) = polynomial or spline

Test:
1. Various polynomial orders (linear, quadratic, cubic, quartic)
2. Recovery rate with continuous model
3. Residual analysis for remaining failures
4. Determine if additional discrete paths are needed

Goal: Find the minimal continuous model that captures geometry evolution.
===========================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, differential_evolution
from collections import Counter

# QFD Constants
alpha_fine = 1.0 / 137.036
beta_vacuum = 1.0 / 3.043233053
M_proton = 938.272
KAPPA_E = 0.0001
SHIELD_FACTOR = 0.52
DELTA_PAIRING = 11.0

# Derived
V_0 = M_proton * (1 - (alpha_fine**2) / beta_vacuum)
beta_nuclear = M_proton * beta_vacuum / 2
E_surface_coeff = beta_nuclear / 15
a_sym_base = (beta_vacuum * M_proton) / 15
a_disp = (alpha_fine * 197.327 / 1.2) * SHIELD_FACTOR

def qfd_energy_full(A, Z):
    """Full QFD energy."""
    N = A - Z
    if Z <= 0 or N < 0:
        return 1e12

    q = Z / A
    lambda_time = KAPPA_E * Z

    E_volume_coeff = V_0 * (1 - lambda_time / (12 * np.pi))
    E_bulk = E_volume_coeff * A
    E_surf = E_surface_coeff * (A**(2/3))
    E_asym = a_sym_base * A * ((1 - 2*q)**2)
    E_vac = a_disp * (Z**2) / (A**(1/3))

    E_pair = 0
    if Z % 2 == 0 and N % 2 == 0:
        E_pair = -DELTA_PAIRING / np.sqrt(A)
    elif Z % 2 == 1 and N % 2 == 1:
        E_pair = +DELTA_PAIRING / np.sqrt(A)

    return E_bulk + E_surf + E_asym + E_vac + E_pair

def find_stable_Z_qfd(A):
    """Find Z with minimum QFD energy."""
    best_Z, best_E = 1, qfd_energy_full(A, 1)
    for Z in range(1, A):
        E = qfd_energy_full(A, Z)
        if E < best_E:
            best_E, best_Z = E, Z
    return best_Z

def empirical_Z(A, c1, c2, c3):
    """Empirical formula."""
    return c1 * (A**(2/3)) + c2 * A + c3

# Load data
with open('qfd_optimized_suite.py', 'r') as f:
    content = f.read()
start = content.find('test_nuclides = [')
end = content.find(']', start) + 1
test_nuclides = eval(content[start:end].replace('test_nuclides = ', ''))

# Get failures
failures = []
for name, Z_exp, A in test_nuclides:
    Z_qfd = find_stable_Z_qfd(A)
    if Z_qfd != Z_exp:
        failures.append({
            'name': name,
            'A': A,
            'Z_exp': Z_exp,
            'Z_qfd': Z_qfd,
            'error': Z_qfd - Z_exp,
        })

print("="*95)
print("CONTINUOUS GEOMETRIC EVOLUTION MODEL")
print("="*95)
print()

print(f"Total failures: {len(failures)}")
print()

# ============================================================================
# FIT CONTINUOUS c1(A) AND c2(A) FUNCTIONS
# ============================================================================
print("="*95)
print("FIT CONTINUOUS c1(A) AND c2(A) FUNCTIONS")
print("="*95)
print()

# Use data from 5-path analysis as anchors
# From previous analysis:
path_data = [
    {'A_center': 30, 'c1': 1.339, 'c2': 0.168, 'c3': -2.96},  # Light
    {'A_center': 80, 'c1': 0.925, 'c2': 0.267, 'c3': -2.23},  # Med-light
    {'A_center': 120, 'c1': 0.774, 'c2': 0.285, 'c3': -1.11}, # Med-heavy
    {'A_center': 160, 'c1': 0.525, 'c2': 0.309, 'c3': 1.91},  # Heavy
    {'A_center': 210, 'c1': 1.046, 'c2': 0.202, 'c3': 4.72},  # Very heavy
]

A_anchors = np.array([p['A_center'] for p in path_data])
c1_anchors = np.array([p['c1'] for p in path_data])
c2_anchors = np.array([p['c2'] for p in path_data])
c3_anchors = np.array([p['c3'] for p in path_data])

# Test various polynomial orders
print("Testing polynomial fits:")
print()

# Define polynomial functions
def poly_1(A, a0, a1):
    return a0 + a1*A

def poly_2(A, a0, a1, a2):
    return a0 + a1*A + a2*(A**2)

def poly_3(A, a0, a1, a2, a3):
    return a0 + a1*A + a2*(A**2) + a3*(A**3)

def poly_4(A, a0, a1, a2, a3, a4):
    return a0 + a1*A + a2*(A**2) + a3*(A**3) + a4*(A**4)

polynomial_funcs = [
    ('Linear', poly_1, 2),
    ('Quadratic', poly_2, 3),
    ('Cubic', poly_3, 4),
    ('Quartic', poly_4, 5),
]

best_models = {}

for name, poly_func, n_params in polynomial_funcs:
    # Fit c1(A)
    try:
        popt_c1, _ = curve_fit(poly_func, A_anchors, c1_anchors)
        c1_fit = poly_func(A_anchors, *popt_c1)
        c1_rmse = np.sqrt(np.mean((c1_fit - c1_anchors)**2))
    except:
        c1_rmse = 999
        popt_c1 = None
    
    # Fit c2(A)
    try:
        popt_c2, _ = curve_fit(poly_func, A_anchors, c2_anchors)
        c2_fit = poly_func(A_anchors, *popt_c2)
        c2_rmse = np.sqrt(np.mean((c2_fit - c2_anchors)**2))
    except:
        c2_rmse = 999
        popt_c2 = None
    
    # Fit c3(A)
    try:
        popt_c3, _ = curve_fit(poly_func, A_anchors, c3_anchors)
        c3_fit = poly_func(A_anchors, *popt_c3)
        c3_rmse = np.sqrt(np.mean((c3_fit - c3_anchors)**2))
    except:
        c3_rmse = 999
        popt_c3 = None
    
    best_models[name] = {
        'poly_func': poly_func,
        'c1_params': popt_c1,
        'c2_params': popt_c2,
        'c3_params': popt_c3,
        'c1_rmse': c1_rmse,
        'c2_rmse': c2_rmse,
        'c3_rmse': c3_rmse,
    }
    
    print(f"{name:<12} c1 RMSE: {c1_rmse:.4f}   c2 RMSE: {c2_rmse:.4f}   c3 RMSE: {c3_rmse:.4f}")

print()

# Select best model (quadratic seems good for U-shape)
best_model_name = 'Quadratic'
best_model = best_models[best_model_name]

print(f"Selected model: {best_model_name}")
print()

# ============================================================================
# TEST CONTINUOUS MODEL ON FAILURES
# ============================================================================
print("="*95)
print("TEST CONTINUOUS MODEL ON ALL FAILURES")
print("="*95)
print()

poly_func = best_model['poly_func']
c1_params = best_model['c1_params']
c2_params = best_model['c2_params']
c3_params = best_model['c3_params']

recovered_continuous = []
still_failed_continuous = []

for f in failures:
    A = f['A']
    Z_exp = f['Z_exp']
    
    # Get continuous coefficients for this A
    c1_A = poly_func(A, *c1_params)
    c2_A = poly_func(A, *c2_params)
    c3_A = poly_func(A, *c3_params)
    
    # Predict Z
    Z_pred = int(round(empirical_Z(A, c1_A, c2_A, c3_A)))
    
    if Z_pred == Z_exp:
        recovered_continuous.append(f)
    else:
        still_failed_continuous.append({
            **f,
            'Z_continuous': Z_pred,
            'continuous_error': Z_pred - Z_exp,
            'c1_A': c1_A,
            'c2_A': c2_A,
            'c3_A': c3_A,
        })

print(f"Recovered with continuous model: {len(recovered_continuous)}/{len(failures)} ({100*len(recovered_continuous)/len(failures):.1f}%)")
print(f"Still failed: {len(still_failed_continuous)}/{len(failures)} ({100*len(still_failed_continuous)/len(failures):.1f}%)")
print()

# ============================================================================
# COMPARISON: DISCRETE VS CONTINUOUS
# ============================================================================
print("="*95)
print("COMPARISON: DISCRETE PATHS vs CONTINUOUS MODEL")
print("="*95)
print()

print(f"{'Model':<30} {'Recovered':<15} {'Rate'}")
print("-"*95)
print(f"{'5 discrete paths':<30} {'52/110':<15} {'47.3%'}")
print(f"{'Continuous quadratic':<30} {f'{len(recovered_continuous)}/110':<15} {100*len(recovered_continuous)/len(failures):.1f}%")
print()

delta = len(recovered_continuous) - 52

if delta > 0:
    print(f"★★ Continuous model is BETTER by {delta} matches!")
    print(f"   Smooth evolution captures geometry more accurately")
elif delta == 0:
    print(f"→ Continuous model matches discrete paths")
    print(f"   Both capture same underlying evolution")
else:
    print(f"→ Discrete paths are better by {abs(delta)} matches")
    print(f"   May need more discrete pathways")

print()

# ============================================================================
# RESIDUAL ANALYSIS: HOW FAR OFF ARE REMAINING FAILURES?
# ============================================================================
print("="*95)
print("RESIDUAL ANALYSIS: DISTANCE FROM CONTINUOUS PATH")
print("="*95)
print()

# Distance from path = |Z_exp - Z_continuous(A)|
distances = [abs(sf['continuous_error']) for sf in still_failed_continuous]

distance_counts = Counter(distances)

print(f"{'Distance':<12} {'Count':<12} {'Percentage'}")
print("-"*95)

for dist in sorted(distance_counts.keys()):
    count = distance_counts[dist]
    pct = 100 * count / len(still_failed_continuous)
    marker = "★" if dist == 1 else "★★" if dist == 2 else ""
    print(f"{dist:<12} {count:<12} {pct:.1f}%  {marker}")

print()

# Statistics
mean_dist = np.mean(distances)
max_dist = max(distances)

print(f"Mean distance from path: {mean_dist:.2f} charges")
print(f"Max distance from path:  {max_dist} charges")
print()

if mean_dist < 1.5:
    print("★ Most failures are VERY CLOSE to continuous path (<1.5 charge average)")
    print("  → Continuous model captures geometry well")
    print("  → Remaining failures likely discrete quantum effects")
elif mean_dist < 2.5:
    print("★ Failures moderately close to path")
    print("  → May benefit from local corrections or finer paths")
else:
    print("→ Failures far from path")
    print("  → Need additional discrete pathways")

print()

# ============================================================================
# DO WE NEED MORE PATHS?
# ============================================================================
print("="*95)
print("DO WE NEED MORE PATHS? CLUSTERING ANALYSIS")
print("="*95)
print()

# Check if remaining failures cluster in specific regions
print("Remaining failures by mass region:")
print()

regions = [
    ('Light (A<60)', lambda f: f['A'] < 60),
    ('Medium-light (60-100)', lambda f: 60 <= f['A'] < 100),
    ('Medium-heavy (100-140)', lambda f: 100 <= f['A'] < 140),
    ('Heavy (140-180)', lambda f: 140 <= f['A'] < 180),
    ('Very heavy (≥180)', lambda f: f['A'] >= 180),
]

print(f"{'Region':<30} {'Remaining':<12} {'Mean dist':<12} {'Need path?'}")
print("-"*95)

for region_name, region_filter in regions:
    region_failures = [sf for sf in still_failed_continuous if region_filter(sf)]
    
    if len(region_failures) == 0:
        continue
    
    mean_region_dist = np.mean([abs(sf['continuous_error']) for sf in region_failures])
    
    need_path = "★ YES" if mean_region_dist > 2.0 and len(region_failures) > 5 else "→ No"
    
    print(f"{region_name:<30} {len(region_failures):<12} {mean_region_dist:<12.2f} {need_path}")

print()

# Check for systematic bias
over_continuous = len([sf for sf in still_failed_continuous if sf['continuous_error'] > 0])
under_continuous = len([sf for sf in still_failed_continuous if sf['continuous_error'] < 0])

print(f"Error direction:")
print(f"  Overpredictions:  {over_continuous} ({100*over_continuous/len(still_failed_continuous):.1f}%)")
print(f"  Underpredictions: {under_continuous} ({100*under_continuous/len(still_failed_continuous):.1f}%)")
print()

if abs(over_continuous - under_continuous) > 0.3 * len(still_failed_continuous):
    print("★ SYSTEMATIC BIAS detected in remaining failures!")
    print("  → Continuous model may need offset correction")
else:
    print("→ No systematic bias - remaining failures are scattered")

print()

# ============================================================================
# SAMPLE REMAINING FAILURES
# ============================================================================
print("="*95)
print("SAMPLE REMAINING FAILURES (furthest from path)")
print("="*95)
print()

# Sort by distance
still_failed_sorted = sorted(still_failed_continuous, key=lambda x: abs(x['continuous_error']), reverse=True)

print(f"{'Nuclide':<12} {'A':<6} {'Z_exp':<8} {'Z_cont':<10} {'Distance':<10} {'Region'}")
print("-"*95)

for sf in still_failed_sorted[:20]:
    region = 'Light' if sf['A'] < 60 else 'Med-light' if sf['A'] < 100 else 'Med-heavy' if sf['A'] < 140 else 'Heavy' if sf['A'] < 180 else 'V.Heavy'
    print(f"{sf['name']:<12} {sf['A']:<6} {sf['Z_exp']:<8} {sf['Z_continuous']:<10} {abs(sf['continuous_error']):<10} {region}")

print()

# ============================================================================
# VISUALIZE CONTINUOUS MODEL
# ============================================================================
print("="*95)
print("CREATING VISUALIZATION...")
print("="*95)
print()

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: c1(A) and c2(A) evolution
ax1 = axes[0, 0]
A_range = np.linspace(1, 240, 500)
c1_continuous = poly_func(A_range, *c1_params)
c2_continuous = poly_func(A_range, *c2_params)

ax1.plot(A_range, c1_continuous, 'b-', linewidth=2, label='c1(A) - Surface')
ax1.plot(A_range, c2_continuous, 'r-', linewidth=2, label='c2(A) - Volume')
ax1.scatter(A_anchors, c1_anchors, c='blue', s=100, marker='o', edgecolors='k', linewidths=2, label='c1 anchors', zorder=5)
ax1.scatter(A_anchors, c2_anchors, c='red', s=100, marker='s', edgecolors='k', linewidths=2, label='c2 anchors', zorder=5)
ax1.set_xlabel('Mass Number A', fontsize=12)
ax1.set_ylabel('Coefficient Value', fontsize=12)
ax1.set_title('Continuous Geometric Evolution: c1(A) and c2(A)', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Recovered vs remaining failures
ax2 = axes[0, 1]
A_recovered = [f['A'] for f in recovered_continuous]
Z_recovered = [f['Z_exp'] for f in recovered_continuous]
A_failed = [sf['A'] for sf in still_failed_continuous]
Z_failed = [sf['Z_exp'] for sf in still_failed_continuous]

ax2.scatter(A_recovered, Z_recovered, c='green', alpha=0.6, s=40, label='Recovered', zorder=3)
ax2.scatter(A_failed, Z_failed, c='red', alpha=0.6, s=40, label='Still failed', zorder=3)

# Plot continuous path
Z_continuous_path = []
for A in A_range:
    c1_A = poly_func(A, *c1_params)
    c2_A = poly_func(A, *c2_params)
    c3_A = poly_func(A, *c3_params)
    Z_continuous_path.append(empirical_Z(A, c1_A, c2_A, c3_A))

ax2.plot(A_range, Z_continuous_path, 'b--', linewidth=2, alpha=0.7, label='Continuous path')

ax2.set_xlabel('Mass Number A', fontsize=12)
ax2.set_ylabel('Proton Number Z', fontsize=12)
ax2.set_title('Continuous Model: Recovered vs Remaining Failures', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Distance distribution
ax3 = axes[1, 0]
ax3.hist(distances, bins=range(0, max(distances)+2), alpha=0.7, color='orange', edgecolor='black')
ax3.set_xlabel('Distance from Continuous Path (charges)', fontsize=12)
ax3.set_ylabel('Count', fontsize=12)
ax3.set_title('Residual Distribution', fontsize=14, fontweight='bold')
ax3.axvline(x=mean_dist, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_dist:.2f}')
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: c1/c2 ratio evolution
ax4 = axes[1, 1]
ratio_continuous = c1_continuous / c2_continuous
ax4.plot(A_range, ratio_continuous, 'purple', linewidth=2)
ax4.axhline(y=4.0, color='gray', linestyle=':', linewidth=1, label='Ratio = 4 (transition)')
ax4.set_xlabel('Mass Number A', fontsize=12)
ax4.set_ylabel('c1/c2 Ratio', fontsize=12)
ax4.set_title('Envelope/Core Dominance Evolution', fontsize=14, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('continuous_evolution_model.png', dpi=150, bbox_inches='tight')
print("Saved: continuous_evolution_model.png")
print()

# ============================================================================
# SUMMARY
# ============================================================================
print("="*95)
print("SUMMARY: CONTINUOUS EVOLUTION MODEL")
print("="*95)
print()

print(f"Continuous quadratic model:")
print(f"  c1(A) = {c1_params[0]:.4f} + {c1_params[1]:.6f}×A + {c1_params[2]:.9f}×A²")
print(f"  c2(A) = {c2_params[0]:.4f} + {c2_params[1]:.6f}×A + {c2_params[2]:.9f}×A²")
print(f"  c3(A) = {c3_params[0]:.4f} + {c3_params[1]:.6f}×A + {c3_params[2]:.9f}×A²")
print()

print(f"Results:")
print(f"  Recovered: {len(recovered_continuous)}/110 ({100*len(recovered_continuous)/len(failures):.1f}%)")
print(f"  Remaining: {len(still_failed_continuous)}/110 ({100*len(still_failed_continuous)/len(failures):.1f}%)")
print(f"  Mean residual: {mean_dist:.2f} charges")
print()

if len(recovered_continuous) > 52:
    print(f"★★★ CONTINUOUS MODEL SUPERIOR!")
    print(f"    Improved by {len(recovered_continuous) - 52} matches over discrete paths")
    print(f"    Smooth geometric evolution confirmed")
elif len(recovered_continuous) >= 45:
    print(f"★★ CONTINUOUS MODEL COMPARABLE")
    print(f"    Similar performance to discrete paths")
    print(f"    Simpler model with fewer parameters")
else:
    print(f"→ Discrete paths perform better")
    print(f"   Need {52 - len(recovered_continuous)} additional discrete corrections")

print()

if mean_dist < 1.5:
    print("Recommendation: CONTINUOUS MODEL IS SUFFICIENT")
    print(f"  • Mean residual < 1.5 charges")
    print(f"  • Remaining failures likely discrete quantum effects")
    print(f"  • No additional geometric paths needed")
elif mean_dist < 2.5:
    print("Recommendation: CONTINUOUS MODEL + LOCAL CORRECTIONS")
    print(f"  • Add 2-3 discrete paths for outlier regions")
    print(f"  • Focus on high-residual clusters")
else:
    print("Recommendation: NEED MORE DISCRETE PATHS")
    print(f"  • Continuous model insufficient")
    print(f"  • Partition into 10-15 paths")

print()
print("="*95)
