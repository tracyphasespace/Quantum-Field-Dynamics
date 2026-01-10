#!/usr/bin/env python3
"""
GEOMETRIC EVOLUTION ANALYSIS
===========================================================================
Hypothesis: The empirical coefficients (c1, c2) are NOT universal constants.
They may evolve with mass scale as the core/envelope geometry changes.

Current empirical formula:
  Z = c1 × A^(2/3) + c2 × A + c3
  
Where:
  c1 ≈ 0.879  (related to 0.529 user mentioned?)
  c2 ≈ 0.258  (related to 1/3.058 = 0.327?)

Test:
1. Fit coefficients separately for light/medium/heavy mass regions
2. Plot errors vs A to identify systematic trends
3. Check if c1, c2 vary smoothly with mass
4. Interpret in terms of evolving core/envelope structure

NO SHELLS, NO NUCLEONS - Pure geometric soliton scaling.
===========================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
from collections import Counter

# QFD Constants
alpha_fine = 1.0 / 137.036
beta_vacuum = 1.0 / 3.058231
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

# Load data
with open('qfd_optimized_suite.py', 'r') as f:
    content = f.read()
start = content.find('test_nuclides = [')
end = content.find(']', start) + 1
test_nuclides = eval(content[start:end].replace('test_nuclides = ', ''))

print("="*95)
print("GEOMETRIC EVOLUTION ANALYSIS")
print("="*95)
print()

# Get QFD predictions and errors
data = []
for name, Z_exp, A in test_nuclides:
    Z_qfd = find_stable_Z_qfd(A)
    error = Z_qfd - Z_exp
    
    data.append({
        'name': name,
        'A': A,
        'Z_exp': Z_exp,
        'Z_qfd': Z_qfd,
        'error': error,
        'q_exp': Z_exp / A,
        'q_qfd': Z_qfd / A,
    })

# ============================================================================
# 1. ERROR TREND VS MASS
# ============================================================================
print("="*95)
print("1. ERROR TREND VS MASS NUMBER")
print("="*95)
print()

# Bin by A ranges
A_bins = [(1, 20), (20, 40), (40, 60), (60, 80), (80, 100),
          (100, 120), (120, 140), (140, 160), (160, 180), (180, 240)]

print(f"{'A range':<15} {'Count':<8} {'Mean error':<15} {'Std error':<15} {'Trend'}")
print("-"*95)

bin_stats = []
for A_min, A_max in A_bins:
    bin_data = [d for d in data if A_min <= d['A'] < A_max]
    
    if len(bin_data) == 0:
        continue
    
    errors = [d['error'] for d in bin_data]
    mean_err = np.mean(errors)
    std_err = np.std(errors)
    
    bin_stats.append({
        'A_center': (A_min + A_max) / 2,
        'mean_error': mean_err,
        'std_error': std_err,
        'count': len(bin_data)
    })
    
    marker = ""
    if abs(mean_err) > 0.5:
        marker = "★ SYSTEMATIC BIAS"
    
    print(f"{A_min}-{A_max:<10} {len(bin_data):<8} {mean_err:<15.3f} {std_err:<15.3f} {marker}")

print()

# Check for systematic trend
A_centers = [b['A_center'] for b in bin_stats]
mean_errors = [b['mean_error'] for b in bin_stats]

# Linear fit to mean errors
if len(A_centers) > 2:
    poly = np.polyfit(A_centers, mean_errors, 1)
    slope = poly[0]
    
    print(f"Linear trend: mean_error = {slope:.6f} × A {poly[1]:+.3f}")
    
    if abs(slope) > 0.001:
        print(f"★ SYSTEMATIC TREND DETECTED!")
        print(f"  Slope: {slope:.6f} charges per nucleon")
        print(f"  Interpretation: QFD envelope geometry evolves with mass")
    else:
        print(f"→ No significant linear trend")
    
    print()

# ============================================================================
# 2. CHARGE FRACTION q = Z/A TREND
# ============================================================================
print("="*95)
print("2. CHARGE FRACTION EVOLUTION")
print("="*95)
print()

print("Does the 'Valley of Stability' q(A) evolve?")
print()

print(f"{'A range':<15} {'<q_exp>':<12} {'<q_qfd>':<12} {'Δq':<12} {'Interpretation'}")
print("-"*95)

for A_min, A_max in A_bins:
    bin_data = [d for d in data if A_min <= d['A'] < A_max]
    
    if len(bin_data) == 0:
        continue
    
    q_exp_mean = np.mean([d['q_exp'] for d in bin_data])
    q_qfd_mean = np.mean([d['q_qfd'] for d in bin_data])
    delta_q = q_qfd_mean - q_exp_mean
    
    marker = ""
    if abs(delta_q) > 0.01:
        if delta_q > 0:
            marker = "QFD too proton-rich"
        else:
            marker = "QFD too neutron-rich"
    
    print(f"{A_min}-{A_max:<10} {q_exp_mean:<12.4f} {q_qfd_mean:<12.4f} {delta_q:+.4f}   {marker}")

print()

# ============================================================================
# 3. FIT EMPIRICAL COEFFICIENTS BY MASS REGION
# ============================================================================
print("="*95)
print("3. EMPIRICAL COEFFICIENT EVOLUTION")
print("="*95)
print()

print("Do c1, c2 change with mass scale?")
print()

def empirical_Z(A, c1, c2, c3):
    """Empirical formula."""
    return c1 * (A**(2/3)) + c2 * A + c3

def fit_coeffs_for_region(data_subset):
    """Fit c1, c2, c3 to data subset."""
    def objective(coeffs):
        c1, c2, c3 = coeffs
        errors = []
        for d in data_subset:
            Z_pred = int(round(empirical_Z(d['A'], c1, c2, c3)))
            errors.append((Z_pred - d['Z_exp'])**2)
        return np.mean(errors)
    
    # Optimize
    result = differential_evolution(
        objective,
        bounds=[(0.0, 2.0), (0.0, 0.5), (-5.0, 5.0)],
        maxiter=100,
        seed=42,
        workers=1,
    )
    
    return result.x

# Mass regions
regions = [
    ('Light (A<60)', lambda d: d['A'] < 60),
    ('Medium (60≤A<140)', lambda d: 60 <= d['A'] < 140),
    ('Heavy (A≥140)', lambda d: d['A'] >= 140),
    ('ALL', lambda d: True),
]

print(f"{'Region':<25} {'c1':<12} {'c2':<12} {'c3':<12}")
print("-"*95)

region_coeffs = []

for region_name, region_filter in regions:
    region_data = [d for d in data if region_filter(d)]
    
    if len(region_data) < 10:
        continue
    
    c1, c2, c3 = fit_coeffs_for_region(region_data)
    
    region_coeffs.append({
        'name': region_name,
        'c1': c1,
        'c2': c2,
        'c3': c3,
    })
    
    print(f"{region_name:<25} {c1:<12.6f} {c2:<12.6f} {c3:<12.6f}")

print()

# Compare to user's values
print("User's mentioned values:")
print("  c1 related to 0.529?")
print("  c2 related to 1/3.058 = 0.327?")
print()

# Check evolution
if len(region_coeffs) >= 3:
    c1_light = region_coeffs[0]['c1']
    c1_medium = region_coeffs[1]['c1']
    c1_heavy = region_coeffs[2]['c1']
    
    c2_light = region_coeffs[0]['c2']
    c2_medium = region_coeffs[1]['c2']
    c2_heavy = region_coeffs[2]['c2']
    
    print("Coefficient evolution:")
    print(f"  c1: {c1_light:.4f} (light) → {c1_medium:.4f} (medium) → {c1_heavy:.4f} (heavy)")
    print(f"  c2: {c2_light:.4f} (light) → {c2_medium:.4f} (medium) → {c2_heavy:.4f} (heavy)")
    print()
    
    c1_variation = (max(c1_light, c1_medium, c1_heavy) - min(c1_light, c1_medium, c1_heavy)) / c1_light
    c2_variation = (max(c2_light, c2_medium, c2_heavy) - min(c2_light, c2_medium, c2_heavy)) / c2_light
    
    print(f"Relative variation:")
    print(f"  c1: {100*c1_variation:.1f}%")
    print(f"  c2: {100*c2_variation:.1f}%")
    print()
    
    if c1_variation > 0.1:
        print(f"★★ c1 VARIES SIGNIFICANTLY WITH MASS!")
        print(f"   Surface coefficient (A^(2/3) term) evolves")
        print(f"   → Envelope geometry changes with mass scale")
    
    if c2_variation > 0.1:
        print(f"★★ c2 VARIES SIGNIFICANTLY WITH MASS!")
        print(f"   Volume coefficient (A term) evolves")
        print(f"   → Core geometry changes with mass scale")
    
    print()

# ============================================================================
# 4. COMPOUNDING EFFECT - DO COEFFS SCALE SMOOTHLY?
# ============================================================================
print("="*95)
print("4. COMPOUNDING GEOMETRIC EVOLUTION")
print("="*95)
print()

print("Fit coefficients as functions of A:")
print("  c1(A) = c1_0 + c1_1 × A")
print("  c2(A) = c2_0 + c2_1 × A")
print()

# Fit coefficients in sliding windows
window_size = 50
A_values = sorted([d['A'] for d in data])
A_min_data = min(A_values)
A_max_data = max(A_values)

windows = []
for A_center in range(A_min_data + 25, A_max_data - 25, 20):
    window_data = [d for d in data if abs(d['A'] - A_center) <= window_size/2]
    
    if len(window_data) < 15:
        continue
    
    c1, c2, c3 = fit_coeffs_for_region(window_data)
    
    windows.append({
        'A_center': A_center,
        'c1': c1,
        'c2': c2,
        'c3': c3,
        'count': len(window_data),
    })

if len(windows) >= 3:
    print(f"{'A_center':<12} {'c1':<12} {'c2':<12} {'c3':<12}")
    print("-"*95)
    
    for w in windows:
        print(f"{w['A_center']:<12} {w['c1']:<12.6f} {w['c2']:<12.6f} {w['c3']:<12.6f}")
    
    print()
    
    # Fit trends
    A_centers_window = [w['A_center'] for w in windows]
    c1_values = [w['c1'] for w in windows]
    c2_values = [w['c2'] for w in windows]
    
    poly_c1 = np.polyfit(A_centers_window, c1_values, 1)
    poly_c2 = np.polyfit(A_centers_window, c2_values, 1)
    
    print("Coefficient trends with mass:")
    print(f"  c1(A) ≈ {poly_c1[1]:.6f} + {poly_c1[0]:.6f} × A")
    print(f"  c2(A) ≈ {poly_c2[1]:.6f} + {poly_c2[0]:.6f} × A")
    print()
    
    if abs(poly_c1[0]) > 0.0001:
        print(f"★ c1 EVOLVES WITH MASS!")
        print(f"  Slope: {poly_c1[0]:.6f} per nucleon")
        print(f"  Physical interpretation: Envelope curvature changes")
    
    if abs(poly_c2[0]) > 0.0001:
        print(f"★ c2 EVOLVES WITH MASS!")
        print(f"  Slope: {poly_c2[0]:.6f} per nucleon")
        print(f"  Physical interpretation: Core/envelope ratio changes")
    
    print()

# ============================================================================
# 5. GEOMETRIC INTERPRETATION
# ============================================================================
print("="*95)
print("5. GEOMETRIC INTERPRETATION")
print("="*95)
print()

print("Empirical formula: Z = c1 × A^(2/3) + c2 × A + c3")
print()
print("Geometric meaning:")
print("  • A^(2/3) term: Surface/envelope contribution")
print("  • A term:       Volume/core contribution")
print("  • c3:           Normalization offset")
print()

# Get global coefficients
c1_global = region_coeffs[3]['c1']  # ALL
c2_global = region_coeffs[3]['c2']
c3_global = region_coeffs[3]['c3']

print(f"Global coefficients (all nuclei):")
print(f"  c1 = {c1_global:.6f}")
print(f"  c2 = {c2_global:.6f}")
print(f"  c3 = {c3_global:.6f}")
print()

# Ratio interpretation
ratio_surface_to_volume = c1_global / c2_global if c2_global > 0 else 0

print(f"Surface/volume ratio (c1/c2): {ratio_surface_to_volume:.3f}")
print()

# Compare to user's values
print("Comparison to user's mentioned values:")
print(f"  c1 = {c1_global:.6f}  vs  0.529  (ratio: {c1_global/0.529:.3f})")
print(f"  c2 = {c2_global:.6f}  vs  1/3.058 = 0.327  (ratio: {c2_global/0.327:.3f})")
print()

if abs(c1_global - 0.529) / 0.529 > 0.1:
    print(f"★ c1 differs from 0.529 by {100*abs(c1_global - 0.529)/0.529:.1f}%")

if abs(c2_global - 0.327) / 0.327 > 0.1:
    print(f"★ c2 differs from 1/β by {100*abs(c2_global - 0.327)/0.327:.1f}%")
elif abs(c2_global - 0.327) / 0.327 < 0.05:
    print(f"★★★ c2 ≈ 1/β within {100*abs(c2_global - 0.327)/0.327:.1f}%!")
    print(f"    Confirms β = {1/c2_global:.6f} ≈ 3.058")

print()

# ============================================================================
# SUMMARY
# ============================================================================
print("="*95)
print("SUMMARY: GEOMETRIC EVOLUTION OF CORE/ENVELOPE")
print("="*95)
print()

print("Key findings:")
print()

# Trend
if abs(slope) > 0.001:
    print(f"1. ★ Systematic error trend: {slope:.6f} charges/nucleon")
    print(f"   → QFD geometry evolves with mass scale")
else:
    print(f"1. → No strong linear trend in errors")

print()

# Coefficient evolution
if c1_variation > 0.1 or c2_variation > 0.1:
    print(f"2. ★★ Coefficients vary with mass:")
    print(f"   c1: {100*c1_variation:.1f}% variation")
    print(f"   c2: {100*c2_variation:.1f}% variation")
    print(f"   → Core/envelope geometry NOT universal")
else:
    print(f"2. → Coefficients relatively stable across mass ranges")

print()

# Interpretation
print(f"3. Physical interpretation:")
print(f"   c1 (surface): Controls envelope curvature")
print(f"   c2 (volume):  Controls core/envelope ratio")
print(f"   c3 (offset):  Normalization/ground state")
print()

print(f"4. Connection to QFD parameters:")
if abs(c2_global - 1/beta_vacuum) / (1/beta_vacuum) < 0.1:
    print(f"   ★★★ c2 ≈ 1/β_vacuum = {1/beta_vacuum:.6f}")
    print(f"       Validates β = {beta_vacuum:.6f}")
else:
    print(f"   c2 = {c2_global:.6f} vs 1/β = {1/beta_vacuum:.6f}")
    print(f"   Different by {100*abs(c2_global - 1/beta_vacuum)/(1/beta_vacuum):.1f}%")

print()
print("="*95)
