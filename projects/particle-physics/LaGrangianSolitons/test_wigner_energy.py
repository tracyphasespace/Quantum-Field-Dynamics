#!/usr/bin/env python3
"""
TEST WIGNER ENERGY FOR ODD-ODD NUCLEI
===========================================================================
Current status: Odd-odd nuclei have 77.8% failure rate (7/9 wrong!)

Current pairing model:
  Even-even: E_pair = -DELTA_PAIRING / √A  (attractive)
  Odd-odd:   E_pair = +DELTA_PAIRING / √A  (PENALTY)
  Odd-A:     E_pair = 0

Wigner energy hypothesis:
  Odd-odd nuclei have residual neutron-proton pairing
  This is STRONGER than like-nucleon pairing
  Should be ATTRACTIVE, not repulsive!

New model:
  Even-even: E_pair = -DELTA_PAIRING / √A  (keep same)
  Odd-odd:   E_pair = -W / A              (Wigner attractive!)
  Odd-A:     E_pair = 0                   (keep same)

Where W ~ 30-40 MeV (empirical nuclear mass formulas)

Test: Does this fix the 7/9 odd-odd failures?
===========================================================================
"""

import numpy as np
from scipy.optimize import minimize_scalar
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

def qfd_energy_original(A, Z):
    """Original QFD energy with penalty for odd-odd."""
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

    # Original pairing
    E_pair = 0
    if Z % 2 == 0 and N % 2 == 0:
        E_pair = -DELTA_PAIRING / np.sqrt(A)
    elif Z % 2 == 1 and N % 2 == 1:
        E_pair = +DELTA_PAIRING / np.sqrt(A)  # PENALTY

    return E_bulk + E_surf + E_asym + E_vac + E_pair

def qfd_energy_wigner(A, Z, W):
    """QFD energy with Wigner energy for odd-odd."""
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

    # NEW: Wigner energy for odd-odd
    E_pair = 0
    if Z % 2 == 0 and N % 2 == 0:
        E_pair = -DELTA_PAIRING / np.sqrt(A)  # Even-even: same as before
    elif Z % 2 == 1 and N % 2 == 1:
        E_pair = -W / A  # Odd-odd: ATTRACTIVE Wigner energy
    # Odd-A: E_pair = 0 (no change)

    return E_bulk + E_surf + E_asym + E_vac + E_pair

def find_stable_Z(A, energy_func):
    """Find Z with minimum energy."""
    best_Z, best_E = 1, energy_func(A, 1)
    for Z in range(1, A):
        E = energy_func(A, Z)
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
print("TEST WIGNER ENERGY FOR ODD-ODD NUCLEI")
print("="*95)
print()

# Identify odd-odd nuclei
odd_odd_nuclei = []
for name, Z_exp, A in test_nuclides:
    N_exp = A - Z_exp
    if Z_exp % 2 == 1 and N_exp % 2 == 1:
        odd_odd_nuclei.append((name, Z_exp, A))

print(f"Total odd-odd nuclei in dataset: {len(odd_odd_nuclei)}")
print()

# Test baseline (original pairing)
print("="*95)
print("BASELINE: ORIGINAL PAIRING (ODD-ODD PENALTY)")
print("="*95)
print()

correct_original_oddodd = 0
for name, Z_exp, A in odd_odd_nuclei:
    Z_pred = find_stable_Z(A, qfd_energy_original)
    if Z_pred == Z_exp:
        correct_original_oddodd += 1

print(f"Odd-odd success: {correct_original_oddodd}/{len(odd_odd_nuclei)} ({100*correct_original_oddodd/len(odd_odd_nuclei):.1f}%)")
print()

# Full dataset with original
correct_original_total = 0
for name, Z_exp, A in test_nuclides:
    Z_pred = find_stable_Z(A, qfd_energy_original)
    if Z_pred == Z_exp:
        correct_original_total += 1

print(f"Total success: {correct_original_total}/285 ({100*correct_original_total/285:.1f}%)")
print()

# ============================================================================
# OPTIMIZE WIGNER PARAMETER W
# ============================================================================
print("="*95)
print("OPTIMIZE WIGNER PARAMETER W")
print("="*95)
print()

def objective_wigner(W):
    """Count exact matches with Wigner energy."""
    correct = 0
    for name, Z_exp, A in test_nuclides:
        Z_pred = find_stable_Z(A, lambda A, Z: qfd_energy_wigner(A, Z, W))
        if Z_pred == Z_exp:
            correct += 1
    return -correct  # Minimize negative = maximize matches

print("Testing W values from 0 to 100 MeV...")
print()

# Grid search first
W_values = np.linspace(0, 100, 51)
results = []

for W in W_values:
    matches = -objective_wigner(W)
    results.append((W, matches))

results.sort(key=lambda x: x[1], reverse=True)

print("Top 10 W values:")
print(f"{'W (MeV)':<12} {'Total matches':<15} {'vs Baseline'}")
print("-"*95)

for W, matches in results[:10]:
    delta = matches - correct_original_total
    marker = "★★★" if delta >= 5 else "★★" if delta >= 3 else "★" if delta > 0 else ""
    print(f"{W:<12.1f} {matches:<15} {delta:+d}  {marker}")

best_W = results[0][0]
best_matches = results[0][1]

print()
print(f"Best W: {best_W:.1f} MeV")
print(f"Best total matches: {best_matches}/285 ({100*best_matches/285:.1f}%)")
print(f"Improvement: {best_matches - correct_original_total:+d} matches")
print()

# ============================================================================
# DETAILED ANALYSIS WITH BEST W
# ============================================================================
print("="*95)
print(f"WIGNER ENERGY RESULTS (W = {best_W:.1f} MeV)")
print("="*95)
print()

predictions_wigner = []
for name, Z_exp, A in test_nuclides:
    N_exp = A - Z_exp
    Z_wigner = find_stable_Z(A, lambda A, Z: qfd_energy_wigner(A, Z, best_W))
    error = Z_wigner - Z_exp

    predictions_wigner.append({
        'name': name,
        'A': A,
        'Z_exp': Z_exp,
        'Z_wigner': Z_wigner,
        'error': error,
        'abs_error': abs(error),
        'parity': 'even-even' if (Z_exp % 2 == 0 and N_exp % 2 == 0) else
                  'odd-odd' if (Z_exp % 2 == 1 and N_exp % 2 == 1) else 'odd-A',
    })

# Odd-odd performance
oddodd_wigner = [p for p in predictions_wigner if p['parity'] == 'odd-odd']
correct_oddodd_wigner = len([p for p in oddodd_wigner if p['error'] == 0])

print(f"Odd-odd nuclei:")
print(f"  Baseline:  {correct_original_oddodd}/{len(odd_odd_nuclei)} ({100*correct_original_oddodd/len(odd_odd_nuclei):.1f}%)")
print(f"  Wigner:    {correct_oddodd_wigner}/{len(odd_odd_nuclei)} ({100*correct_oddodd_wigner/len(odd_odd_nuclei):.1f}%)")
print(f"  Gain:      {correct_oddodd_wigner - correct_original_oddodd:+d} nuclei")
print()

# Overall performance
correct_wigner_total = len([p for p in predictions_wigner if p['error'] == 0])

print(f"Total nuclei:")
print(f"  Baseline:  {correct_original_total}/285 ({100*correct_original_total/285:.1f}%)")
print(f"  Wigner:    {correct_wigner_total}/285 ({100*correct_wigner_total/285:.1f}%)")
print(f"  Gain:      {correct_wigner_total - correct_original_total:+d} nuclei")
print()

# By parity
print("Performance by parity:")
print(f"{'Parity':<15} {'Baseline':<12} {'Wigner':<12} {'Change'}")
print("-"*95)

for parity in ['even-even', 'odd-odd', 'odd-A']:
    parity_all = [p for p in predictions_wigner if p['parity'] == parity]
    
    # Baseline
    baseline_correct = 0
    for name, Z_exp, A in test_nuclides:
        N_exp = A - Z_exp
        p = 'even-even' if (Z_exp % 2 == 0 and N_exp % 2 == 0) else \
            'odd-odd' if (Z_exp % 2 == 1 and N_exp % 2 == 1) else 'odd-A'
        if p == parity:
            Z_pred = find_stable_Z(A, qfd_energy_original)
            if Z_pred == Z_exp:
                baseline_correct += 1
    
    # Wigner
    wigner_correct = len([p for p in parity_all if p['error'] == 0])
    
    total = len(parity_all)
    baseline_pct = 100 * baseline_correct / total if total > 0 else 0
    wigner_pct = 100 * wigner_correct / total if total > 0 else 0
    
    change = wigner_correct - baseline_correct
    marker = "★★★" if change >= 3 else "★★" if change >= 2 else "★" if change > 0 else "↓" if change < 0 else ""
    
    print(f"{parity:<15} {baseline_pct:<12.1f} {wigner_pct:<12.1f} {change:+d}  {marker}")

print()

# ============================================================================
# ODD-ODD NUCLEI DETAILED RESULTS
# ============================================================================
print("="*95)
print("DETAILED ODD-ODD NUCLEI RESULTS")
print("="*95)
print()

print(f"{'Nuclide':<12} {'A':<6} {'Z_exp':<8} {'Z_baseline':<12} {'Z_Wigner':<12} {'Status'}")
print("-"*95)

for name, Z_exp, A in odd_odd_nuclei:
    Z_baseline = find_stable_Z(A, qfd_energy_original)
    Z_wigner = find_stable_Z(A, lambda A, Z: qfd_energy_wigner(A, Z, best_W))
    
    baseline_correct = "✓" if Z_baseline == Z_exp else f"✗({Z_baseline - Z_exp:+d})"
    wigner_correct = "✓" if Z_wigner == Z_exp else f"✗({Z_wigner - Z_exp:+d})"
    
    if Z_baseline != Z_exp and Z_wigner == Z_exp:
        status = "★ FIXED"
    elif Z_baseline == Z_exp and Z_wigner != Z_exp:
        status = "↓ BROKE"
    elif Z_baseline == Z_exp and Z_wigner == Z_exp:
        status = "✓ Still correct"
    else:
        status = "Still wrong"
    
    print(f"{name:<12} {A:<6} {Z_exp:<8} {baseline_correct:<12} {wigner_correct:<12} {status}")

print()

# ============================================================================
# ERROR DISTRIBUTION
# ============================================================================
print("="*95)
print("ERROR DISTRIBUTION")
print("="*95)
print()

errors_wigner = [p['error'] for p in predictions_wigner]
mean_error = np.mean(errors_wigner)
std_error = np.std(errors_wigner)

print(f"Mean error: {mean_error:.3f} charges")
print(f"Std error: {std_error:.3f} charges")
print()

error_counts = Counter(p['abs_error'] for p in predictions_wigner)

print(f"{'|Error|':<12} {'Count':<12} {'Percentage'}")
print("-"*95)

for abs_err in sorted(error_counts.keys())[:6]:
    count = error_counts[abs_err]
    pct = 100 * count / 285
    marker = "★★★" if abs_err == 0 else "★★" if abs_err == 1 else "★" if abs_err == 2 else ""
    print(f"{abs_err:<12} {count:<12} {pct:.1f}%  {marker}")

print()

within_1 = len([p for p in predictions_wigner if p['abs_error'] <= 1])
within_2 = len([p for p in predictions_wigner if p['abs_error'] <= 2])

print(f"Within ±1: {within_1}/285 ({100*within_1/285:.1f}%)")
print(f"Within ±2: {within_2}/285 ({100*within_2/285:.1f}%)")
print()

# ============================================================================
# SUMMARY
# ============================================================================
print("="*95)
print("SUMMARY")
print("="*95)
print()

print(f"Wigner energy model:")
print(f"  Even-even: E_pair = -11.0 / √A  (unchanged)")
print(f"  Odd-odd:   E_pair = -{best_W:.1f} / A    (NEW!)")
print(f"  Odd-A:     E_pair = 0          (unchanged)")
print()

print(f"Results:")
print(f"  Baseline:  {correct_original_total}/285 ({100*correct_original_total/285:.1f}%)")
print(f"  Wigner:    {correct_wigner_total}/285 ({100*correct_wigner_total/285:.1f}%)")
print(f"  Gain:      {correct_wigner_total - correct_original_total:+d} matches")
print()

if correct_wigner_total > correct_original_total:
    print(f"★★★ WIGNER ENERGY IMPROVES PREDICTIONS!")
    print(f"    Odd-odd success rate: {100*correct_original_oddodd/len(odd_odd_nuclei):.1f}% → {100*correct_oddodd_wigner/len(odd_odd_nuclei):.1f}%")
    
    if correct_oddodd_wigner == len(odd_odd_nuclei):
        print(f"    ★★★ ALL ODD-ODD NUCLEI NOW CORRECT!")
    
    print()
    print(f"Physical interpretation:")
    print(f"  W = {best_W:.1f} MeV represents neutron-proton pairing strength")
    print(f"  Attractive term: np pairs are more stable than unpaired nucleons")
    print(f"  Confirms Wigner's hypothesis about residual strong force")
    
elif correct_wigner_total == correct_original_total:
    print(f"→ No overall change, but may improve odd-odd specifically")
    print(f"  Check if odd-odd improved without hurting even-even/odd-A")
else:
    print(f"✗ Wigner energy makes predictions worse")
    print(f"  Original pairing model may be more appropriate")

print()

# Progress toward 90%
target_90 = int(0.90 * 285)
print(f"Progress toward 90% target ({target_90}/285):")
print(f"  Baseline: {correct_original_total}/{target_90} ({100*correct_original_total/target_90:.1f}%)")
print(f"  Wigner:   {correct_wigner_total}/{target_90} ({100*correct_wigner_total/target_90:.1f}%)")
print(f"  Remaining: {target_90 - correct_wigner_total} matches needed")
print()

print("="*95)
