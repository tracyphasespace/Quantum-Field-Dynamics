#!/usr/bin/env python3
"""
Z-SHIFT CORRECTION TEST - FIX SYSTEMATIC UNDERPREDICTION
===========================================================================
Finding: Mean error = -0.126 charges (systematic underprediction)
         42 near-misses (±1), 29 underpredicted vs 13 overpredicted

Hypothesis: Small energy correction proportional to Z can shift the
minimum upward, converting near-misses to exact matches.

Test three correction types:

1. LINEAR Z TERM:
   E_shift = -k_shift × Z
   Negative coefficient favors higher Z (reduces energy for high Z)

2. E_VAC COEFFICIENT ADJUSTMENT:
   E_vac = (a_disp - Δa) × Z² / A^(1/3)
   Reducing a_disp favors higher Z

3. E_ASYM COEFFICIENT ADJUSTMENT:
   E_asym = (a_sym - Δa_sym) × A × (1-2q)²
   Reducing a_sym reduces neutron-rich penalty

Goal: Find optimal correction that maximizes exact matches without
breaking existing successes.
===========================================================================
"""

import numpy as np
from collections import Counter, defaultdict

# QFD Constants
alpha_fine = 1.0 / 137.036
beta_vacuum = 1.0 / 3.058231
M_proton = 938.272
KAPPA_E = 0.0001
SHIELD_FACTOR = 0.52
DELTA_PAIRING = 11.0

# Derived Constants (baseline)
V_0 = M_proton * (1 - (alpha_fine**2) / beta_vacuum)
beta_nuclear = M_proton * beta_vacuum / 2
E_surface_coeff = beta_nuclear / 15
a_sym_base = (beta_vacuum * M_proton) / 15
a_disp_base = (alpha_fine * 197.327 / 1.2) * SHIELD_FACTOR

def qfd_energy_with_shift(A, Z, shift_type='none', correction=0.0):
    """QFD energy with optional Z-shift correction."""
    N = A - Z
    if Z <= 0 or N < 0:
        return 1e12

    q = Z / A
    lambda_time = KAPPA_E * Z

    # Base terms
    E_volume_coeff = V_0 * (1 - lambda_time / (12 * np.pi))
    E_bulk = E_volume_coeff * A
    E_surf = E_surface_coeff * (A**(2/3))

    # Z-dependent terms (apply correction)
    if shift_type == 'linear':
        # Linear Z penalty (negative correction favors high Z)
        E_asym = a_sym_base * A * ((1 - 2*q)**2)
        E_vac = a_disp_base * (Z**2) / (A**(1/3))
        E_shift = -correction * Z  # Negative favors higher Z
    elif shift_type == 'vac_coeff':
        # Reduce E_vac coefficient (favors high Z)
        E_asym = a_sym_base * A * ((1 - 2*q)**2)
        E_vac = (a_disp_base - correction) * (Z**2) / (A**(1/3))
        E_shift = 0
    elif shift_type == 'asym_coeff':
        # Reduce E_asym coefficient (reduces neutron-rich preference)
        E_asym = (a_sym_base - correction) * A * ((1 - 2*q)**2)
        E_vac = a_disp_base * (Z**2) / (A**(1/3))
        E_shift = 0
    else:  # 'none'
        E_asym = a_sym_base * A * ((1 - 2*q)**2)
        E_vac = a_disp_base * (Z**2) / (A**(1/3))
        E_shift = 0

    # Pairing energy
    E_pair = 0
    if Z % 2 == 0 and N % 2 == 0:
        E_pair = -DELTA_PAIRING / np.sqrt(A)
    elif Z % 2 == 1 and N % 2 == 1:
        E_pair = +DELTA_PAIRING / np.sqrt(A)

    return E_bulk + E_surf + E_asym + E_vac + E_pair + E_shift

def find_stable_Z(A, shift_type='none', correction=0.0):
    """Find Z that minimizes energy with optional correction."""
    best_Z, best_E = 1, qfd_energy_with_shift(A, 1, shift_type, correction)

    for Z in range(1, A):
        E = qfd_energy_with_shift(A, Z, shift_type, correction)
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
print("Z-SHIFT CORRECTION TEST - FIX SYSTEMATIC UNDERPREDICTION")
print("="*95)
print()

# Baseline
print("="*95)
print("BASELINE (NO CORRECTION)")
print("="*95)
print()

correct_baseline = 0
baseline_errors = []

for name, Z_exp, A in test_nuclides:
    Z_pred = find_stable_Z(A, 'none', 0.0)
    if Z_pred == Z_exp:
        correct_baseline += 1
    else:
        baseline_errors.append({
            'name': name,
            'A': A,
            'Z_exp': Z_exp,
            'Z_pred': Z_pred,
            'error': Z_pred - Z_exp,
        })

print(f"Baseline: {correct_baseline}/285 ({100*correct_baseline/285:.1f}%)")
print(f"Mean error: {np.mean([e['error'] for e in baseline_errors]):.3f} charges")
print()

# ============================================================================
# TEST 1: LINEAR Z TERM
# ============================================================================
print("="*95)
print("TEST 1: LINEAR Z TERM (E_shift = -k × Z)")
print("="*95)
print()

print("Testing small linear Z corrections...")
print("Negative k favors higher Z (reduces energy for proton-rich)")
print()

linear_shifts = np.linspace(0.0, 2.0, 21)  # Test 0 to 2 MeV

print(f"{'k (MeV)':<12} {'Correct':<12} {'Success %':<12} {'vs Baseline':<15} {'Mean error':<15} {'Marker'}\"")
print("-"*95)

best_linear_k = 0.0
best_linear_correct = correct_baseline

for k in linear_shifts:
    correct = 0
    errors = []

    for name, Z_exp, A in test_nuclides:
        Z_pred = find_stable_Z(A, 'linear', k)
        if Z_pred == Z_exp:
            correct += 1
        else:
            errors.append(Z_pred - Z_exp)

    delta = correct - correct_baseline
    pct = 100 * correct / 285
    mean_err = np.mean(errors) if errors else 0.0

    marker = ""
    if correct > best_linear_correct:
        marker = "★★★"
        best_linear_correct = correct
        best_linear_k = k
    elif correct > correct_baseline:
        marker = "★"

    print(f"{k:<12.3f} {correct:<12} {pct:<12.1f} {delta:+d}  {mean_err:<+15.3f} {marker:<10}")

print()
print(f"Best linear correction: k = {best_linear_k:.3f} MeV")
print(f"Result: {best_linear_correct}/285 ({100*best_linear_correct/285:.1f}%)")
print(f"Improvement: {best_linear_correct - correct_baseline:+d} matches")
print()

# ============================================================================
# TEST 2: E_VAC COEFFICIENT ADJUSTMENT
# ============================================================================
print("="*95)
print("TEST 2: E_VAC COEFFICIENT ADJUSTMENT")
print("="*95)
print()

print(f"Baseline a_disp = {a_disp_base:.4f} MeV·fm")
print("Testing E_vac = (a_disp - Δa) × Z² / A^(1/3)")
print("Reducing a_disp favors higher Z (less Coulomb penalty)")
print()

vac_corrections = np.linspace(0.0, 0.5, 21)  # Test 0 to 0.5 MeV·fm reduction

print(f"{'Δa (MeV·fm)':<15} {'Correct':<12} {'Success %':<12} {'vs Baseline':<15} {'Marker'}\"")
print("-"*95)

best_vac_delta = 0.0
best_vac_correct = correct_baseline

for delta_a in vac_corrections:
    correct = 0

    for name, Z_exp, A in test_nuclides:
        Z_pred = find_stable_Z(A, 'vac_coeff', delta_a)
        if Z_pred == Z_exp:
            correct += 1

    delta = correct - correct_baseline
    pct = 100 * correct / 285

    marker = ""
    if correct > best_vac_correct:
        marker = "★★★"
        best_vac_correct = correct
        best_vac_delta = delta_a
    elif correct > correct_baseline:
        marker = "★"

    print(f"{delta_a:<15.4f} {correct:<12} {pct:<12.1f} {delta:+d}  {marker:<10}")

print()
print(f"Best E_vac correction: Δa = {best_vac_delta:.4f} MeV·fm")
print(f"  → a_disp_new = {a_disp_base - best_vac_delta:.4f} MeV·fm")
print(f"Result: {best_vac_correct}/285 ({100*best_vac_correct/285:.1f}%)")
print(f"Improvement: {best_vac_correct - correct_baseline:+d} matches")
print()

# ============================================================================
# TEST 3: E_ASYM COEFFICIENT ADJUSTMENT
# ============================================================================
print("="*95)
print("TEST 3: E_ASYM COEFFICIENT ADJUSTMENT")
print("="*95)
print()

print(f"Baseline a_sym = {a_sym_base:.4f} MeV")
print("Testing E_asym = (a_sym - Δa_sym) × A × (1-2q)²")
print("Reducing a_sym reduces penalty for asymmetry (less neutron-rich preference)")
print()

asym_corrections = np.linspace(0.0, 5.0, 21)  # Test 0 to 5 MeV reduction

print(f"{'Δa_sym (MeV)':<15} {'Correct':<12} {'Success %':<12} {'vs Baseline':<15} {'Marker'}\"")
print("-"*95)

best_asym_delta = 0.0
best_asym_correct = correct_baseline

for delta_a_sym in asym_corrections:
    correct = 0

    for name, Z_exp, A in test_nuclides:
        Z_pred = find_stable_Z(A, 'asym_coeff', delta_a_sym)
        if Z_pred == Z_exp:
            correct += 1

    delta = correct - correct_baseline
    pct = 100 * correct / 285

    marker = ""
    if correct > best_asym_correct:
        marker = "★★★"
        best_asym_correct = correct
        best_asym_delta = delta_a_sym
    elif correct > correct_baseline:
        marker = "★"

    print(f"{delta_a_sym:<15.4f} {correct:<12} {pct:<12.1f} {delta:+d}  {marker:<10}")

print()
print(f"Best E_asym correction: Δa_sym = {best_asym_delta:.4f} MeV")
print(f"  → a_sym_new = {a_sym_base - best_asym_delta:.4f} MeV")
print(f"Result: {best_asym_correct}/285 ({100*best_asym_correct/285:.1f}%)")
print(f"Improvement: {best_asym_correct - correct_baseline:+d} matches")
print()

# ============================================================================
# BEST CORRECTION ANALYSIS
# ============================================================================
print("="*95)
print("BEST CORRECTION COMPARISON")
print("="*95)
print()

results = [
    ("Linear Z term", best_linear_k, best_linear_correct, 'linear'),
    ("E_vac coefficient", best_vac_delta, best_vac_correct, 'vac_coeff'),
    ("E_asym coefficient", best_asym_delta, best_asym_correct, 'asym_coeff'),
]

best_overall = max(results, key=lambda x: x[2])

print(f"{'Correction Type':<25} {'Parameter':<20} {'Correct':<12} {'Success %':<12} {'Improvement'}\"")
print("-"*95)

for name, param, correct, _ in results:
    pct = 100 * correct / 285
    delta = correct - correct_baseline
    marker = "★★★" if correct == best_overall[2] else ""

    print(f"{name:<25} {param:<20.4f} {correct:<12} {pct:<12.1f} {delta:+d}  {marker:<10}")

print()
print(f"BEST CORRECTION: {best_overall[0]}")
print(f"  Parameter: {best_overall[1]:.4f}")
print(f"  Result: {best_overall[2]}/285 ({100*best_overall[2]/285:.1f}%)")
print(f"  Improvement: {best_overall[2] - correct_baseline:+d} matches ({100*(best_overall[2] - correct_baseline)/285:+.1f}%)")
print()

# ============================================================================
# ANALYZE WHICH NUCLEI ARE FIXED
# ============================================================================
if best_overall[2] > correct_baseline:
    print("="*95)
    print(f"NUCLEI FIXED BY BEST CORRECTION ({best_overall[0].upper()})")
    print("="*95)
    print()

    best_type = best_overall[3]
    best_param = best_overall[1]

    fixed_nuclei = []
    broken_nuclei = []

    for name, Z_exp, A in test_nuclides:
        Z_pred_baseline = find_stable_Z(A, 'none', 0.0)
        Z_pred_corrected = find_stable_Z(A, best_type, best_param)

        if Z_pred_baseline != Z_exp and Z_pred_corrected == Z_exp:
            fixed_nuclei.append({
                'name': name,
                'A': A,
                'Z_exp': Z_exp,
                'Z_baseline': Z_pred_baseline,
                'error_baseline': Z_pred_baseline - Z_exp,
            })
        elif Z_pred_baseline == Z_exp and Z_pred_corrected != Z_exp:
            broken_nuclei.append({
                'name': name,
                'A': A,
                'Z_exp': Z_exp,
                'Z_corrected': Z_pred_corrected,
            })

    print(f"Nuclei fixed: {len(fixed_nuclei)}")
    print(f"Nuclei broken: {len(broken_nuclei)}")
    print()

    if fixed_nuclei:
        print(f"Fixed nuclei (first 30):")
        print(f"{'Nuclide':<12} {'A':<6} {'Z_exp':<8} {'Z_baseline':<12} {'Error_baseline'}\"")
        print("-"*95)

        for n in fixed_nuclei[:30]:
            print(f"{n['name']:<12} {n['A']:<6} {n['Z_exp']:<8} {n['Z_baseline']:<12} {n['error_baseline']:+d}")

        if len(fixed_nuclei) > 30:
            print(f"\n... and {len(fixed_nuclei) - 30} more")

        print()

        # Pattern analysis
        fixed_underpredicted = len([n for n in fixed_nuclei if n['error_baseline'] < 0])
        fixed_overpredicted = len([n for n in fixed_nuclei if n['error_baseline'] > 0])

        print(f"Pattern in fixed nuclei:")
        print(f"  Was underpredicted (Z too low): {fixed_underpredicted}/{len(fixed_nuclei)}")
        print(f"  Was overpredicted (Z too high): {fixed_overpredicted}/{len(fixed_nuclei)}")
        print()

    if broken_nuclei:
        print(f"Broken nuclei (first 20):")
        for n in broken_nuclei[:20]:
            print(f"  {n['name']}: Z_exp={n['Z_exp']}, Z_corrected={n['Z_corrected']}")

        if len(broken_nuclei) > 20:
            print(f"  ... and {len(broken_nuclei) - 20} more")
        print()

# ============================================================================
# SUMMARY
# ============================================================================
print("="*95)
print("SUMMARY: Z-SHIFT CORRECTION TEST")
print("="*95)
print()

print(f"BASELINE: {correct_baseline}/285 (61.4%)")
print(f"  Mean error: -0.126 charges (systematic underprediction)")
print()

print(f"BEST CORRECTION: {best_overall[0]}")
print(f"  Result: {best_overall[2]}/285 ({100*best_overall[2]/285:.1f}%)")
print(f"  Improvement: {best_overall[2] - correct_baseline:+d} matches")
print()

if best_overall[2] > correct_baseline:
    print(f"★★★ CORRECTION IMPROVES PREDICTIONS!")
    print()
    print(f"Mechanism:")
    if best_overall[3] == 'linear':
        print(f"  • Linear Z term: E_shift = -{best_overall[1]:.3f} × Z MeV")
        print(f"  • Reduces energy for high Z (favors proton-rich)")
        print(f"  • Corrects systematic underprediction")
    elif best_overall[3] == 'vac_coeff':
        print(f"  • Reduced E_vac coefficient: a_disp → {a_disp_base - best_overall[1]:.4f} MeV·fm")
        print(f"  • Less Coulomb/vacuum displacement penalty")
        print(f"  • Allows nuclei to hold more charge")
    elif best_overall[3] == 'asym_coeff':
        print(f"  • Reduced E_asym coefficient: a_sym → {a_sym_base - best_overall[1]:.4f} MeV")
        print(f"  • Less asymmetry penalty")
        print(f"  • Reduces preference for neutron-rich configurations")

    print()

    # Progress toward 90%
    target_90 = int(0.90 * 285)
    print(f"Progress toward 90% target ({target_90}/285):")
    print(f"  Current: {best_overall[2]}/{target_90} ({100*best_overall[2]/target_90:.1f}%)")
    print(f"  Remaining: {target_90 - best_overall[2]} matches needed")
    print()

else:
    print(f"✗ Simple Z-shift corrections do NOT improve predictions")
    print(f"  → The systematic underprediction is not correctable by global shifts")
    print(f"  → Need different physics (not just parameter tuning)")
    print()

print("="*95)
