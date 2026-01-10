#!/usr/bin/env python3
"""
HYBRID EMPIRICAL + ENERGY STRESS APPROACH
===========================================================================
Strategy:
1. Empirical formula gives Z_initial (gets within ±1 for 88% of nuclei)
2. Calculate energy E(Z) for Z_initial - 1, Z_initial, Z_initial + 1
3. Pick Z that MINIMIZES total energy (discrete choice)

Energy stress term:
E_stress = E_asym + E_vac + E_pair
(Just the Z-dependent terms, no need for full Hamiltonian)

This combines:
- Speed of empirical formula (narrow search range)
- Accuracy of energy minimization (discrete corrections)
===========================================================================
"""

import numpy as np
from collections import Counter

# QFD Constants
alpha_fine = 1.0 / 137.036
beta_vacuum = 1.0 / 3.058231
M_proton = 938.272
SHIELD_FACTOR = 0.52
DELTA_PAIRING = 11.0

# Derived
a_sym_base = (beta_vacuum * M_proton) / 15
a_disp = (alpha_fine * 197.327 / 1.2) * SHIELD_FACTOR

def empirical_Z_initial(A):
    """Empirical formula for initial guess."""
    # Use optimized global coefficients
    c1 = 0.8790
    c2 = 0.2584
    c3 = -1.8292

    Z_raw = c1 * (A**(2.0/3.0)) + c2 * A + c3
    return int(round(Z_raw))

def energy_stress(A, Z):
    """
    Calculate Z-dependent energy stress.
    Only includes terms that vary with Z at fixed A.
    """
    N = A - Z
    if Z <= 0 or N < 0:
        return 1e12

    q = Z / A

    # Asymmetry energy (favors q=0.5)
    E_asym = a_sym_base * A * ((1 - 2*q)**2)

    # Vacuum displacement (Coulomb-like, Z^2)
    E_vac = a_disp * (Z**2) / (A**(1/3))

    # Pairing energy (discrete even/odd)
    E_pair = 0
    if Z % 2 == 0 and N % 2 == 0:
        E_pair = -DELTA_PAIRING / np.sqrt(A)
    elif Z % 2 == 1 and N % 2 == 1:
        E_pair = +DELTA_PAIRING / np.sqrt(A)

    return E_asym + E_vac + E_pair

def find_Z_hybrid(A):
    """
    Hybrid approach:
    1. Get Z_initial from empirical formula
    2. Check energy at Z_initial ± 1
    3. Return Z with minimum energy
    """
    Z_initial = empirical_Z_initial(A)

    # Search window: Z_initial ± 2 (to be safe)
    Z_min = max(1, Z_initial - 2)
    Z_max = min(A - 1, Z_initial + 3)  # Must have at least 1 neutron

    # Safety check
    if Z_max < Z_min:
        Z_max = Z_min

    candidates = []
    for Z in range(Z_min, Z_max + 1):
        E = energy_stress(A, Z)
        candidates.append((Z, E))

    # Return Z with minimum energy
    if not candidates:
        return max(1, min(A - 1, Z_initial))

    best_Z = min(candidates, key=lambda x: x[1])[0]
    return best_Z

# Load data
with open('qfd_optimized_suite.py', 'r') as f:
    content = f.read()
start = content.find('test_nuclides = [')
end = content.find(']', start) + 1
test_nuclides = eval(content[start:end].replace('test_nuclides = ', ''))

print("="*95)
print("HYBRID EMPIRICAL + ENERGY STRESS APPROACH")
print("="*95)
print()

print("Strategy:")
print("  1. Empirical formula: Z_initial ≈ 0.879×A^(2/3) + 0.258×A - 1.83")
print("  2. Energy stress: E_stress = E_asym + E_vac + E_pair")
print("  3. Discrete choice: Pick Z ∈ {Z_initial-2, ..., Z_initial+2} with minimum E_stress")
print()

# ============================================================================
# TEST HYBRID APPROACH
# ============================================================================
print("="*95)
print("HYBRID RESULTS")
print("="*95)
print()

correct_hybrid = 0
errors_hybrid = []
predictions = []

for name, Z_exp, A in test_nuclides:
    N_exp = A - Z_exp

    Z_initial = empirical_Z_initial(A)
    Z_hybrid = find_Z_hybrid(A)

    error = Z_hybrid - Z_exp

    if Z_hybrid == Z_exp:
        correct_hybrid += 1

    errors_hybrid.append(error)
    predictions.append({
        'name': name,
        'A': A,
        'Z_exp': Z_exp,
        'Z_initial': Z_initial,
        'Z_hybrid': Z_hybrid,
        'error': error,
        'abs_error': abs(error),
        'mod_4': A % 4,
        'parity': 'even-even' if (Z_exp % 2 == 0 and N_exp % 2 == 0) else
                  'odd-odd' if (Z_exp % 2 == 1 and N_exp % 2 == 1) else 'odd-A',
    })

success_rate = 100 * correct_hybrid / 285

print(f"Hybrid (empirical + energy): {correct_hybrid}/285 ({success_rate:.1f}%)")
print()

# Error statistics
mean_error = np.mean(errors_hybrid)
std_error = np.std(errors_hybrid)

print(f"Error statistics:")
print(f"  Mean error: {mean_error:.3f} charges")
print(f"  Std error:  {std_error:.3f} charges")
print()

# ============================================================================
# ERROR DISTRIBUTION
# ============================================================================
print("="*95)
print("ERROR DISTRIBUTION")
print("="*95)
print()

error_counts = Counter(abs(e) for e in errors_hybrid)

print(f"{'|Error|':<12} {'Count':<12} {'Percentage'}\"")
print("-"*95)

for abs_err in sorted(error_counts.keys())[:8]:
    count = error_counts[abs_err]
    pct = 100 * count / 285
    marker = "★★★" if abs_err == 0 else "★★" if abs_err == 1 else "★" if abs_err == 2 else ""
    print(f"{abs_err:<12} {count:<12} {pct:<12.1f}  {marker}")

print()

within_1 = len([p for p in predictions if p['abs_error'] <= 1])
within_2 = len([p for p in predictions if p['abs_error'] <= 2])

print(f"Within ±1: {within_1}/285 ({100*within_1/285:.1f}%)")
print(f"Within ±2: {within_2}/285 ({100*within_2/285:.1f}%)")
print()

# ============================================================================
# BY A MOD 4
# ============================================================================
print("="*95)
print("SUCCESS BY A MOD 4")
print("="*95)
print()

print(f"{'A mod 4':<12} {'Correct':<12} {'Total':<12} {'Success %':<12} {'QFD baseline'}\"")
print("-"*95)

for mod in range(4):
    mod_preds = [p for p in predictions if p['mod_4'] == mod]
    mod_correct = len([p for p in mod_preds if p['error'] == 0])
    mod_total = len(mod_preds)
    mod_pct = 100 * mod_correct / mod_total if mod_total > 0 else 0

    # QFD baseline
    qfd_baseline = {0: 55.1, 1: 77.4, 2: 59.3, 3: 59.6}
    baseline_str = f"{qfd_baseline[mod]:.1f}%"

    marker = "★★★" if mod_pct > 70 else "★" if mod_pct > 60 else ""

    print(f"{mod:<12} {mod_correct:<12} {mod_total:<12} {mod_pct:<12.1f} {baseline_str:<15} {marker}")

print()

# ============================================================================
# ANALYSIS: HOW OFTEN DOES ENERGY CORRECTION HELP?
# ============================================================================
print("="*95)
print("ENERGY CORRECTION ANALYSIS")
print("="*95)
print()

print("How often does energy minimization change the empirical prediction?")
print()

same_as_empirical = len([p for p in predictions if p['Z_initial'] == p['Z_hybrid']])
corrected = len([p for p in predictions if p['Z_initial'] != p['Z_hybrid']])

print(f"Same as empirical:     {same_as_empirical}/285 ({100*same_as_empirical/285:.1f}%)")
print(f"Energy-corrected:      {corrected}/285 ({100*corrected/285:.1f}%)")
print()

# Of the corrected ones, how many became correct?
corrected_preds = [p for p in predictions if p['Z_initial'] != p['Z_hybrid']]
if corrected_preds:
    corrected_to_right = len([p for p in corrected_preds if p['error'] == 0])
    corrected_to_wrong = len([p for p in corrected_preds if p['error'] != 0])

    print(f"Energy correction results:")
    print(f"  Corrected to RIGHT Z: {corrected_to_right}/{corrected} ({100*corrected_to_right/corrected:.1f}%)")
    print(f"  Corrected to WRONG Z: {corrected_to_wrong}/{corrected} ({100*corrected_to_wrong/corrected:.1f}%)")
    print()

# ============================================================================
# COMPARISON
# ============================================================================
print("="*95)
print("COMPARISON TO OTHER METHODS")
print("="*95)
print()

print(f"{'Method':<50} {'Exact':<12} {'Success %'}\"")
print("-"*95)
print(f"{'Pure QFD (full energy minimization)':<50} {'175':<12} {'61.4%'}")
print(f"{'Empirical formula only':<50} {'137':<12} {'48.1%'}")
print(f"{'Hybrid (empirical + energy stress)':<50} {correct_hybrid:<12} {success_rate:.1f}%")
print()

delta_vs_qfd = correct_hybrid - 175
delta_vs_empirical = correct_hybrid - 137

if correct_hybrid > 175:
    print(f"★★★ HYBRID BEATS QFD BASELINE!")
    print(f"  Gain over QFD: {delta_vs_qfd:+d} matches ({100*delta_vs_qfd/285:+.1f}%)")
elif correct_hybrid == 175:
    print(f"★★ HYBRID MATCHES QFD!")
    print(f"  Same performance with narrower search (empirical gives starting point)")
elif correct_hybrid > 137:
    print(f"★ Hybrid improves over empirical formula")
    print(f"  Gain: {delta_vs_empirical:+d} matches ({100*delta_vs_empirical/285:+.1f}%)")
    print(f"  But still below QFD by {175 - correct_hybrid} matches")
else:
    print(f"Energy stress doesn't help empirical formula")

print()

# ============================================================================
# SAMPLE CORRECTIONS
# ============================================================================
print("="*95)
print("SAMPLE ENERGY CORRECTIONS (first 20 where Z_initial ≠ Z_hybrid)")
print("="*95)
print()

corrected_examples = [p for p in predictions if p['Z_initial'] != p['Z_hybrid']][:20]

if corrected_examples:
    print(f"{'Nuclide':<12} {'A':<6} {'Z_exp':<8} {'Z_initial':<12} {'Z_hybrid':<10} {'Correct?'}\"")
    print("-"*95)

    for p in corrected_examples:
        correct_str = "✓" if p['error'] == 0 else f"✗ ({p['error']:+d})"
        print(f"{p['name']:<12} {p['A']:<6} {p['Z_exp']:<8} {p['Z_initial']:<12} {p['Z_hybrid']:<10} {correct_str}")

    print()

# ============================================================================
# SUMMARY
# ============================================================================
print("="*95)
print("SUMMARY")
print("="*95)
print()

print("Hybrid approach combines:")
print("  • Empirical formula: Fast initial guess (within ±1 for 88% of cases)")
print("  • Energy stress: Discrete correction using E_asym + E_vac + E_pair")
print()

print(f"Results: {correct_hybrid}/285 ({success_rate:.1f}%)")
print(f"  vs Pure QFD: {delta_vs_qfd:+d} matches")
print(f"  vs Empirical only: {delta_vs_empirical:+d} matches")
print()

if correct_hybrid >= 175:
    print("★★★ HYBRID ACHIEVES QFD-LEVEL PERFORMANCE!")
    print("    Energy stress successfully corrects empirical errors")
elif correct_hybrid > 137:
    print("★ Energy stress improves empirical predictions")
    print(f"  Corrected {corrected} nuclei, {corrected_to_right} to correct Z")
else:
    print("Energy stress minimal effect - empirical formula dominates")

print()

# Target
target_90 = int(0.90 * 285)
print(f"Progress toward 90% target ({target_90}/285):")
print(f"  Current: {correct_hybrid}/{target_90} ({100*correct_hybrid/target_90:.1f}%)")
print(f"  Remaining: {target_90 - correct_hybrid} matches needed")
print()

print("="*95)
