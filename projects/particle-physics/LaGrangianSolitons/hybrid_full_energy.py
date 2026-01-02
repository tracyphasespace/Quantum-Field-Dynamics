#!/usr/bin/env python3
"""
HYBRID WITH FULL ENERGY - FIX SYSTEMATIC UNDERPREDICTION
===========================================================================
Previous hybrid used E_stress = E_asym + E_vac + E_pair (Z-terms only)
Result: 167/285 (58.6%), systematically underpredicted heavy nuclei

Fix: Use FULL energy including E_bulk + E_surf
E_bulk has weak Z-dependence through lambda_time = KAPPA_E × Z
This should shift minimum upward by 1-2 charges for heavy nuclei

Expected: Match pure QFD (175/285 = 61.4%)
===========================================================================
"""

import numpy as np
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
    """Full QFD energy - complete Hamiltonian."""
    N = A - Z
    if Z <= 0 or N < 0:
        return 1e12

    q = Z / A
    lambda_time = KAPPA_E * Z  # ← Z-dependence in E_bulk!

    E_volume_coeff = V_0 * (1 - lambda_time / (12 * np.pi))
    E_bulk = E_volume_coeff * A
    E_surf = E_surface_coeff * (A**(2/3))
    E_asym = a_sym_base * A * ((1 - 2*q)**2)
    E_vac = a_disp * (Z**2) / (A**(1/3))

    # Pairing energy
    E_pair = 0
    if Z % 2 == 0 and N % 2 == 0:
        E_pair = -DELTA_PAIRING / np.sqrt(A)
    elif Z % 2 == 1 and N % 2 == 1:
        E_pair = +DELTA_PAIRING / np.sqrt(A)

    return E_bulk + E_surf + E_asym + E_vac + E_pair

def empirical_Z_initial(A):
    """Empirical formula for initial guess."""
    c1 = 0.8790
    c2 = 0.2584
    c3 = -1.8292
    Z_raw = c1 * (A**(2.0/3.0)) + c2 * A + c3
    return int(round(Z_raw))

def find_stable_Z_pure(A):
    """Pure QFD: search all Z."""
    best_Z, best_E = 1, qfd_energy_full(A, 1)
    for Z in range(1, A):
        E = qfd_energy_full(A, Z)
        if E < best_E:
            best_E, best_Z = E, Z
    return best_Z

def find_Z_hybrid_full(A):
    """
    Hybrid with FULL energy:
    1. Empirical formula gives Z_initial
    2. Search Z_initial ± 2 using FULL QFD energy
    3. Return Z with minimum E_full
    """
    Z_initial = empirical_Z_initial(A)
    Z_min = max(1, Z_initial - 2)
    Z_max = min(A - 1, Z_initial + 3)

    if Z_max < Z_min:
        Z_max = Z_min

    candidates = []
    for Z in range(Z_min, Z_max + 1):
        E = qfd_energy_full(A, Z)  # ← FULL energy now!
        candidates.append((Z, E))

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
print("HYBRID WITH FULL ENERGY - FIX SYSTEMATIC UNDERPREDICTION")
print("="*95)
print()

print("Strategy:")
print("  1. Empirical formula: Z_initial ≈ 0.879×A^(2/3) + 0.258×A - 1.83")
print("  2. FULL energy: E_total = E_bulk + E_surf + E_asym + E_vac + E_pair")
print("  3. Discrete choice: Pick Z ∈ {Z_initial-2, ..., Z_initial+2} with minimum E_total")
print()
print("Key difference from previous hybrid:")
print("  • Previous: Used E_stress (Z-terms only) → systematic underprediction")
print("  • Now: Uses E_total (includes E_bulk with lambda_time × Z)")
print()

# ============================================================================
# TEST HYBRID WITH FULL ENERGY
# ============================================================================
print("="*95)
print("HYBRID FULL ENERGY RESULTS")
print("="*95)
print()

correct_hybrid_full = 0
errors_hybrid_full = []
predictions = []

for name, Z_exp, A in test_nuclides:
    N_exp = A - Z_exp

    Z_hybrid_full = find_Z_hybrid_full(A)
    error = Z_hybrid_full - Z_exp

    if Z_hybrid_full == Z_exp:
        correct_hybrid_full += 1

    errors_hybrid_full.append(error)
    predictions.append({
        'name': name,
        'A': A,
        'Z_exp': Z_exp,
        'Z_hybrid_full': Z_hybrid_full,
        'error': error,
        'abs_error': abs(error),
        'mod_4': A % 4,
        'parity': 'even-even' if (Z_exp % 2 == 0 and N_exp % 2 == 0) else
                  'odd-odd' if (Z_exp % 2 == 1 and N_exp % 2 == 1) else 'odd-A',
    })

success_rate = 100 * correct_hybrid_full / 285

print(f"Hybrid (full energy): {correct_hybrid_full}/285 ({success_rate:.1f}%)")
print()

# Error statistics
mean_error = np.mean(errors_hybrid_full)
std_error = np.std(errors_hybrid_full)

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

error_counts = Counter(abs(e) for e in errors_hybrid_full)

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
# COMPARISON
# ============================================================================
print("="*95)
print("COMPARISON TO OTHER METHODS")
print("="*95)
print()

print(f"{'Method':<50} {'Exact':<12} {'Success %':<12} {'vs QFD'}\"")
print("-"*95)
print(f"{'Pure QFD (search all Z)':<50} {'175':<12} {'61.4%':<12} {'—'}")
print(f"{'Hybrid with E_stress only':<50} {'167':<12} {'58.6%':<12} {'-8'}")
print(f"{'Hybrid with FULL energy':<50} {correct_hybrid_full:<12} {success_rate:<12.1f} {correct_hybrid_full - 175:+d}")
print()

delta_vs_qfd = correct_hybrid_full - 175
delta_vs_hybrid_stress = correct_hybrid_full - 167

if correct_hybrid_full == 175:
    print(f"★★★ HYBRID FULL MATCHES PURE QFD EXACTLY!")
    print(f"  • Same 175/285 (61.4%) accuracy")
    print(f"  • Empirical formula narrows search (±2 instead of full range)")
    print(f"  • Proves E_bulk Z-dependence (lambda_time) is critical!")
elif correct_hybrid_full > 175:
    print(f"★★★★ HYBRID FULL BEATS PURE QFD!")
    print(f"  Gain: {delta_vs_qfd:+d} matches ({100*delta_vs_qfd/285:+.1f}%)")
    print(f"  Unexpected! Need to investigate why...")
elif correct_hybrid_full > 167:
    print(f"★★ Hybrid full improves over E_stress version")
    print(f"  Gain over E_stress: {delta_vs_hybrid_stress:+d} matches")
    print(f"  Still below QFD by {175 - correct_hybrid_full} matches")
    print(f"  → E_bulk helps but doesn't fully fix underprediction")
else:
    print(f"✗ Full energy doesn't help")
    print(f"  Still at {correct_hybrid_full}/285")

print()

# ============================================================================
# CHECK: DID WE FIX THE 25 UNDERPREDICTION FAILURES?
# ============================================================================
print("="*95)
print("DID WE FIX THE 25 SYSTEMATIC UNDERPREDICTIONS?")
print("="*95)
print()

# Known failures from previous analysis
known_failures = [
    ('Ge-70', 70, 32), ('Zr-91', 91, 40), ('Ru-100', 100, 44),
    ('Ag-109', 109, 47), ('Sn-117', 117, 50), ('Xe-126', 126, 54),
    ('Pr-141', 141, 59), ('Nd-142', 142, 60), ('Sm-148', 148, 62),
    ('Sm-149', 149, 62), ('Er-164', 164, 68), ('Yb-170', 170, 70),
    ('Yb-171', 171, 70), ('Hf-176', 176, 72), ('Hf-177', 177, 72),
    ('W-183', 183, 74), ('Re-185', 185, 75), ('Ir-191', 191, 77),
    ('Pt-192', 192, 78), ('Au-197', 197, 79), ('Hg-198', 198, 80),
    ('Tl-203', 203, 81), ('Pb-204', 204, 82), ('Bi-209', 209, 83),
    ('U-235', 235, 92),
]

fixed_count = 0
still_wrong = []

for name, A, Z_exp in known_failures:
    Z_hybrid_full = find_Z_hybrid_full(A)

    if Z_hybrid_full == Z_exp:
        fixed_count += 1
    else:
        still_wrong.append((name, A, Z_exp, Z_hybrid_full, Z_hybrid_full - Z_exp))

print(f"Previously failed with E_stress: 25 nuclei")
print(f"Now correct with full energy:   {fixed_count}/25 ({100*fixed_count/25:.1f}%)")
print(f"Still wrong:                    {len(still_wrong)}/25")
print()

if still_wrong:
    print(f"Still failing (first 10):")
    print(f"{'Nuclide':<12} {'A':<6} {'Z_exp':<8} {'Z_hybrid':<10} {'Error'}\"")
    print("-"*95)

    for name, A, Z_exp, Z_hybrid, error in still_wrong[:10]:
        print(f"{name:<12} {A:<6} {Z_exp:<8} {Z_hybrid:<10} {error:+d}")

    if len(still_wrong) > 10:
        print(f"... and {len(still_wrong) - 10} more")
    print()

# ============================================================================
# SUMMARY
# ============================================================================
print("="*95)
print("SUMMARY")
print("="*95)
print()

print(f"Hybrid with full energy: {correct_hybrid_full}/285 ({success_rate:.1f}%)")
print(f"  vs Pure QFD: {delta_vs_qfd:+d} matches")
print(f"  vs Hybrid (E_stress): {delta_vs_hybrid_stress:+d} matches")
print()

print("Key insight:")
print("  E_bulk = V_0 × (1 - KAPPA_E×Z/(12π)) × A")
print("  ↑ Weak Z-dependence through lambda_time")
print("  → Shifts energy minimum upward for heavy nuclei")
print("  → Fixes systematic underprediction")
print()

if correct_hybrid_full >= 175:
    print("★★★ SUCCESS! Hybrid with full energy matches/beats pure QFD")
    print("    Empirical formula + local search is sufficient!")
elif fixed_count > 15:
    print(f"★★ PARTIAL SUCCESS: Fixed {fixed_count}/25 underprediction failures")
    print(f"   Full energy helps but not completely")
else:
    print(f"✗ Full energy doesn't significantly help")
    print(f"   Only fixed {fixed_count}/25 failures")

print()

# Progress toward 90%
target_90 = int(0.90 * 285)
print(f"Progress toward 90% target ({target_90}/285):")
print(f"  Current: {correct_hybrid_full}/{target_90} ({100*correct_hybrid_full/target_90:.1f}%)")
print(f"  Remaining: {target_90 - correct_hybrid_full} matches needed")
print()

print("="*95)
