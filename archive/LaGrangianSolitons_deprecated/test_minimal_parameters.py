#!/usr/bin/env python3
"""
MINIMAL PARAMETER TEST - FIND THE ABSOLUTE MINIMUM
===========================================================================
We've removed lambda_time_0 = 0.42 (zero effect).

Now test removing EVERY other parameter to find what's truly essential:
  • KAPPA_E = 0.0001 (Z-dependent correction to lambda)
  • SHIELD_FACTOR = 0.52 (Coulomb screening)
  • DELTA_PAIRING = 11.0 (pairing energy)
  • The "1/15" factors in E_surface and a_sym
  • The "12π" denominator in volume energy

Question: What is the MINIMAL set of parameters that achieves 175/285?

Test configurations:
1. Baseline (current pure QFD): All parameters
2. Remove KAPPA_E (set to 0)
3. Remove SHIELD_FACTOR (set to 1.0, no screening)
4. Remove DELTA_PAIRING (set to 0)
5. Remove all three
6. Try different values for 1/15, 1/12π to see if they're optimized

Also test: Can we achieve better than 175/285 by tweaking these?
===========================================================================
"""

import numpy as np
from collections import defaultdict
import itertools

# Fundamental constants (truly locked)
alpha_fine = 1.0 / 137.036
beta_vacuum = 1.0 / 3.043233053
M_proton = 938.272

# Parameters to test
KAPPA_E_DEFAULT = 0.0001
SHIELD_FACTOR_DEFAULT = 0.52
DELTA_PAIRING_DEFAULT = 11.0

def qfd_energy(A, Z, kappa_e, shield_factor, delta_pairing):
    """QFD energy with adjustable parameters."""
    N = A - Z
    q = Z / A if A > 0 else 0
    lambda_time = kappa_e * Z  # No lambda_time_0!

    V_0 = M_proton * (1 - (alpha_fine**2) / beta_vacuum)
    beta_nuclear = M_proton * beta_vacuum / 2
    E_volume = V_0 * (1 - lambda_time / (12 * np.pi))
    E_surface = beta_nuclear / 15
    a_sym = (beta_vacuum * M_proton) / 15
    a_disp = (alpha_fine * 197.327 / 1.2) * shield_factor

    E_bulk = E_volume * A
    E_surf = E_surface * (A**(2/3))
    E_asym = a_sym * A * ((1 - 2*q)**2)
    E_vac = a_disp * (Z**2) / (A**(1/3))

    E_pair = 0
    if delta_pairing > 0:
        if Z % 2 == 0 and N % 2 == 0:
            E_pair = -delta_pairing / np.sqrt(A)
        elif Z % 2 == 1 and N % 2 == 1:
            E_pair = +delta_pairing / np.sqrt(A)

    return E_bulk + E_surf + E_asym + E_vac + E_pair

def find_stable_Z(A, kappa_e, shield_factor, delta_pairing):
    """Find Z that minimizes energy."""
    best_Z, best_E = 1, qfd_energy(A, 1, kappa_e, shield_factor, delta_pairing)
    for Z in range(1, A):
        E = qfd_energy(A, Z, kappa_e, shield_factor, delta_pairing)
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
print("MINIMAL PARAMETER TEST - FIND THE ABSOLUTE MINIMUM")
print("="*95)
print()

# ============================================================================
# TEST 1: INDIVIDUAL PARAMETER REMOVAL
# ============================================================================
print("="*95)
print("TEST 1: REMOVE EACH PARAMETER INDIVIDUALLY")
print("="*95)
print()

configs = [
    ("BASELINE (all parameters)", KAPPA_E_DEFAULT, SHIELD_FACTOR_DEFAULT, DELTA_PAIRING_DEFAULT),
    ("Remove KAPPA_E (kappa=0)", 0.0, SHIELD_FACTOR_DEFAULT, DELTA_PAIRING_DEFAULT),
    ("Remove SHIELD (shield=1.0)", KAPPA_E_DEFAULT, 1.0, DELTA_PAIRING_DEFAULT),
    ("Remove PAIRING (delta=0)", KAPPA_E_DEFAULT, SHIELD_FACTOR_DEFAULT, 0.0),
    ("MINIMAL (remove all 3)", 0.0, 1.0, 0.0),
]

results = []
baseline_correct = None

print(f"{'Configuration':<35} {'Correct':<12} {'Success %':<12} {'vs Baseline'}")
print("-"*95)

for config_name, ke, sf, dp in configs:
    correct = 0
    for name, Z_exp, A in test_nuclides:
        Z_pred = find_stable_Z(A, ke, sf, dp)
        if Z_pred == Z_exp:
            correct += 1

    if baseline_correct is None:
        baseline_correct = correct
        delta_str = "baseline"
    else:
        delta = correct - baseline_correct
        delta_str = f"{delta:+d} matches"

    results.append({
        'name': config_name,
        'kappa_e': ke,
        'shield': sf,
        'pairing': dp,
        'correct': correct,
        'rate': 100 * correct / len(test_nuclides),
        'delta': correct - baseline_correct if baseline_correct else 0,
    })

    marker = ""
    if baseline_correct and abs(correct - baseline_correct) > 10:
        marker = "★★" if abs(correct - baseline_correct) > 50 else "★"

    print(f"{config_name:<35} {correct:<12} {100*correct/len(test_nuclides):<12.1f} {delta_str:<15} {marker}")

print()

# ============================================================================
# ANALYSIS
# ============================================================================
print("="*95)
print("ANALYSIS: WHICH PARAMETERS ARE ESSENTIAL?")
print("="*95)
print()

for r in results[1:]:  # Skip baseline
    delta = r['delta']
    print(f"{r['name']}:")
    print(f"  Effect: {delta:+d} matches ({delta/285*100:+.1f}%)")

    if abs(delta) < 3:
        print(f"  ✓ NEGLIGIBLE - This parameter can be removed!")
    elif abs(delta) < 10:
        print(f"  • MINOR - Small but measurable effect")
    elif abs(delta) < 50:
        print(f"  ★ SIGNIFICANT - Important for accuracy")
    else:
        print(f"  ★★ ESSENTIAL - Critical parameter!")

    print()

# ============================================================================
# TEST 2: OPTIMIZE REMAINING PARAMETERS
# ============================================================================
print("="*95)
print("TEST 2: CAN WE BEAT 175/285 BY OPTIMIZING PARAMETERS?")
print("="*95)
print()

# Test if shield_factor can be optimized better than 0.52
print("Testing different SHIELD_FACTOR values...")
print()

shield_values = [0.40, 0.45, 0.50, 0.52, 0.55, 0.60, 0.65, 0.70]
best_shield_correct = 0
best_shield_value = 0.52

print(f"{'Shield Factor':<20} {'Correct':<12} {'Success %':<12} {'vs Current'}")
print("-"*95)

for shield in shield_values:
    correct = 0
    for name, Z_exp, A in test_nuclides:
        Z_pred = find_stable_Z(A, KAPPA_E_DEFAULT, shield, DELTA_PAIRING_DEFAULT)
        if Z_pred == Z_exp:
            correct += 1

    delta = correct - baseline_correct
    marker = "★" if correct > baseline_correct else ""

    print(f"{shield:<20.2f} {correct:<12} {100*correct/len(test_nuclides):<12.1f} {delta:+d}  {marker}")

    if correct > best_shield_correct:
        best_shield_correct = correct
        best_shield_value = shield

print()
if best_shield_correct > baseline_correct:
    print(f"★ BETTER shield factor found: {best_shield_value:.2f} → {best_shield_correct}/285")
else:
    print(f"Current shield factor (0.52) is optimal or near-optimal")

print()

# Test if pairing energy can be optimized
print("Testing different DELTA_PAIRING values...")
print()

pairing_values = [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
best_pairing_correct = 0
best_pairing_value = 11.0

print(f"{'Pairing (MeV)':<20} {'Correct':<12} {'Success %':<12} {'vs Current'}")
print("-"*95)

for pairing in pairing_values:
    correct = 0
    for name, Z_exp, A in test_nuclides:
        Z_pred = find_stable_Z(A, KAPPA_E_DEFAULT, SHIELD_FACTOR_DEFAULT, pairing)
        if Z_pred == Z_exp:
            correct += 1

    delta = correct - baseline_correct
    marker = "★" if correct > baseline_correct else ""

    print(f"{pairing:<20.1f} {correct:<12} {100*correct/len(test_nuclides):<12.1f} {delta:+d}  {marker}")

    if correct > best_pairing_correct:
        best_pairing_correct = correct
        best_pairing_value = pairing

print()
if best_pairing_correct > baseline_correct:
    print(f"★ BETTER pairing energy found: {best_pairing_value:.1f} MeV → {best_pairing_correct}/285")
else:
    print(f"Current pairing energy (11.0 MeV) is optimal or near-optimal")

print()

# ============================================================================
# TEST 3: JOINT OPTIMIZATION
# ============================================================================
print("="*95)
print("TEST 3: JOINT OPTIMIZATION OF SHIELD AND PAIRING")
print("="*95)
print()

print("Grid searching for optimal (shield, pairing) combination...")
print()

shield_search = [0.45, 0.50, 0.52, 0.55, 0.60]
pairing_search = [9.0, 10.0, 11.0, 12.0, 13.0]

best_joint_correct = baseline_correct
best_joint_params = (SHIELD_FACTOR_DEFAULT, DELTA_PAIRING_DEFAULT)

for shield, pairing in itertools.product(shield_search, pairing_search):
    correct = 0
    for name, Z_exp, A in test_nuclides:
        Z_pred = find_stable_Z(A, KAPPA_E_DEFAULT, shield, pairing)
        if Z_pred == Z_exp:
            correct += 1

    if correct > best_joint_correct:
        best_joint_correct = correct
        best_joint_params = (shield, pairing)
        print(f"  New best: shield={shield:.2f}, pairing={pairing:.1f} → {correct}/285 ({100*correct/285:.1f}%)")

print()
print(f"Best parameters:")
print(f"  SHIELD_FACTOR = {best_joint_params[0]:.2f}")
print(f"  DELTA_PAIRING = {best_joint_params[1]:.1f} MeV")
print(f"  Result: {best_joint_correct}/285 ({100*best_joint_correct/285:.1f}%)")
print(f"  Improvement: {best_joint_correct - baseline_correct:+d} matches")
print()

# ============================================================================
# SUMMARY
# ============================================================================
print("="*95)
print("SUMMARY: MINIMAL PARAMETER SET")
print("="*95)
print()

print("TRULY FUNDAMENTAL (cannot be changed):")
print("  • α = 1/137.036 (fine structure constant)")
print("  • β = 1/3.043233053 (vacuum stiffness)")
print("  • M_proton = 938.272 MeV")
print()

print("ESSENTIAL PARAMETERS (large effect if removed):")
for r in results[1:]:
    if abs(r['delta']) > 10:
        print(f"  • {r['name'].split('(')[0].strip()}: {r['delta']:+d} matches if removed")
print()

print("NEGLIGIBLE PARAMETERS (can be removed):")
for r in results[1:]:
    if abs(r['delta']) < 3:
        print(f"  • {r['name'].split('(')[0].strip()}: {r['delta']:+d} matches if removed ← CAN REMOVE!")
print()

print("OPTIMIZED PARAMETERS:")
if best_joint_correct > baseline_correct:
    print(f"  ★ Joint optimization found improvement:")
    print(f"    Current:   {baseline_correct}/285 (shield={SHIELD_FACTOR_DEFAULT:.2f}, pairing={DELTA_PAIRING_DEFAULT:.1f})")
    print(f"    Optimized: {best_joint_correct}/285 (shield={best_joint_params[0]:.2f}, pairing={best_joint_params[1]:.1f})")
    print(f"    Gain: {best_joint_correct - baseline_correct:+d} matches")
else:
    print(f"  Current parameters are already optimal")
print()

print("MINIMAL MODEL (if we remove negligible parameters):")
minimal_result = [r for r in results if r['name'] == 'MINIMAL (remove all 3)'][0]
print(f"  Result: {minimal_result['correct']}/285 ({minimal_result['rate']:.1f}%)")
print(f"  Loss: {minimal_result['delta']} matches vs current")
print()

if abs(minimal_result['delta']) < 10:
    print("  → MINIMAL model (only α, β, M_p) achieves >97% of current performance!")
    print("  → All three parameters (KAPPA_E, SHIELD, PAIRING) may be emergent!")
else:
    print(f"  → Need at least some parameters to maintain accuracy")
    print("  → The loss of {-minimal_result['delta']} matches is significant")

print()
print("="*95)
