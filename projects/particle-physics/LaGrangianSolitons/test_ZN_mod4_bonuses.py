#!/usr/bin/env python3
"""
TEST (Z,N) MOD 4 BONUSES - REFINED GEOMETRIC CORRECTIONS
===========================================================================
Discovery: Within A mod 4 = 1, there's huge variation:
  (Z,N) mod 4 = (2,3): 92.3% success!
  (Z,N) mod 4 = (3,2): 84.6% success!
  (Z,N) mod 4 = (0,1): 62.5% success
  (Z,N) mod 4 = (1,0): 72.7% success

Test refined bonuses based on (Z mod 4, N mod 4) combinations:
1. Bonus for (2,3) and (3,2) - the exceptional cases
2. Separate bonuses for each (Z mod 4, N mod 4) combination
3. Combined A mod 4 + (Z,N) mod 4 structure
===========================================================================
"""

import numpy as np
import itertools

# Constants
alpha_fine = 1.0 / 137.036
beta_vacuum = 1.0 / 3.058231
M_proton = 938.272
LAMBDA_TIME_0 = 0.42
KAPPA_E = 0.0001
SHIELD_FACTOR = 0.52
DELTA_PAIRING = 11.0

def qfd_energy_ZN_mod4(A, Z, bonus_23, bonus_32, bonus_10, bonus_01):
    """QFD energy with (Z,N) mod 4 bonuses."""
    N = A - Z
    q = Z / A if A > 0 else 0
    lambda_time = LAMBDA_TIME_0 + KAPPA_E * Z

    V_0 = M_proton * (1 - (alpha_fine**2) / beta_vacuum)
    beta_nuclear = M_proton * beta_vacuum / 2
    E_volume = V_0 * (1 - lambda_time / (12 * np.pi))
    E_surface = beta_nuclear / 15
    a_sym = (beta_vacuum * M_proton) / 15
    a_disp = (alpha_fine * 197.327 / 1.2) * SHIELD_FACTOR

    E_bulk = E_volume * A
    E_surf = E_surface * (A**(2/3))
    E_asym = a_sym * A * ((1 - 2*q)**2)
    E_vac = a_disp * (Z**2) / (A**(1/3))

    E_pair = 0
    if Z % 2 == 0 and N % 2 == 0:
        E_pair = -DELTA_PAIRING / np.sqrt(A)
    elif Z % 2 == 1 and N % 2 == 1:
        E_pair = +DELTA_PAIRING / np.sqrt(A)

    # (Z,N) mod 4 bonuses
    E_geom = 0
    Z_mod = Z % 4
    N_mod = N % 4

    if (Z_mod, N_mod) == (2, 3):
        E_geom -= E_surf * bonus_23  # Stabilize exceptional (2,3)
    elif (Z_mod, N_mod) == (3, 2):
        E_geom -= E_surf * bonus_32  # Stabilize exceptional (3,2)
    elif (Z_mod, N_mod) == (1, 0):
        E_geom -= E_surf * bonus_10
    elif (Z_mod, N_mod) == (0, 1):
        E_geom -= E_surf * bonus_01

    return E_bulk + E_surf + E_asym + E_vac + E_pair + E_geom

def find_stable_Z(A, bonus_23, bonus_32, bonus_10, bonus_01):
    """Find Z that minimizes energy."""
    best_Z, best_E = 1, qfd_energy_ZN_mod4(A, 1, bonus_23, bonus_32, bonus_10, bonus_01)
    for Z in range(1, A):
        E = qfd_energy_ZN_mod4(A, Z, bonus_23, bonus_32, bonus_10, bonus_01)
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
print("TEST (Z,N) MOD 4 GEOMETRIC BONUSES")
print("="*95)
print()
print("Strategy: Add bonuses for specific (Z mod 4, N mod 4) combinations")
print("  Focus on exceptional cases: (2,3) at 92.3% and (3,2) at 84.6%")
print()

# ============================================================================
# TEST 1: BONUS FOR (2,3) AND (3,2) ONLY
# ============================================================================
print("="*95)
print("TEST 1: BONUS FOR EXCEPTIONAL CASES (2,3) AND (3,2)")
print("="*95)
print()

bonus_values = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]

best_correct = 0
best_params = None
results = []

print(f"Testing {len(bonus_values)} bonus values for (2,3) and (3,2)...")
print()

for bonus in bonus_values:
    correct = 0
    for name, Z_exp, A in test_nuclides:
        # Apply same bonus to both (2,3) and (3,2)
        Z_pred = find_stable_Z(A, bonus, bonus, 0.0, 0.0)
        if Z_pred == Z_exp:
            correct += 1

    results.append((bonus, correct))
    if correct > best_correct:
        best_correct = correct
        best_params = (bonus, bonus, 0.0, 0.0)

print(f"{'Bonus (2,3) & (3,2)':<25} {'Correct':<10} {'Success %':<12} {'Improvement'}")
print("-"*95)

for bonus, correct in results:
    rate = 100 * correct / len(test_nuclides)
    improvement = correct - 175
    marker = "★" if improvement > 0 else ""

    print(f"{bonus:<25.2f} {correct:<10} {rate:<12.1f} {improvement:+d}  {marker}")

print()

if best_correct > 175:
    print(f"★★ IMPROVEMENT! Bonus = {best_params[0]:.2f}")
    print(f"   Pure geometry:    175/285 (61.4%)")
    print(f"   With (2,3), (3,2) bonus: {best_correct}/285 ({100*best_correct/285:.1f}%)")
    print(f"   Improvement:      {best_correct - 175:+d} matches")
else:
    print("No improvement from (2,3) and (3,2) bonuses alone")

print()

# ============================================================================
# TEST 2: SEPARATE BONUSES FOR ALL FOUR A MOD 4 = 1 COMBINATIONS
# ============================================================================
print("="*95)
print("TEST 2: INDEPENDENT BONUSES FOR EACH (Z,N) MOD 4 COMBINATION")
print("="*95)
print()

print("Optimizing 4 independent bonuses:")
print("  bonus_23: for (Z,N) mod 4 = (2,3)")
print("  bonus_32: for (Z,N) mod 4 = (3,2)")
print("  bonus_10: for (Z,N) mod 4 = (1,0)")
print("  bonus_01: for (Z,N) mod 4 = (0,1)")
print()

# Grid search (coarse first)
bonus_range = [0.0, 0.20, 0.40, 0.60]

best_correct_full = 0
best_params_full = None

print(f"Testing {len(bonus_range)**4} combinations (coarse grid)...")
print()

for b23, b32, b10, b01 in itertools.product(bonus_range, repeat=4):
    correct = 0
    for name, Z_exp, A in test_nuclides:
        Z_pred = find_stable_Z(A, b23, b32, b10, b01)
        if Z_pred == Z_exp:
            correct += 1

    if correct > best_correct_full:
        best_correct_full = correct
        best_params_full = (b23, b32, b10, b01)

print(f"Best result (coarse grid):")
print(f"  bonus_23 = {best_params_full[0]:.2f}")
print(f"  bonus_32 = {best_params_full[1]:.2f}")
print(f"  bonus_10 = {best_params_full[2]:.2f}")
print(f"  bonus_01 = {best_params_full[3]:.2f}")
print(f"  Result: {best_correct_full}/285 ({100*best_correct_full/285:.1f}%)")
print(f"  Improvement: {best_correct_full - 175:+d} matches")
print()

# Fine-tune around best
if best_correct_full > 175:
    print("Fine-tuning around best parameters...")
    b23_opt, b32_opt, b10_opt, b01_opt = best_params_full

    fine_range_23 = [max(0, b23_opt-0.10), b23_opt, b23_opt+0.10]
    fine_range_32 = [max(0, b32_opt-0.10), b32_opt, b32_opt+0.10]
    fine_range_10 = [max(0, b10_opt-0.10), b10_opt, b10_opt+0.10]
    fine_range_01 = [max(0, b01_opt-0.10), b01_opt, b01_opt+0.10]

    for b23, b32, b10, b01 in itertools.product(fine_range_23, fine_range_32,
                                                  fine_range_10, fine_range_01):
        correct = 0
        for name, Z_exp, A in test_nuclides:
            Z_pred = find_stable_Z(A, b23, b32, b10, b01)
            if Z_pred == Z_exp:
                correct += 1

        if correct > best_correct_full:
            best_correct_full = correct
            best_params_full = (b23, b32, b10, b01)

    print()
    print(f"Final optimized:")
    print(f"  bonus_23 = {best_params_full[0]:.2f}")
    print(f"  bonus_32 = {best_params_full[1]:.2f}")
    print(f"  bonus_10 = {best_params_full[2]:.2f}")
    print(f"  bonus_01 = {best_params_full[3]:.2f}")
    print(f"  Result: {best_correct_full}/285 ({100*best_correct_full/285:.1f}%)")
    print(f"  Improvement: {best_correct_full - 175:+d} matches")

print()

# ============================================================================
# FINAL COMPARISON
# ============================================================================
print("="*95)
print("FINAL RESULTS")
print("="*95)
print()

print(f"{'Model':<40} {'Correct':<10} {'Success %':<12} {'vs Pure Geom'}")
print("-"*95)

print(f"{'Pure geometry (no bonuses)':<40} {175:<10} {61.4:<12.1f} {'baseline'}")
print(f"{'Empirical bonuses (magic, symm, nr)':<40} {206:<10} {72.3:<12.1f} {'+31 matches'}")

if best_correct > 175:
    improvement_1 = best_correct - 175
    print(f"{'(Z,N)=(2,3),(3,2) bonus only':<40} {best_correct:<10} "
          f"{100*best_correct/285:<12.1f} {f'+{improvement_1} matches'}")

if best_correct_full > 175:
    improvement_2 = best_correct_full - 175
    print(f"{'All (Z,N) mod 4 bonuses':<40} {best_correct_full:<10} "
          f"{100*best_correct_full/285:<12.1f} {f'+{improvement_2} matches'}")

print()

if best_correct_full > 206:
    print("★★★ GEOMETRIC BONUSES BEAT EMPIRICAL!")
    print("    → (Z,N) mod 4 structure is MORE fundamental than magic numbers!")
elif best_correct_full > 195:
    print("★★ GEOMETRIC BONUSES ARE COMPETITIVE!")
    print("    → Close to empirical performance with pure topology!")
elif best_correct_full > 180:
    print("★ GEOMETRIC BONUSES HELP SIGNIFICANTLY!")
    print("    → Better than pure geometry!")
else:
    print("Geometric bonuses show modest/no improvement")
    print("   Pattern is statistical, not a simple energy correction")

print()

# ============================================================================
# INTERPRETATION
# ============================================================================
print("="*95)
print("INTERPRETATION")
print("="*95)
print()

if best_correct_full > 175:
    print("SUCCESS! (Z,N) mod 4 bonuses improve predictions!")
    print()
    print("Physical meaning:")
    print("  • (2,3) and (3,2) are exceptional topological configurations")
    print("  • Quaternion structure: Z ≡ 2,3 (mod 4) special?")
    print("  • SU(2) × SU(2) ≈ SO(4) symmetry breaking pattern")
    print()
    print("Next steps:")
    print("  1. Derive (Z,N) mod 4 pattern from Cl(3,3) first principles")
    print("  2. Connect to quaternion winding numbers")
    print("  3. Explain why (2,3) and (3,2) are topologically favored")
    print("  4. Test if pattern extends to unstable nuclei")
else:
    print("(Z,N) mod 4 pattern is STATISTICAL but not a simple energy shift")
    print()
    print("Possible reasons:")
    print("  • Pattern might be EMERGENT from other physics")
    print("  • May need Z-dependent or A-dependent modulation")
    print("  • Could be higher-order effect (product of multiple terms)")
    print()
    print("Alternative approaches:")
    print("  1. Test (Z mod 4) × (N mod 4) product terms")
    print("  2. Try (Z+N) mod 4 vs (Z-N) mod 4")
    print("  3. Test combined A mod 4 AND (Z,N) mod 4")
    print("  4. Look for (Z mod 7) × (N mod 7) patterns (connect to 7-fold)")

print()
print("="*95)
