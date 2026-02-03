#!/usr/bin/env python3
"""
TEST ZERO MAGIC BONUS - EXPOSE HIDDEN STRUCTURE
===========================================================================
Remove magic number bonuses entirely to see:
1. Which nuclei truly need magic vs stable from other physics
2. What hidden structure is masked by magic bonuses
3. If dual resonance + pairing can work without magic
4. Which families depend most on magic

Current with magic:
  Type I:   96.8% (magic=0.05)
  Type II:  93.3% (magic=0.20) ← Uses 4× stronger magic!
  Type III: 91.7% (magic=0.10)
  Type IV:  74.6% (magic=0.10)
  Type V:   63.1% (magic=0.05)
  TOTAL:    72.3% (206/285)

Test: Set all magic=0, keep everything else same
===========================================================================
"""

import numpy as np
from collections import defaultdict

# Constants
alpha_fine = 1.0 / 137.036
beta_vacuum = 1.0 / 3.043233053
M_proton = 938.272
LAMBDA_TIME_0 = 0.42
KAPPA_E = 0.0001
SHIELD_FACTOR = 0.52
DELTA_PAIRING = 11.0

ISOMER_NODES = {2, 8, 20, 28, 50, 82, 126}
SUBSHELL_Z = {6, 14, 16, 32, 34, 38, 40}
SUBSHELL_N = {6, 14, 16, 32, 34, 40, 56, 64, 70}

# Crossover reclassifications
CROSSOVER_RECLASSIFICATIONS = {
    ('Kr-84', 36, 84): 'Type_I',
    ('Rb-87', 37, 87): 'Type_II',
    ('Mo-94', 42, 94): 'Type_V',
    ('Ru-104', 44, 104): 'Type_I',
    ('Cd-114', 48, 114): 'Type_I',
    ('In-115', 49, 115): 'Type_I',
    ('Sn-122', 50, 122): 'Type_II',
    ('Ba-138', 56, 138): 'Type_II',
    ('La-139', 57, 139): 'Type_II',
}

def classify_family(name, Z, A):
    """Reclassified families."""
    key = (name, Z, A)
    if key in CROSSOVER_RECLASSIFICATIONS:
        return CROSSOVER_RECLASSIFICATIONS[key]

    N = A - Z
    nz_ratio = N / Z if Z > 0 else 0

    if A < 40:
        if 0.9 <= nz_ratio <= 1.15:
            return "Type_I"
        else:
            return "Type_III"
    elif 40 <= A < 100:
        if 0.9 <= nz_ratio <= 1.15:
            return "Type_II"
        else:
            return "Type_IV"
    else:
        if 0.9 <= nz_ratio <= 1.15:
            return "Type_II"
        else:
            return "Type_V"

def get_resonance_bonus(Z, N, E_surface, magic, symm, nr, subshell):
    bonus = 0

    # Magic numbers
    if Z in ISOMER_NODES: bonus += E_surface * magic
    if N in ISOMER_NODES: bonus += E_surface * magic
    if Z in ISOMER_NODES and N in ISOMER_NODES:
        bonus += E_surface * magic * 0.5

    # Subshells
    if Z in SUBSHELL_Z: bonus += E_surface * subshell
    if N in SUBSHELL_N: bonus += E_surface * subshell

    # Charge fraction resonance
    nz_ratio = N / Z if Z > 0 else 0
    if 0.95 <= nz_ratio <= 1.15:
        bonus += E_surface * symm
    if 1.15 <= nz_ratio <= 1.30:
        bonus += E_surface * nr

    return bonus

def qfd_energy(A, Z, params):
    magic, symm, nr, subshell = params
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
    E_iso = -get_resonance_bonus(Z, N, E_surface, magic, symm, nr, subshell)

    E_pair = 0
    if Z % 2 == 0 and N % 2 == 0:
        E_pair = -DELTA_PAIRING / np.sqrt(A)
    elif Z % 2 == 1 and N % 2 == 1:
        E_pair = +DELTA_PAIRING / np.sqrt(A)

    return E_bulk + E_surf + E_asym + E_vac + E_iso + E_pair

def find_stable_Z(A, params):
    best_Z, best_E = 1, qfd_energy(A, 1, params)
    for Z in range(1, A):
        E = qfd_energy(A, Z, params)
        if E < best_E:
            best_E, best_Z = E, Z
    return best_Z

# Load data
with open('qfd_optimized_suite.py', 'r') as f:
    content = f.read()
start = content.find('test_nuclides = [')
end = content.find(']', start) + 1
test_nuclides = eval(content[start:end].replace('test_nuclides = ', ''))

# Optimal family parameters (with magic)
family_params_with_magic = {
    "Type_I":   (0.05, 0.40, 0.05, 0.00),
    "Type_II":  (0.20, 0.50, 0.05, 0.00),
    "Type_III": (0.10, 0.30, 0.10, 0.02),
    "Type_IV":  (0.10, 0.10, 0.10, 0.02),
    "Type_V":   (0.05, 0.10, 0.15, 0.00),
}

# Zero magic parameters (set magic=0, keep everything else)
family_params_zero_magic = {
    "Type_I":   (0.00, 0.40, 0.05, 0.00),
    "Type_II":  (0.00, 0.50, 0.05, 0.00),
    "Type_III": (0.00, 0.30, 0.10, 0.02),
    "Type_IV":  (0.00, 0.10, 0.10, 0.02),
    "Type_V":   (0.00, 0.10, 0.15, 0.00),
}

print("="*95)
print("ZERO MAGIC TEST - EXPOSE HIDDEN STRUCTURE")
print("="*95)
print()
print("Setting all magic bonuses to ZERO to see what happens...")
print()

# Classify all nuclei
families = defaultdict(list)
for name, Z_exp, A in test_nuclides:
    family = classify_family(name, Z_exp, A)
    families[family].append((name, Z_exp, A))

# ============================================================================
# TEST 1: Current (With Magic)
# ============================================================================
print("="*95)
print("TEST 1: CURRENT (WITH MAGIC)")
print("="*95)
print()

print(f"{'Family':<15} {'Total':<10} {'Correct':<10} {'Success %':<12} {'Magic Bonus'}")
print("-"*95)

total_correct_with = 0
magic_nuclei_correct_with = 0
magic_nuclei_total = 0

for family in sorted(families.keys()):
    family_nuclides = families[family]
    params = family_params_with_magic[family]

    correct = sum(1 for name, Z_exp, A in family_nuclides
                  if find_stable_Z(A, params) == Z_exp)

    success_rate = 100 * correct / len(family_nuclides)
    total_correct_with += correct

    # Count magic nuclei
    for name, Z_exp, A in family_nuclides:
        N_exp = A - Z_exp
        if Z_exp in ISOMER_NODES or N_exp in ISOMER_NODES:
            magic_nuclei_total += 1
            if find_stable_Z(A, params) == Z_exp:
                magic_nuclei_correct_with += 1

    print(f"{family:<15} {len(family_nuclides):<10} {correct:<10} {success_rate:<12.1f} {params[0]}")

total_success_with = 100 * total_correct_with / len(test_nuclides)
magic_success_with = 100 * magic_nuclei_correct_with / magic_nuclei_total if magic_nuclei_total > 0 else 0

print(f"{'TOTAL':<15} {len(test_nuclides):<10} {total_correct_with:<10} {total_success_with:.1f}%")
print()
print(f"Magic nuclei: {magic_nuclei_correct_with}/{magic_nuclei_total} ({magic_success_with:.1f}%)")
print()

# ============================================================================
# TEST 2: Zero Magic
# ============================================================================
print("="*95)
print("TEST 2: ZERO MAGIC (All magic bonuses = 0)")
print("="*95)
print()

print(f"{'Family':<15} {'Total':<10} {'Correct':<10} {'Success %':<12} {'Change'}")
print("-"*95)

total_correct_zero = 0
magic_nuclei_correct_zero = 0

for family in sorted(families.keys()):
    family_nuclides = families[family]
    params_with = family_params_with_magic[family]
    params_zero = family_params_zero_magic[family]

    correct_with = sum(1 for name, Z_exp, A in family_nuclides
                      if find_stable_Z(A, params_with) == Z_exp)
    correct_zero = sum(1 for name, Z_exp, A in family_nuclides
                       if find_stable_Z(A, params_zero) == Z_exp)

    success_rate = 100 * correct_zero / len(family_nuclides)
    change = correct_zero - correct_with
    total_correct_zero += correct_zero

    # Count magic nuclei
    for name, Z_exp, A in family_nuclides:
        N_exp = A - Z_exp
        if Z_exp in ISOMER_NODES or N_exp in ISOMER_NODES:
            if find_stable_Z(A, params_zero) == Z_exp:
                magic_nuclei_correct_zero += 1

    print(f"{family:<15} {len(family_nuclides):<10} {correct_zero:<10} {success_rate:<12.1f} {change:+d}")

total_success_zero = 100 * total_correct_zero / len(test_nuclides)
total_change = total_correct_zero - total_correct_with
magic_success_zero = 100 * magic_nuclei_correct_zero / magic_nuclei_total if magic_nuclei_total > 0 else 0

print(f"{'TOTAL':<15} {len(test_nuclides):<10} {total_correct_zero:<10} {total_success_zero:.1f}%{'':<7} {total_change:+d}")
print()
print(f"Magic nuclei: {magic_nuclei_correct_zero}/{magic_nuclei_total} ({magic_success_zero:.1f}%) [{magic_nuclei_correct_zero - magic_nuclei_correct_with:+d}]")
print()

# ============================================================================
# ANALYSIS: Which nuclei lost/gained without magic?
# ============================================================================
print("="*95)
print("ANALYSIS: WHAT CHANGED WITHOUT MAGIC?")
print("="*95)
print()

lost_nuclei = []
gained_nuclei = []

for name, Z_exp, A in test_nuclides:
    family = classify_family(name, Z_exp, A)
    params_with = family_params_with_magic[family]
    params_zero = family_params_zero_magic[family]

    Z_pred_with = find_stable_Z(A, params_with)
    Z_pred_zero = find_stable_Z(A, params_zero)

    correct_with = (Z_pred_with == Z_exp)
    correct_zero = (Z_pred_zero == Z_exp)

    N_exp = A - Z_exp
    is_magic = (Z_exp in ISOMER_NODES or N_exp in ISOMER_NODES)
    is_doubly_magic = (Z_exp in ISOMER_NODES and N_exp in ISOMER_NODES)

    if correct_with and not correct_zero:
        lost_nuclei.append({
            'name': name,
            'Z': Z_exp,
            'N': N_exp,
            'A': A,
            'family': family,
            'Z_pred_zero': Z_pred_zero,
            'is_magic': is_magic,
            'is_doubly_magic': is_doubly_magic,
        })
    elif not correct_with and correct_zero:
        gained_nuclei.append({
            'name': name,
            'Z': Z_exp,
            'N': N_exp,
            'A': A,
            'family': family,
            'Z_pred_with': Z_pred_with,
            'is_magic': is_magic,
            'is_doubly_magic': is_doubly_magic,
        })

print(f"Lost without magic: {len(lost_nuclei)} nuclei")
print(f"Gained without magic: {len(gained_nuclei)} nuclei")
print(f"Net change: {len(gained_nuclei) - len(lost_nuclei):+d}")
print()

if lost_nuclei:
    print("LOST WITHOUT MAGIC (was correct, now wrong):")
    print(f"{'Nuclide':<12} {'A':<6} {'Z':<6} {'N':<6} {'Family':<12} {'Pred (no magic)':<15} {'Magic?':<15}")
    print("-"*95)

    for n in lost_nuclei[:30]:  # Show top 30
        magic_status = "Doubly Magic" if n['is_doubly_magic'] else ("Magic" if n['is_magic'] else "Non-magic")
        print(f"{n['name']:<12} {n['A']:<6} {n['Z']:<6} {n['N']:<6} {n['family']:<12} "
              f"{n['Z_pred_zero']:<15} {magic_status}")

    print()

    # Count magic vs non-magic losses
    magic_lost = sum(1 for n in lost_nuclei if n['is_magic'])
    doubly_magic_lost = sum(1 for n in lost_nuclei if n['is_doubly_magic'])
    print(f"  Magic nuclei lost: {magic_lost}/{len(lost_nuclei)} ({100*magic_lost/len(lost_nuclei):.1f}%)")
    print(f"  Doubly magic lost: {doubly_magic_lost}/{len(lost_nuclei)} ({100*doubly_magic_lost/len(lost_nuclei):.1f}%)")
    print()

if gained_nuclei:
    print("GAINED WITHOUT MAGIC (was wrong, now correct):")
    print(f"{'Nuclide':<12} {'A':<6} {'Z':<6} {'N':<6} {'Family':<12} {'Pred (with magic)':<17} {'Magic?':<15}")
    print("-"*95)

    for n in gained_nuclei[:30]:  # Show top 30
        magic_status = "Doubly Magic" if n['is_doubly_magic'] else ("Magic" if n['is_magic'] else "Non-magic")
        print(f"{n['name']:<12} {n['A']:<6} {n['Z']:<6} {n['N']:<6} {n['family']:<12} "
              f"{n['Z_pred_with']:<17} {magic_status}")

    print()

    # Count magic vs non-magic gains
    magic_gained = sum(1 for n in gained_nuclei if n['is_magic'])
    doubly_magic_gained = sum(1 for n in gained_nuclei if n['is_doubly_magic'])
    print(f"  Magic nuclei gained: {magic_gained}/{len(gained_nuclei)} ({100*magic_gained/len(gained_nuclei):.1f}%)")
    print(f"  Doubly magic gained: {doubly_magic_gained}/{len(gained_nuclei)} ({100*doubly_magic_gained/len(gained_nuclei):.1f}%)")
    print()

# ============================================================================
# SUMMARY
# ============================================================================
print("="*95)
print("SUMMARY: WHAT MAGIC BONUSES ACTUALLY DO")
print("="*95)
print()

print(f"With magic:    {total_correct_with}/285 ({total_success_with:.1f}%)")
print(f"Without magic: {total_correct_zero}/285 ({total_success_zero:.1f}%)")
print(f"Difference:    {total_change:+d} matches ({total_success_zero - total_success_with:+.1f} percentage points)")
print()

if total_change < 0:
    print("✗ Magic bonuses ARE needed (accuracy dropped without them)")
    print()
    print(f"  Most affected: {lost_nuclei[0]['family'] if lost_nuclei else 'N/A'}")
    print(f"  Magic nuclei dependency: {100*magic_lost/len(lost_nuclei) if lost_nuclei else 0:.1f}% of losses are magic nuclei")
    print()
    print("Conclusion: Magic numbers are REAL physics, not just fitting.")
    print("            But optimization may have hidden other structure.")
    print()
    print("Next: Re-optimize symm/nr/subshell with magic=0 to find compensating structure")

elif total_change > 0:
    print("✓✓ MAGIC BONUSES WERE OVERFITTING!")
    print()
    print("  Accuracy actually IMPROVED without magic!")
    print(f"  Gained: {len(gained_nuclei)} nuclei")
    print(f"  Lost: {len(lost_nuclei)} nuclei")
    print()
    print("Conclusion: Magic numbers were masking true underlying physics.")
    print("            Dual resonance + pairing are more fundamental!")
    print()
    print("Next: Re-optimize all parameters with magic=0")

else:
    print("= No change (magic bonuses had zero net effect)")
    print()
    print("Conclusion: Magic bonuses exactly balanced - neither helping nor hurting.")

print()
print("="*95)
