#!/usr/bin/env python3
"""
PURE QFD (NO LAMBDA) + EMPIRICAL BONUSES
===========================================================================
Corrected analysis comparing:
1. Pure QFD (no lambda, no bonuses): 175/285 baseline
2. Pure QFD + empirical bonuses: ??? (to be determined)

This will tell us the ACTUAL benefit of empirical bonuses when applied
to the clean pure QFD model (without lambda_time).

Empirical bonuses to test:
- Magic number bonuses (Z,N at 2,8,20,28,50,82,126)
- Symmetric bonuses (N/Z ∈ [0.95, 1.15])
- Neutron-rich bonuses (N/Z ∈ [1.15, 1.30])
===========================================================================
"""

import numpy as np
from collections import Counter
import itertools

# Constants (no lambda!)
alpha_fine = 1.0 / 137.036
beta_vacuum = 1.0 / 3.058231
M_proton = 938.272
KAPPA_E = 0.0001
SHIELD_FACTOR = 0.52
DELTA_PAIRING = 11.0

ISOMER_NODES = {2, 8, 20, 28, 50, 82, 126}

def qfd_energy_with_bonuses(A, Z, magic_bonus, symm_bonus, nr_bonus):
    """Pure QFD (no lambda) + empirical bonuses."""
    N = A - Z
    q = Z / A if A > 0 else 0
    lambda_time = KAPPA_E * Z  # NOT lambda_time_0 + kappa*Z (lambda_0 = 0!)

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

    # Empirical bonuses (in MeV)
    E_bonus = 0

    # Magic number bonus
    if Z in ISOMER_NODES:
        E_bonus -= magic_bonus  # Stabilize (negative = lower energy)
    if N in ISOMER_NODES:
        E_bonus -= magic_bonus

    # Symmetric bonus (N/Z ≈ 1)
    nz_ratio = N / Z if Z > 0 else 0
    if 0.95 <= nz_ratio <= 1.15:
        E_bonus -= symm_bonus

    # Neutron-rich bonus
    if 1.15 <= nz_ratio <= 1.30:
        E_bonus -= nr_bonus

    return E_bulk + E_surf + E_asym + E_vac + E_pair + E_bonus

def find_stable_Z(A, magic_bonus, symm_bonus, nr_bonus):
    """Find Z that minimizes energy."""
    best_Z, best_E = 1, qfd_energy_with_bonuses(A, 1, magic_bonus, symm_bonus, nr_bonus)
    for Z in range(1, A):
        E = qfd_energy_with_bonuses(A, Z, magic_bonus, symm_bonus, nr_bonus)
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
print("PURE QFD (NO LAMBDA) + EMPIRICAL BONUSES")
print("="*95)
print()

# Test 1: Pure QFD (no bonuses)
print("TEST 1: PURE QFD BASELINE (no bonuses)")
print("-"*95)
correct_pure = 0
for name, Z_exp, A in test_nuclides:
    Z_pred = find_stable_Z(A, 0.0, 0.0, 0.0)
    if Z_pred == Z_exp:
        correct_pure += 1

print(f"Result: {correct_pure}/285 ({100*correct_pure/285:.1f}%)")
print()

# Test 2: Grid search for optimal bonuses
print("TEST 2: OPTIMIZE EMPIRICAL BONUSES")
print("-"*95)
print("Grid searching for optimal magic, symm, and nr bonuses...")
print()

# Coarse grid
magic_values = [0.0, 5.0, 10.0, 15.0, 20.0]
symm_values = [0.0, 2.0, 4.0, 6.0, 8.0]
nr_values = [0.0, 1.0, 2.0, 3.0, 4.0]

best_correct = correct_pure
best_params = (0.0, 0.0, 0.0)

total_combinations = len(magic_values) * len(symm_values) * len(nr_values)
print(f"Testing {total_combinations} combinations (coarse grid)...")
print()

for magic, symm, nr in itertools.product(magic_values, symm_values, nr_values):
    correct = 0
    for name, Z_exp, A in test_nuclides:
        Z_pred = find_stable_Z(A, magic, symm, nr)
        if Z_pred == Z_exp:
            correct += 1

    if correct > best_correct:
        best_correct = correct
        best_params = (magic, symm, nr)
        print(f"  New best: magic={magic:.1f}, symm={symm:.1f}, nr={nr:.1f} → {correct}/285 ({100*correct/285:.1f}%)")

print()
print(f"Best parameters (coarse): magic={best_params[0]:.1f}, symm={best_params[1]:.1f}, nr={best_params[2]:.1f}")
print(f"Result: {best_correct}/285 ({100*best_correct/285:.1f}%)")
print()

# Fine-tune around best
if best_correct > correct_pure:
    print("Fine-tuning around best parameters...")
    magic_opt, symm_opt, nr_opt = best_params

    magic_fine = [max(0, magic_opt-2.5), magic_opt, magic_opt+2.5]
    symm_fine = [max(0, symm_opt-1.0), symm_opt, symm_opt+1.0]
    nr_fine = [max(0, nr_opt-0.5), nr_opt, nr_opt+0.5]

    for magic, symm, nr in itertools.product(magic_fine, symm_fine, nr_fine):
        correct = 0
        for name, Z_exp, A in test_nuclides:
            Z_pred = find_stable_Z(A, magic, symm, nr)
            if Z_pred == Z_exp:
                correct += 1

        if correct > best_correct:
            best_correct = correct
            best_params = (magic, symm, nr)
            print(f"  Refined: magic={magic:.1f}, symm={symm:.1f}, nr={nr:.1f} → {correct}/285 ({100*correct/285:.1f}%)")

    print()
    print(f"Optimized parameters: magic={best_params[0]:.1f}, symm={best_params[1]:.1f}, nr={best_params[2]:.1f}")
    print(f"Result: {best_correct}/285 ({100*best_correct/285:.1f}%)")

print()

# Test 3: Identify which nuclei are fixed
print("="*95)
print("NUCLEI FIXED BY EMPIRICAL BONUSES")
print("="*95)
print()

fixed_nuclei = []
for name, Z_exp, A in test_nuclides:
    Z_pred_pure = find_stable_Z(A, 0.0, 0.0, 0.0)
    Z_pred_bonus = find_stable_Z(A, best_params[0], best_params[1], best_params[2])

    correct_pure = (Z_pred_pure == Z_exp)
    correct_bonus = (Z_pred_bonus == Z_exp)

    if (not correct_pure) and correct_bonus:
        N_exp = A - Z_exp
        is_magic_Z = Z_exp in ISOMER_NODES
        is_magic_N = N_exp in ISOMER_NODES
        nz_ratio = N_exp / Z_exp if Z_exp > 0 else 0
        is_symmetric = 0.95 <= nz_ratio <= 1.15
        is_neutron_rich = 1.15 <= nz_ratio <= 1.30

        fixed_nuclei.append({
            'name': name,
            'A': A,
            'Z': Z_exp,
            'N': N_exp,
            'is_magic_Z': is_magic_Z,
            'is_magic_N': is_magic_N,
            'is_symmetric': is_symmetric,
            'is_neutron_rich': is_neutron_rich,
        })

print(f"Total nuclei fixed: {len(fixed_nuclei)}")
print()

if fixed_nuclei:
    print(f"{'Nuclide':<12} {'A':<6} {'Z':<6} {'N':<6} {'Magic':<12} {'Symm':<8} {'NR':<8}")
    print("-"*95)

    for n in sorted(fixed_nuclei, key=lambda x: x['A']):
        magic_str = ""
        if n['is_magic_Z'] and n['is_magic_N']:
            magic_str = "Z+N ★★"
        elif n['is_magic_Z']:
            magic_str = "Z ★"
        elif n['is_magic_N']:
            magic_str = "N ★"

        symm_str = "Yes" if n['is_symmetric'] else ""
        nr_str = "Yes" if n['is_neutron_rich'] else ""

        print(f"{n['name']:<12} {n['A']:<6} {n['Z']:<6} {n['N']:<6} {magic_str:<12} {symm_str:<8} {nr_str:<8}")

    print()

    # Statistics
    magic_count = sum(1 for n in fixed_nuclei if n['is_magic_Z'] or n['is_magic_N'])
    symm_count = sum(1 for n in fixed_nuclei if n['is_symmetric'])
    nr_count = sum(1 for n in fixed_nuclei if n['is_neutron_rich'])

    print("Bonus type breakdown:")
    print(f"  Magic (Z or N): {magic_count}/{len(fixed_nuclei)} ({100*magic_count/len(fixed_nuclei):.1f}%)")
    print(f"  Symmetric:      {symm_count}/{len(fixed_nuclei)} ({100*symm_count/len(fixed_nuclei):.1f}%)")
    print(f"  Neutron-rich:   {nr_count}/{len(fixed_nuclei)} ({100*nr_count/len(fixed_nuclei):.1f}%)")

print()

# Summary
print("="*95)
print("SUMMARY")
print("="*95)
print()
print(f"Pure QFD (no lambda, no bonuses):   {correct_pure}/285 ({100*correct_pure/285:.1f}%)")
print(f"Pure QFD + empirical bonuses:       {best_correct}/285 ({100*best_correct/285:.1f}%)")
print(f"Improvement:                        {best_correct - correct_pure:+d} matches ({100*(best_correct - correct_pure)/285:+.1f}%)")
print()

if best_correct > correct_pure:
    print(f"★ Empirical bonuses improve pure QFD by {best_correct - correct_pure} matches!")
    print(f"  Optimal bonuses: magic={best_params[0]:.1f} MeV, symm={best_params[1]:.1f} MeV, nr={best_params[2]:.1f} MeV")
else:
    print("Empirical bonuses provide no improvement over pure QFD")
    print("  → Geometric structure alone is sufficient!")

print()
print("="*95)
