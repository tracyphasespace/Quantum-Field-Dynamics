#!/usr/bin/env python3
"""
TEST VERY STRONG SYMMETRIC BONUS
===========================================================================
Dual windows gave +5 improvement (183/285), but specific symmetric
failures like Ca-40 are STILL WRONG even with symm=0.20.

Test much stronger symmetric bonuses to see if these can be flipped,
or if there's a fundamental problem with the asymmetry term.
===========================================================================
"""

import numpy as np

# Constants
alpha_fine = 1.0 / 137.036
beta_vacuum = 1.0 / 3.058231
M_proton = 938.272
LAMBDA_TIME_0 = 0.42
KAPPA_E = 0.0001
SHIELD_FACTOR = 0.52
MAGIC_BONUS = 0.10
DELTA_PAIRING = 11.0
ISOMER_NODES = {2, 8, 20, 28, 50, 82, 126}

def get_resonance_bonus(Z, N, E_surface, symm_bonus, nr_bonus):
    bonus = 0
    if Z in ISOMER_NODES: bonus += E_surface * MAGIC_BONUS
    if N in ISOMER_NODES: bonus += E_surface * MAGIC_BONUS
    if Z in ISOMER_NODES and N in ISOMER_NODES:
        bonus += E_surface * MAGIC_BONUS * 0.5

    nz_ratio = N / Z if Z > 0 else 0
    if 0.95 <= nz_ratio <= 1.15:
        bonus += E_surface * symm_bonus
    if 1.15 <= nz_ratio <= 1.30:
        bonus += E_surface * nr_bonus

    return bonus

def qfd_energy(A, Z, symm_bonus, nr_bonus):
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
    E_iso = -get_resonance_bonus(Z, N, E_surface, symm_bonus, nr_bonus)

    E_pair = 0
    if Z % 2 == 0 and N % 2 == 0:
        E_pair = -DELTA_PAIRING / np.sqrt(A)
    elif Z % 2 == 1 and N % 2 == 1:
        E_pair = +DELTA_PAIRING / np.sqrt(A)

    return E_bulk + E_surf + E_asym + E_vac + E_iso + E_pair

def find_stable_Z(A, symm_bonus, nr_bonus):
    best_Z, best_E = 1, qfd_energy(A, 1, symm_bonus, nr_bonus)
    for Z in range(1, A):
        E = qfd_energy(A, Z, symm_bonus, nr_bonus)
        if E < best_E:
            best_E, best_Z = E, Z
    return best_Z

# Load data
with open('qfd_optimized_suite.py', 'r') as f:
    content = f.read()
start = content.find('test_nuclides = [')
end = content.find(']', start) + 1
test_nuclides = eval(content[start:end].replace('test_nuclides = ', ''))

print("="*80)
print("TESTING VERY STRONG SYMMETRIC BONUS")
print("="*80)
print()
print("Trying symm_bonus from 0.20 to 1.00 (nr_bonus fixed at 0.10)")
print()

# Test very strong symmetric bonuses
symm_values = [0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 1.00]
nr_bonus = 0.10

print(f"{'Symm':<8} {'Total Exact':<20} {'Improvement':<15} {'Ca-40':<10} {'S-32':<10} {'Ni-58'}")
print("-"*80)

baseline_exact = 178

for symm in symm_values:
    exact = sum(1 for name, Z_exp, A in test_nuclides
                if find_stable_Z(A, symm, nr_bonus) == Z_exp)

    pct = 100 * exact / len(test_nuclides)
    improvement = exact - baseline_exact

    # Check specific cases
    Z_ca40 = find_stable_Z(40, symm, nr_bonus)
    Z_s32 = find_stable_Z(32, symm, nr_bonus)
    Z_ni58 = find_stable_Z(58, symm, nr_bonus)

    ca40_status = "✓" if Z_ca40 == 20 else f"{Z_ca40}"
    s32_status = "✓" if Z_s32 == 16 else f"{Z_s32}"
    ni58_status = "✓" if Z_ni58 == 28 else f"{Z_ni58}"

    marker = "★" if exact > 183 else ""

    print(f"{symm:<8.2f} {exact}/{len(test_nuclides)} ({pct:.1f}%){'':<6} "
          f"{improvement:+d}{'':<13} {ca40_status:<10} {s32_status:<10} {ni58_status}  {marker}")

print()
print("="*80)
print("ENERGY LANDSCAPE FOR Ca-40")
print("="*80)
print()

# Show energy landscape for Ca-40 with different symmetric bonuses
A = 40
Z_exp = 20

print("Ca-40 (doubly magic Z=20, N=20, N/Z=1.00):")
print()

for symm in [0.15, 0.40, 0.70, 1.00]:
    print(f"symm_bonus = {symm:.2f}:")

    Z_range = range(16, 25)
    for Z in Z_range:
        E = qfd_energy(A, Z, symm, nr_bonus)
        Z_pred = find_stable_Z(A, symm, nr_bonus)
        marker = "←" if Z == Z_pred else ("*" if Z == Z_exp else "")
        print(f"  Z={Z}: E={E:.3f} MeV {marker}")

    E_exp = qfd_energy(A, Z_exp, symm, nr_bonus)
    E_pred = qfd_energy(A, Z_pred, symm, nr_bonus)
    gap = E_exp - E_pred
    print(f"  ΔE = {gap:.3f} MeV, pred={Z_pred}, exp={Z_exp}")
    print()

print("="*80)
