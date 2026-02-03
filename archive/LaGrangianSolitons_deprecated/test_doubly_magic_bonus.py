#!/usr/bin/env python3
"""
TEST DOUBLY MAGIC NUCLEUS BONUS
===========================================================================
Ca-40 (Z=20, N=20) is DOUBLY MAGIC but still predicted wrong (Z=18)!
Energy gap is only 3.95 MeV.

Current doubly magic bonus:
  bonus = E_surface * 0.10 (Z magic)
        + E_surface * 0.10 (N magic)
        + E_surface * 0.05 (doubly magic extra)
        = E_surface * 0.25 total

Test much stronger doubly magic bonus to fix Ca-40, Ca-48, and others.
===========================================================================
"""

import numpy as np

# Constants
alpha_fine = 1.0 / 137.036
beta_vacuum = 1.0 / 3.043233053
M_proton = 938.272
LAMBDA_TIME_0 = 0.42
KAPPA_E = 0.0001
SHIELD_FACTOR = 0.52
MAGIC_BONUS = 0.10
NZ_LOW, NZ_HIGH = 1.15, 1.30
NZ_BONUS = 0.10
DELTA_PAIRING = 11.0
ISOMER_NODES = {2, 8, 20, 28, 50, 82, 126}

def get_resonance_bonus(Z, N, E_surface, doubly_magic_extra):
    bonus = 0
    if Z in ISOMER_NODES: bonus += E_surface * MAGIC_BONUS
    if N in ISOMER_NODES: bonus += E_surface * MAGIC_BONUS
    if Z in ISOMER_NODES and N in ISOMER_NODES:
        bonus += E_surface * doubly_magic_extra  # Test different values
    nz_ratio = N / Z if Z > 0 else 0
    if NZ_LOW <= nz_ratio <= NZ_HIGH:
        bonus += E_surface * NZ_BONUS
    return bonus

def qfd_energy(A, Z, doubly_magic_extra):
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
    E_iso = -get_resonance_bonus(Z, N, E_surface, doubly_magic_extra)

    # Pairing
    E_pair = 0
    if Z % 2 == 0 and N % 2 == 0:
        E_pair = -DELTA_PAIRING / np.sqrt(A)
    elif Z % 2 == 1 and N % 2 == 1:
        E_pair = +DELTA_PAIRING / np.sqrt(A)

    return E_bulk + E_surf + E_asym + E_vac + E_iso + E_pair

def find_stable_Z(A, doubly_magic_extra):
    best_Z, best_E = 1, qfd_energy(A, 1, doubly_magic_extra)
    for Z in range(1, A):
        E = qfd_energy(A, Z, doubly_magic_extra)
        if E < best_E:
            best_E, best_Z = E, Z
    return best_Z

# Load data
with open('qfd_optimized_suite.py', 'r') as f:
    content = f.read()
start = content.find('test_nuclides = [')
end = content.find(']', start) + 1
test_nuclides = eval(content[start:end].replace('test_nuclides = ', ''))

# Identify doubly magic nuclei in test set
doubly_magic_nuclides = [
    (name, Z, A) for name, Z, A in test_nuclides
    if Z in ISOMER_NODES and (A - Z) in ISOMER_NODES
]

print("="*80)
print("TESTING DOUBLY MAGIC NUCLEUS BONUS")
print("="*80)
print()
print(f"Current: doubly_magic_extra = 0.05 → Total bonus = 0.25 * E_surface")
print(f"Result: 178/285 (62.5%)")
print()
print(f"Doubly magic nuclei in test set: {len(doubly_magic_nuclides)}")
for name, Z, A in doubly_magic_nuclides:
    print(f"  {name} (Z={Z}, N={A-Z})")
print()

# Test different doubly magic bonus values
doubly_magic_values = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]

print(f"{'Double Magic':<15} {'Total Exact':<20} {'DM Correct':<15} {'Improvement'}")
print("-"*80)

baseline_exact = 178

for dm_extra in doubly_magic_values:
    exact = sum(1 for name, Z_exp, A in test_nuclides
                if find_stable_Z(A, dm_extra) == Z_exp)

    pct = 100 * exact / len(test_nuclides)
    improvement = exact - baseline_exact

    # Doubly magic correct count
    dm_correct = sum(1 for name, Z_exp, A in doubly_magic_nuclides
                     if find_stable_Z(A, dm_extra) == Z_exp)

    total_bonus = 2 * MAGIC_BONUS + dm_extra

    marker = "★" if exact > baseline_exact else ""

    print(f"{dm_extra:<15.2f} {exact}/{len(test_nuclides)} ({pct:.1f}%){'':<6} "
          f"{dm_correct}/{len(doubly_magic_nuclides):<12} {improvement:+d}  {marker}")

print()
print("="*80)
print("SPECIFIC DOUBLY MAGIC CASES")
print("="*80)
print()

# Check specific problematic cases
test_cases = [
    ('He-4', 2, 4),
    ('O-16', 8, 16),
    ('Ca-40', 20, 40),
    ('Ca-48', 20, 48),
]

for case_name, case_Z, case_A in test_cases:
    if (case_name, case_Z, case_A) not in test_nuclides:
        continue

    print(f"{case_name} (Z={case_Z}, N={case_A - case_Z}):")

    for dm_extra in [0.05, 0.20, 0.40]:
        Z_pred = find_stable_Z(case_A, dm_extra)
        E_exp = qfd_energy(case_A, case_Z, dm_extra)
        E_pred = qfd_energy(case_A, Z_pred, dm_extra)
        gap = E_exp - E_pred

        status = "✓" if Z_pred == case_Z else f"✗ (pred {Z_pred})"

        print(f"  dm_extra={dm_extra:.2f}: {status}, ΔE={gap:.2f} MeV")
    print()

print("="*80)
