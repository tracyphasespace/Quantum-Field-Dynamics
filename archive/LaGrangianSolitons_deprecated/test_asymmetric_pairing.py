#!/usr/bin/env python3
"""
TEST ASYMMETRIC PAIRING ENERGY
===========================================================================
The analysis shows odd-odd nuclei have CATASTROPHIC 77.8% fail rate!

Current: E_pair(even-even) = -11.0/√A, E_pair(odd-odd) = +11.0/√A

This is clearly insufficient. Test much stronger odd-odd destabilization:
- δ_ee = 11.0 MeV (stabilize even-even, keep same)
- δ_oo = 15, 20, 25, 30 MeV (strongly destabilize odd-odd)
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
ISOMER_NODES = {2, 8, 20, 28, 50, 82, 126}

def get_resonance_bonus(Z, N, E_surface):
    bonus = 0
    if Z in ISOMER_NODES: bonus += E_surface * MAGIC_BONUS
    if N in ISOMER_NODES: bonus += E_surface * MAGIC_BONUS
    if Z in ISOMER_NODES and N in ISOMER_NODES:
        bonus += E_surface * MAGIC_BONUS * 0.5
    nz_ratio = N / Z if Z > 0 else 0
    if NZ_LOW <= nz_ratio <= NZ_HIGH:
        bonus += E_surface * NZ_BONUS
    return bonus

def qfd_energy(A, Z, delta_ee, delta_oo):
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
    E_iso = -get_resonance_bonus(Z, N, E_surface)

    # ASYMMETRIC PAIRING
    E_pair = 0
    if Z % 2 == 0 and N % 2 == 0:
        E_pair = -delta_ee / np.sqrt(A)
    elif Z % 2 == 1 and N % 2 == 1:
        E_pair = +delta_oo / np.sqrt(A)  # Stronger destabilization

    return E_bulk + E_surf + E_asym + E_vac + E_iso + E_pair

def find_stable_Z(A, delta_ee, delta_oo):
    best_Z, best_E = 1, qfd_energy(A, 1, delta_ee, delta_oo)
    for Z in range(1, A):
        E = qfd_energy(A, Z, delta_ee, delta_oo)
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
print("TESTING ASYMMETRIC PAIRING ENERGY")
print("="*80)
print()
print("Current (symmetric): δ_ee = δ_oo = 11.0 MeV → 178/285 (62.5%)")
print()
print("Hypothesis: Odd-odd needs MUCH stronger destabilization")
print()

# Test asymmetric pairing
delta_ee = 11.0  # Keep even-even stabilization same
delta_oo_values = [11.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]

print(f"{'δ_oo (MeV)':<12} {'Total Exact':<20} {'EE Fail':<12} {'OO Fail':<12} {'Improvement'}")
print("-"*80)

baseline_exact = 178

for delta_oo in delta_oo_values:
    exact = sum(1 for name, Z_exp, A in test_nuclides
                if find_stable_Z(A, delta_ee, delta_oo) == Z_exp)

    pct = 100 * exact / len(test_nuclides)
    improvement = exact - baseline_exact

    # Even-even fail rate
    ee_total = sum(1 for name, Z_exp, A in test_nuclides
                   if Z_exp % 2 == 0 and (A-Z_exp) % 2 == 0)
    ee_correct = sum(1 for name, Z_exp, A in test_nuclides
                     if Z_exp % 2 == 0 and (A-Z_exp) % 2 == 0
                     and find_stable_Z(A, delta_ee, delta_oo) == Z_exp)
    ee_fail_rate = 100 * (1 - ee_correct/ee_total)

    # Odd-odd fail rate
    oo_total = sum(1 for name, Z_exp, A in test_nuclides
                   if Z_exp % 2 == 1 and (A-Z_exp) % 2 == 1)
    oo_correct = sum(1 for name, Z_exp, A in test_nuclides
                     if Z_exp % 2 == 1 and (A-Z_exp) % 2 == 1
                     and find_stable_Z(A, delta_ee, delta_oo) == Z_exp)
    oo_fail_rate = 100 * (1 - oo_correct/oo_total) if oo_total > 0 else 0

    marker = "★" if exact > baseline_exact else ""

    print(f"{delta_oo:<12.1f} {exact}/{len(test_nuclides)} ({pct:.1f}%){'':<6} "
          f"{ee_fail_rate:<12.1f} {oo_fail_rate:<12.1f} {improvement:+d}  {marker}")

print()
print("="*80)
print("ANALYSIS")
print("="*80)
print()
print("Checking if asymmetric pairing helps...")
print()
print("Key metrics:")
print("  - Total exact matches (higher is better)")
print("  - Odd-odd fail rate (should decrease from 77.8%)")
print("  - Even-even fail rate (should stay ~40%)")
print()
print("="*80)
