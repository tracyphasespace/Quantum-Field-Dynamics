#!/usr/bin/env python3
"""
TEST DUAL RESONANCE WINDOWS
===========================================================================
PATTERN DISCOVERED: Systematic ΔZ=-2 for nuclei with N/Z ~ 1.0-1.1

All these failures are BELOW the current resonance [1.15, 1.30]:
  S-32, Ar-36, Ca-40, Ti-46, Cr-50, Fe-54, Ni-58 (all N/Z ≈ 1.0-1.1)

Hypothesis: Need TWO resonance windows:
  1. Symmetric window: N/Z ∈ [0.95, 1.15] (stabilize symmetric nuclei)
  2. Neutron-rich window: N/Z ∈ [1.15, 1.30] (current, heavier systems)

This makes physical sense in QFD:
  - Light nuclei prefer symmetry (N≈Z)
  - Heavy nuclei need excess neutrons (N>Z) for stability
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
    """Two resonance windows: symmetric + neutron-rich."""
    bonus = 0

    # Magic numbers
    if Z in ISOMER_NODES: bonus += E_surface * MAGIC_BONUS
    if N in ISOMER_NODES: bonus += E_surface * MAGIC_BONUS
    if Z in ISOMER_NODES and N in ISOMER_NODES:
        bonus += E_surface * MAGIC_BONUS * 0.5

    # Dual resonance windows
    nz_ratio = N / Z if Z > 0 else 0

    # Symmetric window (for light nuclei)
    if 0.95 <= nz_ratio <= 1.15:
        bonus += E_surface * symm_bonus

    # Neutron-rich window (for heavy nuclei)
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

    # Pairing
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
print("TESTING DUAL RESONANCE WINDOWS")
print("="*80)
print()
print("Current (single window): N/Z ∈ [1.15, 1.30], bonus=0.10")
print("                         → 178/285 (62.5%)")
print()
print("New (dual windows):")
print("  Window 1 (symmetric):  N/Z ∈ [0.95, 1.15]")
print("  Window 2 (neutron-rich): N/Z ∈ [1.15, 1.30]")
print()

# Test different bonus combinations
test_configs = [
    # (symm_bonus, nr_bonus, description)
    (0.00, 0.10, "No symm, keep nr=0.10 (baseline)"),
    (0.05, 0.10, "Weak symm, nr=0.10"),
    (0.10, 0.10, "Equal symm=nr=0.10"),
    (0.15, 0.10, "Strong symm, nr=0.10"),
    (0.20, 0.10, "Very strong symm, nr=0.10"),
    (0.10, 0.05, "symm=0.10, weak nr"),
    (0.10, 0.15, "symm=0.10, strong nr"),
]

print(f"{'Symm':<8} {'NR':<8} {'Total Exact':<20} {'Description':<35} {'Improv'}")
print("-"*80)

baseline_exact = 178

for symm, nr, desc in test_configs:
    exact = sum(1 for name, Z_exp, A in test_nuclides
                if find_stable_Z(A, symm, nr) == Z_exp)

    pct = 100 * exact / len(test_nuclides)
    improvement = exact - baseline_exact

    marker = "★" if exact > baseline_exact else ""

    print(f"{symm:<8.2f} {nr:<8.2f} {exact}/{len(test_nuclides)} ({pct:.1f}%){'':<6} "
          f"{desc:<35} {improvement:+d}  {marker}")

print()
print("="*80)
print("SPECIFIC TEST CASES (Symmetric Failures)")
print("="*80)
print()

# Test specific symmetric nuclei
test_cases = [
    ('S-32', 16, 32),
    ('Ar-36', 18, 36),
    ('Ca-40', 20, 40),
    ('Ti-46', 22, 46),
    ('Cr-50', 24, 50),
    ('Fe-54', 26, 54),
    ('Ni-58', 28, 58),
]

# Test with best configuration
best_symm, best_nr = 0.10, 0.10  # Equal bonuses

for case_name, case_Z, case_A in test_cases:
    if (case_name, case_Z, case_A) not in test_nuclides:
        continue

    Z_pred_baseline = find_stable_Z(case_A, 0.00, 0.10)
    Z_pred_dual = find_stable_Z(case_A, best_symm, best_nr)

    nz_ratio = (case_A - case_Z) / case_Z

    status_baseline = "✓" if Z_pred_baseline == case_Z else f"✗ pred={Z_pred_baseline}"
    status_dual = "✓" if Z_pred_dual == case_Z else f"✗ pred={Z_pred_dual}"

    improvement = "FIXED" if Z_pred_baseline != case_Z and Z_pred_dual == case_Z else ""

    print(f"{case_name:<8} (N/Z={nz_ratio:.2f}): baseline {status_baseline:<15} "
          f"→ dual {status_dual:<15} {improvement}")

print()
print("="*80)
