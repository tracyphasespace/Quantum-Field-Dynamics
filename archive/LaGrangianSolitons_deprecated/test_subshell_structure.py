#!/usr/bin/env python3
"""
TEST SUBSHELL STRUCTURE BONUSES
===========================================================================
Current state: 186/285 (65.3%)
Heavy nuclei: 58.3% success

Mass-dependent asymmetry FAILED - constant coefficient is optimal!

Next hypothesis: SUBSHELL CLOSURES beyond magic numbers

Magic numbers: {2, 8, 20, 28, 50, 82, 126}

Observed subshell closures in nuclear physics:
  Z subshells: 6 (C), 14 (Si), 16 (S), 32 (Ge), 34 (Se), 38 (Sr), 40 (Zr)
  N subshells: 6, 14, 16, 32, 34, 40, 56, 64, 70

Pattern from failures:
  S-32  (Z=16): ΔZ=-2  ← Z=16 subshell?
  Ar-36 (Z=18): ΔZ=-2
  Ca-40 (Z=20): ΔZ=-2  ← Z=20 magic, but still wrong
  Cr-50 (Z=24): ΔZ=-2
  Fe-54 (Z=26): ΔZ=-2
  Ni-58 (Z=28): ΔZ=-2  ← Z=28 magic, but still wrong
  Zn-64 (Z=30): ΔZ=-2  ← Z=30 subshell?

Test adding weak bonuses at subshell closures.
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

# OPTIMAL DUAL-RESONANCE CONFIGURATION
MAGIC_BONUS = 0.10
SYMM_BONUS = 0.30
NR_BONUS = 0.10
DELTA_PAIRING = 11.0

ISOMER_NODES = {2, 8, 20, 28, 50, 82, 126}

# Subshell closures (empirically observed in nuclear physics)
# These are weaker than magic numbers but still show enhanced stability
SUBSHELL_Z = {6, 14, 16, 32, 34, 38, 40}
SUBSHELL_N = {6, 14, 16, 32, 34, 40, 56, 64, 70}

def get_resonance_bonus(Z, N, E_surface, subshell_bonus):
    """
    Resonance bonus with subshell structure.

    subshell_bonus: Bonus strength for subshell closures (< magic bonus)
    """
    bonus = 0

    # Magic numbers (strong)
    if Z in ISOMER_NODES: bonus += E_surface * MAGIC_BONUS
    if N in ISOMER_NODES: bonus += E_surface * MAGIC_BONUS
    if Z in ISOMER_NODES and N in ISOMER_NODES:
        bonus += E_surface * MAGIC_BONUS * 0.5

    # Subshell closures (weak)
    if Z in SUBSHELL_Z: bonus += E_surface * subshell_bonus
    if N in SUBSHELL_N: bonus += E_surface * subshell_bonus

    # Charge fraction resonance (dual windows)
    nz_ratio = N / Z if Z > 0 else 0
    if 0.95 <= nz_ratio <= 1.15:
        bonus += E_surface * SYMM_BONUS
    if 1.15 <= nz_ratio <= 1.30:
        bonus += E_surface * NR_BONUS

    return bonus

def qfd_energy(A, Z, subshell_bonus):
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
    E_iso = -get_resonance_bonus(Z, N, E_surface, subshell_bonus)

    E_pair = 0
    if Z % 2 == 0 and N % 2 == 0:
        E_pair = -DELTA_PAIRING / np.sqrt(A)
    elif Z % 2 == 1 and N % 2 == 1:
        E_pair = +DELTA_PAIRING / np.sqrt(A)

    return E_bulk + E_surf + E_asym + E_vac + E_iso + E_pair

def find_stable_Z(A, subshell_bonus):
    best_Z, best_E = 1, qfd_energy(A, 1, subshell_bonus)
    for Z in range(1, A):
        E = qfd_energy(A, Z, subshell_bonus)
        if E < best_E:
            best_E, best_Z = E, Z
    return best_Z

# Load data
with open('qfd_optimized_suite.py', 'r') as f:
    content = f.read()
start = content.find('test_nuclides = [')
end = content.find(']', start) + 1
test_nuclides = eval(content[start:end].replace('test_nuclides = ', ''))

print("="*90)
print("TESTING SUBSHELL STRUCTURE BONUSES")
print("="*90)
print()
print("Current (no subshells): 186/285 (65.3%)")
print(f"  Magic bonus: {MAGIC_BONUS}")
print()
print("Subshell closures:")
print(f"  Z subshells: {sorted(SUBSHELL_Z)}")
print(f"  N subshells: {sorted(SUBSHELL_N)}")
print()
print("Testing subshell bonus strengths (< magic bonus):")
print()

baseline_exact = 186

# Test different subshell bonus strengths
subshell_values = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]

print(f"{'Subshell':<12} {'Total':<20} {'Light':<12} {'Heavy':<12} {'Improvement'}")
print("-"*90)

for sub_bonus in subshell_values:
    exact = sum(1 for name, Z_exp, A in test_nuclides
                if find_stable_Z(A, sub_bonus) == Z_exp)

    # Light nuclei (A<40)
    light_correct = sum(1 for name, Z_exp, A in test_nuclides
                        if A < 40 and find_stable_Z(A, sub_bonus) == Z_exp)
    light_total = sum(1 for name, Z_exp, A in test_nuclides if A < 40)

    # Heavy nuclei (A≥100)
    heavy_correct = sum(1 for name, Z_exp, A in test_nuclides
                        if A >= 100 and find_stable_Z(A, sub_bonus) == Z_exp)
    heavy_total = sum(1 for name, Z_exp, A in test_nuclides if A >= 100)

    pct = 100 * exact / len(test_nuclides)
    improvement = exact - baseline_exact
    light_pct = 100 * light_correct / light_total
    heavy_pct = 100 * heavy_correct / heavy_total

    marker = "★" if exact > baseline_exact else ""

    print(f"{sub_bonus:<12.2f} {exact}/{len(test_nuclides)} ({pct:.1f}%){'':<6} "
          f"{light_pct:<12.1f} {heavy_pct:<12.1f} {improvement:+d}  {marker}")

print()

# ============================================================================
# TEST SPECIFIC CASES
# ============================================================================
print("="*90)
print("SPECIFIC SUBSHELL CASES")
print("="*90)
print()

# These are the stubborn ΔZ=-2 failures with Z at or near subshells
test_cases = [
    ('S-32', 16, 32),    # Z=16 subshell
    ('Ar-36', 18, 36),   # Z=18 (near Z=20 magic)
    ('Ca-40', 20, 40),   # Z=20 magic
    ('Cr-50', 24, 50),   # Between Z=20 and Z=28
    ('Fe-54', 26, 54),   # Near Z=28 magic
    ('Ni-58', 28, 58),   # Z=28 magic
    ('Zn-64', 30, 64),   # Z=30 (near Z=28)
    ('Ge-76', 32, 76),   # Z=32 subshell
    ('Se-74', 34, 74),   # Z=34 subshell
]

for case_name, case_Z, case_A in test_cases:
    if (case_name, case_Z, case_A) not in test_nuclides:
        continue

    Z_pred_baseline = find_stable_Z(case_A, 0.00)
    Z_pred_sub = find_stable_Z(case_A, 0.05)

    N_exp = case_A - case_Z
    Z_at_subshell = case_Z in SUBSHELL_Z
    N_at_subshell = N_exp in SUBSHELL_N
    Z_at_magic = case_Z in ISOMER_NODES
    N_at_magic = N_exp in ISOMER_NODES

    status_baseline = "✓" if Z_pred_baseline == case_Z else f"✗ {Z_pred_baseline}"
    status_sub = "✓" if Z_pred_sub == case_Z else f"✗ {Z_pred_sub}"

    notes = []
    if Z_at_magic: notes.append("Z_mag")
    elif Z_at_subshell: notes.append("Z_sub")
    if N_at_magic: notes.append("N_mag")
    elif N_at_subshell: notes.append("N_sub")

    improvement = "FIXED" if Z_pred_baseline != case_Z and Z_pred_sub == case_Z else ""

    print(f"{case_name:<10} (Z={case_Z:2d}, N={N_exp:2d}): "
          f"baseline {status_baseline:<8} → sub=0.05 {status_sub:<8} "
          f"{', '.join(notes):<20} {improvement}")

print()

# ============================================================================
# SUMMARY
# ============================================================================
print("="*90)
print("SUMMARY")
print("="*90)
print()
print("Baseline (no subshells): 186/285 (65.3%)")
print("  Light: 92.3%, Heavy: 58.3%")
print()
print("Subshell bonuses test whether intermediate shell closures")
print("(Z=14, 16, 32, 34, etc.) provide additional stability.")
print()
print("If no improvement: Shell structure is already captured by magic numbers")
print("                   and resonance windows. Missing physics is elsewhere.")
print()
print("="*90)
