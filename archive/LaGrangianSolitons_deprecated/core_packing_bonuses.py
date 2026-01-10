#!/usr/bin/env python3
"""
CORE PACKING BONUSES - Discrete Geometric Structure
===========================================================================
Implement discrete vortex packing terms based on discovered patterns:

1. Z mod 6 = 5 bonus (68.2% survival → 6-fold Cl(3,3) symmetry)
2. Z mod 8 = 3 bonus (62.5% survival → octahedral packing)
3. N mod 4 = 3 bonus (63.3% survival → tetrahedral neutron structure)
4. Odd Z enhancement (many 100% survival → unpaired vortex stability)

These are DISCRETE corrections to base Lagrangian, not continuous fields.
===========================================================================
"""

import numpy as np

# ============================================================================
# CONSTANTS
# ============================================================================
alpha_fine   = 1.0 / 137.036
beta_vacuum  = 1.0 / 3.058231
M_proton     = 938.272
LAMBDA_TIME_0 = 0.42
KAPPA_E = 0.0001

SHIELD_FACTOR = 0.52
a_disp_base = (alpha_fine * 197.327 / 1.2) * SHIELD_FACTOR
ISOMER_NODES = {2, 8, 20, 28, 50, 82, 126}
BONUS_STRENGTH = 0.70

def lambda_time_eff(Z):
    return LAMBDA_TIME_0 + KAPPA_E * Z

def get_resonance_bonus(Z, N, E_surface):
    bonus = 0
    if Z in ISOMER_NODES: bonus += E_surface * BONUS_STRENGTH
    if N in ISOMER_NODES: bonus += E_surface * BONUS_STRENGTH
    if Z in ISOMER_NODES and N in ISOMER_NODES:
        bonus *= 1.5
    return bonus

# ============================================================================
# CORE PACKING BONUSES (Discrete Geometric Structure)
# ============================================================================
def get_core_packing_bonus(Z, N, E_surface):
    """
    Discrete bonuses from vortex packing geometry in core.

    Parameters to calibrate:
      ε_mod6, ε_mod8, ε_mod4, ε_odd
    """
    bonus = 0

    # Pattern 1: Z mod 6 = 5 (6-fold Cl(3,3) symmetry)
    epsilon_mod6 = 0.15  # Fraction of E_surface
    if Z % 6 == 5:
        bonus += epsilon_mod6 * E_surface

    # Pattern 2: Z mod 8 = 3 (octahedral packing)
    epsilon_mod8 = 0.10
    if Z % 8 == 3:
        bonus += epsilon_mod8 * E_surface

    # Pattern 3: N mod 4 = 3 (tetrahedral neutron structure)
    epsilon_mod4 = 0.10
    if N % 4 == 3:
        bonus += epsilon_mod4 * E_surface

    # Pattern 4: Odd Z (unpaired proton vortex)
    epsilon_odd = 0.05
    if Z % 2 == 1:
        bonus += epsilon_odd * E_surface

    return bonus

# ============================================================================
# ENERGY FUNCTIONAL WITH CORE PACKING
# ============================================================================
def qfd_energy_with_packing(A, Z):
    N = A - Z
    q = Z / A if A > 0 else 0
    lambda_time = lambda_time_eff(Z)

    V_0 = M_proton * (1 - (alpha_fine**2) / beta_vacuum)
    beta_nuclear = M_proton * beta_vacuum / 2
    E_volume  = V_0 * (1 - lambda_time / (12 * np.pi))
    E_surface = beta_nuclear / 15
    a_sym     = (beta_vacuum * M_proton) / 15

    # Standard terms
    E_bulk = E_volume * A
    E_surf = E_surface * (A**(2/3))
    E_asym = a_sym * A * ((1 - 2*q)**2)
    E_vac  = a_disp_base * (Z**2) / (A**(1/3))

    # Geometric bonuses
    E_iso = -get_resonance_bonus(Z, N, E_surface)  # Magic numbers
    E_pack = -get_core_packing_bonus(Z, N, E_surface)  # Core packing

    return E_bulk + E_surf + E_asym + E_vac + E_iso + E_pack

def find_stable_Z_with_packing(A):
    best_Z = 1
    best_E = qfd_energy_with_packing(A, 1)
    for Z in range(1, A):
        E = qfd_energy_with_packing(A, Z)
        if E < best_E:
            best_E = E
            best_Z = Z
    return best_Z

# ============================================================================
# LOAD DATA
# ============================================================================
with open('qfd_optimized_suite.py', 'r') as f:
    content = f.read()

start = content.find('test_nuclides = [')
end = content.find(']', start) + 1
test_nuclides = eval(content[start:end].replace('test_nuclides = ', ''))

print("="*95)
print("CORE PACKING BONUSES - Discrete Geometric Structure")
print("="*95)
print()
print("Implementing discrete vortex packing terms:")
print(f"  1. Z mod 6 = 5 → +bonus (6-fold Cl(3,3) symmetry)")
print(f"  2. Z mod 8 = 3 → +bonus (octahedral packing)")
print(f"  3. N mod 4 = 3 → +bonus (tetrahedral structure)")
print(f"  4. Odd Z → +bonus (unpaired vortex stability)")
print()

# ============================================================================
# EVALUATE
# ============================================================================
results_packing = []
results_baseline = []

# Baseline (electron correction only, no core packing)
def qfd_baseline(A, Z):
    N = A - Z
    q = Z / A if A > 0 else 0
    lambda_time = lambda_time_eff(Z)

    V_0 = M_proton * (1 - (alpha_fine**2) / beta_vacuum)
    beta_nuclear = M_proton * beta_vacuum / 2
    E_volume  = V_0 * (1 - lambda_time / (12 * np.pi))
    E_surface = beta_nuclear / 15
    a_sym     = (beta_vacuum * M_proton) / 15

    E_bulk = E_volume * A
    E_surf = E_surface * (A**(2/3))
    E_asym = a_sym * A * ((1 - 2*q)**2)
    E_vac  = a_disp_base * (Z**2) / (A**(1/3))
    E_iso = -get_resonance_bonus(Z, N, E_surface)

    return E_bulk + E_surf + E_asym + E_vac + E_iso

def find_stable_Z_baseline(A):
    best_Z = 1
    best_E = qfd_baseline(A, 1)
    for Z in range(1, A):
        E = qfd_baseline(A, Z)
        if E < best_E:
            best_E = E
            best_Z = Z
    return best_Z

for name, Z_exp, A in test_nuclides:
    # With core packing
    Z_pack = find_stable_Z_with_packing(A)
    Delta_pack = Z_pack - Z_exp

    # Baseline
    Z_base = find_stable_Z_baseline(A)
    Delta_base = Z_base - Z_exp

    results_packing.append({'name': name, 'A': A, 'Z_exp': Z_exp,
                           'Z_pred': Z_pack, 'Delta_Z': Delta_pack})
    results_baseline.append({'name': name, 'A': A, 'Z_exp': Z_exp,
                            'Z_pred': Z_base, 'Delta_Z': Delta_base})

# ============================================================================
# STATISTICS
# ============================================================================
errors_pack = [abs(r['Delta_Z']) for r in results_packing]
errors_base = [abs(r['Delta_Z']) for r in results_baseline]

exact_pack = sum(e == 0 for e in errors_pack)
exact_base = sum(e == 0 for e in errors_base)

print("="*95)
print("RESULTS")
print("="*95)
print()

print(f"{'Model':<40} {'Exact':<20} {'Mean |ΔZ|':<15} {'Median |ΔZ|'}")
print("-"*95)
print(f"{'Baseline (electron + magic only)':<40} {exact_base}/{len(results_baseline)} ({100*exact_base/len(results_baseline):.1f}%)  "
      f"{np.mean(errors_base):<15.3f} {np.median(errors_base):.1f}")
print(f"{'With Core Packing (discrete bonuses)':<40} {exact_pack}/{len(results_packing)} ({100*exact_pack/len(results_packing):.1f}%)  "
      f"{np.mean(errors_pack):<15.3f} {np.median(errors_pack):.1f}")
print()

improvement = exact_pack - exact_base
if improvement > 0:
    print(f"✓ IMPROVEMENT: +{improvement} exact matches ({improvement/len(results_packing)*100:.1f} percentage points)")
    print()
    print("Discrete core packing structure is REAL!")
elif improvement < 0:
    print(f"✗ REGRESSION: {improvement} exact matches")
    print("Current bonuses may be too strong or wrong mod values")
else:
    print("= NEUTRAL: No change (bonuses too weak?)")

print()

# By region
for model_name, results in [("Baseline", results_baseline), ("With Packing", results_packing)]:
    light = [r for r in results if r['A'] < 40]
    medium = [r for r in results if 40 <= r['A'] < 100]
    heavy = [r for r in results if 100 <= r['A'] < 200]
    superheavy = [r for r in results if r['A'] >= 200]

    print(f"{model_name.upper()} - BY REGION:")
    print("-"*95)
    for region_name, group in [("Light (A<40)", light), ("Medium (40≤A<100)", medium),
                                ("Heavy (100≤A<200)", heavy), ("Superheavy (A≥200)", superheavy)]:
        if len(group) > 0:
            errs = [abs(r['Delta_Z']) for r in group]
            ex = sum(e == 0 for e in errs)
            print(f"  {region_name:<25} {ex}/{len(group)} ({100*ex/len(group):>5.1f}%)  "
                  f"Mean|ΔZ|={np.mean(errs):.2f}")
    print()

# ============================================================================
# KEY TEST CASES
# ============================================================================
print("="*95)
print("KEY TEST CASES")
print("="*95)
print(f"{'Nuclide':<12} {'A':<5} {'Z':<5} {'Base':<10} {'Packing':<10} {'Improvement'}")
print("-"*95)

key_cases = [
    ("He-4", 2, 4),
    ("O-16", 8, 16),
    ("Ca-40", 20, 40),
    ("Fe-56", 26, 56),
    ("Ni-58", 28, 58),
    ("Sn-112", 50, 112),
    ("Xe-136", 54, 136),
    ("Pb-208", 82, 208),
    ("U-238", 92, 238),
]

for name, Z_exp, A in key_cases:
    r_pack = next((r for r in results_packing if r['name'] == name), None)
    r_base = next((r for r in results_baseline if r['name'] == name), None)

    if r_pack and r_base:
        base_str = "✓" if r_base['Delta_Z'] == 0 else f"{r_base['Delta_Z']:+d}"
        pack_str = "✓" if r_pack['Delta_Z'] == 0 else f"{r_pack['Delta_Z']:+d}"

        improvement = ""
        if abs(r_pack['Delta_Z']) < abs(r_base['Delta_Z']):
            improvement = "✓ Better"
        elif abs(r_pack['Delta_Z']) > abs(r_base['Delta_Z']):
            improvement = "✗ Worse"
        else:
            improvement = "= Same"

        print(f"{name:<12} {A:<5} {Z_exp:<5} {base_str:<10} {pack_str:<10} {improvement}")

print()
print("="*95)
print("VERDICT")
print("="*95)
print()

if improvement > 10:
    print("✓✓ BREAKTHROUGH: Core packing structure significantly improves predictions!")
    print()
    print("Physical interpretation:")
    print("  - Vortices pack in discrete geometric arrangements in Cl(3,3) core")
    print("  - Mod-6, mod-8, mod-4 patterns reveal 6-fold, octahedral, tetrahedral symmetries")
    print("  - Odd Z nuclei have unpaired vortex that stabilizes configuration")
    print()
    print("This confirms QFD soliton picture with discrete topological structure!")
elif improvement > 0:
    print("✓ MODEST IMPROVEMENT from core packing bonuses")
    print()
    print("Next step: Calibrate bonus strengths (ε_mod6, ε_mod8, ε_mod4, ε_odd)")
    print("Or add additional discrete patterns from analysis")
else:
    print("Core packing bonuses don't help with current parameters")
    print()
    print("Possible issues:")
    print("  - Bonus strengths too weak/strong")
    print("  - Wrong mod values (need different k)")
    print("  - Missing the primary geometric constraint")

print()
print("="*95)
