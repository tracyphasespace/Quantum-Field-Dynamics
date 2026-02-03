#!/usr/bin/env python3
"""
COMPREHENSIVE SIGN FLIP TEST
===========================================================================
Test BOTH positive and negative signs for all parameters:

1. Shield factor (currently 0.52 > 0)
   - Positive: Reduces displacement
   - Negative: ANTI-shielding (increases displacement)

2. Bonus strength (currently 0.70 > 0)
   - Positive: Magic numbers have LOWER energy (stable)
   - Negative: Magic numbers have HIGHER energy (anti-magic)

3. Vortex shielding κ_vortex
   - Positive: More electrons → less displacement penalty
   - Negative: More electrons → more displacement penalty

4. Electron temporal κ_e (already tested both signs)
   - For completeness, re-verify optimal sign

This could reveal if we have ANY term with the WRONG SIGN.
===========================================================================
"""

import numpy as np
from itertools import product

# Constants
alpha_fine   = 1.0 / 137.036
beta_vacuum  = 1.0 / 3.043233053
M_proton     = 938.272
LAMBDA_TIME_0 = 0.42

ISOMER_NODES = {2, 8, 20, 28, 50, 82, 126}

def get_resonance_bonus(Z, N, E_surface, bonus_strength):
    """Magic number bonus with SIGNED strength."""
    bonus = 0
    if Z in ISOMER_NODES: bonus += E_surface * bonus_strength
    if N in ISOMER_NODES: bonus += E_surface * bonus_strength
    if Z in ISOMER_NODES and N in ISOMER_NODES:
        bonus *= 1.5
    return bonus

def qfd_energy_signed(A, Z, shield_factor=0.52, bonus_strength=0.70,
                     kappa_vortex=0.0, kappa_e=0.0001):
    """
    QFD energy with SIGNED parameters.

    All parameters can be positive OR negative to test physics direction.
    """
    N = A - Z
    q = Z / A if A > 0 else 0

    # Temporal metric (signed κ_e)
    lambda_time = LAMBDA_TIME_0 + kappa_e * Z

    V_0 = M_proton * (1 - (alpha_fine**2) / beta_vacuum)
    beta_nuclear = M_proton * beta_vacuum / 2
    E_volume  = V_0 * (1 - lambda_time / (12 * np.pi))
    E_surface = beta_nuclear / 15
    a_sym     = (beta_vacuum * M_proton) / 15

    # Standard terms
    E_bulk = E_volume * A
    E_surf = E_surface * (A**(2/3))
    E_asym = a_sym * A * ((1 - 2*q)**2)

    # Displacement with SIGNED shielding
    a_disp_bare = (alpha_fine * 197.327 / 1.2)

    # Vortex shielding (signed κ_vortex)
    if kappa_vortex != 0:
        # Simple linear form for sign test
        vortex_factor = 1 + kappa_vortex * Z
        shield_total = shield_factor * vortex_factor
    else:
        shield_total = shield_factor

    a_disp = a_disp_bare * shield_total
    E_vac = a_disp * (Z**2) / (A**(1/3))

    # Magic number bonus (SIGNED)
    E_iso = -get_resonance_bonus(Z, N, E_surface, bonus_strength)

    return E_bulk + E_surf + E_asym + E_vac + E_iso

def find_stable_Z_signed(A, **kwargs):
    """Find stable Z with signed parameters."""
    best_Z = 1
    best_E = qfd_energy_signed(A, 1, **kwargs)
    for Z in range(1, A):
        E = qfd_energy_signed(A, Z, **kwargs)
        if E < best_E:
            best_E = E
            best_Z = Z
    return best_Z

# Load data
with open('qfd_optimized_suite.py', 'r') as f:
    content = f.read()
start = content.find('test_nuclides = [')
end = content.find(']', start) + 1
test_nuclides = eval(content[start:end].replace('test_nuclides = ', ''))

print("="*95)
print("COMPREHENSIVE SIGN FLIP TEST")
print("="*95)
print()
print("Testing BOTH positive and negative signs for all parameters")
print()

# Baseline
baseline_exact = sum(1 for name, Z_exp, A in test_nuclides
                     if find_stable_Z_signed(A) == Z_exp)

print(f"Baseline (shield=0.52, bonus=0.70, κ_vortex=0, κ_e=0.0001):")
print(f"  {baseline_exact}/{len(test_nuclides)} ({100*baseline_exact/len(test_nuclides):.1f}%)")
print()

# ============================================================================
# TEST 1: SHIELD FACTOR SIGN
# ============================================================================
print("="*95)
print("TEST 1: Shield Factor Sign")
print("="*95)
print()
print("  Positive: Shields displacement (reduces Z² penalty)")
print("  Negative: ANTI-shields (increases Z² penalty)")
print()

shield_values = [-0.70, -0.52, -0.30, 0.30, 0.52, 0.70, 1.00]

best_shield = {'shield': 0.52, 'exact': baseline_exact}

for shield in shield_values:
    exact = sum(1 for name, Z_exp, A in test_nuclides
                if find_stable_Z_signed(A, shield_factor=shield) == Z_exp)

    pct = 100 * exact / len(test_nuclides)
    improvement = exact - baseline_exact

    marker = "★" if exact > best_shield['exact'] else ""

    print(f"  shield = {shield:+.2f}:  {exact}/{len(test_nuclides)} ({pct:.1f}%)  "
          f"({improvement:+d})  {marker}")

    if exact > best_shield['exact']:
        best_shield = {'shield': shield, 'exact': exact}

print()
if best_shield['shield'] != 0.52:
    print(f"✓ BETTER shield = {best_shield['shield']:.2f} "
          f"({best_shield['exact']}/{len(test_nuclides)}, "
          f"+{best_shield['exact'] - baseline_exact})")
else:
    print("= Baseline shield = 0.52 is optimal")

print()

# ============================================================================
# TEST 2: BONUS STRENGTH SIGN
# ============================================================================
print("="*95)
print("TEST 2: Bonus Strength Sign")
print("="*95)
print()
print("  Positive: Magic numbers STABILIZED (lower energy)")
print("  Negative: Magic numbers DESTABILIZED (anti-magic)")
print()

bonus_values = [-1.00, -0.70, -0.50, -0.30, 0.0, 0.30, 0.50, 0.70, 1.00]

best_bonus = {'bonus': 0.70, 'exact': baseline_exact}

for bonus in bonus_values:
    exact = sum(1 for name, Z_exp, A in test_nuclides
                if find_stable_Z_signed(A, bonus_strength=bonus) == Z_exp)

    pct = 100 * exact / len(test_nuclides)
    improvement = exact - baseline_exact

    marker = "★" if exact > best_bonus['exact'] else ""

    print(f"  bonus = {bonus:+.2f}:  {exact}/{len(test_nuclides)} ({pct:.1f}%)  "
          f"({improvement:+d})  {marker}")

    if exact > best_bonus['exact']:
        best_bonus = {'bonus': bonus, 'exact': exact}

print()
if best_bonus['bonus'] != 0.70:
    print(f"✓ BETTER bonus = {best_bonus['bonus']:.2f} "
          f"({best_bonus['exact']}/{len(test_nuclides)}, "
          f"+{best_bonus['exact'] - baseline_exact})")
else:
    print("= Baseline bonus = 0.70 is optimal")

print()

# ============================================================================
# TEST 3: VORTEX SHIELDING SIGN (Linear)
# ============================================================================
print("="*95)
print("TEST 3: Vortex Shielding Sign (Linear κ_vortex)")
print("="*95)
print()
print("  Positive: More electrons → LESS displacement penalty")
print("  Negative: More electrons → MORE displacement penalty")
print()

kappa_vortex_values = [-0.03, -0.02, -0.01, -0.005, 0.0,
                       0.005, 0.01, 0.02, 0.03]

best_kappa_v = {'kappa': 0.0, 'exact': baseline_exact}

for kappa in kappa_vortex_values:
    exact = sum(1 for name, Z_exp, A in test_nuclides
                if find_stable_Z_signed(A, kappa_vortex=kappa) == Z_exp)

    pct = 100 * exact / len(test_nuclides)
    improvement = exact - baseline_exact

    marker = "★" if exact > best_kappa_v['exact'] else ""

    print(f"  κ_vortex = {kappa:+.3f}:  {exact}/{len(test_nuclides)} ({pct:.1f}%)  "
          f"({improvement:+d})  {marker}")

    if exact > best_kappa_v['exact']:
        best_kappa_v = {'kappa': kappa, 'exact': exact}

print()
if best_kappa_v['kappa'] != 0.0:
    print(f"✓ BETTER κ_vortex = {best_kappa_v['kappa']:.3f} "
          f"({best_kappa_v['exact']}/{len(test_nuclides)}, "
          f"+{best_kappa_v['exact'] - baseline_exact})")
else:
    print("= κ_vortex = 0 is optimal (no vortex shielding)")

print()

# ============================================================================
# TEST 4: ELECTRON TEMPORAL SIGN (Re-verify)
# ============================================================================
print("="*95)
print("TEST 4: Electron Temporal Sign (Re-verify κ_e)")
print("="*95)
print()
print("  Positive: More electrons → RAISE λ_time → LOWER E_volume")
print("  Negative: More electrons → LOWER λ_time → RAISE E_volume")
print()

kappa_e_values = [-0.005, -0.002, -0.001, -0.0001, 0.0,
                  0.0001, 0.0002, 0.0005, 0.001, 0.002]

best_kappa_e = {'kappa': 0.0001, 'exact': baseline_exact}

for kappa in kappa_e_values:
    exact = sum(1 for name, Z_exp, A in test_nuclides
                if find_stable_Z_signed(A, kappa_e=kappa) == Z_exp)

    pct = 100 * exact / len(test_nuclides)
    improvement = exact - baseline_exact

    marker = "★" if exact > best_kappa_e['exact'] else ""

    print(f"  κ_e = {kappa:+.4f}:  {exact}/{len(test_nuclides)} ({pct:.1f}%)  "
          f"({improvement:+d})  {marker}")

    if exact > best_kappa_e['exact']:
        best_kappa_e = {'kappa': kappa, 'exact': exact}

print()
if best_kappa_e['kappa'] != 0.0001:
    print(f"✓ DIFFERENT OPTIMAL: κ_e = {best_kappa_e['kappa']:.4f} "
          f"({best_kappa_e['exact']}/{len(test_nuclides)}, "
          f"+{best_kappa_e['exact'] - baseline_exact})")
else:
    print("= κ_e = +0.0001 confirmed optimal")

print()

# ============================================================================
# FINAL RESULTS
# ============================================================================
print("="*95)
print("COMPREHENSIVE SIGN FLIP RESULTS")
print("="*95)
print()

print(f"Baseline:  {baseline_exact}/{len(test_nuclides)} ({100*baseline_exact/len(test_nuclides):.1f}%)")
print()

sign_changes = []

if best_shield['shield'] != 0.52:
    sign_changes.append(f"  Shield: {0.52:.2f} → {best_shield['shield']:.2f} "
                       f"(+{best_shield['exact'] - baseline_exact})")

if best_bonus['bonus'] != 0.70:
    sign_changes.append(f"  Bonus: {0.70:.2f} → {best_bonus['bonus']:.2f} "
                       f"(+{best_bonus['exact'] - baseline_exact})")

if best_kappa_v['kappa'] != 0.0:
    sign_changes.append(f"  κ_vortex: {0.0:.3f} → {best_kappa_v['kappa']:.3f} "
                       f"(+{best_kappa_v['exact'] - baseline_exact})")

if best_kappa_e['kappa'] != 0.0001:
    sign_changes.append(f"  κ_e: {0.0001:.4f} → {best_kappa_e['kappa']:.4f} "
                       f"(+{best_kappa_e['exact'] - baseline_exact})")

if sign_changes:
    print("✓ SIGN CHANGES FOUND:")
    print()
    for change in sign_changes:
        print(change)
    print()
    print("⚠️  Some parameters had WRONG SIGN!")
else:
    print("= NO SIGN CHANGES")
    print()
    print("All baseline parameter signs confirmed optimal.")

print()

# ============================================================================
# COMBINED OPTIMIZATION (if any changes found)
# ============================================================================
if sign_changes:
    print("="*95)
    print("COMBINED OPTIMIZATION (All Best Signs Together)")
    print("="*95)
    print()

    combined_exact = sum(1 for name, Z_exp, A in test_nuclides
                        if find_stable_Z_signed(A,
                                               shield_factor=best_shield['shield'],
                                               bonus_strength=best_bonus['bonus'],
                                               kappa_vortex=best_kappa_v['kappa'],
                                               kappa_e=best_kappa_e['kappa']) == Z_exp)

    combined_pct = 100 * combined_exact / len(test_nuclides)
    combined_improvement = combined_exact - baseline_exact

    print(f"Baseline:  {baseline_exact}/{len(test_nuclides)} ({100*baseline_exact/len(test_nuclides):.1f}%)")
    print(f"Combined:  {combined_exact}/{len(test_nuclides)} ({combined_pct:.1f}%)")
    print()
    print(f"Total improvement: {combined_improvement:+d} matches ({combined_improvement/len(test_nuclides)*100:+.1f} pp)")

    if combined_improvement > 0:
        print()
        print("✓✓ BREAKTHROUGH: Sign corrections improve predictions!")
        print()
        print("Optimal configuration:")
        print(f"  shield_factor = {best_shield['shield']:.3f}")
        print(f"  bonus_strength = {best_bonus['bonus']:.3f}")
        print(f"  κ_vortex = {best_kappa_v['kappa']:.4f}")
        print(f"  κ_e = {best_kappa_e['kappa']:.4f}")

print()
print("="*95)
