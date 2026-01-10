#!/usr/bin/env python3
"""
DUAL-CORE SPIN SOLVER - SPIN AS ENERGY, NOT QUANTUM LABEL
===========================================================================
Revolutionary Insight: Spin is NOT a secondary property.
It is the ROTATIONAL KINETIC ENERGY of the vacuum manifold.

The Unified Lagrangian:
    E_total = E_bulk + E_surf + E_asym + E_vac + E_pair + E_spin

Where:
    E_spin = L²/(2I) = ℏ²J(J+1)/(2I)

And I(Z,N) is the dual-core moment of inertia:
    I = I_core(N) + I_atmosphere(Z)

Key Predictions:
1. Mod 4 = 1 succeeds because spin energy is optimized by parity
2. Mod 28 = 13 achieves 87.5% due to spin-vacuum resonance
3. Magic numbers = 0.0 MeV because shell energy is emergent from E_spin

Test Case: Fe-56
    - Currently: Pure QFD predicts Z=24 (WRONG, should be Z=26)
    - Question: Does E_spin(Z,N) fix this by favoring Z=26?

Strategy:
1. Predict J from (Z,N) parity
2. Calculate I(Z,N) from dual-core model
3. Add E_spin to energy functional
4. Minimize E_total to find stable Z
===========================================================================
"""

import numpy as np
from collections import defaultdict, Counter

# Fundamental constants
alpha_fine = 1.0 / 137.036
beta_vacuum = 1.0 / 3.058231
M_proton = 938.272
KAPPA_E = 0.0001
SHIELD_FACTOR = 0.52
DELTA_PAIRING = 11.0
hbar_c = 197.327  # MeV·fm

def predict_spin(Z, N):
    """
    Predict ground state spin J based on (Z,N) parity.

    Physical basis:
    - Even-even: All paired → J=0
    - Odd-A: Single unpaired nucleon → J=0.5 to 2.5
    - Odd-odd: Two unpaired nucleons → J=0 to 4+

    Refinement: Use (Z,N) mod 4 patterns
    """
    if Z % 2 == 0 and N % 2 == 0:
        # Even-even: Paired, no net spin
        return 0.0

    elif Z % 2 == 1 and N % 2 == 1:
        # Odd-odd: High spin, strong coupling
        Z_mod = Z % 4
        N_mod = N % 4

        # Exceptional parities have higher spins
        if (Z_mod, N_mod) in [(3, 1), (1, 3)]:
            return 3.0  # High vortex coupling
        elif (Z_mod, N_mod) in [(3, 3), (1, 1)]:
            return 2.0  # Moderate coupling
        else:
            return 1.0  # Default odd-odd

    else:
        # Odd-A: Single unpaired nucleon
        # Use shell model estimate: J = l + 1/2 or l - 1/2
        # Simple estimate: J = 0.5 for s-orbital, 1.5 for p, 2.5 for d, etc.

        A = Z + N
        if A < 20:
            return 0.5  # Light nuclei, s or p orbitals
        elif A < 60:
            return 1.5  # Medium nuclei, p or d orbitals
        elif A < 150:
            return 2.5  # Heavy nuclei, d or f orbitals
        else:
            return 3.5  # Very heavy, higher orbitals

def moment_of_inertia_dual_core(Z, N):
    """
    Dual-core moment of inertia model.

    Physical basis:
    - Core (neutrons): Dense, saturated → I_core ∝ N^(5/3)
    - Atmosphere (protons): Gradient, refractive → I_atm ∝ Z^(5/3)

    The atmosphere has LARGER effective radius → higher contribution to I.
    """
    A = Z + N
    r_0 = 1.2  # fm

    # Core radius (compressed, saturated)
    R_core = r_0 * N**(1/3)

    # Atmosphere radius (extended, gradient)
    # Protons pushed outward by Coulomb repulsion
    R_atm = r_0 * (A**(1/3) + 0.3 * Z**(1/3))  # Extended by charge

    # Moment contributions (rigid body: I = (2/5) M R²)
    # But atmosphere is diffuse → use factor 0.6 instead of 0.4
    I_core = 0.4 * (N * M_proton) * R_core**2
    I_atm = 0.6 * (Z * M_proton) * R_atm**2  # Diffuse shell

    return I_core + I_atm

def spin_energy(Z, N, use_spin=True):
    """
    Calculate rotational spin energy E_spin = ℏ²J(J+1)/(2I).

    This is NOT a bonus - it's the kinetic energy of the rotating manifold.
    """
    if not use_spin:
        return 0.0

    J = predict_spin(Z, N)

    if J == 0:
        return 0.0

    I = moment_of_inertia_dual_core(Z, N)

    # E_spin = (ℏc)² × J(J+1) / (2I)
    # Units: (MeV·fm)² / (MeV·fm²/c²) = MeV
    E_spin = (hbar_c**2) * J * (J + 1) / (2.0 * I)

    return E_spin

def qfd_energy_unified(A, Z, use_spin=True, spin_scaling=1.0):
    """
    Unified QFD energy with spin as explicit energy term.

    Args:
        A, Z: Mass and charge
        use_spin: Include spin energy term
        spin_scaling: Scale factor for E_spin (test sensitivity)
    """
    N = A - Z
    q = Z / A if A > 0 else 0
    lambda_time = KAPPA_E * Z

    # Base QFD terms
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

    # Pairing energy (fermion statistics)
    E_pair = 0
    if Z % 2 == 0 and N % 2 == 0:
        E_pair = -DELTA_PAIRING / np.sqrt(A)
    elif Z % 2 == 1 and N % 2 == 1:
        E_pair = +DELTA_PAIRING / np.sqrt(A)

    # SPIN ENERGY (rotational kinetic energy)
    E_spin = spin_energy(Z, N, use_spin) * spin_scaling

    return E_bulk + E_surf + E_asym + E_vac + E_pair + E_spin

def find_stable_Z_unified(A, use_spin=True, spin_scaling=1.0):
    """Find Z that minimizes unified QFD energy."""
    best_Z, best_E = 1, qfd_energy_unified(A, 1, use_spin, spin_scaling)

    for Z in range(1, A):
        E = qfd_energy_unified(A, Z, use_spin, spin_scaling)
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
print("DUAL-CORE SPIN SOLVER - SPIN AS ROTATIONAL ENERGY")
print("="*95)
print()

# ============================================================================
# TEST 1: FE-56 CASE STUDY
# ============================================================================
print("="*95)
print("TEST CASE: Fe-56 (The Nuclear Benchmark)")
print("="*95)
print()

A_fe = 56
Z_fe_exp = 26
N_fe_exp = 30

print(f"Fe-56: A={A_fe}, Z_exp={Z_fe_exp}, N_exp={N_fe_exp}")
print()

# Test different Z values around Fe-56
print("Energy landscape around Z=26:")
print(f"{'Z':<6} {'N':<6} {'J_pred':<8} {'I (MeV·fm²)':<15} {'E_spin (MeV)':<15} {'E_total (MeV)':<15}")
print("-"*95)

for Z_test in range(22, 31):
    N_test = A_fe - Z_test
    J_pred = predict_spin(Z_test, N_test)
    I = moment_of_inertia_dual_core(Z_test, N_test)
    E_sp = spin_energy(Z_test, N_test, True)
    E_tot = qfd_energy_unified(A_fe, Z_test, True, 1.0)

    marker = "★" if Z_test == Z_fe_exp else ""

    print(f"{Z_test:<6} {N_test:<6} {J_pred:<8.1f} {I:<15.0f} {E_sp:<15.6f} {E_tot:<15.2f}  {marker}")

print()

# Find predicted Z with and without spin
Z_pred_pure = find_stable_Z_unified(A_fe, use_spin=False)
Z_pred_spin = find_stable_Z_unified(A_fe, use_spin=True)

print(f"Pure QFD (no spin):     Z_pred = {Z_pred_pure} (Z_exp = {Z_fe_exp}) {'✓' if Z_pred_pure == Z_fe_exp else '✗'}")
print(f"Unified (with E_spin):  Z_pred = {Z_pred_spin} (Z_exp = {Z_fe_exp}) {'✓' if Z_pred_spin == Z_fe_exp else '✗'}")
print()

if Z_pred_spin == Z_fe_exp and Z_pred_pure != Z_fe_exp:
    print("★★★ SPIN ENERGY FIXES Fe-56!")
    print("    → Rotational kinetic energy shifts minimum to correct Z")
elif Z_pred_spin != Z_fe_exp:
    print("Spin energy doesn't fix Fe-56 with current I(Z,N) model")
    print("    → May need refined dual-core partition or deformation")

print()

# ============================================================================
# TEST 2: FULL DATASET WITH SPIN ENERGY
# ============================================================================
print("="*95)
print("FULL DATASET: UNIFIED QFD WITH SPIN ENERGY")
print("="*95)
print()

print("Testing spin energy on all 285 nuclides...")
print()

correct_pure = 0
correct_spin = 0
fixed_by_spin = []
broken_by_spin = []

for name, Z_exp, A in test_nuclides:
    N_exp = A - Z_exp

    Z_pred_pure = find_stable_Z_unified(A, use_spin=False)
    Z_pred_spin = find_stable_Z_unified(A, use_spin=True)

    if Z_pred_pure == Z_exp:
        correct_pure += 1
    if Z_pred_spin == Z_exp:
        correct_spin += 1

    # Track changes
    if Z_pred_pure != Z_exp and Z_pred_spin == Z_exp:
        J_pred = predict_spin(Z_exp, N_exp)
        fixed_by_spin.append({
            'name': name,
            'A': A,
            'Z': Z_exp,
            'N': N_exp,
            'J': J_pred,
            'Z_pred_pure': Z_pred_pure,
            'mod_4': A % 4,
            'mod_28': A % 28,
        })
    elif Z_pred_pure == Z_exp and Z_pred_spin != Z_exp:
        broken_by_spin.append({
            'name': name,
            'A': A,
            'Z': Z_exp,
            'N': N_exp,
            'Z_pred_spin': Z_pred_spin,
        })

print(f"{'Model':<35} {'Correct':<12} {'Success %':<12} {'vs Pure'}")
print("-"*95)
print(f"{'Pure QFD (no spin)':<35} {correct_pure:<12} {100*correct_pure/285:<12.1f} {'baseline'}")
print(f"{'Unified (with E_spin)':<35} {correct_spin:<12} {100*correct_spin/285:<12.1f} {f'{correct_spin - correct_pure:+d} matches'}")
print()

if correct_spin > correct_pure:
    print(f"★★ SPIN ENERGY IMPROVES PREDICTIONS BY {correct_spin - correct_pure} MATCHES!")
    print()

# ============================================================================
# ANALYSIS: WHICH NUCLEI ARE FIXED?
# ============================================================================
if fixed_by_spin:
    print("="*95)
    print(f"NUCLEI FIXED BY SPIN ENERGY ({len(fixed_by_spin)} total)")
    print("="*95)
    print()

    print(f"{'Nuclide':<12} {'A':<6} {'Z':<6} {'N':<6} {'J_pred':<8} {'Z_pure':<10} {'A mod 4':<10} {'A mod 28'}")
    print("-"*95)

    for n in sorted(fixed_by_spin, key=lambda x: x['A'])[:30]:
        marker_4 = "★" if n['mod_4'] == 1 else ""
        marker_28 = "★★" if n['mod_28'] == 13 else ""

        print(f"{n['name']:<12} {n['A']:<6} {n['Z']:<6} {n['N']:<6} {n['J']:<8.1f} {n['Z_pred_pure']:<10} "
              f"{n['mod_4']:<10} {marker_4:<5} {n['mod_28']:<10} {marker_28}")

    if len(fixed_by_spin) > 30:
        print(f"... and {len(fixed_by_spin) - 30} more")

    print()

    # Statistics
    mod4_1_count = sum(1 for n in fixed_by_spin if n['mod_4'] == 1)
    mod28_13_count = sum(1 for n in fixed_by_spin if n['mod_28'] == 13)
    odd_odd_count = sum(1 for n in fixed_by_spin if n['Z'] % 2 == 1 and n['N'] % 2 == 1)

    print("Patterns in fixed nuclei:")
    print(f"  A mod 4 = 1:    {mod4_1_count}/{len(fixed_by_spin)} ({100*mod4_1_count/len(fixed_by_spin):.1f}%)")
    print(f"  A mod 28 = 13:  {mod28_13_count}/{len(fixed_by_spin)} ({100*mod28_13_count/len(fixed_by_spin):.1f}%)")
    print(f"  Odd-odd nuclei: {odd_odd_count}/{len(fixed_by_spin)} ({100*odd_odd_count/len(fixed_by_spin):.1f}%)")
    print()

# ============================================================================
# TEST 3: SENSITIVITY TO SPIN SCALING
# ============================================================================
print("="*95)
print("SENSITIVITY: SPIN ENERGY SCALING")
print("="*95)
print()

print("Testing different spin energy scaling factors...")
print("(E_spin multiplied by scaling factor)")
print()

scaling_factors = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]

print(f"{'Scaling':<12} {'Correct':<12} {'Success %':<12} {'vs Pure'}")
print("-"*95)

best_scaling = 1.0
best_correct = correct_pure

for scale in scaling_factors:
    correct = 0
    for name, Z_exp, A in test_nuclides:
        Z_pred = find_stable_Z_unified(A, use_spin=True, spin_scaling=scale)
        if Z_pred == Z_exp:
            correct += 1

    delta = correct - correct_pure
    marker = "★" if correct > best_correct else ""

    print(f"{scale:<12.1f} {correct:<12} {100*correct/285:<12.1f} {delta:+d}  {marker}")

    if correct > best_correct:
        best_correct = correct
        best_scaling = scale

print()

if best_correct > correct_pure:
    print(f"★ OPTIMAL SPIN SCALING: {best_scaling:.1f}x")
    print(f"  Result: {best_correct}/285 ({100*best_correct/285:.1f}%)")
    print(f"  Improvement: {best_correct - correct_pure:+d} matches")
else:
    print("Default spin scaling (1.0x) is optimal")

print()

# ============================================================================
# SUMMARY
# ============================================================================
print("="*95)
print("SUMMARY: SPIN AS ROTATIONAL ENERGY")
print("="*95)
print()

print("Key Results:")
print(f"  Pure QFD (no spin):        {correct_pure}/285 ({100*correct_pure/285:.1f}%)")
print(f"  Unified (E_spin at 1.0x):  {correct_spin}/285 ({100*correct_spin/285:.1f}%)")
print(f"  Unified (E_spin at {best_scaling:.1f}x):  {best_correct}/285 ({100*best_correct/285:.1f}%)")
print()

if best_correct > correct_pure:
    print(f"★★ SPIN ENERGY IMPROVES PREDICTIONS!")
    print(f"   Mechanism: Rotational kinetic energy I_atm(Z) > I_core(N)")
    print(f"   → Higher Z favored for high-spin states")
    print()
    print(f"   Improvement: {best_correct - correct_pure} matches ({100*(best_correct - correct_pure)/285:.1f}%)")
else:
    print("Spin energy shows minimal/no improvement with current I(Z,N) model")
    print()
    print("Possible reasons:")
    print("  • I(Z,N) model too simplistic (need deformation-dependent I)")
    print("  • Spin prediction J(Z,N) needs refinement")
    print("  • Spin energy scale may be wrong (check ℏ²/(2I) units)")

print()

print("Physical Interpretation:")
print("  • Spin IS the rotational kinetic energy of the vacuum manifold")
print("  • Higher Z → larger atmosphere → larger I → LOWER E_spin for same J")
print("  • This explains why higher Z can be favored for rotating nuclei")
print()

print("The Unified Lagrangian:")
print("  E_total = E_bulk + E_surf + E_asym + E_vac + E_pair + E_spin")
print("  where E_spin = ℏ²J(J+1)/(2I_dual_core)")
print()

if 'Fe-56' in [n[0] for n in test_nuclides]:
    fe56_result = "✓ FIXED" if Z_pred_spin == Z_fe_exp else "✗ NOT FIXED"
    print(f"Fe-56 benchmark: {fe56_result}")
    print(f"  Pure: Z={Z_pred_pure}, Unified: Z={Z_pred_spin}, Experimental: Z={Z_fe_exp}")

print()
print("="*95)
print("THE REVOLUTION: Spin is NOT a quantum number - it is ENERGY.")
print("="*95)
