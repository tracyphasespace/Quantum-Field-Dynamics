#!/usr/bin/env python3
"""
ISOBAR SPIN ANALYSIS - COLLECTIVE ROTATION DIFFERENTIATES ISOBARS
===========================================================================
Discovery: Isobars (same A, different Z) have different spins because
the dual-core partition (frozen neutron core vs gradient proton atmosphere)
creates different moments of inertia.

Key Insight:
  • A mod 4 = 1 creates "opposite parity harmony" (one even, one odd)
  • Clean spin signature: even component pairs off, odd determines net spin
  • Rotation energy E_rot = ℏ²J(J+1)/(2I) depends on moment I

Strategy:
1. Identify all stable isobar pairs in dataset
2. Calculate their experimental spins
3. Model moment of inertia I(Z,N) based on core/atmosphere partition
4. Test if adding rotation energy differentiates isobars correctly

GOAL: Fix the remaining ~100 failures by adding collective rotation
===========================================================================
"""

import numpy as np
from collections import defaultdict

# Fundamental constants
alpha_fine = 1.0 / 137.036
beta_vacuum = 1.0 / 3.043233053
M_proton = 938.272
KAPPA_E = 0.0001
SHIELD_FACTOR = 0.52
DELTA_PAIRING = 11.0
hbar_c = 197.327  # MeV·fm

# Nuclear spins from literature (ground states)
NUCLEAR_SPINS = {
    # Format: (name, Z, A): J (total angular momentum)
    ('H-1', 1, 1): 0.5,
    ('H-2', 1, 2): 1.0,
    ('He-3', 2, 3): 0.5,
    ('He-4', 2, 4): 0.0,
    ('Li-6', 3, 6): 1.0,
    ('Li-7', 3, 7): 1.5,
    ('Be-9', 4, 9): 1.5,
    ('B-10', 5, 10): 3.0,
    ('B-11', 5, 11): 1.5,
    ('C-12', 6, 12): 0.0,
    ('C-13', 6, 13): 0.5,
    ('N-14', 7, 14): 1.0,
    ('N-15', 7, 15): 0.5,
    ('O-16', 8, 16): 0.0,
    ('O-17', 8, 17): 2.5,
    ('O-18', 8, 18): 0.0,
    ('F-19', 9, 19): 0.5,
    ('Ne-20', 10, 20): 0.0,
    ('Na-23', 11, 23): 1.5,
    ('Mg-24', 12, 24): 0.0,
    ('Al-27', 13, 27): 2.5,
    ('Si-28', 14, 28): 0.0,
    ('P-31', 15, 31): 0.5,
    ('S-32', 16, 32): 0.0,
    ('Cl-35', 17, 35): 1.5,
    ('Cl-37', 17, 37): 1.5,
    ('Ar-36', 18, 36): 0.0,
    ('Ar-40', 18, 40): 0.0,
    ('K-39', 19, 39): 1.5,
    ('K-40', 19, 40): 4.0,  # Odd-odd, high spin
    ('K-41', 19, 41): 1.5,
    ('Ca-40', 20, 40): 0.0,
    ('Ca-48', 20, 48): 0.0,
    # Add more as needed
}

def qfd_energy_pure(A, Z):
    """Pure QFD energy - no bonuses, no rotation."""
    N = A - Z
    q = Z / A if A > 0 else 0
    lambda_time = KAPPA_E * Z

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

    return E_bulk + E_surf + E_asym + E_vac + E_pair

def moment_of_inertia(A, Z, model='dual_core'):
    """
    Calculate moment of inertia based on core/atmosphere partition.

    Models:
    - 'rigid_sphere': I = (2/5) M R²
    - 'dual_core': I depends on N/Z partition
    - 'shell_model': Empirical from shell model
    """
    N = A - Z
    r_0 = 1.2  # fm (nuclear radius parameter)
    R = r_0 * A**(1/3)  # fm
    M_nucleus = A * M_proton  # Total mass (MeV/c²)

    if model == 'rigid_sphere':
        # Rigid sphere: I = (2/5) M R²
        I = 0.4 * M_nucleus * R**2
        return I

    elif model == 'dual_core':
        # Dual-core partition model
        # Core (neutrons): Dense, rigid → low contribution to I
        # Atmosphere (protons): Diffuse, extended → high contribution to I

        # Fraction of mass in atmosphere vs core
        f_atm = Z / A  # Proton fraction (gradient atmosphere)
        f_core = N / A  # Neutron fraction (frozen core)

        # Effective radii (core is compressed, atmosphere is extended)
        R_core = r_0 * N**(1/3)  # Smaller radius for core
        R_atm = r_0 * (A**(1/3) + 0.5)  # Extended radius for atmosphere

        # Moment contributions
        I_core = 0.4 * (N * M_proton) * R_core**2  # Rigid sphere for core
        I_atm = 0.6 * (Z * M_proton) * R_atm**2  # More diffuse for atmosphere

        I_total = I_core + I_atm
        return I_total

    elif model == 'parity_dependent':
        # Moment depends on (Z,N) mod 4 parity
        Z_mod = Z % 4
        N_mod = N % 4

        # Base moment (rigid sphere)
        I_base = 0.4 * M_nucleus * R**2

        # Parity corrections
        if (Z_mod, N_mod) in [(2, 3), (3, 2)]:
            # Exceptional parities: enhanced atmosphere → larger I
            I_total = I_base * 1.2
        elif (Z_mod, N_mod) in [(0, 0), (2, 2)]:
            # Paired parities: compressed core → smaller I
            I_total = I_base * 0.8
        else:
            I_total = I_base

        return I_total

    else:
        raise ValueError(f"Unknown model: {model}")

def rotation_energy(A, Z, J, model='dual_core'):
    """
    Calculate collective rotation energy E_rot = ℏ²J(J+1)/(2I).

    Args:
        A, Z: Mass and charge
        J: Total angular momentum
        model: 'dual_core', 'rigid_sphere', 'parity_dependent'
    """
    if J == 0:
        return 0.0

    I = moment_of_inertia(A, Z, model)

    # E_rot = ℏ²J(J+1)/(2I)
    # Units: (MeV·fm)² / (MeV/c² · fm²) = MeV
    # Need to convert: ℏc = 197.327 MeV·fm, so ℏ² = (197.327)²/(c²)
    # For nuclear units, use: E_rot in MeV, I in MeV·fm²/c²

    # Simplified: E_rot = A₀ × J(J+1) / I where A₀ ≈ ℏ²c²/(2 MeV·fm²)
    # Typical value: A₀ ≈ 0.01-0.1 MeV for heavy nuclei

    # Use dimensional analysis:
    # ℏ = 197.327 MeV·fm / c
    # I in units of M_proton·fm² where M_proton = 938.272 MeV/c²
    # E_rot = (ℏc)² × J(J+1) / (2 × I × c²)

    E_rot = (hbar_c**2) * J * (J + 1) / (2.0 * I)

    return E_rot

def qfd_energy_with_rotation(A, Z, J, model='dual_core'):
    """QFD energy including collective rotation."""
    E_static = qfd_energy_pure(A, Z)
    E_rot = rotation_energy(A, Z, J, model)
    return E_static + E_rot

# Load data
with open('qfd_optimized_suite.py', 'r') as f:
    content = f.read()
start = content.find('test_nuclides = [')
end = content.find(']', start) + 1
test_nuclides = eval(content[start:end].replace('test_nuclides = ', ''))

print("="*95)
print("ISOBAR SPIN ANALYSIS - COLLECTIVE ROTATION DIFFERENTIATES ISOBARS")
print("="*95)
print()

# ============================================================================
# IDENTIFY STABLE ISOBAR PAIRS
# ============================================================================
print("="*95)
print("STABLE ISOBAR PAIRS IN DATASET")
print("="*95)
print()

# Group nuclei by mass number A
nuclei_by_A = defaultdict(list)
for name, Z, A in test_nuclides:
    nuclei_by_A[A].append((name, Z, A))

# Find isobar pairs (same A, different Z)
isobar_pairs = []
for A, nuclei in nuclei_by_A.items():
    if len(nuclei) >= 2:
        isobar_pairs.append((A, nuclei))

print(f"Found {len(isobar_pairs)} mass numbers with multiple stable isobars:")
print()

print(f"{'Mass A':<10} {'Isobars':<50} {'Count'}")
print("-"*95)

for A, nuclei in sorted(isobar_pairs, key=lambda x: x[0]):
    names = ', '.join(n[0] for n in nuclei)
    print(f"{A:<10} {names:<50} {len(nuclei)}")

print()

# ============================================================================
# DETAILED ANALYSIS OF SELECTED ISOBAR PAIRS
# ============================================================================
print("="*95)
print("DETAILED ISOBAR ANALYSIS: A=40 (K-40 vs Ca-40)")
print("="*95)
print()

# Example: A=40 has K-40 (Z=19) and Ca-40 (Z=20)
A_example = 40
isobars_40 = [n for n in test_nuclides if n[2] == A_example]

if isobars_40:
    print(f"Isobars with A={A_example}:")
    print()

    for name, Z, A in sorted(isobars_40, key=lambda x: x[1]):
        N = A - Z
        J = NUCLEAR_SPINS.get((name, Z, A), None)

        print(f"{name}:")
        print(f"  Z={Z}, N={N}")
        print(f"  Z mod 4 = {Z % 4}, N mod 4 = {N % 4}")
        print(f"  Parity type: Z-{'even' if Z % 2 == 0 else 'odd'}, N-{'even' if N % 2 == 0 else 'odd'}")

        if J is not None:
            print(f"  Experimental spin: J={J}")

            # Calculate moments of inertia
            I_rigid = moment_of_inertia(A, Z, 'rigid_sphere')
            I_dual = moment_of_inertia(A, Z, 'dual_core')
            I_parity = moment_of_inertia(A, Z, 'parity_dependent')

            print(f"  Moment of inertia:")
            print(f"    Rigid sphere: I={I_rigid:.1f} MeV·fm²/c²")
            print(f"    Dual core:    I={I_dual:.1f} MeV·fm²/c²")
            print(f"    Parity dep:   I={I_parity:.1f} MeV·fm²/c²")

            # Rotation energies
            E_rot_rigid = rotation_energy(A, Z, J, 'rigid_sphere')
            E_rot_dual = rotation_energy(A, Z, J, 'dual_core')
            E_rot_parity = rotation_energy(A, Z, J, 'parity_dependent')

            print(f"  Rotation energy (J={J}):")
            print(f"    Rigid sphere: E_rot={E_rot_rigid:.4f} MeV")
            print(f"    Dual core:    E_rot={E_rot_dual:.4f} MeV")
            print(f"    Parity dep:   E_rot={E_rot_parity:.4f} MeV")

            # Total energies
            E_static = qfd_energy_pure(A, Z)
            E_total_dual = E_static + E_rot_dual

            print(f"  QFD energies:")
            print(f"    Static:       E={E_static:.2f} MeV")
            print(f"    + Rotation:   E={E_total_dual:.2f} MeV")
        else:
            print(f"  Experimental spin: Unknown")

        print()

# ============================================================================
# TEST: DOES ROTATION ENERGY IMPROVE ISOBAR PREDICTIONS?
# ============================================================================
print("="*95)
print("TEST: ROTATION ENERGY FOR ISOBAR DIFFERENTIATION")
print("="*95)
print()

print("Testing if rotation energy improves predictions for isobar pairs...")
print()

# For each isobar set, check if rotation energy helps pick the right Z
improved_by_rotation = []
worsened_by_rotation = []

for A, nuclei in isobar_pairs:
    # Get experimental Z
    Z_exp_list = [Z for (name, Z, A_val) in nuclei]

    for name_exp, Z_exp, A_val in nuclei:
        N_exp = A - Z_exp

        # Get experimental spin (if known)
        J_exp = NUCLEAR_SPINS.get((name_exp, Z_exp, A), None)

        if J_exp is None:
            continue  # Skip if spin unknown

        # Pure QFD prediction (no rotation)
        Z_pred_pure = None
        E_min_pure = None
        for Z_test in Z_exp_list:
            E = qfd_energy_pure(A, Z_test)
            if E_min_pure is None or E < E_min_pure:
                E_min_pure = E
                Z_pred_pure = Z_test

        # QFD with rotation energy
        Z_pred_rot = None
        E_min_rot = None
        for Z_test in Z_exp_list:
            E = qfd_energy_with_rotation(A, Z_test, J_exp, 'dual_core')
            if E_min_rot is None or E < E_min_rot:
                E_min_rot = E
                Z_pred_rot = Z_test

        # Check if rotation improved
        correct_pure = (Z_pred_pure == Z_exp)
        correct_rot = (Z_pred_rot == Z_exp)

        if not correct_pure and correct_rot:
            improved_by_rotation.append({
                'name': name_exp,
                'A': A,
                'Z_exp': Z_exp,
                'N_exp': N_exp,
                'J': J_exp,
                'Z_pred_pure': Z_pred_pure,
                'Z_pred_rot': Z_pred_rot,
            })
        elif correct_pure and not correct_rot:
            worsened_by_rotation.append({
                'name': name_exp,
                'A': A,
                'Z_exp': Z_exp,
                'N_exp': N_exp,
                'J': J_exp,
                'Z_pred_pure': Z_pred_pure,
                'Z_pred_rot': Z_pred_rot,
            })

print(f"Isobars improved by rotation energy: {len(improved_by_rotation)}")
print(f"Isobars worsened by rotation energy: {len(worsened_by_rotation)}")
print()

if improved_by_rotation:
    print("IMPROVED BY ROTATION:")
    print(f"{'Nuclide':<12} {'A':<6} {'Z_exp':<8} {'J':<6} {'Z_pure':<10} {'Z_rot':<10}")
    print("-"*95)
    for n in improved_by_rotation:
        print(f"{n['name']:<12} {n['A']:<6} {n['Z_exp']:<8} {n['J']:<6.1f} {n['Z_pred_pure']:<10} {n['Z_pred_rot']:<10}")
    print()

if worsened_by_rotation:
    print("WORSENED BY ROTATION:")
    print(f"{'Nuclide':<12} {'A':<6} {'Z_exp':<8} {'J':<6} {'Z_pure':<10} {'Z_rot':<10}")
    print("-"*95)
    for n in worsened_by_rotation:
        print(f"{n['name']:<12} {n['A']:<6} {n['Z_exp']:<8} {n['J']:<6.1f} {n['Z_pred_pure']:<10} {n['Z_pred_rot']:<10}")
    print()

# ============================================================================
# SUMMARY
# ============================================================================
print("="*95)
print("SUMMARY: COLLECTIVE ROTATION IN QFD")
print("="*95)
print()

print(f"Total stable isobar sets: {len(isobar_pairs)}")
print(f"Isobars with known spins: {len([n for n in NUCLEAR_SPINS if n[2] in [A for A, _ in isobar_pairs]])}")
print()

if improved_by_rotation:
    print(f"★ Rotation energy IMPROVES {len(improved_by_rotation)} isobar predictions!")
    print("  → Moment of inertia differentiates core/atmosphere partition")
elif len(improved_by_rotation) == 0 and len(worsened_by_rotation) == 0:
    print("Need more experimental spin data to test rotation hypothesis")
    print("  → Only tested isobars with known J values")
else:
    print("Rotation energy shows mixed/no improvement")
    print("  → May need better I(Z,N) model or more spin data")

print()

print("Physical interpretation:")
print("  • Higher Z → More gradient atmosphere → Larger moment I")
print("  • Same J but larger I → Lower E_rot → More stable")
print("  • Explains why some isobars are stable over others")
print()

print("Next steps:")
print("  1. Add more experimental spin data (NUCLEAR_SPINS dictionary)")
print("  2. Refine I(Z,N) model based on core/atmosphere density profiles")
print("  3. Test rotation energy on full 285 nuclide dataset")
print("  4. Combine rotation with deformation (E_rot depends on deformation!)")
print()

print("="*95)
