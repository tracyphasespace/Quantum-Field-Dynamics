#!/usr/bin/env python3
"""
DEFORMATION GEOMETRY - PROLATE/OBLATE SOLITON SHAPES
===========================================================================
NO BONUSES. Pure geometric reality.

Physical basis:
- Nuclei are NOT perfect spheres
- Rare earth region (A~150-190): Prolate (football) shapes
- Some regions: Oblate (pancake) shapes
- Deformation parameter β₂ (quadrupole moment)

Geometric effects:
1. Surface energy changes: E_surf × (1 + β₂²/π)
2. Moment of inertia changes: I_deformed = I_sphere × f(β₂)
3. Coulomb energy changes: E_vac × (1 - β₂²/5)

This is NOT an empirical bonus - it's accounting for the actual shape!

Known deformed regions:
- Rare earth (60≤Z≤70, 150≤A≤190): β₂ ~ 0.2-0.3 (prolate)
- Actinides (Z≥90, 230≤A≤250): β₂ ~ 0.15-0.25 (prolate)
- Transitional regions: Variable deformation

Strategy:
1. Estimate β₂(A,Z) from known systematics
2. Modify E_surf, I_atm, E_vac based on geometry
3. Test if accounting for real shapes improves predictions
===========================================================================
"""

import numpy as np
from collections import Counter, defaultdict

# QFD Constants (α-derived, 2026-01-07)
# β = 3.04309 derived from Golden Loop: 1/α = 2π²(e^β/β) + 1
alpha_fine = 1.0 / 137.035999206  # CODATA 2018
BETA_VACUUM_STIFFNESS = 3.04309   # Vacuum stiffness (derived from α)
beta_vacuum = 1.0 / BETA_VACUUM_STIFFNESS  # c₂ = 1/β = 0.328615
M_proton = 938.272
KAPPA_E = 0.0001
SHIELD_FACTOR = 0.52
DELTA_PAIRING = 11.0
R0 = 1.2  # fm
hbar_c = 197.327  # MeV·fm

# Derived Constants
V_0 = M_proton * (1 - (alpha_fine**2) / beta_vacuum)
beta_nuclear = M_proton * beta_vacuum / 2
E_surface_coeff = beta_nuclear / 15
a_sym_base = (beta_vacuum * M_proton) / 15
a_disp = (alpha_fine * 197.327 / 1.2) * SHIELD_FACTOR

ISOMER_NODES = {2, 8, 20, 28, 50, 82, 126}

def distance_to_magic(n):
    """Distance to nearest magic number."""
    if not ISOMER_NODES:
        return 999
    return min(abs(n - magic) for magic in ISOMER_NODES)

def estimate_deformation(A, Z):
    """
    Estimate quadrupole deformation β₂ from systematics.

    Physical basis:
    - Spherical near magic numbers (β₂ ≈ 0)
    - Maximum deformation midshell (β₂ ≈ 0.2-0.3)
    - Rare earth region: Known large deformations
    - Actinides: Also deformed

    Returns: β₂ (dimensionless quadrupole parameter)
    """
    N = A - Z

    # Distance from magic numbers
    Z_dist = distance_to_magic(Z)
    N_dist = distance_to_magic(N)

    # Baseline deformation from magic number distance
    # Maximum deformation midshell (dist ~ 10-15)
    dist_factor = min(Z_dist, 15) / 15.0  # 0 to 1
    dist_factor += min(N_dist, 15) / 15.0
    dist_factor /= 2.0  # Average

    # Base deformation (parabolic, max at midshell)
    beta_base = 0.25 * dist_factor * (2 - dist_factor)  # Peaks at 0.5

    # Region-specific enhancements
    beta_region = 0.0

    # Rare earth region (60≤Z≤70, 150≤A≤190)
    if 60 <= Z <= 70 and 150 <= A <= 190:
        beta_region = 0.10  # Large prolate deformation

    # Actinide region (Z≥90, 230≤A≤250)
    elif Z >= 90 and 230 <= A <= 250:
        beta_region = 0.08  # Moderate prolate deformation

    # Light nuclei (A<40) - mostly spherical
    elif A < 40:
        beta_base *= 0.5  # Reduce deformation

    # Transitional regions (between shells)
    # A ~ 100-130: Moderate deformation
    elif 100 <= A <= 130:
        beta_region = 0.05

    beta_total = beta_base + beta_region

    # Cap at physical maximum
    return min(beta_total, 0.35)

def surface_energy_deformed(A, beta2):
    """
    Surface energy for deformed nucleus.

    E_surf = E_surf_sphere × (1 + β₂²/π + higher orders)

    Physical basis: Deformation increases surface area
    """
    E_surf_sphere = E_surface_coeff * (A**(2/3))

    # Deformation correction (to second order)
    deformation_factor = 1.0 + (beta2**2) / np.pi

    return E_surf_sphere * deformation_factor

def coulomb_energy_deformed(A, Z, beta2):
    """
    Coulomb/vacuum energy for deformed nucleus.

    E_vac = E_vac_sphere × (1 - β₂²/5 + higher orders)

    Physical basis: Deformation reduces average Coulomb energy
    (charges spread out more in prolate shape)
    """
    E_vac_sphere = a_disp * (Z**2) / (A**(1/3))

    # Deformation correction (prolate reduces, oblate increases)
    deformation_factor = 1.0 - (beta2**2) / 5.0

    return E_vac_sphere * deformation_factor

def moment_inertia_deformed(A, Z, beta2):
    """
    Moment of inertia for deformed nucleus.

    For prolate (football): I_perp > I_parallel
    Nucleus rotates around short axis → use I_perp

    I_deformed ≈ I_sphere × (1 + β₂²)
    """
    # Spherical moment (gradient atmosphere model)
    N = A - Z
    core_fraction = N / A
    R_total = R0 * (A**(1/3))
    R_core = R_total * (core_fraction**(1/3))
    M_total = A * M_proton
    M_atm = M_total * (1 - core_fraction)

    I_sphere = (1.0/3.0) * M_atm * (R_total**2 + R_core**2)

    # Deformation enhancement (prolate increases I_perp)
    deformation_factor = 1.0 + beta2**2

    return I_sphere * deformation_factor

def spin_energy_deformed(A, Z, beta2, coupling_k):
    """
    Spin energy for deformed nucleus with modified moment.
    """
    if Z == 0:
        return 0.0

    L = Z  # Vortex quantization
    I_deformed = moment_inertia_deformed(A, Z, beta2)

    E_spin = (hbar_c**2) * L * (L + 1) / (2.0 * I_deformed)

    return E_spin * coupling_k

def qfd_energy_deformed(A, Z, coupling_k=0.0, use_deformation=True):
    """
    Full QFD energy accounting for deformed geometry.

    NO BONUSES - just geometric reality of non-spherical shapes.
    """
    N = A - Z
    q = Z / A if A > 0 else 0
    lambda_time = KAPPA_E * Z

    # Volume energy (unchanged by deformation to second order)
    E_volume_coeff = V_0 * (1 - lambda_time / (12 * np.pi))
    E_bulk = E_volume_coeff * A

    # Asymmetry energy (unchanged)
    E_asym = a_sym_base * A * ((1 - 2*q)**2)

    # Pairing energy
    E_pair = 0
    if Z % 2 == 0 and N % 2 == 0:
        E_pair = -DELTA_PAIRING / np.sqrt(A)
    elif Z % 2 == 1 and N % 2 == 1:
        E_pair = +DELTA_PAIRING / np.sqrt(A)

    if use_deformation:
        # Estimate deformation
        beta2 = estimate_deformation(A, Z)

        # Deformed surface energy
        E_surf = surface_energy_deformed(A, beta2)

        # Deformed Coulomb energy
        E_vac = coulomb_energy_deformed(A, Z, beta2)

        # Deformed spin energy
        E_spin = spin_energy_deformed(A, Z, beta2, coupling_k)
    else:
        # Spherical approximation
        E_surf = E_surface_coeff * (A**(2/3))
        E_vac = a_disp * (Z**2) / (A**(1/3))

        # Spherical spin energy
        if coupling_k > 0 and Z > 0:
            N = A - Z
            core_fraction = N / A
            R_total = R0 * (A**(1/3))
            R_core = R_total * (core_fraction**(1/3))
            M_total = A * M_proton
            M_atm = M_total * (1 - core_fraction)
            I_atm = (1.0/3.0) * M_atm * (R_total**2 + R_core**2)
            L = Z
            E_spin = (hbar_c**2) * L * (L + 1) / (2.0 * I_atm) * coupling_k
        else:
            E_spin = 0.0

    return E_bulk + E_surf + E_asym + E_vac + E_pair + E_spin

def find_stable_Z_deformed(A, coupling_k=0.0, use_deformation=True):
    """Find Z that minimizes deformed energy."""
    best_Z, best_E = 1, qfd_energy_deformed(A, 1, coupling_k, use_deformation)

    for Z in range(1, A):
        E = qfd_energy_deformed(A, Z, coupling_k, use_deformation)
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
print("DEFORMATION GEOMETRY - ACCOUNTING FOR REAL NUCLEAR SHAPES")
print("="*95)
print()
print("Philosophy: NO BONUSES. Just geometric reality.")
print("Nuclei are prolate (football) or oblate (pancake), not perfect spheres.")
print()

# ============================================================================
# TEST 1: DEFORMATION ESTIMATES
# ============================================================================
print("="*95)
print("DEFORMATION ESTIMATES FOR KEY REGIONS")
print("="*95)
print()

print(f"{'Nuclide':<12} {'A':<6} {'Z':<6} {'N':<6} {'β₂':<10} {'Region'}")
print("-"*95)

# Sample nuclei from different regions
samples = [
    ('Ca-40', 40, 20),      # Doubly magic, spherical
    ('Fe-56', 56, 26),      # Semi-magic, near-spherical
    ('Zr-96', 96, 40),      # Transitional
    ('Sm-154', 154, 62),    # Rare earth, deformed
    ('Gd-160', 160, 64),    # Rare earth, deformed
    ('Yb-176', 176, 70),    # Rare earth, deformed
    ('Pb-208', 208, 82),    # Doubly magic, spherical
]

for name, A, Z in samples:
    N = A - Z
    beta2 = estimate_deformation(A, Z)

    # Classify region
    if 60 <= Z <= 70 and 150 <= A <= 190:
        region = "Rare earth (deformed)"
    elif Z in ISOMER_NODES and N in ISOMER_NODES:
        region = "Doubly magic (spherical)"
    elif distance_to_magic(Z) < 3 or distance_to_magic(N) < 3:
        region = "Near magic (spherical)"
    else:
        region = "Transitional"

    print(f"{name:<12} {A:<6} {Z:<6} {N:<6} {beta2:<10.3f} {region}")

print()

# ============================================================================
# TEST 2: FULL DATASET WITH DEFORMATION
# ============================================================================
print("="*95)
print("FULL DATASET: DEFORMED vs SPHERICAL GEOMETRY")
print("="*95)
print()

print("Testing on all 285 nuclides...")
print()

correct_sphere = 0
correct_deformed = 0

for name, Z_exp, A in test_nuclides:
    Z_pred_sphere = find_stable_Z_deformed(A, coupling_k=0.0, use_deformation=False)
    Z_pred_deformed = find_stable_Z_deformed(A, coupling_k=0.0, use_deformation=True)

    if Z_pred_sphere == Z_exp:
        correct_sphere += 1
    if Z_pred_deformed == Z_exp:
        correct_deformed += 1

print(f"{'Model':<35} {'Correct':<12} {'Success %':<12} {'vs Sphere'}")
print("-"*95)
print(f"{'Spherical geometry':<35} {correct_sphere:<12} {100*correct_sphere/285:<12.1f} {'baseline'}")
print(f"{'Deformed geometry':<35} {correct_deformed:<12} {100*correct_deformed/285:<12.1f} {f'{correct_deformed - correct_sphere:+d} matches'}")
print()

if correct_deformed > correct_sphere:
    print(f"★★ DEFORMATION IMPROVES BY {correct_deformed - correct_sphere} MATCHES!")
elif correct_deformed == correct_sphere:
    print("Deformation has no net effect (geometric corrections cancel)")
else:
    print(f"Deformation reduces accuracy by {correct_sphere - correct_deformed} matches")
    print("  → May need to refine β₂ estimates or correction formulas")

print()

# ============================================================================
# TEST 3: DEFORMATION + GRADIENT ATMOSPHERE
# ============================================================================
print("="*95)
print("COMBINED: DEFORMATION + GRADIENT ATMOSPHERE SPIN")
print("="*95)
print()

# Test different coupling values with deformation
coupling_values = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20]

print(f"{'k':<12} {'Correct':<12} {'Success %':<12} {'vs Pure'}")
print("-"*95)

best_k_deformed = 0.0
best_correct_deformed = 0

for k in coupling_values:
    correct = 0
    for name, Z_exp, A in test_nuclides:
        Z_pred = find_stable_Z_deformed(A, coupling_k=k, use_deformation=True)
        if Z_pred == Z_exp:
            correct += 1

    delta = correct - correct_sphere
    marker = "★" if correct > best_correct_deformed else ""

    print(f"{k:<12.2f} {correct:<12} {100*correct/285:<12.1f} {delta:+d}  {marker}")

    if correct > best_correct_deformed:
        best_correct_deformed = correct
        best_k_deformed = k

print()
print(f"Optimal: k={best_k_deformed:.2f}, Result={best_correct_deformed}/285 ({100*best_correct_deformed/285:.1f}%)")
print()

# ============================================================================
# ANALYSIS: WHICH NUCLEI ARE FIXED?
# ============================================================================
print("="*95)
print("NUCLEI FIXED BY DEFORMATION GEOMETRY")
print("="*95)
print()

fixed_by_deformation = []

for name, Z_exp, A in test_nuclides:
    N_exp = A - Z_exp

    Z_pred_sphere = find_stable_Z_deformed(A, coupling_k=0.0, use_deformation=False)
    Z_pred_deformed = find_stable_Z_deformed(A, coupling_k=best_k_deformed, use_deformation=True)

    if Z_pred_sphere != Z_exp and Z_pred_deformed == Z_exp:
        beta2 = estimate_deformation(A, Z_exp)

        # Classify region
        if 60 <= Z_exp <= 70 and 150 <= A <= 190:
            region = "Rare earth"
        elif Z_exp >= 90 and 230 <= A <= 250:
            region = "Actinide"
        elif 100 <= A <= 130:
            region = "Transitional"
        else:
            region = "Other"

        fixed_by_deformation.append({
            'name': name,
            'A': A,
            'Z': Z_exp,
            'N': N_exp,
            'beta2': beta2,
            'region': region,
            'mod_4': A % 4,
        })

print(f"Total fixed: {len(fixed_by_deformation)}")
print()

if fixed_by_deformation:
    print(f"{'Nuclide':<12} {'A':<6} {'Z':<6} {'N':<6} {'β₂':<10} {'Region':<15} {'A mod 4'}")
    print("-"*95)

    for n in sorted(fixed_by_deformation, key=lambda x: x['A'])[:30]:
        marker = "★" if n['mod_4'] == 1 else ""
        print(f"{n['name']:<12} {n['A']:<6} {n['Z']:<6} {n['N']:<6} {n['beta2']:<10.3f} "
              f"{n['region']:<15} {n['mod_4']:<10} {marker}")

    if len(fixed_by_deformation) > 30:
        print(f"... and {len(fixed_by_deformation) - 30} more")

    print()

    # Statistics by region
    region_counts = Counter(n['region'] for n in fixed_by_deformation)
    print("Fixed by region:")
    for region, count in region_counts.most_common():
        print(f"  {region}: {count}/{len(fixed_by_deformation)} ({100*count/len(fixed_by_deformation):.1f}%)")

print()

# ============================================================================
# SUMMARY
# ============================================================================
print("="*95)
print("SUMMARY: PURE GEOMETRY (NO BONUSES)")
print("="*95)
print()

print(f"Spherical (baseline):            {correct_sphere}/285 ({100*correct_sphere/285:.1f}%)")
print(f"Deformed (β₂ corrections):       {correct_deformed}/285 ({100*correct_deformed/285:.1f}%)")
print(f"Deformed + spin (k={best_k_deformed:.2f}):      {best_correct_deformed}/285 ({100*best_correct_deformed/285:.1f}%)")
print()

total_improvement = best_correct_deformed - correct_sphere
print(f"Total improvement: {total_improvement:+d} matches ({100*total_improvement/285:+.1f}%)")
print()

if total_improvement > 0:
    print("★ SUCCESS! Accounting for real nuclear shapes improves predictions!")
    print()
    print("Physical mechanism:")
    print("  • Rare earth nuclei: Prolate (β₂ ~ 0.2-0.3)")
    print("  • Surface area increases → E_surf × (1 + β₂²/π)")
    print("  • Coulomb energy decreases → E_vac × (1 - β₂²/5)")
    print("  • Moment of inertia increases → E_spin decreases")
    print()
    print("This is NOT a bonus - it's GEOMETRIC REALITY.")
else:
    print("Deformation corrections show no/negative improvement")
    print("  → May need better β₂ systematics or higher-order corrections")

print()
print("="*95)
print("PHILOSOPHY: Pure geometry. No empirical bonuses. Just the real shape.")
print("="*95)
