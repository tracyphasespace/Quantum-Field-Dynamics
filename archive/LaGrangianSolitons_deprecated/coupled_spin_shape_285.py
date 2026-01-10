#!/usr/bin/env python3
"""
COUPLED SPIN-SHAPE OPTIMIZATION - 285 NUCLIDES
===========================================================================
Self-consistent optimization over (Z, J, β₂) to find true ground states.

Energy functional:
  E_total = E_bulk + E_surf(β₂) + E_asym + E_vac + E_iso + E_rot(J,β₂) + E_pair(J)

For each A:
  1. For each Z: minimize E over (J, β₂) → find ground state
  2. Pick Z with lowest ground state energy

Physical couplings:
  - β₂ affects surface energy and moment of inertia
  - J affects rotational energy (depends on I(β₂))
  - Pairing favors J=0 for even-even nuclei
===========================================================================
"""

import numpy as np
from scipy.optimize import minimize_scalar

# ============================================================================
# FUNDAMENTAL CONSTANTS
# ============================================================================
alpha_fine   = 1.0 / 137.036
beta_vacuum  = 1.0 / 3.058231
lambda_time  = 0.42
M_proton     = 938.272

V_0 = M_proton * (1 - (alpha_fine**2) / beta_vacuum)
beta_nuclear = M_proton * beta_vacuum / 2

E_volume  = V_0 * (1 - lambda_time / (12 * np.pi))
E_surface = beta_nuclear / 15
a_sym     = (beta_vacuum * M_proton) / 15

hbar_c = 197.327
r_0 = 1.2

# Optimal parameters from sweep
SHIELD_FACTOR = 0.52
BONUS_STRENGTH = 0.70

a_disp = (alpha_fine * hbar_c / r_0) * SHIELD_FACTOR

ISOMER_NODES = {2, 8, 20, 28, 50, 82, 126}
ISOMER_BONUS = E_surface * BONUS_STRENGTH

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def get_isomer_bonus(Z, N):
    """Isomer resonance bonus."""
    bonus = 0
    if Z in ISOMER_NODES: bonus += ISOMER_BONUS
    if N in ISOMER_NODES: bonus += ISOMER_BONUS
    if Z in ISOMER_NODES and N in ISOMER_NODES:
        bonus *= 1.5
    return bonus

def get_moment_of_inertia(A, beta2):
    """
    Moment of inertia depends on deformation.
    For β₂=0 (spherical): I ∝ A^(5/3)
    For β₂≠0 (deformed): I increases with |β₂|
    """
    r0 = 1.2  # fm
    mass = 931.5  # MeV/c²

    # Spherical moment of inertia
    I_sphere = (2.0/5.0) * mass * A * (r0 * A**(1.0/3.0))**2

    # Deformation enhancement (Bohr-Mottelson)
    # Prolate (β₂>0) or oblate (β₂<0) both increase I
    enhancement = 1.0 + 0.5 * abs(beta2)

    return I_sphere * enhancement / (hbar_c**2)  # Convert to MeV⁻¹

def get_pairing_energy(Z, N, J):
    """
    Pairing energy for even-even nuclei with J=0.
    Topological interpretation: Paired vortices with opposite winding.
    """
    A = Z + N

    # Only even-even nuclei with J=0 get pairing bonus
    if Z % 2 == 0 and N % 2 == 0:
        if J == 0:
            return -12.0 / np.sqrt(A)  # Empirical pairing gap

    return 0.0

def get_allowed_spins(Z, N):
    """
    Allowed ground state spins based on even/odd character.
    """
    if Z % 2 == 0 and N % 2 == 0:
        # Even-even: J=0 (paired)
        return [0]
    elif Z % 2 == 1 and N % 2 == 1:
        # Odd-odd: J=1,2,3 typically
        return [1, 2, 3]
    else:
        # Odd-A: J=1/2, 3/2, 5/2, 7/2
        return [0.5, 1.5, 2.5, 3.5]

# ============================================================================
# COUPLED ENERGY FUNCTIONAL
# ============================================================================
def coupled_energy(A, Z, J, beta2):
    """
    Total energy including spin-shape coupling.

    Parameters:
      A: Mass number
      Z: Proton number
      J: Total angular momentum
      beta2: Quadrupole deformation parameter

    Returns:
      Total energy in MeV
    """
    N = A - Z
    q = Z / A if A > 0 else 0

    # Standard geometric terms (parameter-free)
    E_bulk = E_volume * A
    E_asym = a_sym * A * ((1 - 2*q)**2)
    E_vac  = a_disp * (Z**2) / (A**(1/3))
    E_iso  = -get_isomer_bonus(Z, N)

    # Deformation-dependent surface energy
    # f(β₂) = 1 + (2/3)β₂² for small deformations
    f_shape = 1.0 + (2.0/3.0) * beta2**2
    E_surf = E_surface * (A**(2/3)) * f_shape

    # Rotational energy (couples to deformation via I)
    I = get_moment_of_inertia(A, beta2)
    rot_const = 1.0 / (2.0 * I)  # ℏ²/(2I) in MeV
    E_rot = rot_const * J * (J + 1.0)

    # Pairing energy (favors J=0 for even-even)
    E_pair = get_pairing_energy(Z, N, J)

    return E_bulk + E_surf + E_asym + E_vac + E_iso + E_rot + E_pair

# ============================================================================
# GROUND STATE OPTIMIZATION
# ============================================================================
def find_ground_state(A, Z):
    """
    Find ground state (J, β₂) for given (A, Z).

    Returns:
      E_gs: Ground state energy
      (J_gs, beta2_gs): Optimal configuration
    """
    N = A - Z
    allowed_J = get_allowed_spins(Z, N)

    best_E = float('inf')
    best_config = (0, 0.0)

    for J in allowed_J:
        # For each J, optimize β₂
        result = minimize_scalar(
            lambda b2: coupled_energy(A, Z, J, b2),
            bounds=(-0.6, 0.6),
            method='bounded'
        )

        beta2_opt = result.x
        E = result.fun

        if E < best_E:
            best_E = E
            best_config = (J, beta2_opt)

    return best_E, best_config

def find_stable_Z_coupled(A):
    """
    Find most stable Z for given A, optimizing over (Z, J, β₂).

    Returns:
      Z_pred: Predicted proton number
      (J_gs, beta2_gs): Ground state configuration
    """
    if A <= 2:
        return 1, (0, 0.0)

    best_Z = 1
    best_E = float('inf')
    best_config = (0, 0.0)

    for Z in range(1, A):
        E_gs, (J_gs, beta2_gs) = find_ground_state(A, Z)

        if E_gs < best_E:
            best_E = E_gs
            best_Z = Z
            best_config = (J_gs, beta2_gs)

    return best_Z, best_config

# ============================================================================
# LOAD TEST DATA
# ============================================================================
# Load 285 nuclide dataset
import sys
sys.path.append('.')

# Read experimental data from qfd_optimized_suite.py
with open('qfd_optimized_suite.py', 'r') as f:
    content = f.read()

# Extract test_nuclides list
start = content.find('test_nuclides = [')
end = content.find(']', start) + 1
test_data_str = content[start:end]

# Parse it
test_nuclides = eval(test_data_str.replace('test_nuclides = ', ''))

print("="*95)
print("COUPLED SPIN-SHAPE OPTIMIZATION - 285 NUCLIDES")
print("="*95)
print()
print(f"Energy functional: E_total = E_bulk + E_surf(β₂) + E_asym + E_vac + E_iso + E_rot(J,β₂) + E_pair(J)")
print()
print(f"Parameters:")
print(f"  E_volume  = {E_volume:.3f} MeV")
print(f"  E_surface = {E_surface:.3f} MeV")
print(f"  a_sym     = {a_sym:.3f} MeV")
print(f"  a_disp    = {a_disp:.3f} MeV (shield={SHIELD_FACTOR:.2f})")
print(f"  Isomer bonus = {ISOMER_BONUS:.3f} MeV ({BONUS_STRENGTH:.2f} × E_surface)")
print()
print(f"Running coupled optimization on {len(test_nuclides)} nuclides...")
print(f"(Optimizing over Z, J, β₂ for each mass number)")
print()

# ============================================================================
# RUN OPTIMIZATION
# ============================================================================
results = []
for i, (name, Z_exp, A) in enumerate(test_nuclides):
    N_exp = A - Z_exp

    # Find optimal (Z, J, β₂)
    Z_pred, (J_gs, beta2_gs) = find_stable_Z_coupled(A)
    N_pred = A - Z_pred
    Delta_Z = Z_pred - Z_exp

    results.append({
        'name': name,
        'A': A,
        'Z_exp': Z_exp,
        'N_exp': N_exp,
        'Z_pred': Z_pred,
        'N_pred': N_pred,
        'Delta_Z': Delta_Z,
        'J_gs': J_gs,
        'beta2_gs': beta2_gs,
    })

    # Progress indicator
    if (i + 1) % 50 == 0:
        print(f"  Processed {i+1}/{len(test_nuclides)} nuclides...")

print()
print("="*95)
print("RESULTS")
print("="*95)
print()

# ============================================================================
# STATISTICS
# ============================================================================
errors = [abs(r['Delta_Z']) for r in results]
exact = sum(e == 0 for e in errors)
within_1 = sum(e <= 1 for e in errors)
within_2 = sum(e <= 2 for e in errors)

print(f"OVERALL PERFORMANCE:")
print(f"  Total nuclides:  {len(results)}")
print(f"  Exact matches:   {exact}/{len(results)} ({100*exact/len(results):.1f}%)")
print(f"  Within ±1:       {within_1}/{len(results)} ({100*within_1/len(results):.1f}%)")
print(f"  Within ±2:       {within_2}/{len(results)} ({100*within_2/len(results):.1f}%)")
print(f"  Mean |ΔZ|:       {np.mean(errors):.3f} charges")
print(f"  Median |ΔZ|:     {np.median(errors):.1f} charges")
print(f"  Max |ΔZ|:        {np.max(errors):.0f} charges")
print()

# By mass region
light = [r for r in results if r['A'] < 40]
medium = [r for r in results if 40 <= r['A'] < 100]
heavy = [r for r in results if 100 <= r['A'] < 200]
superheavy = [r for r in results if r['A'] >= 200]

print("PERFORMANCE BY MASS REGION:")
print("-"*95)
for region_name, group in [("Light (A<40)", light), ("Medium (40≤A<100)", medium),
                            ("Heavy (100≤A<200)", heavy), ("Superheavy (A≥200)", superheavy)]:
    if len(group) > 0:
        errs = [abs(r['Delta_Z']) for r in group]
        ex = sum(e == 0 for e in errs)
        print(f"{region_name:<25} N={len(group):<3} Exact={ex}/{len(group)} ({100*ex/len(group):>5.1f}%)  "
              f"Mean|ΔZ|={np.mean(errs):.2f}")

print()

# Spin statistics
J0_results = [r for r in results if r['J_gs'] == 0]
J_nonzero = [r for r in results if r['J_gs'] != 0]

print("GROUND STATE SPIN DISTRIBUTION:")
print("-"*95)
print(f"J=0 configurations:   {len(J0_results)}/{len(results)} ({100*len(J0_results)/len(results):.1f}%)")
print(f"J≠0 configurations:   {len(J_nonzero)}/{len(results)} ({100*len(J_nonzero)/len(results):.1f}%)")
print()

# Deformation statistics
spherical = [r for r in results if abs(r['beta2_gs']) < 0.05]
deformed = [r for r in results if abs(r['beta2_gs']) >= 0.05]

print("GROUND STATE DEFORMATION DISTRIBUTION:")
print("-"*95)
print(f"Spherical (|β₂|<0.05): {len(spherical)}/{len(results)} ({100*len(spherical)/len(results):.1f}%)")
print(f"Deformed (|β₂|≥0.05):  {len(deformed)}/{len(results)} ({100*len(deformed)/len(results):.1f}%)")
print()

if len(deformed) > 0:
    beta2_values = [r['beta2_gs'] for r in deformed]
    print(f"Deformed nuclei:")
    print(f"  Mean β₂:     {np.mean(beta2_values):.3f}")
    print(f"  Median β₂:   {np.median(beta2_values):.3f}")
    print(f"  Range:       [{np.min(beta2_values):.3f}, {np.max(beta2_values):.3f}]")
    print()

# ============================================================================
# KEY TEST CASES
# ============================================================================
print("="*95)
print("KEY TEST CASES")
print("="*95)
print(f"{'Nuclide':<10} {'A':<5} {'Z_exp':<6} {'Z_pred':<6} {'ΔZ':<6} {'J_gs':<8} {'β₂_gs':<10} {'Description'}")
print("-"*95)

key_cases = [
    ("He-4", 2, 4, "Doubly magic (N=2, Z=2)"),
    ("O-16", 8, 16, "Doubly magic (N=8, Z=8)"),
    ("Ca-40", 20, 40, "Doubly magic (N=20, Z=20)"),
    ("Fe-56", 26, 56, "Most stable nucleus"),
    ("Ni-58", 28, 58, "Magic Z=28"),
    ("Sn-112", 50, 112, "Magic Z=50"),
    ("Pb-208", 82, 208, "Doubly magic (N=126, Z=82)"),
    ("U-238", 92, 238, "Heaviest natural"),
]

for name, Z_exp, A, desc in key_cases:
    # Find result
    r = next((res for res in results if res['name'] == name), None)
    if r:
        status = "✓" if r['Delta_Z'] == 0 else f"{r['Delta_Z']:+d}"
        print(f"{name:<10} {A:<5} {Z_exp:<6} {r['Z_pred']:<6} {status:<6} "
              f"{r['J_gs']:<8.1f} {r['beta2_gs']:<10.3f} {desc}")

print()

# ============================================================================
# COMPARISON TO BASELINE
# ============================================================================
print("="*95)
print("COMPARISON TO BASELINE (spherical, no spin)")
print("="*95)
print()
print(f"Previous accuracy (optimized parameters, spherical):  44.6% exact")
print(f"Current accuracy (coupled spin-shape):                {100*exact/len(results):.1f}% exact")
print()

improvement = 100*exact/len(results) - 44.6
if improvement > 5:
    print(f"✓✓ SIGNIFICANT IMPROVEMENT: +{improvement:.1f}%")
    print("Spin-shape coupling resolves survivors!")
elif improvement > 0:
    print(f"✓ MODEST IMPROVEMENT: +{improvement:.1f}%")
    print("Coupling helps but not transformative.")
elif improvement > -5:
    print(f"NEUTRAL: {improvement:+.1f}%")
    print("Coupling doesn't significantly change predictions.")
else:
    print(f"REGRESSION: {improvement:+.1f}%")
    print("Baseline model was better.")

print()
print("="*95)
print("INTERPRETATION")
print("="*95)
print()
print("The coupled spin-shape optimization reveals:")
print("1. Spin-deformation correlation in ground states")
print("2. Role of pairing (J=0) vs rotational excitation")
print("3. Which survivors are spherical vs deformed")
print()
print("Next step: Analyze which nuclides benefit from coupling.")
print("="*95)
