#!/usr/bin/env python3
"""
QFD SURVIVOR SEARCH - TOPOLOGICAL ECCENTRICITY MODEL
===========================================================================
Allows solitons to optimize their eccentricity to minimize total energy.

Physical mechanism:
  - G_surf = 1 + ecc² → Surface energy increases with deformation
  - G_disp = 1/(1 + ecc) → Vacuum displacement cost decreases with stretching
  - Trade-off finds optimal shape for each (A, Z) configuration

Key insight: "Survivors" in the 44.6% may be those that found the optimal
geometric sweet spot in eccentricity space, not those that are spherical.
===========================================================================
"""

import numpy as np

# ============================================================================
# 1. FUNDAMENTAL CONSTANTS & DERIVED PARAMETERS
# ============================================================================
alpha_fine   = 1.0 / 137.036
beta_vacuum  = 1.0 / 3.043233053
lambda_time  = 0.42
M_proton     = 938.272

V_0 = M_proton * (1 - (alpha_fine**2) / beta_vacuum)
beta_nuclear = M_proton * beta_vacuum / 2
E_volume  = V_0 * (1 - lambda_time / (12 * np.pi))
E_surface = beta_nuclear / 15
a_sym     = (beta_vacuum * M_proton) / 15

# Calibrated Displacement and Resonance
SHIELD_FACTOR = 0.52
a_disp = (alpha_fine * 197.327 / 1.2) * SHIELD_FACTOR
ISOMER_NODES = {2, 8, 20, 28, 50, 82, 126}
BONUS_STRENGTH = 0.70

def get_resonance_bonus(Z, N):
    bonus = 0
    if Z in ISOMER_NODES: bonus += E_surface * BONUS_STRENGTH
    if N in ISOMER_NODES: bonus += E_surface * BONUS_STRENGTH
    if Z in ISOMER_NODES and N in ISOMER_NODES:
        bonus *= 1.5  # Doubly magic enhancement
    return bonus

# ============================================================================
# 2. SURVIVOR ENERGY FUNCTIONAL (Shape-Shifting Active)
# ============================================================================
def qfd_survivor_energy(A, Z, ecc):
    """
    Energy functional with eccentricity-dependent geometry factors.

    Parameters:
      A: Mass number
      Z: Proton number
      ecc: Eccentricity (0 = sphere, >0 = ellipsoidal deformation)

    Returns:
      Total energy in MeV
    """
    # Geometric factors
    G_surf = 1.0 + (ecc**2)         # Surface increases with deformation
    G_disp = 1.0 / (1.0 + ecc)      # Displacement decreases (less packed)

    N = A - Z
    q = Z / A if A > 0 else 0

    # Standard terms
    E_bulk = E_volume * A
    E_asym = a_sym * A * ((1 - 2*q)**2)
    E_iso  = -get_resonance_bonus(Z, N)

    # Deformation-modified terms
    E_surf = E_surface * (A**(2/3)) * G_surf
    E_vac  = a_disp * (Z**2) / (A**(1/3)) * G_disp

    return E_bulk + E_surf + E_asym + E_vac + E_iso

def find_survivor_state(A):
    """
    Searches for the global minimum in (Z, eccentricity) space.

    Returns:
      Z_pred: Predicted proton number
      ecc_pred: Optimal eccentricity
    """
    best_Z, best_ecc, min_E = 0, 0.0, float('inf')

    for z in range(1, A):
        # Test discrete topological deformation steps
        for ecc in np.linspace(0, 0.25, 6):
            energy = qfd_survivor_energy(A, z, ecc)
            if energy < min_E:
                min_E, best_Z, best_ecc = energy, z, ecc

    return best_Z, best_ecc

# ============================================================================
# 3. VERIFICATION: THE XE-136 SURVIVAL TEST
# ============================================================================
print("="*80)
print("QFD SURVIVOR SEARCH - Xe-136 TEST CASE")
print("="*80)
print()
print("Physics: Solitons optimize eccentricity to balance surface vs displacement")
print()

A_Xe, Z_Xe_exp = 136, 54
Z_pred, ecc_pred = find_survivor_state(A_Xe)

print("RESULT:")
print("-" * 80)
print(f"  Mass number A:      {A_Xe}")
print(f"  Experimental Z:     {Z_Xe_exp}")
print(f"  Predicted Z:        {Z_pred}")
print(f"  ΔZ:                 {Z_pred - Z_Xe_exp:+d}")
print(f"  Predicted Ecc:      {ecc_pred:.3f}")
print()

if Z_pred == Z_Xe_exp:
    print("✓ SUCCESS: Topological deformation resolves Xe-136!")
else:
    print(f"✗ Still off by {abs(Z_pred - Z_Xe_exp)} charges")
    print(f"  Previous spherical model predicted Z={50} (ΔZ=-4)")
    print(f"  With eccentricity: Z={Z_pred} (ΔZ={Z_pred - Z_Xe_exp:+d})")
    if abs(Z_pred - Z_Xe_exp) < 4:
        print("  → Partial improvement!")

print()

# Compare energy landscape with and without eccentricity
print("ENERGY LANDSCAPE COMPARISON:")
print("-" * 80)
print(f"{'Z':<6} {'E_sphere':<15} {'E_optimal':<15} {'ecc_opt':<10} {'Improvement'}")
print("-" * 80)

for Z_test in [50, 52, 54, 56]:
    # Spherical (ecc=0)
    E_sphere = qfd_survivor_energy(A_Xe, Z_test, 0.0)

    # Optimal eccentricity for this Z
    best_ecc_Z = 0.0
    best_E_Z = E_sphere
    for ecc in np.linspace(0, 0.25, 20):
        E = qfd_survivor_energy(A_Xe, Z_test, ecc)
        if E < best_E_Z:
            best_E_Z = E
            best_ecc_Z = ecc

    improvement = E_sphere - best_E_Z
    marker = "← exp" if Z_test == Z_Xe_exp else ""

    print(f"{Z_test:<6} {E_sphere:<15.3f} {best_E_Z:<15.3f} {best_ecc_Z:<10.3f} "
          f"{improvement:+.3f} MeV {marker}")

print()
print("="*80)
print("INTERPRETATION")
print("="*80)
print()
print("If optimal ecc > 0 at experimental Z:")
print("  → Soliton prefers deformed shape (prolate/oblate)")
print("  → Spherical approximation was too rigid")
print()
print("If Z_pred moves closer to Z_exp:")
print("  → Eccentricity freedom partially resolves magic number pull")
print("  → Survivors are shape-shifters, not spheres!")
print()
print("="*80)
