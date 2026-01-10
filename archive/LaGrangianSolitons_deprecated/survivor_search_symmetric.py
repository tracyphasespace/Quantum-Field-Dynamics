#!/usr/bin/env python3
"""
QFD SURVIVOR SEARCH - SYMMETRIC COUPLING (CORRECTED)
===========================================================================
Tests corrected eccentricity coupling where BOTH terms increase with deformation:

  G_surf = 1 + ecc²
  G_disp = 1 + k·ecc²   (k = 0.5, symmetric penalty)

Physical rationale:
  - Ellipsoid has MORE surface area than sphere → G_surf > 1 ✓
  - Ellipsoid has HIGHER peak density → G_disp > 1 ✓
  - Both terms penalize deformation, so only nuclei that benefit
    (via resonance locking or other effects) will deform
===========================================================================
"""

import numpy as np

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

SHIELD_FACTOR = 0.52
a_disp = (alpha_fine * 197.327 / 1.2) * SHIELD_FACTOR
ISOMER_NODES = {2, 8, 20, 28, 50, 82, 126}
BONUS_STRENGTH = 0.70

# CORRECTED: Displacement coupling strength
K_DISP = 0.5  # Symmetric with surface (both penalize deformation)

def get_resonance_bonus(Z, N):
    bonus = 0
    if Z in ISOMER_NODES: bonus += E_surface * BONUS_STRENGTH
    if N in ISOMER_NODES: bonus += E_surface * BONUS_STRENGTH
    if Z in ISOMER_NODES and N in ISOMER_NODES:
        bonus *= 1.5
    return bonus

# ============================================================================
# CORRECTED ENERGY FUNCTIONAL
# ============================================================================
def qfd_survivor_energy_symmetric(A, Z, ecc):
    """
    Energy with SYMMETRIC eccentricity coupling.

    Both G_surf and G_disp > 1 for ecc > 0 (deformation is penalized).
    """
    G_surf = 1.0 + (ecc**2)           # Increases surface
    G_disp = 1.0 + K_DISP * (ecc**2)  # Increases displacement (corrected!)

    N = A - Z
    q = Z / A if A > 0 else 0

    E_bulk = E_volume * A
    E_surf = E_surface * (A**(2/3)) * G_surf
    E_asym = a_sym * A * ((1 - 2*q)**2)
    E_vac  = a_disp * (Z**2) / (A**(1/3)) * G_disp
    E_iso  = -get_resonance_bonus(Z, N)

    return E_bulk + E_surf + E_asym + E_vac + E_iso

def find_survivor_state_symmetric(A, ecc_max=0.25, n_ecc=6):
    """Find optimal (Z, ecc) with symmetric coupling."""
    best_Z, best_ecc, min_E = 0, 0.0, float('inf')

    for z in range(1, A):
        for ecc in np.linspace(0, ecc_max, n_ecc):
            energy = qfd_survivor_energy_symmetric(A, z, ecc)
            if energy < min_E:
                min_E, best_Z, best_ecc = energy, z, ecc

    return best_Z, best_ecc

# ============================================================================
# QUICK TEST: XE-136
# ============================================================================
print("="*80)
print("CORRECTED SYMMETRIC COUPLING - Xe-136 TEST")
print("="*80)
print()
print(f"Coupling: G_surf = 1 + ecc², G_disp = 1 + {K_DISP}·ecc²")
print("(Both terms increase with deformation)")
print()

A_Xe, Z_Xe_exp = 136, 54
Z_pred, ecc_pred = find_survivor_state_symmetric(A_Xe)

print("RESULT:")
print("-" * 80)
print(f"  Experimental Z:     {Z_Xe_exp}")
print(f"  Predicted Z:        {Z_pred}")
print(f"  ΔZ:                 {Z_pred - Z_Xe_exp:+d}")
print(f"  Optimal ecc:        {ecc_pred:.3f}")
print()

if Z_pred == Z_Xe_exp:
    print("✓ SUCCESS: Symmetric coupling resolves Xe-136!")
elif abs(Z_pred - Z_Xe_exp) < 4:
    print(f"✓ IMPROVEMENT: Closer than asymmetric model (ΔZ={Z_pred - Z_Xe_exp:+d} vs +5)")
else:
    print(f"Still problematic (ΔZ={Z_pred - Z_Xe_exp:+d})")

print()

# Energy landscape
print("ENERGY LANDSCAPE (Symmetric vs Asymmetric):")
print("-" * 80)
print(f"{'Z':<6} {'E_sym(ecc=0)':<15} {'E_sym(opt)':<15} {'ecc_opt':<10}")
print("-" * 80)

for Z_test in [50, 52, 54, 56]:
    E_sphere = qfd_survivor_energy_symmetric(A_Xe, Z_test, 0.0)

    best_ecc = 0.0
    best_E = E_sphere
    for ecc in np.linspace(0, 0.25, 20):
        E = qfd_survivor_energy_symmetric(A_Xe, Z_test, ecc)
        if E < best_E:
            best_E = E
            best_ecc = ecc

    marker = "← exp" if Z_test == Z_Xe_exp else ""
    print(f"{Z_test:<6} {E_sphere:<15.3f} {best_E:<15.3f} {best_ecc:<10.3f} {marker}")

print()
print("If ecc_opt ≈ 0: Sphere is optimal (deformation penalized on both fronts)")
print("If ecc_opt > 0: Some other benefit (resonance?) overcomes penalties")
print()
print("="*80)
