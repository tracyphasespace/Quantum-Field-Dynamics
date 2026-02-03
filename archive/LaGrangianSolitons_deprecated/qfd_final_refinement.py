#!/usr/bin/env python3
"""
QFD MASS FORMULA - FINAL REFINEMENT
===========================================================================
Refining the two best formulas from qfd_named_constants.py:
  - Formula D: RMS 1.31%
  - Formula E: RMS 0.81%

Both have E_volume ≈ M_proton (within 1%), which is correct!
Need to find the right combination for E_surface.
===========================================================================
"""

import numpy as np

# Constants
alpha_fine   = 1.0 / 137.036
beta_vacuum  = 1.0 / 3.043233053
lambda_time  = 0.42
M_proton     = 938.272  # MeV

# Targets
E_volume_target   = 927.652  # MeV
E_surface_target  = 10.195   # MeV

# Test nuclei
test_nuclei = [
    ("H-1",   1, 1, 938.272),
    ("He-4",  4, 2, 3727.379),
    ("C-12", 12, 6, 11174.862),
    ("O-16", 16, 8, 14895.079),
    ("Ca-40",40,20, 37211.000),
    ("Fe-56",56,26, 52102.500),
]

def test_formula(name, E_vol, E_surf):
    """Test a formula given E_volume and E_surface coefficients."""
    print(f"\n{name}")
    print("-"*85)
    print(f"E_volume  = {E_vol:>10.3f} MeV  (target: {E_volume_target:.3f}, error: {100*(E_vol-E_volume_target)/E_volume_target:+.2f}%)")
    print(f"E_surface = {E_surf:>10.3f} MeV  (target: {E_surface_target:.3f}, error: {100*(E_surf-E_surface_target)/E_surface_target:+.2f}%)")
    print()
    print(f"{'Nucleus':<8} {'A':>3} {'Exp(MeV)':>11} {'QFD(MeV)':>11} {'Error':>11} {'%':>9}")
    print("-"*85)

    errors = []
    for name_n, A, Z, m_exp in test_nuclei:
        m_qfd = E_vol * A + E_surf * (A ** (2/3))
        error = m_qfd - m_exp
        error_pct = 100 * error / m_exp
        errors.append(abs(error_pct))

        print(f"{name_n:<8} {A:>3} {m_exp:>11.2f} {m_qfd:>11.2f} "
              f"{error:>+11.2f} {error_pct:>+8.2f}%")

    rms = np.sqrt(np.mean([e**2 for e in errors]))
    print(f"{'':>36} RMS error: {rms:>6.2f}%")
    return rms

print("="*85)
print("QFD MASS FORMULA - FINAL REFINEMENT")
print("="*85)
print(f"\nConstants:")
print(f"  alpha_fine   = {alpha_fine:.6f}")
print(f"  beta_vacuum  = {beta_vacuum:.6f}")
print(f"  lambda_time  = {lambda_time}")
print(f"  M_proton     = {M_proton} MeV")
print()

# Original winners from previous test
E_vol_E = M_proton
E_surf_E = M_proton * alpha_fine * beta_vacuum
test_formula("Formula E (original, 0.81% RMS)", E_vol_E, E_surf_E)

E_vol_D = M_proton / (1 + alpha_fine * beta_vacuum)
E_surf_D = M_proton * alpha_fine / lambda_time
test_formula("Formula D (original, 1.31% RMS)", E_vol_D, E_surf_D)

# Refinement 1: Average E_surface from D and E
E_vol_1 = M_proton
E_surf_1 = (E_surf_D + E_surf_E) / 2
test_formula("Refinement 1: E_surf = avg(D, E)", E_vol_1, E_surf_1)

# Refinement 2: Geometric mean
E_vol_2 = M_proton
E_surf_2 = np.sqrt(E_surf_D * E_surf_E)
test_formula("Refinement 2: E_surf = sqrt(D × E)", E_vol_2, E_surf_2)

# Refinement 3: E_surface with 1/sqrt(lambda_time) factor
E_vol_3 = M_proton
E_surf_3 = M_proton * alpha_fine * beta_vacuum / np.sqrt(lambda_time)
test_formula("Refinement 3: E_surf = M_p × α × β / sqrt(λ)", E_vol_3, E_surf_3)

# Refinement 4: E_surface with square root of beta
E_vol_4 = M_proton
E_surf_4 = M_proton * alpha_fine * np.sqrt(beta_vacuum) / lambda_time
test_formula("Refinement 4: E_surf = M_p × α × sqrt(β) / λ", E_vol_4, E_surf_4)

# Refinement 5: E_surface with (alpha/lambda) × beta
E_vol_5 = M_proton
E_surf_5 = M_proton * (alpha_fine / lambda_time) * beta_vacuum
test_formula("Refinement 5: E_surf = M_p × (α/λ) × β", E_vol_5, E_surf_5)

# Refinement 6: Combine D's volume with adjusted surface
E_vol_6 = M_proton / (1 + alpha_fine * beta_vacuum)
E_surf_6 = M_proton * alpha_fine * beta_vacuum / lambda_time
test_formula("Refinement 6: D's E_vol, E_surf = M_p × α × β / λ", E_vol_6, E_surf_6)

# Refinement 7: Target-matched (use fitted values for comparison)
E_vol_7 = E_volume_target
E_surf_7 = E_surface_target
test_formula("Target (fitted from data)", E_vol_7, E_surf_7)

# Refinement 8: E_volume from D, E_surface empirically scaled
E_vol_8 = M_proton / (1 + alpha_fine * beta_vacuum)
E_surf_8 = M_proton * alpha_fine / lambda_time * (10.195 / 16.302)  # Scale D's E_surf to match target
test_formula("Refinement 8: D scaled to match targets", E_vol_8, E_surf_8)

# Refinement 9: Try lambda_time^2 in denominator
E_vol_9 = M_proton
E_surf_9 = M_proton * alpha_fine * beta_vacuum / (lambda_time ** 2)
test_formula("Refinement 9: E_surf = M_p × α × β / λ²", E_vol_9, E_surf_9)

# Refinement 10: Try (alpha × beta)^(1/3)
E_vol_10 = M_proton
E_surf_10 = M_proton * (alpha_fine * beta_vacuum) ** (1/3)
test_formula("Refinement 10: E_surf = M_p × (α × β)^(1/3)", E_vol_10, E_surf_10)

print("\n" + "="*85)
print("CONCLUSION")
print("="*85)
print("Looking for RMS error < 0.5% with fundamental constants only (no fitting)")
print("="*85)
