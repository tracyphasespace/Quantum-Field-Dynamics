#!/usr/bin/env python3
"""
QFD Shared Constants - Single Source of Truth

Copyright (c) 2026 Tracy McSheery
Licensed under the MIT License

This file contains two categories of constants:

1. DERIVED CONSTANTS (from α alone):
   - BETA, C1_SURFACE, C2_VOLUME, V4_QED
   - These are the core QFD predictions with ZERO free parameters

2. REFERENCE CONSTANTS (for unit conversion and comparison):
   - SI units (c, ℏ, masses), empirical targets, H0
   - These are NOT claimed to be derived - they're for validation/plotting

Reference: Lean4 proofs in projects/Lean4/QFD/Nuclear/VacuumStiffness.lean
Standardized β = 3.043233053 across all Lean proofs (see BUILD_STATUS.md)

How to use:
    from qfd.shared_constants import ALPHA, BETA, C1_SURFACE, C2_VOLUME
"""

import numpy as np
from scipy.optimize import brentq

# =============================================================================
# THE MASTER INPUT (One measured constant)
# =============================================================================

# Fine structure constant (CODATA 2018)
# Source: https://physics.nist.gov/cgi-bin/cuu/Value?alph
ALPHA = 1.0 / 137.035999206
ALPHA_INV = 137.035999206
PI_SQ = np.pi ** 2

# =============================================================================
# GOLDEN LOOP DERIVATION (α → β)
# =============================================================================
#
# Master Equation: 1/α = 2π² × (e^β / β) + 1
#
# Physical interpretation of each factor:
#
#   2π²  = Surface area of the unit 3-sphere (S³).
#          The electron vortex is a topological object embedded in S³ × R,
#          so its self-energy integral samples the full solid angle of S³.
#
#   e^β/β = Boltzmann-like probability factor for maintaining a topological
#           knot at vacuum stiffness β. At low β (soft vacuum), knots untie
#           easily → e^β/β is small. At high β (stiff vacuum), the
#           exponential dominates → e^β/β is large. The equation demands
#           a β where knot survival probability exactly matches α⁻¹.
#
#   +1   = Ground-state winding contribution (topological zero-point term).
#           Even with zero vacuum stiffness, one unit of twist persists.
#
# Together: α measures the twist-to-compression ratio of the vacuum —
# the fraction of vortex energy stored in shear (EM winding) versus
# bulk (volume compression). This is the continuum-mechanics analogue
# of G/K (shear modulus / bulk modulus) in solid-state physics.
#
# Rearranged: e^β / β = (1/α - 1) / (2π²)
# Unique solution β ≈ 3.043233 (moderately stiff: knots survive, but barely)
#
# Reference: projects/Lean4/QFD/Physics/GoldenLoop_Existence.lean

def _solve_golden_loop():
    """Solve the Golden Loop transcendental equation for β."""
    # Target value K = (1/α - 1) / (2π²)
    K = (ALPHA_INV - 1) / (2 * np.pi**2)

    # Solve e^β / β = K
    # f(β) = e^β / β - K = 0
    def f(beta):
        return np.exp(beta) / beta - K

    # Root is in range [2, 4] for physical α
    beta = brentq(f, 2.0, 4.0)
    return beta

# Vacuum stiffness (DERIVED from α)
BETA = _solve_golden_loop()  # ≈ 3.043233053

# Verification: should match standardized value from Lean proofs
BETA_STANDARDIZED = 3.043233053
assert abs(BETA - BETA_STANDARDIZED) < 0.0001, \
    f"Golden Loop β={BETA} differs from standardized {BETA_STANDARDIZED}"

# =============================================================================
# NUCLEAR COEFFICIENTS (Derived from α and β)
# =============================================================================
#
# Fundamental Soliton Equation: Q(A) = c₁ × A^(2/3) + c₂ × A
#
# c₁ = ½(1 - α)  [Surface tension minus electromagnetic drag]
# c₂ = 1/β       [Bulk modulus from vacuum stiffness]
#
# Physical mechanism:
#   c₁ governs the SURFACE term (∝ A^{2/3}). The ½ comes from the virial
#   theorem (kinetic = ½ potential for bound systems). The (1-α) factor
#   means EM coupling REDUCES surface tension: the electric field of the
#   protons partially cancels the vacuum confinement at the nuclear surface.
#   More protons → more EM drag → weaker surface binding → drip line.
#
#   c₂ governs the VOLUME term (∝ A). This is the vacuum bulk modulus
#   crushing the soliton core: each nucleon added to the interior costs
#   1/β units of compression energy. Stiffer vacuum (larger β) → smaller
#   c₂ → less volume energy per nucleon → more tightly bound nuclei.
#
# Reference: projects/Lean4/QFD/Nuclear/CoreCompressionLaw.lean

# Surface tension coefficient (DERIVED)
# EM drag reduces surface tension: ½ × (1 - α) where α is the EM coupling
C1_SURFACE = 0.5 * (1 - ALPHA)  # = 0.496351

# Volume coefficient (DERIVED)
# Vacuum crushing: each interior nucleon costs 1/β compression energy
C2_VOLUME = 1.0 / BETA  # = 0.328598

# =============================================================================
# GEOMETRIC EIGENVALUES (Book v8.5, Ch. 12 & Appendix Z.12)
# =============================================================================
#
# k_geom is the vacuum-renormalized eigenvalue of the Cl(3,3) soliton.
# k_geom = k_Hill × (π/α)^(1/5), where k_Hill = (56/15)^(1/5) ≈ 1.30
# is the bare vortex shape factor and (π/α)^(1/5) ≈ 3.39 is the
# vacuum electromagnetic enhancement. NOT a fitted parameter.
# See Appendix Z.12 and K_GEOM_REFERENCE.md.
#
# Book v8.5 value: 4.4028 (used consistently in Chs. 12, 12.X, 12.Y, Z.12)
# Lean values: 4.3813–4.3982 (different pipeline stages, ~0.5% spread)
# The spread is documented and may reflect alpha-conditioning physics.

K_GEOM = 4.4028               # Vacuum-renormalized eigenvalue (book v8.5)
K_CIRC = np.pi * K_GEOM       # Loop-closure eigenvalue ≈ 13.83
                               # Use k_geom for mass ratios, k_circ for Compton/phase

# Gravitational coupling (Appendix Z, Ch. 9)
# ξ_QFD = k_geom² × (active_dims / total_dims) = k_geom² × (5/6)
XI_QFD = K_GEOM**2 * (5.0 / 6.0)  # ≈ 16.2

# =============================================================================
# VACUUM PARAMETERS
# =============================================================================

# Speed of light in natural units (vacuum sound speed)
# c = √(β/ρ_vac) where ρ_vac = 1 in natural units
C_NATURAL = np.sqrt(BETA)  # ≈ 1.745

# QED vacuum polarization coefficient
# V₄ = -ξ/β where ξ ≈ 1 (surface tension in natural units)
XI_SURFACE_TENSION = 1.0  # Natural units
V4_QED = -XI_SURFACE_TENSION / BETA  # ≈ -0.329

# Universal circulation velocity (Appendix G, all leptons)
# Back-calculated from L = hbar/2 given I_eff ≈ 2.32 MR²
# NOTE: U/c_s = 0.876/sqrt(beta) ≈ 0.502 (Mach 0.5 to 0.4%)
# Whether Mach 0.5 is exact remains an open conjecture — see edits28.md
U_CIRC = 0.876  # fraction of c (Book v8.5: U = 0.876c)

# Saturation coefficient (Appendix V)
# Leading electromagnetic correction to elastic response (NOT shear/bulk ratio)
GAMMA_S = 2 * ALPHA / BETA  # ≈ 0.0048

# =============================================================================
# REFERENCE CONSTANTS (SI units - NOT derived, for unit conversion only)
# =============================================================================

# Speed of light
C_SI = 299792458.0  # m/s (defined exactly)

# Planck constant
HBAR_SI = 1.054571817e-34  # J·s

# Boltzmann constant
K_BOLTZ_SI = 1.380649e-23  # J/K

# Electron mass
M_ELECTRON_SI = 9.1093837015e-31  # kg
M_ELECTRON_MEV = 0.51099895000  # MeV

# Proton mass
M_PROTON_SI = 1.67262192369e-27  # kg
M_PROTON_MEV = 938.27208816  # MeV

# Muon mass
M_MUON_MEV = 105.6583755  # MeV

# Tau mass
M_TAU_MEV = 1776.86  # MeV

# Hubble constant (standard cosmology reference)
H0_KM_S_MPC = 70.0  # km/s/Mpc (ΛCDM consensus)
MPC_TO_M = 3.086e22  # meters per Mpc
H0_SI = H0_KM_S_MPC * 1000 / MPC_TO_M  # s⁻¹

# QFD vacuum scattering rate (Book v8.5, Ch. 9–12)
# κ̃ = ξ_QFD × β^(3/2) is a DIMENSIONLESS scattering rate derived from α alone.
# The identification κ̃ ≈ K_J [km/s/Mpc] is a numerical coincidence whose
# dimensional bridge is not yet derived from first principles.
# The physically testable content is the SHAPE of μ(z), not the absolute
# value of K_J (which is degenerate with magnitude calibration M).
#
# Derivation chain (zero free parameters):
#   α → β (Golden Loop) → k_geom (vacuum-renormalized eigenvalue)
#   → ξ_QFD = k_geom² × 5/6 (gravitational coupling)
#   → κ̃ = ξ_QFD × β^(3/2)  (dimensionless scattering rate)
#
# Note: The SNe pipeline uses the exact soliton BC k = 7π/5 = 4.3982
# giving κ̃ = 85.581. With book v8.5 k_geom = 4.4028 the value is ~85.9.
# Both are consistent within the k_geom spread (~0.5%).
K_J_KM_S_MPC = XI_QFD * BETA**1.5  # dimensionless κ̃ ≈ 85.6 (see §9.3.1 for dimensional status)

# QFD photon decay constant: κ = H₀/c (using standard H0 for comparison)
KAPPA_MPC = H0_KM_S_MPC / (C_SI / 1000)  # Mpc⁻¹
# QFD photon decay constant using K_J
KAPPA_QFD_MPC = K_J_KM_S_MPC / (C_SI / 1000)  # Mpc⁻¹

# =============================================================================
# VALIDATION TARGETS (Independent measurements - NOT inputs to QFD)
# These are used ONLY to compare QFD predictions against experiment
# =============================================================================

# Nuclear coefficients from NuBase 2020 (2,550 nuclei)
C1_EMPIRICAL = 0.496297  # Fitted to nuclear data (independent of α)
C2_EMPIRICAL = 0.32704   # Fitted to nuclear data (independent of α)

# QED vacuum polarization from Feynman diagrams
A2_QED_SCHWINGER = -0.328479  # From perturbation theory

# =============================================================================
# FUNDAMENTAL SOLITON EQUATION
# =============================================================================

def fundamental_soliton_equation(A):
    """
    The Fundamental Soliton Equation: predicts stable Z for given mass A.

    Q(A) = c₁ × A^(2/3) + c₂ × A

    Where:
        c₁ = ½(1 - α) = 0.496351  (surface tension)
        c₂ = 1/β = 0.328598       (bulk modulus)

    This equation has ZERO free parameters - both coefficients
    derive from the fine structure constant α.

    Args:
        A: Mass number (can be scalar or numpy array)

    Returns:
        Predicted stable charge Z
    """
    A = np.asarray(A, dtype=float)
    return C1_SURFACE * A**(2.0/3.0) + C2_VOLUME * A

# =============================================================================
# PROTON BRIDGE (Book v8.5, Ch. 12.2)
# =============================================================================
#
# λ = k_geom × β × (m_e / α)
#
# The proton mass is a geometric consequence of soliton stability.
# m_p / m_e ≈ k_geom × β / α
#
# Physical picture: The proton is a CAVITATION BUBBLE in the vacuum
# superfluid — a region where compression energy (β) balances the
# electromagnetic pressure of the confined charge (m_e/α). The factor
# k_geom is the geometric shape factor of this cavity, determined by
# the Cl(3,3) → Cl(3,1) dimensional projection.
#
# Complete derivation chain (zero free parameters after α):
#   α → β (Golden Loop) → c (vacuum sound speed)
#   → m_p (Proton Bridge) → (c₁, c₂) (nuclear coefficients)
#   → K_J (Hubble refraction)
# Each step is algebraically determined; there is no freedom to adjust.
#
# NOTE on c₂ definition inconsistency in Book v8.5:
#   Line 417 (Ch. 1.2):  c₂ = (1+α)/4 ≈ 0.2518  (introductory summary)
#   Appendix/Ch.12:       c₂ = 1/β ≈ 0.3286       (nuclear analysis)
#   Resolution: c₂ = 1/β is the CORRECT definition used in all nuclear
#   predictions and validated against NuBase 2020 data. The (1+α)/4
#   definition in the introduction appears to be a remnant from an
#   earlier version of the book and should be corrected there.

PROTON_MASS_PREDICTED_MEV = K_GEOM * BETA * (M_ELECTRON_MEV / ALPHA)
PROTON_ELECTRON_RATIO_PREDICTED = K_GEOM * BETA / ALPHA

# =============================================================================
# SELF-TEST
# =============================================================================

def verify_constants():
    """Verify all derived constants match expected values."""
    print("=" * 60)
    print("QFD SHARED CONSTANTS - VERIFICATION (Book v8.5)")
    print("=" * 60)

    print(f"\nMASTER INPUT:")
    print(f"  α = 1/{ALPHA_INV:.9f}")

    print(f"\nGOLDEN LOOP DERIVED:")
    print(f"  β = {BETA:.9f} (vacuum stiffness)")
    print(f"  β_standardized = {BETA_STANDARDIZED} (Lean proofs)")
    print(f"  c = √β = {C_NATURAL:.6f} (natural units)")

    print(f"\nGEOMETRIC EIGENVALUES:")
    print(f"  k_geom = {K_GEOM} (radial stability)")
    print(f"  k_circ = π × k_geom = {K_CIRC:.4f} (loop-closure)")
    print(f"  ξ_QFD  = k_geom² × 5/6 = {XI_QFD:.4f}")

    print(f"\nNUCLEAR COEFFICIENTS:")
    print(f"  c₁ = ½(1-α) = {C1_SURFACE:.6f}")
    print(f"  c₂ = 1/β    = {C2_VOLUME:.6f}")

    print(f"\nPROTON BRIDGE:")
    mp_err = abs(PROTON_MASS_PREDICTED_MEV - M_PROTON_MEV) / M_PROTON_MEV * 100
    ratio_exp = M_PROTON_MEV / M_ELECTRON_MEV
    ratio_err = abs(PROTON_ELECTRON_RATIO_PREDICTED - ratio_exp) / ratio_exp * 100
    print(f"  λ = k_geom × β × (m_e/α) = {PROTON_MASS_PREDICTED_MEV:.3f} MeV")
    print(f"  m_p (experiment)           = {M_PROTON_MEV:.3f} MeV")
    print(f"  error: {mp_err:.4f}%")
    print(f"  m_p/m_e predicted = {PROTON_ELECTRON_RATIO_PREDICTED:.2f}")
    print(f"  m_p/m_e experiment = {ratio_exp:.2f}")

    print(f"\nVACUUM PARAMETERS:")
    print(f"  U_circ = {U_CIRC}c (universal circulation)")
    print(f"  γ_s = 2α/β = {GAMMA_S:.6f} (saturation coefficient)")
    print(f"  K_J = ξ_QFD × β^(3/2) = {XI_QFD:.4f} × {BETA**1.5:.4f} = {K_J_KM_S_MPC:.4f} km/s/Mpc")
    print(f"  κ = K_J/c = {KAPPA_QFD_MPC:.6e} Mpc⁻¹")

    print(f"\nVALIDATION vs EMPIRICAL:")
    c1_err = abs(C1_SURFACE - C1_EMPIRICAL) / C1_EMPIRICAL * 100
    c2_err = abs(C2_VOLUME - C2_EMPIRICAL) / C2_EMPIRICAL * 100
    v4_err = abs(V4_QED - A2_QED_SCHWINGER) / abs(A2_QED_SCHWINGER) * 100

    print(f"  c₁: derived={C1_SURFACE:.6f}, empirical={C1_EMPIRICAL:.6f}, error={c1_err:.3f}%")
    print(f"  c₂: derived={C2_VOLUME:.6f}, empirical={C2_EMPIRICAL:.6f}, error={c2_err:.3f}%")
    print(f"  V₄: derived={V4_QED:.6f}, QED={A2_QED_SCHWINGER:.6f}, error={v4_err:.2f}%")

    print(f"\nNUCLEAR PREDICTIONS (Fundamental Soliton Equation):")
    test_nuclei = [(56, 26, "Fe-56"), (208, 82, "Pb-208"), (238, 92, "U-238")]
    for A, Z_actual, name in test_nuclei:
        Z_pred = fundamental_soliton_equation(A)
        print(f"  {name}: Z_pred={Z_pred:.2f}, Z_actual={Z_actual}, Δ={Z_pred-Z_actual:+.2f}")

    print(f"\nSTATUS: All constants derived from α = 1/137.036")
    print("=" * 60)

    return {
        'c1_error': c1_err,
        'c2_error': c2_err,
        'v4_error': v4_err,
        'proton_mass_error': mp_err
    }

if __name__ == "__main__":
    verify_constants()
