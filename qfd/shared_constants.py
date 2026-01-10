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
# Rearranged: e^β / β = (1/α - 1) / (2π²)
#
# This transcendental equation has a unique solution β ≈ 3.043233
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
# Reference: projects/Lean4/QFD/Nuclear/CoreCompressionLaw.lean

# Surface tension coefficient (DERIVED)
# Physical interpretation: Virial theorem geometry (½) minus EM drag (α)
C1_SURFACE = 0.5 * (1 - ALPHA)  # = 0.496351

# Volume coefficient (DERIVED)
# Physical interpretation: Vacuum bulk modulus (saturation limit)
C2_VOLUME = 1.0 / BETA  # = 0.328598

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

# Hubble constant
H0_KM_S_MPC = 70.0  # km/s/Mpc
MPC_TO_M = 3.086e22  # meters per Mpc
H0_SI = H0_KM_S_MPC * 1000 / MPC_TO_M  # s⁻¹

# QFD photon decay constant: κ = H₀/c
KAPPA_MPC = H0_KM_S_MPC / (C_SI / 1000)  # Mpc⁻¹

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
# SELF-TEST
# =============================================================================

def verify_constants():
    """Verify all derived constants match expected values."""
    print("=" * 60)
    print("QFD SHARED CONSTANTS - VERIFICATION")
    print("=" * 60)

    print(f"\nMASTER INPUT:")
    print(f"  α = 1/{ALPHA_INV:.9f}")

    print(f"\nGOLDEN LOOP DERIVED:")
    print(f"  β = {BETA:.9f} (vacuum stiffness)")
    print(f"  β_standardized = {BETA_STANDARDIZED} (Lean proofs)")
    print(f"  c = √β = {C_NATURAL:.6f} (natural units)")

    print(f"\nNUCLEAR COEFFICIENTS:")
    print(f"  c₁ = ½(1-α) = {C1_SURFACE:.6f}")
    print(f"  c₂ = 1/β    = {C2_VOLUME:.6f}")

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
        'v4_error': v4_err
    }

if __name__ == "__main__":
    verify_constants()
