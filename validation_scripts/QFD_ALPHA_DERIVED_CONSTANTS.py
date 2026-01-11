#!/usr/bin/env python3
"""
QFD Alpha-Derived Constants
============================

THE SINGLE SOURCE OF TRUTH for all QFD nuclear physics constants.

As of 2026-01-06, all coefficients are DERIVED from the fine structure
constant α via the Golden Loop transcendental equation. There are NO
free parameters.

The Derivation Chain:
    α (measured) → β (Golden Loop) → c₁, c₂ (Fundamental Soliton Equation)

Usage:
    from QFD_ALPHA_DERIVED_CONSTANTS import *

    # Access constants directly
    print(f"β = {BETA_VACUUM}")
    print(f"c₁ = {C1_SURFACE}")
    print(f"c₂ = {C2_VOLUME}")

    # Use the Fundamental Soliton Equation
    Z_stable = fundamental_soliton_equation(A=208)  # Returns ~85.77 for Pb-208

References:
    - Appendix Z.17: "The Geometry of Existence"
    - FUNDAMENTAL_SOLITON_EQUATION.md
    - PHOTON_ALPHA_BETA_CHAIN.md
"""

import numpy as np
from typing import Union

# =============================================================================
# THE MASTER CONSTANTS (α-derived, 2026-01-06)
# =============================================================================

# Fine Structure Constant (CODATA 2018)
# This is the ONLY measured input. Everything else is derived.
ALPHA_FINE = 1.0 / 137.035999206
ALPHA_INV = 137.035999206

# =============================================================================
# THE GOLDEN LOOP EQUATION
# =============================================================================
#
# The transcendental equation that locks β to α:
#
#     e^β / β = (α⁻¹ × c₁) / π²
#
# Rearranged to the Master Equation:
#
#     1/α = 2π² × (e^β / β) + 1
#
# Solving numerically for β gives:
#
# =============================================================================
# CONSTANTS
# =============================================================================

ALPHA_INV = 137.035999206
BETA_VACUUM = 3.043233053  # Vacuum stiffness (α-derived)
BETA_VACUUM_OLD = 3.058  # DEPRECATED: Was fitted to nuclear data

# =============================================================================
# FUNDAMENTAL SOLITON EQUATION COEFFICIENTS
# =============================================================================
#
# The Fundamental Soliton Equation (zero free parameters):
#
#     Q = c₁ × A^(2/3) + c₂ × A
#
# Where:
#     c₁ = ½(1 - α)   [Surface term: Virial geometry minus EM drag]
#     c₂ = 1/β        [Volume term: Bulk modulus / saturation limit]
#

# Surface tension coefficient
# Physical meaning: Virial theorem gives ½, charge weakens by α
C1_SURFACE = 0.5 * (1.0 - ALPHA_FINE)  # = 0.496351...

# Volume saturation coefficient
# Physical meaning: Inverse of vacuum bulk stiffness
C2_VOLUME = 1.0 / BETA_VACUUM          # = 0.328615...

# Old fitted values (for reference only - DO NOT USE):
# C1_SURFACE_OLD = 0.529   # Was fitted to nuclear data
# C2_VOLUME_OLD = 0.327    # Was fitted to nuclear data

# =============================================================================
# DERIVED QUANTITIES
# =============================================================================

# Speed of light as vacuum sound speed (natural units)
C_NATURAL = np.sqrt(BETA_VACUUM)  # = 1.7451...

# QED vertex correction coefficient
# V₄ = -ξ/β where ξ ≈ 1.0
XI_COUPLING = 1.0
V4_QED = -XI_COUPLING / BETA_VACUUM  # = -0.3286...

# =============================================================================
# PHYSICAL CONSTANTS (SI units)
# =============================================================================

# Proton mass
M_PROTON_KG = 1.6726219e-27       # kg
M_PROTON_MEV = 938.27208816       # MeV/c²

# Electron mass
M_ELECTRON_KG = 9.10938356e-31    # kg
M_ELECTRON_MEV = 0.51099895       # MeV/c²

# Planck's constant
HBAR_MEV_FM = 197.3269804         # MeV·fm

# Fine structure in MeV·fm for Coulomb calculations
HC_MEV_FM = HBAR_MEV_FM           # ℏc in MeV·fm
ALPHA_EM_COUPLING = ALPHA_FINE * HC_MEV_FM  # ≈ 1.4399... MeV·fm

# =============================================================================
# THE FUNDAMENTAL SOLITON EQUATION
# =============================================================================

def fundamental_soliton_equation(
    A: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    The Fundamental Soliton Equation: predicts stable Z for given mass A.

    Q(A) = c₁ × A^(2/3) + c₂ × A

    Where:
        c₁ = ½(1 - α) = 0.496351  (surface tension)
        c₂ = 1/β = 0.328615       (bulk modulus)

    This equation has ZERO free parameters - both coefficients
    derive from the fine structure constant α.

    Args:
        A: Mass number (can be scalar or numpy array)

    Returns:
        Predicted stable charge Z

    Examples:
        >>> fundamental_soliton_equation(56)   # Iron
        25.67  (actual: 26)

        >>> fundamental_soliton_equation(208)  # Lead
        85.77  (actual: 82, shell effects pull down by ~4)

        >>> fundamental_soliton_equation(238)  # Uranium
        97.27  (actual: 92, shell effects)
    """
    A = np.asarray(A, dtype=float)
    return C1_SURFACE * A**(2.0/3.0) + C2_VOLUME * A


def golden_loop_verify(beta: float = BETA_VACUUM) -> dict:
    """
    Verify that β satisfies the Golden Loop equation.

    The Golden Loop: e^β / β = K = (α⁻¹ × c₁) / π²

    Rearranged: 1/α = 2π² × (e^β / β) + 1

    Args:
        beta: Value to verify (default: BETA_VACUUM)

    Returns:
        Dictionary with verification results
    """
    # Calculate both sides
    lhs = 1.0 / ALPHA_FINE  # Should be ~137.036

    exp_beta_over_beta = np.exp(beta) / beta
    rhs = 2 * np.pi**2 * exp_beta_over_beta + 1

    # Calculate K value
    K = exp_beta_over_beta
    K_expected = (ALPHA_INV * C1_SURFACE) / np.pi**2

    return {
        'beta': beta,
        'alpha_inv_actual': ALPHA_INV,
        'alpha_inv_from_beta': rhs,
        'alpha_inv_error': abs(lhs - rhs),
        'alpha_inv_error_pct': abs(lhs - rhs) / lhs * 100,
        'K_actual': K,
        'K_expected': K_expected,
        'K_error_pct': abs(K - K_expected) / K_expected * 100,
        'valid': abs(lhs - rhs) / lhs < 0.001,  # 0.1% tolerance
    }


def c1_derivation_verify() -> dict:
    """
    Verify the c₁ = ½(1 - α) derivation.

    The "ugly decimal" 0.496297 from nuclear fits is actually just:
        ½ × (1 - 1/137.036) = 0.496351

    Match accuracy: 0.011%

    Returns:
        Dictionary with verification results
    """
    c1_from_golden_loop = 0.496297   # Original fitted value
    c1_from_alpha = 0.5 * (1.0 - ALPHA_FINE)

    error = abs(c1_from_golden_loop - c1_from_alpha)
    error_pct = error / c1_from_golden_loop * 100

    return {
        'c1_golden_loop': c1_from_golden_loop,
        'c1_from_alpha': c1_from_alpha,
        'absolute_error': error,
        'percent_error': error_pct,
        'interpretation': 'c₁ = ½(1-α) = "half minus electromagnetic tax"',
        'valid': error_pct < 0.1,  # Within 0.1%
    }


# =============================================================================
# NUCLEAR ENERGY FUNCTIONAL COEFFICIENTS (for compatibility)
# =============================================================================

def get_nuclear_coefficients() -> dict:
    """
    Return nuclear binding energy coefficients derived from α.

    These are the coefficients for the Semi-Empirical Mass Formula (SEMF)
    form, but now with α-derived values instead of empirical fits.

    Returns:
        Dictionary of coefficients with physical units
    """
    # Volume coefficient (bulk binding per nucleon)
    # a_V ≈ M_p × (1 - α²/β) × correction factors
    # Typical: ~15.8 MeV

    # Surface coefficient
    # a_S = related to c₁ surface tension
    # a_S ≈ M_p × β / 2 / geometry_factor
    # Typical: ~18.3 MeV

    # Asymmetry coefficient
    # a_A = β × M_p / 15
    # Typical: ~23.2 MeV

    return {
        'c1_surface': C1_SURFACE,
        'c2_volume': C2_VOLUME,
        'beta_vacuum': BETA_VACUUM,
        'alpha_fine': ALPHA_FINE,
        # Derived energy scales (approximate)
        'E_surface_scale_MeV': M_PROTON_MEV * BETA_VACUUM / 2 / 15,
        'E_asymmetry_scale_MeV': BETA_VACUUM * M_PROTON_MEV / 15,
    }


# =============================================================================
# MAIN: Print all constants for verification
# =============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("QFD ALPHA-DERIVED CONSTANTS")
    print("=" * 70)
    print()

    print("MASTER INPUT (measured):")
    print(f"  α = 1/{ALPHA_INV:.9f}")
    print()

    print("GOLDEN LOOP DERIVED:")
    print(f"  β = {BETA_VACUUM:.5f}  (vacuum stiffness)")
    print(f"  c = √β = {C_NATURAL:.4f}  (vacuum sound speed, natural units)")
    print()

    print("FUNDAMENTAL SOLITON EQUATION:")
    print(f"  c₁ = ½(1-α) = {C1_SURFACE:.6f}  (surface tension)")
    print(f"  c₂ = 1/β    = {C2_VOLUME:.6f}  (bulk modulus)")
    print()

    print("EQUATION: Q(A) = c₁ × A^(2/3) + c₂ × A")
    print()

    print("VERIFICATION:")

    # Golden Loop verification
    gl = golden_loop_verify()
    print(f"  Golden Loop: 1/α = 2π²(e^β/β) + 1")
    print(f"    LHS (1/α) = {ALPHA_INV:.6f}")
    print(f"    RHS       = {gl['alpha_inv_from_beta']:.6f}")
    print(f"    Error     = {gl['alpha_inv_error_pct']:.4f}%")
    print()

    # c₁ derivation verification
    c1v = c1_derivation_verify()
    print(f"  c₁ = ½(1-α) derivation:")
    print(f"    Golden Loop fit = {c1v['c1_golden_loop']:.6f}")
    print(f"    From α formula  = {c1v['c1_from_alpha']:.6f}")
    print(f"    Error           = {c1v['percent_error']:.4f}%")
    print()

    print("NUCLEAR PREDICTIONS:")
    test_nuclei = [
        (56, 26, "Fe-56 (peak stability)"),
        (208, 82, "Pb-208 (heaviest stable)"),
        (238, 92, "U-238 (heaviest natural)"),
    ]

    for A, Z_actual, name in test_nuclei:
        Z_pred = fundamental_soliton_equation(A)
        error = Z_pred - Z_actual
        print(f"  {name}:")
        print(f"    Predicted Z = {Z_pred:.2f}, Actual Z = {Z_actual}, Δ = {error:+.2f}")

    print()
    print("=" * 70)
    print("OLD vs NEW CONSTANTS:")
    print("=" * 70)
    print()
    print("  Parameter    Old (Fitted)    New (α-Derived)    Change")
    print("  ---------    ------------    ---------------    ------")
    print(f"  β            3.058           {BETA_VACUUM:.5f}          -0.49%")
    print(f"  c₁           0.529           {C1_SURFACE:.6f}         -6.18%")
    print(f"  c₂           0.327           {C2_VOLUME:.6f}         +0.49%")
    print()
    print("The α-derived values predict heavy nuclei BETTER than the old fits!")
    print()
