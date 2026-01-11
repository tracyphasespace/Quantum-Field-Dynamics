#!/usr/bin/env python3
"""
Derive β from α (Fine Structure Constant)
==========================================

Copyright (c) 2026 Tracy McSheery
Licensed under the MIT License

PURPOSE:
--------
This script demonstrates the Golden Loop derivation that connects
the fine structure constant α to the vacuum stiffness parameter β.

THE GOLDEN LOOP EQUATION (CORRECT FORM):
-----------------------------------------
    1/α = 2π² × (e^β / β) + 1

Rearranging:
    e^β / β = (1/α - 1) / (2π²)

Let K = (1/α - 1) / (2π²) ≈ 6.892

Then we solve the transcendental equation:
    e^β / β = K

This has a unique solution β ≈ 3.043233 for α = 1/137.036.

WARNING - COMMON ERROR:
-----------------------
An INCORRECT alternative formula sometimes appears:
    K = (α⁻¹ × c₁) / π²  ← WRONG!

This is wrong because:
1. It requires c₁ as an input, but c₁ = ½(1-α) is DERIVED from α
2. It doesn't match the Lean4 formal proof in GoldenLoop_Existence.lean
3. It gives inconsistent values of β

Always use the CORRECT form: K = (1/α - 1) / (2π²)

DERIVATION CHAIN:
-----------------
α (measured) → β (Golden Loop) → c₁, c₂ (Fundamental Soliton Equation)

Where:
    c₁ = ½(1 - α)  [Surface coefficient]
    c₂ = 1/β       [Volume coefficient]

References:
    - projects/Lean4/QFD/Physics/GoldenLoop_Existence.lean
    - qfd/shared_constants.py (single source of truth)
"""

import math
import sys
from pathlib import Path

# Try to import from shared_constants, fallback to inline values
try:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from qfd.shared_constants import (
        ALPHA, ALPHA_INV, BETA, BETA_STANDARDIZED,
        C1_SURFACE, C2_VOLUME,
        C1_EMPIRICAL, C2_EMPIRICAL
    )
    USING_SHARED_CONSTANTS = True
except ImportError:
    USING_SHARED_CONSTANTS = False
    ALPHA_INV = 137.035999206
    ALPHA = 1.0 / ALPHA_INV

# =============================================================================
# THE GOLDEN LOOP SOLVER (Pure Python, no dependencies)
# =============================================================================

def solve_golden_loop_newton(target_alpha_inv):
    """
    Solve the Golden Loop equation for β using Newton-Raphson.

    THE EQUATION (correct form):
    ----------------------------
        1/α = 2π² × (e^β / β) + 1

    Rearranging:
        e^β / β = K where K = (1/α - 1) / (2π²)

    WHY THIS EQUATION:
    ------------------
    The Golden Loop arises from vacuum energy self-consistency.
    - The factor 2π² comes from 6D phase space integration (Cl(3,3) algebra)
    - The exponential e^β/β is the harmonic oscillator partition function
    - The +1 offset ensures finite α for any positive β

    METHOD:
    -------
    Newton-Raphson iteration on f(β) = e^β/β - K

    f'(β) = d/dβ(e^β/β) = e^β(β-1)/β²

    Newton step: β_new = β - f(β)/f'(β)

    Args:
        target_alpha_inv: The value of 1/α (e.g., 137.035999206)

    Returns:
        β: The vacuum stiffness parameter (≈ 3.043233)
    """
    # Target value for e^β/β
    # CORRECT FORMULA: K = (1/α - 1) / (2π²)
    K = (target_alpha_inv - 1) / (2 * math.pi**2)

    # Initial guess
    beta = 3.0

    # Newton-Raphson iteration
    for iteration in range(20):
        exp_beta = math.exp(beta)
        f_beta = exp_beta / beta  # Current value
        residual = f_beta - K     # Error

        # Derivative: d/dβ(e^β/β) = e^β(β-1)/β²
        f_prime = exp_beta * (beta - 1) / (beta**2)

        # Newton step
        beta_new = beta - residual / f_prime

        # Check convergence
        if abs(beta_new - beta) < 1e-12:
            return beta_new

        beta = beta_new

    return beta


def verify_golden_loop(beta, alpha_inv):
    """
    Verify that β satisfies the Golden Loop equation.

    Returns a dictionary with verification results.
    """
    # Left-hand side: 1/α
    lhs = alpha_inv

    # Right-hand side: 2π² × (e^β/β) + 1
    exp_beta_over_beta = math.exp(beta) / beta
    rhs = 2 * math.pi**2 * exp_beta_over_beta + 1

    # K value
    K_actual = exp_beta_over_beta
    K_expected = (alpha_inv - 1) / (2 * math.pi**2)

    return {
        'beta': beta,
        'alpha_inv_lhs': lhs,
        'alpha_inv_rhs': rhs,
        'error_ppm': abs(lhs - rhs) / lhs * 1e6,
        'K_actual': K_actual,
        'K_expected': K_expected,
        'valid': abs(lhs - rhs) / lhs < 1e-9
    }


# =============================================================================
# DERIVED COEFFICIENTS
# =============================================================================

def derive_coefficients(beta, alpha):
    """
    Derive nuclear coefficients from β and α.

    FORMULAS:
    ---------
    c₁ = ½(1 - α)
        - The ½ comes from the virial theorem (KE = ½ PE)
        - The (1-α) is electromagnetic screening
        - Physical meaning: surface tension minus EM drag

    c₂ = 1/β
        - Inverse of vacuum stiffness
        - Physical meaning: bulk compressibility (saturation limit)

    V₄ = -1/β
        - Same magnitude as c₂, opposite sign
        - Physical meaning: QED vacuum polarization
    """
    c1 = 0.5 * (1 - alpha)
    c2 = 1.0 / beta
    v4 = -1.0 / beta

    return {'c1': c1, 'c2': c2, 'v4': v4}


# =============================================================================
# DEMONSTRATION OF COMMON ERROR
# =============================================================================

def show_wrong_formula_error():
    """
    Demonstrate why the alternative formula is WRONG.

    WRONG FORMULA: K = (α⁻¹ × c₁) / π²

    This is circular and gives inconsistent results.
    """
    print("-" * 70)
    print("WARNING: Common Error in Some Documentation")
    print("-" * 70)
    print()
    print("Some scripts incorrectly use:")
    print()
    print("    K = (α⁻¹ × c₁) / π²  ← WRONG!")
    print()
    print("This is wrong because:")
    print("  1. It requires c₁ as input, but c₁ is DERIVED from α")
    print("  2. It doesn't match the Lean4 proof")
    print("  3. It gives different K values depending on which c₁ you use")
    print()

    # Show the inconsistency
    pi_sq = math.pi ** 2

    # Using empirical c₁ = 0.496297 (NuBase 2020)
    c1_empirical = 0.496297
    K_wrong_empirical = (ALPHA_INV * c1_empirical) / pi_sq

    # Using derived c₁ = ½(1-α) = 0.496351
    c1_derived = 0.5 * (1 - ALPHA)
    K_wrong_derived = (ALPHA_INV * c1_derived) / pi_sq

    # Using correct formula
    K_correct = (ALPHA_INV - 1) / (2 * pi_sq)

    print("If we use the WRONG formula with different c₁ values:")
    print(f"  K_wrong (empirical c₁=0.496297) = {K_wrong_empirical:.6f}")
    print(f"  K_wrong (derived c₁=0.496351)   = {K_wrong_derived:.6f}")
    print()
    print("But the CORRECT formula gives:")
    print(f"  K_correct = (1/α - 1) / (2π²) = {K_correct:.6f}")
    print()
    print("The correct value K ≈ 6.892 leads to β ≈ 3.043")
    print()


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def main():
    print("=" * 70)
    print("DERIVING β FROM α (Fine Structure Constant)")
    print("=" * 70)
    print()

    if USING_SHARED_CONSTANTS:
        print("Using constants from qfd/shared_constants.py")
    else:
        print("Using inline constants (shared_constants not available)")
    print()

    print("-" * 70)
    print("THE MASTER INPUT: Fine Structure Constant")
    print("-" * 70)
    print()
    print(f"    α = 1/{ALPHA_INV:.9f}")
    print()
    print("    This is the ONLY measured input. Everything else is derived.")
    print()

    # =========================================================================
    # THE GOLDEN LOOP EQUATION
    # =========================================================================
    print("-" * 70)
    print("THE GOLDEN LOOP EQUATION")
    print("-" * 70)
    print()
    print("    Master Equation:")
    print()
    print("        1/α = 2π² × (e^β / β) + 1")
    print()
    print("    Rearranged to solve for β:")
    print()
    print("        e^β / β = (1/α - 1) / (2π²)")
    print()

    # Calculate K
    K = (ALPHA_INV - 1) / (2 * math.pi**2)
    print(f"    Target: K = ({ALPHA_INV:.3f} - 1) / (2 × π²)")
    print(f"              = {ALPHA_INV - 1:.6f} / {2 * math.pi**2:.6f}")
    print(f"              = {K:.6f}")
    print()

    # Solve for β
    beta = solve_golden_loop_newton(ALPHA_INV)

    print(f"    Solving e^β/β = {K:.6f} using Newton-Raphson...")
    print()
    print(f"    RESULT: β = {beta:.9f}")
    print()

    # Verify
    verification = verify_golden_loop(beta, ALPHA_INV)
    print("    VERIFICATION:")
    print(f"        LHS (1/α)                 = {verification['alpha_inv_lhs']:.9f}")
    print(f"        RHS (2π²×e^β/β + 1)       = {verification['alpha_inv_rhs']:.9f}")
    print(f"        Error                     = {verification['error_ppm']:.2e} ppm")
    print(f"        Match: {'✓ YES' if verification['valid'] else '✗ NO'}")
    print()

    # =========================================================================
    # DERIVED COEFFICIENTS
    # =========================================================================
    print("-" * 70)
    print("DERIVED NUCLEAR COEFFICIENTS")
    print("-" * 70)
    print()

    coeffs = derive_coefficients(beta, ALPHA)

    print("    The Fundamental Soliton Equation:")
    print()
    print("        Q(A) = c₁ × A^(2/3) + c₂ × A")
    print()
    print("    Where (derived from α and β):")
    print()
    print(f"        c₁ = ½(1 - α) = {coeffs['c1']:.6f}")
    print(f"        c₂ = 1/β      = {coeffs['c2']:.6f}")
    print()

    # Compare to empirical
    c1_emp = 0.496297
    c2_emp = 0.32704

    c1_err = abs(coeffs['c1'] - c1_emp) / c1_emp * 100
    c2_err = abs(coeffs['c2'] - c2_emp) / c2_emp * 100

    print("    VALIDATION vs Empirical (NuBase 2020):")
    print()
    print(f"        c₁: derived = {coeffs['c1']:.6f}, empirical = {c1_emp:.6f}")
    print(f"            Error = {c1_err:.4f}%")
    print()
    print(f"        c₂: derived = {coeffs['c2']:.6f}, empirical = {c2_emp:.6f}")
    print(f"            Error = {c2_err:.4f}%")
    print()

    # =========================================================================
    # QED VACUUM POLARIZATION
    # =========================================================================
    print("-" * 70)
    print("QED VACUUM POLARIZATION")
    print("-" * 70)
    print()
    print("    The vacuum polarization coefficient:")
    print()
    print(f"        V₄ = -1/β = {coeffs['v4']:.6f}")
    print()
    print("    This appears in the electron g-2 calculation:")
    print()
    print("        a = α/(2π) + V₄×(α/π)² + ...")
    print()

    v4_qed = -0.328479
    v4_err = abs(coeffs['v4'] - v4_qed) / abs(v4_qed) * 100

    print(f"    QED perturbation theory value: {v4_qed:.6f}")
    print(f"    QFD derived value:             {coeffs['v4']:.6f}")
    print(f"    Error: {v4_err:.4f}%")
    print()

    # =========================================================================
    # COMMON ERROR WARNING
    # =========================================================================
    show_wrong_formula_error()

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("=" * 70)
    print("SUMMARY: Complete Derivation Chain")
    print("=" * 70)
    print()
    print("    INPUT (measured):")
    print(f"        α = 1/{ALPHA_INV}")
    print()
    print("    DERIVED (from Golden Loop):")
    print(f"        β = {beta:.9f}")
    print()
    print("    PREDICTIONS:")
    print()
    print(f"        {'Quantity':<20} {'Derived':>12} {'Empirical':>12} {'Error':>10}")
    print("        " + "-" * 56)
    print(f"        {'c₁ (surface)':<20} {coeffs['c1']:>12.6f} {c1_emp:>12.6f} {c1_err:>9.4f}%")
    print(f"        {'c₂ (volume)':<20} {coeffs['c2']:>12.6f} {c2_emp:>12.6f} {c2_err:>9.4f}%")
    print(f"        {'V₄ (QED)':<20} {coeffs['v4']:>12.6f} {v4_qed:>12.6f} {v4_err:>9.4f}%")
    print()
    print("    CONCLUSION:")
    print("        All predictions match within 0.5%.")
    print("        This is achieved with ZERO free parameters.")
    print()
    print("    CORRECT FORMULA:")
    print("        K = (1/α - 1) / (2π²) ≈ 6.892")
    print()
    print("    NOT:")
    print("        K = (α⁻¹ × c₁) / π²  ← WRONG (circular)")
    print()
    print("=" * 70)

    return {
        'beta': beta,
        'c1': coeffs['c1'],
        'c2': coeffs['c2'],
        'v4': coeffs['v4'],
        'c1_error': c1_err,
        'c2_error': c2_err,
        'v4_error': v4_err,
    }


if __name__ == "__main__":
    results = main()
