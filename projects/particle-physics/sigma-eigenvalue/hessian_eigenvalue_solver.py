#!/usr/bin/env python3
"""
hessian_eigenvalue_solver.py -- Compute Hessian eigenvalues of the Hill vortex

PURPOSE: Test whether the shear modulus σ = β³/(4π²) emerges as the ℓ=2
eigenvalue of the energy Hessian evaluated at the Hill vortex ground state.

PHYSICS:
    The Hill vortex in the QFD vacuum has energy functional:
        E[ψ] = ∫ [½|∇ψ|² + V(ρ)] d⁶x
    where V(ρ) = -μ²ρ + λρ² is the Mexican-hat potential with λ = β.

    Second variation (Hessian) around ground state ψ₀:
        L[ψ₀] η = -∇² η + V''(ρ₀(r)) η

    Decompose into hyperspherical harmonics (6D):
        η(r,Ω) = Σ f_ℓ(r) Y_ℓm(Ω)

    Radial Sturm-Liouville problem for each ℓ:
        -f''(r) - (d-1)/r f'(r) + [ℓ(ℓ+d-2)/r² + V''(ρ₀(r))] f(r) = λ f(r)

    where d = 6 (6D phase space), so the centrifugal term is ℓ(ℓ+4)/r².

    KEY IDENTIFICATION:
        ℓ = 0 → isotropic compression → V₄ (bulk modulus, should give β)
        ℓ = 2 → shape distortion → V₆ (shear modulus, should give σ)
        ℓ = 4 → higher torsion → V₈ (torsional stiffness, should give δₛ)

    If σ = β³/(4π²) emerges from the ℓ=2 eigenvalue, the constitutive
    postulate in Appendix V becomes a derived result.

REFERENCE:
    - Book v8.9, Appendix V (σ = β³/(4π²) constitutive postulate)
    - Book v8.9, Appendix Z.4 (spectral gap theorem, Hessian)
    - Book v8.9, Appendix Z.8 (soliton spectrum)
    - Book v8.9, Appendix Z.12 (Hill vortex eigenvalue k_geom)
    - RED_TEAM_ROADMAP.md, Gap 3

Copyright (c) 2026 Tracy McSheery
Licensed under the MIT License
"""

import sys
import os
import numpy as np
from scipy.integrate import solve_bvp
from scipy.linalg import eigh_tridiagonal

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from qfd.shared_constants import ALPHA, BETA, XI_QFD, K_GEOM


# =============================================================================
# PHYSICAL PARAMETERS
# =============================================================================

# Dimensionless vacuum stiffness (from Golden Loop)
beta = BETA  # ≈ 3.043233053

# Mexican-hat potential V(ρ) = -μ²ρ + λρ²
# At the ground state: V'(ρ₀) = 0 → ρ₀ = μ²/(2λ)
# Second derivative: V''(ρ₀) = 2λ = 2β (at the minimum)
# Away from minimum: V''(ρ) = -μ² + 6λρ² (for V = -μ²ρ² + λρ⁴ quartic)
#
# For the Hill vortex, the density profile is parabolic:
# ρ(r) = ρ₀(1 - r²/a²) for r ≤ a, 0 outside
# so V''(ρ(r)) varies with position.

# Vortex radius (normalized to 1)
a = 1.0

# Ground state central density (normalized)
rho_0 = 1.0

# Quartic coupling (identified with β)
lam = beta


# =============================================================================
# HILL VORTEX DENSITY PROFILE
# =============================================================================

def hill_density(r):
    """Parabolic density profile of the Hill spherical vortex.

    ρ(r) = ρ₀(1 - r²/a²) inside, 0 outside.
    """
    if np.isscalar(r):
        if r <= a:
            return rho_0 * (1.0 - (r / a) ** 2)
        return 0.0
    result = np.zeros_like(r)
    inside = r <= a
    result[inside] = rho_0 * (1.0 - (r[inside] / a) ** 2)
    return result


def V_second_derivative(r):
    """Second derivative of the Mexican-hat potential at the Hill vortex profile.

    For V(ρ) = -μ²ρ² + λρ⁴  (quartic Mexican hat):
        V'(ρ) = -2μ²ρ + 4λρ³
        V''(ρ) = -2μ² + 12λρ²

    At the minimum ρ₀: V''(ρ₀) = -2μ² + 12λρ₀² = 2λ(6ρ₀² - μ²/λ)
    With μ² = 2λρ₀² (minimum condition for V' = 0):
        V''(ρ₀) = 2λ(6ρ₀² - 2ρ₀²) = 8λρ₀² = 8β (at center)

    For the parabolic profile ρ(r) = ρ₀(1 - r²/a²):
        V''(r) = -2μ² + 12λ[ρ₀(1 - r²/a²)]²
               = -4λρ₀² + 12λρ₀²(1 - r²/a²)²
               = 4λρ₀²[-1 + 3(1 - r²/a²)²]
    """
    rho = hill_density(r)
    mu_sq = 2.0 * lam * rho_0 ** 2  # From minimum condition
    return -2.0 * mu_sq + 12.0 * lam * rho ** 2


# =============================================================================
# RADIAL STURM-LIOUVILLE SOLVER
# =============================================================================

def radial_eigenvalues(ell, N=500, r_max=3.0):
    """Solve the radial eigenvalue problem for angular quantum number ℓ.

    -f''(r) - (d-1)/r f'(r) + V_eff(r) f(r) = λ f(r)

    where d = 6, V_eff(r) = ℓ(ℓ+d-2)/r² + V''(ρ₀(r))

    Uses finite-difference discretization on uniform grid.

    Parameters
    ----------
    ell : int
        Angular quantum number (0, 2, 4, ...)
    N : int
        Number of grid points
    r_max : float
        Outer boundary (should be >> a)

    Returns
    -------
    eigenvalues : array
        Lowest few eigenvalues of the radial Hessian
    """
    d = 6  # Phase space dimension
    dr = r_max / (N + 1)
    r = np.linspace(dr, r_max - dr, N)  # Exclude r=0 and r=r_max

    # Centrifugal term: ℓ(ℓ + d - 2) / r²
    centrifugal = ell * (ell + d - 2) / r ** 2

    # Effective potential on the diagonal
    V_eff = centrifugal + V_second_derivative(r)

    # Finite-difference Laplacian in d dimensions:
    # -f'' - (d-1)/r f' ≈ -[f_{i+1} - 2f_i + f_{i-1}]/dr²
    #                       - (d-1)/(r_i) × [f_{i+1} - f_{i-1}]/(2dr)
    #
    # This gives a tridiagonal matrix:
    # diagonal: 2/dr² + V_eff
    # upper:    -1/dr² - (d-1)/(2r_i dr)
    # lower:    -1/dr² + (d-1)/(2r_i dr)

    diag = 2.0 / dr ** 2 + V_eff

    # Off-diagonal terms
    coeff_deriv = (d - 1) / (2.0 * r * dr)
    upper = -1.0 / dr ** 2 - coeff_deriv[:-1]  # f_{i+1} coefficient
    lower = -1.0 / dr ** 2 + coeff_deriv[1:]    # f_{i-1} coefficient

    # Solve tridiagonal eigenvalue problem
    eigenvalues = eigh_tridiagonal(diag, upper, eigvals_only=True,
                                   select='i', select_range=(0, min(9, N - 1)))

    return eigenvalues


# =============================================================================
# EFFECTIVE ELASTIC MODULI FROM EIGENVALUES
# =============================================================================

def extract_elastic_moduli():
    """Compute eigenvalues for ℓ = 0, 2, 4 and extract elastic moduli.

    The mapping:
        ℓ = 0 lowest eigenvalue → proportional to V₄ (bulk modulus)
        ℓ = 2 lowest eigenvalue → proportional to V₆ (shear modulus)
        ℓ = 4 lowest eigenvalue → proportional to V₈ (torsional stiffness)
    """
    results = {}

    for ell in [0, 2, 4]:
        evals = radial_eigenvalues(ell)
        results[ell] = evals

    return results


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def main():
    W = 72

    print()
    print("=" * W)
    print("  HESSIAN EIGENVALUE ANALYSIS: Hill Vortex in QFD Vacuum")
    print("=" * W)

    print(f"\n  INPUT PARAMETERS:")
    print(f"    β (vacuum stiffness)    = {beta:.10f}")
    print(f"    λ (quartic coupling)    = {lam:.10f}")
    print(f"    a (vortex radius)       = {a:.4f}")
    print(f"    ρ₀ (central density)    = {rho_0:.4f}")

    print(f"\n  CONSTITUTIVE POSTULATE (to be tested):")
    sigma_postulate = beta ** 3 / (4 * np.pi ** 2)
    print(f"    σ = β³/(4π²) = {sigma_postulate:.6f}")

    print(f"\n  POTENTIAL CHECK:")
    print(f"    V''(r=0)   = {V_second_derivative(0.0):.4f}  (center)")
    print(f"    V''(r=a/2) = {V_second_derivative(a / 2):.4f}  (midpoint)")
    print(f"    V''(r=a)   = {V_second_derivative(a):.4f}  (surface)")
    print(f"    V''(r=2a)  = {V_second_derivative(2 * a):.4f}  (exterior)")

    # Compute eigenvalues
    print(f"\n{'EIGENVALUE COMPUTATION':^{W}}")
    print("-" * W)

    results = extract_elastic_moduli()

    mode_names = {0: "Compression (V₄/bulk)", 2: "Shear (V₆/σ)", 4: "Torsion (V₈/δₛ)"}

    for ell in [0, 2, 4]:
        evals = results[ell]
        print(f"\n  ℓ = {ell} — {mode_names[ell]}:")
        print(f"    Lowest eigenvalues:")
        for i, ev in enumerate(evals[:5]):
            print(f"      λ_{i} = {ev:12.6f}")

    # Extract the key ratios
    print(f"\n{'RATIO ANALYSIS':^{W}}")
    print("-" * W)

    lambda_0 = results[0][0]  # Bulk mode
    lambda_2 = results[2][0]  # Shear mode
    lambda_4 = results[4][0]  # Torsion mode

    print(f"\n  Lowest eigenvalues:")
    print(f"    λ₀ (ℓ=0, bulk)    = {lambda_0:.6f}")
    print(f"    λ₂ (ℓ=2, shear)   = {lambda_2:.6f}")
    print(f"    λ₄ (ℓ=4, torsion)  = {lambda_4:.6f}")

    print(f"\n  Ratios:")
    if lambda_0 != 0:
        print(f"    λ₂/λ₀ = {lambda_2 / lambda_0:.6f}")
        print(f"    λ₄/λ₀ = {lambda_4 / lambda_0:.6f}")
    if lambda_2 != 0:
        print(f"    λ₄/λ₂ = {lambda_4 / lambda_2:.6f}")

    print(f"\n  Comparison with constitutive postulate:")
    print(f"    σ_postulate = β³/(4π²)           = {sigma_postulate:.6f}")
    print(f"    β (bulk reference)                = {beta:.6f}")
    print(f"    σ_postulate/β                     = {sigma_postulate / beta:.6f}")
    print(f"    β²/(4π²) (= σ/β)                 = {beta ** 2 / (4 * np.pi ** 2):.6f}")

    # The key test: does λ₂/λ₀ ≈ σ/β = β²/(4π²)?
    if lambda_0 > 0:
        ratio_computed = lambda_2 / lambda_0
        ratio_postulate = beta ** 2 / (4 * np.pi ** 2)
        deviation = abs(ratio_computed / ratio_postulate - 1.0) * 100
        print(f"\n  KEY TEST:")
        print(f"    λ₂/λ₀ (computed)  = {ratio_computed:.6f}")
        print(f"    β²/(4π²) (target) = {ratio_postulate:.6f}")
        print(f"    Deviation         = {deviation:.2f}%")

        if deviation < 5:
            print(f"\n  *** σ = β³/(4π²) is CONSISTENT with the Hessian eigenvalue ***")
        elif deviation < 20:
            print(f"\n  ** σ = β³/(4π²) is APPROXIMATELY consistent (within {deviation:.0f}%) **")
        else:
            print(f"\n  !! σ = β³/(4π²) does NOT match the Hessian eigenvalue !!")
            print(f"  !! The constitutive postulate may need revision !!")

    # Self-tests
    print(f"\n{'SELF-TESTS':^{W}}")
    print("-" * W)

    tests = []

    # T1: Eigenvalues should be real
    all_real = all(np.all(np.isreal(results[ell][:5])) for ell in [0, 2, 4])
    tests.append(("All eigenvalues real", all_real, "Hessian is self-adjoint"))

    # T2: ℓ=2 eigenvalue should be > ℓ=0 (centrifugal barrier raises energy)
    ok = lambda_2 > lambda_0
    tests.append(("λ₂ > λ₀ (centrifugal ordering)", ok,
                  f"{lambda_2:.4f} > {lambda_0:.4f}"))

    # T3: ℓ=4 eigenvalue should be > ℓ=2
    ok = lambda_4 > lambda_2
    tests.append(("λ₄ > λ₂ (centrifugal ordering)", ok,
                  f"{lambda_4:.4f} > {lambda_2:.4f}"))

    # T4: Bulk eigenvalue should be positive (stable ground state)
    ok = lambda_0 > 0
    tests.append(("λ₀ > 0 (stable ground state)", ok, f"λ₀ = {lambda_0:.4f}"))

    all_pass = True
    for name, ok, detail in tests:
        status = 'PASS' if ok else 'FAIL'
        print(f"  [{status}] {name}: {detail}")
        if not ok:
            all_pass = False

    print(f"\n  Overall: {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")

    # Summary
    print(f"\n{'=' * W}")
    print("  SUMMARY")
    print(f"{'=' * W}")
    print(f"  This solver computes the Hessian eigenvalues of the Hill vortex")
    print(f"  energy functional in 6D, decomposed by angular quantum number ℓ.")
    print(f"  The ℓ=2 eigenvalue corresponds to the shear mode (V₆).")
    print(f"")
    print(f"  CAVEAT: This is a simplified 1D radial computation with a")
    print(f"  parabolic density profile. The full 6D Cl(3,3) computation")
    print(f"  requires the multivector structure of the Hill vortex, which")
    print(f"  may modify the angular mode coupling. This solver provides")
    print(f"  the numerical scaffolding to test the postulate; the full")
    print(f"  derivation requires the algebra outlined in RED_TEAM_ROADMAP.md.")
    print(f"{'=' * W}")


if __name__ == '__main__':
    main()
