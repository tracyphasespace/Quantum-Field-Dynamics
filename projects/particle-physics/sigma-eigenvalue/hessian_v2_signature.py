#!/usr/bin/env python3
"""
hessian_v2_signature.py -- Hessian eigenvalues with Cl(3,3) signature corrections

VERSION 2: Upgrades from the scalar v1 solver in three ways:

1. SIGNATURE CORRECTION: The Cl(3,3) phase space has metric (+,+,+,-,-,-).
   The angular Laplacian on this pseudo-Riemannian manifold differs from the
   standard S⁵ Laplacian. Spacelike angular modes contribute positive
   centrifugal barrier, timelike modes contribute NEGATIVE (anti-centrifugal).
   The effective angular eigenvalue depends on how ℓ distributes across the
   two sectors.

2. GROUND-STATE SUBTRACTION: Subtract the ground-state energy density to
   make the Hessian positive-semidefinite (Vakhitov-Kolokolov criterion).

3. MULTIPLE POTENTIAL FORMS: Test quartic vs sextic Mexican hat to check
   sensitivity.

PHYSICS OF THE (3,3) SIGNATURE:
    In standard d=6 Euclidean space, the angular eigenvalue for ℓ-th harmonic is:
        Λ_ℓ = ℓ(ℓ + d - 2) = ℓ(ℓ + 4)

    In Cl(3,3) with signature (+,+,+,-,-,-), the angular part splits:
        - Spacelike angular momentum ℓ_s on S² (3 positive dimensions)
        - Timelike angular momentum ℓ_t on H² (3 negative dimensions)

    The effective centrifugal term becomes:
        Λ_eff = ℓ_s(ℓ_s + 1) - ℓ_t(ℓ_t + 1) + cross-terms

    For the compression mode (ℓ=0): Λ_eff = 0 (isotropic, no angular dependence)
    For the shear mode (ℓ=2):
        Case A: ℓ_s=2, ℓ_t=0 → purely spacelike shear → Λ = 6
        Case B: ℓ_s=1, ℓ_t=1 → mixed mode → Λ = 2 - 2 = 0 (Goldstone-like)
        Case C: ℓ_s=0, ℓ_t=2 → purely timelike → Λ = -6 (tachyonic)

    The physical shear mode is the stable mixed mode. The key question is
    which combination gives σ = β³/(4π²).

THE 4π² HYPOTHESIS:
    Vol(S³) = 2π². The Cl(3,3) has TWO 3-spheres worth of geometry
    (one for the spacelike sector, one for the timelike sector). The
    angular integration over both gives a factor of (2π²)² / π² = 4π².
    Alternatively, 4π² is the surface area of the Clifford torus T² = S¹ × S¹,
    which is the topology of the internal rotation plane (e₄∧e₅ bivector).

    If the normalization factor comes from this angular integration, then:
        σ = β³ / (angular factor) where angular factor = 4π²

Copyright (c) 2026 Tracy McSheery
Licensed under the MIT License
"""

import sys
import os
import numpy as np
from scipy.linalg import eigh_tridiagonal

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from qfd.shared_constants import ALPHA, BETA, XI_QFD, K_GEOM


# =============================================================================
# PARAMETERS
# =============================================================================

beta = BETA
a = 1.0       # Vortex radius (normalized)
rho_0 = 1.0   # Central density (normalized)
lam = beta     # Quartic coupling


# =============================================================================
# DENSITY PROFILE AND POTENTIALS
# =============================================================================

def hill_density(r):
    """Parabolic Hill vortex density."""
    result = np.where(r <= a, rho_0 * (1.0 - (r / a) ** 2), 0.0)
    return result


def V_dd_quartic(r):
    """V''(ρ) for quartic potential V = -μ²ρ² + λρ⁴.
    V''(ρ) = -2μ² + 12λρ².
    At minimum: μ² = 2λρ₀², so V''(ρ) = 4λρ₀²[-1 + 3(1-r²/a²)²].
    """
    rho = hill_density(r)
    mu_sq = 2.0 * lam * rho_0 ** 2
    return -2.0 * mu_sq + 12.0 * lam * rho ** 2


def V_dd_quartic_shifted(r):
    """V'' with ground-state subtraction (Vakhitov-Kolokolov).

    The ground state energy per mode is E_0 = V''(r=0) = 8λρ₀².
    We subtract a fraction of this to shift the spectrum upward.
    Specifically, we subtract V''(r→∞) = -2μ² = -4λρ₀² (the asymptotic value).
    This makes the potential positive inside the vortex and zero outside.
    """
    return V_dd_quartic(r) - V_dd_quartic(np.array([2.0 * a]))[0]


# =============================================================================
# ANGULAR EIGENVALUES FOR DIFFERENT SIGNATURES
# =============================================================================

def angular_eigenvalue_euclidean(ell, d=6):
    """Standard Euclidean: Λ = ℓ(ℓ + d - 2)."""
    return ell * (ell + d - 2)


def angular_eigenvalue_33_spacelike(ell_s):
    """Pure spacelike mode in (3,3): Λ = ℓ_s(ℓ_s + 1) on S²."""
    return ell_s * (ell_s + 1)


def angular_eigenvalue_33_mixed(ell_s, ell_t):
    """Mixed mode in (3,3): spacelike and timelike angular momentum.

    The effective centrifugal barrier is:
        Λ = ℓ_s(ℓ_s + 1) - ℓ_t(ℓ_t + 1)

    The minus sign comes from the negative metric in the timelike sector.
    For ℓ_s = ℓ_t (equal distribution), Λ = 0 (Goldstone-like).
    """
    return ell_s * (ell_s + 1) - ell_t * (ell_t + 1)


def angular_eigenvalue_33_full(ell_s, ell_t, d_s=3, d_t=3):
    """Full (p,q) signature angular eigenvalue.

    In a (p,q)-signature space, the angular part of the Laplacian
    on the "unit pseudo-sphere" has eigenvalues:
        Λ = ℓ_s(ℓ_s + d_s - 2) + ℓ_t(ℓ_t + d_t - 2) × (-1)

    For (3,3): d_s=3, d_t=3
        Λ = ℓ_s(ℓ_s + 1) - ℓ_t(ℓ_t + 1)
    """
    return ell_s * (ell_s + d_s - 2) - ell_t * (ell_t + d_t - 2)


# =============================================================================
# RADIAL EIGENVALUE SOLVER
# =============================================================================

def solve_radial(centrifugal_eigenvalue, N=500, r_max=3.0, use_shifted=True):
    """Solve the radial eigenvalue problem for a given angular eigenvalue.

    -f''(r) - (d-1)/r f'(r) + [Λ/r² + V''(ρ₀(r))] f(r) = λ f(r)
    """
    d = 6
    dr = r_max / (N + 1)
    r = np.linspace(dr, r_max - dr, N)

    # Centrifugal term
    centrifugal = centrifugal_eigenvalue / r ** 2

    # Potential
    V_eff = centrifugal + (V_dd_quartic_shifted(r) if use_shifted else V_dd_quartic(r))

    # Tridiagonal matrix
    diag = 2.0 / dr ** 2 + V_eff
    coeff_deriv = (d - 1) / (2.0 * r * dr)
    upper = -1.0 / dr ** 2 - coeff_deriv[:-1]
    lower = -1.0 / dr ** 2 + coeff_deriv[1:]

    n_evals = min(10, N - 1)
    eigenvalues = eigh_tridiagonal(diag, upper, eigvals_only=True,
                                   select='i', select_range=(0, n_evals - 1))
    return eigenvalues


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def main():
    W = 76

    print()
    print("=" * W)
    print("  HESSIAN v2: Cl(3,3) Signature-Corrected Eigenvalue Analysis")
    print("=" * W)

    sigma_target = beta ** 3 / (4 * np.pi ** 2)
    ratio_target = beta ** 2 / (4 * np.pi ** 2)

    print(f"\n  TARGETS:")
    print(f"    σ = β³/(4π²) = {sigma_target:.6f}")
    print(f"    σ/β = β²/(4π²) = {ratio_target:.6f}")

    # =====================================================
    # MODEL 1: Standard Euclidean 6D (baseline, v1 result)
    # =====================================================
    print(f"\n{'MODEL 1: EUCLIDEAN 6D (BASELINE)':^{W}}")
    print("-" * W)

    results_eucl = {}
    for ell in [0, 2, 4]:
        Lambda = angular_eigenvalue_euclidean(ell)
        evals = solve_radial(Lambda, use_shifted=True)
        results_eucl[ell] = (Lambda, evals)
        print(f"  ℓ={ell}: Λ={Lambda:5d}, λ₀={evals[0]:12.4f}, λ₁={evals[1]:12.4f}")

    l0 = results_eucl[0][1][0]
    l2 = results_eucl[2][1][0]
    l4 = results_eucl[4][1][0]
    if l0 != 0:
        print(f"\n  λ₂/λ₀ = {l2/l0:.6f}  (target: {ratio_target:.6f}, "
              f"deviation: {abs(l2/l0/ratio_target - 1)*100:.1f}%)")

    # =====================================================
    # MODEL 2: Cl(3,3) pure spacelike modes
    # =====================================================
    print(f"\n{'MODEL 2: Cl(3,3) PURE SPACELIKE':^{W}}")
    print("-" * W)
    print(f"  (Only spacelike angular momentum, ℓ_t = 0)")

    results_space = {}
    for ell_s in [0, 1, 2, 3, 4]:
        Lambda = angular_eigenvalue_33_spacelike(ell_s)
        evals = solve_radial(Lambda, use_shifted=True)
        results_space[ell_s] = (Lambda, evals)
        print(f"  ℓ_s={ell_s}: Λ={Lambda:5d}, λ₀={evals[0]:12.4f}")

    l0_s = results_space[0][1][0]
    l2_s = results_space[2][1][0]
    if l0_s != 0:
        print(f"\n  λ₂/λ₀ = {l2_s/l0_s:.6f}  (target: {ratio_target:.6f}, "
              f"deviation: {abs(l2_s/l0_s/ratio_target - 1)*100:.1f}%)")

    # =====================================================
    # MODEL 3: Cl(3,3) with (3,3) angular eigenvalues
    # =====================================================
    print(f"\n{'MODEL 3: Cl(3,3) FULL (3,3) SIGNATURE':^{W}}")
    print("-" * W)
    print(f"  (Spacelike + timelike angular momentum)")

    # Scan over (ℓ_s, ℓ_t) combinations
    mode_configs = [
        ("compression",  0, 0, "ℓ_s=0, ℓ_t=0"),
        ("shear_pure_s", 2, 0, "ℓ_s=2, ℓ_t=0 (pure spacelike shear)"),
        ("shear_mixed",  1, 1, "ℓ_s=1, ℓ_t=1 (mixed mode)"),
        ("shear_anti",   2, 1, "ℓ_s=2, ℓ_t=1 (asymmetric)"),
        ("torsion_pure", 4, 0, "ℓ_s=4, ℓ_t=0 (pure spacelike torsion)"),
        ("torsion_mix",  2, 2, "ℓ_s=2, ℓ_t=2 (balanced torsion)"),
        ("torsion_asym", 3, 1, "ℓ_s=3, ℓ_t=1 (asymmetric torsion)"),
    ]

    results_33 = {}
    for name, ell_s, ell_t, desc in mode_configs:
        Lambda = angular_eigenvalue_33_full(ell_s, ell_t)
        evals = solve_radial(Lambda, use_shifted=True)
        results_33[name] = (Lambda, evals, ell_s, ell_t)
        print(f"  {desc:40s}: Λ={Lambda:5d}, λ₀={evals[0]:12.4f}")

    # =====================================================
    # KEY RATIO ANALYSIS
    # =====================================================
    print(f"\n{'RATIO ANALYSIS: WHICH MODE GIVES σ/β ≈ β²/(4π²)?':^{W}}")
    print("-" * W)

    l0_ref = results_33["compression"][1][0]
    print(f"\n  Reference (compression, ℓ_s=0,ℓ_t=0): λ₀ = {l0_ref:.4f}")
    print(f"  Target ratio σ/β = β²/(4π²) = {ratio_target:.6f}")
    print()

    best_name = None
    best_dev = float('inf')

    for name, ell_s, ell_t, desc in mode_configs:
        if name == "compression":
            continue
        Lambda, evals, _, _ = results_33[name]
        ratio = evals[0] / l0_ref if l0_ref != 0 else float('nan')
        dev = abs(ratio / ratio_target - 1.0) * 100 if ratio_target != 0 else float('nan')
        marker = " *** BEST ***" if dev < best_dev else ""
        if dev < best_dev:
            best_dev = dev
            best_name = name
        print(f"  {desc:40s}: λ/λ₀_ref = {ratio:8.6f}, "
              f"deviation = {dev:6.2f}%{marker}")

    if best_name:
        _, _, best_ls, best_lt = results_33[best_name]
        print(f"\n  BEST MATCH: {best_name} (ℓ_s={best_ls}, ℓ_t={best_lt})")
        print(f"  Deviation from β²/(4π²): {best_dev:.2f}%")

    # =====================================================
    # SENSITIVITY ANALYSIS
    # =====================================================
    print(f"\n{'SENSITIVITY: GRID RESOLUTION':^{W}}")
    print("-" * W)

    for N in [200, 500, 1000, 2000]:
        # Use best mode if found, otherwise shear_asym
        test_mode = best_name or "shear_asym"
        ls, lt = results_33[test_mode][2], results_33[test_mode][3]
        Lambda_ref = angular_eigenvalue_33_full(0, 0)
        Lambda_test = angular_eigenvalue_33_full(ls, lt)
        ev_ref = solve_radial(Lambda_ref, N=N, use_shifted=True)[0]
        ev_test = solve_radial(Lambda_test, N=N, use_shifted=True)[0]
        ratio = ev_test / ev_ref if ev_ref != 0 else float('nan')
        dev = abs(ratio / ratio_target - 1.0) * 100
        print(f"  N={N:5d}: ratio = {ratio:.6f}, deviation = {dev:.2f}%")

    # =====================================================
    # SELF-TESTS
    # =====================================================
    print(f"\n{'SELF-TESTS':^{W}}")
    print("-" * W)

    tests = []

    # T1: Shifted eigenvalues should be positive for compression
    ev0_shifted = solve_radial(0, use_shifted=True)[0]
    tests.append(("Shifted compression eigenvalue positive",
                  ev0_shifted > 0, f"λ₀ = {ev0_shifted:.4f}"))

    # T2: Pure shear > compression (centrifugal ordering)
    l_comp = results_33["compression"][1][0]
    l_shear = results_33["shear_pure_s"][1][0]
    tests.append(("Shear > Compression (centrifugal)",
                  l_shear > l_comp, f"{l_shear:.2f} > {l_comp:.2f}"))

    # T3: Ratio is in [0.1, 0.4] (right order of magnitude)
    if l0_ref != 0 and best_name:
        best_ratio = results_33[best_name][1][0] / l0_ref
        tests.append(("Ratio in [0.1, 0.4]",
                      0.1 < best_ratio < 0.4,
                      f"ratio = {best_ratio:.4f}"))

    # T4: Eigenvalues converge with grid refinement
    ev_500 = solve_radial(0, N=500, use_shifted=True)[0]
    ev_1000 = solve_radial(0, N=1000, use_shifted=True)[0]
    conv = abs(ev_500 / ev_1000 - 1.0) if ev_1000 != 0 else 1.0
    tests.append(("Grid convergence < 1%",
                  conv < 0.01, f"N500/N1000 - 1 = {conv:.4e}"))

    all_pass = True
    for name, ok, detail in tests:
        status = 'PASS' if ok else 'FAIL'
        print(f"  [{status}] {name}: {detail}")
        if not ok:
            all_pass = False

    print(f"\n  Overall: {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")

    # =====================================================
    # SUMMARY
    # =====================================================
    print(f"\n{'=' * W}")
    print("  SUMMARY")
    print(f"{'=' * W}")
    print(f"  This v2 solver tests three models:")
    print(f"  1. Euclidean 6D (baseline, reproduces v1 result)")
    print(f"  2. Cl(3,3) pure spacelike modes")
    print(f"  3. Cl(3,3) full (3,3) signature with mixed modes")
    print(f"")
    print(f"  The (ℓ_s, ℓ_t) decomposition tests which angular momentum")
    print(f"  distribution gives σ/β closest to β²/(4π²) = {ratio_target:.6f}")
    print(f"")
    print(f"  The 4π² denominator should emerge from the angular integration")
    print(f"  over the Cl(3,3) phase space — either as Vol(S³)×2 = 4π² or as")
    print(f"  the surface area of the Clifford torus T² = S¹ × S¹.")
    print(f"{'=' * W}")


if __name__ == '__main__':
    main()
