#!/usr/bin/env python3
"""
hessian_v4b_selfadjoint.py -- Self-adjoint form of the radial Hessian

CRITICAL OBSERVATION: All previous solvers (v1-v3) used eigh_tridiagonal()
on an ASYMMETRIC tridiagonal matrix. The (d-1)/r first-derivative term
makes upper ≠ lower diagonals:
    upper[i] = -1/dr² - (d-1)/(2r_i dr)
    lower[i] = -1/dr² + (d-1)/(2r_i dr)

eigh_tridiagonal() assumes symmetry (lower = upper), which is WRONG.
The absolute eigenvalues were nonsensical (large negative), though the
RATIO might have been partially protected by error cancellation.

FIX: Use the standard self-adjoint substitution u(r) = r^((d-1)/2) f(r).
The equation becomes:

    -u'' + V_eff_SA(r) u = λ u

where:
    V_eff_SA(r) = [Λ + (d-1)(d-3)/4] / r² + V''(ρ₀(r))

This IS symmetric and eigh_tridiagonal is exactly correct.

For d=6: the extra centrifugal term is (d-1)(d-3)/4 = 15/4 = 3.75
    Compression (Λ=0): V_cent = 3.75/r²
    Shear (3,1) (Λ=10): V_cent = 13.75/r²

KEY QUESTION: Does the self-adjoint form change the eigenvalue RATIO?
If yes, the 2.83% residual was a numerical artifact of the wrong solver.
If no, the 2.83% is genuine and the (d-1)/r error cancels in ratios.

Copyright (c) 2026 Tracy McSheery
Licensed under the MIT License
"""

import sys
import os
import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.optimize import brentq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from qfd.shared_constants import ALPHA, BETA


# =============================================================================
# PARAMETERS
# =============================================================================

beta = BETA
lam = beta
rho_0 = 1.0
a = 1.0


# =============================================================================
# PHYSICS
# =============================================================================

def parabolic_profile(r):
    return np.where(r <= a, rho_0 * (1.0 - (r/a)**2), 0.0)

def Vpp_quartic(rho):
    mu_sq = 2.0 * lam * rho_0**2
    return -2.0 * mu_sq + 12.0 * lam * rho**2


# =============================================================================
# SELF-ADJOINT SOLVER
# =============================================================================

def solve_selfadjoint(Lambda, d_eff=6.0, N=1000, r_max=3.0):
    """Solve the self-adjoint form of the radial eigenvalue problem.

    After substitution u(r) = r^((d-1)/2) f(r):
        -u'' + [(Λ + (d-1)(d-3)/4)/r² + V''(ρ₀(r))] u = λ u

    This is a SYMMETRIC tridiagonal eigenvalue problem where
    eigh_tridiagonal is exactly correct.
    """
    dr = r_max / (N + 1)
    r = np.linspace(dr, r_max - dr, N)

    rho = parabolic_profile(r)
    V_dd = Vpp_quartic(rho)

    # Self-adjoint centrifugal term includes the coordinate-change correction
    sa_correction = (d_eff - 1) * (d_eff - 3) / 4.0
    centrifugal = (Lambda + sa_correction) / r**2

    V_eff = centrifugal + V_dd

    # SYMMETRIC tridiagonal: -u'' + V_eff u = λ u
    # Only one off-diagonal (symmetric): -1/dr²
    diag = 2.0 / dr**2 + V_eff
    offdiag = np.full(N - 1, -1.0 / dr**2)

    evals = eigh_tridiagonal(diag, offdiag, eigvals_only=True,
                              select='i', select_range=(0, min(9, N-1)))
    return evals


def solve_asymmetric(Lambda, d_eff=6.0, N=1000, r_max=3.0):
    """Original (WRONG) asymmetric solver for comparison.

    Uses eigh_tridiagonal with upper diagonal only, ignoring the
    asymmetry from the (d-1)/r first-derivative term.
    """
    dr = r_max / (N + 1)
    r = np.linspace(dr, r_max - dr, N)

    rho = parabolic_profile(r)
    V_dd = Vpp_quartic(rho)

    centrifugal = Lambda / r**2
    V_eff = centrifugal + V_dd

    diag = 2.0 / dr**2 + V_eff
    coeff_deriv = (d_eff - 1) / (2.0 * r * dr)
    upper = -1.0 / dr**2 - coeff_deriv[:-1]  # upper ≠ lower!

    evals = eigh_tridiagonal(diag, upper, eigvals_only=True,
                              select='i', select_range=(0, min(9, N-1)))
    return evals


def angular_eigenvalue_33(ell_s, ell_t):
    return ell_s * (ell_s + 1) - ell_t * (ell_t + 1)


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def main():
    W = 78
    ratio_target = beta**2 / (4 * np.pi**2)

    print()
    print("=" * W)
    print("  HESSIAN v4b: Self-Adjoint vs Asymmetric Comparison")
    print("=" * W)
    print(f"  β = {beta:.10f}")
    print(f"  Target ratio σ/β = β²/(4π²) = {ratio_target:.6f}")
    print(f"  Self-adjoint correction: (d-1)(d-3)/4 = {5*3/4.0} for d=6")

    # ===== Compare eigenvalues =====
    print(f"\n{'EIGENVALUE COMPARISON: SELF-ADJOINT vs ASYMMETRIC':^{W}}")
    print("-" * W)

    modes = [
        ("Compression (0,0)", 0, 0),
        ("Shear (3,1)",       3, 1),
        ("Shear (2,0)",       2, 0),
        ("Torsion (4,0)",     4, 0),
    ]

    results_sa = {}
    results_asym = {}

    for name, ls, lt in modes:
        Lambda = angular_eigenvalue_33(ls, lt)
        ev_sa = solve_selfadjoint(Lambda, N=2000)
        ev_asym = solve_asymmetric(Lambda, N=2000)

        results_sa[(ls,lt)] = ev_sa
        results_asym[(ls,lt)] = ev_asym

        print(f"\n  {name} (Λ={Lambda}):")
        print(f"    Self-adjoint λ₀ = {ev_sa[0]:12.4f}  λ₁ = {ev_sa[1]:12.4f}")
        print(f"    Asymmetric   λ₀ = {ev_asym[0]:12.4f}  λ₁ = {ev_asym[1]:12.4f}")
        print(f"    SA positive? {ev_sa[0] > 0}")

    # ===== Ratio comparison =====
    print(f"\n{'RATIO COMPARISON':^{W}}")
    print("-" * W)

    ev_comp_sa = results_sa[(0,0)][0]
    ev_comp_asym = results_asym[(0,0)][0]

    for name, ls, lt in modes[1:]:
        Lambda = angular_eigenvalue_33(ls, lt)
        ev_sa = results_sa[(ls,lt)][0]
        ev_asym = results_asym[(ls,lt)][0]

        ratio_sa = ev_sa / ev_comp_sa if ev_comp_sa != 0 else float('nan')
        ratio_asym = ev_asym / ev_comp_asym if ev_comp_asym != 0 else float('nan')
        dev_sa = abs(ratio_sa / ratio_target - 1) * 100
        dev_asym = abs(ratio_asym / ratio_target - 1) * 100

        print(f"  {name:20s}: SA ratio = {ratio_sa:.6f} ({dev_sa:.2f}%)  "
              f"Asym ratio = {ratio_asym:.6f} ({dev_asym:.2f}%)")

    # ===== Full mode scan with self-adjoint =====
    print(f"\n{'SELF-ADJOINT MODE SCAN (N=2000)':^{W}}")
    print("-" * W)

    all_modes = [
        (1, 0), (2, 0), (3, 0), (4, 0),
        (1, 1), (2, 1), (3, 1), (4, 1),
        (2, 2), (3, 2), (4, 2),
        (3, 3), (4, 3),
    ]

    for ls, lt in all_modes:
        Lambda = angular_eigenvalue_33(ls, lt)
        ev = solve_selfadjoint(Lambda, N=2000)
        ratio = ev[0] / ev_comp_sa if ev_comp_sa != 0 else float('nan')
        dev = abs(ratio / ratio_target - 1) * 100
        marker = " ***" if dev < 5 else ""
        print(f"  (ℓ_s={ls}, ℓ_t={lt}): Λ={Lambda:5d}, "
              f"ratio = {ratio:10.6f}, dev = {dev:6.2f}%{marker}")

    # ===== Grid convergence for self-adjoint =====
    print(f"\n{'SELF-ADJOINT GRID CONVERGENCE':^{W}}")
    print("-" * W)

    Lambda_31 = angular_eigenvalue_33(3, 1)
    for N in [200, 500, 1000, 2000, 4000]:
        ev_c = solve_selfadjoint(0, N=N)[0]
        ev_s = solve_selfadjoint(Lambda_31, N=N)[0]
        ratio = ev_s / ev_c if ev_c != 0 else float('nan')
        dev = abs(ratio / ratio_target - 1) * 100
        print(f"  N={N:5d}: ratio = {ratio:.6f}, dev = {dev:.2f}%  "
              f"(comp={ev_c:.2f}, shear={ev_s:.2f})")

    # ===== d_eff scan with self-adjoint =====
    print(f"\n{'SELF-ADJOINT: d_eff SCAN':^{W}}")
    print("-" * W)

    for d_eff in [4.0, 5.0, 5.5, 6.0, 6.5, 7.0, 8.0]:
        ev_c = solve_selfadjoint(0, d_eff=d_eff, N=2000)[0]
        ev_s = solve_selfadjoint(Lambda_31, d_eff=d_eff, N=2000)[0]
        ratio = ev_s / ev_c if ev_c != 0 else float('nan')
        dev = abs(ratio / ratio_target - 1) * 100
        print(f"  d_eff={d_eff:.1f}: ratio = {ratio:.6f}, dev = {dev:.2f}%")

    # ===== Find exact d_eff (self-adjoint) =====
    print(f"\n{'FIND EXACT d_eff (SELF-ADJOINT)':^{W}}")
    print("-" * W)

    def ratio_minus_target_sa(d_eff):
        ev_c = solve_selfadjoint(0, d_eff=d_eff, N=2000)[0]
        ev_s = solve_selfadjoint(Lambda_31, d_eff=d_eff, N=2000)[0]
        return ev_s / ev_c - ratio_target

    try:
        d_exact_sa = brentq(ratio_minus_target_sa, 5.0, 8.0)
        print(f"  Exact d_eff (self-adjoint): {d_exact_sa:.6f}")
        print(f"  (d_eff - 1) = {d_exact_sa - 1:.6f}")
        print(f"  For comparison, asymmetric gave d_eff = 6.1647")

        # Check nice numbers
        print(f"\n  Nearby nice numbers:")
        candidates = {
            "6":           6.0,
            "6 + 1/6":     6.0 + 1.0/6,
            "6 + α":       6.0 + ALPHA,
            "6 + 1/(2π)":  6.0 + 1/(2*np.pi),
            "6 + β/(4π²)": 6.0 + beta/(4*np.pi**2),
            "6 + 1/β":     6.0 + 1/beta,
            "37/6":        37.0/6,
        }
        for desc, val in sorted(candidates.items(),
                                 key=lambda x: abs(x[1] - d_exact_sa)):
            diff = val - d_exact_sa
            print(f"    {desc:20s} = {val:.6f}  diff = {diff:+.6f}")
    except ValueError:
        print(f"  No root found in [5, 8]")

    # ===== Find exact Λ (self-adjoint) =====
    print(f"\n{'FIND EXACT Λ (SELF-ADJOINT, d_eff=6)':^{W}}")
    print("-" * W)

    def ratio_minus_target_Lambda(Lambda):
        ev_c = solve_selfadjoint(0, d_eff=6.0, N=2000)[0]
        ev_s = solve_selfadjoint(Lambda, d_eff=6.0, N=2000)[0]
        return ev_s / ev_c - ratio_target

    Lambda_exact = brentq(ratio_minus_target_Lambda, 5.0, 15.0)
    delta_Lambda = Lambda_exact - 10.0
    print(f"  Exact Λ: {Lambda_exact:.6f}")
    print(f"  δΛ from 10: {delta_Lambda:+.6f}")
    print(f"  δΛ/10: {delta_Lambda/10*100:+.3f}%")

    # Summary
    print(f"\n{'=' * W}")
    print(f"  SUMMARY")
    print(f"{'=' * W}")
    print(f"  The self-adjoint form correctly handles the (d-1)/r term.")
    print(f"  If SA and asymmetric give DIFFERENT ratios, the 2.83% was")
    print(f"  partly a solver artifact. If SAME, the residual is genuine.")
    print(f"{'=' * W}")


if __name__ == '__main__':
    main()
