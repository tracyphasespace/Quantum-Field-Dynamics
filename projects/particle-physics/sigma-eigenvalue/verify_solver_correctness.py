#!/usr/bin/env python3
"""
verify_solver_correctness.py -- Verify which eigenvalues are correct

Three methods to compute eigenvalues of the SAME physical operator:

1. ASYMMETRIC (v1-v3): eigh_tridiagonal on non-symmetric matrix (SUSPECTED BUG)
2. SELF-ADJOINT (v4b): eigh_tridiagonal on properly symmetric matrix
3. GENERAL: scipy.linalg.eigvals on the full asymmetric matrix (ground truth)

If method 3 agrees with method 2, the 2.83% was an artifact.
If method 3 agrees with method 1, the self-adjoint transform has an error.

Copyright (c) 2026 Tracy McSheery
Licensed under the MIT License
"""

import sys
import os
import numpy as np
from scipy.linalg import eigh_tridiagonal, eigvalsh

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from qfd.shared_constants import BETA

beta = BETA
lam = beta
rho_0 = 1.0
a = 1.0


def parabolic_profile(r):
    return np.where(r <= a, rho_0 * (1.0 - (r/a)**2), 0.0)

def Vpp_quartic(rho):
    mu_sq = 2.0 * lam * rho_0**2
    return -2.0 * mu_sq + 12.0 * lam * rho**2

def angular_eigenvalue_33(ell_s, ell_t):
    return ell_s * (ell_s + 1) - ell_t * (ell_t + 1)


def method_1_asymmetric(Lambda, N, r_max=3.0, d_eff=6.0):
    """v1-v3 method: eigh_tridiagonal with upper diagonal only."""
    dr = r_max / (N + 1)
    r = np.linspace(dr, r_max - dr, N)
    rho = parabolic_profile(r)
    V_dd = Vpp_quartic(rho)
    V_eff = Lambda / r**2 + V_dd

    diag = 2.0 / dr**2 + V_eff
    coeff_deriv = (d_eff - 1) / (2.0 * r * dr)
    upper = -1.0 / dr**2 - coeff_deriv[:-1]

    evals = eigh_tridiagonal(diag, upper, eigvals_only=True,
                              select='i', select_range=(0, 2))
    return evals


def method_2_selfadjoint(Lambda, N, r_max=3.0, d_eff=6.0):
    """v4b method: eigh_tridiagonal on properly symmetric matrix."""
    dr = r_max / (N + 1)
    r = np.linspace(dr, r_max - dr, N)
    rho = parabolic_profile(r)
    V_dd = Vpp_quartic(rho)

    sa_correction = (d_eff - 1) * (d_eff - 3) / 4.0
    V_eff = (Lambda + sa_correction) / r**2 + V_dd

    diag = 2.0 / dr**2 + V_eff
    offdiag = np.full(N - 1, -1.0 / dr**2)

    evals = eigh_tridiagonal(diag, offdiag, eigvals_only=True,
                              select='i', select_range=(0, 2))
    return evals


def method_3_general(Lambda, N, r_max=3.0, d_eff=6.0):
    """Ground truth: build full asymmetric matrix, use general eigensolver."""
    dr = r_max / (N + 1)
    r = np.linspace(dr, r_max - dr, N)
    rho = parabolic_profile(r)
    V_dd = Vpp_quartic(rho)
    V_eff = Lambda / r**2 + V_dd

    coeff_deriv = (d_eff - 1) / (2.0 * r * dr)

    # Build full N×N matrix
    H = np.zeros((N, N))
    for i in range(N):
        H[i, i] = 2.0 / dr**2 + V_eff[i]
    for i in range(N - 1):
        H[i, i+1] = -1.0 / dr**2 - coeff_deriv[i]      # upper
        H[i+1, i] = -1.0 / dr**2 + coeff_deriv[i+1]    # lower (DIFFERENT!)

    # General eigenvalue solver (handles non-symmetric matrices)
    all_evals = np.linalg.eigvals(H)

    # For non-symmetric matrices, eigenvalues might be complex
    # Sort by real part
    all_evals = np.sort(np.real(all_evals))
    return all_evals[:3]


def method_4_symmetrized(Lambda, N, r_max=3.0, d_eff=6.0):
    """Symmetrize the matrix using the weight function w(r) = r^(d-1).

    The operator L is self-adjoint in L²(r^{d-1} dr):
        ⟨f, Lg⟩_w = ⟨Lf, g⟩_w

    The weighted matrix H_w[i,j] = w_i^{1/2} H[i,j] w_j^{-1/2}
    is symmetric. This is the correct way to handle the inner product.
    """
    dr = r_max / (N + 1)
    r = np.linspace(dr, r_max - dr, N)
    rho = parabolic_profile(r)
    V_dd = Vpp_quartic(rho)
    V_eff = Lambda / r**2 + V_dd

    coeff_deriv = (d_eff - 1) / (2.0 * r * dr)

    # Weight function w(r) = r^(d-1)
    w = r ** (d_eff - 1)
    sqrt_w = np.sqrt(w)

    # Build the weighted-symmetric matrix
    # H_w = D^{1/2} H D^{-1/2} where D = diag(w)
    diag_vals = 2.0 / dr**2 + V_eff  # Same diagonal

    # Off-diagonals: upper[i] → upper[i] * sqrt(w[i]) / sqrt(w[i+1])
    #                lower[i] → lower[i] * sqrt(w[i+1]) / sqrt(w[i])
    upper_orig = -1.0 / dr**2 - coeff_deriv[:-1]
    lower_orig = -1.0 / dr**2 + coeff_deriv[1:]

    # Geometric mean of upper and lower gives the symmetric version
    # because H_w upper = upper * sqrt(w[i]/w[i+1])
    #          H_w lower = lower * sqrt(w[i+1]/w[i])
    # For symmetric: H_w[i,i+1] = H_w[i+1,i] = sqrt(upper * lower)

    # Actually, the similarity transform gives:
    # H_w[i,i+1] = H[i,i+1] * sqrt(w[i]) / sqrt(w[i+1])
    # H_w[i+1,i] = H[i+1,i] * sqrt(w[i+1]) / sqrt(w[i])
    # For this to be symmetric: H[i,i+1]*sqrt(w[i]/w[i+1]) = H[i+1,i]*sqrt(w[i+1]/w[i])
    # i.e. H[i,i+1] * w[i] = H[i+1,i] * w[i+1]
    # This is true iff the operator is self-adjoint w.r.t. w(r) dr.

    # Check: H[i,i+1] * w[i] = (-1/dr² - (d-1)/(2r_i dr)) * r_i^(d-1)
    # H[i+1,i] * w[i+1] = (-1/dr² + (d-1)/(2r_{i+1} dr)) * r_{i+1}^(d-1)
    # These are NOT exactly equal on a discrete grid (they ARE in continuum).
    # The discrepancy is O(dr) and vanishes as N→∞.

    upper_w = upper_orig * sqrt_w[:-1] / sqrt_w[1:]
    lower_w = lower_orig * sqrt_w[1:] / sqrt_w[:-1]

    # Average to symmetrize (removes O(dr) discretization error)
    offdiag_w = (upper_w + lower_w) / 2.0

    evals = eigh_tridiagonal(diag_vals, offdiag_w, eigvals_only=True,
                              select='i', select_range=(0, 2))
    return evals


# =============================================================================
# MAIN
# =============================================================================

def main():
    W = 78
    ratio_target = beta**2 / (4 * np.pi**2)

    print()
    print("=" * W)
    print("  SOLVER CORRECTNESS VERIFICATION")
    print("=" * W)
    print(f"  Target ratio = β²/(4π²) = {ratio_target:.6f}")

    N = 500  # Moderate grid for general eigensolver feasibility

    # Test modes
    Lambda_comp = 0
    Lambda_31 = angular_eigenvalue_33(3, 1)  # = 10

    print(f"\n{'RAW EIGENVALUES (N={N})':^{W}}")
    print("-" * W)

    methods = [
        ("M1: Asymmetric (v1-v3 bug)", method_1_asymmetric),
        ("M2: Self-adjoint",           method_2_selfadjoint),
        ("M3: General eigensolver",    method_3_general),
        ("M4: Weight-symmetrized",     method_4_symmetrized),
    ]

    results = {}
    for name, fn in methods:
        ev_comp = fn(Lambda_comp, N)
        ev_shear = fn(Lambda_31, N)
        results[name] = (ev_comp, ev_shear)

        print(f"\n  {name}:")
        print(f"    Compression λ₀ = {ev_comp[0]:14.4f}")
        print(f"    Shear (3,1) λ₀ = {ev_shear[0]:14.4f}")
        ratio = ev_shear[0] / ev_comp[0] if ev_comp[0] != 0 else float('nan')
        dev = abs(ratio / ratio_target - 1) * 100
        print(f"    Ratio = {ratio:.6f}, deviation = {dev:.2f}%")

    # ===== Cross-check: are M3 eigenvalues real? =====
    print(f"\n{'COMPLEX CHECK (M3)':^{W}}")
    print("-" * W)

    dr = 3.0 / (N + 1)
    r = np.linspace(dr, 3.0 - dr, N)
    d_eff = 6.0
    coeff_deriv = (d_eff - 1) / (2.0 * r * dr)

    # Check symmetry of the full matrix
    rho = parabolic_profile(r)
    V_dd = Vpp_quartic(rho)
    V_eff = Lambda_comp / r**2 + V_dd

    H = np.zeros((N, N))
    for i in range(N):
        H[i, i] = 2.0 / dr**2 + V_eff[i]
    for i in range(N - 1):
        H[i, i+1] = -1.0 / dr**2 - coeff_deriv[i]
        H[i+1, i] = -1.0 / dr**2 + coeff_deriv[i+1]

    asym = np.max(np.abs(H - H.T))
    print(f"  Matrix asymmetry ||H - H^T||_∞ = {asym:.4f}")

    all_evals = np.linalg.eigvals(H)
    max_imag = np.max(np.abs(np.imag(all_evals)))
    print(f"  Max imaginary part = {max_imag:.6e}")
    print(f"  Eigenvalues are {'real' if max_imag < 1e-6 else 'COMPLEX'}")

    min_eval = np.min(np.real(all_evals))
    max_eval = np.max(np.real(all_evals))
    print(f"  Eigenvalue range: [{min_eval:.2f}, {max_eval:.2f}]")

    # ===== Which method agrees with which? =====
    print(f"\n{'AGREEMENT TABLE':^{W}}")
    print("-" * W)

    method_names = [name for name, _ in methods]
    for i in range(len(methods)):
        for j in range(i+1, len(methods)):
            ni, nj = method_names[i], method_names[j]
            # Compare compression eigenvalues
            ev_i = results[ni][0][0]
            ev_j = results[nj][0][0]
            rel_diff = abs(ev_i - ev_j) / max(abs(ev_i), abs(ev_j)) * 100
            # Compare ratios
            ratio_i = results[ni][1][0] / results[ni][0][0]
            ratio_j = results[nj][1][0] / results[nj][0][0]
            ratio_diff = abs(ratio_i - ratio_j)

            print(f"  {ni[:20]:20s} vs {nj[:20]:20s}: "
                  f"λ₀_diff = {rel_diff:.1f}%, ratio_diff = {ratio_diff:.6f}")

    # ===== Convergence of M2 and M4 =====
    print(f"\n{'CONVERGENCE: M2 (SA) and M4 (Weight-Sym)':^{W}}")
    print("-" * W)

    for Ng in [100, 200, 500, 1000, 2000]:
        ev_c_sa = method_2_selfadjoint(Lambda_comp, Ng)[0]
        ev_s_sa = method_2_selfadjoint(Lambda_31, Ng)[0]
        ratio_sa = ev_s_sa / ev_c_sa

        ev_c_w = method_4_symmetrized(Lambda_comp, Ng)[0]
        ev_s_w = method_4_symmetrized(Lambda_31, Ng)[0]
        ratio_w = ev_s_w / ev_c_w

        dev_sa = abs(ratio_sa / ratio_target - 1) * 100
        dev_w = abs(ratio_w / ratio_target - 1) * 100

        print(f"  N={Ng:5d}: SA={ratio_sa:.6f} ({dev_sa:.2f}%), "
              f"W-Sym={ratio_w:.6f} ({dev_w:.2f}%)")

    print(f"\n{'=' * W}")
    print(f"  VERDICT")
    print(f"{'=' * W}")
    print(f"  If M3 (general) agrees with M2 (SA): the 2.83% was a BUG.")
    print(f"  If M3 agrees with M1 (asymmetric): there's a subtlety.")
    print(f"  M4 (weight-symmetrized) should agree with M2 and M3.")
    print(f"{'=' * W}")


if __name__ == '__main__':
    main()
