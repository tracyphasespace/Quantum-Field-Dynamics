#!/usr/bin/env python3
"""
hessian_v4_bivector_coupling.py -- Bivector coupling corrections to the Hessian

The v3 profile scan proved the eigenvalue ratio is ENTIRELY determined by:
  1. The angular eigenvalue Λ (currently = 10 for the (3,1) mode)
  2. The effective radial dimension d_eff (currently = 6)

The density profile drops out completely. This means the 2.83% residual
must come from a correction to Λ or d_eff in the full Cl(3,3) Hessian.

This solver investigates:
  Phase A: What Λ_exact gives the target ratio at d_eff=6?
  Phase B: What physical mechanism produces the correction δΛ?
  Phase C: The 2-channel (scalar + bivector) Hessian

BIVECTOR COUPLING PHYSICS:
    The Hill vortex ground state is NOT a pure scalar. It has:
        ψ₀ = ρ₀(r) + B₀(r) e₄∧e₅
    where B₀ is the bivector amplitude (internal rotation/spin).

    For the Hill vortex: B₀(r) = ρ₀(r) × U(r)/c where U(r) is the
    circulation velocity. In solid-body rotation: U(r) = Ω × r.

    The scalar perturbation η_s couples to the bivector through the
    kinetic cross-term in the Lagrangian:
        δ²E/δρ_s δB₀ = ⟨∇η_s, ∇(η_B e₄∧e₅)⟩_{Cl(3,3)}

    This coupling adds off-diagonal terms to the Hessian matrix and
    shifts the eigenvalues. The shift depends on the spin-orbit
    coupling strength, which is proportional to the circulation speed.

Copyright (c) 2026 Tracy McSheery
Licensed under the MIT License
"""

import sys
import os
import numpy as np
from scipy.linalg import eigh_tridiagonal, eigh
from scipy.optimize import brentq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from qfd.shared_constants import ALPHA, BETA, XI_QFD, K_GEOM


# =============================================================================
# PARAMETERS
# =============================================================================

beta = BETA
lam = beta
rho_0 = 1.0
a = 1.0


# =============================================================================
# CORE SOLVER (from v3, profile-independent)
# =============================================================================

def parabolic_profile(r):
    return np.where(r <= a, rho_0 * (1.0 - (r/a)**2), 0.0)


def Vpp_quartic(rho):
    mu_sq = 2.0 * lam * rho_0**2
    return -2.0 * mu_sq + 12.0 * lam * rho**2


def solve_lowest_eigenvalue(Lambda, N=1000, r_max=3.0, d_eff=6.0):
    """Solve for lowest eigenvalue given angular eigenvalue Λ and d_eff."""
    dr = r_max / (N + 1)
    r = np.linspace(dr, r_max - dr, N)

    rho = parabolic_profile(r)
    V_dd = Vpp_quartic(rho)
    V_eff = Lambda / r**2 + V_dd

    diag = 2.0 / dr**2 + V_eff
    coeff_deriv = (d_eff - 1) / (2.0 * r * dr)
    upper = -1.0 / dr**2 - coeff_deriv[:-1]

    evals = eigh_tridiagonal(diag, upper, eigvals_only=True,
                              select='i', select_range=(0, 0))
    return evals[0]


def compute_ratio(Lambda_shear, N=1000, r_max=3.0, d_eff=6.0):
    """Ratio of shear to compression eigenvalue."""
    ev_comp = solve_lowest_eigenvalue(0.0, N, r_max, d_eff)
    ev_shear = solve_lowest_eigenvalue(Lambda_shear, N, r_max, d_eff)
    return ev_shear / ev_comp if ev_comp != 0 else float('nan')


# =============================================================================
# PHASE A: FIND EXACT Λ
# =============================================================================

def find_exact_Lambda(d_eff=6.0, N=2000, r_max=3.0):
    """Find the angular eigenvalue that gives exact ratio = β²/(4π²)."""
    ratio_target = beta**2 / (4 * np.pi**2)

    def residual(Lambda):
        return compute_ratio(Lambda, N, r_max, d_eff) - ratio_target

    # Bracket: Λ=10 gives 0.228, target is 0.235, need Λ slightly > 10
    Lambda_exact = brentq(residual, 8.0, 15.0)
    return Lambda_exact


# =============================================================================
# PHASE B: BIVECTOR ANGULAR MOMENTUM CORRECTION
# =============================================================================

def bivector_angular_correction():
    """Compute the angular momentum correction from the bivector field.

    The Hill vortex ground state has a bivector component B₀ = ρ₀ × (U/c)
    in the e₄∧e₅ plane. This bivector carries angular momentum:
        L_bivector = ∫ ρ₀ × B₀ × r² dV ∝ ∫ ρ₀² × U × r² dV

    For solid-body rotation U(r) = Ω r inside the vortex:
        L_biv ∝ ∫₀ᵃ ρ₀²(r) × Ω r × r^(d-1) dr

    The effective angular momentum quantum number shift is:
        δℓ² ≈ ⟨L_biv⟩ / ⟨L_total⟩

    In QFD, the circulation speed U_circ = c√β/2 ≈ 0.872c.
    The fractional bivector amplitude is B₀/ρ₀ = U_circ/c = √β/2.
    """

    # Bivector fraction (circulation speed / c)
    U_circ_over_c = np.sqrt(beta) / 2.0  # ≈ 0.872

    # The bivector field carries intrinsic angular momentum in the e₄∧e₅ plane.
    # This manifests as a correction to the effective angular eigenvalue.
    #
    # In the (3,3) decomposition:
    #   Λ = ℓ_s(ℓ_s + 1) - ℓ_t(ℓ_t + 1)
    #
    # The bivector adds a spin-orbit coupling:
    #   Λ_eff = Λ + δΛ_SO
    #
    # For a bivector in the e₄∧e₅ plane (one spacelike + one timelike direction):
    #   The spin S = 1 (bivector has spin-1 under rotations in that plane)
    #   Spin-orbit coupling: δΛ_SO = 2 × S × ℓ_t × cos(θ_SO)
    #   where θ_SO is the angle between spin and orbital angular momentum
    #
    # For the (3,1) mode with ℓ_t = 1:
    #   δΛ_SO = 2 × 1 × 1 × (U/c)² = 2(U/c)² = 2β/4 = β/2
    #   This is a SPIN-ORBIT coupling, proportional to U²/c² = β/4

    delta_Lambda_SO = beta / 2.0  # ≈ 1.52

    return U_circ_over_c, delta_Lambda_SO


# =============================================================================
# PHASE C: 2-CHANNEL HESSIAN (SCALAR + BIVECTOR)
# =============================================================================

def solve_2channel(ell_s, ell_t, coupling_strength, N=1000, r_max=3.0):
    """Solve the 2-channel (scalar + bivector) Hessian.

    The Hessian matrix for perturbations (η_s, η_B) is:
        H = | -∇² + V_ss + Λ_s/r²     κ(r)              |
            | κ(r)                       -∇² + V_BB + Λ_B/r² |

    where:
    - V_ss = V''(ρ₀) for scalar-scalar (same as single-channel)
    - V_BB = V''(ρ₀) for bivector-bivector (same potential)
    - κ(r) = coupling_strength × B₀(r) / r² (spin-orbit coupling)
    - Λ_s = ℓ_s(ℓ_s+1) - ℓ_t(ℓ_t+1) (angular eigenvalue for scalar)
    - Λ_B = Λ_s + δΛ_spin (bivector carries intrinsic spin)

    The coupling κ(r) comes from the kinetic cross-term:
        ⟨∇η_s, ∇(η_B × e₄∧e₅)⟩ = coupling × B₀(r) × angular overlap
    """
    d_eff = 6.0
    dr = r_max / (N + 1)
    r = np.linspace(dr, r_max - dr, N)

    rho = parabolic_profile(r)
    V_dd = Vpp_quartic(rho)

    Lambda_s = ell_s * (ell_s + 1) - ell_t * (ell_t + 1)

    # Bivector ground state amplitude: B₀ = ρ₀ × U_circ/c
    U_over_c = np.sqrt(beta) / 2.0
    B0 = rho * U_over_c

    # Bivector intrinsic angular momentum: spin-1 in the e₄∧e₅ plane
    # adds +2 to the angular eigenvalue (from ℓ_B = ℓ ± S, S=1)
    # The coupling between scalar ℓ and bivector ℓ±1 modes gives
    # an intermediate effective Λ
    delta_spin = 2.0  # Bivector carries spin-1 → δΛ = 2S = 2
    Lambda_B = Lambda_s + delta_spin

    # Build 2N×2N block Hessian
    # Block structure: [scalar_block, coupling; coupling, bivector_block]

    # Scalar block (N×N tridiagonal)
    V_eff_s = Lambda_s / r**2 + V_dd
    diag_s = 2.0 / dr**2 + V_eff_s
    coeff_deriv = (d_eff - 1) / (2.0 * r * dr)
    upper_s = -1.0 / dr**2 - coeff_deriv[:-1]
    lower_s = -1.0 / dr**2 + coeff_deriv[1:]

    # Bivector block (N×N tridiagonal, same structure but different Λ)
    V_eff_B = Lambda_B / r**2 + V_dd
    diag_B = 2.0 / dr**2 + V_eff_B

    # Coupling: κ(r) = coupling_strength × B₀(r) × angular overlap / r
    # The angular overlap between scalar ℓ and bivector ℓ±1 modes
    # is given by Clebsch-Gordan coefficients ≈ 1/√(2ℓ+1)
    kappa = coupling_strength * B0

    # Build full 2N×2N dense matrix (small N for feasibility)
    H = np.zeros((2*N, 2*N))

    # Scalar diagonal
    for i in range(N):
        H[i, i] = diag_s[i]
    for i in range(N-1):
        H[i, i+1] = upper_s[i]
        H[i+1, i] = lower_s[i]

    # Bivector diagonal
    for i in range(N):
        H[N+i, N+i] = diag_B[i]
    for i in range(N-1):
        H[N+i, N+i+1] = upper_s[i]  # Same kinetic structure
        H[N+i+1, N+i] = lower_s[i]

    # Off-diagonal coupling
    for i in range(N):
        H[i, N+i] = kappa[i]
        H[N+i, i] = kappa[i]

    # Solve for lowest eigenvalues
    eigenvalues = np.linalg.eigvalsh(H)
    return eigenvalues[:10]


# =============================================================================
# MAIN
# =============================================================================

def main():
    W = 78
    ratio_target = beta**2 / (4 * np.pi**2)
    Lambda_31 = 3 * 4 - 1 * 2  # = 10

    print()
    print("=" * W)
    print("  HESSIAN v4: Bivector Coupling Analysis")
    print("=" * W)
    print(f"  β = {beta:.10f}")
    print(f"  Target ratio = β²/(4π²) = {ratio_target:.6f}")
    print(f"  Current (3,1) Λ = {Lambda_31}")
    print(f"  Current ratio at Λ={Lambda_31} = {compute_ratio(Lambda_31, N=2000):.6f}")

    # ===== PHASE A: Find exact Λ =====
    print(f"\n{'PHASE A: EXACT ANGULAR EIGENVALUE':^{W}}")
    print("-" * W)

    Lambda_exact = find_exact_Lambda(d_eff=6.0, N=2000)
    delta_Lambda = Lambda_exact - Lambda_31

    print(f"  Exact Λ for ratio = β²/(4π²): {Lambda_exact:.6f}")
    print(f"  Current Λ (3,1):              {Lambda_31}")
    print(f"  Required correction δΛ:       {delta_Lambda:+.6f}")
    print(f"  δΛ/Λ (fractional):            {delta_Lambda/Lambda_31*100:+.3f}%")

    # Check against nice numbers
    print(f"\n  Is δΛ a recognizable quantity?")
    candidates = {
        "β/2":         beta / 2,
        "β/π":         beta / np.pi,
        "2α":          2 * ALPHA,
        "α×β":         ALPHA * beta,
        "β²/(4π²)":    beta**2 / (4 * np.pi**2),
        "1/(2π)":      1 / (2 * np.pi),
        "β-3":         beta - 3,
        "√β - √3":    np.sqrt(beta) - np.sqrt(3),
        "α/π × β":    ALPHA / np.pi * beta,
        "(5/6)×β/(4π²)": (5./6) * beta / (4 * np.pi**2),
        "1/d = 1/6":  1./6,
    }
    for desc, val in sorted(candidates.items(), key=lambda x: abs(x[1] - delta_Lambda)):
        diff_pct = (val - delta_Lambda) / delta_Lambda * 100 if delta_Lambda != 0 else float('nan')
        print(f"    {desc:20s} = {val:.6f}  (diff: {diff_pct:+.1f}%)")

    # ===== PHASE B: Bivector angular correction =====
    print(f"\n{'PHASE B: BIVECTOR ANGULAR MOMENTUM':^{W}}")
    print("-" * W)

    U_over_c, delta_SO = bivector_angular_correction()
    print(f"  U_circ/c = √β/2 = {U_over_c:.6f}")
    print(f"  Naive spin-orbit δΛ = β/2 = {delta_SO:.6f}")
    print(f"  Required δΛ = {delta_Lambda:.6f}")

    # What fraction of spin-orbit coupling gives exact match?
    fraction = delta_Lambda / delta_SO if delta_SO != 0 else float('nan')
    print(f"  Coupling fraction needed: {fraction:.4f}")
    print(f"  (i.e., actual coupling = {fraction:.4f} × naive estimate)")

    # Alternative: the bivector correction is through U²/c² = β/4
    delta_U2 = beta / 4.0
    fraction_U2 = delta_Lambda / delta_U2
    print(f"\n  Alternative: δΛ = (β/4) × f")
    print(f"  β/4 = {delta_U2:.6f}")
    print(f"  f = {fraction_U2:.6f}")

    # ===== PHASE C: 2-Channel Hessian =====
    print(f"\n{'PHASE C: 2-CHANNEL (SCALAR + BIVECTOR) HESSIAN':^{W}}")
    print("-" * W)

    # Scan coupling strength to find what closes the gap
    N_small = 200  # Small grid for dense matrix feasibility

    # First: single-channel reference
    ratio_1ch = compute_ratio(Lambda_31, N=N_small)
    print(f"  Single-channel reference (N={N_small}): ratio = {ratio_1ch:.6f}")

    # Compression mode (2-channel)
    print(f"\n  Scanning coupling strength κ:")
    best_kappa = 0
    best_dev = float('inf')

    for kappa_val in [0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]:
        # Compression: (0,0) mode
        ev_comp = solve_2channel(0, 0, kappa_val, N=N_small)
        # Shear: (3,1) mode
        ev_shear = solve_2channel(3, 1, kappa_val, N=N_small)

        ratio = ev_shear[0] / ev_comp[0] if ev_comp[0] != 0 else float('nan')
        dev = abs(ratio / ratio_target - 1) * 100
        marker = " ***" if dev < best_dev else ""
        if dev < best_dev:
            best_dev = dev
            best_kappa = kappa_val

        print(f"    κ={kappa_val:6.1f}: comp₀={ev_comp[0]:12.2f}, "
              f"shear₀={ev_shear[0]:12.2f}, ratio={ratio:.6f}, "
              f"dev={dev:.2f}%{marker}")

    print(f"\n  Best coupling κ = {best_kappa}, deviation = {best_dev:.2f}%")

    # Fine scan around best
    if best_kappa > 0:
        print(f"\n  Fine scan around κ = {best_kappa}:")
        for kappa_val in np.linspace(best_kappa * 0.5, best_kappa * 1.5, 11):
            ev_comp = solve_2channel(0, 0, kappa_val, N=N_small)
            ev_shear = solve_2channel(3, 1, kappa_val, N=N_small)
            ratio = ev_shear[0] / ev_comp[0] if ev_comp[0] != 0 else float('nan')
            dev = abs(ratio / ratio_target - 1) * 100
            print(f"    κ={kappa_val:8.3f}: ratio={ratio:.6f}, dev={dev:.2f}%")

    # ===== SUMMARY =====
    print(f"\n{'=' * W}")
    print(f"  SUMMARY")
    print(f"{'=' * W}")
    print(f"  Phase A: Exact Λ = {Lambda_exact:.4f} (vs integer 10)")
    print(f"           Correction δΛ = {delta_Lambda:+.4f}")
    print(f"  Phase B: Spin-orbit estimate δΛ = β/2 = {delta_SO:.4f}")
    print(f"           Required fraction: {fraction:.4f}")
    print(f"  Phase C: Best 2-channel κ = {best_kappa}")
    print(f"           Best deviation = {best_dev:.2f}%")
    print(f"{'=' * W}")


if __name__ == '__main__':
    main()
