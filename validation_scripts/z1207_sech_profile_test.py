#!/usr/bin/env python3
"""
Test: Does a smooth sech boundary profile produce η_topo ≈ 0.029?

The claim: replacing the sharp Hill profile φ = 1 - y² with a physical
sech-smoothed boundary gives A_sech/A_hill ≈ 1.029, naturally producing
the topological correction without extracting it from the known k_geom.

We test this for various skin depths δ and see if any physically
motivated δ gives the right η.
"""

import numpy as np
from scipy.integrate import quad

# Target
ETA_TARGET = 0.0294  # extracted from book k_geom vs π/α scaling


def hill_profile(r):
    """Standard Hill vortex: φ = 1 - r² for r ≤ 1, 0 outside."""
    if r > 1:
        return 0.0
    return 1.0 - r**2


def hill_gradient(r):
    """dφ/dr for Hill vortex."""
    if r > 1:
        return 0.0
    return -2.0 * r


def sech_profile(r, delta):
    """Smooth sech boundary: transitions from Hill-like core to vacuum.

    φ(r) = (1 - r²) × 0.5×(1 + tanh((1-r)/δ))

    This keeps the parabolic core but smooths the boundary over width δ.
    """
    core = max(0, 1.0 - r**2)
    # Smooth cutoff centered at r=1 with width δ
    cutoff = 0.5 * (1.0 + np.tanh((1.0 - r) / delta))
    return core * cutoff


def sech_gradient(r, delta, dr=1e-6):
    """Numerical gradient of sech profile."""
    return (sech_profile(r + dr, delta) - sech_profile(r - dr, delta)) / (2 * dr)


def compute_A(profile_grad, rmax=2.0):
    """Curvature integral A = (1/2) ∫ |∇φ|² d³y = 2π ∫ (dφ/dr)² r² dr."""
    integrand = lambda r: profile_grad(r)**2 * 4 * np.pi * r**2
    val, _ = quad(integrand, 0, rmax, limit=200)
    return 0.5 * val


def compute_B(profile, rmax=2.0):
    """Compression integral B = (1/2) ∫ (φ-1)² d³y = 2π ∫ (φ-1)² r² dr."""
    integrand = lambda r: (profile(r) - 1.0)**2 * 4 * np.pi * r**2
    val, _ = quad(integrand, 0, rmax, limit=200)
    return 0.5 * val


def main():
    print("=" * 72)
    print("  SECH PROFILE TEST: Can η_topo ≈ 0.029 be derived?")
    print("=" * 72)

    # Bare Hill values
    A_hill = compute_A(hill_gradient)
    B_hill = compute_B(hill_profile)
    A_exact = 8 * np.pi / 5
    B_exact = 2 * np.pi / 7

    print(f"\n  Bare Hill vortex:")
    print(f"    A_hill = {A_hill:.6f}  (exact: {A_exact:.6f})")
    print(f"    B_hill = {B_hill:.6f}  (exact: {B_exact:.6f})")
    print(f"    A/B    = {A_hill/B_hill:.6f}  (exact: {A_exact/B_exact:.6f})")

    # Scan skin depths
    print(f"\n  Scanning sech profiles with various skin depth δ:")
    print(f"  Target: η = A_sech/A_hill - 1 ≈ {ETA_TARGET:.4f}")
    print(f"\n  {'δ':>8s} {'A_sech':>10s} {'B_sech':>10s} {'A/B ratio':>10s}"
          f" {'η_A':>10s} {'η_AB':>10s} {'status':>10s}")
    print(f"  {'─'*8} {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")

    best_delta = None
    best_err = 999

    for delta in [0.01, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25,
                  0.30, 0.35, 0.40, 0.50, 0.60, 0.80, 1.00]:
        grad_func = lambda r, d=delta: sech_gradient(r, d)
        prof_func = lambda r, d=delta: sech_profile(r, d)

        A_s = compute_A(grad_func, rmax=3.0)
        B_s = compute_B(prof_func, rmax=3.0)

        # η from A alone (curvature enhancement)
        eta_A = A_s / A_hill - 1

        # η from A/B ratio (what matters for k_geom)
        ratio_s = A_s / B_s
        ratio_h = A_hill / B_hill
        eta_AB = ratio_s / ratio_h - 1

        err = abs(eta_AB - ETA_TARGET)
        if err < best_err:
            best_err = err
            best_delta = delta

        status = "◄ MATCH" if abs(eta_AB - ETA_TARGET) < 0.005 else ""
        print(f"  {delta:8.3f} {A_s:10.6f} {B_s:10.6f} {ratio_s:10.6f}"
              f" {eta_A:+10.4f} {eta_AB:+10.4f} {status:>10s}")

    print(f"\n  Best match: δ = {best_delta:.3f} (err = {best_err:.4f})")

    # Check if ANY delta gives η ≈ 0.029
    if best_err < 0.005:
        print(f"\n  RESULT: YES — sech profile with δ ≈ {best_delta:.3f} produces η ≈ 0.029")
        print(f"  But is δ = {best_delta:.3f} physically motivated?")

        # Physical motivation: δ should relate to β
        beta = 3.043233053
        print(f"\n  Physical skin depths:")
        print(f"    1/β     = {1/beta:.4f}")
        print(f"    1/√β    = {1/np.sqrt(beta):.4f}")
        print(f"    α       = {1/137.036:.6f}")
        print(f"    1/(2π)  = {1/(2*np.pi):.4f}")
        print(f"    β/π²    = {beta/np.pi**2:.4f}")
    else:
        print(f"\n  RESULT: NO — no sech skin depth produces η ≈ 0.029")
        print(f"  The sech profile hypothesis does NOT explain the topological correction.")
        print(f"  The closest match gives η = {ETA_TARGET + best_err:.4f} or {ETA_TARGET - best_err:.4f}")


if __name__ == "__main__":
    main()
