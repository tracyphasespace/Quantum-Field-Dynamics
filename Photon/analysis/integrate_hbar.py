#!/usr/bin/env python3
"""
QFD: Numerical Integration of Vortex Spin

Critical Test: Does Hill Vortex geometry produce ℏ ~ c?

If YES → Quantization is geometric (Theory of Everything)
If NO → Need to adjust vortex model or abandon emergence
"""

import numpy as np
from scipy import integrate

def integrate_hbar():
    print("=== NUMERICAL INTEGRATION OF VORTEX SPIN ===")
    print("Goal: Calculate emergent hbar from Vortex Geometry")

    # 1. SETUP PARAMETERS (Natural Units)
    # We work in units where R=1 and rho_vac=1 to find the GEOMETRIC FACTOR
    R = 1.0
    beta = 3.058

    # Emergent speed of light from previous step
    # c = sqrt(beta/rho)
    c_vac = np.sqrt(beta / 1.0)

    print(f"Input β: {beta}")
    print(f"Emergent c: {c_vac:.4f} (vacuum shear velocity)")

    # 2. DEFINE VELOCITY FIELDS
    # Hill Vortex Stream Function (Internal r < R)
    # psi ~ r^2 * (R^2 - r^2) * sin^2(theta)

    def poloidal_velocity(r, theta):
        """
        Standard Hill Vortex rolling motion.
        v_p = (1/r sin theta) * dpsi/dr ...
        """
        # Simplified magnitude profile for integration weight
        # Highest at core, zero at axis and rim
        return 1.5 * c_vac * (r/R) * (1 - (r/R)**2)

    def toroidal_velocity(r, theta):
        """
        The "Spin" Component.
        In QFD, the vacuum stiffness 'squeezes' the vortex.
        To resist collapse, it must spin.

        Ansatz: Swirl is coupled to the pressure gradient (Bernoulli).
        v_phi ~ proportional to poloidal flow (Helical flow)
        """
        # A stable vortex knot typically has equal poloidal and toroidal energy
        # (Equipartition of vorticity).
        return poloidal_velocity(r, theta)

    # 3. INTEGRATE ANGULAR MOMENTUM
    # L = Integral( r_perp * v_phi * rho * dV )
    # r_perp = r * sin(theta)
    # dV = r^2 sin(theta) dr dtheta dphi

    print("\nIntegration Grid: Spherical (r, θ, φ)")

    def angular_momentum_integrand(r, theta, phi):
        # r_perp (lever arm)
        lever = r * np.sin(theta)

        # v_tangential (swirl velocity)
        v_swirl = toroidal_velocity(r, theta)

        # Mass density (uniform vacuum + perturbation)
        # For simplicity in this geometry check, we assume density ~ 1
        rho = 1.0

        # dV factor (Jacobian)
        jacobian = r**2 * np.sin(theta)

        return lever * v_swirl * rho * jacobian

    # Perform Triple Integration
    # r: 0 to R
    # theta: 0 to pi
    # phi: 0 to 2pi

    # We can do phi analytically (2pi)
    # We integrate r and theta numerically

    def r_theta_integrand(theta, r):
        return angular_momentum_integrand(r, theta, 0)

    # Scipy dblquad integrates (y, x) -> (theta, r)
    total_L, error = integrate.dblquad(
        r_theta_integrand,
        0, R,           # r limits
        lambda r: 0,    # theta min
        lambda r: np.pi # theta max
    )

    # Multiply by 2pi for phi integration
    total_L *= (2 * np.pi)

    print(f"\nIntegration Result:")
    print(f"Geometric Angular Momentum (L): {total_L:.4f}")
    print(f"Integration Error Estimate: {error:.2e}")

    # 4. COMPARE TO EXPECTED HBAR
    # In natural units, we define the 'Action' of the vortex.
    # If this is a fundamental fermion, L should be hbar/2.

    emergent_hbar = 2 * total_L

    print(f"Implied Planck Constant (ℏ = 2L): {emergent_hbar:.4f}")

    # 5. CHECK CONSISTENCY
    # Does this match the c_vac scaling?
    # hbar ~ c * L_scale
    print(f"\nConsistency Check:")
    print(f"Is ℏ ≈ c? {emergent_hbar:.4f} vs {c_vac:.4f}")
    ratio = emergent_hbar / c_vac
    print(f"Ratio ℏ/c: {ratio:.4f}")

    target_ratio = 1.0 # In natural units c=hbar=1
    diff = abs(ratio - target_ratio)

    if diff < 0.1:
        print("✅ SUCCESS: Geometry yields ℏ ~ c (Natural Units verified)")
        print("   → Quantization is GEOMETRIC")
        print("   → QFD is a Theory of Everything candidate")
    else:
        print(f"⚠️  OFFSET: Geometry factor mismatch.")
        print(f"   Correction needed: {1/ratio:.4f}")
        print(f"   → Either adjust vortex model or abandon emergence hypothesis")

    # 6. BREAKDOWN OF CONTRIBUTIONS
    print("\n=== PHYSICAL INTERPRETATION ===")
    print(f"1. Vacuum stiffness β = {beta} sets wave speed c = {c_vac:.4f}")
    print(f"2. Vortex stability requires rim velocity ~ c")
    print(f"3. Geometric integration of angular momentum L = {total_L:.4f}")
    print(f"4. Fermion spin S = ℏ/2 implies ℏ = {emergent_hbar:.4f}")
    print(f"5. In natural units, ℏ/c = {ratio:.4f}")

    if diff < 0.1:
        print("\n🎯 OBSERVATION:")
        print("   Geometric integration yields Γ_vortex = ℏ/c ratio")
        print("   IF electron is Hill Vortex, THEN ℏ ~ Γ·M·R·c")
        print("   This is a dimensional consistency check, not a derivation")

    return emergent_hbar, c_vac, ratio

if __name__ == "__main__":
    integrate_hbar()
