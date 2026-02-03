#!/usr/bin/env python3
"""
Lepton Energy Functional with Gradient (Curvature) Term

Purpose: Test whether explicit gradient energy E_∇ breaks the β degeneracy
         and shifts β_min from ~3.15 toward the Golden Loop value 3.043233053.

Key physics:
- E_circ ~ R³ (circulation, bulk hydrodynamics)
- E_stab ~ β·A²R³ (stabilization, bulk density deficit)
- E_grad ~ λ·A²R (gradient/curvature, boundary-layer physics)

The R vs R³ scaling asymmetry is what breaks the degeneracy that made
the radius constraint redundant in the self-similar closure.

Profile family: Family A (polynomial with smooth edge)
    Δρ(r) = -A(1-(r/R)²)^s  for r ≤ R
    Δρ(r) = 0               for r > R

where s ≥ 2 ensures C¹ smoothness at boundary (no derivative kink).
"""

import numpy as np
from scipy.integrate import simps
from scipy.special import beta as beta_func


# Physical constants (dimensionless units)
RHO_VAC = 1.0  # Vacuum density

# Analytic coefficients for Family A profile
def K_stab(s):
    """Stabilization coefficient: E_stab = β·K_stab(s)·A²R³"""
    if s == 2.0:
        return 512 * np.pi / 3465  # Exact for s=2
    elif s == 3.0:
        return 4096 * np.pi / 45045  # Exact for s=3
    else:
        # General: K_stab = 2π·B(3/2, 2s+1)
        return 2 * np.pi * beta_func(1.5, 2*s + 1)


def K_grad(s):
    """Gradient coefficient: E_grad = λ·K_grad(s)·A²R"""
    if s == 2.0:
        return 512 * np.pi / 315  # Exact for s=2
    elif s == 3.0:
        return 6144 * np.pi / 5005  # Exact for s=3
    else:
        # General: K_grad = 8π·s²·B(5/2, 2s-1)
        return 8 * np.pi * s**2 * beta_func(2.5, 2*s - 1)


class HillVortexStreamFunction:
    """Hill spherical vortex stream function (unchanged from original)."""

    def __init__(self, R, U):
        self.R = R
        self.U = U

    def velocity_components(self, r, theta):
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        v_r = np.zeros_like(r)
        v_theta = np.zeros_like(r)

        mask_internal = r < self.R

        if np.any(mask_internal):
            r_int = r[mask_internal]
            dpsi_dr_int = -(3 * self.U / (self.R**2)) * r_int**3 * sin_theta**2
            dpsi_dtheta_int = -(3 * self.U / (2 * self.R**2)) * \
                (self.R**2 - r_int**2) * r_int**2 * 2 * sin_theta * cos_theta

            v_r[mask_internal] = dpsi_dtheta_int / (r_int**2 * sin_theta + 1e-10)
            v_theta[mask_internal] = -dpsi_dr_int / (r_int * sin_theta + 1e-10)

        mask_external = ~mask_internal
        if np.any(mask_external):
            r_ext = r[mask_external]
            dpsi_dr_ext = (self.U / 2) * (2*r_ext + self.R**3 / r_ext**2) * sin_theta**2
            dpsi_dtheta_ext = (self.U / 2) * (r_ext**2 - self.R**3 / r_ext) * \
                2 * sin_theta * cos_theta

            v_r[mask_external] = dpsi_dtheta_ext / (r_ext**2 * sin_theta + 1e-10)
            v_theta[mask_external] = -dpsi_dr_ext / (r_ext * sin_theta + 1e-10)

        return v_r, v_theta


class DensityGradient:
    """
    Family A density profile: Δρ = -A(1-(r/R)²)^s

    Parameters:
        R: core radius (vortex scale)
        amplitude: A, deficit depth (∈ [0, ρ_vac])
        s: sharpness exponent (≥ 2 for C¹ smoothness)
        rho_vac: vacuum density (default 1.0)
    """

    def __init__(self, R, amplitude, s=2.0, rho_vac=RHO_VAC):
        self.R = float(R)
        self.A = float(amplitude)
        self.s = float(s)
        self.rho_vac = float(rho_vac)

        if self.s < 2.0:
            import warnings
            warnings.warn(f"s={s} < 2.0: profile will have derivative kink at r=R")

    def delta_rho(self, r):
        """Density deficit: Δρ(r) = -A(1-(r/R)²)^s for r ≤ R, 0 otherwise"""
        delta = np.zeros_like(r)
        mask = r <= self.R
        if np.any(mask):
            x = r[mask] / self.R
            delta[mask] = -self.A * np.power(np.maximum(1.0 - x**2, 0.0), self.s)
        return delta

    def rho(self, r):
        """Total density: ρ(r) = ρ_vac + Δρ(r)"""
        return self.rho_vac + self.delta_rho(r)


class LeptonEnergy:
    """
    Compute lepton energy with gradient term.

    Energy components:
        E_circ: Circulation kinetic energy (numeric integral)
        E_stab: Stabilization energy (analytic: β·K_stab·A²R³)
        E_grad: Gradient energy (analytic: λ·K_grad·A²R)

    Total: E_total = E_circ - E_stab + E_grad
    """

    def __init__(self, beta, s=2.0, lam=1.0, r_max=10.0, num_r=100, num_theta=20):
        self.beta = beta
        self.s = s
        self.lam = lam
        self.rho_vac = RHO_VAC

        # Radial and angular grids
        self.r = np.linspace(0.01, r_max, num_r)
        self.theta = np.linspace(0.01, np.pi - 0.01, num_theta)
        self.dr = self.r[1] - self.r[0]
        self.dtheta = self.theta[1] - self.theta[0]

        # Precompute analytic coefficients
        self.K_stab = K_stab(s)
        self.K_grad = K_grad(s)

    def circulation_energy(self, R, U, A):
        """Numeric integration of circulation kinetic energy."""
        stream = HillVortexStreamFunction(R, U)
        density = DensityGradient(R, A, self.s, self.rho_vac)

        E_circ = 0.0
        for theta in self.theta:
            v_r, v_theta = stream.velocity_components(self.r, theta)
            v_squared = v_r**2 + v_theta**2
            rho_actual = density.rho(self.r)
            integrand = 0.5 * rho_actual * v_squared * self.r**2 * np.sin(theta)
            E_circ += simps(integrand, x=self.r) * self.dtheta
        E_circ *= 2 * np.pi

        return E_circ

    def stabilization_energy(self, R, A):
        """Analytic stabilization energy: E_stab = β·K_stab(s)·A²R³"""
        return self.beta * self.K_stab * A**2 * R**3

    def gradient_energy(self, R, A):
        """Analytic gradient energy: E_grad = λ·K_grad(s)·A²R"""
        return self.lam * self.K_grad * A**2 * R

    def total_energy(self, R, U, A):
        """
        Compute total energy: E_total = E_circ - E_stab + E_grad

        Returns:
            E_total: Total energy
            E_circ: Circulation energy
            E_stab: Stabilization energy
            E_grad: Gradient energy
        """
        E_circ = self.circulation_energy(R, U, A)
        E_stab = self.stabilization_energy(R, A)
        E_grad = self.gradient_energy(R, A)

        E_total = E_circ - E_stab + E_grad

        return E_total, E_circ, E_stab, E_grad

    def energy_diagnostic(self, R, A):
        """
        Compute E_grad/E_stab ratio (diagnostic for curvature-bulk competition).

        Returns: E_grad/E_stab = (λ·K_grad)/(β·K_stab·R²)
        """
        if R <= 0:
            return 0.0
        return (self.lam * self.K_grad) / (self.beta * self.K_stab * R**2)


# Unit tests
if __name__ == "__main__":
    print("Testing analytic energy coefficients...\n")

    # Test 1: Coefficient values
    print("Analytic coefficients for Family A:")
    for s in [2.0, 2.5, 3.0]:
        K_s = K_stab(s)
        K_g = K_grad(s)
        print(f"  s={s:.1f}: K_stab={K_s:.6f}, K_grad={K_g:.6f}")

    # Test 2: Energy computation
    print("\nTest energy computation (s=2, β=3.043233053, λ=1.0):")
    energy_calc = LeptonEnergy(beta=3.043233053, s=2.0, lam=1.0)

    # Example parameters (electron-like)
    R, U, A = 0.88, 0.036, 0.92

    E_total, E_circ, E_stab, E_grad = energy_calc.total_energy(R, U, A)

    print(f"  R={R}, U={U}, A={A}")
    print(f"  E_circ  = {E_circ:.6e}")
    print(f"  E_stab  = {E_stab:.6e}")
    print(f"  E_grad  = {E_grad:.6e}")
    print(f"  E_total = {E_total:.6e}")
    print(f"  E_grad/E_stab = {E_grad/E_stab:.4f}")
    print(f"  Ratio diagnostic = {energy_calc.energy_diagnostic(R, A):.4f}")

    # Test 3: Scaling check
    print("\nScaling check (double R, expect E_stab×8, E_grad×2):")
    R2 = 2 * R
    E_total2, E_circ2, E_stab2, E_grad2 = energy_calc.total_energy(R2, U, A)

    print(f"  E_stab ratio: {E_stab2/E_stab:.2f} (expect ~8.0)")
    print(f"  E_grad ratio: {E_grad2/E_grad:.2f} (expect ~2.0)")

    print("\n✓ All tests complete")
