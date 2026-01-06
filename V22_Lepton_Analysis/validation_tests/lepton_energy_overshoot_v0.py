#!/usr/bin/env python3
"""
Lepton Energy Functional with Overshoot Shell v0

Purpose: Break spatial orthogonality by allowing ρ > 1 in region where v² is large

MINIMAL CHANGE (v0):
  - ONE new parameter: B ≥ 0 (shell amplitude)
  - TIED geometry: R_shell = R_c + w, Δ = 0.1 × R_shell (hard-coded)
  - REGULARIZER: +λ_B B² penalty (prevents B from being universal knob)

Density ansatz:
    ρ(r) = 1 - A·f_void(r; R_c, w) + B·exp[-(r - R_shell)²/(2Δ²)]

where:
    f_void(r) = (1 - (r/R_outer)²)² · T(r)  (existing deficit profile)
    R_shell = R_c + w (tied to existing geometry)
    Δ = epsilon · R_shell with epsilon = 0.1 (fixed)

This allows τ to "carry ballast" (ρ > 1) where velocity is concentrated,
while e, μ remain at B ≈ 0 (no ballast needed).
"""

import numpy as np
from scipy.integrate import simps
from lepton_energy_boundary_layer import (
    build_smart_radial_grid,
    HillVortexStreamFunction,
    DensityBoundaryLayer,
    RHO_VAC,
)


class DensityBoundaryLayerWithOvershoot(DensityBoundaryLayer):
    """
    Extends DensityBoundaryLayer with overshoot shell capability.

    ρ(r) = 1 - A·f_void(r) + B·f_shell(r)

    where:
        f_void(r) = (1 - (r/R_outer)²)² · T(r)  (existing deficit)
        f_shell(r) = exp[-(r - R_shell)²/(2Δ²)]  (overshoot shell)
        R_shell = R_c + w (tied)
        Δ = 0.1 × R_shell (hard-coded for v0)
    """

    def __init__(self, R_c, w, A, B, rho_vac=RHO_VAC, epsilon_shell=0.1):
        """
        Parameters
        ----------
        R_c : float
            Core radius
        w : float
            Boundary layer thickness
        A : float
            Deficit amplitude
        B : float
            Overshoot shell amplitude (B ≥ 0)
        rho_vac : float
            Vacuum density (default 1.0)
        epsilon_shell : float
            Shell width parameter: Δ = epsilon · R_shell (default 0.1)
        """
        # Initialize parent (deficit profile)
        super().__init__(R_c, w, A, rho_vac)

        # Overshoot shell parameters
        self.B = float(B)
        self.epsilon_shell = float(epsilon_shell)

        # Tied geometry
        self.R_shell = self.R_c + self.w
        self.Delta = self.epsilon_shell * self.R_shell

    def shell_profile(self, r):
        """
        Gaussian overshoot shell centered at R_shell

        f_shell(r) = exp[-(r - R_shell)²/(2Δ²)]
        """
        return np.exp(-((r - self.R_shell) ** 2) / (2 * self.Delta**2))

    def rho(self, r):
        """
        Total density with deficit + overshoot:

        ρ(r) = ρ_vac - A·f_void(r) + B·f_shell(r)
        """
        # Get deficit from parent
        rho_deficit = super().rho(r)

        # Add overshoot shell
        rho_overshoot = self.B * self.shell_profile(r)

        return rho_deficit + rho_overshoot

    def delta_rho(self, r):
        """
        Total density variation from vacuum:

        Δρ(r) = ρ(r) - ρ_vac = -A·f_void(r) + B·f_shell(r)
        """
        # Deficit (negative)
        delta_deficit = super().delta_rho(r)

        # Overshoot (positive)
        delta_overshoot = self.B * self.shell_profile(r)

        return delta_deficit + delta_overshoot


class LeptonEnergyWithOvershoot:
    """
    Energy functional with overshoot shell capability.

    E_total = E_circ - E_stab + E_grad + E_bulk

    where:
        E_circ = ∫ ½ρ(r) v²(r) d³r  (now sensitive to ρ via overshoot)
        E_stab = β ∫ (Δρ)² d³r
        E_grad = λ ∫ |∇ρ|² d³r
        E_bulk = λ_B B²  (regularizer)

    The overshoot shell places ρ > 1 where v² is large, breaking orthogonality.
    """

    def __init__(
        self,
        beta,
        w,
        lam,
        lambda_B=0.01,
        epsilon_shell=0.1,
        r_min=0.01,
        r_max=10.0,
        num_theta=50,
        R_c_leptons=None,
    ):
        """
        Parameters
        ----------
        beta : float
            Vacuum stiffness
        w : float
            Boundary layer thickness (absolute scale)
        lam : float
            Gradient energy coefficient
        lambda_B : float
            Bulk regularizer coefficient (default 0.01)
        epsilon_shell : float
            Shell width parameter (default 0.1)
        """
        self.beta = beta
        self.w = w
        self.lam = lam
        self.lambda_B = lambda_B
        self.epsilon_shell = epsilon_shell
        self.rho_vac = RHO_VAC

        # Build smart radial grid
        if R_c_leptons is None:
            R_c_leptons = [0.13, 0.50, 0.88]  # muon, tau, electron

        self.r = build_smart_radial_grid(
            r_min=r_min,
            r_max=r_max,
            w=w,
            R_c_leptons=R_c_leptons,
            dr_fine_factor=25.0,
            dr_coarse=0.02,
        )

        # Angular grid
        self.theta = np.linspace(0.01, np.pi - 0.01, num_theta)
        self.dtheta = self.theta[1] - self.theta[0]

    def circulation_energy(self, R, U, A, B):
        """
        Circulation kinetic energy with overshoot:

        E_circ = ∫ ½ρ(r) v²(r) d³r

        Now ρ(r) includes overshoot shell, so E_circ is profile-sensitive.
        """
        stream = HillVortexStreamFunction(R, U)
        density = DensityBoundaryLayerWithOvershoot(
            R - self.w, self.w, A, B, self.rho_vac, self.epsilon_shell
        )

        E_circ = 0.0
        for theta in self.theta:
            v_r, v_theta = stream.velocity_components(self.r, theta)
            v_squared = v_r**2 + v_theta**2
            rho_actual = density.rho(self.r)
            integrand = 0.5 * rho_actual * v_squared * self.r**2 * np.sin(theta)
            E_circ += simps(integrand, x=self.r) * self.dtheta

        E_circ *= 2 * np.pi

        return E_circ

    def stabilization_energy(self, R_c, A, B):
        """
        Stabilization energy: E_stab = β ∫ (Δρ)² d³r

        Now includes overshoot contribution.
        """
        density = DensityBoundaryLayerWithOvershoot(
            R_c, self.w, A, B, self.rho_vac, self.epsilon_shell
        )
        delta_rho = density.delta_rho(self.r)

        integrand = delta_rho**2 * self.r**2
        integral = simps(integrand, x=self.r)

        E_stab = self.beta * 4 * np.pi * integral

        return E_stab

    def gradient_energy(self, R_c, A, B):
        """
        Gradient energy: E_grad = λ ∫ |∇ρ|² d³r

        Now includes overshoot contribution.
        """
        density = DensityBoundaryLayerWithOvershoot(
            R_c, self.w, A, B, self.rho_vac, self.epsilon_shell
        )

        # Compute ∇ρ (radial component, spherical symmetry)
        rho_vals = density.rho(self.r)
        grad_rho = np.gradient(rho_vals, self.r)

        integrand = grad_rho**2 * self.r**2
        integral = simps(integrand, x=self.r)

        E_grad = self.lam * 4 * np.pi * integral

        return E_grad

    def bulk_regularizer(self, B):
        """
        Bulk regularizer: E_bulk = λ_B B²

        Prevents B from becoming a universal knob.
        Small λ_B → only matters if B grows large.
        """
        return self.lambda_B * B**2

    def total_energy(self, R_c, U, A, B):
        """
        Total energy with overshoot:

        E_total = E_circ - E_stab + E_grad + E_bulk

        Parameters
        ----------
        R_c : float
            Core radius
        U : float
            Circulation strength
        A : float
            Deficit amplitude
        B : float
            Overshoot shell amplitude

        Returns
        -------
        E_total, E_circ, E_stab, E_grad, E_bulk
        """
        R = R_c + self.w  # Vortex radius

        E_circ = self.circulation_energy(R, U, A, B)
        E_stab = self.stabilization_energy(R_c, A, B)
        E_grad = self.gradient_energy(R_c, A, B)
        E_bulk = self.bulk_regularizer(B)

        E_total = E_circ - E_stab + E_grad + E_bulk

        return E_total, E_circ, E_stab, E_grad, E_bulk


# ========================================================================
# Quick test
# ========================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("OVERSHOOT SHELL v0 - BASIC TEST")
    print("=" * 70)
    print()

    # Test parameters
    beta = 3.15
    w = 0.020
    lam = 0.006653
    lambda_B = 0.01

    energy_calc = LeptonEnergyWithOvershoot(beta=beta, w=w, lam=lam, lambda_B=lambda_B)

    # Test point: muon-like
    R_c = 0.30
    U = 0.05
    A = 0.99

    print("Test: Muon-like parameters with varying B")
    print(f"  R_c = {R_c}, U = {U}, A = {A}")
    print()
    print(f"{'B':<8} {'E_circ':<12} {'E_stab':<12} {'E_bulk':<12} {'E_total':<12} {'I=E_circ/U²':<12}")
    print("-" * 70)

    for B in [0.0, 0.25, 0.5, 1.0]:
        E_total, E_circ, E_stab, E_grad, E_bulk = energy_calc.total_energy(R_c, U, A, B)
        I = E_circ / (U**2)

        print(f"{B:<8.2f} {E_circ:<12.6f} {E_stab:<12.6f} {E_bulk:<12.6f} {E_total:<12.6f} {I:<12.2f}")

    print()
    print("✓ If I changes with B, profile-sensitivity restored!")
    print()
