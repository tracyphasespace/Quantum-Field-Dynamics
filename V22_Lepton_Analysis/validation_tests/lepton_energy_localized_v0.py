#!/usr/bin/env python3
"""
Lepton Energy Functional with Localized Vortex v0

Purpose: Break spatial orthogonality by localizing the Hill vortex flow,
         forcing kinetic energy into the region where ρ(r) varies.

MINIMAL CHANGE (v0):
  - Add velocity envelope: v_eff(r) = v_Hill(r) × g(r; R_v, p)
  - g(r) = exp[-(r/R_v)^p] with p=8 (steep suppression)
  - NO new fit parameter: R_v = k × R_shell (tied, sweep k)

This forces E_circ integral to be dominated by r < R_v where ρ ≠ 1,
making the functional profile-sensitive.

Diagnostics:
  - F_inner: fraction of E_circ from r < R_cut (structured region)
  - I sensitivity: ΔI/I when varying A (or other structure parameter)
"""

import numpy as np
from scipy.integrate import simps
from typing import Tuple

from lepton_energy_boundary_layer import (
    build_smart_radial_grid,
    HillVortexStreamFunction,
    DensityBoundaryLayer,
    RHO_VAC,
)


class LocalizedHillVortex(HillVortexStreamFunction):
    """
    Hill vortex with localization envelope to suppress external flow.

    v_eff(r, θ) = v_Hill(r, θ) × g(r; R_v, p)

    where g(r) = exp[-(r/R_v)^p] with p=8 (default, steep but smooth)
    """

    def __init__(self, R, U, R_v, p=8):
        """
        Parameters
        ----------
        R : float
            Hill vortex radius
        U : float
            Circulation strength
        R_v : float
            Localization radius (cutoff scale)
        p : int
            Envelope steepness (default 8, very steep)
        """
        super().__init__(R, U)
        self.R_v = float(R_v)
        self.p = int(p)

    def localization_envelope(self, r):
        """
        Smooth exponential envelope: g(r) = exp[-(r/R_v)^p]

        With p=8, this is very steep but C^∞ smooth.
        """
        return np.exp(-((r / self.R_v) ** self.p))

    def velocity_components(self, r, theta):
        """
        Velocity components with localization:

        v_r_eff = v_r × g(r)
        v_θ_eff = v_θ × g(r)
        """
        # Get Hill vortex velocities
        v_r, v_theta = super().velocity_components(r, theta)

        # Apply localization envelope
        g = self.localization_envelope(r)

        return v_r * g, v_theta * g


class LeptonEnergyLocalized:
    """
    Energy functional with localized Hill vortex.

    E_total = E_circ - E_stab + E_grad

    where:
        E_circ = ∫ ½ρ(r) v_eff²(r) d³r  (now localized, profile-sensitive)
        E_stab = β ∫ (Δρ)² d³r
        E_grad = λ ∫ |∇ρ|² d³r

    Diagnostics:
        - F_inner: fraction of E_circ from r < R_cut
        - I = E_circ/U²: should vary with A if profile-sensitive
    """

    def __init__(
        self,
        beta,
        w,
        lam,
        k_localization=1.5,
        p_envelope=8,
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
            Boundary layer thickness
        lam : float
            Gradient energy coefficient
        k_localization : float
            Localization factor: R_v = k × R_shell (default 1.5)
        p_envelope : int
            Envelope steepness (default 8)
        """
        self.beta = beta
        self.w = w
        self.lam = lam
        self.k_localization = k_localization
        self.p_envelope = p_envelope
        self.rho_vac = RHO_VAC

        # Build smart radial grid
        if R_c_leptons is None:
            R_c_leptons = [0.13, 0.50, 0.88]

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

    def circulation_energy_with_diagnostics(
        self, R, U, A, R_c=None
    ) -> Tuple[float, float, float]:
        """
        Compute circulation energy with localization and diagnostics.

        Returns
        -------
        E_circ : float
            Total circulation energy
        F_inner : float
            Fraction of E_circ from r < R_cut (structured region)
        I : float
            Prefactor I = E_circ/U²
        """
        # Localization radius
        R_shell = R  # Hill boundary
        R_v = self.k_localization * R_shell

        # Cutoff for "inner" diagnostic (structured region)
        if R_c is not None:
            R_cut = R_c + self.w
        else:
            R_cut = R_shell

        # Create localized vortex and density
        stream = LocalizedHillVortex(R, U, R_v, self.p_envelope)
        density = DensityBoundaryLayer(R - self.w, self.w, A, self.rho_vac)

        # Integrate over angles
        E_circ_total = 0.0
        E_circ_inner = 0.0

        for theta in self.theta:
            v_r, v_theta = stream.velocity_components(self.r, theta)
            v_squared = v_r**2 + v_theta**2
            rho_actual = density.rho(self.r)

            integrand = 0.5 * rho_actual * v_squared * self.r**2 * np.sin(theta)

            # Total energy
            E_circ_total += simps(integrand, x=self.r) * self.dtheta

            # Inner energy (r < R_cut only)
            mask_inner = self.r <= R_cut
            if np.any(mask_inner):
                integrand_inner = integrand.copy()
                integrand_inner[~mask_inner] = 0.0
                E_circ_inner += simps(integrand_inner, x=self.r) * self.dtheta

        E_circ_total *= 2 * np.pi
        E_circ_inner *= 2 * np.pi

        # Diagnostics
        F_inner = E_circ_inner / E_circ_total if E_circ_total > 0 else 0.0
        I = E_circ_total / (U**2) if U > 0 else 0.0

        return E_circ_total, F_inner, I

    def circulation_energy(self, R, U, A):
        """Simple interface (no diagnostics)"""
        E_circ, _, _ = self.circulation_energy_with_diagnostics(R, U, A)
        return E_circ

    def stabilization_energy(self, R_c, A):
        """Stabilization energy (unchanged)"""
        density = DensityBoundaryLayer(R_c, self.w, A, self.rho_vac)
        delta_rho = density.delta_rho(self.r)

        integrand = delta_rho**2 * self.r**2
        integral = simps(integrand, x=self.r)

        return self.beta * 4 * np.pi * integral

    def gradient_energy(self, R_c, A):
        """Gradient energy (unchanged)"""
        density = DensityBoundaryLayer(R_c, self.w, A, self.rho_vac)

        rho_vals = density.rho(self.r)
        grad_rho = np.gradient(rho_vals, self.r)

        integrand = grad_rho**2 * self.r**2
        integral = simps(integrand, x=self.r)

        return self.lam * 4 * np.pi * integral

    def total_energy(self, R_c, U, A):
        """
        Total energy: E_total = E_circ - E_stab + E_grad

        Returns
        -------
        E_total, E_circ, E_stab, E_grad
        """
        R = R_c + self.w

        E_circ = self.circulation_energy(R, U, A)
        E_stab = self.stabilization_energy(R_c, A)
        E_grad = self.gradient_energy(R_c, A)

        E_total = E_circ - E_stab + E_grad

        return E_total, E_circ, E_stab, E_grad


# ========================================================================
# Run 1A: Localization Sensitivity Sweep
# ========================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("RUN 1A: LOCALIZATION SENSITIVITY SWEEP")
    print("=" * 70)
    print()

    # Test parameters (muon-like)
    beta = 3.15
    w = 0.020
    lam = 0.006653

    R_c = 0.30
    U = 0.05
    R_shell = R_c + w

    print(f"Fixed parameters: R_c={R_c}, U={U}, R_shell={R_shell:.3f}")
    print()
    print("Sweep: k ∈ {1.0, 1.5, 2.0, 3.0}, A ∈ {0.70, 0.99}")
    print()

    # k sweep
    k_values = [1.0, 1.5, 2.0, 3.0]
    A_values = [0.70, 0.99]

    print("=" * 90)
    print(f"{'k':<8} {'R_v':<10} {'A':<8} {'I':<12} {'F_inner':<12} {'ΔI/I (%)':<12}")
    print("=" * 90)

    results_sweep = []

    for k in k_values:
        # Create energy calculator with this k
        energy_calc = LeptonEnergyLocalized(
            beta=beta, w=w, lam=lam, k_localization=k, p_envelope=8
        )

        R_v = k * R_shell

        I_results = []
        F_inner_results = []

        for A in A_values:
            E_circ, F_inner, I = energy_calc.circulation_energy_with_diagnostics(
                R_shell, U, A, R_c=R_c
            )

            I_results.append(I)
            F_inner_results.append(F_inner)

        # Compute sensitivity
        I_low = I_results[0]  # A=0.70
        I_high = I_results[1]  # A=0.99
        delta_I_pct = 100 * abs(I_high - I_low) / I_low if I_low > 0 else 0

        # Display for each A
        for i, A in enumerate(A_values):
            I = I_results[i]
            F_inner = F_inner_results[i]

            if i == 0:
                # First row: show delta
                print(f"{k:<8.1f} {R_v:<10.3f} {A:<8.2f} {I:<12.2f} {F_inner:<12.4f} {delta_I_pct:<12.2f}")
            else:
                # Second row: no delta (already shown)
                print(f"{'':<8} {'':<10} {A:<8.2f} {I:<12.2f} {F_inner:<12.4f} {'':<12}")

        print("-" * 90)

        results_sweep.append({
            "k": k,
            "R_v": R_v,
            "F_inner_avg": np.mean(F_inner_results),
            "delta_I_pct": delta_I_pct,
            "pass_F_inner": np.mean(F_inner_results) >= 0.5,
            "pass_sensitivity": delta_I_pct >= 1.0,
        })

    print("=" * 90)
    print()

    # Decision
    print("=" * 70)
    print("ACCEPTANCE CRITERIA")
    print("=" * 70)
    print()

    print(f"{'k':<8} {'F_inner≥0.5':<15} {'ΔI/I≥1%':<15} {'PASS':<10}")
    print("-" * 70)

    for res in results_sweep:
        pass_overall = res["pass_F_inner"] and res["pass_sensitivity"]

        print(
            f"{res['k']:<8.1f} "
            f"{'✓' if res['pass_F_inner'] else '✗':<15} "
            f"{'✓' if res['pass_sensitivity'] else '✗':<15} "
            f"{'PASS' if pass_overall else 'FAIL':<10}"
        )

    print()

    # Recommend best k
    passing = [r for r in results_sweep if r["pass_F_inner"] and r["pass_sensitivity"]]

    if passing:
        # Choose smallest k that passes (most conservative)
        best = min(passing, key=lambda x: x["k"])
        print(f"✓ RECOMMENDED: k = {best['k']:.1f}")
        print(f"    R_v = {best['R_v']:.3f}")
        print(f"    F_inner = {best['F_inner_avg']:.2%}")
        print(f"    ΔI/I = {best['delta_I_pct']:.2f}%")
        print()
        print("Proceed to Run 2 (e,μ regression) with this k fixed.")
    else:
        print("✗ NO k PASSES BOTH CRITERIA")
        print()
        print("Options:")
        print("  1. Try larger k (e.g., k=5 or k=10)")
        print("  2. Try steeper envelope (p=12 or p=16)")
        print("  3. Pivot to vacuum-subtraction approach")
        print()

        # Show which criterion failed
        if not any(r["pass_F_inner"] for r in results_sweep):
            print("Issue: F_inner < 0.5 for all k")
            print("  → External flow still dominates even with localization")
        if not any(r["pass_sensitivity"] for r in results_sweep):
            print("Issue: ΔI/I < 1% for all k")
            print("  → Functional still profile-insensitive")

    print()
    print("=" * 70)
