#!/usr/bin/env python3
"""
Lepton Energy Functional with Localized Vortex v1

CORRECTED ENVELOPE (outside-only):
  g(r) = 1                           for r ≤ R_v
  g(r) = exp(-((r-R_v)/Δ_v)^p)      for r > R_v

This preserves the interior and boundary velocities (where v² is large)
while cutting the far-field tail that caused profile insensitivity.

Key differences from v0:
  - v0: exp(-(r/R_v)^p) suppressed boundary by exp(-1) ≈ 0.37
  - v1: g(r) = 1 at boundary, only damps beyond R_v
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


class LocalizedHillVortexV1(HillVortexStreamFunction):
    """
    Hill vortex with outside-only localization envelope.

    v_eff(r, θ) = v_Hill(r, θ) × g(r; R_v, Δ_v, p)

    where:
        g(r) = 1                           for r ≤ R_v
        g(r) = exp(-((r-R_v)/Δ_v)^p)      for r > R_v
    """

    def __init__(self, R, U, R_v, delta_v, p=6):
        """
        Parameters
        ----------
        R : float
            Hill vortex radius
        U : float
            Circulation strength
        R_v : float
            Localization radius (cutoff scale)
        delta_v : float
            Transition width (Δ_v)
        p : int
            Envelope steepness (default 6, smooth but effective)
        """
        super().__init__(R, U)
        self.R_v = float(R_v)
        self.delta_v = float(delta_v)
        self.p = int(p)

    def localization_envelope(self, r):
        """
        Outside-only exponential envelope:

        g(r) = 1                           for r ≤ R_v
        g(r) = exp(-((r-R_v)/Δ_v)^p)      for r > R_v
        """
        g = np.ones_like(r)
        mask = r > self.R_v
        if np.any(mask):
            g[mask] = np.exp(-(((r[mask] - self.R_v) / self.delta_v) ** self.p))
        return g

    def velocity_components(self, r, theta):
        """
        Velocity components with outside-only localization:

        v_r_eff = v_r × g(r)
        v_θ_eff = v_θ × g(r)
        """
        # Get Hill vortex velocities
        v_r, v_theta = super().velocity_components(r, theta)

        # Apply localization envelope (preserves r ≤ R_v)
        g = self.localization_envelope(r)

        return v_r * g, v_theta * g


class LeptonEnergyLocalizedV1:
    """
    Energy functional with outside-only localized Hill vortex.

    E_total = E_circ - E_stab + E_grad

    where:
        E_circ = ∫ ½ρ(r) v_eff²(r) d³r  (now localized outside R_v)
        E_stab = β ∫ (Δρ)² d³r
        E_grad = λ ∫ |∇ρ|² d³r

    Diagnostics:
        - F_inner: fraction of E_circ from r < R_v
        - I = E_circ/U²: should vary with A if profile-sensitive
    """

    def __init__(
        self,
        beta,
        w,
        lam,
        k_localization=1.5,
        delta_v_factor=0.25,
        p_envelope=6,
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
        delta_v_factor : float
            Transition width factor: Δ_v = delta_v_factor × R_v (default 0.25)
        p_envelope : int
            Envelope steepness (default 6)
        """
        self.beta = beta
        self.w = w
        self.lam = lam
        self.k_localization = k_localization
        self.delta_v_factor = delta_v_factor
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

    def circulation_energy_vacuum(self, R, U) -> float:
        """
        Compute circulation energy with VACUUM density (ρ ≡ 1).

        This is the reference state for renormalization:
        same velocity field, same localization, but no density structure.

        Returns
        -------
        E_circ_vac : float
            Vacuum circulation energy
        """
        # Localization radius
        R_v = self.k_localization * R
        delta_v = self.delta_v_factor * R_v

        # Create localized vortex (same as actual)
        stream = LocalizedHillVortexV1(R, U, R_v, delta_v, self.p_envelope)

        # Integrate over angles with ρ = 1 everywhere
        E_circ_vac = 0.0

        for theta in self.theta:
            v_r, v_theta = stream.velocity_components(self.r, theta)
            v_squared = v_r**2 + v_theta**2

            # Vacuum: ρ = 1 everywhere
            integrand = 0.5 * 1.0 * v_squared * self.r**2 * np.sin(theta)

            E_circ_vac += simps(integrand, x=self.r) * self.dtheta

        E_circ_vac *= 2 * np.pi

        return E_circ_vac

    def circulation_energy_with_diagnostics(
        self, R, U, A, R_c=None
    ) -> Tuple[float, float, float, float, float]:
        """
        Compute RENORMALIZED circulation energy with diagnostics.

        Returns
        -------
        dE_circ : float
            Excess circulation energy (ΔE_circ = E_circ[ρ] - E_circ[ρ=1])
        F_inner : float
            Fraction of E_circ_actual from r < R_v (localization boundary)
        I : float
            Prefactor I = E_circ_actual/U²
        E_circ_actual : float
            Actual circulation energy (for auditing)
        E_circ_vac : float
            Vacuum circulation energy (for auditing)
        """
        # Localization radius
        R_shell = R  # Hill boundary
        R_v = self.k_localization * R_shell
        delta_v = self.delta_v_factor * R_v

        # Create localized vortex and density
        stream = LocalizedHillVortexV1(R, U, R_v, delta_v, self.p_envelope)
        density = DensityBoundaryLayer(R - self.w, self.w, A, self.rho_vac)

        # Integrate over angles
        E_circ_actual = 0.0
        E_circ_inner = 0.0

        for theta in self.theta:
            v_r, v_theta = stream.velocity_components(self.r, theta)
            v_squared = v_r**2 + v_theta**2
            rho_actual = density.rho(self.r)

            integrand = 0.5 * rho_actual * v_squared * self.r**2 * np.sin(theta)

            # Total energy
            E_circ_actual += simps(integrand, x=self.r) * self.dtheta

            # Inner energy (r < R_v only, i.e., unmodulated region)
            mask_inner = self.r <= R_v
            if np.any(mask_inner):
                integrand_inner = integrand.copy()
                integrand_inner[~mask_inner] = 0.0
                E_circ_inner += simps(integrand_inner, x=self.r) * self.dtheta

        E_circ_actual *= 2 * np.pi
        E_circ_inner *= 2 * np.pi

        # Compute vacuum reference
        E_circ_vac = self.circulation_energy_vacuum(R, U)

        # Renormalized (excess) circulation energy
        dE_circ = E_circ_actual - E_circ_vac

        # Diagnostics
        F_inner = E_circ_inner / E_circ_actual if E_circ_actual > 0 else 0.0
        I = E_circ_actual / (U**2) if U > 0 else 0.0

        return dE_circ, F_inner, I, E_circ_actual, E_circ_vac

    def circulation_energy(self, R, U, A):
        """
        Simple interface (no diagnostics).

        Returns actual circulation energy (not excess).
        """
        _, _, _, E_circ, _ = self.circulation_energy_with_diagnostics(R, U, A)
        return E_circ

    def stabilization_energy(self, R_c, A):
        """
        Stabilization energy with co-localization.

        Apply same localization window g(r) to preserve electron cancellation regime.
        E_stab = β × 4π × ∫ g(r) × (Δρ)² r² dr
        """
        density = DensityBoundaryLayer(R_c, self.w, A, self.rho_vac)
        delta_rho = density.delta_rho(self.r)

        # Compute localization envelope at same R_v used for circulation
        R_shell = R_c + self.w
        R_v = self.k_localization * R_shell
        delta_v = self.delta_v_factor * R_v

        # Create localized vortex just to get g(r)
        stream = LocalizedHillVortexV1(R_shell, 1.0, R_v, delta_v, self.p_envelope)
        g = stream.localization_envelope(self.r)

        # Co-localized integrand
        integrand = g * delta_rho**2 * self.r**2
        integral = simps(integrand, x=self.r)

        return self.beta * 4 * np.pi * integral

    def gradient_energy(self, R_c, A):
        """
        Gradient energy with co-localization.

        Apply same localization window g(r) for Hamiltonian consistency.
        E_grad = λ × 4π × ∫ g(r) × |∇ρ|² r² dr
        """
        density = DensityBoundaryLayer(R_c, self.w, A, self.rho_vac)

        rho_vals = density.rho(self.r)
        grad_rho = np.gradient(rho_vals, self.r)

        # Compute localization envelope
        R_shell = R_c + self.w
        R_v = self.k_localization * R_shell
        delta_v = self.delta_v_factor * R_v

        stream = LocalizedHillVortexV1(R_shell, 1.0, R_v, delta_v, self.p_envelope)
        g = stream.localization_envelope(self.r)

        # Co-localized integrand
        integrand = g * grad_rho**2 * self.r**2
        integral = simps(integrand, x=self.r)

        return self.lam * 4 * np.pi * integral

    def total_energy(self, R_c, U, A):
        """
        Total energy with CORRECTED SIGN CONVENTION:

        E_total = E_circ + E_stab + E_grad

        All terms are positive-definite penalties:
        - E_circ: kinetic energy (½ρv²)
        - E_stab: stabilization penalty β∫(Δρ)²
        - E_grad: gradient penalty λ∫|∇ρ|²

        NO vacuum subtraction - not needed with correct signs.

        Returns
        -------
        E_total : float
            Total energy (all penalties add)
        E_circ : float
            Circulation energy
        E_stab : float
            Stabilization penalty
        E_grad : float
            Gradient penalty
        """
        R = R_c + self.w

        # Use actual E_circ (not vacuum-subtracted)
        # Get from diagnostics but discard excess energy
        _, _, _, E_circ, _ = self.circulation_energy_with_diagnostics(R, U, A)

        E_stab = self.stabilization_energy(R_c, A)
        E_grad = self.gradient_energy(R_c, A)

        # CORRECTED: All terms ADD (positive-definite penalties)
        E_total = E_circ + E_stab + E_grad

        return E_total, E_circ, E_stab, E_grad


# ========================================================================
# Run 1A: Localization Sensitivity Sweep (v1 - outside-only envelope)
# ========================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("RUN 1A v1: LOCALIZATION SENSITIVITY SWEEP")
    print("(Outside-only envelope)")
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
    print("Envelope: g(r) = 1 for r ≤ R_v, exp(-((r-R_v)/Δ_v)^p) for r > R_v")
    print("Settings: p=6, Δ_v=0.25×R_v")
    print()
    print("Sweep: k ∈ {1.0, 1.5, 2.0}, A ∈ {0.70, 0.99}")
    print()

    # k sweep
    k_values = [1.0, 1.5, 2.0]
    A_values = [0.70, 0.99]
    delta_v_factor = 0.25
    p = 6

    print("=" * 100)
    print(f"{'k':<8} {'R_v':<10} {'Δ_v':<10} {'A':<8} {'I':<12} {'F_inner':<12} {'ΔI/I (%)':<12}")
    print("=" * 100)

    results_sweep = []

    for k in k_values:
        # Create energy calculator with this k
        energy_calc = LeptonEnergyLocalizedV1(
            beta=beta, w=w, lam=lam, k_localization=k, delta_v_factor=delta_v_factor, p_envelope=p
        )

        R_v = k * R_shell
        delta_v = delta_v_factor * R_v

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
                print(f"{k:<8.1f} {R_v:<10.3f} {delta_v:<10.3f} {A:<8.2f} {I:<12.2f} {F_inner:<12.4f} {delta_I_pct:<12.2f}")
            else:
                # Second row: no delta (already shown)
                print(f"{'':<8} {'':<10} {'':<10} {A:<8.2f} {I:<12.2f} {F_inner:<12.4f} {'':<12}")

        print("-" * 100)

        results_sweep.append({
            "k": k,
            "R_v": R_v,
            "delta_v": delta_v,
            "F_inner_avg": np.mean(F_inner_results),
            "delta_I_pct": delta_I_pct,
            "pass_F_inner": np.mean(F_inner_results) >= 0.5,
            "pass_sensitivity": delta_I_pct >= 1.0,
        })

    print("=" * 100)
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
        print(f"    Δ_v = {best['delta_v']:.3f}")
        print(f"    F_inner = {best['F_inner_avg']:.2%}")
        print(f"    ΔI/I = {best['delta_I_pct']:.2f}%")
        print()
        print("Proceed to energy-balance precheck with this k.")
    else:
        print("✗ NO k PASSES BOTH CRITERIA")
        print()
        print("This would indicate outside-only localization insufficient.")
        print("Consider pivot to vacuum-subtraction approach.")

    print()
    print("=" * 70)
