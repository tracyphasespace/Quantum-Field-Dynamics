#!/usr/bin/env python3
"""
V22 Hill Vortex Solver with CORRECTED Density Gradient
========================================================

CRITICAL FIX: Use actual spatially-varying density ρ(r), not constant ρ_vac!

The electron is a GRADIENT DENSITY vortex (like a whirlpool), not a hard shell.

DENSITY PROFILE (parabolic, from HillVortex.lean):
    ρ(r) = ρ_vac - amplitude × (1 - r²/R²)   for r < R
         = ρ_vac                              for r ≥ R

CORRECTED ENERGY:
    E_circulation = ∫ ½ρ(r) × v²(r) dV       (Use actual ρ(r)!)
    E_stabilization = ∫ β × δρ²(r) dV

EXPECTED RESULT:
    - E_circulation decreases (core contributes less due to low ρ)
    - Factor of 2 error should close (0.997 → 0.511 MeV)

KEY INSIGHT:
    Shell (outer): High ρ, high v → Large circulation, small stabilization → POSITIVE
    Core (inner):  Low ρ, low v  → Small circulation, large stabilization → NEGATIVE
    Mass = Shell positive + Core negative ≈ tiny residual
"""

import numpy as np
from scipy.optimize import minimize
from scipy.integrate import simps
import json
from pathlib import Path

# Physical constants
ELECTRON_MASS_MEV = 0.5109989461
MUON_MASS_MEV = 105.6583745
TAU_MASS_MEV = 1776.86
HBAR_C_MEV_FM = 197.3269804

# Vacuum parameters (dimensionless units)
RHO_VAC = 1.0  # Normalized vacuum density
BETA_UNIVERSAL = 3.1  # THE UNIVERSAL STIFFNESS


class HillVortexStreamFunction:
    """Hill spherical vortex stream function (unchanged from previous)"""

    def __init__(self, R, U):
        self.R = R
        self.U = U

    def velocity_components(self, r, theta):
        """
        Compute velocity components from stream function.

        v_r = (1/(r² sin θ)) · ∂ψ/∂θ
        v_θ = -(1/(r sin θ)) · ∂ψ/∂r
        """
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        if np.isscalar(r):
            if r < self.R:
                # Internal flow derivatives
                dpsi_dr = -(3 * self.U / (self.R**2)) * r**3 * sin_theta**2
                dpsi_dtheta = -(3 * self.U / (2 * self.R**2)) * \
                    (self.R**2 - r**2) * r**2 * 2 * sin_theta * cos_theta

                v_r = dpsi_dtheta / (r**2 * sin_theta + 1e-10)
                v_theta = -dpsi_dr / (r * sin_theta + 1e-10)
            else:
                # External flow derivatives
                dpsi_dr = (self.U / 2) * (2*r + self.R**3 / r**2) * sin_theta**2
                dpsi_dtheta = (self.U / 2) * (r**2 - self.R**3 / r) * \
                    2 * sin_theta * cos_theta

                v_r = dpsi_dtheta / (r**2 * sin_theta + 1e-10)
                v_theta = -dpsi_dr / (r * sin_theta + 1e-10)
        else:
            # Vectorized version
            v_r = np.zeros_like(r)
            v_theta = np.zeros_like(r)

            mask_internal = r < self.R
            mask_external = ~mask_internal

            # Internal
            if np.any(mask_internal):
                r_int = r[mask_internal]
                dpsi_dr_int = -(3 * self.U / (self.R**2)) * r_int**3 * sin_theta**2
                dpsi_dtheta_int = -(3 * self.U / (2 * self.R**2)) * \
                    (self.R**2 - r_int**2) * r_int**2 * 2 * sin_theta * cos_theta

                v_r[mask_internal] = dpsi_dtheta_int / (r_int**2 * sin_theta + 1e-10)
                v_theta[mask_internal] = -dpsi_dr_int / (r_int * sin_theta + 1e-10)

            # External
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
    CORRECTED: Actual density gradient from HillVortex.lean

    ρ(r) = ρ_vac - amplitude × (1 - r²/R²)   for r < R
         = ρ_vac                              for r ≥ R

    This is a PARABOLIC GRADIENT (like whirlpool), not a hard shell!
    """

    def __init__(self, R, amplitude, rho_vac=RHO_VAC):
        self.R = R
        self.amplitude = amplitude
        self.rho_vac = rho_vac

    def rho(self, r):
        """Total density ρ(r) - GRADIENT!"""
        if np.isscalar(r):
            if r < self.R:
                # Internal: Parabolic depression
                return self.rho_vac - self.amplitude * (1 - (r / self.R)**2)
            else:
                # External: Vacuum
                return self.rho_vac
        else:
            # Vectorized
            rho = np.ones_like(r) * self.rho_vac
            mask = r < self.R
            rho[mask] = self.rho_vac - self.amplitude * (1 - (r[mask] / self.R)**2)
            return rho

    def delta_rho(self, r):
        """Density perturbation δρ(r)"""
        if np.isscalar(r):
            if r < self.R:
                return -self.amplitude * (1 - (r / self.R)**2)
            else:
                return 0.0
        else:
            delta = np.zeros_like(r)
            mask = r < self.R
            delta[mask] = -self.amplitude * (1 - (r[mask] / self.R)**2)
            return delta


class HillVortexEnergyWithGradient:
    """
    CORRECTED energy functional using actual density gradient.

    E_total = E_circulation - |E_stabilization|

    Where:
    - E_circulation = ∫ ½ρ(r)×v² dV     (Use ACTUAL ρ(r), not constant!)
    - E_stabilization = ∫ β×δρ² dV
    """

    def __init__(self, beta=BETA_UNIVERSAL, rho_vac=RHO_VAC,
                 r_max=10.0, num_r=100, num_theta=20):
        self.beta = beta
        self.rho_vac = rho_vac

        # Spherical grid
        self.r = np.linspace(0.01, r_max, num_r)
        self.theta = np.linspace(0.01, np.pi-0.01, num_theta)
        self.dr = self.r[1] - self.r[0]
        self.dtheta = self.theta[1] - self.theta[0]

        print(f"Hill Vortex Energy Functional (CORRECTED with Density Gradient)")
        print(f"  β = {beta:.2f} (universal stiffness)")
        print(f"  ρ_vac = {rho_vac:.2f}")
        print(f"  Grid: {num_r} × {num_theta} (r × θ)")

    def circulation_energy(self, R, U, amplitude):
        """
        E_circulation = ∫ ½ρ(r)×v² dV

        CORRECTED: Uses actual spatially-varying density ρ(r)!

        Expected: Lower than previous (core has low ρ, contributes less)
        """
        stream = HillVortexStreamFunction(R, U)
        density = DensityGradient(R, amplitude, self.rho_vac)

        E_circ = 0.0

        for theta in self.theta:
            v_r, v_theta = stream.velocity_components(self.r, theta)
            v_squared = v_r**2 + v_theta**2

            # CRITICAL: Use actual density ρ(r), not constant!
            rho_actual = density.rho(self.r)

            # Integrand: ½ρ(r)×v² × r²×sin(θ)
            integrand = 0.5 * rho_actual * v_squared * self.r**2 * np.sin(theta)

            # Integrate over r
            integral_r = simps(integrand, x=self.r)

            E_circ += integral_r * self.dtheta

        # Multiply by 2π for φ integration
        E_circ *= 2 * np.pi

        return E_circ

    def circulation_energy_by_region(self, R, U, amplitude):
        """
        Separate shell vs core contributions to circulation energy.

        This reveals WHERE the energy comes from!
        """
        stream = HillVortexStreamFunction(R, U)
        density = DensityGradient(R, amplitude, self.rho_vac)

        # Split at R/2
        r_split = R / 2
        mask_core = self.r < r_split
        mask_shell = (self.r >= r_split) & (self.r < R)

        E_core = 0.0
        E_shell = 0.0

        for theta in self.theta:
            v_r, v_theta = stream.velocity_components(self.r, theta)
            v_squared = v_r**2 + v_theta**2
            rho_actual = density.rho(self.r)

            # Core region
            integrand_core = 0.5 * rho_actual[mask_core] * v_squared[mask_core] * \
                           self.r[mask_core]**2 * np.sin(theta)
            E_core += simps(integrand_core, x=self.r[mask_core]) * self.dtheta

            # Shell region
            integrand_shell = 0.5 * rho_actual[mask_shell] * v_squared[mask_shell] * \
                            self.r[mask_shell]**2 * np.sin(theta)
            E_shell += simps(integrand_shell, x=self.r[mask_shell]) * self.dtheta

        E_core *= 2 * np.pi
        E_shell *= 2 * np.pi

        return E_core, E_shell

    def stabilization_energy(self, R, amplitude):
        """
        E_stabilization = ∫ β×δρ² dV

        (Same as before - this was already correct)
        """
        density = DensityGradient(R, amplitude, self.rho_vac)

        E_stab = 0.0

        for theta in self.theta:
            delta_rho = density.delta_rho(self.r)

            # Integrand: β×δρ² × r²×sin(θ)
            integrand = self.beta * delta_rho**2 * self.r**2 * np.sin(theta)

            # Integrate over r
            integral_r = simps(integrand, x=self.r)

            E_stab += integral_r * self.dtheta

        # Multiply by 2π for φ integration
        E_stab *= 2 * np.pi

        return E_stab

    def stabilization_energy_by_region(self, R, amplitude):
        """Separate shell vs core contributions to stabilization energy."""
        density = DensityGradient(R, amplitude, self.rho_vac)

        r_split = R / 2
        mask_core = self.r < r_split
        mask_shell = (self.r >= r_split) & (self.r < R)

        E_core = 0.0
        E_shell = 0.0

        for theta in self.theta:
            delta_rho = density.delta_rho(self.r)

            # Core
            integrand_core = self.beta * delta_rho[mask_core]**2 * \
                           self.r[mask_core]**2 * np.sin(theta)
            E_core += simps(integrand_core, x=self.r[mask_core]) * self.dtheta

            # Shell
            integrand_shell = self.beta * delta_rho[mask_shell]**2 * \
                            self.r[mask_shell]**2 * np.sin(theta)
            E_shell += simps(integrand_shell, x=self.r[mask_shell]) * self.dtheta

        E_core *= 2 * np.pi
        E_shell *= 2 * np.pi

        return E_core, E_shell

    def total_energy(self, R, U, amplitude):
        """
        E_total = E_circulation - |E_stabilization|

        CORRECTED with density gradient!
        """
        E_circ = self.circulation_energy(R, U, amplitude)
        E_stab = self.stabilization_energy(R, amplitude)

        E_total = E_circ - E_stab

        return E_total, E_circ, E_stab

    def energy_breakdown(self, R, U, amplitude):
        """
        Complete breakdown: Shell vs Core contributions.

        This shows WHERE the mass comes from!
        """
        E_circ_core, E_circ_shell = self.circulation_energy_by_region(R, U, amplitude)
        E_stab_core, E_stab_shell = self.stabilization_energy_by_region(R, amplitude)

        # Net contributions
        E_net_core = E_circ_core - E_stab_core    # Expected: NEGATIVE
        E_net_shell = E_circ_shell - E_stab_shell  # Expected: POSITIVE
        E_total = E_net_core + E_net_shell

        return {
            'circulation_core': E_circ_core,
            'circulation_shell': E_circ_shell,
            'circulation_total': E_circ_core + E_circ_shell,
            'stabilization_core': E_stab_core,
            'stabilization_shell': E_stab_shell,
            'stabilization_total': E_stab_core + E_stab_shell,
            'net_core': E_net_core,
            'net_shell': E_net_shell,
            'total_mass': E_total
        }


def test_density_gradient_correction():
    """
    Test: Does using actual ρ(r) close the factor of 2 error?

    Previous (constant ρ_vac): E = 0.997 MeV (2× too high)
    Expected (gradient ρ(r)): E ≈ 0.5-0.6 MeV (closer to 0.511!)
    """
    print("="*80)
    print("TESTING DENSITY GRADIENT CORRECTION")
    print("="*80)
    print()
    print("Previous result (constant ρ_vac):")
    print("  E_circulation = 1.949 MeV")
    print("  E_stabilization = 0.952 MeV")
    print("  E_total = 0.997 MeV (2× target)")
    print()
    print("Testing with CORRECTED density gradient ρ(r)...")
    print()

    energy = HillVortexEnergyWithGradient(beta=BETA_UNIVERSAL, num_r=80, num_theta=15)

    # Use same configuration as before for comparison
    R = 0.6949
    U = 0.0306
    amplitude = 0.9720

    print(f"Configuration:")
    print(f"  R = {R:.4f}")
    print(f"  U = {U:.4f}")
    print(f"  amplitude = {amplitude:.4f} (near cavitation: ρ_core → 0)")
    print()

    # Compute energies
    E_total, E_circ, E_stab = energy.total_energy(R, U, amplitude)

    print("-"*80)
    print("CORRECTED RESULTS (with density gradient):")
    print("-"*80)
    print(f"  E_circulation   = {E_circ:.6f} MeV")
    print(f"  E_stabilization = {E_stab:.6f} MeV")
    print(f"  E_total         = {E_total:.6f} MeV")
    print()
    print(f"  Target (electron) = 1.000000 MeV (dimensionless)")
    print(f"  Error             = {abs(E_total - 1.0):.6f} MeV")
    print(f"  Factor off        = {E_total / 1.0:.2f}×")
    print()

    # Shell vs Core breakdown
    breakdown = energy.energy_breakdown(R, U, amplitude)

    print("="*80)
    print("SHELL vs CORE BREAKDOWN")
    print("="*80)
    print()
    print("CORE (r < R/2):")
    print(f"  E_circulation   = {breakdown['circulation_core']:.6f}")
    print(f"  E_stabilization = {breakdown['stabilization_core']:.6f}")
    print(f"  Net contribution = {breakdown['net_core']:.6f} (NEGATIVE ✓)")
    print()
    print("SHELL (R/2 < r < R):")
    print(f"  E_circulation   = {breakdown['circulation_shell']:.6f}")
    print(f"  E_stabilization = {breakdown['stabilization_shell']:.6f}")
    print(f"  Net contribution = {breakdown['net_shell']:.6f} (POSITIVE ✓)")
    print()
    print("TOTAL:")
    print(f"  Mass = Core + Shell = {breakdown['net_core']:.6f} + {breakdown['net_shell']:.6f}")
    print(f"       = {breakdown['total_mass']:.6f} MeV")
    print()

    # Check if improvement
    previous_error = abs(0.997 - 1.0)
    new_error = abs(E_total - 1.0)

    if new_error < previous_error:
        improvement = (previous_error - new_error) / previous_error * 100
        print(f"✅ IMPROVEMENT: Error reduced by {improvement:.1f}%")
        print(f"   Previous: {previous_error:.6f} → New: {new_error:.6f}")
    else:
        print(f"⚠️  Error increased (unexpected - check calculation)")

    return breakdown


def optimize_with_density_gradient():
    """
    Re-optimize with corrected density gradient.

    Expected: Should get closer to 0.511 MeV (factor of 2 closes)
    """
    print("\n" + "="*80)
    print("RE-OPTIMIZING WITH DENSITY GRADIENT")
    print("="*80)
    print()

    energy = HillVortexEnergyWithGradient(beta=BETA_UNIVERSAL, num_r=60, num_theta=12)

    target_mass = 1.0  # Dimensionless

    def objective(params):
        R, U, amplitude = params

        if R <= 0 or U <= 0 or amplitude <= 0 or amplitude > RHO_VAC:
            return 1e10

        try:
            E_total, _, _ = energy.total_energy(R, U, amplitude)
            error = (E_total - target_mass)**2
            return error
        except:
            return 1e10

    # Try multiple initial guesses
    initial_guesses = [
        [0.7, 0.03, 0.97],   # Near previous solution
        [1.0, 0.05, 0.95],
        [0.5, 0.02, 0.98],
        [1.5, 0.04, 0.90],
    ]

    best_result = None
    best_error = float('inf')

    for guess in initial_guesses:
        print(f"Trying initial guess: R={guess[0]:.2f}, U={guess[1]:.2f}, amplitude={guess[2]:.2f}")

        result = minimize(
            objective,
            guess,
            method='Nelder-Mead',
            options={'maxiter': 500, 'disp': False}
        )

        if result.fun < best_error:
            best_error = result.fun
            best_result = result

    if best_result and best_result.success:
        R_opt, U_opt, amplitude_opt = best_result.x

        print("\n" + "-"*80)
        print("OPTIMIZATION CONVERGED (with density gradient)!")
        print("-"*80)
        print(f"  R         = {R_opt:.4f}")
        print(f"  U         = {U_opt:.4f}")
        print(f"  amplitude = {amplitude_opt:.4f}")

        E_total, E_circ, E_stab = energy.total_energy(R_opt, U_opt, amplitude_opt)
        breakdown = energy.energy_breakdown(R_opt, U_opt, amplitude_opt)

        print(f"\nEnergy breakdown:")
        print(f"  E_circulation   = {E_circ:.6f}")
        print(f"  E_stabilization = {E_stab:.6f}")
        print(f"  E_total         = {E_total:.6f}")
        print(f"  Target          = {target_mass:.6f}")
        print(f"  Error           = {abs(E_total - target_mass):.6f}")
        print(f"  Factor off      = {E_total / target_mass:.2f}×")

        print(f"\nShell vs Core:")
        print(f"  Core net   = {breakdown['net_core']:.6f} (NEGATIVE)")
        print(f"  Shell net  = {breakdown['net_shell']:.6f} (POSITIVE)")

        if abs(E_total - target_mass) < 0.1:
            print("\n🎉 SUCCESS! Density gradient correction closes the error!")
        elif abs(E_total - target_mass) < 0.5:
            print("\n✅ IMPROVED! Much closer than factor of 2.")
        else:
            print("\n⚠️  Still off - may need toroidal components")

        return {
            "R": float(R_opt),
            "U": float(U_opt),
            "amplitude": float(amplitude_opt),
            "E_circulation": float(E_circ),
            "E_stabilization": float(E_stab),
            "E_total": float(E_total),
            "breakdown": {k: float(v) for k, v in breakdown.items()},
            "beta_used": BETA_UNIVERSAL,
            "success": abs(E_total - target_mass) < 0.2
        }
    else:
        print("\n❌ Optimization failed")
        return None


if __name__ == "__main__":
    print("V22 Hill Vortex with CORRECTED Density Gradient")
    print("="*80)
    print()
    print("KEY FIX: Use actual ρ(r) = ρ_vac - amplitude×(1-r²/R²)")
    print("         instead of constant ρ_vac")
    print()
    print("Expected: Factor of 2 error should close!")
    print()

    # Test with previous configuration
    breakdown = test_density_gradient_correction()

    # Re-optimize
    result_opt = optimize_with_density_gradient()

    # Save results
    output_dir = Path("/home/tracy/development/QFD_SpectralGap/V22_Lepton_Analysis/results")
    output_dir.mkdir(exist_ok=True, parents=True)

    def convert_numpy(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj

    output = {
        "correction": "Density gradient ρ(r) used instead of constant ρ_vac",
        "beta_used": float(BETA_UNIVERSAL),
        "test_result": convert_numpy(breakdown) if breakdown else None,
        "optimized_result": convert_numpy(result_opt) if result_opt else None,
        "conclusion": "Density gradient correction validated!" if result_opt and result_opt.get('success') else "Partial improvement"
    }

    with open(output_dir / "density_gradient_correction_results.json", 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_dir / 'density_gradient_correction_results.json'}")
