#!/usr/bin/env python3
"""
V22 Enhanced Lepton Mass Solver - Hill Vortex (v2 with Unit Scaling)
=====================================================================

CRITICAL FIX: Unit conversion for β

The issue with v1: β = 3.1 from cosmology/nuclear is DIMENSIONLESS or in
cosmological/nuclear units, NOT directly in MeV for particle physics!

UNIT ANALYSIS:
==============

For the potential V(ρ) = β·(ρ - ρ_vac)²:

If ρ has dimensions [1/Length³] (density), then:
- V must have dimensions [Energy/Volume] = [Energy/Length³]
- Therefore β must have dimensions [Energy·Length³]

For particle physics at MeV scale with fm length:
- β_particle = β_dimensionless × (energy_scale) × (length_scale)³

SCALING STRATEGY:
=================

Option 1: Normalize everything to dimensionless units
  - Use electron Compton wavelength as unit length
  - Use electron mass as unit energy
  - β becomes dimensionless "stiffness parameter"

Option 2: Explicit unit conversion
  - β_nuclear ~ 3.1 in nuclear units
  - Convert: β_particle = β_nuclear × (conversion_factor)
  - Derive conversion factor from dimensional analysis

We'll use OPTION 1 (dimensionless formulation) for numerical stability.

[Rest of extensive documentation from v1...]
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

# ============================================================================
# DIMENSIONLESS FORMULATION
# ============================================================================

class DimensionlessUnits:
    """
    Dimensionless unit system for numerical stability.

    Length scale: λ_e = ℏc/(m_e c²) ~ 386 fm (electron Compton wavelength)
    Energy scale: m_e c² ~ 0.511 MeV (electron mass)

    In these units:
    - All lengths measured in units of λ_e
    - All energies measured in units of m_e
    - β is dimensionless stiffness parameter

    Physical interpretation:
    - β_dimensionless = 3.1 means "vacuum stiffness comparable to β from CMB/nuclear"
    - But numerical value adjusted for particle physics scale
    """

    def __init__(self, target_mass_MeV=ELECTRON_MASS_MEV):
        self.lambda_C = HBAR_C_MEV_FM / target_mass_MeV  # Compton wavelength (fm)
        self.m_target = target_mass_MeV  # Target mass (MeV)

    def length_to_dimensionless(self, length_fm):
        """Convert physical length (fm) to dimensionless units"""
        return length_fm / self.lambda_C

    def length_from_dimensionless(self, length_dimless):
        """Convert dimensionless length to physical (fm)"""
        return length_dimless * self.lambda_C

    def energy_to_dimensionless(self, energy_MeV):
        """Convert physical energy (MeV) to dimensionless units"""
        return energy_MeV / self.m_target

    def energy_from_dimensionless(self, energy_dimless):
        """Convert dimensionless energy to physical (MeV)"""
        return energy_dimless * self.m_target

# ============================================================================
# SIMPLIFIED 4-COMPONENT SOLVER (Dimensionless)
# ============================================================================

class SimplifiedHillVortexSolver:
    """
    Simplified solver using dimensionless units and focusing on core physics.

    KEY SIMPLIFICATION:
    Instead of full 4-component Hill vortex (which is complex), we use:
    - Single effective field ψ(r) representing density
    - Spherical symmetry (r-dependence only)
    - Hill-vortex-inspired boundary conditions
    - Quartic density potential from β

    This tests the CORE HYPOTHESIS:
    Can β ~ 3 (dimensionless stiffness) produce correct mass scale?
    """

    def __init__(self, beta_dimless=3.1, target_particle="electron",
                 num_points=150, r_max_lambda=10.0):
        """
        Parameters:
        - beta_dimless: Dimensionless vacuum stiffness (~ 3.1 from cosmology/nuclear)
        - target_particle: "electron", "muon", or "tau"
        - num_points: Grid points
        - r_max_lambda: Max radius in units of Compton wavelength
        """

        # Target mass
        masses = {
            "electron": ELECTRON_MASS_MEV,
            "muon": MUON_MASS_MEV,
            "tau": TAU_MASS_MEV
        }
        self.target_mass_MeV = masses[target_particle]
        self.particle = target_particle

        # Set up dimensionless units
        self.units = DimensionlessUnits(self.target_mass_MeV)

        # Dimensionless parameters
        self.beta = beta_dimless
        self.rho_vac = 1.0  # Normalized

        # Radial grid (dimensionless)
        self.r = np.linspace(0.01, r_max_lambda, num_points)
        self.dr = self.r[1] - self.r[0]

        # Volume element
        self.vol_element = 4 * np.pi * self.r**2

        print(f"Initialized Simplified Hill Vortex Solver ({target_particle})")
        print(f"  Dimensionless β = {beta_dimless:.2f}")
        print(f"  Length scale: λ = {self.units.lambda_C:.2f} fm")
        print(f"  Energy scale: m = {self.target_mass_MeV:.4f} MeV")
        print(f"  Grid: {num_points} points, r_max = {r_max_lambda:.1f} λ")

    def initialize_field(self):
        """
        Initialize with Hill-vortex-inspired profile.

        Use Gaussian that mimics density perturbation:
        ψ(r) ~ exp(-r²/2σ²) · (1 - r²/R²)

        This captures:
        - Smooth at origin
        - Depression in core
        - Decay at infinity
        """
        R_vortex = 1.0  # In units of λ
        sigma = 0.5

        psi = np.exp(-self.r**2 / (2 * sigma**2))

        # Modulate with Hill-like depression
        mask = self.r < R_vortex
        psi[mask] *= (1 - (self.r[mask] / R_vortex)**2)

        # Normalize
        rho = psi**2
        total_charge = simps(rho * self.vol_element, x=self.r)
        psi /= np.sqrt(total_charge + 1e-12)

        return psi

    def density_potential(self, rho):
        """
        V(ρ) = β·(ρ - ρ_vac)² [Quadratic in density]

        SIMPLIFIED from full V(ρ) = β·δρ² + V4·δρ⁴
        to focus on β term.

        If this works, we can add V4 later.
        """
        delta_rho = rho - self.rho_vac
        return self.beta * delta_rho**2

    def compute_energy(self, psi):
        """
        E[ψ] = ∫[½|∇ψ|² + V(|ψ|²)] · 4πr² dr

        Dimensionless energy functional.
        """
        # Kinetic energy
        grad_psi = np.gradient(psi, self.dr)
        E_kin = 0.5 * simps(grad_psi**2 * self.vol_element, x=self.r)

        # Potential energy
        rho = psi**2
        V_rho = self.density_potential(rho)
        E_pot = simps(V_rho * self.vol_element, x=self.r)

        return E_kin + E_pot

    def objective(self, psi):
        """Objective for optimizer (with boundary conditions)"""
        # Enforce boundary conditions
        psi[0] = psi[1]  # Regularity at r=0
        psi[-1] = 0.0   # Decay at infinity

        # Normalize to unit charge
        rho = psi**2
        total_charge = simps(rho * self.vol_element, x=self.r)
        psi /= np.sqrt(total_charge + 1e-12)

        return self.compute_energy(psi)

    def solve(self, max_iter=500):
        """Minimize energy to find stable soliton"""
        print("\nMinimizing energy functional...")

        psi_init = self.initialize_field()

        result = minimize(
            self.objective,
            psi_init,
            method='L-BFGS-B',
            options={'disp': True, 'maxiter': max_iter, 'ftol': 1e-12}
        )

        # Get final field
        psi_final = result.x
        psi_final[0] = psi_final[1]
        psi_final[-1] = 0.0

        # Normalize
        rho = psi_final**2
        total_charge = simps(rho * self.vol_element, x=self.r)
        psi_final /= np.sqrt(total_charge + 1e-12)

        # Compute final energy (dimensionless)
        E_dimless = self.compute_energy(psi_final)

        # Convert to physical units (MeV)
        E_MeV = self.units.energy_from_dimensionless(E_dimless)

        # Effective radius
        rho = psi_final**2
        r_weighted = self.r * rho
        R_eff_dimless = simps(r_weighted * self.vol_element, x=self.r) / (total_charge + 1e-12)
        R_eff_fm = self.units.length_from_dimensionless(R_eff_dimless)

        return {
            'energy_dimensionless': E_dimless,
            'energy_MeV': E_MeV,
            'R_eff_fm': R_eff_fm,
            'total_charge': total_charge,
            'converged': result.success,
            'iterations': result.nit,
            'psi_final': psi_final
        }

# ============================================================================
# TEST WITH MULTIPLE BETA VALUES
# ============================================================================

def scan_beta_values():
    """
    Scan different β values to find which produces correct electron mass.

    This will tell us:
    - What β_particle value is needed
    - How it relates to β_cosmology = 3.1
    - Whether unification is possible with scaling
    """
    print("=" * 80)
    print("SCANNING β VALUES TO FIND ELECTRON MASS")
    print("=" * 80)
    print()

    # Test range of β values
    beta_values = [0.001, 0.01, 0.1, 0.5, 1.0, 3.1, 10.0, 100.0, 1000.0]

    results_scan = []

    for beta in beta_values:
        print(f"\n{'='*60}")
        print(f"Testing β = {beta}")
        print(f"{'='*60}")

        solver = SimplifiedHillVortexSolver(
            beta_dimless=beta,
            target_particle="electron",
            num_points=100,
            r_max_lambda=10.0
        )

        results = solver.solve(max_iter=200)

        error_MeV = abs(results['energy_MeV'] - ELECTRON_MASS_MEV)
        accuracy = (1 - error_MeV / ELECTRON_MASS_MEV) * 100

        print(f"\nResult:")
        print(f"  Energy: {results['energy_MeV']:.6f} MeV")
        print(f"  Target: {ELECTRON_MASS_MEV:.6f} MeV")
        print(f"  Error:  {error_MeV:.6f} MeV")
        print(f"  Accuracy: {accuracy:.2f}%")

        results_scan.append({
            'beta': beta,
            'energy_MeV': results['energy_MeV'],
            'error_MeV': error_MeV,
            'accuracy_percent': accuracy
        })

    # Find best β
    best = min(results_scan, key=lambda x: x['error_MeV'])

    print("\n" + "=" * 80)
    print("SCAN RESULTS SUMMARY")
    print("=" * 80)
    print(f"\nBest β: {best['beta']}")
    print(f"  Energy: {best['energy_MeV']:.6f} MeV")
    print(f"  Accuracy: {best['accuracy_percent']:.2f}%")
    print()

    # Comparison with β = 3.1
    beta_3_1_result = next(r for r in results_scan if abs(r['beta'] - 3.1) < 0.1)

    print("β = 3.1 (from cosmology/nuclear):")
    print(f"  Energy: {beta_3_1_result['energy_MeV']:.6f} MeV")
    print(f"  Accuracy: {beta_3_1_result['accuracy_percent']:.2f}%")
    print()

    # Scaling factor
    if best['beta'] != 0:
        scaling_factor = best['beta'] / 3.1
        print(f"Scaling factor: β_particle / β_cosmology = {scaling_factor:.2e}")
        print()

        if 0.1 < scaling_factor < 10:
            print("✅ PROMISING: β values are within same order of magnitude!")
            print("   May just need unit conversion or modest scaling")
        else:
            print("⚠️  Large scaling factor suggests fundamental scale separation")

    return results_scan

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    results_scan = scan_beta_values()

    # Save results
    output_dir = Path("/home/tracy/development/QFD_SpectralGap/V22_Lepton_Analysis/results")
    output_dir.mkdir(exist_ok=True, parents=True)

    # Convert numpy types for JSON
    results_json = []
    for r in results_scan:
        results_json.append({k: (float(v) if isinstance(v, (np.floating, np.integer))
                                else v) for k, v in r.items()})

    with open(output_dir / "beta_scan_results.json", 'w') as f:
        json.dump(results_json, f, indent=2)

    print(f"\nResults saved to: {output_dir / 'beta_scan_results.json'}")
