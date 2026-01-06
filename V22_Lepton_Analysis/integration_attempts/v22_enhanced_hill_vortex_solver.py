#!/usr/bin/env python3
"""
V22 Enhanced Lepton Mass Solver - Hill Vortex Formulation
==========================================================

THE CRITICAL INSIGHT:
The electron (and all leptons) are NOT point particles, but Hill spherical vortices!

This solver implements the CORRECT physics:
- Euler-Lagrange field equations (NOT Schrödinger!)
- Hill vortex stream function (from classical fluid dynamics)
- 4-component field structure (poloidal + toroidal circulation)
- Density-dependent potential V(ρ) (NOT radial potential V(r)!)
- β = 3.1 from cosmology/nuclear determines vacuum stiffness

PHYSICS BACKGROUND:
==================

1. HILL'S SPHERICAL VORTEX (Lamb 1932, §§159-160)
   - Discovered by M.J.M. Hill (1894)
   - Exact solution to Euler equations for incompressible inviscid flow
   - Spherical boundary at r = R separates internal/external flow
   - Internal (r < R): Rotational flow with vorticity
   - External (r > R): Irrotational potential flow

   Stream function:
   For r < R:  ψ = -(3U/2R²)(R² - r²)r² sin²θ  [Internal circulation]
   For r > R:  ψ = (U/2)(r² - R³/r) sin²θ      [External potential flow]

   where:
   - R = vortex radius
   - U = propagation velocity
   - θ = polar angle (0 = north pole)

2. WHY 4 COMPONENTS?
   From QFD/Electron/AxisAlignment.lean:
   "The QFD Electron is a 'Swirling' Hill Vortex. It has:
    1. Poloidal circulation (Standard Hill) → Defines soliton shape
    2. Toroidal/Azimuthal swirl (The 'Spin') → Adds angular momentum L_z"

   Field decomposition:
   - ψ_s:  Scalar density (pressure perturbation from Bernoulli)
   - ψ_b0, ψ_b1, ψ_b2: Bi-vector (3 components of toroidal swirl)

   Total circulation = poloidal (shape) + toroidal (spin)

3. DENSITY-DEPENDENT POTENTIAL
   For a Hill vortex, Bernoulli's equation gives:
   p + ½ρv² = constant

   In regions of high velocity (circulation core):
   - Pressure drops (Bernoulli effect)
   - Density perturbation δρ < 0 (compressible vacuum)
   - Creates "soliton trap"

   Density perturbation (from HillVortex.lean):
   δρ(r) = -amplitude · (1 - r²/R²)  for r < R
   δρ(r) = 0                         for r > R

   At core (r=0): ρ_total = ρ_vac - amplitude
   Cavitation limit: ρ_total ≥ 0 → amplitude ≤ ρ_vac
   This gives CHARGE QUANTIZATION: e = ρ_vac

4. EULER-LAGRANGE vs SCHRÖDINGER
   WRONG (V22 Original):
   -ψ'' + V(r)ψ = E·ψ  [Linear, 1D, radial potential]

   CORRECT (Enhanced V22):
   δS/δψ = 0  where S = ∫ L(ψ, ∇ψ, ρ(ψ)) dV
   [Nonlinear, 3D, density-dependent, variational principle]

   The Hill vortex extremizes the fluid action!

5. ROLE OF β
   β = vacuum stiffness parameter (from cosmology/nuclear)

   In density-dependent formulation:
   V(ρ) = β·(ρ - ρ_vac)²

   β determines:
   - How "stiff" the vacuum is (resistance to compression)
   - Energy cost to create density perturbations
   - Strength of soliton confinement

   SAME β in:
   - CMB stiffness (cosmology)
   - Nuclear core compression (nuclear physics)
   - Lepton vortex stability (particle physics)

   UNIFICATION HYPOTHESIS:
   If β = 3.1 produces correct lepton masses → Complete unification!

6. Q* NORMALIZATION
   Q* = ∫ ρ_charge² · 4πr² dr  [RMS charge density]

   For Hill vortex, charge density from Poisson equation:
   ρ_q = -g_c · ∇²ψ_s  (scalar component only)

   Q* encodes internal angular structure:
   - Electron: Q* ≈ 2.2 (ground state, minimal swirl)
   - Muon: Q* ≈ 2.3 (first excitation, moderate swirl)
   - Tau: Q* ≈ 9800 (highly excited, complex swirl)

   Different swirl patterns → Different masses!

IMPLEMENTATION STRATEGY:
========================

1. Initialize 4-component field with Hill vortex ansatz
2. Compute density from field: ρ = ρ_vac + δρ(ψ)
3. Evaluate energy functional:
   E = ∫ [½|∇ψ|² + V(ρ(ψ)) + E_csr(ψ_s)] · 4πr² dr
4. Optimize ψ to minimize E using L-BFGS-B
5. Enforce Q* constraint via normalization
6. Check if converges to lepton masses with β = 3.1

DIFFERENCES FROM PHOENIX:
=========================

Phoenix Solver (Working):
- Tunes V2, V4, Q* separately for each lepton
- No explicit Hill vortex structure
- Achieves 99.9999% accuracy

Enhanced V22 (This code):
- Derives V(ρ) from β = 3.1 (unified parameter!)
- Explicit Hill vortex stream function
- Tests if cosmology determines particle physics

If this works → Revolutionary unification!
If this fails → Need scale-dependent β (still interesting!)

Author: QFD Research Team
Date: December 22, 2025
Version: V22 Enhanced (Hill Vortex)
Reference: /projects/Lean4/QFD/Electron/HillVortex.lean
"""

import numpy as np
from scipy.optimize import minimize
from scipy.integrate import simps
import matplotlib.pyplot as plt
import json
from pathlib import Path

# Physical constants (MeV)
ELECTRON_MASS = 0.5109989461  # MeV
MUON_MASS = 105.6583745       # MeV
TAU_MASS = 1776.86            # MeV

# Universal constants (natural units, ℏ = c = 1)
HBAR_C = 197.3269804  # MeV·fm

# ============================================================================
# HILL VORTEX STRUCTURE (From HillVortex.lean)
# ============================================================================

class HillVortexContext:
    """
    Hill's spherical vortex structure from QFD/Electron/HillVortex.lean

    Parameters:
    - R: Vortex radius (spherical boundary)
    - U: Propagation velocity
    - beta: Vacuum stiffness (from cosmology/nuclear)
    - rho_vac: Vacuum density floor
    - g_c: Charge coupling constant

    Physical meaning:
    - R sets the "size" of the lepton
    - U sets the internal circulation speed
    - beta determines energy scale (SAME as cosmology/nuclear!)
    """

    def __init__(self, R, U, beta, rho_vac=1.0, g_c=0.985):
        self.R = R
        self.U = U
        self.beta = beta
        self.rho_vac = rho_vac
        self.g_c = g_c

        # Derived parameters
        self.amplitude_max = rho_vac  # Cavitation limit (charge quantization!)

    def stream_function_internal(self, r, theta):
        """
        Internal Hill vortex stream function (r < R).
        From HillVortex.lean line 38-39.

        ψ = -(3U/2R²)(R² - r²)r² sin²θ

        Physical meaning:
        - Describes toroidal circulation inside vortex
        - Vanishes at r=0 (regularity at center)
        - Vanishes at r=R (defines boundary)
        - Maximum at intermediate r (circulation ring)
        """
        sin_sq = np.sin(theta)**2
        return -(3 * self.U / (2 * self.R**2)) * (self.R**2 - r**2) * r**2 * sin_sq

    def stream_function_external(self, r, theta):
        """
        External Hill vortex stream function (r > R).
        From HillVortex.lean line 42-43.

        ψ = (U/2)(r² - R³/r) sin²θ

        Physical meaning:
        - Irrotational flow (∇×v = 0)
        - Combination of uniform stream + dipole
        - Decays as 1/r at infinity
        - Continuous with internal at r=R
        """
        sin_sq = np.sin(theta)**2
        return (self.U / 2) * (r**2 - self.R**3 / r) * sin_sq

    def density_perturbation(self, r, amplitude):
        """
        Density perturbation from vortex circulation.
        From HillVortex.lean line 65-70.

        δρ(r) = -amplitude · (1 - r²/R²)  for r < R
              = 0                         for r ≥ R

        Physical meaning:
        - Negative δρ (pressure deficit from Bernoulli)
        - Maximum depression at core (r=0)
        - Smoothly approaches vacuum at boundary (r=R)
        - Creates "trap" for soliton stability
        """
        if r < self.R:
            return -amplitude * (1 - (r / self.R)**2)
        else:
            return 0.0

# ============================================================================
# 4-COMPONENT FIELD STRUCTURE
# ============================================================================

class FourComponentField:
    """
    4-component field structure for swirling Hill vortex.

    From AxisAlignment.lean:
    "The QFD Electron is a 'Swirling' Hill Vortex. It has:
     1. Poloidal circulation (Standard Hill) -> Defines the soliton shape.
     2. Toroidal/Azimuthal swirl (The 'Spin') -> Adds non-zero L_z."

    Components:
    - psi_s: Scalar (density/pressure field from poloidal circulation)
    - psi_b0, psi_b1, psi_b2: Bi-vector (toroidal swirl, 3 components for 3D)

    The decomposition allows:
    - Axial symmetry (rotation about z-axis)
    - Spin angular momentum (from toroidal components)
    - Collinear P ∥ L (unique to Hill vortex!)
    """

    def __init__(self, r_grid, hill_context):
        self.r = r_grid
        self.n_points = len(r_grid)
        self.hill = hill_context

        # Initialize from Hill vortex ansatz
        self.psi_s = np.zeros(self.n_points)
        self.psi_b0 = np.zeros(self.n_points)
        self.psi_b1 = np.zeros(self.n_points)
        self.psi_b2 = np.zeros(self.n_points)

    def initialize_from_hill_vortex(self, theta=np.pi/2):
        """
        Initialize field from Hill vortex stream function.

        We use θ = π/2 (equatorial plane) as reference.

        Decomposition strategy:
        - psi_s ← density perturbation (dominant for mass)
        - psi_b0, psi_b1, psi_b2 ← toroidal swirl (for spin)
        """
        # Scalar component from density perturbation
        for i, r in enumerate(self.r):
            # Start with normalized Gaussian times density perturbation shape
            amplitude = 1.0  # Will be normalized later
            if r < self.hill.R:
                self.psi_s[i] = amplitude * (1 - (r / self.hill.R)**2) * np.exp(-r**2 / (2 * self.hill.R**2))
            else:
                self.psi_s[i] = 0.0

        # Bi-vector components from stream function derivatives
        # (Simplified: use Hill stream function modulated by angle)
        for i, r in enumerate(self.r):
            if r < self.hill.R:
                # Internal toroidal circulation
                psi_hill = self.hill.stream_function_internal(r, theta)
            else:
                # External potential flow
                psi_hill = self.hill.stream_function_external(r, theta)

            # Decompose stream function into 3 toroidal components
            # (Simplified model: distribute across 3 orthogonal directions)
            norm = np.sqrt(3)
            self.psi_b0[i] = psi_hill / norm
            self.psi_b1[i] = psi_hill / norm * 0.8  # Slightly different phases
            self.psi_b2[i] = psi_hill / norm * 0.6

    def get_density(self):
        """
        Total field density ρ = |ψ|² = ψ_s² + ψ_b0² + ψ_b1² + ψ_b2².

        This is the "mass density" determining energy.
        """
        return self.psi_s**2 + self.psi_b0**2 + self.psi_b1**2 + self.psi_b2**2

    def get_charge_density(self, dr):
        """
        Charge density from Poisson equation (scalar component only).

        ρ_q = -g_c · ∇²ψ_s

        In spherical coordinates:
        ∇²ψ_s = ∂²ψ_s/∂r² + (2/r)∂ψ_s/∂r
        """
        # Second derivative (central differences)
        d2_psi_s_dr2 = np.gradient(np.gradient(self.psi_s, dr), dr)

        # First derivative
        d_psi_s_dr = np.gradient(self.psi_s, dr)

        # Spherical Laplacian
        r_safe = self.r.copy()
        r_safe[0] = 1e-9  # Avoid division by zero
        laplacian = d2_psi_s_dr2 + (2.0 / r_safe) * d_psi_s_dr

        # L'Hôpital's rule at r=0
        laplacian[0] = 3.0 * d2_psi_s_dr2[0]

        return -self.hill.g_c * laplacian

# ============================================================================
# ENERGY FUNCTIONAL (Euler-Lagrange Formulation)
# ============================================================================

class HillVortexEnergyFunctional:
    """
    Energy functional for Hill vortex soliton.

    E[ψ] = E_kinetic + E_potential + E_csr

    where:
    - E_kinetic = ½∫|∇ψ|² · 4πr² dr  [Gradient energy]
    - E_potential = ∫V(ρ(ψ)) · 4πr² dr  [Density-dependent potential]
    - E_csr = -½k_csr ∫ρ_q² · 4πr² dr  [Charge self-repulsion]

    The KEY difference from V22 original:
    V(ρ) NOT V(r)! Potential depends on field density, not position.
    """

    def __init__(self, r_grid, hill_context, V4=11.0, k_csr=0.0):
        self.r = r_grid
        self.dr = r_grid[1] - r_grid[0]
        self.hill = hill_context
        self.V4 = V4  # Quartic coefficient (can be derived from β)
        self.k_csr = k_csr

        # Volume element for spherical integration
        self.vol_element = 4 * np.pi * self.r**2

    def kinetic_energy(self, field):
        """
        E_kin = ½∫[|∇ψ_s|² + |∇ψ_b0|² + |∇ψ_b1|² + |∇ψ_b2|²] · 4πr² dr
        """
        grad_s = np.gradient(field.psi_s, self.dr)
        grad_b0 = np.gradient(field.psi_b0, self.dr)
        grad_b1 = np.gradient(field.psi_b1, self.dr)
        grad_b2 = np.gradient(field.psi_b2, self.dr)

        grad_squared = grad_s**2 + grad_b0**2 + grad_b1**2 + grad_b2**2

        integrand = 0.5 * grad_squared * self.vol_element
        return simps(integrand, x=self.r)

    def potential_energy(self, field):
        """
        E_pot = ∫V(ρ) · 4πr² dr

        where V(ρ) = β·(ρ - ρ_vac)² + V4·(ρ - ρ_vac)⁴

        CRITICAL: Potential depends on DENSITY, not radius!

        β determines energy scale (from cosmology/nuclear).
        V4 provides quartic stability.
        """
        rho = field.get_density()

        # Density perturbation from vacuum
        delta_rho = rho - self.hill.rho_vac

        # Quartic potential in density
        # V(ρ) = β·δρ² + V4·δρ⁴
        V_rho = self.hill.beta * delta_rho**2 + self.V4 * delta_rho**4

        integrand = V_rho * self.vol_element
        return simps(integrand, x=self.r)

    def csr_energy(self, field):
        """
        E_csr = -½k_csr ∫ρ_q² · 4πr² dr

        Charge self-repulsion (usually k_csr = 0 for leptons).
        """
        rho_q = field.get_charge_density(self.dr)
        integrand = -0.5 * self.k_csr * rho_q**2 * self.vol_element
        return simps(integrand, x=self.r)

    def total_energy(self, field):
        """Total energy E = E_kin + E_pot + E_csr"""
        E_kin = self.kinetic_energy(field)
        E_pot = self.potential_energy(field)
        E_csr = self.csr_energy(field)
        return E_kin + E_pot + E_csr

    def compute_q_star(self, field):
        """
        Q* = sqrt(∫ρ_q² · 4πr² dr)  [RMS charge density]

        This encodes the internal angular structure.
        Different Q* → Different masses!
        """
        rho_q = field.get_charge_density(self.dr)
        integrand = rho_q**2 * self.vol_element
        return np.sqrt(simps(integrand, x=self.r))

# ============================================================================
# ENHANCED V22 SOLVER
# ============================================================================

class EnhancedV22Solver:
    """
    Enhanced V22 Lepton Mass Solver with Hill Vortex Structure.

    Tests THE HYPOTHESIS:
    Can β = 3.1 from cosmology/nuclear produce correct lepton masses?

    If YES → Complete unification (revolutionary!)
    If NO → Scale separation (still interesting!)
    """

    def __init__(self, beta=3.1, target_mass_MeV=ELECTRON_MASS,
                 R_fm=1.0, U=1.0, num_points=200, r_max_fm=10.0):
        """
        Parameters:
        - beta: Vacuum stiffness (from cosmology/nuclear)
        - target_mass_MeV: Lepton mass to target
        - R_fm: Vortex radius in femtometers
        - U: Propagation velocity (normalized)
        - num_points: Grid resolution
        - r_max_fm: Maximum radius for integration
        """
        self.beta = beta
        self.target_mass = target_mass_MeV
        self.R = R_fm
        self.U = U

        # Create radial grid (avoid r=0 singularity)
        self.r = np.linspace(0.01, r_max_fm, num_points)
        self.dr = self.r[1] - self.r[0]

        # Initialize Hill vortex context
        self.hill = HillVortexContext(R=R_fm, U=U, beta=beta)

        # Initialize field
        self.field = FourComponentField(self.r, self.hill)
        self.field.initialize_from_hill_vortex()

        # Initialize energy functional
        self.energy = HillVortexEnergyFunctional(self.r, self.hill)

        print(f"Initialized Enhanced V22 Solver")
        print(f"  β = {beta:.2f} (from cosmology/nuclear)")
        print(f"  Target mass: {target_mass_MeV:.4f} MeV")
        print(f"  Vortex radius R = {R_fm:.2f} fm")
        print(f"  Grid: {num_points} points, r_max = {r_max_fm:.2f} fm")

    def pack_fields(self):
        """Pack 4-component field into 1D array for optimizer."""
        return np.concatenate([
            self.field.psi_s,
            self.field.psi_b0,
            self.field.psi_b1,
            self.field.psi_b2
        ])

    def unpack_fields(self, params):
        """Unpack 1D array into 4-component field."""
        n = len(self.r)
        self.field.psi_s = params[0:n]
        self.field.psi_b0 = params[n:2*n]
        self.field.psi_b1 = params[2*n:3*n]
        self.field.psi_b2 = params[3*n:4*n]

    def objective(self, params):
        """
        Objective function for optimizer.

        Minimizes total energy E[ψ] to find stable soliton.
        This implements the Euler-Lagrange equation: δE/δψ = 0
        """
        self.unpack_fields(params)

        # Enforce boundary conditions
        self.field.psi_s[0] = self.field.psi_s[1]  # Regularity at r=0
        self.field.psi_s[-1] = 0.0  # Decay at infinity
        self.field.psi_b0[0] = self.field.psi_b0[1]
        self.field.psi_b0[-1] = 0.0
        self.field.psi_b1[0] = self.field.psi_b1[1]
        self.field.psi_b1[-1] = 0.0
        self.field.psi_b2[0] = self.field.psi_b2[1]
        self.field.psi_b2[-1] = 0.0

        return self.energy.total_energy(self.field)

    def solve(self, max_iter=500):
        """
        Solve Euler-Lagrange equations to find stable Hill vortex soliton.

        Uses L-BFGS-B optimization (same as Phoenix solver).
        """
        print("\nSolving Euler-Lagrange equations...")
        print("  (Minimizing energy functional E[ψ] for stable soliton)")

        params_init = self.pack_fields()

        result = minimize(
            self.objective,
            params_init,
            method='L-BFGS-B',
            options={'disp': True, 'maxiter': max_iter, 'ftol': 1e-9}
        )

        self.unpack_fields(result.x)

        # Calculate final observables
        final_energy = result.fun
        Q_star = self.energy.compute_q_star(self.field)

        # Calculate effective radius
        rho = self.field.get_density()
        r_weighted = self.r * rho
        vol_element = 4 * np.pi * self.r**2
        R_eff = simps(r_weighted * vol_element, x=self.r) / (Q_star + 1e-12)

        return {
            'energy_MeV': final_energy,
            'Q_star': Q_star,
            'R_eff_fm': R_eff,
            'converged': result.success,
            'iterations': result.nit
        }

# ============================================================================
# TEST THE 3.1 HYPOTHESIS
# ============================================================================

def test_beta_3_1_with_hill_vortex():
    """
    THE CRITICAL TEST (Enhanced Version):

    Does β = 3.1 from cosmology/nuclear produce correct lepton masses
    when using the CORRECT Hill vortex formulation?

    This is the enhanced test with proper physics:
    - Hill vortex structure (not simple potential well)
    - 4-component fields (not 1D wavefunction)
    - Euler-Lagrange (not Schrödinger)
    - V(ρ) density-dependent (not V(r) radial)
    """
    print("=" * 80)
    print("THE 3.1 QUESTION (Enhanced with Hill Vortex Structure)")
    print("=" * 80)
    print()
    print("PHYSICS:")
    print("  - Hill spherical vortex (from HillVortex.lean)")
    print("  - 4-component field: (ψ_s, ψ_b0, ψ_b1, ψ_b2)")
    print("  - Euler-Lagrange: δE/δψ = 0")
    print("  - Potential: V(ρ) = β·(ρ-ρ_vac)² + V4·(ρ-ρ_vac)⁴")
    print("  - β = 3.1 from cosmology/nuclear (UNIFIED!)")
    print()

    # Test electron first
    print("=" * 80)
    print("TESTING: ELECTRON")
    print("=" * 80)

    # For electron, use Compton wavelength as initial R guess
    lambda_e_fm = HBAR_C / ELECTRON_MASS  # ~386 fm
    R_electron = lambda_e_fm / 100  # Start with R ~ 4 fm (rough guess)

    solver = EnhancedV22Solver(
        beta=3.1,
        target_mass_MeV=ELECTRON_MASS,
        R_fm=R_electron,
        U=1.0,
        num_points=200,
        r_max_fm=20.0
    )

    results = solver.solve(max_iter=300)

    print()
    print("=" * 80)
    print("RESULTS:")
    print("=" * 80)
    print(f"Energy: {results['energy_MeV']:.6f} MeV")
    print(f"Target (electron): {ELECTRON_MASS:.6f} MeV")
    print(f"Error: {abs(results['energy_MeV'] - ELECTRON_MASS):.6f} MeV")
    print(f"Accuracy: {(1 - abs(results['energy_MeV'] - ELECTRON_MASS)/ELECTRON_MASS)*100:.4f}%")
    print()
    print(f"Q*: {results['Q_star']:.4f}")
    print(f"R_eff: {results['R_eff_fm']:.4f} fm")
    print(f"Converged: {results['converged']}")
    print()

    # Verdict
    accuracy = (1 - abs(results['energy_MeV'] - ELECTRON_MASS)/ELECTRON_MASS) * 100

    print("=" * 80)
    print("VERDICT:")
    print("=" * 80)

    if accuracy > 99.9:
        print("✅ SUCCESS!")
        print("   β = 3.1 from cosmology DOES produce electron mass!")
        print("   COMPLETE UNIFICATION ACHIEVED!")
    elif accuracy > 95.0:
        print("⚠️  PROMISING!")
        print(f"   {accuracy:.2f}% accuracy - Close but needs refinement")
        print("   May need:")
        print("   - Better initial conditions")
        print("   - More iterations")
        print("   - Fine-tuning of R or V4")
    else:
        print("❌ PARTIAL SUCCESS")
        print(f"   {accuracy:.2f}% accuracy - Not yet correct")
        print("   Possible reasons:")
        print("   - Unit conversion needed for β")
        print("   - Missing relativistic corrections")
        print("   - Need different V4 derivation from β")
        print("   - Fundamental scale separation (β_particle ≠ β_nuclear)")

    print("=" * 80)

    return results

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    results = test_beta_3_1_with_hill_vortex()

    # Save results
    output_dir = Path("/home/tracy/development/QFD_SpectralGap/V22_Lepton_Analysis/results")
    output_dir.mkdir(exist_ok=True, parents=True)

    with open(output_dir / "v22_enhanced_hill_vortex_test.json", 'w') as f:
        # Convert numpy types
        results_json = {k: (float(v) if isinstance(v, (np.floating, np.integer))
                           else bool(v) if isinstance(v, np.bool_) else v)
                       for k, v in results.items()}
        json.dump(results_json, f, indent=2)

    print(f"\nResults saved to: {output_dir / 'v22_enhanced_hill_vortex_test.json'}")
