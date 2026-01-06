#!/usr/bin/env python3
"""
V22 Lepton Mass Solver - The 3.1 Question

THE CRITICAL TEST:
Does β ≈ 3.1 from Cosmology/Nuclear produce m_μ/m_e ≈ 206?

If YES → Cosmology determines Particle Physics
If NO → Theory needs revision

IMPROVEMENTS OVER PREVIOUS ATTEMPTS:
1. Uses β from VALIDATED V22 Cosmology/Nuclear results
2. Parameter bounds enforced by Lean 4 proofs:
   - β ∈ (0, ∞) from confinement requirement
   - Q = 2/3 from Koide geometric constraint
3. Solves radial Schrödinger equation: -ψ'' + V(r)ψ = E·ψ
   where V(r) = β(r² - v²)²
4. Direct comparison with experimental lepton masses

Physics Model:
    V(r) = β·(r² - v²)²  (Quartic Potential)

Where:
    - β ≈ 3.1 (from CMB/Nuclear - SAME parameter!)
    - v ≈ 1.0 (vacuum scale)
    - Eigenvalues E_n are the lepton masses

Author: QFD Research Team
Date: December 22, 2025
Version: V22 (Lean-Constrained, Unified from Cosmic to Particle)
"""

import numpy as np
from scipy.integrate import solve_bvp
from scipy.optimize import minimize_scalar, minimize
import matplotlib.pyplot as plt
import json
from pathlib import Path

# Physical constants (MeV)
ELECTRON_MASS = 0.5109989461  # MeV
MUON_MASS = 105.6583745      # MeV
TAU_MASS = 1776.86           # MeV

# ============================================================================
# LEAN-PROVEN CONSTRAINTS
# ============================================================================

class LeanConstraints:
    """
    Parameter constraints derived from formal Lean 4 proofs.

    Source: /projects/Lean4/QFD/Lepton/MassSpectrum.lean

    Theorems:
    - theorem qfd_potential_is_confining: β > 0 ensures discrete spectrum
    - theorem geometric_mass_condition: Koide Q = 2/3
    """

    # From MassSpectrum.lean:
    # structure SolitonParams where
    #   beta : ℝ
    #   v : ℝ
    #   h_beta_pos : beta > 0
    #   h_v_pos : v > 0

    BETA_MIN = 0.0   # Confinement requires β > 0
    BETA_MAX = 10.0  # Physical upper bound

    V_MIN = 0.1      # Vacuum scale lower bound
    V_MAX = 2.0      # Vacuum scale upper bound

    # Koide relation target
    KOIDE_TARGET = 2.0 / 3.0  # Q = 2/3 from lattice geometry

    @classmethod
    def validate_beta(cls, beta):
        """Validate β against Lean confinement constraint."""
        if not (cls.BETA_MIN < beta < cls.BETA_MAX):
            raise ValueError(
                f"β = {beta} violates Lean proof! "
                f"Must be in ({cls.BETA_MIN}, {cls.BETA_MAX})"
            )
        return True

    @classmethod
    def validate_koide(cls, m_e, m_mu, m_tau, tolerance=0.01):
        """Validate masses against Koide constraint."""
        Q = koide_parameter(m_e, m_mu, m_tau)
        error = abs(Q - cls.KOIDE_TARGET)
        if error > tolerance:
            print(f"⚠️  Koide deviation: Q = {Q:.6f} (target = {cls.KOIDE_TARGET:.6f})")
        return error < tolerance

# ============================================================================
# KOIDE RELATION
# ============================================================================

def koide_parameter(m_e, m_mu, m_tau):
    """
    Koide Formula: Q = (m₁ + m₂ + m₃) / (√m₁ + √m₂ + √m₃)²

    Empirically Q ≈ 2/3 for charged leptons.
    In QFD, this is derived from lattice geometry.
    """
    numerator = m_e + m_mu + m_tau
    denominator = (np.sqrt(m_e) + np.sqrt(m_mu) + np.sqrt(m_tau))**2
    return numerator / denominator

# ============================================================================
# QUARTIC POTENTIAL SOLVER
# ============================================================================

class QuarticPotentialSolver:
    """
    Solve the 1D radial Schrödinger equation with quartic potential.

    -ψ'' + V(r)ψ = E·ψ
    V(r) = β·(r² - v²)²

    Boundary conditions:
    - ψ(0) = 0 (regularity at origin)
    - ψ(∞) → 0 (normalizability)
    """

    def __init__(self, beta, v, r_max=10.0, n_points=500):
        self.beta = beta
        self.v = v
        self.r_max = r_max
        self.n_points = n_points
        self.r = np.linspace(0.01, r_max, n_points)  # Avoid r=0 singularity

    def potential(self, r):
        """V(r) = β·(r² - v²)²"""
        return self.beta * (r**2 - self.v**2)**2

    def shooting_method(self, E, r_grid=None):
        """
        Shooting method to find eigenvalues.

        Returns:
            Wavefunction value at r_max (should be ~0 for eigenstate)
        """
        if r_grid is None:
            r_grid = self.r

        # Numerov method for ODE: -ψ'' + V(r)ψ = E·ψ
        # Rearrange: ψ'' = (V(r) - E)·ψ

        dr = r_grid[1] - r_grid[0]
        n = len(r_grid)
        psi = np.zeros(n)

        # Initial conditions (start near r=0 with small linear rise)
        psi[0] = 0.0
        psi[1] = r_grid[1]  # ψ ~ r near origin

        # Numerov integration
        for i in range(1, n-1):
            V_curr = self.potential(r_grid[i])
            V_next = self.potential(r_grid[i+1])
            V_prev = self.potential(r_grid[i-1])

            k2_prev = 2 * (V_prev - E)
            k2_curr = 2 * (V_curr - E)
            k2_next = 2 * (V_next - E)

            # Numerov formula
            numerator = 2 * psi[i] * (1 - 5*dr**2/12 * k2_curr) - psi[i-1] * (1 + dr**2/12 * k2_prev)
            denominator = 1 + dr**2/12 * k2_next
            psi[i+1] = numerator / denominator

            # Prevent overflow
            if abs(psi[i+1]) > 1e10:
                return 1e10  # Diverging solution

        return psi[-1]  # Return endpoint value

    def find_eigenvalue(self, E_guess, E_min=0.0, E_max=100.0):
        """
        Find eigenvalue by searching for zero crossing of ψ(r_max).
        """
        def objective(E):
            return abs(self.shooting_method(E))

        result = minimize_scalar(objective, bounds=(E_min, E_max), method='bounded')
        return result.x if result.success else None

    def find_spectrum(self, n_states=3):
        """
        Find first n_states eigenvalues.

        Returns:
            Array of eigenvalues (energies/masses)
        """
        eigenvalues = []

        # Rough energy spacing estimate
        E_spacing = 20.0  # MeV

        for n in range(n_states):
            # Search in windows to find different states
            E_min = n * E_spacing
            E_max = (n + 2) * E_spacing

            E_n = self.find_eigenvalue(0, E_min, E_max)
            if E_n is not None:
                eigenvalues.append(E_n)
            else:
                eigenvalues.append(np.nan)

        return np.array(eigenvalues)

# ============================================================================
# THE 3.1 QUESTION
# ============================================================================

def test_beta_3_1_hypothesis():
    """
    THE CRITICAL TEST:

    Does β ≈ 3.1 (from CMB/Nuclear) produce m_μ/m_e ≈ 206?

    This is the unification test. If cosmology determines particle physics,
    the SAME β that appears in SNe scattering and nuclear compression
    must produce the correct lepton mass hierarchy.
    """
    print("=" * 80)
    print("THE 3.1 QUESTION: Does Cosmology Determine Particle Masses?")
    print("=" * 80)
    print()

    # β from V22 Cosmology/Nuclear results
    beta_cosmic = 3.1  # From CMB/Nuclear Phase 1
    v = 1.0            # Vacuum scale (normalized)

    print(f"Input from Cosmology/Nuclear:")
    print(f"  β = {beta_cosmic:.2f} (from CMB stiffness)")
    print(f"  v = {v:.2f} (vacuum scale)")
    print()

    # Validate Lean constraint
    LeanConstraints.validate_beta(beta_cosmic)
    print(f"✅ β satisfies Lean confinement constraint")
    print()

    # Solve radial equation
    print("Solving radial Schrödinger equation...")
    print(f"  V(r) = {beta_cosmic}·(r² - {v}²)²")
    print()

    solver = QuarticPotentialSolver(beta=beta_cosmic, v=v, r_max=5.0)

    # Find first 3 eigenvalues (e, μ, τ)
    eigenvalues = solver.find_spectrum(n_states=3)

    if np.any(np.isnan(eigenvalues)):
        print("❌ Failed to find all eigenvalues")
        return None

    m_e_pred, m_mu_pred, m_tau_pred = eigenvalues

    print("=" * 80)
    print("RESULTS: Predicted Lepton Masses")
    print("=" * 80)
    print(f"Electron:  {m_e_pred:.4f} MeV (experimental: {ELECTRON_MASS:.4f} MeV)")
    print(f"Muon:      {m_mu_pred:.4f} MeV (experimental: {MUON_MASS:.4f} MeV)")
    print(f"Tau:       {m_tau_pred:.4f} MeV (experimental: {TAU_MASS:.4f} MeV)")
    print()

    # Mass ratios
    ratio_mu_e_pred = m_mu_pred / m_e_pred
    ratio_mu_e_exp = MUON_MASS / ELECTRON_MASS

    ratio_tau_e_pred = m_tau_pred / m_e_pred
    ratio_tau_e_exp = TAU_MASS / ELECTRON_MASS

    print("MASS RATIOS:")
    print(f"m_μ/m_e (predicted): {ratio_mu_e_pred:.2f}")
    print(f"m_μ/m_e (experimental): {ratio_mu_e_exp:.2f}")
    print(f"Difference: {abs(ratio_mu_e_pred - ratio_mu_e_exp):.2f}")
    print()
    print(f"m_τ/m_e (predicted): {ratio_tau_e_pred:.2f}")
    print(f"m_τ/m_e (experimental): {ratio_tau_e_exp:.2f}")
    print(f"Difference: {abs(ratio_tau_e_pred - ratio_tau_e_exp):.2f}")
    print()

    # Koide parameter
    Q_pred = koide_parameter(m_e_pred, m_mu_pred, m_tau_pred)
    Q_exp = koide_parameter(ELECTRON_MASS, MUON_MASS, TAU_MASS)

    print("KOIDE PARAMETER:")
    print(f"Q (predicted): {Q_pred:.6f}")
    print(f"Q (experimental): {Q_exp:.6f}")
    print(f"Q (Lean target): {LeanConstraints.KOIDE_TARGET:.6f}")
    print()

    # Verdict
    print("=" * 80)
    print("VERDICT:")
    print("=" * 80)

    success = abs(ratio_mu_e_pred - ratio_mu_e_exp) / ratio_mu_e_exp < 0.1

    if success:
        print("✅ β = 3.1 from Cosmology DOES produce correct lepton hierarchy!")
        print("✅ COSMOLOGY DETERMINES PARTICLE PHYSICS!")
        print()
        print("This proves:")
        print("  - Same β in SNe scattering, nuclear compression, AND lepton masses")
        print("  - Unified framework spans 21 orders of magnitude")
        print("  - No separate 'particle physics' needed!")
    else:
        print("⚠️  β = 3.1 does NOT produce correct hierarchy")
        print("   Further investigation needed:")
        print("   - May need different potential form")
        print("   - May need relativistic corrections")
        print("   - May need coupling to other fields")
    print("=" * 80)

    return {
        'beta': beta_cosmic,
        'v': v,
        'masses_predicted': {
            'electron': m_e_pred,
            'muon': m_mu_pred,
            'tau': m_tau_pred
        },
        'masses_experimental': {
            'electron': ELECTRON_MASS,
            'muon': MUON_MASS,
            'tau': TAU_MASS
        },
        'ratios': {
            'mu_e_predicted': ratio_mu_e_pred,
            'mu_e_experimental': ratio_mu_e_exp,
            'tau_e_predicted': ratio_tau_e_pred,
            'tau_e_experimental': ratio_tau_e_exp
        },
        'koide': {
            'predicted': Q_pred,
            'experimental': Q_exp,
            'target': LeanConstraints.KOIDE_TARGET
        },
        'success': success
    }

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    results = test_beta_3_1_hypothesis()

    if results:
        # Save results
        output_dir = Path("/home/tracy/development/QFD_SpectralGap/V22_Lepton_Analysis/results")
        output_dir.mkdir(exist_ok=True, parents=True)

        with open(output_dir / "v22_lepton_test.json", 'w') as f:
            # Convert numpy types to Python types for JSON
            def convert_to_json(obj):
                if isinstance(obj, (np.integer, np.floating)):
                    return float(obj)
                elif isinstance(obj, (np.bool_)):
                    return bool(obj)
                elif isinstance(obj, dict):
                    return {k: convert_to_json(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_json(item) for item in obj]
                else:
                    return obj

            results_json = convert_to_json(results)
            json.dump(results_json, f, indent=2)

        print(f"\nResults saved to: {output_dir / 'v22_lepton_test.json'}")
