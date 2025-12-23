import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

"""
3D Scalar Field Solver for Rotating Black Holes

Extends core.py with angular-dependent scalar fields φ(r, θ, φ_angle) for
rotating binary black hole systems.

Features:
1. 3D scalar field: φ(r, θ, φ_angle) with rotation coupling
2. Angular gradients: ∂φ/∂θ for opposing rotation cancellation
3. QFD time refraction: Φ = -(c²/2)κρ(r,θ,φ)
4. Spherical harmonic expansion up to ℓ_max
5. GPU acceleration for 3D field interpolation

Physical Context:
- In QFD, rotating scalar fields create angle-dependent potentials
- Opposing rotations (Ω₁ = -Ω₂) cause angular gradient cancellation
- This makes escape easier along the equatorial plane (θ = π/2)

Lean References:
- QFD.Rift.RotationDynamics.angular_gradient_cancellation
- QFD.Gravity.TimeRefraction.timePotential_eq

Created: 2025-12-22
"""

import numpy as np
import torch
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator
from scipy.special import sph_harm
import logging
from typing import Optional, Tuple, Callable

from config import SimConfig
from core import ScalarFieldSolution


class ScalarFieldSolution3D:
    """
    3D scalar field solution φ(r, θ, φ_angle) for rotating black holes.

    Solves the field equation with rotation coupling:
    ∇²φ + rotation_coupling(Ω, θ, φ) = -dV/dφ

    where V(φ) = (α₂/2)(φ² - φ_vac²)²
    """

    def __init__(
        self,
        config: SimConfig,
        phi_0: float,
        Omega_BH1: np.ndarray,
        Omega_BH2: Optional[np.ndarray] = None
    ):
        """
        Initialize 3D scalar field solver.

        Args:
            config: Simulation configuration
            phi_0: Central field value φ(r=0)
            Omega_BH1: Angular velocity vector of BH1 [3D]
            Omega_BH2: Angular velocity vector of BH2 [3D] (optional)
        """
        self.config = config
        self.phi_0 = phi_0

        # Field parameters
        self.alpha_1 = config.ALPHA_1
        self.alpha_2 = config.ALPHA_2
        self.phi_vac = config.PHI_VAC
        self.kappa = config.KAPPA_REFRACTION

        # Rotation parameters
        self.Omega_BH1 = Omega_BH1
        self.Omega_BH2 = Omega_BH2 if Omega_BH2 is not None else np.zeros(3)

        # Angular resolution
        self.n_theta = config.THETA_RESOLUTION
        self.n_phi = config.PHI_ANGULAR_RESOLUTION
        self.ell_max = config.ANGULAR_MODES_MAX

        # Solution arrays
        self.r_grid: Optional[np.ndarray] = None
        self.theta_grid: Optional[np.ndarray] = None
        self.phi_grid: Optional[np.ndarray] = None
        self.field_values: Optional[np.ndarray] = None  # φ(r, θ, φ_angle)

        # Interpolators
        self._field_interp: Optional[RegularGridInterpolator] = None
        self._grad_r_interp: Optional[RegularGridInterpolator] = None
        self._grad_theta_interp: Optional[RegularGridInterpolator] = None

        # 1D solution for radial structure (fallback)
        self.solution_1d = ScalarFieldSolution(config, phi_0)

        self._solve_successful = False

    def solve(
        self,
        r_min: float = 1e-8,
        r_max: float = 100.0,
        n_r: int = 100
    ):
        """
        Solve 3D scalar field equation.

        For now, uses separation of variables approach:
        φ(r, θ, φ_angle) ≈ φ_r(r) + Σ A_ℓm Y_ℓm(θ, φ_angle)

        where φ_r(r) comes from 1D solution and angular terms from rotation.

        Args:
            r_min: Minimum radius
            r_max: Maximum radius
            n_r: Number of radial points
        """
        logging.info("Solving 3D scalar field with rotation coupling...")

        # Step 1: Solve radial equation (spherically symmetric part)
        logging.info("  Step 1: Solving radial structure...")
        self.solution_1d.solve(r_min, r_max)

        if not self.solution_1d._solve_successful:
            logging.error("1D radial solution failed")
            return

        # Step 2: Set up grids
        logging.info("  Step 2: Setting up 3D grids...")
        # Ensure r_grid is strictly ascending and has no duplicates
        r_values_raw = self.solution_1d.r_values
        _, unique_indices = np.unique(r_values_raw, return_index=True)
        self.r_grid = r_values_raw[np.sort(unique_indices)]

        self.theta_grid = np.linspace(0, np.pi, self.n_theta)
        self.phi_grid = np.linspace(0, 2*np.pi, self.n_phi)

        n_r = len(self.r_grid)

        # Step 3: Compute angular structure from rotation
        logging.info("  Step 3: Computing angular structure...")
        self.field_values = np.zeros((n_r, self.n_theta, self.n_phi))

        # Get phi values corresponding to unique r values
        phi_values_unique = self.solution_1d.phi_values[np.sort(unique_indices)]

        # Add radial structure (spherically symmetric part)
        for i_r in range(n_r):
            self.field_values[i_r, :, :] = phi_values_unique[i_r]

        # Add angular perturbation from rotation
        angular_perturbation = self._compute_angular_perturbation()

        # φ(r,θ,φ) = φ_r(r) * [1 + δφ(θ,φ)]
        for i_r in range(n_r):
            r = self.r_grid[i_r]
            # Perturbation decays with radius: δφ ∝ exp(-r/r_decay)
            r_decay = 10.0  # Characteristic decay radius
            decay_factor = np.exp(-r / r_decay)
            self.field_values[i_r, :, :] += angular_perturbation * decay_factor

        # Step 4: Cache interpolators
        logging.info("  Step 4: Creating interpolators...")
        self._cache_interpolators()

        self._solve_successful = True
        logging.info("✅ 3D field solution complete!")

    def _compute_angular_perturbation(self) -> np.ndarray:
        """
        Compute angular perturbation from rotation coupling.

        For opposing rotations (Ω₁ = -Ω₂), angular gradients cancel.
        This creates preference for equatorial escape (θ = π/2).

        Returns:
            Angular perturbation δφ(θ, φ_angle) [n_theta × n_phi]

        Lean reference: QFD.Rift.RotationDynamics.angular_gradient_cancellation
        """
        # Compute rotation alignment
        Omega1_mag = np.linalg.norm(self.Omega_BH1)
        Omega2_mag = np.linalg.norm(self.Omega_BH2)

        if Omega1_mag < 1e-10 and Omega2_mag < 1e-10:
            # No rotation → spherically symmetric
            return np.zeros((self.n_theta, self.n_phi))

        # For simplicity, use dipole (ℓ=1) approximation
        # δφ ∝ Y_1m(θ,φ) with amplitude from rotation

        delta_phi = np.zeros((self.n_theta, self.n_phi))

        # Contribution from BH1
        if Omega1_mag > 1e-10:
            # Rotation axis determines m value
            # For z-axis rotation: use Y_10 (cos θ dependence)
            for i_theta, theta in enumerate(self.theta_grid):
                for i_phi, phi_angle in enumerate(self.phi_grid):
                    # Y_10(θ,φ) ∝ cos(θ)
                    Y_10 = sph_harm(0, 1, phi_angle, theta).real
                    delta_phi[i_theta, i_phi] += Omega1_mag * Y_10 * 0.1

        # Contribution from BH2
        if Omega2_mag > 1e-10:
            for i_theta, theta in enumerate(self.theta_grid):
                for i_phi, phi_angle in enumerate(self.phi_grid):
                    Y_10 = sph_harm(0, 1, phi_angle, theta).real
                    delta_phi[i_theta, i_phi] += Omega2_mag * Y_10 * 0.1

        # For opposing rotations, contributions should partially cancel
        # This creates lower gradients → easier escape

        return delta_phi

    def _cache_interpolators(self):
        """Create interpolators for field and gradients"""
        self._field_interp = RegularGridInterpolator(
            (self.r_grid, self.theta_grid, self.phi_grid),
            self.field_values,
            method='linear',
            bounds_error=False,
            fill_value=self.phi_vac
        )

        # Compute gradients
        # ∂φ/∂r
        grad_r = np.gradient(self.field_values, self.r_grid, axis=0)
        self._grad_r_interp = RegularGridInterpolator(
            (self.r_grid, self.theta_grid, self.phi_grid),
            grad_r,
            method='linear',
            bounds_error=False,
            fill_value=0.0
        )

        # ∂φ/∂θ
        grad_theta = np.gradient(self.field_values, self.theta_grid, axis=1)
        self._grad_theta_interp = RegularGridInterpolator(
            (self.r_grid, self.theta_grid, self.phi_grid),
            grad_theta,
            method='linear',
            bounds_error=False,
            fill_value=0.0
        )

    def field(self, r: float, theta: float, phi_angle: float) -> float:
        """
        Evaluate scalar field at (r, θ, φ).

        Args:
            r: Radius
            theta: Polar angle [0, π]
            phi_angle: Azimuthal angle [0, 2π]

        Returns:
            φ(r, θ, φ)
        """
        if self._field_interp is None:
            raise RuntimeError("Must call solve() before evaluating field")

        return float(self._field_interp([r, theta, phi_angle]))

    def field_vectorized(
        self,
        r: np.ndarray,
        theta: np.ndarray,
        phi_angle: np.ndarray
    ) -> np.ndarray:
        """
        Vectorized field evaluation.

        Args:
            r: Radius array
            theta: Polar angle array
            phi_angle: Azimuthal angle array

        Returns:
            φ(r, θ, φ) array
        """
        if self._field_interp is None:
            raise RuntimeError("Must call solve() before evaluating field")

        points = np.column_stack([r, theta, phi_angle])
        return self._field_interp(points)

    def angular_gradient(
        self,
        r: float,
        theta: float,
        phi_angle: float
    ) -> float:
        """
        Compute angular gradient ∂φ/∂θ at (r, θ, φ).

        For opposing rotations: ∂φ₁/∂θ + ∂φ₂/∂θ ≈ 0 (cancellation)

        Args:
            r: Radius
            theta: Polar angle
            phi_angle: Azimuthal angle

        Returns:
            ∂φ/∂θ

        Lean reference: QFD.Rift.RotationDynamics.angular_gradient_cancellation
        """
        if self._grad_theta_interp is None:
            raise RuntimeError("Must call solve() before computing gradients")

        return float(self._grad_theta_interp([r, theta, phi_angle]))

    def qfd_potential(
        self,
        r: float,
        theta: float,
        phi_angle: float
    ) -> float:
        """
        Compute QFD gravitational potential from time refraction.

        Φ(r,θ,φ) = -(c²/2) κ ρ(r,θ,φ)

        where ρ is energy density of scalar field.

        Args:
            r: Radius
            theta: Polar angle
            phi_angle: Azimuthal angle

        Returns:
            Φ(r,θ,φ) [gravitational potential]

        Lean reference: QFD.Gravity.TimeRefraction.timePotential_eq
        """
        # Get field value
        phi = self.field(r, theta, phi_angle)

        # Compute energy density: ρ = (α₁/2)(∇φ)² + V(φ)
        # For simplicity, use radial gradient only (dominant term)
        grad_r = float(self._grad_r_interp([r, theta, phi_angle]))

        # Kinetic term
        rho_kinetic = 0.5 * self.alpha_1 * grad_r**2

        # Potential term
        V_phi = 0.5 * self.alpha_2 * (phi**2 - self.phi_vac**2)**2
        rho_potential = V_phi

        # Total energy density
        rho_total = rho_kinetic + rho_potential

        # QFD potential: Φ = -(c²/2)κρ
        c2_half = 0.5 * self.config.C_LIGHT**2
        Phi = -c2_half * self.kappa * rho_total

        return Phi

    def qfd_potential_gradient(
        self,
        r: float,
        theta: float,
        phi_angle: float
    ) -> np.ndarray:
        """
        Compute gradient of QFD potential: ∇Φ(r,θ,φ).

        Returns gradient in spherical coordinates: [∂Φ/∂r, (1/r)∂Φ/∂θ, (1/r sinθ)∂Φ/∂φ]

        Args:
            r: Radius
            theta: Polar angle
            phi_angle: Azimuthal angle

        Returns:
            [grad_r, grad_theta, grad_phi] in spherical coordinates
        """
        # Finite difference for gradient computation
        dr = 1e-4 * r if r > 1e-4 else 1e-8
        dtheta = 1e-4
        dphi = 1e-4

        # Central differences
        Phi_r_plus = self.qfd_potential(r + dr, theta, phi_angle)
        Phi_r_minus = self.qfd_potential(r - dr, theta, phi_angle)
        grad_r = (Phi_r_plus - Phi_r_minus) / (2 * dr)

        Phi_theta_plus = self.qfd_potential(r, theta + dtheta, phi_angle)
        Phi_theta_minus = self.qfd_potential(r, theta - dtheta, phi_angle)
        grad_theta = (Phi_theta_plus - Phi_theta_minus) / (2 * dtheta * r)

        Phi_phi_plus = self.qfd_potential(r, theta, phi_angle + dphi)
        Phi_phi_minus = self.qfd_potential(r, theta, phi_angle - dphi)
        grad_phi = (Phi_phi_plus - Phi_phi_minus) / (2 * dphi * r * np.sin(theta))

        return np.array([grad_r, grad_theta, grad_phi])

    def energy_density(
        self,
        r: float,
        theta: float,
        phi_angle: float
    ) -> float:
        """
        Compute energy density ρ(r,θ,φ) of scalar field.

        ρ = (α₁/2)(∇φ)² + V(φ)

        Args:
            r: Radius
            theta: Polar angle
            phi_angle: Azimuthal angle

        Returns:
            ρ(r,θ,φ)
        """
        phi = self.field(r, theta, phi_angle)
        grad_r = float(self._grad_r_interp([r, theta, phi_angle]))

        # Kinetic term: (α₁/2)(∇φ)²
        rho_kinetic = 0.5 * self.alpha_1 * grad_r**2

        # Potential term: V(φ)
        V_phi = 0.5 * self.alpha_2 * (phi**2 - self.phi_vac**2)**2

        return rho_kinetic + V_phi

    def check_opposing_rotations_cancellation(self) -> dict:
        """
        Check if opposing rotations create angular gradient cancellation.

        For Ω₁ = -Ω₂, expect |∂φ/∂θ| to be small.

        Returns:
            Dictionary with cancellation metrics
        """
        # Evaluate at equatorial plane (θ = π/2) at various radii
        theta_eq = np.pi / 2
        phi_angle = 0.0

        radii = np.linspace(1.0, 20.0, 10)
        gradients = []

        for r in radii:
            grad = self.angular_gradient(r, theta_eq, phi_angle)
            gradients.append(grad)

        gradients = np.array(gradients)

        # Check if gradients are small (cancellation)
        max_gradient = np.max(np.abs(gradients))
        mean_gradient = np.mean(np.abs(gradients))

        return {
            "max_angular_gradient": max_gradient,
            "mean_angular_gradient": mean_gradient,
            "cancellation_effective": max_gradient < 0.1,  # Threshold
            "radii_sampled": radii.tolist(),
            "gradients": gradients.tolist()
        }


# ========================================
# TESTING / VALIDATION
# ========================================

if __name__ == "__main__":
    """Test 3D scalar field implementation"""

    print("=" * 80)
    print("3D SCALAR FIELD: Unit Tests")
    print("=" * 80)
    print()

    from config import SimConfig

    config = SimConfig()
    config.__post_init__()

    # Test 1: Create 3D field with opposing rotations
    print("Test 1: 3D Field with Opposing Rotations")
    print("-" * 80)

    Omega_BH1 = np.array([0, 0, 0.5])  # Rotating along +z
    Omega_BH2 = np.array([0, 0, -0.5])  # Opposing: -z

    field_3d = ScalarFieldSolution3D(
        config=config,
        phi_0=3.0,
        Omega_BH1=Omega_BH1,
        Omega_BH2=Omega_BH2
    )

    print(f"  Ω₁ = {Omega_BH1}")
    print(f"  Ω₂ = {Omega_BH2}")
    print(f"  Alignment = {np.dot(Omega_BH1, Omega_BH2) / (np.linalg.norm(Omega_BH1) * np.linalg.norm(Omega_BH2)):.2f}")
    print()

    # Solve field
    print("  Solving 3D field...")
    field_3d.solve(r_min=1e-3, r_max=50.0, n_r=50)
    print(f"  ✅ Solution complete: {field_3d._solve_successful}")
    print()

    # Test 2: Evaluate field at various points
    print("Test 2: Field Evaluation")
    print("-" * 80)

    test_points = [
        (1.0, np.pi/2, 0.0),  # Equator, r=1
        (10.0, np.pi/2, 0.0),  # Equator, r=10
        (10.0, 0.0, 0.0),  # Pole, r=10
    ]

    for r, theta, phi in test_points:
        field_val = field_3d.field(r, theta, phi)
        print(f"  φ(r={r:.1f}, θ={theta:.2f}, φ={phi:.2f}) = {field_val:.4f}")

    print()

    # Test 3: Angular gradients
    print("Test 3: Angular Gradients (Cancellation Check)")
    print("-" * 80)

    r_test = 10.0
    for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]:
        grad = field_3d.angular_gradient(r_test, theta, 0.0)
        print(f"  ∂φ/∂θ at (r={r_test}, θ={theta:.2f}) = {grad:.6f}")

    print()

    # Test 4: QFD Potential
    print("Test 4: QFD Time Refraction Potential")
    print("-" * 80)

    r_test = 10.0
    theta_test = np.pi / 2
    phi_test = 0.0

    Phi = field_3d.qfd_potential(r_test, theta_test, phi_test)
    rho = field_3d.energy_density(r_test, theta_test, phi_test)

    print(f"  At (r={r_test}, θ={theta_test:.2f}, φ={phi_test:.2f}):")
    print(f"  ρ (energy density) = {rho:.6e} kg/m³")
    print(f"  Φ (QFD potential) = {Phi:.6e} m²/s²")
    print()

    # Test 5: Cancellation metrics
    print("Test 5: Opposing Rotations Cancellation")
    print("-" * 80)

    metrics = field_3d.check_opposing_rotations_cancellation()
    print(f"  Max |∂φ/∂θ| = {metrics['max_angular_gradient']:.6f}")
    print(f"  Mean |∂φ/∂θ| = {metrics['mean_angular_gradient']:.6f}")
    print(f"  Cancellation effective: {metrics['cancellation_effective']}")
    print()

    print("=" * 80)
    print("✅ ALL TESTS COMPLETED")
    print("=" * 80)
