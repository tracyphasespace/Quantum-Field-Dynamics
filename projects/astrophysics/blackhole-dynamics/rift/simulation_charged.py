import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

"""
Charged Particle Dynamics for Black Hole Rift Simulations

Extends simulation.py with:
1. Coulomb forces between charged particles
2. Thermal pressure forces
3. QFD gravitational forces (angle-dependent)
4. Particle charge tracking (electrons vs ions)
5. N-body charge interactions

Physical Framework:
- Total force: F = F_grav + F_coulomb + F_thermal
- F_grav = -m ∇Φ(r,θ,φ) [QFD time refraction, angle-dependent]
- F_coulomb = Σ k_e q₁q₂/r² (all pairwise interactions)
- F_thermal = -∇P/ρ where P = nkT

Lean References:
- QFD.Rift.ChargeEscape.modified_schwarzschild_escape
- QFD.EM.Coulomb.force_law
- QFD.Gravity.TimeRefraction.timePotential_eq

Created: 2025-12-22
"""

import numpy as np
import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass, field
from scipy.integrate import solve_ivp

from config import SimConfig
from rift.rotation_dynamics import Particle
from rift.core_3d import ScalarFieldSolution3D


@dataclass
class ChargedParticleState:
    """
    State vector for a charged particle.

    Attributes:
        position: [x, y, z] in meters
        velocity: [vx, vy, vz] in m/s
        mass: kg
        charge: Coulombs
        particle_type: 'electron' or 'ion'
    """
    position: np.ndarray  # [3]
    velocity: np.ndarray  # [3]
    mass: float
    charge: float
    particle_type: str

    def to_vector(self) -> np.ndarray:
        """Convert to state vector [x, y, z, vx, vy, vz]"""
        return np.concatenate([self.position, self.velocity])

    @staticmethod
    def from_vector(Y: np.ndarray, mass: float, charge: float, particle_type: str):
        """Create from state vector"""
        return ChargedParticleState(
            position=Y[:3],
            velocity=Y[3:],
            mass=mass,
            charge=charge,
            particle_type=particle_type
        )


class ChargedParticleDynamics:
    """
    Simulates charged particle dynamics in rotating binary black hole system.

    Includes:
    - QFD gravitational forces (3D, angle-dependent)
    - Coulomb forces (N-body pairwise)
    - Thermal pressure forces
    """

    def __init__(
        self,
        config: SimConfig,
        field_3d: ScalarFieldSolution3D,
        BH1_position: np.ndarray,
        BH2_position: Optional[np.ndarray] = None,
        include_thermal: bool = False
    ):
        """
        Initialize charged particle dynamics.

        Args:
            config: Simulation configuration
            field_3d: 3D scalar field solution
            BH1_position: Position of BH1 [x, y, z]
            BH2_position: Position of BH2 [x, y, z] (optional)
            include_thermal: Include thermal pressure forces (default: False)
        """
        self.config = config
        self.field_3d = field_3d
        self.BH1_pos = BH1_position
        self.BH2_pos = BH2_position if BH2_position is not None else np.zeros(3)
        self.include_thermal = include_thermal

        # Physical constants
        self.k_coulomb = config.K_COULOMB
        self.k_boltzmann = config.K_BOLTZMANN
        self.c_light = config.C_LIGHT

        # Plasma parameters
        self.T_plasma = config.T_PLASMA_CORE
        self.n_density = config.N_DENSITY_SURFACE

        logging.info(f"ChargedParticleDynamics initialized:")
        logging.info(f"  T_plasma = {self.T_plasma:.2e} K")
        logging.info(f"  n_density = {self.n_density:.2e} m⁻³")
        logging.info(f"  include_thermal = {self.include_thermal}")

    def cartesian_to_spherical(self, position: np.ndarray) -> Tuple[float, float, float]:
        """
        Convert Cartesian to spherical coordinates.

        Args:
            position: [x, y, z]

        Returns:
            (r, theta, phi) where:
            r = √(x²+y²+z²)
            theta ∈ [0, π] (polar angle from z-axis)
            phi ∈ [0, 2π] (azimuthal angle in xy-plane)
        """
        x, y, z = position
        r = np.sqrt(x**2 + y**2 + z**2)

        if r < 1e-10:
            return 0.0, 0.0, 0.0

        theta = np.arccos(np.clip(z / r, -1, 1))
        phi = np.arctan2(y, x)
        if phi < 0:
            phi += 2 * np.pi

        return r, theta, phi

    def compute_qfd_gravitational_force(
        self,
        particle: ChargedParticleState
    ) -> np.ndarray:
        """
        Compute QFD gravitational force on particle.

        F_grav = -m ∇Φ(r,θ,φ)

        where Φ = -(c²/2)κρ(r,θ,φ) [QFD time refraction]

        Args:
            particle: Particle state

        Returns:
            Force vector [Fx, Fy, Fz] in Newtons

        Lean reference: QFD.Gravity.TimeRefraction.timePotential_eq
        """
        # Convert to spherical coordinates
        r, theta, phi_angle = self.cartesian_to_spherical(particle.position)

        if r < 1e-10:
            return np.zeros(3)

        # Get potential gradient in spherical coords
        grad_sph = self.field_3d.qfd_potential_gradient(r, theta, phi_angle)
        # grad_sph = [∂Φ/∂r, (1/r)∂Φ/∂θ, (1/r sinθ)∂Φ/∂φ]

        # Convert to Cartesian
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        sin_phi = np.sin(phi_angle)
        cos_phi = np.cos(phi_angle)

        # Transformation matrix: spherical → Cartesian gradient
        # ∇_x = sin(θ)cos(φ) ∂/∂r + cos(θ)cos(φ)/r ∂/∂θ - sin(φ)/(r sin(θ)) ∂/∂φ
        # ∇_y = sin(θ)sin(φ) ∂/∂r + cos(θ)sin(φ)/r ∂/∂θ + cos(φ)/(r sin(θ)) ∂/∂φ
        # ∇_z = cos(θ) ∂/∂r - sin(θ)/r ∂/∂θ

        grad_r, grad_theta_over_r, grad_phi_over_r_sin_theta = grad_sph

        grad_x = (sin_theta * cos_phi * grad_r +
                  cos_theta * cos_phi * grad_theta_over_r -
                  sin_phi * grad_phi_over_r_sin_theta)

        grad_y = (sin_theta * sin_phi * grad_r +
                  cos_theta * sin_phi * grad_theta_over_r +
                  cos_phi * grad_phi_over_r_sin_theta)

        grad_z = (cos_theta * grad_r -
                  sin_theta * grad_theta_over_r)

        grad_Phi_cart = np.array([grad_x, grad_y, grad_z])

        # Force: F = -m ∇Φ
        F_grav = -particle.mass * grad_Phi_cart

        return F_grav

    def compute_coulomb_force(
        self,
        particle_index: int,
        all_particles: List[ChargedParticleState]
    ) -> np.ndarray:
        """
        Compute Coulomb force on particle from all other particles.

        F_c = Σ k_e q₁q₂/r² r̂

        where sum is over all other particles.

        Args:
            particle_index: Index of particle to compute force on
            all_particles: List of all particles

        Returns:
            Force vector [Fx, Fy, Fz] in Newtons

        Lean reference: QFD.EM.Coulomb.force_law
        """
        particle_i = all_particles[particle_index]
        F_coulomb = np.zeros(3)

        for j, particle_j in enumerate(all_particles):
            if j == particle_index:
                continue

            # Vector from j to i
            r_vec = particle_i.position - particle_j.position
            r = np.linalg.norm(r_vec)

            if r < 1e-10:
                # Avoid singularity (particles at same position)
                continue

            r_hat = r_vec / r

            # Coulomb force: F = k_e q₁q₂/r² r̂
            F_magnitude = self.k_coulomb * particle_i.charge * particle_j.charge / (r**2)
            F_coulomb += F_magnitude * r_hat

        return F_coulomb

    def compute_thermal_pressure_force(
        self,
        particle: ChargedParticleState
    ) -> np.ndarray:
        """
        Compute thermal pressure force on particle.

        F_thermal = -∇P / ρ

        where P = nkT (ideal gas pressure)
        and ρ = n * m_avg (mass density)

        For simplicity, assume pressure gradient points radially outward
        from BH center with scale height H ~ kT/(m g).

        Args:
            particle: Particle state

        Returns:
            Force vector [Fx, Fy, Fz] in Newtons

        Lean reference: QFD.Rift.ChargeEscape.thermal_energy_contribution
        """
        # Position relative to BH1
        r_vec = particle.position - self.BH1_pos
        r = np.linalg.norm(r_vec)

        if r < 1e-10:
            return np.zeros(3)

        r_hat = r_vec / r

        # Pressure: P = nkT
        P = self.n_density * self.k_boltzmann * self.T_plasma

        # Scale height: H ~ kT / (m g)
        # For simplicity, use exponential decay: P(r) = P₀ exp(-r/H)
        H = 1.0  # Scale height in meters (approximate)

        # Pressure gradient: dP/dr = -P/H exp(-r/H) ≈ -P/H (for r << H)
        dP_dr = -P / H

        # Force per unit mass: F/m = -∇P/ρ = -(dP/dr) / ρ
        # ρ = n * m_avg ≈ n * m_particle
        rho = self.n_density * particle.mass

        F_magnitude = -dP_dr / rho if rho > 1e-30 else 0.0

        F_thermal = F_magnitude * r_hat

        return F_thermal

    def total_force(
        self,
        particle_index: int,
        all_particles: List[ChargedParticleState]
    ) -> np.ndarray:
        """
        Compute total force on particle.

        F_total = F_grav + F_coulomb + F_thermal (if enabled)

        Args:
            particle_index: Index of particle
            all_particles: List of all particles

        Returns:
            Total force vector [Fx, Fy, Fz]
        """
        particle = all_particles[particle_index]

        # Gravitational force (QFD, angle-dependent)
        F_grav = self.compute_qfd_gravitational_force(particle)

        # Coulomb force (pairwise interactions)
        F_coulomb = self.compute_coulomb_force(particle_index, all_particles)

        # Thermal pressure force (optional)
        if self.include_thermal:
            F_thermal = self.compute_thermal_pressure_force(particle)
        else:
            F_thermal = np.zeros(3)

        # Total force
        F_total = F_grav + F_coulomb + F_thermal

        return F_total

    def equations_of_motion(
        self,
        t: float,
        Y_flat: np.ndarray,
        particle_metadata: List[Tuple[float, float, str]]
    ) -> np.ndarray:
        """
        Equations of motion for N charged particles.

        Y_flat = [x₁, y₁, z₁, vx₁, vy₁, vz₁, x₂, y₂, z₂, vx₂, vy₂, vz₂, ...]

        dY/dt = [v₁, a₁, v₂, a₂, ...]

        where aᵢ = F_total(i) / mᵢ

        Args:
            t: Time
            Y_flat: Flattened state vector (6*N elements)
            particle_metadata: List of (mass, charge, type) for each particle

        Returns:
            dY/dt flattened
        """
        N = len(particle_metadata)

        if len(Y_flat) != 6 * N:
            raise ValueError(f"State vector size mismatch: {len(Y_flat)} != 6*{N}")

        # Reconstruct particle states
        particles = []
        for i in range(N):
            Y_i = Y_flat[6*i:6*(i+1)]
            mass, charge, ptype = particle_metadata[i]
            particles.append(ChargedParticleState.from_vector(Y_i, mass, charge, ptype))

        # Compute derivatives
        dY_dt = np.zeros(6 * N)

        for i in range(N):
            particle = particles[i]

            # Velocity
            dY_dt[6*i:6*i+3] = particle.velocity

            # Acceleration: a = F_total / m
            F_total = self.total_force(i, particles)
            acceleration = F_total / particle.mass

            dY_dt[6*i+3:6*i+6] = acceleration

        return dY_dt

    def simulate_charged_particles(
        self,
        particles_initial: List[ChargedParticleState],
        t_span: Tuple[float, float],
        t_eval: Optional[np.ndarray] = None,
        method: str = 'LSODA'
    ) -> dict:
        """
        Simulate N charged particles.

        Args:
            particles_initial: List of initial particle states
            t_span: (t_start, t_end)
            t_eval: Times to evaluate solution
            method: ODE solver method

        Returns:
            Dictionary with:
            - t: Time array
            - particles: List of particle trajectories
            - success: Whether integration succeeded
        """
        N = len(particles_initial)

        # Flatten initial conditions
        Y0 = np.concatenate([p.to_vector() for p in particles_initial])

        # Extract metadata
        metadata = [(p.mass, p.charge, p.particle_type) for p in particles_initial]

        # Define ODE function
        def ode_func(t, Y):
            return self.equations_of_motion(t, Y, metadata)

        logging.info(f"Simulating {N} charged particles...")
        logging.info(f"  Time span: {t_span}")
        logging.info(f"  Method: {method}")

        # Solve ODE
        sol = solve_ivp(
            ode_func,
            t_span,
            Y0,
            method=method,
            t_eval=t_eval,
            rtol=self.config.ODE_RTOL,
            atol=self.config.ODE_ATOL
        )

        if not sol.success:
            logging.warning(f"ODE solver failed: {sol.message}")

        # Unpack solution into particle trajectories
        particle_trajectories = []
        for i in range(N):
            traj = {
                'position': sol.y[6*i:6*i+3, :],  # [3 × n_times]
                'velocity': sol.y[6*i+3:6*i+6, :],
                'mass': metadata[i][0],
                'charge': metadata[i][1],
                'type': metadata[i][2]
            }
            particle_trajectories.append(traj)

        return {
            't': sol.t,
            'particles': particle_trajectories,
            'success': sol.success,
            'message': sol.message
        }


# ========================================
# TESTING / VALIDATION
# ========================================

if __name__ == "__main__":
    """Test charged particle dynamics"""

    print("=" * 80)
    print("CHARGED PARTICLE DYNAMICS: Unit Tests")
    print("=" * 80)
    print()

    from config import SimConfig
    from rift.core_3d import ScalarFieldSolution3D

    config = SimConfig()
    config.__post_init__()

    # Test 1: Create 3D field
    print("Test 1: Initialize 3D Field and Dynamics")
    print("-" * 80)

    Omega_BH1 = np.array([0, 0, 0.5])
    Omega_BH2 = np.array([0, 0, -0.5])

    field_3d = ScalarFieldSolution3D(
        config=config,
        phi_0=3.0,
        Omega_BH1=Omega_BH1,
        Omega_BH2=Omega_BH2
    )

    field_3d.solve(r_min=1e-3, r_max=50.0, n_r=50)
    print(f"  ✅ 3D field solved")

    BH1_pos = np.array([0.0, 0.0, 0.0])
    dynamics = ChargedParticleDynamics(config, field_3d, BH1_pos)
    print(f"  ✅ Dynamics initialized")
    print()

    # Test 2: Coulomb force between two particles
    print("Test 2: Coulomb Force")
    print("-" * 80)

    particle1 = ChargedParticleState(
        position=np.array([0.0, 0.0, 0.0]),
        velocity=np.array([0.0, 0.0, 0.0]),
        mass=config.M_ELECTRON,
        charge=config.Q_ELECTRON,
        particle_type='electron'
    )

    particle2 = ChargedParticleState(
        position=np.array([1.0, 0.0, 0.0]),  # 1m away
        velocity=np.array([0.0, 0.0, 0.0]),
        mass=config.M_PROTON,
        charge=config.Q_PROTON,
        particle_type='ion'
    )

    particles = [particle1, particle2]

    F_coulomb_1 = dynamics.compute_coulomb_force(0, particles)
    F_coulomb_2 = dynamics.compute_coulomb_force(1, particles)

    print(f"  Particle 1 (electron) at [0, 0, 0]")
    print(f"  Particle 2 (proton) at [1, 0, 0]")
    print(f"  F_coulomb on electron: {F_coulomb_1}")
    print(f"  F_coulomb on proton: {F_coulomb_2}")
    print(f"  ✅ Forces equal and opposite: {np.allclose(F_coulomb_1, -F_coulomb_2)}")
    print()

    # Test 3: QFD gravitational force
    print("Test 3: QFD Gravitational Force")
    print("-" * 80)

    particle_test = ChargedParticleState(
        position=np.array([10.0, 0.0, 0.0]),
        velocity=np.array([0.0, 0.0, 0.0]),
        mass=config.M_ELECTRON,
        charge=config.Q_ELECTRON,
        particle_type='electron'
    )

    F_grav = dynamics.compute_qfd_gravitational_force(particle_test)
    print(f"  Particle at [10, 0, 0]")
    print(f"  F_grav: {F_grav}")
    print(f"  |F_grav|: {np.linalg.norm(F_grav):.6e} N")
    print(f"  ✅ Gravitational force computed")
    print()

    # Test 4: Total force
    print("Test 4: Total Force (Grav + Coulomb + Thermal)")
    print("-" * 80)

    F_total = dynamics.total_force(0, particles)
    print(f"  F_total on particle 1: {F_total}")
    print(f"  ✅ Total force computed")
    print()

    # Test 5: Short trajectory simulation
    print("Test 5: Trajectory Simulation (2 particles)")
    print("-" * 80)

    result = dynamics.simulate_charged_particles(
        particles_initial=particles,
        t_span=(0.0, 1e-9),  # 1 nanosecond
        method='RK45'
    )

    print(f"  Success: {result['success']}")
    print(f"  Number of time steps: {len(result['t'])}")
    print(f"  Final positions:")
    for i, traj in enumerate(result['particles']):
        final_pos = traj['position'][:, -1]
        print(f"    Particle {i+1}: {final_pos}")
    print(f"  ✅ Simulation complete")
    print()

    print("=" * 80)
    print("✅ ALL TESTS COMPLETED")
    print("=" * 80)
