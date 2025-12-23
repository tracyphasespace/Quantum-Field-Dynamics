import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

#!/usr/bin/env python3
"""
Binary Black Hole Rift Simulation - L1 Saddle Point Ejection

Integrates:
1. Existing TwoBodySystem with L1 saddle point finding
2. 3D scalar field with opposing rotations (Ω₁ = -Ω₂)
3. Charged particle dynamics through the saddle point

Demonstrates:
- Gravitational ejection through L1 Lagrange point
- How opposing rotations enable escape (angular gradient cancellation)
- Charge-mediated plasma dynamics in rift zone

Usage:
    python rift/binary_rift_simulation.py
"""

import numpy as np
import logging
from typing import List, Tuple, Dict

# Original code
from config import SimConfig
from core import ScalarFieldSolution, TwoBodySystem

# Rift physics
from rift.core_3d import ScalarFieldSolution3D
from rift.simulation_charged import ChargedParticleState, ChargedParticleDynamics
from rift.rotation_dynamics import RotationDynamics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BinaryRiftSystem:
    """
    Binary black hole system with rift physics.

    Combines:
    - TwoBodySystem (binary configuration + L1 point)
    - 3D scalar field with rotations
    - Charged particle dynamics
    """

    def __init__(
        self,
        config: SimConfig,
        M1: float,
        M2: float,
        separation: float,
        Omega1: np.ndarray,
        Omega2: np.ndarray
    ):
        """
        Initialize binary rift system.

        Args:
            config: Simulation configuration
            M1: Mass of BH1 (soliton mass units)
            M2: Mass of BH2 (soliton mass units)
            separation: Distance between BHs [m]
            Omega1: Angular velocity of BH1 [rad/s]
            Omega2: Angular velocity of BH2 [rad/s] (should be -Omega1 for rifts!)
        """
        self.config = config
        self.M1 = M1
        self.M2 = M2
        self.separation = separation
        self.Omega1 = Omega1
        self.Omega2 = Omega2

        # Step 1: Create reference scalar field solution
        logger.info("Step 1: Solving reference 1D scalar field...")
        phi_0 = 3.0
        self.solution_ref = ScalarFieldSolution(config, phi_0)
        self.solution_ref.solve()

        logger.info(f"  Reference mass: {self.solution_ref.mass:.3e}")
        logger.info(f"  Reference r_core: {self.solution_ref.r_core:.3e}")

        # Step 2: Create binary system
        logger.info("Step 2: Setting up binary system...")
        self.binary_system = TwoBodySystem(config, self.solution_ref, M1, M2)
        self.binary_system.set_separation(separation)

        # BH positions
        self.BH1_pos = np.array([0.0, 0.0, 0.0])
        self.BH2_pos = np.array([separation, 0.0, 0.0])

        logger.info(f"  BH1 at {self.BH1_pos}, mass = {M1:.3e}")
        logger.info(f"  BH2 at {self.BH2_pos}, mass = {M2:.3e}")
        logger.info(f"  Separation: {separation:.3e} m")

        # Step 3: Find L1 saddle point
        logger.info("Step 3: Finding L1 saddle point...")
        self.saddle_point, self.saddle_energy = self.binary_system.find_saddle_point()

        logger.info(f"  L1 saddle point: {self.saddle_point}")
        logger.info(f"  Saddle energy: {self.saddle_energy:.3e}")

        # Step 4: Create 3D field with rotations
        logger.info("Step 4: Solving 3D scalar field with rotations...")
        self.field_3d = ScalarFieldSolution3D(
            config=config,
            phi_0=phi_0,
            Omega_BH1=Omega1,
            Omega_BH2=Omega2
        )
        self.field_3d.solve(r_min=1e-3, r_max=separation*2, n_r=50)

        # Check rotation alignment
        alignment = Omega1.dot(Omega2) / (np.linalg.norm(Omega1) * np.linalg.norm(Omega2))
        logger.info(f"  Rotation alignment: {alignment:.3f} {'(OPPOSING!)' if alignment < -0.5 else ''}")

        # Check angular gradient cancellation
        metrics = self.field_3d.check_opposing_rotations_cancellation()
        logger.info(f"  Max |∂φ/∂θ|: {metrics['max_angular_gradient']:.6f}")
        logger.info(f"  Cancellation effective: {metrics['cancellation_effective']}")

        # Step 5: Setup charged particle dynamics
        logger.info("Step 5: Initializing charged particle dynamics...")
        self.dynamics = ChargedParticleDynamics(
            config,
            self.field_3d,
            self.BH1_pos,
            self.BH2_pos,
            include_thermal=False  # Focus on gravity + Coulomb
        )

        # Step 6: Setup rotation dynamics
        self.rotation_dynamics = RotationDynamics(config)

        logger.info("✅ Binary rift system initialized!")
        print()

    def create_particles_near_L1(
        self,
        n_electrons: int = 2,
        n_ions: int = 2,
        offset_distance: float = 0.1
    ) -> List[ChargedParticleState]:
        """
        Create charged particles near L1 saddle point.

        Args:
            n_electrons: Number of electrons
            n_ions: Number of ions
            offset_distance: Distance from L1 [m]

        Returns:
            List of charged particles
        """
        particles = []

        # L1 position (x-axis between BHs)
        L1_x = self.saddle_point[0]

        logger.info(f"Creating particles near L1 at x={L1_x:.3f} m")

        # Thermal velocity (Maxwell-Boltzmann)
        T = self.config.T_PLASMA_CORE
        k_B = self.config.K_BOLTZMANN

        # Create electrons (lighter, faster)
        v_th_electron = np.sqrt(2 * k_B * T / self.config.M_ELECTRON)

        for i in range(n_electrons):
            # Place slightly offset from L1
            theta = 2 * np.pi * i / n_electrons
            position = np.array([
                L1_x + offset_distance * np.cos(theta),
                offset_distance * np.sin(theta),
                0.0
            ])

            # Random thermal velocity
            velocity = np.random.randn(3) * v_th_electron * 0.1  # 10% of thermal

            particles.append(ChargedParticleState(
                position=position,
                velocity=velocity,
                mass=self.config.M_ELECTRON,
                charge=self.config.Q_ELECTRON,
                particle_type='electron'
            ))

        # Create ions (heavier, slower)
        v_th_ion = np.sqrt(2 * k_B * T / self.config.M_PROTON)

        for i in range(n_ions):
            # Place slightly offset from L1
            theta = 2 * np.pi * i / n_ions + np.pi / n_ions  # Offset from electrons
            position = np.array([
                L1_x + offset_distance * np.cos(theta),
                offset_distance * np.sin(theta),
                0.0
            ])

            # Random thermal velocity
            velocity = np.random.randn(3) * v_th_ion * 0.1

            particles.append(ChargedParticleState(
                position=position,
                velocity=velocity,
                mass=self.config.M_PROTON,
                charge=self.config.Q_PROTON,
                particle_type='ion'
            ))

        logger.info(f"  Created {n_electrons} electrons, {n_ions} ions")
        logger.info(f"  Electron v_thermal: {v_th_electron:.2e} m/s")
        logger.info(f"  Ion v_thermal: {v_th_ion:.2e} m/s")

        return particles

    def compute_effective_potential_1d(
        self,
        x_array: np.ndarray
    ) -> np.ndarray:
        """
        Compute effective potential along x-axis (between BHs).

        Shows the saddle point structure.

        Args:
            x_array: x-coordinates [m]

        Returns:
            Potential values
        """
        potentials = np.zeros_like(x_array)

        for i, x in enumerate(x_array):
            q = np.array([x, 0.0, 0.0])
            potentials[i] = self.binary_system.total_potential(q)

        return potentials

    def simulate_rift_ejection(
        self,
        particles: List[ChargedParticleState],
        t_span: Tuple[float, float],
        t_eval: np.ndarray = None
    ) -> Dict:
        """
        Simulate particle ejection through L1 saddle point.

        Args:
            particles: Initial particle states
            t_span: (t_start, t_end) in seconds
            t_eval: Times to evaluate solution

        Returns:
            Simulation results
        """
        logger.info(f"Simulating rift ejection...")
        logger.info(f"  Time span: {t_span[0]:.2e} to {t_span[1]:.2e} s")
        logger.info(f"  {len(particles)} particles")

        # Simulate
        result = self.dynamics.simulate_charged_particles(
            particles_initial=particles,
            t_span=t_span,
            t_eval=t_eval,
            method='RK45'
        )

        # Analyze escape
        L1_x = self.saddle_point[0]

        for i, traj in enumerate(result['particles']):
            pos_initial = traj['position'][:, 0]
            pos_final = traj['position'][:, -1]

            # Check if particle crossed L1
            x_initial = pos_initial[0]
            x_final = pos_final[0]

            crossed_L1 = (x_initial < L1_x and x_final > L1_x) or (x_initial > L1_x and x_final < L1_x)

            escaped = np.linalg.norm(pos_final) > self.separation

            particle_type = traj['type']
            logger.info(f"  {particle_type} {i}: crossed_L1={crossed_L1}, escaped={escaped}")

        result['L1_position'] = self.saddle_point
        result['BH1_position'] = self.BH1_pos
        result['BH2_position'] = self.BH2_pos
        result['separation'] = self.separation

        return result

    def get_system_info(self) -> Dict:
        """Get summary of system configuration."""

        alignment = self.Omega1.dot(self.Omega2) / (
            np.linalg.norm(self.Omega1) * np.linalg.norm(self.Omega2)
        )

        return {
            'M1': self.M1,
            'M2': self.M2,
            'separation': self.separation,
            'L1_position': self.saddle_point,
            'L1_energy': self.saddle_energy,
            'Omega1': self.Omega1,
            'Omega2': self.Omega2,
            'rotation_alignment': alignment,
            'opposing_rotations': alignment < -0.5,
            'BH1_position': self.BH1_pos,
            'BH2_position': self.BH2_pos
        }


def main():
    """Run binary rift simulation with L1 ejection."""

    print("="*80)
    print("BINARY BLACK HOLE RIFT SIMULATION")
    print("L1 Saddle Point Ejection with Opposing Rotations")
    print("="*80)
    print()

    # Configuration
    config = SimConfig()
    config.__post_init__()

    # Binary system parameters
    M1 = 1.0  # Reference mass units
    M2 = 1.0  # Equal mass binary
    separation = 50.0  # meters

    # Opposing rotations (CRITICAL for rift physics!)
    Omega_magnitude = config.OMEGA_BH1_MAGNITUDE
    Omega1 = np.array([0.0, 0.0, Omega_magnitude])
    Omega2 = np.array([0.0, 0.0, -Omega_magnitude])  # OPPOSING!

    print(f"Binary System Configuration:")
    print(f"  M1 = {M1} (soliton masses)")
    print(f"  M2 = {M2} (soliton masses)")
    print(f"  Separation = {separation} m")
    print(f"  Ω₁ = {Omega1} rad/s")
    print(f"  Ω₂ = {Omega2} rad/s (opposing!)")
    print()

    # Create binary rift system
    system = BinaryRiftSystem(config, M1, M2, separation, Omega1, Omega2)

    # Create particles near L1
    particles = system.create_particles_near_L1(n_electrons=3, n_ions=3, offset_distance=0.5)

    print()
    print(f"Initial Particle Configuration:")
    for i, p in enumerate(particles):
        r = np.linalg.norm(p.position)
        v = np.linalg.norm(p.velocity)
        print(f"  {p.particle_type} {i}: r={r:.3f} m, v={v:.2e} m/s")
    print()

    # Simulate ejection
    t_span = (0.0, 1e-6)  # 1 microsecond
    t_eval = np.linspace(t_span[0], t_span[1], 1000)

    result = system.simulate_rift_ejection(particles, t_span, t_eval)

    print()
    print("="*80)
    print("SIMULATION COMPLETE")
    print("="*80)
    print(f"Success: {result['success']}")
    print(f"Time steps: {len(result['t'])}")
    print()

    # System summary
    info = system.get_system_info()
    print("System Summary:")
    print(f"  L1 position: {info['L1_position']}")
    print(f"  L1 energy: {info['L1_energy']:.3e}")
    print(f"  Rotation alignment: {info['rotation_alignment']:.3f}")
    print(f"  Opposing rotations: {info['opposing_rotations']}")
    print()

    # Next: Create visualizations
    print("To visualize results:")
    print("  from rift.binary_rift_visualization import plot_L1_ejection")
    print("  plot_L1_ejection(system, result)")
    print()

    return system, result


if __name__ == "__main__":
    system, result = main()
