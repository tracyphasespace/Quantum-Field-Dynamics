import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

"""
QFD Black Hole Rotation Dynamics and Spin Evolution

Implements the spin-sorting ratchet mechanism where rift eruptions drive
binary black hole systems toward opposing rotations (Ω₁ ≈ -Ω₂).

Physical Mechanism:
1. Plasma erupts from modified Schwarzschild surface
2. Ejecta with favorable angular momentum → ESCAPES
3. Ejecta with unfavorable angular momentum → RECAPTURED
4. Recaptured material deposits L back into BH
5. Net torque drives system toward Ω₁ = -Ω₂ (equilibrium)

Mathematical Framework:
- Angular momentum: L = r × p
- Net torque: τ_net = ∫ L_recaptured dm - ∫ L_escaped dm
- Spin evolution: dΩ/dt = τ_net / I
- Equilibrium: rotation_alignment → -1

Lean References:
- QFD.Rift.SpinSorting.spin_sorting_equilibrium
- QFD.Rift.SpinSorting.net_torque_evolution
- QFD.Rift.RotationDynamics.angular_gradient_cancellation

Created: 2025-12-22
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from config import SimConfig


@dataclass
class Particle:
    """
    Charged particle with position, velocity, mass, and charge.

    Attributes:
        pos: Position vector [x, y, z] in meters
        vel: Velocity vector [vx, vy, vz] in m/s
        mass: Particle mass in kg
        charge: Particle charge in Coulombs
        particle_type: 'electron' or 'ion'
    """
    pos: np.ndarray  # [3] array
    vel: np.ndarray  # [3] array
    mass: float
    charge: float
    particle_type: str  # 'electron' or 'ion'

    def __post_init__(self):
        """Validate particle data"""
        assert self.pos.shape == (3,), "Position must be 3D vector"
        assert self.vel.shape == (3,), "Velocity must be 3D vector"
        assert self.mass > 0, "Mass must be positive"
        assert self.particle_type in ['electron', 'ion'], "Type must be 'electron' or 'ion'"


@dataclass
class SpinState:
    """
    Black hole rotation state.

    Attributes:
        Omega: Angular velocity vector [Ωx, Ωy, Ωz] in rad/s
        L: Angular momentum vector [Lx, Ly, Lz] in kg⋅m²/s
        I: Moment of inertia in kg⋅m²
    """
    Omega: np.ndarray  # [3] array [rad/s]
    L: np.ndarray      # [3] array [kg⋅m²/s]
    I: float           # [kg⋅m²]

    def __post_init__(self):
        """Validate spin state"""
        assert self.Omega.shape == (3,), "Omega must be 3D vector"
        assert self.L.shape == (3,), "L must be 3D vector"
        assert self.I > 0, "Moment of inertia must be positive"

    @property
    def magnitude(self) -> float:
        """Angular velocity magnitude |Ω|"""
        return np.linalg.norm(self.Omega)

    @property
    def angular_momentum_magnitude(self) -> float:
        """Angular momentum magnitude |L|"""
        return np.linalg.norm(self.L)


class RotationDynamics:
    """
    Manages angular momentum transfer and spin evolution for binary black holes.

    Tracks:
    - Angular momentum of individual particles
    - Net torque from rift eruptions
    - Spin evolution: dΩ/dt = τ / I
    - Convergence to opposing rotations
    """

    def __init__(self, config: SimConfig):
        """
        Initialize rotation dynamics.

        Args:
            config: Simulation configuration
        """
        self.config = config

        # Black hole spin states
        self.spin_BH1 = self._init_spin_state(
            config.OMEGA_BH1_MAGNITUDE,
            config.I_MOMENT_BH1,
            axis=np.array([0, 0, 1])  # Default: z-axis
        )

        self.spin_BH2 = self._init_spin_state(
            config.OMEGA_BH2_MAGNITUDE,
            config.I_MOMENT_BH2,
            axis=self._get_BH2_rotation_axis(config.ROTATION_ALIGNMENT)
        )

        # History tracking
        self.rift_events: List[dict] = []
        self.alignment_history: List[float] = []

    def _init_spin_state(
        self,
        omega_magnitude: float,
        I: float,
        axis: np.ndarray
    ) -> SpinState:
        """
        Initialize spin state with given magnitude and axis.

        Args:
            omega_magnitude: |Ω| in units of c/r_g (dimensionless)
            I: Moment of inertia [kg⋅m²]
            axis: Rotation axis (will be normalized)

        Returns:
            SpinState with Ω and L
        """
        # Normalize axis
        axis = axis / np.linalg.norm(axis)

        # Convert Ω from geometric units to rad/s
        # Ω [c/r_g] → Ω [rad/s] requires characteristic timescale
        # For now, use geometric units directly
        Omega = omega_magnitude * axis

        # L = I Ω
        L = I * Omega

        return SpinState(Omega=Omega, L=L, I=I)

    def _get_BH2_rotation_axis(self, alignment: float) -> np.ndarray:
        """
        Compute BH2 rotation axis from alignment parameter.

        alignment = Ω₁·Ω₂/(|Ω₁||Ω₂|) = cos(angle)

        Args:
            alignment: Rotation alignment [-1, 1]
                       -1 = opposing, 0 = perpendicular, +1 = aligned

        Returns:
            Rotation axis for BH2
        """
        # BH1 is along z-axis
        # BH2 makes angle θ = arccos(alignment) with z-axis

        theta = np.arccos(np.clip(alignment, -1, 1))

        # BH2 axis in x-z plane (simple case)
        return np.array([np.sin(theta), 0, np.cos(theta)])

    @staticmethod
    def angular_momentum(r: np.ndarray, p: np.ndarray) -> np.ndarray:
        """
        Compute angular momentum L = r × p.

        Args:
            r: Position vector [x, y, z]
            p: Momentum vector [px, py, pz] = m*v

        Returns:
            Angular momentum vector [Lx, Ly, Lz]

        Lean reference: QFD.Rift.SpinSorting.angular_momentum
        """
        return np.cross(r, p)

    def rotation_alignment(
        self,
        Omega1: np.ndarray,
        Omega2: np.ndarray
    ) -> float:
        """
        Compute rotation alignment parameter.

        alignment = Ω₁·Ω₂ / (|Ω₁| |Ω₂|) = cos(angle)

        Args:
            Omega1: Angular velocity of BH1 [3D vector]
            Omega2: Angular velocity of BH2 [3D vector]

        Returns:
            Alignment in range [-1, 1]
            -1 = opposing (Ω₁ = -Ω₂)
            0 = perpendicular
            +1 = aligned (Ω₁ = Ω₂)

        Lean reference: QFD.Rift.RotationDynamics.rotation_alignment
        """
        norm1 = np.linalg.norm(Omega1)
        norm2 = np.linalg.norm(Omega2)

        if norm1 < 1e-10 or norm2 < 1e-10:
            return 0.0  # Degenerate case

        return np.dot(Omega1, Omega2) / (norm1 * norm2)

    def opposing_rotations(
        self,
        Omega1: np.ndarray,
        Omega2: np.ndarray,
        threshold: float = -0.5
    ) -> bool:
        """
        Check if rotations are opposing.

        Args:
            Omega1: Angular velocity of BH1
            Omega2: Angular velocity of BH2
            threshold: Alignment threshold (default: -0.5)

        Returns:
            True if alignment < threshold (opposing)

        Lean reference: QFD.Rift.RotationDynamics.opposing_rotations
        """
        alignment = self.rotation_alignment(Omega1, Omega2)
        return alignment < threshold

    def compute_net_torque(
        self,
        particles_recaptured: List[Particle],
        particles_escaped: List[Particle],
        BH_position: np.ndarray
    ) -> np.ndarray:
        """
        Compute net torque on black hole from rift eruption.

        τ_net = ∫ L_recaptured dm - ∫ L_escaped dm

        Physical meaning:
        - Recaptured material: Deposits angular momentum → spins up BH
        - Escaped material: Removes angular momentum → spins down BH

        Args:
            particles_recaptured: List of recaptured particles
            particles_escaped: List of escaped particles
            BH_position: Position of black hole center [x, y, z]

        Returns:
            Net torque vector [τx, τy, τz] in kg⋅m²/s²

        Lean reference: QFD.Rift.SpinSorting.net_torque
        """
        torque_recaptured = np.zeros(3)
        for particle in particles_recaptured:
            r = particle.pos - BH_position
            p = particle.mass * particle.vel
            L = self.angular_momentum(r, p)
            torque_recaptured += L  # dL/dt contribution

        torque_escaped = np.zeros(3)
        for particle in particles_escaped:
            r = particle.pos - BH_position
            p = particle.mass * particle.vel
            L = self.angular_momentum(r, p)
            torque_escaped += L

        # Net torque: deposited - removed
        return torque_recaptured - torque_escaped

    def evolve_spin(
        self,
        spin_state: SpinState,
        torque: np.ndarray,
        dt: float
    ) -> SpinState:
        """
        Evolve spin state by one timestep.

        dΩ/dt = τ / I

        Args:
            spin_state: Current spin state
            torque: Net torque [kg⋅m²/s²]
            dt: Time step [s]

        Returns:
            Updated spin state

        Lean reference: QFD.Rift.SpinSorting.angular_velocity_evolution
        """
        # dΩ/dt = τ / I
        dOmega_dt = torque / spin_state.I

        # Update Ω
        Omega_new = spin_state.Omega + dOmega_dt * dt

        # Update L = I Ω
        L_new = spin_state.I * Omega_new

        return SpinState(Omega=Omega_new, L=L_new, I=spin_state.I)

    def check_equilibrium(
        self,
        Omega1: np.ndarray,
        Omega2: np.ndarray,
        epsilon: float = 0.1
    ) -> bool:
        """
        Check if system is in equilibrium (opposing rotations).

        Equilibrium: |alignment - (-1)| < epsilon

        Args:
            Omega1: Angular velocity of BH1
            Omega2: Angular velocity of BH2
            epsilon: Tolerance (default: 0.1)

        Returns:
            True if system is in equilibrium

        Lean reference: QFD.Rift.SpinSorting.spin_sorting_equilibrium
        """
        alignment = self.rotation_alignment(Omega1, Omega2)
        return abs(alignment - (-1.0)) < epsilon

    def track_rift_event(
        self,
        particles_ejected: List[Particle],
        particles_escaped: List[Particle],
        particles_recaptured: List[Particle],
        BH1_pos: np.ndarray,
        BH2_pos: np.ndarray
    ) -> dict:
        """
        Track a single rift eruption event.

        Args:
            particles_ejected: All particles ejected in rift
            particles_escaped: Particles that escaped to infinity
            particles_recaptured: Particles recaptured by BH
            BH1_pos: Position of BH1
            BH2_pos: Position of BH2

        Returns:
            Event summary dictionary
        """
        # Compute torques
        torque_BH1 = self.compute_net_torque(
            particles_recaptured,
            particles_escaped,
            BH1_pos
        )

        # Current alignment
        alignment = self.rotation_alignment(
            self.spin_BH1.Omega,
            self.spin_BH2.Omega
        )

        # Escape fraction
        escape_fraction = len(particles_escaped) / max(len(particles_ejected), 1)

        event = {
            "rift_index": len(self.rift_events),
            "num_ejected": len(particles_ejected),
            "num_escaped": len(particles_escaped),
            "num_recaptured": len(particles_recaptured),
            "escape_fraction": escape_fraction,
            "torque_BH1": torque_BH1,
            "alignment_before": alignment,
            "Omega_BH1_magnitude": self.spin_BH1.magnitude,
            "Omega_BH2_magnitude": self.spin_BH2.magnitude,
        }

        self.rift_events.append(event)
        self.alignment_history.append(alignment)

        return event

    def get_convergence_metrics(self) -> dict:
        """
        Compute metrics for convergence to opposing rotations.

        Returns:
            Dictionary with convergence metrics
        """
        if len(self.alignment_history) == 0:
            return {
                "converged": False,
                "current_alignment": self.rotation_alignment(
                    self.spin_BH1.Omega,
                    self.spin_BH2.Omega
                ),
                "num_rifts": 0
            }

        current_alignment = self.alignment_history[-1]
        converged = self.check_equilibrium(
            self.spin_BH1.Omega,
            self.spin_BH2.Omega
        )

        # Convergence rate (if enough history)
        if len(self.alignment_history) > 5:
            recent_change = abs(
                self.alignment_history[-1] - self.alignment_history[-5]
            )
            convergence_rate = recent_change / 5  # per rift event
        else:
            convergence_rate = None

        return {
            "converged": converged,
            "current_alignment": current_alignment,
            "target_alignment": -1.0,
            "distance_to_equilibrium": abs(current_alignment - (-1.0)),
            "num_rifts": len(self.rift_events),
            "convergence_rate": convergence_rate,
            "alignment_history": self.alignment_history.copy()
        }


def compute_angular_gradient(
    phi_field: np.ndarray,
    theta_grid: np.ndarray,
    r: float
) -> np.ndarray:
    """
    Compute angular gradient ∂φ/∂θ at fixed radius.

    For opposing rotations: ∂φ₁/∂θ + ∂φ₂/∂θ ≈ 0 (cancellation)

    Args:
        phi_field: Scalar field values on (θ, φ_angle) grid
        theta_grid: θ values [0, π]
        r: Radius at which to compute gradient

    Returns:
        ∂φ/∂θ array

    Lean reference: QFD.Rift.RotationDynamics.angular_gradient_cancellation
    """
    # Finite difference: ∂φ/∂θ ≈ (φ[i+1] - φ[i-1]) / (2Δθ)
    dtheta = theta_grid[1] - theta_grid[0]

    grad_theta = np.zeros_like(phi_field)
    grad_theta[1:-1] = (phi_field[2:] - phi_field[:-2]) / (2 * dtheta)

    # Boundary conditions (one-sided differences)
    grad_theta[0] = (phi_field[1] - phi_field[0]) / dtheta
    grad_theta[-1] = (phi_field[-1] - phi_field[-2]) / dtheta

    return grad_theta


# ========================================
# TESTING / VALIDATION
# ========================================

if __name__ == "__main__":
    """Test rotation dynamics implementation"""

    print("=" * 80)
    print("ROTATION DYNAMICS: Unit Tests")
    print("=" * 80)
    print()

    config = SimConfig()
    config.__post_init__()

    rot = RotationDynamics(config)

    # Test 1: Angular momentum
    print("Test 1: Angular Momentum")
    r = np.array([1.0, 0.0, 0.0])
    p = np.array([0.0, 1.0, 0.0])
    L = rot.angular_momentum(r, p)
    print(f"  r = {r}")
    print(f"  p = {p}")
    print(f"  L = r × p = {L}")
    print(f"  Expected: [0, 0, 1]")
    assert np.allclose(L, [0, 0, 1]), "Angular momentum test failed"
    print("  ✅ PASSED")
    print()

    # Test 2: Rotation alignment
    print("Test 2: Rotation Alignment")
    Omega1 = np.array([0, 0, 1])
    Omega2_aligned = np.array([0, 0, 1])
    Omega2_opposing = np.array([0, 0, -1])
    Omega2_perp = np.array([1, 0, 0])

    align_same = rot.rotation_alignment(Omega1, Omega2_aligned)
    align_opp = rot.rotation_alignment(Omega1, Omega2_opposing)
    align_perp = rot.rotation_alignment(Omega1, Omega2_perp)

    print(f"  Ω₁ = {Omega1}")
    print(f"  Ω₂ (aligned) = {Omega2_aligned}, alignment = {align_same:.2f}")
    print(f"  Ω₂ (opposing) = {Omega2_opposing}, alignment = {align_opp:.2f}")
    print(f"  Ω₂ (perpendicular) = {Omega2_perp}, alignment = {align_perp:.2f}")

    assert np.isclose(align_same, 1.0), "Aligned case failed"
    assert np.isclose(align_opp, -1.0), "Opposing case failed"
    assert np.isclose(align_perp, 0.0), "Perpendicular case failed"
    print("  ✅ PASSED")
    print()

    # Test 3: Opposing rotations check
    print("Test 3: Opposing Rotations Check")
    is_opp = rot.opposing_rotations(Omega1, Omega2_opposing)
    is_not_opp = rot.opposing_rotations(Omega1, Omega2_aligned)
    print(f"  opposing_rotations(Ω₁, -Ω₁) = {is_opp} (expected True)")
    print(f"  opposing_rotations(Ω₁, Ω₁) = {is_not_opp} (expected False)")
    assert is_opp == True, "Opposing check failed"
    assert is_not_opp == False, "Aligned check failed"
    print("  ✅ PASSED")
    print()

    # Test 4: Equilibrium check
    print("Test 4: Equilibrium Check")
    equilibrium = rot.check_equilibrium(Omega1, Omega2_opposing, epsilon=0.1)
    not_equilibrium = rot.check_equilibrium(Omega1, Omega2_aligned, epsilon=0.1)
    print(f"  check_equilibrium(Ω₁, -Ω₁) = {equilibrium} (expected True)")
    print(f"  check_equilibrium(Ω₁, Ω₁) = {not_equilibrium} (expected False)")
    assert equilibrium == True, "Equilibrium check failed"
    assert not_equilibrium == False, "Non-equilibrium check failed"
    print("  ✅ PASSED")
    print()

    print("=" * 80)
    print("✅ ALL TESTS PASSED")
    print("=" * 80)
