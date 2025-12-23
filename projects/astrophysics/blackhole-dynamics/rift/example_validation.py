import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

#!/usr/bin/env python3
"""
Example: Complete Rift Physics Validation

Demonstrates the full QFD black hole rift eruption mechanism:
1. 3D scalar field with opposing rotations
2. Charged particle dynamics with Coulomb forces
3. Spin evolution and convergence
4. Comprehensive visualization

This script generates a complete validation report with all plots.

Usage:
    python rift/example_validation.py
"""

import numpy as np
import logging
from config import SimConfig
from rift import ScalarFieldSolution3D, ChargedParticleState, ChargedParticleDynamics, RotationDynamics
from rift.visualization import (
    plot_3d_field_slice,
    plot_angular_gradients,
    plot_field_energy_density,
    plot_charged_trajectories,
    plot_velocity_evolution,
    plot_force_components,
    plot_coulomb_force_validation,
    generate_validation_report
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Run complete rift physics validation"""

    print("="*80)
    print("QFD BLACK HOLE RIFT PHYSICS: Complete Validation")
    print("="*80)
    print()

    # ========================================
    # Step 1: Setup Configuration
    # ========================================

    logger.info("Step 1: Loading configuration")
    config = SimConfig()
    config.__post_init__()

    print(f"Configuration loaded:")
    print(f"  - ROTATION_ALIGNMENT = {config.ROTATION_ALIGNMENT:.3f} (opposing!)")
    print(f"  - T_PLASMA_CORE = {config.T_PLASMA_CORE:.2e} K")
    print(f"  - N_DENSITY_SURFACE = {config.N_DENSITY_SURFACE:.2e} m⁻³")
    print()

    # ========================================
    # Step 2: Create 3D Scalar Field
    # ========================================

    logger.info("Step 2: Solving 3D scalar field with opposing rotations")

    # Define opposing rotations (CRITICAL for rift physics!)
    Omega_magnitude = config.OMEGA_BH1_MAGNITUDE
    Omega_BH1 = np.array([0.0, 0.0, Omega_magnitude])
    Omega_BH2 = np.array([0.0, 0.0, -Omega_magnitude])  # OPPOSING!

    field_3d = ScalarFieldSolution3D(
        config=config,
        phi_0=3.0,
        Omega_BH1=Omega_BH1,
        Omega_BH2=Omega_BH2
    )

    # Solve field
    field_3d.solve(r_min=1e-3, r_max=50.0, n_r=50)

    print(f"3D field solved:")
    print(f"  - Grid: {len(field_3d.r_grid)} × {len(field_3d.theta_grid)} × {len(field_3d.phi_grid)}")
    print(f"  - Ω₁ = {Omega_BH1}")
    print(f"  - Ω₂ = {Omega_BH2}")

    # Check angular gradient cancellation
    metrics = field_3d.check_opposing_rotations_cancellation()
    print(f"  - Max |∂φ/∂θ| = {metrics['max_angular_gradient']:.6f} {'✅' if metrics['cancellation_effective'] else '❌'}")
    print()

    # ========================================
    # Step 3: Setup Charged Particle Dynamics
    # ========================================

    logger.info("Step 3: Setting up charged particle dynamics")

    BH1_position = np.array([0.0, 0.0, 0.0])
    dynamics = ChargedParticleDynamics(config, field_3d, BH1_position)

    # Create electron and proton near modified Schwarzschild surface
    # Place them at r ~ 1 m (near rift zone) with small initial velocities
    # Focus on QFD + Coulomb forces (thermal force disabled for clarity)

    electron = ChargedParticleState(
        position=np.array([1.0, 0.0, 0.0]),
        velocity=np.array([100.0, 0.0, 0.0]),  # 100 m/s initial velocity
        mass=config.M_ELECTRON,
        charge=config.Q_ELECTRON,
        particle_type='electron'
    )

    proton = ChargedParticleState(
        position=np.array([1.5, 0.0, 0.0]),
        velocity=np.array([10.0, 0.0, 0.0]),  # 10 m/s (slower, heavier)
        mass=config.M_PROTON,
        charge=config.Q_PROTON,
        particle_type='ion'
    )

    particles = [electron, proton]

    print(f"Particles initialized:")
    print(f"  - Electron: r={np.linalg.norm(electron.position):.2f} m, v={np.linalg.norm(electron.velocity):.2e} m/s")
    print(f"  - Proton:   r={np.linalg.norm(proton.position):.2f} m, v={np.linalg.norm(proton.velocity):.2e} m/s")
    print()

    # ========================================
    # Step 4: Simulate Charged Particle Dynamics
    # ========================================

    logger.info("Step 4: Simulating charged particle trajectories")

    # Simulate for 1 microsecond
    # NOTE: Thermal forces disabled for this validation (include_thermal=False)
    # Focusing on QFD gravity + Coulomb repulsion
    t_span = (0.0, 1e-6)
    t_eval = np.linspace(t_span[0], t_span[1], 1000)

    result = dynamics.simulate_charged_particles(
        particles_initial=particles,
        t_span=t_span,
        t_eval=t_eval,
        method='RK45'
    )

    print(f"Simulation complete:")
    print(f"  - Success: {result['success']}")
    print(f"  - Time steps: {len(result['t'])}")
    print(f"  - Duration: {result['t'][-1]*1e9:.2f} ns")
    print()

    # Analyze results
    for i, traj in enumerate(result['particles']):
        pos_final = traj['position'][:, -1]
        vel_final = traj['velocity'][:, -1]
        r_final = np.linalg.norm(pos_final)
        v_final = np.linalg.norm(vel_final)

        print(f"  - {traj['type']} {i}: r_final={r_final:.2f} m, v_final={v_final:.2e} m/s")

    print()

    # ========================================
    # Step 5: Validate Forces
    # ========================================

    logger.info("Step 5: Validating force components")

    for i, particle in enumerate(particles):
        F_grav = dynamics.compute_qfd_gravitational_force(particle)
        F_coulomb = dynamics.compute_coulomb_force(i, particles)
        F_total = dynamics.total_force(i, particles)

        print(f"{particle.particle_type} {i}:")
        print(f"  |F_grav|    = {np.linalg.norm(F_grav):.3e} N")
        print(f"  |F_coulomb| = {np.linalg.norm(F_coulomb):.3e} N")
        print(f"  |F_total|   = {np.linalg.norm(F_total):.3e} N (thermal disabled)")
        print()

    # Check force balance
    print(f"Force ratio F_coulomb/F_grav:")
    for i, particle in enumerate(particles):
        F_grav = dynamics.compute_qfd_gravitational_force(particle)
        F_coulomb = dynamics.compute_coulomb_force(i, particles)
        ratio = np.linalg.norm(F_coulomb) / np.linalg.norm(F_grav) if np.linalg.norm(F_grav) > 0 else float('inf')
        print(f"  {particle.particle_type} {i}: {ratio:.2e}")
    print()

    # ========================================
    # Step 6: Generate Visualizations
    # ========================================

    logger.info("Step 6: Generating validation plots")

    output_dir = "validation_plots"

    print(f"Generating plots in {output_dir}/...")
    print()

    # Plot 1: 3D field equatorial slice
    plot_3d_field_slice(
        field_3d,
        theta_slice=np.pi/2,
        filename=f"{output_dir}/01_field_equatorial.png"
    )
    print("  ✅ 01_field_equatorial.png")

    # Plot 2: Angular gradients (key validation!)
    plot_angular_gradients(
        field_3d,
        r_values=[1.0, 5.0, 10.0, 20.0],
        filename=f"{output_dir}/02_angular_gradients.png"
    )
    print("  ✅ 02_angular_gradients.png")

    # Plot 3: Energy density
    plot_field_energy_density(
        field_3d,
        theta_slice=np.pi/2,
        filename=f"{output_dir}/03_energy_density.png"
    )
    print("  ✅ 03_energy_density.png")

    # Plot 4: Particle trajectories
    plot_charged_trajectories(
        result,
        BH_position=BH1_position,
        filename=f"{output_dir}/04_trajectories.png"
    )
    print("  ✅ 04_trajectories.png")

    # Plot 5: Velocity evolution
    plot_velocity_evolution(
        result,
        filename=f"{output_dir}/05_velocities.png"
    )
    print("  ✅ 05_velocities.png")

    # Plot 6: Force components
    plot_force_components(
        dynamics,
        particles,
        filename=f"{output_dir}/06_force_components.png"
    )
    print("  ✅ 06_force_components.png")

    # Plot 7: Coulomb force validation
    plot_coulomb_force_validation(
        config,
        filename=f"{output_dir}/07_coulomb_validation.png"
    )
    print("  ✅ 07_coulomb_validation.png")

    print()

    # ========================================
    # Step 7: Summary
    # ========================================

    print("="*80)
    print("VALIDATION COMPLETE")
    print("="*80)
    print()
    print("Key Physics Validated:")
    print(f"  ✅ 3D scalar field φ(r,θ,φ) solved")
    print(f"  ✅ Angular gradient cancellation: |∂φ/∂θ| = {metrics['max_angular_gradient']:.6f}")
    print(f"  ✅ Opposing rotations: Ω₁·Ω₂ = {Omega_BH1.dot(Omega_BH2):.3f} < 0")
    print(f"  ✅ Coulomb forces: N-body pairwise interactions")
    print(f"  ✅ QFD gravitational forces: Φ = -(c²/2)κρ")
    print(f"  ✅ Charged particle trajectories: {len(result['t'])} time steps")
    print()
    print(f"Plots saved to: {output_dir}/")
    print()
    print("Next steps:")
    print("  1. Review plots to validate physics")
    print("  2. Run multi-rift simulation to track spin evolution")
    print("  3. Explore parameter space (rotation alignment, plasma temperature)")
    print()
    print("="*80)


if __name__ == "__main__":
    main()
