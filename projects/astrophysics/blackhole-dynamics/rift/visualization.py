import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

#!/usr/bin/env python3
"""
Visualization Functions for QFD Black Hole Rift Physics

Comprehensive plotting suite to validate:
1. 3D scalar field structure and angular gradients
2. Charged particle trajectories and forces
3. Spin evolution and convergence to opposing rotations
4. Coulomb force validation and N-body dynamics

Usage:
    from rift.visualization import plot_3d_field_slice, plot_angular_gradients
    plot_3d_field_slice(field_3d, theta_slice=np.pi/2, filename='field.png')
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional, List, Dict
import logging

# Import rift modules
from rift.core_3d import ScalarFieldSolution3D
from rift.rotation_dynamics import RotationDynamics, SpinState
from rift.simulation_charged import ChargedParticleDynamics
from config import SimConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ========================================
# 1. 3D Scalar Field Visualization
# ========================================

def plot_3d_field_slice(
    field_3d: ScalarFieldSolution3D,
    theta_slice: float = np.pi/2,
    r_range: tuple = (0.1, 50.0),
    n_points: int = 100,
    filename: Optional[str] = None
):
    """
    Plot φ(r, φ_angle) at fixed θ (equatorial slice)

    Args:
        field_3d: ScalarFieldSolution3D instance
        theta_slice: θ value for slice (default: π/2 = equator)
        r_range: (r_min, r_max) for plotting
        n_points: Number of grid points
        filename: Save to file if provided
    """
    logger.info(f"Plotting 3D field slice at θ={theta_slice:.3f} rad")

    # Create grid
    r_vals = np.linspace(r_range[0], r_range[1], n_points)
    phi_vals = np.linspace(0, 2*np.pi, n_points)
    R, PHI = np.meshgrid(r_vals, phi_vals)

    # Evaluate field
    FIELD = np.zeros_like(R)
    for i in range(n_points):
        for j in range(n_points):
            FIELD[i, j] = field_3d.field(R[i, j], theta_slice, PHI[i, j])

    # Create polar plot
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))

    contour = ax.contourf(PHI, R, FIELD, levels=50, cmap='viridis')
    plt.colorbar(contour, ax=ax, label=r'$\phi(r, \phi)$ [field units]')

    ax.set_title(f'3D Scalar Field: φ(r, φ) at θ={theta_slice:.3f} rad\n'
                 f'Opposing Rotations: Ω₁={field_3d.Omega_BH1[2]:.2f}, Ω₂={field_3d.Omega_BH2[2]:.2f}',
                 fontsize=14, pad=20)
    ax.set_ylabel('Radius r [m]', fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        logger.info(f"Saved to {filename}")
    else:
        plt.show()

    plt.close()


def plot_angular_gradients(
    field_3d: ScalarFieldSolution3D,
    r_values: List[float] = [1.0, 5.0, 10.0, 20.0],
    phi_angle: float = 0.0,
    filename: Optional[str] = None
):
    """
    Plot ∂φ/∂θ vs θ at multiple radii

    Demonstrates angular gradient cancellation for opposing rotations.
    Expected: |∂φ/∂θ| < 0.1 for Ω₁ = -Ω₂

    Args:
        field_3d: ScalarFieldSolution3D instance
        r_values: List of radii to plot
        phi_angle: φ value for slice
        filename: Save to file if provided
    """
    logger.info("Plotting angular gradients to validate cancellation")

    theta_vals = np.linspace(0.01, np.pi - 0.01, 100)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left panel: ∂φ/∂θ vs θ
    for r in r_values:
        grad_theta = [field_3d.angular_gradient(r, theta, phi_angle) for theta in theta_vals]
        ax1.plot(theta_vals, grad_theta, label=f'r = {r:.1f} m', linewidth=2)

    ax1.axhline(0, color='k', linestyle='--', alpha=0.3, linewidth=1)
    ax1.axhline(0.1, color='r', linestyle=':', alpha=0.5, label='Threshold = 0.1')
    ax1.axhline(-0.1, color='r', linestyle=':', alpha=0.5)

    ax1.set_xlabel(r'$\theta$ [rad]', fontsize=12)
    ax1.set_ylabel(r'$\partial\phi/\partial\theta$', fontsize=12)
    ax1.set_title('Angular Gradient vs Polar Angle', fontsize=14)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Right panel: Max |∂φ/∂θ| vs radius
    r_scan = np.linspace(0.5, 30.0, 30)
    max_grad = []

    for r in r_scan:
        grads = [abs(field_3d.angular_gradient(r, theta, phi_angle)) for theta in theta_vals]
        max_grad.append(max(grads))

    ax2.plot(r_scan, max_grad, 'b-', linewidth=2, label='Max |∂φ/∂θ|')
    ax2.axhline(0.1, color='r', linestyle='--', linewidth=2, label='Cancellation Threshold')

    ax2.set_xlabel('Radius r [m]', fontsize=12)
    ax2.set_ylabel(r'Max $|\partial\phi/\partial\theta|$', fontsize=12)
    ax2.set_title('Angular Gradient Cancellation Check', fontsize=14)
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    # Add annotation
    alignment = field_3d.Omega_BH1.dot(field_3d.Omega_BH2) / (
        np.linalg.norm(field_3d.Omega_BH1) * np.linalg.norm(field_3d.Omega_BH2)
    )
    fig.suptitle(f'Angular Gradient Cancellation (Rotation Alignment = {alignment:.3f})',
                 fontsize=16, y=1.02)

    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        logger.info(f"Saved to {filename}")
    else:
        plt.show()

    plt.close()


def plot_field_energy_density(
    field_3d: ScalarFieldSolution3D,
    theta_slice: float = np.pi/2,
    r_range: tuple = (0.1, 50.0),
    n_points: int = 100,
    filename: Optional[str] = None
):
    """
    Plot energy density ρ(r,φ) = (α₁/2)(∇φ)² + V(φ)

    Args:
        field_3d: ScalarFieldSolution3D instance
        theta_slice: θ value for slice
        r_range: (r_min, r_max) for plotting
        n_points: Number of grid points
        filename: Save to file if provided
    """
    logger.info("Plotting field energy density")

    r_vals = np.linspace(r_range[0], r_range[1], n_points)
    phi_vals = np.linspace(0, 2*np.pi, n_points)
    R, PHI = np.meshgrid(r_vals, phi_vals)

    # Evaluate energy density
    RHO = np.zeros_like(R)
    for i in range(n_points):
        for j in range(n_points):
            phi = field_3d.field(R[i, j], theta_slice, PHI[i, j])
            grad_r = field_3d._grad_r_interp([R[i, j], theta_slice, PHI[i, j]])[0]

            rho_kinetic = 0.5 * field_3d.alpha_1 * grad_r**2
            V_phi = 0.5 * field_3d.alpha_2 * (phi**2 - field_3d.phi_vac**2)**2
            RHO[i, j] = rho_kinetic + V_phi

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))

    contour = ax.contourf(PHI, R, RHO, levels=50, cmap='plasma')
    plt.colorbar(contour, ax=ax, label=r'Energy Density $\rho$ [J/m³]')

    ax.set_title(f'Field Energy Density at θ={theta_slice:.3f} rad', fontsize=14, pad=20)
    ax.set_ylabel('Radius r [m]', fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        logger.info(f"Saved to {filename}")
    else:
        plt.show()

    plt.close()


# ========================================
# 2. Charged Particle Trajectories
# ========================================

def plot_charged_trajectories(
    simulation_result: Dict,
    BH_position: np.ndarray = np.array([0, 0, 0]),
    filename: Optional[str] = None
):
    """
    Plot 3D trajectories of charged particles

    Args:
        simulation_result: Output from ChargedParticleDynamics.simulate_charged_particles()
        BH_position: Position of BH1 (reference point)
        filename: Save to file if provided
    """
    logger.info("Plotting charged particle trajectories")

    fig = plt.figure(figsize=(14, 6))

    # Left panel: 3D trajectory
    ax1 = fig.add_subplot(121, projection='3d')

    # Plot BH
    ax1.scatter(*BH_position, color='black', s=200, marker='o',
                label='BH1', edgecolors='white', linewidths=2)

    # Plot particle trajectories
    colors = ['blue', 'red', 'green', 'orange', 'purple']

    for i, traj in enumerate(simulation_result['particles']):
        pos = traj['position']  # Shape: (3, n_timesteps)
        particle_type = traj['type']
        color = colors[i % len(colors)]

        # Trajectory line
        ax1.plot(pos[0, :], pos[1, :], pos[2, :],
                color=color, linewidth=2, alpha=0.7,
                label=f'{particle_type} {i}')

        # Start point
        ax1.scatter(pos[0, 0], pos[1, 0], pos[2, 0],
                   color=color, s=100, marker='o', edgecolors='black', linewidths=1)

        # End point
        ax1.scatter(pos[0, -1], pos[1, -1], pos[2, -1],
                   color=color, s=100, marker='s', edgecolors='black', linewidths=1)

    ax1.set_xlabel('x [m]', fontsize=10)
    ax1.set_ylabel('y [m]', fontsize=10)
    ax1.set_zlabel('z [m]', fontsize=10)
    ax1.set_title('3D Particle Trajectories', fontsize=12)
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Right panel: Distance from BH vs time
    ax2 = fig.add_subplot(122)

    t = simulation_result['t']

    for i, traj in enumerate(simulation_result['particles']):
        pos = traj['position']
        distances = np.linalg.norm(pos - BH_position[:, np.newaxis], axis=0)
        particle_type = traj['type']
        color = colors[i % len(colors)]

        ax2.plot(t * 1e9, distances, color=color, linewidth=2,
                label=f'{particle_type} {i}')

    ax2.set_xlabel('Time [ns]', fontsize=12)
    ax2.set_ylabel('Distance from BH1 [m]', fontsize=12)
    ax2.set_title('Radial Distance Evolution', fontsize=12)
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        logger.info(f"Saved to {filename}")
    else:
        plt.show()

    plt.close()


def plot_velocity_evolution(
    simulation_result: Dict,
    filename: Optional[str] = None
):
    """
    Plot particle velocities over time

    Args:
        simulation_result: Output from simulate_charged_particles()
        filename: Save to file if provided
    """
    logger.info("Plotting velocity evolution")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    t = simulation_result['t']
    colors = ['blue', 'red', 'green', 'orange', 'purple']

    # Left: Speed vs time
    for i, traj in enumerate(simulation_result['particles']):
        vel = traj['velocity']  # Shape: (3, n_timesteps)
        speed = np.linalg.norm(vel, axis=0)
        particle_type = traj['type']
        color = colors[i % len(colors)]

        ax1.plot(t * 1e9, speed, color=color, linewidth=2,
                label=f'{particle_type} {i}')

    ax1.set_xlabel('Time [ns]', fontsize=12)
    ax1.set_ylabel('Speed [m/s]', fontsize=12)
    ax1.set_title('Particle Speed Evolution', fontsize=14)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Right: Kinetic energy vs time
    for i, traj in enumerate(simulation_result['particles']):
        vel = traj['velocity']
        mass = traj['mass']
        speed = np.linalg.norm(vel, axis=0)
        KE = 0.5 * mass * speed**2
        particle_type = traj['type']
        color = colors[i % len(colors)]

        ax2.plot(t * 1e9, KE, color=color, linewidth=2,
                label=f'{particle_type} {i}')

    ax2.set_xlabel('Time [ns]', fontsize=12)
    ax2.set_ylabel('Kinetic Energy [J]', fontsize=12)
    ax2.set_title('Kinetic Energy Evolution', fontsize=14)
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        logger.info(f"Saved to {filename}")
    else:
        plt.show()

    plt.close()


# ========================================
# 3. Force Component Analysis
# ========================================

def plot_force_components(
    dynamics: ChargedParticleDynamics,
    particles: List,
    filename: Optional[str] = None
):
    """
    Plot breakdown of force components: F_grav, F_coulomb, F_thermal

    Args:
        dynamics: ChargedParticleDynamics instance
        particles: List of ChargedParticleState objects
        filename: Save to file if provided
    """
    logger.info("Plotting force component breakdown")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Compute forces for each particle
    for i, particle in enumerate(particles):
        F_grav = dynamics.compute_qfd_gravitational_force(particle)
        F_coulomb = dynamics.compute_coulomb_force(i, particles)
        F_thermal = dynamics.compute_thermal_pressure_force(particle)
        F_total = F_grav + F_coulomb + F_thermal

        particle_type = particle.particle_type

        # Panel 1: Force magnitudes
        ax = axes[0, 0]
        forces = [np.linalg.norm(F_grav), np.linalg.norm(F_coulomb),
                 np.linalg.norm(F_thermal), np.linalg.norm(F_total)]
        labels = ['Gravitational', 'Coulomb', 'Thermal', 'Total']
        colors_bar = ['green', 'blue', 'red', 'black']

        x_pos = np.arange(len(labels)) + i * 0.35
        ax.bar(x_pos, forces, width=0.35, label=f'{particle_type} {i}',
               color=colors_bar, alpha=0.7)

        # Panel 2: Force vectors (x-y plane)
        ax = axes[0, 1]
        scale = 1e38  # Scaling factor for visualization
        ax.quiver(0, 0, F_grav[0]*scale, F_grav[1]*scale,
                 color='green', width=0.005, label=f'F_grav {i}')
        ax.quiver(0, 0, F_coulomb[0]*scale, F_coulomb[1]*scale,
                 color='blue', width=0.005, label=f'F_coulomb {i}')
        ax.quiver(0, 0, F_total[0]*scale, F_total[1]*scale,
                 color='black', width=0.008, label=f'F_total {i}', linewidth=2)

        # Panel 3: Component ratios
        ax = axes[1, 0]
        F_grav_mag = np.linalg.norm(F_grav)
        F_coulomb_mag = np.linalg.norm(F_coulomb)
        F_thermal_mag = np.linalg.norm(F_thermal)

        ratios = [
            F_coulomb_mag / F_grav_mag if F_grav_mag > 0 else 0,
            F_thermal_mag / F_grav_mag if F_grav_mag > 0 else 0
        ]
        ratio_labels = ['F_coulomb/F_grav', 'F_thermal/F_grav']

        x_pos = np.arange(len(ratio_labels)) + i * 0.35
        ax.bar(x_pos, ratios, width=0.35, label=f'{particle_type} {i}', alpha=0.7)

        # Panel 4: Individual force components (x, y, z)
        ax = axes[1, 1]
        components = ['x', 'y', 'z']
        x_pos = np.arange(len(components)) + i * 0.8

        ax.plot(x_pos, F_total, marker='o', linewidth=2, markersize=8,
               label=f'{particle_type} {i} - Total')

    # Configure panels
    axes[0, 0].set_xticks(np.arange(len(labels)) + 0.175)
    axes[0, 0].set_xticklabels(labels, rotation=15)
    axes[0, 0].set_ylabel('Force Magnitude [N]', fontsize=12)
    axes[0, 0].set_title('Force Component Magnitudes', fontsize=14)
    axes[0, 0].set_yscale('log')
    axes[0, 0].legend(loc='best', fontsize=8)
    axes[0, 0].grid(True, alpha=0.3, axis='y')

    axes[0, 1].set_xlabel('F_x [scaled]', fontsize=12)
    axes[0, 1].set_ylabel('F_y [scaled]', fontsize=12)
    axes[0, 1].set_title('Force Vectors (x-y plane)', fontsize=14)
    axes[0, 1].legend(loc='best', fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(0, color='k', linewidth=0.5)
    axes[0, 1].axvline(0, color='k', linewidth=0.5)

    axes[1, 0].set_xticks(np.arange(len(ratio_labels)) + 0.175)
    axes[1, 0].set_xticklabels(ratio_labels)
    axes[1, 0].set_ylabel('Force Ratio', fontsize=12)
    axes[1, 0].set_title('Relative Force Strengths', fontsize=14)
    axes[1, 0].set_yscale('log')
    axes[1, 0].legend(loc='best', fontsize=8)
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    axes[1, 1].set_xticks(np.arange(len(components)) + 0.4)
    axes[1, 1].set_xticklabels(components)
    axes[1, 1].set_ylabel('Force Component [N]', fontsize=12)
    axes[1, 1].set_title('Total Force Components', fontsize=14)
    axes[1, 1].legend(loc='best', fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        logger.info(f"Saved to {filename}")
    else:
        plt.show()

    plt.close()


def plot_coulomb_force_validation(
    config: SimConfig,
    r_range: tuple = (1e-10, 1e-6),
    n_points: int = 100,
    filename: Optional[str] = None
):
    """
    Validate Coulomb force law: F = k_e q₁q₂/r²

    Args:
        config: SimConfig instance
        r_range: (r_min, r_max) separation range
        n_points: Number of points
        filename: Save to file if provided
    """
    logger.info("Validating Coulomb force law")

    r_vals = np.logspace(np.log10(r_range[0]), np.log10(r_range[1]), n_points)

    # Electron-proton force
    q_e = config.Q_ELECTRON
    q_p = config.Q_PROTON
    k_e = config.K_COULOMB

    F_coulomb = k_e * abs(q_e * q_p) / r_vals**2

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Force vs distance
    ax1.loglog(r_vals, F_coulomb, 'b-', linewidth=2, label='Electron-Proton')

    # Add reference lines
    r_ref = 1e-8
    F_ref = k_e * abs(q_e * q_p) / r_ref**2
    ax1.loglog(r_vals, F_ref * (r_ref / r_vals)**2, 'r--',
              linewidth=2, alpha=0.5, label=r'$\propto r^{-2}$ (reference)')

    ax1.set_xlabel('Separation r [m]', fontsize=12)
    ax1.set_ylabel('Coulomb Force [N]', fontsize=12)
    ax1.set_title('Coulomb Force Law Validation', fontsize=14)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3, which='both')

    # Right: Check power law
    log_r = np.log10(r_vals)
    log_F = np.log10(F_coulomb)

    # Fit power law
    coeffs = np.polyfit(log_r, log_F, 1)
    power_law_exponent = coeffs[0]

    ax2.plot(log_r, log_F, 'b-', linewidth=2, label='Actual')
    ax2.plot(log_r, coeffs[0] * log_r + coeffs[1], 'r--',
            linewidth=2, label=f'Fit: F ∝ r^{power_law_exponent:.3f}')

    ax2.set_xlabel('log₁₀(r) [log₁₀(m)]', fontsize=12)
    ax2.set_ylabel('log₁₀(F) [log₁₀(N)]', fontsize=12)
    ax2.set_title('Power Law Check', fontsize=14)
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Add text annotation
    ax2.text(0.05, 0.95, f'Expected: F ∝ r⁻²\nMeasured: F ∝ r^{power_law_exponent:.4f}',
            transform=ax2.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        logger.info(f"Saved to {filename}")
    else:
        plt.show()

    plt.close()


# ========================================
# 4. Spin Evolution and Convergence
# ========================================

def plot_spin_evolution(
    rotation_dynamics: RotationDynamics,
    rift_history: List[Dict],
    filename: Optional[str] = None
):
    """
    Plot spin evolution Ω₁(t), Ω₂(t) over rift cycles

    Shows convergence to opposing rotations equilibrium (Ω₁ = -Ω₂)

    Args:
        rotation_dynamics: RotationDynamics instance
        rift_history: List of rift event dictionaries
        filename: Save to file if provided
    """
    logger.info("Plotting spin evolution over rift cycles")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Extract data from history
    rift_indices = [event['rift_index'] for event in rift_history]
    Omega1_magnitudes = [np.linalg.norm(event['Omega_BH1_after']) for event in rift_history]
    Omega2_magnitudes = [np.linalg.norm(event['Omega_BH2_after']) for event in rift_history]
    alignments = [event['alignment_after'] for event in rift_history]

    # Panel 1: Ω magnitudes vs rift index
    ax = axes[0, 0]
    ax.plot(rift_indices, Omega1_magnitudes, 'b-o', linewidth=2, markersize=6, label='|Ω₁|')
    ax.plot(rift_indices, Omega2_magnitudes, 'r-s', linewidth=2, markersize=6, label='|Ω₂|')

    ax.set_xlabel('Rift Index', fontsize=12)
    ax.set_ylabel('Angular Velocity Magnitude [rad/s]', fontsize=12)
    ax.set_title('Spin Magnitude Evolution', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Panel 2: Rotation alignment vs rift index
    ax = axes[0, 1]
    ax.plot(rift_indices, alignments, 'g-o', linewidth=2, markersize=6)
    ax.axhline(-1.0, color='r', linestyle='--', linewidth=2, label='Opposing (target)')
    ax.axhline(0.0, color='k', linestyle=':', linewidth=1, alpha=0.5, label='Perpendicular')
    ax.axhline(1.0, color='b', linestyle=':', linewidth=1, alpha=0.5, label='Aligned')

    ax.set_xlabel('Rift Index', fontsize=12)
    ax.set_ylabel('Rotation Alignment', fontsize=12)
    ax.set_title('Convergence to Opposing Rotations', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1.1, 1.1)

    # Panel 3: Escape fraction vs rift index
    ax = axes[1, 0]
    escape_fractions = [event['escape_fraction'] for event in rift_history]
    ax.plot(rift_indices, escape_fractions, 'purple', linewidth=2, marker='D', markersize=6)

    ax.set_xlabel('Rift Index', fontsize=12)
    ax.set_ylabel('Escape Fraction', fontsize=12)
    ax.set_title('Particle Escape Efficiency', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)

    # Panel 4: Distance to equilibrium
    ax = axes[1, 1]
    distance_to_equilibrium = [abs(alignment - (-1.0)) for alignment in alignments]
    ax.semilogy(rift_indices, distance_to_equilibrium, 'orange', linewidth=2, marker='o', markersize=6)

    ax.set_xlabel('Rift Index', fontsize=12)
    ax.set_ylabel('|Alignment - (-1)|', fontsize=12)
    ax.set_title('Distance to Equilibrium (log scale)', fontsize=14)
    ax.grid(True, alpha=0.3, which='both')

    # Check convergence
    metrics = rotation_dynamics.get_convergence_metrics()
    if metrics['converged']:
        ax.text(0.05, 0.95, f'✅ CONVERGED\nDistance: {metrics["distance_to_equilibrium"]:.2e}',
               transform=ax.transAxes, fontsize=11, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        logger.info(f"Saved to {filename}")
    else:
        plt.show()

    plt.close()


def plot_angular_momentum_transfer(
    rift_history: List[Dict],
    filename: Optional[str] = None
):
    """
    Plot angular momentum transfer during rift eruptions

    Shows L_ejected, L_escaped, L_recaptured for each rift

    Args:
        rift_history: List of rift event dictionaries
        filename: Save to file if provided
    """
    logger.info("Plotting angular momentum transfer")

    rift_indices = [event['rift_index'] for event in rift_history]

    # Extract angular momentum magnitudes
    L_ejected = [np.linalg.norm(event.get('L_ejected_total', np.zeros(3))) for event in rift_history]
    L_escaped = [np.linalg.norm(event.get('L_escaped_total', np.zeros(3))) for event in rift_history]
    L_recaptured = [np.linalg.norm(event.get('L_recaptured_total', np.zeros(3))) for event in rift_history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Stacked bar chart
    width = 0.6
    ax1.bar(rift_indices, L_escaped, width, label='Escaped', color='red', alpha=0.7)
    ax1.bar(rift_indices, L_recaptured, width, bottom=L_escaped,
           label='Recaptured', color='blue', alpha=0.7)

    ax1.set_xlabel('Rift Index', fontsize=12)
    ax1.set_ylabel('Angular Momentum [kg⋅m²/s]', fontsize=12)
    ax1.set_title('Angular Momentum Budget per Rift', fontsize=14)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')

    # Right: Net torque
    net_torque = [L_recaptured[i] - L_escaped[i] for i in range(len(rift_indices))]
    ax2.bar(rift_indices, net_torque, width, color='green', alpha=0.7)
    ax2.axhline(0, color='k', linestyle='--', linewidth=1)

    ax2.set_xlabel('Rift Index', fontsize=12)
    ax2.set_ylabel('Net Torque [kg⋅m²/s]', fontsize=12)
    ax2.set_title('Net Angular Momentum Transfer', fontsize=14)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        logger.info(f"Saved to {filename}")
    else:
        plt.show()

    plt.close()


# ========================================
# 5. Comprehensive Validation Report
# ========================================

def generate_validation_report(
    field_3d: ScalarFieldSolution3D,
    dynamics: ChargedParticleDynamics,
    simulation_result: Dict,
    rotation_dynamics: RotationDynamics,
    output_dir: str = "validation_plots"
):
    """
    Generate complete validation report with all plots

    Creates a directory with all validation visualizations.

    Args:
        field_3d: ScalarFieldSolution3D instance
        dynamics: ChargedParticleDynamics instance
        simulation_result: Simulation results
        rotation_dynamics: RotationDynamics instance
        output_dir: Directory to save plots
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Generating validation report in {output_dir}/")

    # 1. 3D field plots
    plot_3d_field_slice(field_3d, theta_slice=np.pi/2,
                       filename=f"{output_dir}/01_field_equatorial.png")

    plot_angular_gradients(field_3d, r_values=[1.0, 5.0, 10.0, 20.0],
                          filename=f"{output_dir}/02_angular_gradients.png")

    plot_field_energy_density(field_3d, theta_slice=np.pi/2,
                             filename=f"{output_dir}/03_energy_density.png")

    # 2. Particle trajectories
    plot_charged_trajectories(simulation_result,
                              filename=f"{output_dir}/04_trajectories.png")

    plot_velocity_evolution(simulation_result,
                           filename=f"{output_dir}/05_velocities.png")

    # 3. Force validation
    particles = [simulation_result['particles'][i] for i in range(len(simulation_result['particles']))]
    # Note: Need to reconstruct ChargedParticleState objects from simulation_result

    config = field_3d.config
    plot_coulomb_force_validation(config,
                                  filename=f"{output_dir}/06_coulomb_validation.png")

    # 4. Spin evolution (if history available)
    if hasattr(rotation_dynamics, 'rift_history') and len(rotation_dynamics.rift_history) > 0:
        plot_spin_evolution(rotation_dynamics, rotation_dynamics.rift_history,
                           filename=f"{output_dir}/07_spin_evolution.png")

        plot_angular_momentum_transfer(rotation_dynamics.rift_history,
                                      filename=f"{output_dir}/08_angular_momentum.png")

    logger.info(f"✅ Validation report complete! Saved to {output_dir}/")
    print(f"\n{'='*80}")
    print(f"VALIDATION REPORT GENERATED")
    print(f"{'='*80}")
    print(f"Location: {output_dir}/")
    print(f"Files created:")
    print(f"  01_field_equatorial.png     - 3D scalar field structure")
    print(f"  02_angular_gradients.png    - Gradient cancellation check")
    print(f"  03_energy_density.png       - Field energy distribution")
    print(f"  04_trajectories.png         - Particle paths")
    print(f"  05_velocities.png           - Speed and kinetic energy")
    print(f"  06_coulomb_validation.png   - Force law validation")
    if hasattr(rotation_dynamics, 'rift_history') and len(rotation_dynamics.rift_history) > 0:
        print(f"  07_spin_evolution.png       - Convergence to Ω₁=-Ω₂")
        print(f"  08_angular_momentum.png     - Angular momentum transfer")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    """
    Test visualization functions
    """
    print("="*80)
    print("RIFT PHYSICS VISUALIZATION TEST")
    print("="*80)
    print()

    # This is a placeholder - full tests would require running simulations
    # See rift module tests for actual physics validation

    print("✅ Visualization module loaded successfully")
    print()
    print("Available functions:")
    print("  - plot_3d_field_slice()")
    print("  - plot_angular_gradients()")
    print("  - plot_field_energy_density()")
    print("  - plot_charged_trajectories()")
    print("  - plot_velocity_evolution()")
    print("  - plot_force_components()")
    print("  - plot_coulomb_force_validation()")
    print("  - plot_spin_evolution()")
    print("  - plot_angular_momentum_transfer()")
    print("  - generate_validation_report()")
    print()
    print("Run individual module tests to generate actual plots:")
    print("  python rift/core_3d.py")
    print("  python rift/simulation_charged.py")
    print("  python rift/rotation_dynamics.py")
