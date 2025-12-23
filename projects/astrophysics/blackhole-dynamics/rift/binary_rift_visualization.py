import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

#!/usr/bin/env python3
"""
Binary Rift Visualization - L1 Saddle Point Ejection

Visualizations showing:
1. Gravitational potential saddle structure
2. Particle ejection through L1 Lagrange point
3. Effect of opposing rotations on escape
4. Binary BH system configuration
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def plot_potential_saddle_1d(
    system,
    x_range: tuple = None,
    n_points: int = 500,
    filename: Optional[str] = None
):
    """
    Plot gravitational potential along x-axis showing L1 saddle point.

    Args:
        system: BinaryRiftSystem instance
        x_range: (x_min, x_max) or None for auto
        n_points: Number of grid points
        filename: Save to file if provided
    """
    logger.info("Plotting 1D potential saddle structure")

    if x_range is None:
        # Auto range: slightly beyond BHs
        x_range = (-5.0, system.separation + 5.0)

    x_vals = np.linspace(x_range[0], x_range[1], n_points)
    potentials = system.compute_effective_potential_1d(x_vals)

    # L1 position
    L1_x = system.saddle_point[0]
    L1_energy = system.saddle_energy

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot potential
    ax.plot(x_vals, potentials, 'b-', linewidth=2, label='Total Potential')

    # Mark BH positions
    ax.axvline(system.BH1_pos[0], color='black', linestyle='--', linewidth=2,
              alpha=0.5, label=f'BH1 (M={system.M1:.1f})')
    ax.axvline(system.BH2_pos[0], color='gray', linestyle='--', linewidth=2,
              alpha=0.5, label=f'BH2 (M={system.M2:.1f})')

    # Mark L1 saddle point
    ax.plot(L1_x, L1_energy, 'ro', markersize=15, label=f'L1 Saddle Point',
           zorder=10)
    ax.axvline(L1_x, color='red', linestyle=':', linewidth=1, alpha=0.5)

    # Annotations
    ax.annotate('L1', xy=(L1_x, L1_energy),
               xytext=(L1_x, L1_energy + (potentials.max()-potentials.min())*0.1),
               fontsize=14, color='red', weight='bold',
               ha='center',
               arrowprops=dict(arrowstyle='->', color='red', lw=2))

    ax.set_xlabel('x [m]', fontsize=14)
    ax.set_ylabel('Gravitational Potential Φ', fontsize=14)
    ax.set_title('Potential Saddle Structure - L1 Lagrange Point\n'
                f'Binary Separation: {system.separation:.1f} m', fontsize=16)
    ax.legend(loc='best', fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        logger.info(f"Saved to {filename}")
    else:
        plt.show()

    plt.close()


def plot_potential_saddle_2d(
    system,
    x_range: tuple = None,
    y_range: tuple = (-10, 10),
    n_points: int = 100,
    filename: Optional[str] = None
):
    """
    Plot 2D gravitational potential showing saddle structure.

    Args:
        system: BinaryRiftSystem instance
        x_range: (x_min, x_max) for x-axis
        y_range: (y_min, y_max) for y-axis
        n_points: Grid resolution
        filename: Save to file if provided
    """
    logger.info("Plotting 2D potential saddle surface")

    if x_range is None:
        x_range = (-5.0, system.separation + 5.0)

    x_vals = np.linspace(x_range[0], x_range[1], n_points)
    y_vals = np.linspace(y_range[0], y_range[1], n_points)
    X, Y = np.meshgrid(x_vals, y_vals)

    # Compute potential at each grid point
    Z = np.zeros_like(X)
    for i in range(n_points):
        for j in range(n_points):
            q = np.array([X[i, j], Y[i, j], 0.0])
            Z[i, j] = system.binary_system.total_potential(q)

    # Clip extreme values for visualization
    Z_clipped = np.clip(Z, np.percentile(Z, 1), np.percentile(Z, 99))

    fig = plt.figure(figsize=(16, 6))

    # Left: Contour plot
    ax1 = fig.add_subplot(121)
    contour = ax1.contourf(X, Y, Z_clipped, levels=50, cmap='viridis')
    plt.colorbar(contour, ax=ax1, label='Potential Φ')

    # Mark BHs
    ax1.plot(system.BH1_pos[0], system.BH1_pos[1], 'wo', markersize=15,
            markeredgecolor='black', markeredgewidth=2, label='BH1')
    ax1.plot(system.BH2_pos[0], system.BH2_pos[1], 'wo', markersize=15,
            markeredgecolor='gray', markeredgewidth=2, label='BH2')

    # Mark L1
    ax1.plot(system.saddle_point[0], system.saddle_point[1], 'r*',
            markersize=20, label='L1 Saddle', zorder=10)

    ax1.set_xlabel('x [m]', fontsize=12)
    ax1.set_ylabel('y [m]', fontsize=12)
    ax1.set_title('Potential Contours (Top View)', fontsize=14)
    ax1.legend(loc='best', fontsize=10)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    # Right: 3D surface
    ax2 = fig.add_subplot(122, projection='3d')
    surf = ax2.plot_surface(X, Y, Z_clipped, cmap='viridis', alpha=0.8,
                           linewidth=0, antialiased=True)

    # Mark L1 on 3D plot
    ax2.scatter([system.saddle_point[0]], [system.saddle_point[1]],
               [system.saddle_energy],
               color='red', s=200, marker='*', label='L1', zorder=10)

    ax2.set_xlabel('x [m]', fontsize=10)
    ax2.set_ylabel('y [m]', fontsize=10)
    ax2.set_zlabel('Potential Φ', fontsize=10)
    ax2.set_title('Saddle Surface (3D View)', fontsize=14)

    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        logger.info(f"Saved to {filename}")
    else:
        plt.show()

    plt.close()


def plot_L1_ejection(
    system,
    result: Dict,
    filename: Optional[str] = None
):
    """
    Plot particle trajectories through L1 saddle point.

    Args:
        system: BinaryRiftSystem instance
        result: Simulation results from simulate_rift_ejection()
        filename: Save to file if provided
    """
    logger.info("Plotting L1 ejection trajectories")

    fig = plt.figure(figsize=(16, 10))

    # Panel 1: Top view (x-y plane)
    ax1 = fig.add_subplot(221)

    # Plot BHs
    ax1.plot(system.BH1_pos[0], system.BH1_pos[1], 'ko', markersize=20,
            label='BH1', zorder=5)
    ax1.plot(system.BH2_pos[0], system.BH2_pos[1], 'ko', markersize=20,
            label='BH2', zorder=5, markerfacecolor='gray')

    # Plot L1
    ax1.plot(system.saddle_point[0], system.saddle_point[1], 'r*',
            markersize=25, label='L1 Saddle', zorder=10)

    # Plot particle trajectories
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']

    for i, traj in enumerate(result['particles']):
        pos = traj['position']
        particle_type = traj['type']
        color = colors[i % len(colors)]

        # Trajectory
        ax1.plot(pos[0, :], pos[1, :], color=color, linewidth=2,
                alpha=0.7, label=f'{particle_type} {i}')

        # Start point
        ax1.plot(pos[0, 0], pos[1, 0], 'o', color=color, markersize=8,
                markeredgecolor='black', markeredgewidth=1)

        # End point
        ax1.plot(pos[0, -1], pos[1, -1], 's', color=color, markersize=8,
                markeredgecolor='black', markeredgewidth=1)

    ax1.set_xlabel('x [m]', fontsize=12)
    ax1.set_ylabel('y [m]', fontsize=12)
    ax1.set_title('Particle Trajectories (Top View)', fontsize=14)
    ax1.legend(loc='best', fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # Panel 2: 3D view
    ax2 = fig.add_subplot(222, projection='3d')

    # Plot BHs
    ax2.scatter([system.BH1_pos[0]], [system.BH1_pos[1]], [0],
               color='black', s=200, marker='o', label='BH1')
    ax2.scatter([system.BH2_pos[0]], [system.BH2_pos[1]], [0],
               color='gray', s=200, marker='o', label='BH2')

    # Plot L1
    ax2.scatter([system.saddle_point[0]], [system.saddle_point[1]], [0],
               color='red', s=300, marker='*', label='L1')

    # Plot trajectories
    for i, traj in enumerate(result['particles']):
        pos = traj['position']
        color = colors[i % len(colors)]
        particle_type = traj['type']

        ax2.plot(pos[0, :], pos[1, :], pos[2, :], color=color,
                linewidth=2, alpha=0.7, label=f'{particle_type} {i}')

    ax2.set_xlabel('x [m]', fontsize=10)
    ax2.set_ylabel('y [m]', fontsize=10)
    ax2.set_zlabel('z [m]', fontsize=10)
    ax2.set_title('3D Trajectories', fontsize=14)
    ax2.legend(loc='best', fontsize=8)

    # Panel 3: x-position vs time
    ax3 = fig.add_subplot(223)

    t = result['t'] * 1e6  # Convert to microseconds

    # Mark L1 position
    ax3.axhline(system.saddle_point[0], color='red', linestyle='--',
               linewidth=2, label='L1 position', alpha=0.5)

    for i, traj in enumerate(result['particles']):
        pos = traj['position']
        color = colors[i % len(colors)]
        particle_type = traj['type']

        ax3.plot(t, pos[0, :], color=color, linewidth=2,
                label=f'{particle_type} {i}')

    ax3.set_xlabel('Time [μs]', fontsize=12)
    ax3.set_ylabel('x position [m]', fontsize=12)
    ax3.set_title('Position Along Axis (Through L1)', fontsize=14)
    ax3.legend(loc='best', fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Panel 4: Distance from L1 vs time
    ax4 = fig.add_subplot(224)

    L1_pos = system.saddle_point

    for i, traj in enumerate(result['particles']):
        pos = traj['position']
        color = colors[i % len(colors)]
        particle_type = traj['type']

        # Distance from L1
        dist_from_L1 = np.linalg.norm(pos - L1_pos[:, np.newaxis], axis=0)

        ax4.plot(t, dist_from_L1, color=color, linewidth=2,
                label=f'{particle_type} {i}')

    ax4.set_xlabel('Time [μs]', fontsize=12)
    ax4.set_ylabel('Distance from L1 [m]', fontsize=12)
    ax4.set_title('Escape from L1 Saddle Point', fontsize=14)
    ax4.legend(loc='best', fontsize=8)
    ax4.grid(True, alpha=0.3)

    # Overall title
    alignment = system.Omega1.dot(system.Omega2) / (
        np.linalg.norm(system.Omega1) * np.linalg.norm(system.Omega2)
    )
    fig.suptitle(f'L1 Saddle Point Ejection - Rotation Alignment: {alignment:.3f}',
                fontsize=16, y=0.995)

    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        logger.info(f"Saved to {filename}")
    else:
        plt.show()

    plt.close()


def plot_rotation_effect_comparison(
    config,
    M1: float,
    M2: float,
    separation: float,
    alignments: list = [-1.0, 0.0, 1.0],
    filename: Optional[str] = None
):
    """
    Compare escape dynamics for different rotation alignments.

    Shows how opposing rotations (alignment = -1) enable escape.

    Args:
        config: SimConfig
        M1, M2: BH masses
        separation: Binary separation
        alignments: List of rotation alignments to compare
        filename: Save to file if provided
    """
    from rift.binary_rift_simulation import BinaryRiftSystem

    logger.info("Comparing rotation alignments...")

    fig, axes = plt.subplots(2, len(alignments), figsize=(6*len(alignments), 10))

    Omega_magnitude = config.OMEGA_BH1_MAGNITUDE

    for idx, alignment in enumerate(alignments):
        # Define rotations
        Omega1 = np.array([0.0, 0.0, Omega_magnitude])

        if abs(alignment) < 0.01:  # Perpendicular
            Omega2 = np.array([Omega_magnitude, 0.0, 0.0])
        else:
            Omega2 = alignment * Omega1

        # Create system
        logger.info(f"  Alignment {alignment:.1f}...")
        system = BinaryRiftSystem(config, M1, M2, separation, Omega1, Omega2)

        # Plot potential saddle
        ax_top = axes[0, idx] if len(alignments) > 1 else axes[0]

        x_vals = np.linspace(-5, separation+5, 200)
        potentials = system.compute_effective_potential_1d(x_vals)

        ax_top.plot(x_vals, potentials, 'b-', linewidth=2)
        ax_top.axvline(system.saddle_point[0], color='r', linestyle='--',
                      label='L1')
        ax_top.axvline(0, color='k', linestyle=':', alpha=0.5, label='BH1')
        ax_top.axvline(separation, color='gray', linestyle=':', alpha=0.5,
                      label='BH2')

        ax_top.set_xlabel('x [m]')
        ax_top.set_ylabel('Potential Φ')
        ax_top.set_title(f'Alignment = {alignment:.1f}')
        ax_top.legend()
        ax_top.grid(True, alpha=0.3)

        # Plot angular gradient cancellation
        ax_bottom = axes[1, idx] if len(alignments) > 1 else axes[1]

        theta_vals = np.linspace(0.01, np.pi - 0.01, 100)
        r = system.saddle_point[0]  # At L1

        grad_theta = [system.field_3d.angular_gradient(r, theta, 0.0)
                     for theta in theta_vals]

        ax_bottom.plot(theta_vals, grad_theta, 'g-', linewidth=2)
        ax_bottom.axhline(0, color='k', linestyle='--', alpha=0.5)
        ax_bottom.axhline(0.1, color='r', linestyle=':', alpha=0.5,
                         label='Threshold')
        ax_bottom.axhline(-0.1, color='r', linestyle=':', alpha=0.5)

        max_grad = max(abs(g) for g in grad_theta)
        ax_bottom.set_xlabel('θ [rad]')
        ax_bottom.set_ylabel('∂φ/∂θ')
        ax_bottom.set_title(f'Max |∂φ/∂θ| = {max_grad:.4f}')
        ax_bottom.legend()
        ax_bottom.grid(True, alpha=0.3)

    fig.suptitle('Effect of Rotation Alignment on Rift Physics', fontsize=16)
    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        logger.info(f"Saved to {filename}")
    else:
        plt.show()

    plt.close()


if __name__ == "__main__":
    """Test visualizations"""
    print("Binary rift visualization module loaded.")
    print()
    print("Usage:")
    print("  from rift.binary_rift_simulation import BinaryRiftSystem")
    print("  from rift.binary_rift_visualization import *")
    print()
    print("  # Create system and run simulation")
    print("  system = BinaryRiftSystem(...)")
    print("  result = system.simulate_rift_ejection(...)")
    print()
    print("  # Visualize")
    print("  plot_potential_saddle_1d(system)")
    print("  plot_potential_saddle_2d(system)")
    print("  plot_L1_ejection(system, result)")
