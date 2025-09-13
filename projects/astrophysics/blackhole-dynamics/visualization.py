import matplotlib.pyplot as plt
import numpy as np
import os
import logging
from config import SimConfig
from core import TwoBodySystem, ScalarFieldSolution
from simulation import HamiltonianDynamics

def plot_scalar_field_profiles(config: SimConfig, solutions: list, filename: str):
    logging.info(f"Generating plot: {filename}")
    if not solutions:
        logging.warning("No solutions for plotting scalar field profiles.")
        return
    plt.figure(figsize=(10, 8))
    try:
        cmap_func = plt.colormaps['viridis']
    except AttributeError:
        cmap_func = plt.get_cmap('viridis')
    colors = cmap_func(np.linspace(0, 1, len(solutions)))
    for i, sol in enumerate(solutions):
        if sol.r_core is None or sol.r_core <= 0 or sol.r_values is None:
            logging.warning(f"Skipping solution {i} in plot: invalid data.")
            continue
        phi_ratio = sol.phi_0 / sol.phi_vac
        r_norm = sol.r_values / sol.r_core
        phi_norm = sol.phi_values / sol.phi_vac
        uniq_r, idx = np.unique(r_norm, return_index=True)
        uniq_phi = phi_norm[idx]
        plt.plot(uniq_r, uniq_phi, label=rf"$\phi(0)/\phi_{{vac}}={phi_ratio:.1f}$", lw=2, color=colors[i])
    plt.xscale('log')
    plt.xlabel("$r/R_{core}$")
    plt.ylabel(r"$\phi(r)/\phi_{vac}$")
    plt.grid(True, which='both', alpha=0.3)
    plt.legend()
    plt.ylim(bottom=0.9)
    plt.title("Non-singular Scalar Field Profiles")
    # ... (add inset plot logic if desired) ...
    try:
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        logging.info(f"Figure saved: {filename}")
    except Exception as e:
        logging.error(f"ERROR saving {filename}: {e}")
    plt.close()

def plot_potential_contours(config: SimConfig, system: TwoBodySystem, filename: str):
    # ... (add plotting logic) ...
    pass

def plot_saddle_energy_vs_distance(config: SimConfig, D_values_norm: np.ndarray, all_saddle_energies: np.ndarray, mass_ratios_plot: np.ndarray, filename: str):
    # ... (add plotting logic) ...
    pass

def plot_particle_trajectories(config: SimConfig, dynamics: HamiltonianDynamics, filename: str):
    # ... (add plotting logic) ...
    pass

def plot_escape_fraction(config: SimConfig, D_values_norm: np.ndarray, mass_ratios_plot: np.ndarray, escape_fractions_plot: np.ndarray, filename: str):
    # ... (add plotting logic) ...
    pass