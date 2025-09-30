import logging
import time
from pathlib import Path
import numpy as np

from config import SimConfig
from core import ScalarFieldSolution, TwoBodySystem
from simulation import HamiltonianDynamics, analyze_escape_statistics_parallel
from visualization import (
    plot_scalar_field_profiles,
    plot_potential_contours,
    plot_saddle_energy_vs_distance,
    plot_particle_trajectories,
    plot_escape_fraction,
)

def run_simulation_pipeline(config: SimConfig):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s (%(funcName)s PID:%(process)d): %(message)s')
    logging.info(f"Starting Simulation. PyTorch: {config.USE_GPU_ODE}, Device: {config.DEVICE}")
    start_time = time.time()
    results = {}

    Path(config.FIGURE_DIR).mkdir(exist_ok=True)

    # --- Part 1: Scalar Field Solution ---
    if config.RUN_FIELD_SOLUTION:
        logging.info("\n=== Part 1: Solving Scalar Field Equation ===")
        solutions = []
        for phi0_val in config.PHI_0_VALUES:
            solution = ScalarFieldSolution(config, phi0_val)
            solution.solve()
            if solution._solve_successful:
                solutions.append(solution)
        if not solutions:
            logging.critical("P1: Failed to solve any scalar field. Aborting.")
            return
        ref_solution = next((s for s in solutions if s.mass and s.r_core), None)
        if ref_solution is None:
            logging.critical("P1: No valid reference solution found. Aborting.")
            return
        logging.info(f"P1: Using ref_solution phi_0={ref_solution.phi_0:.1f} (M={ref_solution.mass:.2e}, Rc={ref_solution.r_core:.2e})")
        results['ref_solution'] = ref_solution
        results['all_solutions'] = solutions
        if config.GENERATE_FIGURES:
            plot_scalar_field_profiles(config, results['all_solutions'], f"{config.FIGURE_DIR}/fig1_phi_profiles.png")

    # --- Part 2: Saddle Point Analysis ---
    if config.RUN_SADDLE_ANALYSIS:
        logging.info("\n=== Part 2: Saddle Point Analysis ===")
        ref_solution = results['ref_solution']
        R_core = ref_solution.r_core or 1.0
        M_focus = ref_solution.mass or 1.0
        system = TwoBodySystem(config, ref_solution, M_focus, M_focus)
        results['system'] = system
        if config.GENERATE_FIGURES:
            plot_potential_contours(config, system, f"{config.FIGURE_DIR}/fig2_potential_contours.png")
        
        D_values_test_saddle = np.array([5.0*R_core, 10.0*R_core, 20.0*R_core, 50.0*R_core])
        mass_ratios_cfg_test_saddle = np.array([0.5, 1.0, 2.0])
        all_saddle_energies_plot = np.full((len(mass_ratios_cfg_test_saddle), len(D_values_test_saddle)), np.nan)
        
        for i, ratio_cfg in enumerate(mass_ratios_cfg_test_saddle):
            system.update_masses(M_focus, ratio_cfg * M_focus)
            _, s_energies = system.analyze_saddle_vs_separation(D_values_test_saddle, use_gpu_search=config.USE_GPU_SADDLE_SEARCH)
            all_saddle_energies_plot[i,:] = s_energies
        results['D_values_norm_saddle_plot'] = D_values_test_saddle / R_core
        results['mass_ratios_saddle_plot'] = mass_ratios_cfg_test_saddle
        results['all_saddle_energies_saddle_plot'] = all_saddle_energies_plot
        if config.GENERATE_FIGURES: 
            plot_saddle_energy_vs_distance(config, results['D_values_norm_saddle_plot'], 
                                           results['all_saddle_energies_saddle_plot'], 
                                           results['mass_ratios_saddle_plot'], f"{config.FIGURE_DIR}/fig3_saddle_energy.png")

    # --- Part 3: Particle Trajectory Simulation ---
    if config.RUN_TRAJECTORY_SIM:
        logging.info("\n=== Part 3: Particle Trajectory Simulation (Test Single) ===")
        system = results['system']
        dynamics = HamiltonianDynamics(config, system)
        results['dynamics'] = dynamics
        if config.GENERATE_FIGURES:
            plot_particle_trajectories(config, dynamics, f"{config.FIGURE_DIR}/fig4_trajectories.png")

    # --- Part 4: Statistical Analysis ---
    if config.RUN_ESCAPE_STATS:
        logging.info("\n=== Part 4: Statistical Analysis of Escape Fractions ===")
        ref_solution_stats = results['ref_solution']
        escape_fractions_res = analyze_escape_statistics_parallel(config, ref_solution_stats)
        results['escape_fractions'] = escape_fractions_res
        if config.GENERATE_FIGURES and 'escape_fractions' in results:
            plot_escape_fraction(config, np.array(config.D_VALUES_STATS_NORM), np.array(config.MASS_RATIOS_STATS), 
                               results['escape_fractions'], f"{config.FIGURE_DIR}/fig5_escape_fraction.png")

    total_time = time.time() - start_time
    logging.info(f"Simulation Pipeline Completed in {total_time:.2f} seconds.")
    return results

if __name__ == "__main__":
    config = SimConfig()
    run_simulation_pipeline(config)