#!/usr/bin/env python3
"""
Tune Geometric Exponents for Stability Model
==========================================

This script performs a Design of Experiments (DoE) to find the optimal
set of geometric exponents (aU, aR, aI, aK) that minimizes the lifetime
prediction error from the physically-constrained stability model.

It uses the 'Rosetta Stone' data as the input for this analysis.

Usage:
    python examples/tune_geometric_exponents.py
"""

import json
import logging
import sys
import itertools
from pathlib import Path
import numpy as np

# Add src to path for the example
SCRIPT_DIR = Path(__file__).parent
SRC_DIR = SCRIPT_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from utils.stability_analysis import StabilityPredictor

def setup_logging():
    """Setup logging for the example."""
    logging.basicConfig(
        level=logging.WARNING,  # Set to WARNING to keep the output clean
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def find_latest_results(base_dir: Path, particle: str) -> Path:
    """Find the latest ladder step results.json for a particle."""
    particle_dir = base_dir / particle
    if not particle_dir.exists():
        raise FileNotFoundError(f"Directory not found: {particle_dir}")

    ladder_dirs = sorted([d for d in particle_dir.iterdir() if d.is_dir() and d.name.startswith('ladder_')], reverse=True)
    if not ladder_dirs:
        raise FileNotFoundError(f"No ladder directories found in {particle_dir}")

    latest_ladder_dir = ladder_dirs[0]
    results_file = latest_ladder_dir / "results.json"

    if not results_file.exists():
        raise FileNotFoundError(f"results.json not found in {latest_ladder_dir}")

    return results_file

def load_simulation_results(results_path: Path) -> dict:
    """Load the simulation results from JSON and associated .npy files."""
    with open(results_path, 'r') as f:
        data = json.load(f)

    for key, value in data.items():
        if isinstance(value, dict) and 'npy_file' in value:
            npy_path = results_path.parent / value['npy_file']
            if npy_path.exists():
                data[key] = np.load(npy_path)
            else:
                print(f"Warning: Could not find .npy file: {npy_path}")
                data[key] = None
    return data

def main():
    """Run the DoE for tuning geometric exponents."""
    setup_logging()

    # --- 1. Load the Rosetta Stone Data ---
    base_output_dir = Path("rosetta_run_outputs")
    if not base_output_dir.exists():
        print(f"Error: Rosetta Stone output directory '{base_output_dir}' not found.")
        return 1

    try:
        print("Loading Rosetta Stone simulation data...")
        electron_file = find_latest_results(base_output_dir, "electron")
        muon_file = find_latest_results(base_output_dir, "muon")
        tau_file = find_latest_results(base_output_dir, "tau")
        
        electron_results = load_simulation_results(electron_file)
        muon_results = load_simulation_results(muon_file)
        tau_results = load_simulation_results(tau_file)
        print("Data loaded successfully.")

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        return 1

    # --- 2. Define the Search Space for Exponents ---
    print("\nDefining search space for exponents...")
    # For a first pass, we use a coarse-grained search space.
    # A more refined search can be done later.
    doe_space = {
        'aU': [2.0, 4.0, 6.0, 8.0],
        'aR': [0.5, 1.0, 1.5],
        'aI': [0.5, 1.0, 1.5],
        'aK': [0.5, 1.0, 1.5]
    }
    keys, values = zip(*doe_space.items())
    exponent_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    print(f"{len(exponent_combinations)} combinations will be tested.")

    # --- 3. Run the Optimization Loop ---
    print("\nRunning Design of Experiments... (This may take a moment)")
    results_history = []
    best_result = {'error': float('inf')}

    for i, exponents in enumerate(exponent_combinations):
        # Instantiate the predictor with the current set of exponents
        predictor = StabilityPredictor(exponents=exponents)

        # Run the constrained analysis
        analysis_results = predictor.analyze_stability(
            electron_results=electron_results,
            muon_results=muon_results,
            tau_results=tau_results
        )

        # Calculate the total error (sum of relative errors)
        errors = analysis_results['prediction_errors']
        total_error = abs(errors['muon_relative_error']) + abs(errors['tau_relative_error'])

        run_info = {
            'exponents': exponents,
            'error': total_error,
            'k_csr': analysis_results['fitted_parameters']['k_csr'],
            'k0': analysis_results['fitted_parameters']['k0']
        }
        results_history.append(run_info)

        if total_error < best_result['error']:
            best_result = run_info
        
        # Print progress
        sys.stdout.write(f"\rProgress: {i+1}/{len(exponent_combinations)}")
        sys.stdout.flush()

    print("\n\nDoE run complete.")

    # --- 4. Report the Best Model ---
    print("\n" + "="*60)
    print("OPTIMAL GEOMETRIC EXPONENTS REPORT")
    print("="*60)
    if best_result['error'] == float('inf'):
        print("No successful fit found.")
    else:
        print("Best physical fit found with the following parameters:")
        print(f"\n  Exponents: {best_result['exponents']}")
        print(f"  Resulting minimum error (sum of relative errors): {best_result['error']:.4f}")
        print(f"  Resulting physical parameters:")
        print(f"    k_csr = {best_result['k_csr']:.6g}")
        print(f"    k0    = {best_result['k0']:.6g}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
