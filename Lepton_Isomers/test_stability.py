#!/usr/bin/env python3
"""
Test Stability Analysis
=======================

This script tests the StabilityPredictor from stability_analysis.py by
loading the results from the quick simulations and running the analysis.

Usage:
    python test_stability.py
"""

import json
import logging
import sys
from pathlib import Path
import numpy as np

# Add src to path for the test
SCRIPT_DIR = Path(__file__).parent
SRC_DIR = SCRIPT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from utils.stability_analysis import StabilityPredictor

def setup_logging():
    """Setup logging for the test."""
    logging.basicConfig(
        level=logging.INFO,
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

    # Load numpy arrays referenced in the JSON
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
    """Load data and run stability analysis."""
    setup_logging()

    base_output_dir = Path("rosetta_run_outputs")
    if not base_output_dir.exists():
        print(f"Error: Output directory '{base_output_dir}' not found.")
        print("Please run 'python examples/run_all_particles_quick.py' first.")
        return 1

    try:
        print("Locating the latest simulation results...")
        electron_file = find_latest_results(base_output_dir, "electron")
        muon_file = find_latest_results(base_output_dir, "muon")
        tau_file = find_latest_results(base_output_dir, "tau")
        print(f"  Electron: {electron_file}")
        print(f"  Muon: {muon_file}")
        print(f"  Tau: {tau_file}")

        print("\nLoading results...")
        electron_results = load_simulation_results(electron_file)
        muon_results = load_simulation_results(muon_file)
        tau_results = load_simulation_results(tau_file)

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please ensure the quick simulations ran successfully.")
        return 1

    print("\nInitializing Stability Predictor...")
    predictor = StabilityPredictor()

    print("\nRunning stability analysis...")
    analysis_results = predictor.analyze_stability(
        electron_results=electron_results,
        muon_results=muon_results,
        tau_results=tau_results,
        output_dir=Path("stability_analysis_test_output")
    )

    print("\n" + "-"*70)
    print("STABILITY ANALYSIS TEST COMPLETE")
    print("-"*70)

    # Print the summary report
    report = predictor.format_summary_report(analysis_results)
    print(report)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
