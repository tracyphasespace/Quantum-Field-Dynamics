# src/orchestration/run_free_electron.py
import argparse
import logging
from pathlib import Path
from src.solvers.phoenix_solver import solve_psi_field
from src.utils.analysis import analyze_results
from src.utils.io import save_results

def main():
    """Orchestrate a free electron simulation."""
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Run free electron simulation.")
    parser.add_argument("--particle", type=str, default="electron", help="Particle to simulate.")
    parser.add_argument("--grid_size", type=int, default=64, help="Grid size.")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU if available.")
    parser.add_argument("--output_dir", type=str, default="data/output", help="Output directory.")
    args = parser.parse_args()

    # Run simulation
    logging.info(f"Running simulation for {args.particle}...")
    results = solve_psi_field(
        particle=args.particle,
        grid_size=args.grid_size,
        use_gpu=args.use_gpu
    )

    # Save raw results
    output_dir = Path(args.output_dir) / args.particle
    output_dir.mkdir(parents=True, exist_ok=True)
    save_results(results, output_dir / "results")

    # Analyze and export
    analysis = analyze_results(results)
    save_results(analysis, output_dir / "analysis")

    logging.info(f"Simulation completed. Results saved to {output_dir}.")

if __name__ == "__main__":
    main()
