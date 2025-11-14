"""
V17 Stage 1 Parallel Runner

This script manages the parallel execution of the Stage 1 optimization.
It loads the full supernova dataset, prepares a task for each supernova,
and distributes the tasks to a pool of worker processes.
"""
import argparse
import json
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import sys
from pathlib import Path

# Add core and stages directories to path
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))
sys.path.insert(0, str(Path(__file__).parent.parent / "stages"))

from v17_data import LightcurveLoader
from stage1_optimize_v17 import run_single_sn_optimization

def main():
    parser = argparse.ArgumentParser(description="V17 Stage 1 Parallel Runner")
    parser.add_argument('--lightcurves', type=str, default='pipeline/data/lightcurves_unified_v2_min3.csv', help='Path to the full lightcurve data file')
    parser.add_argument('--out', type=str, default='pipeline/results/stage1_v17', help='Output directory for results')
    parser.add_argument('--limit', type=int, default=None, help='Optional: Limit the number of SNe to process for testing')
    parser.add_argument('--n-cores', type=int, default=None, help='Number of CPU cores to use. Defaults to all available.')
    args = parser.parse_args()

    # --- Global Configuration ---
    global_params = {'k_J': 10.7, 'eta_prime': -8.0, 'xi': -7.0}
    config = {
        'nu': 5.0,
        'ftol': 1e-6,
        'max_iter': 500
    }
    
    print("--- V17 Stage 1 Optimization Runner ---")
    print(f"Loading lightcurves from: {args.lightcurves}")
    
    # 1. Load all data
    loader = LightcurveLoader(Path(args.lightcurves))
    all_photometry = loader.get_all_photometry() # Gets a dict of {snid: Photometry}
    
    # 2. Prepare tasks for the processing pool
    tasks = []
    snids_to_process = sorted(all_photometry.keys())
    if args.limit:
        snids_to_process = snids_to_process[:args.limit]

    for snid in snids_to_process:
        tasks.append({
            "snid": snid,
            "photometry": all_photometry[snid],
            "global_params": global_params,
            "config": config
        })
        
    print(f"Prepared {len(tasks)} supernovae for optimization.")

    # 3. Run optimization in parallel
    num_cores = args.n_cores if args.n_cores else cpu_count()
    print(f"Starting parallel optimization on {num_cores} cores...")
    
    output_path = Path(args.out)
    output_path.mkdir(parents=True, exist_ok=True)

    with Pool(processes=num_cores) as pool:
        # Use tqdm for a progress bar
        results_iterator = pool.imap_unordered(run_single_sn_optimization, tasks)
        
        for result in tqdm(results_iterator, total=len(tasks), desc="Processing SNe"):
            snid = result['snid']
            # 4. Save each result to its own file
            with open(output_path / f"{snid}.json", 'w') as f:
                # Convert NamedTuple to dict for JSON serialization
                if result['success']:
                    # The params are already a dict from the worker
                    pass
                json.dump(result, f, indent=2)

    print("\n--- Stage 1 Optimization Complete ---")
    print(f"Results saved to: {output_path}")

if __name__ == "__main__":
    main()
