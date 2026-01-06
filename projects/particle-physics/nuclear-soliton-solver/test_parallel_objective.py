#!/usr/bin/env python3
"""
Test parallel objective function with GPU memory monitoring.
"""
import sys
import json
import time
from pathlib import Path
import pandas as pd

sys.path.insert(0, 'src')
from parallel_objective import ParallelObjective

def test_parallel(max_workers=2):
    """Test with different worker counts."""
    print("=" * 70)
    print(f"PARALLEL OBJECTIVE TEST (max_workers={max_workers})")
    print("=" * 70)
    print()
    
    # Load minimal test config
    with open('experiments/nuclear_heavy_minimal_test.runspec.json') as f:
        config = json.load(f)
    
    # Get parameters
    params = {}
    for p in config['parameters']:
        if not p.get('frozen', False):
            name = p['name'].replace('nuclear.', '').replace('solver.', '')
            params[name] = p['value']
    
    # Load AME data
    ame_data = pd.read_csv('data/ame2020_system_energies.csv')
    
    # Get target isotopes (8 from minimal test)
    target_isotopes = [(50, 120), (79, 197), (80, 200), 
                      (82, 206), (82, 207), (82, 208),
                      (92, 235), (92, 238)]
    
    print(f"Testing on {len(target_isotopes)} isotopes")
    print(f"Workers: {max_workers}")
    print(f"Grid: 32, Iters: 150 (fast mode)")
    print()
    
    # Create objective
    objective = ParallelObjective(
        target_isotopes=target_isotopes,
        ame_data=ame_data,
        max_workers=max_workers,
        grid_points=32,
        iters_outer=150,
        device='cuda',
        verbose=True
    )
    
    print()
    print("Starting evaluation...")
    print("-" * 70)
    
    start = time.time()
    loss = objective(params)
    elapsed = time.time() - start
    
    print("-" * 70)
    print(f"Completed in {elapsed:.1f} seconds")
    print(f"Final loss: {loss:.6f}")
    print()
    
    # Calculate speedup estimate
    time_per_isotope = elapsed / len(target_isotopes)
    print(f"Time per isotope: {time_per_isotope:.1f}s (with {max_workers} workers)")
    print()
    
    # Check GPU memory
    import subprocess
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', 
                               '--format=csv,noheader,nounits'],
                              capture_output=True, text=True)
        mem_mb = int(result.stdout.strip())
        print(f"GPU memory after test: {mem_mb} MB ({mem_mb/1024:.2f} GB)")
        if mem_mb > 3072:
            print("⚠️  WARNING: Exceeded 3GB target!")
        else:
            print("✓ Within 3GB memory target")
    except:
        print("Could not check GPU memory")
    
    return elapsed, loss

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=2,
                       help='Number of parallel workers (2-4)')
    args = parser.parse_args()
    
    test_parallel(max_workers=args.workers)
