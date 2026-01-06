#!/usr/bin/env python3
"""
Simple single-job optimization with visible progress.
No multiprocessing, no hidden output - just run and see results.
"""
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, 'src')
from runspec_adapter import RunSpecAdapter

def main():
    print("=" * 70)
    print("SIMPLE SINGLE OPTIMIZATION - Visible Progress")
    print("=" * 70)
    print()
    
    config_path = 'experiments/nuclear_heavy_minimal_test.runspec.json'
    
    # Load and show config
    with open(config_path) as f:
        config = json.load(f)
    
    print(f"Config: {config_path}")
    print(f"maxiter: {config['solver']['options']['maxiter']}")
    print(f"popsize: {config['solver']['options']['popsize']}")
    print(f"Device: {config['solver']['scf_solver_options']['device']}")
    print()
    
    # Estimate runtime
    maxiter = config['solver']['options']['maxiter']
    popsize = config['solver']['options']['popsize']
    
    # Count isotopes from dataset
    import pandas as pd
    ame_data = pd.read_csv('data/ame2020_system_energies.csv')
    cuts = config['datasets'][0]['cuts']

    # Check if using explicit target_isotopes list or A_min/A_max range
    if 'target_isotopes' in cuts:
        n_isotopes = len(cuts['target_isotopes'])
    elif 'A_min' in cuts and 'A_max' in cuts:
        isotopes = ame_data[(ame_data['A'] >= cuts['A_min']) & (ame_data['A'] <= cuts['A_max'])]
        n_isotopes = len(isotopes)
    else:
        n_isotopes = len(ame_data)  # fallback
    
    if 'target_isotopes' in cuts:
        print(f"Dataset: {n_isotopes} explicit target isotopes")
    elif 'A_min' in cuts:
        print(f"Dataset: {n_isotopes} isotopes (A={cuts['A_min']}-{cuts['A_max']})")
    else:
        print(f"Dataset: {n_isotopes} isotopes (full dataset)")
    print()
    
    # Estimate
    evals_per_iter = popsize
    total_evals = maxiter * evals_per_iter * n_isotopes
    mins_per_eval = 2  # rough estimate for heavy isotopes
    estimated_mins = total_evals * mins_per_eval / 60
    
    print(f"Estimated evaluations: {maxiter} iter × {popsize} pop × {n_isotopes} isotopes = {total_evals}")
    print(f"Estimated time: ~{estimated_mins:.0f} minutes ({estimated_mins/60:.1f} hours)")
    print()
    print("Starting optimization...")
    print("-" * 70)
    
    start = time.time()
    
    adapter = RunSpecAdapter(config_path)
    result = adapter.run(verbose=True)
    
    elapsed = time.time() - start
    
    print("-" * 70)
    print(f"Completed in {elapsed/60:.1f} minutes")
    print()
    print(f"Status: {result.get('status')}")
    
    if result.get('status') == 'success':
        print(f"Best score: {result.get('best_score', 'N/A')}")
        if 'metrics' in result:
            print("Metrics:")
            for k, v in result['metrics'].items():
                print(f"  {k}: {v}")
        
        # Save result
        output_file = Path('results') / f'simple_opt_{int(time.time())}.json'
        output_file.parent.mkdir(exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nResult saved to: {output_file}")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")

if __name__ == '__main__':
    main()
