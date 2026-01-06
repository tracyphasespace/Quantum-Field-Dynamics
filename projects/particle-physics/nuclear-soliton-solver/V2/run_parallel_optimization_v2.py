#!/usr/bin/env python3
"""
Run overnight optimization with ParallelObjective for GPU-efficient parallel evaluation.

Usage:
  python3 run_parallel_optimization.py --maxiter 20 --popsize 15 --workers 3

For overnight run:
  nohup python3 run_parallel_optimization.py --maxiter 100 --popsize 15 --workers 3 > overnight_opt.log 2>&1 &
"""
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm

sys.path.insert(0, 'src')
from parallel_objective import ParallelObjective

def main():
    parser = argparse.ArgumentParser(description='Run parallel optimization')
    parser.add_argument('--runspec', default='experiments/nuclear_heavy_minimal_test.runspec.json',
                       help='Path to RunSpec config file')
    parser.add_argument('--init_params', default='parameters/trial32_universal.params.json',
                       help='Path to initial parameters JSON file for refinement')
    parser.add_argument('--maxiter', type=int, default=500,
                       help='Maximum DE iterations (generations)')
    parser.add_argument('--popsize', type=int, default=8,
                       help='Population size per parameter')
    parser.add_argument('--workers', type=int, default=2,
                       help='Number of parallel isotope workers (2-4 for 4GB GPU)')
    parser.add_argument('--grid', type=int, default=32,
                       help='Grid resolution (32=fast, 48=accurate)')
    parser.add_argument('--iters', type=int, default=150,
                       help='SCF iterations (150=fast, 360=accurate)')
    parser.add_argument('--device', default='cuda',
                       help='Device: cuda or cpu')
    args = parser.parse_args()

    print("=" * 80)
    print("PARALLEL REFINEMENT OPTIMIZATION WITH GPU ACCELERATION")
    print("=" * 80)
    print(f"Start time: {datetime.now()}")
    print()

    # Load initial parameters for refinement
    with open(args.init_params) as f:
        init_params_config = json.load(f)

    # Load RunSpec
    with open(args.runspec) as f:
        config = json.load(f)

    print(f"RunSpec: {config['experiment_id']}")
    print(f"Initial parameters for refinement: {args.init_params}")
    print()

    # Extract parameters and bounds
    params_initial = {}
    param_names = []
    
    # Use parameters from the init_params file as the starting point
    for p in init_params_config['parameters']:
        name = p['name'].replace('nuclear.', '').replace('solver.', '')
        value = p['value']
        frozen = p.get('frozen', False)

        if not frozen:
            params_initial[name] = value
            param_names.append(name)
            
    # The bounds will be taken from the runspec file, but the initial values from trial32
    params_bounds = {}
    for p in config['parameters']:
        name = p['name'].replace('nuclear.', '').replace('solver.', '')
        if not p.get('frozen', False) and p.get('bounds'):
            params_bounds[name] = tuple(p['bounds'])

    print(f"Refining {len(param_names)} parameters from {args.init_params}:")
    x0 = []
    for name in param_names:
        x0.append(params_initial[name])
        print(f"  {name}: {params_initial[name]:.6f} Bounds: {params_bounds.get(name)}")
    print()


    # Load AME data
    ame_data = pd.read_csv('data/ame2020_system_energies.csv')

    # Get target isotopes from RunSpec
    cuts = config['datasets'][0]['cuts']
    target_isotopes = [(iso['Z'], iso['A']) for iso in cuts['target_isotopes']]

    print(f"Target isotopes: {len(target_isotopes)}")
    for Z, A in target_isotopes:
        print(f"  {Z}-{A}")
    print()

    # Setup ParallelObjective
    print(f"ParallelObjective configuration:")
    print(f"  Workers: {args.workers}")
    print(f"  Grid: {args.grid}")
    print(f"  SCF iterations: {args.iters}")
    print(f"  Device: {args.device}")
    print()

    parallel_obj = ParallelObjective(
        target_isotopes=target_isotopes,
        ame_data=ame_data,
        max_workers=args.workers,
        grid_points=args.grid,
        iters_outer=args.iters,
        device=args.device,
        verbose=False  # Don't print every isotope
    )

    # Vectorized objective wrapper with TQDM progress bar
    eval_count = [0]
    best_loss = [float('inf')]
    start_time = time.time()

    pbar = tqdm(total=args.maxiter, desc="Nelder-Mead", unit="eval",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] Loss: {postfix}')

    def objective_vec(x):
        """Convert parameter vector to dict and evaluate."""
        params = params_initial.copy()
        for i, name in enumerate(param_names):
            params[name] = x[i]

        loss = parallel_obj(params)
        eval_count[0] += 1
        pbar.update(1)

        if loss < best_loss[0]:
            best_loss[0] = loss
            pbar.set_postfix_str(f"Best: {loss:.2e}")

        return loss

    # Run optimization
    print("=" * 80)
    print(f"STARTING REFINEMENT")
    print(f"  Method: Nelder-Mead")
    print(f"  Max function evaluations: {args.maxiter}")
    print("=" * 80)
    print()

    result = minimize(
        objective_vec,
        x0=np.array(x0),
        method='Nelder-Mead',
        options={'maxfev': args.maxiter, 'disp': True}
    )

    pbar.close()  # Close progress bar
    print()
    print("=" * 80)
    print("OPTIMIZATION COMPLETE")
    print("=" * 80)
    print(f"End time: {datetime.now()}")
    print(f"Total time: {time.time() - start_time:.1f}s")
    print(f"Total evaluations: {eval_count[0]}")
    print(f"Final loss: {result.fun:.6f}")
    print()

    # Extract optimized parameters
    optimized_params = params_initial.copy()
    for i, name in enumerate(param_names):
        optimized_params[name] = result.x[i]

    print("Optimized parameters:")
    for name in param_names:
        initial = params_initial[name]
        final = optimized_params[name]
        change_pct = 100.0 * (final - initial) / initial if initial != 0 else 0
        print(f"  {name:20s}: {initial:10.6f} → {final:10.6f} ({change_pct:+.1f}%)")
    print()

    # Save results
    output_file = f"optimization_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output = {
        'experiment_id': config['experiment_id'],
        'timestamp': datetime.now().isoformat(),
        'optimization': {
            'method': 'Nelder-Mead',
            'max_evals': args.maxiter,
            'total_evaluations': eval_count[0],
            'final_loss': float(result.fun),
            'success': result.success,
            'message': result.message
        },
        'parameters_initial': params_initial,
        'parameters_optimized': optimized_params,
        'target_isotopes': [{'Z': Z, 'A': A} for Z, A in target_isotopes],
        'hardware': {
            'parallel_workers': args.workers,
            'grid_points': args.grid,
            'iters_outer': args.iters,
            'device': args.device
        }
    }

    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to: {output_file}")
    print()

    # Check GPU memory
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used',
                               '--format=csv,noheader,nounits'],
                              capture_output=True, text=True)
        mem_mb = int(result.stdout.strip())
        print(f"Final GPU memory: {mem_mb} MB ({mem_mb/1024:.2f} GB)")
        if mem_mb > 3072:
            print("⚠️  WARNING: Exceeded 3GB target")
        else:
            print("✓ Within 3GB memory target")
    except:
        pass

    print()
    print("Done!")

if __name__ == '__main__':
    main()
