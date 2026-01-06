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
from scipy.optimize import differential_evolution
from tqdm import tqdm

sys.path.insert(0, 'src')
from parallel_objective import ParallelObjective

def main():
    parser = argparse.ArgumentParser(description='Run parallel optimization')
    parser.add_argument('--runspec', default='experiments/nuclear_heavy_minimal_test.runspec.json',
                       help='Path to RunSpec config file')
    parser.add_argument('--maxiter', type=int, default=3,
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
    parser.add_argument('--lr-psi', type=float, default=0.015,
                          help='Learning rate for psi field in SCF solver')
    parser.add_argument('--lr-b', type=float, default=0.005,
                        help='Learning rate for B field in SCF solver')
    args = parser.parse_args()

    print("=" * 80)
    print("PARALLEL OPTIMIZATION WITH GPU ACCELERATION")
    print("=" * 80)
    print(f"Start time: {datetime.now()}")
    print()

    # Load RunSpec
    with open(args.runspec) as f:
        config = json.load(f)

    print(f"RunSpec: {config['experiment_id']}")
    print()

    # Extract ALL parameters from the runspec, including frozen ones
    all_params = {}
    for p in config['parameters']:
        name = p['name'].replace('nuclear.', '').replace('solver.', '')
        all_params[name] = p['value']

    # Identify which parameters are being optimized (not frozen)
    params_bounds = {}
    param_names = []
    for p in config['parameters']:
        name = p['name'].replace('nuclear.', '').replace('solver.', '')
        if not p.get('frozen', False):
            if p.get('bounds'):
                params_bounds[name] = tuple(p['bounds'])
                param_names.append(name)
    
    # Store the initial values of ONLY the optimized parameters for final report
    params_initial_optimized = {name: all_params[name] for name in param_names}

    print(f"Optimizing {len(param_names)} parameters:")
    for name in param_names:
        print(f"  {name}: {all_params[name]:.6f} {params_bounds[name]}")
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
    print(f"  LR Psi: {args.lr_psi}, LR B: {args.lr_b}")
    print()

    parallel_obj = ParallelObjective(
        target_isotopes=target_isotopes,
        ame_data=ame_data,
        max_workers=args.workers,
        grid_points=args.grid,
        iters_outer=args.iters,
        device=args.device,
        lr_psi=args.lr_psi,
        lr_B=args.lr_b,
        verbose=False  # Don't print every isotope
    )

    # Prepare bounds for differential_evolution
    bounds_list = [params_bounds[name] for name in param_names]

    # Vectorized objective wrapper with TQDM progress bar
    eval_count = [0]
    best_loss = [float('inf')]
    start_time = time.time()

    # Estimate total evaluations for progress bar
    total_evals = args.maxiter * args.popsize * len(param_names)
    pbar = tqdm(total=total_evals, desc="Optimization", unit="eval",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] Loss: {postfix}')

    def objective_vec(x):
        """Convert parameter vector to dict and evaluate."""
        # Start with all parameters (including frozen)
        params = all_params.copy()
        # Update with the current values from the optimizer
        for i, name in enumerate(param_names):
            params[name] = x[i]

        loss = parallel_obj(params)
        eval_count[0] += 1
        pbar.update(1)

        if loss < best_loss[0]:
            best_loss[0] = loss
            pbar.set_postfix_str(f"Best: {loss:.2e}")

        return loss

    def callback(xk, convergence):
        """Progress callback for differential_evolution."""
        # Update progress bar description with convergence info
        pbar.set_description(f"Optimization (conv={convergence:.4f})")
        return False  # Continue optimization

    # Run optimization
    print("=" * 80)
    print(f"STARTING OPTIMIZATION")
    print(f"  Method: differential_evolution")
    print(f"  Generations (maxiter): {args.maxiter}")
    print(f"  Population size: {args.popsize}")
    print(f"  Total evaluations: ~{args.maxiter * args.popsize * len(param_names)}")
    print("=" * 80)
    print()

    result = differential_evolution(
        objective_vec,
        bounds_list,
        maxiter=args.maxiter,
        popsize=args.popsize,
        workers=1,  # Sequential DE (parallel WITHIN each objective call)
        seed=42,
        disp=True,
        callback=callback,
        atol=0.01,
        tol=0.01
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
    optimized_params = all_params.copy()
    for i, name in enumerate(param_names):
        optimized_params[name] = result.x[i]

    print("Optimized parameters:")
    for name in param_names:
        initial = params_initial_optimized[name]
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
            'method': 'differential_evolution',
            'maxiter': args.maxiter,
            'popsize': args.popsize,
            'total_evaluations': eval_count[0],
            'final_loss': float(result.fun),
            'success': result.success,
            'message': result.message
        },
        'parameters_initial': all_params,
        'parameters_optimized': optimized_params,
        'target_isotopes': [{'Z': Z, 'A': A} for Z, A in target_isotopes],
        'hardware': {
            'parallel_workers': args.workers,
            'grid_points': args.grid,
            'iters_outer': args.iters,
            'device': args.device,
            'lr_psi': args.lr_psi,
            'lr_b': args.lr_b
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
