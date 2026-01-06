#!/usr/bin/env python3
"""Quick test with 4 workers on 2 isotopes"""
import sys
import json
sys.path.insert(0, 'src')
import pandas as pd
from parallel_objective import ParallelObjective
from scipy.optimize import differential_evolution
from tqdm import tqdm
import time

# Load config
with open('experiments/nuclear_heavy_minimal_test.runspec.json') as f:
    config = json.load(f)

# Get first 3 parameters for quick test
params_initial = {}
param_names = []
param_bounds = {}
for p in config['parameters'][:3]:  # Just 3 params for speed
    if not p.get('frozen', False):
        name = p['name'].replace('nuclear.', '').replace('solver.', '')
        params_initial[name] = p['value']
        param_names.append(name)
        param_bounds[name] = tuple(p['bounds'])

print(f'Testing with 4 workers, 2 isotopes, 3 parameters')
print(f'Parameters: {param_names}')
print()

# Load AME data
ame_data = pd.read_csv('data/ame2020_system_energies.csv')

# Use just 2 isotopes
cuts = config['datasets'][0]['cuts']
target_isotopes = [(cuts['target_isotopes'][0]['Z'], cuts['target_isotopes'][0]['A']),
                   (cuts['target_isotopes'][1]['Z'], cuts['target_isotopes'][1]['A'])]

# Create objective with 4 workers
parallel_obj = ParallelObjective(
    target_isotopes=target_isotopes,
    ame_data=ame_data,
    max_workers=4,  # ← 4 workers instead of 2
    grid_points=32,
    iters_outer=150,
    device='cuda',
    early_stop_vir=0.18,
    verbose=False
)

# Prepare bounds
bounds_list = [param_bounds[name] for name in param_names]

# Setup progress bar
total_evals = 2 * 6 * 3  # 2 iters × 6 popsize × 3 params = 36 evals
pbar = tqdm(total=total_evals, desc="Quick Test (4 workers)", unit="eval")
best_loss = [float('inf')]

def objective_vec(x):
    params = params_initial.copy()
    for i, name in enumerate(param_names):
        params[name] = x[i]
    loss = parallel_obj(params)
    pbar.update(1)
    if loss < best_loss[0]:
        best_loss[0] = loss
        pbar.set_postfix_str(f"Best: {loss:.2e}")
    return loss

def callback(xk, convergence):
    pbar.set_description(f"Quick Test (conv={convergence:.4f})")
    return False

print('Starting optimization...')
start = time.time()

result = differential_evolution(
    objective_vec,
    bounds_list,
    maxiter=2,
    popsize=6,
    workers=1,
    seed=42,
    callback=callback,
    atol=0.01,
    tol=0.01
)

pbar.close()
elapsed = time.time() - start

print()
print(f'✓ Completed in {elapsed:.1f} seconds')
print(f'Final loss: {result.fun:.6e}')
print(f'Time per evaluation: {elapsed/total_evals:.1f}s')
print()
print('Testing 4 workers - check GPU memory usage!')
