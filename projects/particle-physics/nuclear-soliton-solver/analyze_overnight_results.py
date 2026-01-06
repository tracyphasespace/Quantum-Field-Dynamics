#!/usr/bin/env python3
"""
Analyze the overnight optimization results in detail.
"""
import sys
import json
import pandas as pd
sys.path.insert(0, 'src')

from parallel_objective import ParallelObjective

# Load results
with open('optimization_result_20251231_005319.json') as f:
    results = json.load(f)

params_initial = results['parameters_initial']
params_optimized = results['parameters_optimized']
target_isotopes = [(iso['Z'], iso['A']) for iso in results['target_isotopes']]

# Load AME data
ame_data = pd.read_csv('data/ame2020_system_energies.csv')

# Create objective function
objective = ParallelObjective(
    target_isotopes=target_isotopes,
    ame_data=ame_data,
    max_workers=4,
    grid_points=32,
    iters_outer=150,
    device='cuda',
    early_stop_vir=0.18,
    verbose=True
)

print("=" * 80)
print("ANALYZING OVERNIGHT OPTIMIZATION RESULTS")
print("=" * 80)
print()

print("Initial Parameters:")
for name, value in params_initial.items():
    print(f"  {name:20s}: {value:10.6f}")
print()

print("Optimized Parameters:")
for name, value in params_optimized.items():
    change = 100 * (value - params_initial[name]) / params_initial[name]
    print(f"  {name:20s}: {value:10.6f} ({change:+.1f}%)")
print()

print("=" * 80)
print("EVALUATING INITIAL PARAMETERS")
print("=" * 80)
loss_initial = objective(params_initial)
print(f"\nInitial Loss: {loss_initial:.6f}")
print()

print("=" * 80)
print("EVALUATING OPTIMIZED PARAMETERS")
print("=" * 80)
loss_optimized = objective(params_optimized)
print(f"\nOptimized Loss: {loss_optimized:.6f}")
print()

print("=" * 80)
print("COMPARISON")
print("=" * 80)
print(f"Loss improvement: {loss_initial:.6f} → {loss_optimized:.6f}")
if loss_optimized < loss_initial:
    improvement_pct = 100 * (loss_initial - loss_optimized) / loss_initial
    print(f"Improvement: {improvement_pct:.2f}%")
else:
    print("No improvement achieved")
print()

print("Done!")
