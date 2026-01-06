#!/usr/bin/env python3
"""
Detailed analysis of optimized parameters - run each isotope individually.
"""
import sys
import json
import pandas as pd
sys.path.insert(0, 'src')

from parallel_objective import run_solver_subprocess
from qfd_metaopt_ame2020 import M_PROTON, M_NEUTRON, M_ELECTRON

# Load results
with open('optimization_result_20251231_005319.json') as f:
    results = json.load(f)

params_optimized = results['parameters_optimized']
target_isotopes = [(iso['Z'], iso['A']) for iso in results['target_isotopes']]

# Load AME data for comparison
ame_data = pd.read_csv('data/ame2020_system_energies.csv')
exp_data = {}
for Z, A in target_isotopes:
    row = ame_data[(ame_data['Z'] == Z) & (ame_data['A'] == A)]
    if not row.empty:
        exp_data[(Z, A)] = {
            'E_exp': float(row.iloc[0]['E_exp_MeV']),
            'sigma': float(row.iloc[0].get('E_uncertainty_MeV', 1.0))
        }

print("=" * 80)
print("DETAILED ANALYSIS: OPTIMIZED PARAMETERS")
print("=" * 80)
print()

print("Running solver for each isotope with optimized parameters...")
print()

isotope_results = []

for Z, A in target_isotopes:
    print(f"Solving {Z}-{A}...", end=" ", flush=True)

    result = run_solver_subprocess(
        A=A, Z=Z,
        params=params_optimized,
        grid_points=32,
        iters_outer=150,
        device='cuda',
        early_stop_vir=0.18
    )

    E_model = result.get('E_model', 0)
    virial = result.get('virial', 999)
    converged = result.get('converged', False)

    # Calculate full energy
    N = A - Z
    M_constituents = Z * M_PROTON + N * M_NEUTRON + Z * M_ELECTRON
    pred_E = M_constituents + E_model

    # Get experimental value
    exp_E = exp_data.get((Z, A), {}).get('E_exp', 0)

    # Calculate error
    if exp_E != 0:
        rel_error = (pred_E - exp_E) / exp_E
        error_pct = 100 * rel_error
    else:
        error_pct = 999

    isotope_results.append({
        'Z': Z, 'A': A,
        'E_model': E_model,
        'pred_E': pred_E,
        'exp_E': exp_E,
        'error_pct': error_pct,
        'virial': virial,
        'converged': converged
    })

    status = "✓" if converged else "✗"
    print(f"{status} E_model={E_model:.1f} MeV, vir={virial:.3f}, error={error_pct:+.2f}%")

print()
print("=" * 80)
print("SUMMARY TABLE")
print("=" * 80)
print()
print("Isotope   E_model(MeV)  Virial  Conv?  Pred(MeV)   Exp(MeV)    Error(%)")
print("-" * 80)

for r in isotope_results:
    conv_mark = "✓" if r['converged'] else "✗"
    print(f"{r['Z']:2d}-{r['A']:3d}   {r['E_model']:8.1f}     {r['virial']:6.3f}   {conv_mark}   "
          f"{r['pred_E']:9.1f}  {r['exp_E']:9.1f}   {r['error_pct']:+7.2f}")

print()
print("=" * 80)
print("VIRIAL ANALYSIS")
print("=" * 80)
print()

converged_count = sum(1 for r in isotope_results if r['converged'])
high_virial = [r for r in isotope_results if r['virial'] > 0.18]

print(f"Converged solutions (|virial| < 0.18): {converged_count}/8")
print(f"High virial solutions (|virial| > 0.18): {len(high_virial)}/8")
print()

if high_virial:
    print("Isotopes with high virial:")
    for r in high_virial:
        print(f"  {r['Z']:2d}-{r['A']:3d}: virial = {r['virial']:.3f}")
    print()

print("=" * 80)
print("BINDING ENERGY ERRORS")
print("=" * 80)
print()

errors = [r['error_pct'] for r in isotope_results if abs(r['error_pct']) < 100]
if errors:
    import numpy as np
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    print(f"Mean error: {mean_error:+.2f}%")
    print(f"Std deviation: {std_error:.2f}%")
    print(f"RMS error: {np.sqrt(np.mean([e**2 for e in errors])):.2f}%")
else:
    print("No valid predictions to analyze")

print()
print("Done!")
