#!/usr/bin/env python3
"""
Validate Trial 32 parameters on representative isotopes with FULL solver resolution.

This uses the production settings (48 grid points, 360 iterations) instead of
the fast search settings (32 grid, 150 iterations) used during optimization.
"""

import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from qfd_metaopt_ame2020 import (
    load_ame2020_data,
    run_qfd_solver,
    M_PROTON,
    M_NEUTRON,
    M_ELECTRON,
)

# Trial 32 parameters
TRIAL32_PARAMS = {
    'c_v2_base': 2.201711,
    'c_v2_iso': 0.027035,
    'c_v2_mass': -0.000205,
    'c_v4_base': 5.282364,
    'c_v4_size': -0.085018,
    'alpha_e_scale': 1.007419,
    'beta_e_scale': 0.504312,
    'c_sym': 25.0,
    'kappa_rho': 0.029816,
}

# Representative isotopes from each region
TEST_ISOTOPES = {
    'light': [
        (2, 4, 'He-4'),
        (6, 12, 'C-12'),
        (8, 16, 'O-16'),
        (14, 28, 'Si-28'),
        (20, 40, 'Ca-40'),
        (20, 48, 'Ca-48'),
        (26, 56, 'Fe-56'),
    ],
    'medium': [
        (28, 62, 'Ni-62'),
        (29, 63, 'Cu-63'),
        (30, 64, 'Zn-64'),
        (47, 107, 'Ag-107'),
        (50, 120, 'Sn-120'),
    ],
    'heavy': [
        (79, 197, 'Au-197'),
        (80, 200, 'Hg-200'),
        (82, 206, 'Pb-206'),
        (82, 207, 'Pb-207'),
        (82, 208, 'Pb-208'),
        (92, 238, 'U-238'),
    ],
}

def validate_isotope(Z, A, name, E_exp_MeV, params):
    """Validate a single isotope with full solver resolution."""

    print(f"\n  {name} (Z={Z}, A={A}):")
    print(f"    E_exp = {E_exp_MeV:.2f} MeV")

    # Run with FULL resolution (fast_mode=False)
    data = run_qfd_solver(A, Z, params, verbose=False, fast_mode=False)

    if not data or not data.get('physical_success'):
        print(f"    ✗ FAILED TO CONVERGE")
        if data:
            vir = data.get('virial_abs', data.get('virial', 999))
            print(f"    Virial: {vir:.3f}")
        return None

    # Compute total energy
    E_interaction = data['E_model']
    N = A - Z
    M_constituents = Z * M_PROTON + N * M_NEUTRON + Z * M_ELECTRON
    E_total_QFD = M_constituents + E_interaction

    # Error analysis
    rel_error = (E_total_QFD - E_exp_MeV) / E_exp_MeV
    virial_abs = abs(float(data.get('virial_abs', data.get('virial', 0.0))))

    print(f"    E_QFD = {E_total_QFD:.2f} MeV")
    print(f"    Error = {rel_error*100:+.2f}%")
    print(f"    Virial = {virial_abs:.3f} {'✓' if virial_abs < 0.18 else '✗ (>0.18)'}")

    status = '✓' if abs(rel_error) < 0.10 and virial_abs < 0.18 else '✗'
    print(f"    Status: {status}")

    return {
        'name': name,
        'Z': Z,
        'A': A,
        'E_exp_MeV': E_exp_MeV,
        'E_total_QFD': E_total_QFD,
        'rel_error': rel_error,
        'rel_error_pct': rel_error * 100,
        'virial_abs': virial_abs,
        'converged': virial_abs < 0.18,
    }

def main():
    print("="*70)
    print("TRIAL 32 VALIDATION - FULL SOLVER RESOLUTION")
    print("="*70)
    print("\nSettings: 48 grid points, 360 iterations")
    print("Acceptance: |error| < 10%, virial < 0.18")

    # Load AME2020 data
    ame_df = load_ame2020_data('data/ame2020_system_energies.csv')

    all_results = {}

    for region in ['light', 'medium', 'heavy']:
        print(f"\n{'='*70}")
        print(f"{region.upper()} REGION")
        print(f"{'='*70}")

        results = []

        for Z, A, name in TEST_ISOTOPES[region]:
            # Find experimental value
            row = ame_df[(ame_df['Z'] == Z) & (ame_df['A'] == A)]
            if row.empty:
                print(f"\n  {name}: ✗ NOT FOUND in AME2020")
                continue

            E_exp_MeV = float(row.iloc[0]['E_exp_MeV'])

            result = validate_isotope(Z, A, name, E_exp_MeV, TRIAL32_PARAMS)
            if result:
                results.append(result)

        all_results[region] = results

        # Regional summary
        if results:
            errors = [r['rel_error_pct'] for r in results]
            virials = [r['virial_abs'] for r in results]
            converged = [r for r in results if r['converged']]

            print(f"\n  {region.upper()} SUMMARY:")
            print(f"    Converged: {len(converged)}/{len(results)}")
            print(f"    Mean error: {np.mean(errors):+.2f}%")
            print(f"    RMS error: {np.sqrt(np.mean([e**2 for e in errors])):.2f}%")
            print(f"    Max |error|: {max([abs(e) for e in errors]):.2f}%")
            print(f"    Mean virial: {np.mean(virials):.3f}")
            print(f"    Max virial: {max(virials):.3f}")

    # Overall summary
    print(f"\n{'='*70}")
    print("OVERALL SUMMARY")
    print(f"{'='*70}")

    for region in ['light', 'medium', 'heavy']:
        results = all_results[region]
        if results:
            converged = [r for r in results if r['converged']]
            errors = [r['rel_error_pct'] for r in converged]

            print(f"\n{region.upper()} ({len(converged)}/{len(results)} converged):")
            if errors:
                print(f"  Mean error: {np.mean(errors):+.2f}%")
                print(f"  Error range: {min(errors):+.2f}% to {max(errors):+.2f}%")

    # Save results
    output_file = Path("validation_trial32_full.json")
    with open(output_file, 'w') as f:
        json.dump({
            'parameters': TRIAL32_PARAMS,
            'results_by_region': all_results,
            'settings': {
                'grid_points': 48,
                'iterations': 360,
                'fast_mode': False,
            }
        }, f, indent=2)

    print(f"\n\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()
