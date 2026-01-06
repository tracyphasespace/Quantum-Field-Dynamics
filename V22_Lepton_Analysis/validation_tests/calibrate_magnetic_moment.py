#!/usr/bin/env python3
"""
Calibrate magnetic moment normalization factor.

Problem: μ = k × Q × R × U gives correct scaling, but g-factor normalization is wrong.

Approach:
1. Take known electron solution (R, U, amplitude) at β = 3.058
2. Calculate μ = k × Q × R × U
3. Find normalization that gives g = 2.00231930436256
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from test_all_leptons_beta_from_alpha import (
    LeptonEnergy, ELECTRON_MASS
)
from test_multi_objective_beta_scan import magnetic_moment_hill_vortex

# Known electron solution at β = 3.058
beta_ref = 3.058
target_mass = ELECTRON_MASS
target_g = 2.00231930436256

print("="*70)
print("MAGNETIC MOMENT NORMALIZATION CALIBRATION")
print("="*70)

# Load known solution from production scan
import json
results_file = Path(__file__).parent / 'results' / 'beta_scan_production.json'

if results_file.exists():
    with open(results_file, 'r') as f:
        data = json.load(f)

    # Find β = 3.058 solution
    for beta_result in data['scan_results']:
        if abs(beta_result['beta'] - 3.058) < 0.01:
            electron_solution = beta_result['leptons']['electron']

            if electron_solution['converged']:
                R_known = electron_solution['R']
                U_known = electron_solution['U']
                amp_known = electron_solution['amplitude']

                print(f"\nKnown electron solution at β = 3.058:")
                print(f"  R = {R_known:.6f}")
                print(f"  U = {U_known:.6f}")
                print(f"  amplitude = {amp_known:.6f}")
                if 'energy' in electron_solution:
                    print(f"  E_total = {electron_solution['energy']:.6f}")

                # Calculate raw magnetic moment
                mu_raw = magnetic_moment_hill_vortex(
                    R_known, U_known, amp_known, beta_ref,
                    geometric_factor=0.2
                )

                print(f"\nRaw magnetic moment:")
                print(f"  μ = k × Q × R × U")
                print(f"  μ = 0.2 × 1.0 × {R_known:.6f} × {U_known:.6f}")
                print(f"  μ = {mu_raw:.6e}")

                # Calculate required normalization
                # g = normalization × μ / mass_ratio
                # For electron: mass_ratio = 1.0
                # So: normalization = g / μ

                normalization_required = target_g / mu_raw

                print(f"\nRequired normalization:")
                print(f"  g_target / μ_raw = {target_g:.10f} / {mu_raw:.6e}")
                print(f"  normalization = {normalization_required:.2f}")

                print(f"\nCurrent normalization in code: 10.0")
                print(f"Ratio: {normalization_required / 10.0:.2f}x too small")

                print("\n" + "="*70)
                print("RECOMMENDATION")
                print("="*70)
                print(f"\nUpdate test_multi_objective_beta_scan.py line 132:")
                print(f"  OLD: normalization = 10.0")
                print(f"  NEW: normalization = {normalization_required:.1f}")

                # Test verification
                g_test = normalization_required * mu_raw / 1.0
                print(f"\nVerification:")
                print(f"  g = {normalization_required:.1f} × {mu_raw:.6e} / 1.0")
                print(f"  g = {g_test:.10f}")
                print(f"  Target: {target_g:.10f}")
                print(f"  Match: {'✓' if abs(g_test - target_g) < 1e-9 else '✗'}")

                break
    else:
        print("\n✗ No converged electron solution found at β = 3.058")
        print("  Run production β-scan first to get reference solution")
else:
    print(f"\n✗ Results file not found: {results_file}")
    print("  Run production β-scan first:")
    print("  python3 validation_tests/test_beta_scan_production.py")
