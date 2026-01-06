#!/usr/bin/env python3
"""
Test CCL-seeded optimization on a single isotope.

Uses Core Compression Law to seed initial parameters,
then runs optimization to see if it converges better than blind search.
"""
import sys
import json
import time
import pandas as pd
import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, 'src')

from ccl_seeded_solver import seed_density_parameters, backbone_charge, charge_stress
from parallel_objective import run_solver_subprocess
from qfd_metaopt_ame2020 import M_PROTON, M_NEUTRON, M_ELECTRON

# Test isotope: Pb-208 (should be stable, well-studied)
Z, A = 82, 208

print("=" * 80)
print("CCL-SEEDED SINGLE-ISOTOPE OPTIMIZATION TEST")
print("=" * 80)
print()

print(f"Target: {Z}-{A} (Lead-208)")
print()

# Load experimental binding energy
ame_data = pd.read_csv('data/ame2020_system_energies.csv')
row = ame_data[(ame_data['Z'] == Z) & (ame_data['A'] == A)]
E_exp = float(row.iloc[0]['E_exp_MeV'])

print(f"Experimental data:")
print(f"  E_total_exp = {E_exp:.1f} MeV")
print()

# CCL predictions
Q_ccl = backbone_charge(A)
stress = charge_stress(Z, A)

print(f"CCL predictions:")
print(f"  Q_backbone = {Q_ccl:.2f}")
print(f"  Stress     = {stress:.2f}")
print(f"  Status     = {'Stable' if stress < 1.5 else 'Marginal'}")
print()

# Seed parameters using CCL
params_seeded = seed_density_parameters(Z, A)

print("=" * 80)
print("TESTING SEEDED PARAMETERS")
print("=" * 80)
print()

print("Running solver with CCL-seeded parameters...")
result_seeded = run_solver_subprocess(
    A=A, Z=Z,
    params=params_seeded,
    grid_points=48,  # Higher resolution for single test
    iters_outer=360,  # More iterations for convergence
    device='cuda',
    early_stop_vir=0.18
)

E_model_seeded = result_seeded.get('E_model', 0)
virial_seeded = result_seeded.get('virial', 999)
converged_seeded = result_seeded.get('converged', False)

# Calculate full energy
N = A - Z
M_constituents = Z * M_PROTON + N * M_NEUTRON + Z * M_ELECTRON
pred_E_seeded = M_constituents + E_model_seeded

# Calculate error
if E_exp != 0:
    error_seeded = (pred_E_seeded - E_exp) / E_exp * 100
else:
    error_seeded = 999

print()
print("Results with CCL-seeded parameters:")
print(f"  E_model     = {E_model_seeded:.1f} MeV")
print(f"  Virial      = {virial_seeded:.3f}")
print(f"  Converged   = {converged_seeded}")
print(f"  Pred E_tot  = {pred_E_seeded:.1f} MeV")
print(f"  Exp E_tot   = {E_exp:.1f} MeV")
print(f"  Error       = {error_seeded:+.2f}%")
print()

if not converged_seeded:
    print("⚠️  WARNING: Solver did not converge with seeded parameters")
    print("   Trying with default initial parameters for comparison...")
    print()

    # Default parameters (from original RunSpec)
    params_default = {
        'c_v2_base': 2.201711,
        'c_v2_iso': 0.027035,
        'c_v2_mass': -0.000205,
        'c_v4_base': 5.282364,
        'c_v4_size': -0.085018,
        'alpha_e_scale': 1.007419,
        'beta_e_scale': 0.504312,
        'c_sym': 25.0,
        'kappa_rho': 0.029816
    }

    result_default = run_solver_subprocess(
        A=A, Z=Z,
        params=params_default,
        grid_points=48,
        iters_outer=360,
        device='cuda',
        early_stop_vir=0.18
    )

    E_model_default = result_default.get('E_model', 0)
    virial_default = result_default.get('virial', 999)
    converged_default = result_default.get('converged', False)
    pred_E_default = M_constituents + E_model_default
    error_default = (pred_E_default - E_exp) / E_exp * 100 if E_exp != 0 else 999

    print("Results with default parameters:")
    print(f"  E_model     = {E_model_default:.1f} MeV")
    print(f"  Virial      = {virial_default:.3f}")
    print(f"  Converged   = {converged_default}")
    print(f"  Pred E_tot  = {pred_E_default:.1f} MeV")
    print(f"  Error       = {error_default:+.2f}%")
    print()

    print("=" * 80)
    print("COMPARISON")
    print("=" * 80)
    print()
    print(f"                Seeded     Default")
    print(f"  Virial:       {virial_seeded:7.3f}    {virial_default:7.3f}")
    print(f"  Converged:    {str(converged_seeded):7s}    {str(converged_default):7s}")
    print(f"  Error:        {error_seeded:+6.2f}%   {error_default:+6.2f}%")
    print()

    if virial_seeded < virial_default:
        print("✓ CCL seeding improved virial convergence")
    else:
        print("✗ CCL seeding did not improve virial")

else:
    print("✓ SUCCESS: Solver converged with CCL-seeded parameters!")
    print()
    print("  This suggests CCL seeding helps guide the solver to")
    print("  physical solutions in parameter space.")

print()
print("=" * 80)
print("NEXT STEPS")
print("=" * 80)
print()

if converged_seeded or (not converged_seeded and virial_seeded < 50):
    print("CCL seeding shows promise. Recommendations:")
    print()
    print("1. Test on more isotopes (especially Sn-120, Au-197, U-238)")
    print("2. Run multi-isotope optimization with seeded initial values")
    print("3. Compare convergence rate vs. blind optimization")
    print("4. Analyze which parameters CCL seeding adjusts most effectively")
else:
    print("CCL seeding alone insufficient. Additional strategies needed:")
    print()
    print("1. Increase grid resolution: 48 → 64 points")
    print("2. Increase SCF iterations: 360 → 500")
    print("3. Add CCL constraint to loss function")
    print("4. Investigate virial penalty coefficient (may be too harsh)")

print()
print("Done!")
