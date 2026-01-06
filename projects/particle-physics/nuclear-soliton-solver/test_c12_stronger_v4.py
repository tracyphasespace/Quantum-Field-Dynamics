"""Test C-12 with stronger V4 potential to fix sign flip

Current problem:
  c_v2_base = 3.643 → V4 = -46 MeV → E_total = +82.46 MeV (WRONG SIGN)

Hypothesis:
  c_v2_base = 15.0 → V4 ≈ -250 MeV → E_total ≈ -81 MeV (CORRECT SIGN)

This test tries 3 values: 10, 15, 20 to bracket the correct regime.
"""
import sys
sys.path.insert(0, 'src')
from parallel_objective import run_solver_direct
from qfd_metaopt_ame2020 import M_PROTON
import pandas as pd

# Load experimental target
ame_data = pd.read_csv('data/ame2020_system_energies.csv')
row = ame_data[(ame_data['Z'] == 6) & (ame_data['A'] == 12)]
exp_mass_total = float(row.iloc[0]['E_exp_MeV'])
target_stability = exp_mass_total - (12 * M_PROTON)

print("=" * 80)
print("C-12 V4 STRENGTH TEST - Fixing Sign Flip")
print("=" * 80)
print()
print(f"Target stability energy: {target_stability:.2f} MeV (negative = stable)")
print()

# Test three c_v2_base values
test_values = [10.0, 15.0, 20.0]

for c_v2_base_test in test_values:
    params = {
        'c_v2_base': c_v2_base_test,  # ← TESTING THIS
        'c_v2_iso': 0.0135,
        'c_v2_mass': 0.0005,
        'c_v4_base': 9.33,
        'c_v4_size': -0.129,
        'alpha_e_scale': 1.181,
        'beta_e_scale': 0.523,
        'c_sym': 0.0,
        'kappa_rho': 0.044
    }

    print(f"Testing c_v2_base = {c_v2_base_test:.1f}...")
    result = run_solver_direct(A=12, Z=6, params=params, grid_points=32,
                               iters_outer=150, device='cuda')

    if result['status'] == 'success':
        E_model = result['E_model']
        virial = result['virial']
        error = E_model - target_stability

        sign_check = "✓ NEGATIVE" if E_model < 0 else "✗ POSITIVE"
        error_pct = abs(error) / abs(target_stability) * 100

        print(f"  E_model = {E_model:+8.2f} MeV  {sign_check}")
        print(f"  Virial  = {virial:7.4f}")
        print(f"  Error   = {error:+8.2f} MeV ({error_pct:.1f}%)")

        if E_model < 0 and abs(error) < 20:
            print(f"  🎯 EXCELLENT! Sign correct and error < 20 MeV")
        elif E_model < 0:
            print(f"  ⚠ Sign correct but error still large")
        else:
            print(f"  ✗ Still positive - need stronger V4")
    else:
        print(f"  FAILED: {result.get('error', 'Unknown error')}")

    print()

print("=" * 80)
print("Analysis:")
print("  If E_model flips to negative: V4 strength hypothesis CONFIRMED")
print("  If still positive: Need even larger c_v2_base or finer grid")
print("=" * 80)
