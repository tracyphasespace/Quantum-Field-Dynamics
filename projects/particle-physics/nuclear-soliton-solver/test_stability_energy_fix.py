"""Test corrected stability energy comparison for C-12"""
import sys
sys.path.insert(0, 'src')
from parallel_objective import run_solver_direct
from qfd_metaopt_ame2020 import M_PROTON
import pandas as pd

# C-12 optimized params with c_sym=0
params = {
    'c_v2_base': 3.643,
    'c_v2_iso': 0.0135,
    'c_v2_mass': 0.0005,
    'c_v4_base': 9.33,
    'c_v4_size': -0.129,
    'alpha_e_scale': 1.181,
    'beta_e_scale': 0.523,
    'c_sym': 0.0,
    'kappa_rho': 0.044
}

# Run solver
result = run_solver_direct(A=12, Z=6, params=params, grid_points=32, iters_outer=150, device='cuda')

# Load AME data
ame_data = pd.read_csv('data/ame2020_system_energies.csv')
row = ame_data[(ame_data['Z'] == 6) & (ame_data['A'] == 12)]
exp_mass_total = float(row.iloc[0]['E_exp_MeV'])

print("=" * 80)
print("C-12 STABILITY ENERGY TEST (Corrected QFD Comparison)")
print("=" * 80)
print()
print("Experimental Data:")
print(f"  Total mass (AME): {exp_mass_total:.2f} MeV")
print()
print("QFD Baseline:")
print(f"  M_proton = {M_PROTON:.2f} MeV")
print(f"  A = 12")
print(f"  Vacuum baseline = 12 × {M_PROTON:.2f} = {12 * M_PROTON:.2f} MeV")
print()
print("Target Stability Energy:")
target_stability = exp_mass_total - (12 * M_PROTON)
print(f"  E_target = {exp_mass_total:.2f} - {12*M_PROTON:.2f}")
print(f"  E_target = {target_stability:.2f} MeV")
print(f"  {'✓ NEGATIVE (stable)' if target_stability < 0 else '✗ POSITIVE (unstable)'}")
print()
print("Solver Result:")
solved_stability = result['E_model']
print(f"  E_model = {solved_stability:.2f} MeV")
print(f"  {'✓ NEGATIVE (stable)' if solved_stability < 0 else '✗ POSITIVE (unstable)'}")
print(f"  Virial = {result['virial']:.4f}")
print()
print("Comparison:")
error_abs = solved_stability - target_stability
error_rel = error_abs / abs(target_stability) if abs(target_stability) > 0.1 else error_abs
print(f"  Stability error = {solved_stability:.2f} - ({target_stability:.2f})")
print(f"                  = {error_abs:.2f} MeV")
print(f"  Relative error  = {error_rel*100:.2f}%")
print(f"  Loss term       = {error_rel**2:.6e}")
print()
print("Diagnosis:")
if abs(solved_stability) > 60 and abs(target_stability) > 60:
    ratio = abs(solved_stability) / abs(target_stability)
    print(f"  Magnitude ratio: {ratio:.3f}")
    if 0.9 < ratio < 1.1:
        print(f"  ✓ MAGNITUDE CORRECT (~{abs(target_stability):.0f} MeV)")
    if solved_stability * target_stability < 0:
        print(f"  ✗ SIGN FLIPPED: Solver found repulsive (+) instead of attractive (-)")
        print(f"  → Need to flip V4 sign or minimize instead of maximize")
    elif abs(error_rel) < 0.1:
        print(f"  ✓ EXCELLENT FIT: Both magnitude and sign correct!")
    else:
        print(f"  ⚠ Large error despite correct sign")
else:
    print(f"  Values too small to diagnose sign flip")

