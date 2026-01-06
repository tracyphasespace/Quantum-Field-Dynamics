"""Test if C-12 works with c_sym=0"""
import sys
sys.path.insert(0, 'src')
from qfd_metaopt_ame2020 import run_qfd_solver, M_PROTON, M_NEUTRON, M_ELECTRON
import pandas as pd

# C-12 optimized parameters BUT with c_sym=0
params_with_csym = {
    'c_v2_base': 3.643,
    'c_v2_iso': 0.0135,
    'c_v2_mass': 0.0005,
    'c_v4_base': 9.33,
    'c_v4_size': -0.129,
    'alpha_e_scale': 1.181,
    'beta_e_scale': 0.523,
    'c_sym': 22.3,  # Original
    'kappa_rho': 0.044
}

params_no_csym = {
    'c_v2_base': 3.643,
    'c_v2_iso': 0.0135,
    'c_v2_mass': 0.0005,
    'c_v4_base': 9.33,
    'c_v4_size': -0.129,
    'alpha_e_scale': 1.181,
    'beta_e_scale': 0.523,
    'c_sym': 0.0,  # ABOLISHED
    'kappa_rho': 0.044
}

ame_data = pd.read_csv('data/ame2020_system_energies.csv')
Z, A = 6, 12
row = ame_data[(ame_data['Z'] == Z) & (ame_data['A'] == A)]
E_exp = float(row.iloc[0]['E_exp_MeV'])

print("=" * 80)
print("C-12 TEST: Does c_sym=0 break the solver?")
print("=" * 80)

for label, params in [("WITH c_sym=22.3", params_with_csym), ("WITHOUT c_sym (=0)", params_no_csym)]:
    print(f"\n{label}:")
    print(f"  c_sym = {params['c_sym']:.1f}")
    
    result = run_qfd_solver(A, Z, params, verbose=False, fast_mode=True, device='cuda')
    
    if result is None or result.get('status') != 'ok':
        print(f"  ✗ SOLVER FAILED!")
        continue
    
    E_interaction = result['E_model']
    N = A - Z
    M_constituents = Z * M_PROTON + N * M_NEUTRON + Z * M_ELECTRON
    E_pred = M_constituents + E_interaction
    error_MeV = E_pred - E_exp
    error_pct = 100 * error_MeV / E_exp
    virial = result.get('virial_abs', abs(result.get('virial', 999)))
    
    print(f"  E_exp   = {E_exp:.1f} MeV")
    print(f"  E_pred  = {E_pred:.1f} MeV")
    print(f"  Error   = {error_MeV:+.1f} MeV ({error_pct:+.4f}%)")
    print(f"  Virial  = {virial:.4f}")
    print(f"  Status  = {'✓ Good' if abs(error_pct) < 1.0 and virial < 0.5 else '✗ Poor'}")

print("\n" + "=" * 80)
print("DIAGNOSIS")
print("=" * 80)
print("If c_sym=0 makes C-12 fail, then we need to re-optimize with c_sym=0.")
print("If c_sym=0 still works for C-12, then the octet failure is from heavy nuclei.")

