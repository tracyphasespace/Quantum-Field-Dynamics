"""Test C-12 with the ACTUAL optimized parameters from the golden probe run"""
import sys
sys.path.insert(0, 'src')
from qfd_metaopt_ame2020 import run_qfd_solver, M_PROTON, M_NEUTRON, M_ELECTRON
import pandas as pd

# ACTUAL OPTIMIZED parameters from c12_golden_test.log (final values!)
params_optimized = {
    'c_v2_base': 3.642990,
    'c_v2_iso': 0.013487,
    'c_v2_mass': 0.000517,
    'c_v4_base': 9.326908,
    'c_v4_size': -0.129277,
    'alpha_e_scale': 1.181133,
    'beta_e_scale': 0.523409,
    'c_sym': 22.300922,  # Original optimized value
    'kappa_rho': 0.043896
}

# Same but with c_sym=0
params_no_csym = params_optimized.copy()
params_no_csym['c_sym'] = 0.0

ame_data = pd.read_csv('data/ame2020_system_energies.csv')
Z, A = 6, 12
row = ame_data[(ame_data['Z'] == Z) & (ame_data['A'] == A)]
E_exp = float(row.iloc[0]['E_exp_MeV'])

print("=" * 80)
print("C-12 TEST: Using ACTUAL OPTIMIZED parameters from golden probe")
print("=" * 80)

for label, params in [("WITH c_sym=22.3 (optimized)", params_optimized), 
                      ("WITH c_sym=0 (abolished)", params_no_csym)]:
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
    rel_error = error_MeV / E_exp
    virial = result.get('virial_abs', abs(result.get('virial', 999)))
    
    # Calculate loss components (matching parallel_objective.py)
    energy_loss = rel_error ** 2
    if abs(E_interaction) > 1.0:
        virial_loss = (virial / abs(E_interaction)) ** 2
    else:
        virial_loss = virial ** 2
    total_loss = energy_loss + 0.5 * virial_loss
    
    print(f"  E_exp       = {E_exp:.1f} MeV")
    print(f"  E_pred      = {E_pred:.1f} MeV")
    print(f"  Error       = {error_MeV:+.1f} MeV ({error_pct:+.4f}%)")
    print(f"  Virial      = {virial:.4f}")
    print(f"  Total loss  = {total_loss:.6e}")
    print(f"  Status      = {'✓ EXCELLENT' if total_loss < 0.001 else '✗ Poor'}")

