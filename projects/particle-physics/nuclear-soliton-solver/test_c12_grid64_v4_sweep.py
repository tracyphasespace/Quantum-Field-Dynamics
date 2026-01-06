"""Test C-12 with grid=64 and varying c_v2_base to find sign flip

Best result so far: grid=64, c_v2_base=3.643 → E_total = +48.1 MeV

Hypothesis: Slightly stronger V4 with grid=64 might cross zero without over-compressing.

Test c_v2_base = [3.64, 5, 6, 7, 8, 9, 10] with grid=64
"""
import sys
import torch
sys.path.insert(0, 'src')
from qfd_solver import Phase8Model, RotorParams, scf_minimize, torch_det_seed
from qfd_metaopt_ame2020 import M_PROTON
import pandas as pd

# Load target
ame_data = pd.read_csv('data/ame2020_system_energies.csv')
row = ame_data[(ame_data['Z'] == 6) & (ame_data['A'] == 12)]
exp_mass_total = float(row.iloc[0]['E_exp_MeV'])
target_stability = exp_mass_total - (12 * M_PROTON)

print("=" * 95)
print("C-12 COMBINED TEST: Grid=64 + V4 Strength Sweep")
print("=" * 95)
print()
print(f"Target stability energy: {target_stability:.2f} MeV")
print()
print(f"{'c_v2':>6s}  {'E_total':>8s}  {'T_total':>8s}  {'V4_total':>9s}  {'V6_total':>9s}  {'Virial':>8s}  {'Error':>9s}  {'Sign':>6s}")
print("-" * 95)

test_c_v2 = [3.643, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

for c_v2_base_test in test_c_v2:
    params = {
        'c_v2_base': c_v2_base_test,
        'c_v2_iso': 0.0135,
        'c_v2_mass': 0.0005,
        'c_v4_base': 9.33,
        'c_v4_size': -0.129,
        'alpha_e_scale': 1.181,
        'beta_e_scale': 0.523,
        'c_sym': 0.0,
        'kappa_rho': 0.044
    }

    seed = 4242
    torch_det_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    rotor = RotorParams(lambda_R2=3e-4, lambda_R3=1e-3, B_target=0.01)
    model = Phase8Model(
        A=12, Z=6,
        grid=64,  # Finer grid
        dx=1.0,
        c_v2_base=params['c_v2_base'],
        c_v2_iso=params['c_v2_iso'],
        c_v2_mass=params['c_v2_mass'],
        c_v4_base=params['c_v4_base'],
        c_v4_size=params['c_v4_size'],
        rotor=rotor,
        device=str(device),
        coulomb_mode="spectral",
        alpha_coul=1.0,
        kappa_rho=params['kappa_rho'],
        alpha_e_scale=params['alpha_e_scale'],
        beta_e_scale=params['beta_e_scale'],
        c_sym=params['c_sym'],
        alpha_model="exp",
        coulomb_twopi=False,
        mass_penalty_N=0.0,
        mass_penalty_e=0.0,
        project_mass_each=False,
        project_e_each=False,
    )

    model.initialize_fields(seed=seed, init_mode="gauss_cluster")

    best_result, virial, energy_terms = scf_minimize(
        model, iters_outer=200, lr_psi=0.015, lr_B=0.005,
        early_stop_vir=0.18, verbose=False
    )

    E_total = float(best_result["E"])
    T_total = float(energy_terms['T_N'] + energy_terms['T_e'] + energy_terms.get('T_rotor', 0))
    V4_total = float(energy_terms.get('V4_N', 0) + energy_terms.get('V4_e', 0))
    V6_total = float(energy_terms.get('V6_N', 0) + energy_terms.get('V6_e', 0))
    vir = float(abs(virial))
    error = E_total - target_stability

    sign_status = "NEG✓" if E_total < 0 else "POS✗"
    if E_total < 0 and abs(error) < 30:
        sign_status = "WIN!"

    print(f"{c_v2_base_test:6.2f}  {E_total:+8.1f}  {T_total:+8.1f}  {V4_total:+9.1f}  {V6_total:+9.1f}  {vir:8.4f}  {error:+9.1f}  {sign_status:>6s}")

print()
print("=" * 95)
print("ANALYSIS:")
print("  Looking for the sign flip point where E_total goes from positive to negative")
print("  If no flip found, need to investigate SCF initialization or loss function")
print("=" * 95)
