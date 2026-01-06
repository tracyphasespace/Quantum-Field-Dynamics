"""Test C-12 with VERY STRONG penalties to force sign flip

Previous result: penalty=10 → E_total = +4.5 MeV (very close to zero!)

Now test: penalty = [10, 20, 50, 100, 200] to push through zero
"""
import sys
import torch
sys.path.insert(0, 'src')
from qfd_solver import Phase8Model, RotorParams, torch_det_seed
from qfd_metaopt_ame2020 import M_PROTON
import pandas as pd

# Load target
ame_data = pd.read_csv('data/ame2020_system_energies.csv')
row = ame_data[(ame_data['Z'] == 6) & (ame_data['A'] == 12)]
exp_mass_total = float(row.iloc[0]['E_exp_MeV'])
target_stability = exp_mass_total - (12 * M_PROTON)

params = {
    'c_v2_base': 7.0,
    'c_v2_iso': 0.0135,
    'c_v2_mass': 0.0005,
    'c_v4_base': 9.33,
    'c_v4_size': -0.129,
    'alpha_e_scale': 1.181,
    'beta_e_scale': 0.523,
    'c_sym': 0.0,
    'kappa_rho': 0.044
}

print("=" * 95)
print("C-12 STRONG PENALTY TEST - Forcing Sign Flip")
print("=" * 95)
print()
print(f"Target: {target_stability:.2f} MeV")
print()
print(f"{'Penalty':>8s}  {'E_total':>8s}  {'T_total':>8s}  {'V4_total':>9s}  {'V6_total':>9s}  {'Virial':>8s}  {'Sign':>6s}")
print("-" * 95)

for penalty_weight in [10.0, 20.0, 50.0, 100.0, 200.0]:
    seed = 4242
    torch_det_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    rotor = RotorParams(lambda_R2=3e-4, lambda_R3=1e-3, B_target=0.01)
    model = Phase8Model(
        A=12, Z=6, grid=64, dx=1.0,
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

    model.initialize_fields(seed=seed, init_mode="gauss")

    optim = torch.optim.Adam([
        {"params": [model.psi_N], "lr": 0.015},
        {"params": [model.psi_e], "lr": 0.015},
        {"params": [model.B_N], "lr": 0.005},
    ])

    for it in range(1, 301):  # More iterations for strong penalties
        optim.zero_grad()
        energies = model.energies()
        total = sum(energies.values())
        vir = model.virial(energies)

        positive_penalty = torch.relu(total) ** 2
        loss = total + 10.0 * vir*vir + penalty_weight * positive_penalty

        loss.backward()
        optim.step()
        model.projections()

    # Final evaluation
    with torch.no_grad():
        energies = model.energies()
        E_total = float(sum(energies.values()))
        T_total = float(energies['T_N'] + energies['T_e'] + energies.get('T_rotor', 0))
        V4_total = float(energies.get('V4_N', 0) + energies.get('V4_e', 0))
        V6_total = float(energies.get('V6_N', 0) + energies.get('V6_e', 0))
        vir = float(abs(model.virial(energies)))

    sign_status = "NEG✓" if E_total < 0 else "POS✗"
    if E_total < 0:
        sign_status = "NEG✓✓"

    print(f"{penalty_weight:8.1f}  {E_total:+8.1f}  {T_total:+8.1f}  {V4_total:+9.1f}  {V6_total:+9.1f}  {vir:8.4f}  {sign_status:>6s}")

print()
print("=" * 95)
print("Looking for the magic threshold where E_total crosses zero...")
print("=" * 95)
