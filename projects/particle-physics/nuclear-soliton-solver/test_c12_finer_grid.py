"""Test C-12 with finer grid to reduce kinetic energy artifacts

Current problem: Solver finds compressed states with high kinetic energy.

Hypothesis: Coarse grid (32 points) overestimates gradient energy.
            Finer grid (48 or 64 points) should give lower kinetic energy.

Test grid = 48 with original c_v2_base = 3.643
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

# Original parameters
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

print("=" * 90)
print("C-12 GRID RESOLUTION TEST - Reducing Kinetic Energy Artifacts")
print("=" * 90)
print()
print(f"Target stability energy: {target_stability:.2f} MeV")
print()
print(f"{'Grid':>5s}  {'E_total':>8s}  {'T_total':>8s}  {'V4_total':>9s}  {'V6_total':>9s}  {'Virial':>8s}  {'Error':>8s}")
print("-" * 90)

for grid_size in [32, 48, 64]:
    seed = 4242
    torch_det_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    rotor = RotorParams(lambda_R2=3e-4, lambda_R3=1e-3, B_target=0.01)
    model = Phase8Model(
        A=12, Z=6,
        grid=grid_size,  # ← TESTING THIS
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

    # SCF with more iterations for finer grids
    iters = 150 if grid_size == 32 else 200
    best_result, virial, energy_terms = scf_minimize(
        model, iters_outer=iters, lr_psi=0.015, lr_B=0.005,
        early_stop_vir=0.18, verbose=False
    )

    E_total = float(best_result["E"])
    T_total = float(energy_terms['T_N'] + energy_terms['T_e'] + energy_terms.get('T_rotor', 0))
    V4_total = float(energy_terms.get('V4_N', 0) + energy_terms.get('V4_e', 0))
    V6_total = float(energy_terms.get('V6_N', 0) + energy_terms.get('V6_e', 0))
    vir = float(abs(virial))
    error = E_total - target_stability

    print(f"{grid_size:5d}  {E_total:+8.1f}  {T_total:+8.1f}  {V4_total:+9.1f}  {V6_total:+9.1f}  {vir:8.4f}  {error:+8.1f}")

print()
print("=" * 90)
print("KEY INSIGHT TO LOOK FOR:")
print("  - Does finer grid reduce T_total (kinetic energy)?")
print("  - Does E_total approach -81.33 MeV target?")
print("  - Does sign flip from positive to negative?")
print("=" * 90)
