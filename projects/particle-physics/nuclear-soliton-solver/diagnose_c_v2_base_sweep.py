"""Diagnose energy components as we vary c_v2_base

Goal: Understand why increasing c_v2_base makes E_model MORE positive,
when we expected it to make V4 stronger (more negative).
"""
import sys
import torch
sys.path.insert(0, 'src')
from qfd_solver import Phase8Model, RotorParams, scf_minimize, torch_det_seed

# Test multiple c_v2_base values
test_values = [3.643, 5.0, 7.0, 10.0, 15.0, 20.0]

print("=" * 90)
print("C-12 ENERGY COMPONENT BREAKDOWN vs c_v2_base")
print("=" * 90)
print()
print(f"{'c_v2':>6s}  {'E_total':>8s}  {'T_total':>8s}  {'V4_total':>9s}  {'V6_total':>9s}  {'alpha_eff':>9s}  {'virial':>8s}")
print("-" * 90)

for c_v2_base_test in test_values:
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

    # Create model
    seed = 4242
    torch_det_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    rotor = RotorParams(lambda_R2=3e-4, lambda_R3=1e-3, B_target=0.01)
    model = Phase8Model(
        A=12, Z=6,
        grid=32,
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

    # Initialize
    model.initialize_fields(seed=seed, init_mode="gauss_cluster")

    # Quick SCF
    best_result, virial, energy_terms = scf_minimize(
        model, iters_outer=150, lr_psi=0.015, lr_B=0.005,
        early_stop_vir=0.18, verbose=False
    )

    # Extract components
    E_total = float(best_result["E"])
    T_total = float(energy_terms['T_N'] + energy_terms['T_e'] + energy_terms.get('T_rotor', 0))
    V4_total = float(energy_terms.get('V4_N', 0) + energy_terms.get('V4_e', 0))
    V6_total = float(energy_terms.get('V6_N', 0) + energy_terms.get('V6_e', 0))
    alpha_eff = float(model.alpha_eff)
    vir = float(abs(virial))

    print(f"{c_v2_base_test:6.2f}  {E_total:+8.1f}  {T_total:+8.1f}  {V4_total:+9.1f}  {V6_total:+9.1f}  {alpha_eff:9.4f}  {vir:8.4f}")

print()
print("=" * 90)
print("KEY INSIGHT TO LOOK FOR:")
print("  - Does V4_total become MORE negative as c_v2_base increases?")
print("  - Or does V4_total become LESS negative (less attractive)?")
print("  - What happens to alpha_eff (the actual V4 coefficient)?")
print()
print("HYPOTHESIS:")
print("  If alpha_eff DECREASES as c_v2_base increases, then the problem is")
print("  in the alpha_eff calculation (lines 137-154 in qfd_solver.py).")
print("=" * 90)
