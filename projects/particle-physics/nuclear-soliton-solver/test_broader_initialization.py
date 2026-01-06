"""Test broader initialization to access the expanded-branch (low-energy) state

GOAL: Reduce E_proton from 55 MeV to ~10 MeV by starting with a larger initial radius.

Current initialization (qfd_solver.py:164):
  R0 = 1.2 × A^(1/3)  → Compact, leads to compressed branch

New initialization:
  R0 = 2.5 × A^(1/3)  → Expanded, should favor low-kinetic-energy branch

Test both single proton and C-12 to verify the approach.
"""
import sys
import torch
sys.path.insert(0, 'src')
from qfd_solver import Phase8Model, RotorParams, torch_det_seed
from qfd_metaopt_ame2020 import M_PROTON
import pandas as pd

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

def solve_with_custom_init(A, Z, R0_multiplier, grid=64, iters=200):
    """Solve with custom initialization radius"""
    seed = 4242
    torch_det_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    rotor = RotorParams(lambda_R2=3e-4, lambda_R3=1e-3, B_target=0.01)
    model = Phase8Model(
        A=A, Z=Z, grid=grid, dx=1.0,
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

    # CUSTOM INITIALIZATION with broader radius
    with torch.no_grad():
        r2 = model._r2_grid()
        R0 = R0_multiplier * max(1.0, A) ** (1.0/3.0)
        sigma2_n = (0.60 * R0) ** 2
        sigma2_e = (1.00 * R0) ** 2

        gn = torch.exp(-r2 / (2.0 * sigma2_n))
        ge = torch.exp(-r2 / (2.0 * sigma2_e))

        torch.manual_seed(seed)
        model.psi_N.copy_(gn + 1e-3 * torch.randn_like(gn))
        model.psi_e.copy_(ge + 1e-3 * torch.randn_like(ge))
        model.B_N.zero_().add_(1e-3 * torch.randn_like(model.B_N))

        # Normalize to A and Z
        dV = model.dV
        nN = torch.sqrt((model.psi_N * model.psi_N).sum() * dV + 1e-24)
        ne = torch.sqrt((model.psi_e * model.psi_e).sum() * dV + 1e-24)
        if float(nN) > 0:
            model.psi_N.mul_((0.5 * A) / float(nN))
        if float(ne) > 0:
            model.psi_e.mul_((0.5 * Z) / float(ne))

    # Run SCF
    from qfd_solver import scf_minimize
    best_result, virial, energy_terms = scf_minimize(
        model, iters_outer=iters, lr_psi=0.015, lr_B=0.005,
        early_stop_vir=0.18, verbose=False
    )

    return best_result, virial, energy_terms

print("=" * 95)
print("BROADER INITIALIZATION TEST - Accessing the Expanded Branch")
print("=" * 95)
print()

# Load experimental target for C-12
ame_data = pd.read_csv('data/ame2020_system_energies.csv')
row = ame_data[(ame_data['Z'] == 6) & (ame_data['A'] == 12)]
exp_mass_total = float(row.iloc[0]['E_exp_MeV'])
target_stability = exp_mass_total - (12 * M_PROTON)

print(f"Target for C-12: E_stability = {target_stability:.2f} MeV")
print()

# Test different R0 multipliers
test_cases = [
    ("Original (1.2)", 1.2),
    ("Broader (2.0)", 2.0),
    ("Broader (2.5)", 2.5),
    ("Broader (3.0)", 3.0),
]

print("=" * 95)
print("SINGLE PROTON (A=1, Z=1) - Finding the Low-Energy State")
print("=" * 95)
print()
print(f"{'R0_mult':>10s}  {'E_proton':>10s}  {'T_total':>10s}  {'V4_total':>10s}  {'Virial':>10s}  {'Status':>15s}")
print("-" * 95)

E_proton_values = {}

for label, R0_mult in test_cases:
    best, vir, energies = solve_with_custom_init(A=1, Z=1, R0_multiplier=R0_mult, grid=64, iters=200)

    E_total = float(best["E"])
    T_total = float(energies['T_N'] + energies['T_e'] + energies.get('T_rotor', 0))
    V4_total = float(energies.get('V4_N', 0) + energies.get('V4_e', 0))
    vir_abs = abs(float(vir))

    E_proton_values[R0_mult] = E_total

    status = "✓ Low-energy!" if E_total < 20 else "✗ Compressed"

    print(f"{R0_mult:>10.1f}  {E_total:>10.2f}  {T_total:>10.2f}  {V4_total:>10.2f}  {vir_abs:>10.4f}  {status:>15s}")

print()
print("=" * 95)
print("C-12 NUCLEUS - Testing Stability Energy with Corrected Baseline")
print("=" * 95)
print()
print(f"{'R0_mult':>10s}  {'E_C12':>10s}  {'E_proton':>10s}  {'E_stab':>10s}  {'Target':>10s}  {'Error':>10s}  {'Status':>10s}")
print("-" * 95)

for label, R0_mult in test_cases:
    best, vir, energies = solve_with_custom_init(A=12, Z=6, R0_multiplier=R0_mult, grid=64, iters=200)

    E_C12 = float(best["E"])
    E_proton = E_proton_values.get(R0_mult, 55.0)
    E_stab = E_C12 - 12 * E_proton
    error = E_stab - target_stability

    sign_check = "NEG✓" if E_stab < 0 else "POS✗"
    if E_stab < 0 and abs(error) < 30:
        sign_check = "WIN!✓✓"
    elif E_stab < 0 and abs(error) < 100:
        sign_check = "CLOSE✓"

    print(f"{R0_mult:>10.1f}  {E_C12:>10.2f}  {E_proton:>10.2f}  {E_stab:>10.2f}  {target_stability:>10.2f}  {error:>+10.2f}  {sign_check:>10s}")

print()
print("=" * 95)
print("KEY METRICS:")
print("  - Target E_proton: ~10 MeV (to get E_stab ≈ -81 MeV)")
print("  - If E_proton drops below 20 MeV: Expanded branch accessed ✓")
print("  - If E_stab < 0 and |error| < 30 MeV: SUCCESS ✓✓")
print("=" * 95)
