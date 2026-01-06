"""Diagnose individual energy components for C-12"""
import sys
import torch
sys.path.insert(0, 'src')
from qfd_solver import Phase8Model, RotorParams, torch_det_seed

# C-12 params
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

# Quick SCF (just a few steps to see settled state)
from qfd_solver import scf_minimize
best_result, virial, energy_terms = scf_minimize(
    model, iters_outer=150, lr_psi=0.015, lr_B=0.005,
    early_stop_vir=0.18, verbose=False
)

print("=" * 80)
print("C-12 ENERGY COMPONENT BREAKDOWN")
print("=" * 80)
print()
print("Model Parameters:")
print(f"  alpha_eff (V4 coefficient) = {model.alpha_eff:.4f}")
print(f"  beta_eff (V6 coefficient)  = {model.beta_eff:.4f}")
print()
print("Energy Components (MeV):")
print()

total = 0.0
for key in sorted(energy_terms.keys()):
    val = float(energy_terms[key])
    total += val
    sign = '+' if val >= 0 else ''
    print(f"  {key:15s} = {sign}{val:10.2f} MeV")

print(f"  {'─'*15}   {'─'*10}")
print(f"  {'TOTAL':15s} = {'+' if total >= 0 else ''}{total:10.2f} MeV")
print()
print(f"Virial = {virial:.4f}")
print()
print("Analysis:")
kinetic_total = energy_terms['T_N'] + energy_terms['T_e'] + energy_terms.get('T_rotor', 0)
v4_total = energy_terms.get('V4_N', 0) + energy_terms.get('V4_e', 0)
v6_total = energy_terms.get('V6_N', 0) + energy_terms.get('V6_e', 0)
print(f"  Total Kinetic: {float(kinetic_total):.2f} MeV (should be positive)")
print(f"  Total V4:      {float(v4_total):.2f} MeV (should be NEGATIVE)")
print(f"  Total V6:      {float(v6_total):.2f} MeV (should be positive)")
print()
if v4_total > 0:
    print("  ✗ ERROR: V4 is POSITIVE (repulsive) - should be NEGATIVE (attractive)!")
else:
    print("  ✓ V4 is correctly negative (attractive)")
    
balance = float(kinetic_total + v4_total)
print()
print(f"  Kinetic + V4 balance: {balance:.2f} MeV")
if balance > 0:
    print("    → Kinetic dominates (repulsive regime)")
else:
    print("    → V4 dominates (attractive regime)")

