"""Compute the baseline energy of a single proton soliton

HYPOTHESIS: The solver computes TOTAL field energy, not stability energy.
            E_stability = E_(C-12) - 12 × E_(single proton)

This test computes E_(single proton) to establish the baseline.
"""
import sys
import torch
sys.path.insert(0, 'src')
from qfd_solver import Phase8Model, RotorParams, scf_minimize, torch_det_seed

params = {
    'c_v2_base': 7.0,  # Best parameters from grid=64 sweep
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
print("BASELINE TEST: Single Proton Energy")
print("=" * 90)
print()
print("Solving for A=1, Z=1 (single proton) with same parameters as C-12...")
print()

seed = 4242
torch_det_seed(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

rotor = RotorParams(lambda_R2=3e-4, lambda_R3=1e-3, B_target=0.01)
model = Phase8Model(
    A=1, Z=1,  # Single proton!
    grid=64,
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

model.initialize_fields(seed=seed, init_mode="gauss")

best_result, virial, energy_terms = scf_minimize(
    model, iters_outer=200, lr_psi=0.015, lr_B=0.005,
    early_stop_vir=0.18, verbose=False
)

E_proton = float(best_result["E"])

print("Results:")
print(f"  E_proton (single) = {E_proton:+.2f} MeV")
print(f"  Virial            = {abs(virial):.4f}")
print()
print("Energy components:")
for key in sorted(energy_terms.keys()):
    val = float(energy_terms[key])
    if abs(val) > 0.01:  # Only show non-zero terms
        print(f"  {key:15s} = {val:+10.2f} MeV")

print()
print("=" * 90)
print("BASELINE HYPOTHESIS:")
print(f"  If C-12 E_model ≈ +46 MeV (from earlier tests)")
print(f"  And single proton E_proton = {E_proton:+.2f} MeV")
print(f"  Then 12 scattered protons = 12 × {E_proton:.2f} = {12*E_proton:+.2f} MeV")
print()
print(f"  Stability energy = E_(C-12) - E_(12 protons)")
print(f"                   = +46 - ({12*E_proton:+.2f})")
print(f"                   = {46 - 12*E_proton:+.2f} MeV")
print()
print("  Target stability = -81.33 MeV")
print()
if abs(46 - 12*E_proton - (-81.33)) < 20:
    print("  ✓✓✓ HYPOTHESIS CONFIRMED! This accounts for the sign flip!")
else:
    print("  ✗ Doesn't match. Need different explanation.")
print("=" * 90)
