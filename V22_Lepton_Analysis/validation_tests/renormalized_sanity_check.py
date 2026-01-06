#!/usr/bin/env python3
"""
Corrected Sign Convention Sanity Check

Test that E_total = E_circ + E_stab + E_grad (all penalties add)
fixes the electron positivity issue.

Configuration: k=1.5, Δv/Rv=0.5, p=6 (best sensitivity + localization balance)
"""

import numpy as np
from lepton_energy_localized_v1 import LeptonEnergyLocalizedV1
from profile_likelihood_boundary_layer import calibrate_lambda

# Physical constants
M_E = 0.511
M_MU = 105.7

# Test parameters
beta = 3.15
w = 0.020
eta_target = 0.03
R_c_ref = 0.88
lam = calibrate_lambda(eta_target, beta, R_c_ref)

# Best configuration from U_star scan
k = 1.5
delta_v_factor = 0.5
p = 6

# Representative parameters (moderate)
R_c_e = 0.60
U_e = 0.095  # Within bounds
A_e = 0.75

R_c_mu = 0.18
U_mu = 0.18
A_mu = 0.75

print("=" * 80)
print("CORRECTED SIGN CONVENTION SANITY CHECK")
print("=" * 80)
print()
print("Testing: E_total = E_circ + E_stab + E_grad (all penalties add)")
print()

print("Configuration:")
print(f"  k = {k}")
print(f"  Δv/Rv = {delta_v_factor}")
print(f"  p = {p}")
print()

print("Parameters:")
print(f"  Electron: R_c={R_c_e}, U={U_e}, A={A_e}")
print(f"  Muon:     R_c={R_c_mu}, U={U_mu}, A={A_mu}")
print()

# Create energy calculator
energy_calc = LeptonEnergyLocalizedV1(
    beta=beta,
    w=w,
    lam=lam,
    k_localization=k,
    delta_v_factor=delta_v_factor,
    p_envelope=p,
)

# Compute renormalized energies
print("=" * 80)
print("ELECTRON")
print("=" * 80)

E_e, dE_circ_e, E_stab_e, E_grad_e = energy_calc.total_energy(R_c_e, U_e, A_e)

# Get detailed diagnostics
R_e = R_c_e + w
dE_circ_e_full, F_inner_e, I_e, E_circ_actual_e, E_circ_vac_e = \
    energy_calc.circulation_energy_with_diagnostics(R_e, U_e, A_e, R_c_e)

print()
print("Diagnostics:")
print(f"  E_circ (actual):    {E_circ_actual_e:12.6f}")
print(f"  E_circ (vacuum):    {E_circ_vac_e:12.6f}")
print(f"  ΔE_circ (deficit):  {dE_circ_e:12.6f} (expected < 0 for ρ < 1)")
print(f"  F_inner:            {F_inner_e:12.4f}")
print()

print("Energy components (corrected signs - all add):")
print(f"  E_circ:   {E_circ_actual_e:12.6f} (kinetic)")
print(f"  E_stab:   {E_stab_e:12.6f} (penalty)")
print(f"  E_grad:   {E_grad_e:12.6f} (penalty)")
print(f"  E_total:  {E_e:12.6f} (sum)")
print()

e_positive = E_e > 0
print(f"✓ E_total > 0:  {e_positive} ({'PASS' if e_positive else 'FAIL'})")
print()

# Muon
print("=" * 80)
print("MUON")
print("=" * 80)

E_mu, dE_circ_mu, E_stab_mu, E_grad_mu = energy_calc.total_energy(R_c_mu, U_mu, A_mu)

R_mu = R_c_mu + w
dE_circ_mu_full, F_inner_mu, I_mu, E_circ_actual_mu, E_circ_vac_mu = \
    energy_calc.circulation_energy_with_diagnostics(R_mu, U_mu, A_mu, R_c_mu)

print()
print("Diagnostics:")
print(f"  E_circ (actual):    {E_circ_actual_mu:12.6f}")
print(f"  E_circ (vacuum):    {E_circ_vac_mu:12.6f}")
print(f"  ΔE_circ (deficit):  {dE_circ_mu:12.6f} (expected < 0 for ρ < 1)")
print(f"  F_inner:            {F_inner_mu:12.4f}")
print()

print("Energy components (corrected signs - all add):")
print(f"  E_circ:   {E_circ_actual_mu:12.6f} (kinetic)")
print(f"  E_stab:   {E_stab_mu:12.6f} (penalty)")
print(f"  E_grad:   {E_grad_mu:12.6f} (penalty)")
print(f"  E_total:  {E_mu:12.6f} (sum)")
print()

mu_positive = E_mu > 0
print(f"✓ E_total > 0:  {mu_positive} ({'PASS' if mu_positive else 'FAIL'})")
print()

# Summary
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print()

print("Sign convention audit:")
print(f"  Old (wrong):  E_total = E_circ - E_stab + E_grad")
print(f"  New (correct): E_total = E_circ + E_stab + E_grad (all penalties add)")
print()

print("Energy scale ordering:")
print(f"  E_total,μ / E_total,e = {E_mu/E_e:.2f}")
print(f"  Expected (m_μ/m_e):     {M_MU/M_E:.2f}")
print()

both_positive = e_positive and mu_positive

if both_positive:
    print("✓ SANITY CHECK PASSED")
    print()
    print("Both electron and muon have E_total > 0 at reasonable parameters.")
    print("Corrected sign convention makes all penalty terms add consistently.")
    print()

    # Check if ordering is reasonable
    ratio_actual = E_mu / E_e if E_e > 0 else 0
    ratio_target = M_MU / M_E

    if 50 < ratio_actual < 500:
        print(f"Energy ratio ({ratio_actual:.1f}) is in reasonable range for mass ratio ({ratio_target:.1f}).")
        print()
        print("NEXT: Proceed to e,μ regression (Run 2)")
    else:
        print(f"⚠ Energy ratio ({ratio_actual:.1f}) may be too far from mass ratio ({ratio_target:.1f}).")
        print("  May require global scale factor S, but structure is correct.")
        print()
        print("NEXT: Test e,μ regression to see if optimizer finds stable fit")
else:
    print("✗ SANITY CHECK FAILED")
    print()
    if not e_positive:
        print(f"  Electron E_total = {E_e:.6f} < 0")
    if not mu_positive:
        print(f"  Muon E_total = {E_mu:.6f} < 0")
    print()
    print("Even with corrected signs, energies remain negative.")
    print("May need bulk potential or different physics.")

print()
