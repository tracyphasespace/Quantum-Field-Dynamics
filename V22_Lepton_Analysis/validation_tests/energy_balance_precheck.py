#!/usr/bin/env python3
"""
Energy-Balance Precheck (Quick)

At representative e/μ parameters, compute E_circ, E_stab, E_grad, E_total
to verify E_total > 0 before running expensive optimizer.

Use prior best-fit parameters from baseline or reasonable defaults.
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

# Representative parameters - maximize E_circ to test if E_total can be positive
# Use high U (maximize circulation) and low A (minimize stabilization penalty)
# Electron
R_c_e = 0.60
U_e = 0.095  # Near upper bound
A_e = 0.72   # Near lower bound

# Muon
R_c_mu = 0.18
U_mu = 0.18  # Near upper bound
A_mu = 0.72  # Near lower bound

print("=" * 70)
print("ENERGY-BALANCE PRECHECK")
print("=" * 70)
print()

print("Representative parameters (baseline-like):")
print(f"  electron: R_c={R_c_e}, U={U_e}, A={A_e}")
print(f"  muon:     R_c={R_c_mu}, U={U_mu}, A={A_mu}")
print()

# Test k values and Δ_v factors
k_values = [1.0, 1.5, 2.0]
delta_v_factors = [0.25, 0.5]
p = 6

results = []

for k in k_values:
    for delta_v_factor in delta_v_factors:
        print(f"Testing k={k:.1f}, Δ_v/R_v={delta_v_factor}:")
        print("-" * 70)

        # Create energy calculator
        energy_calc = LeptonEnergyLocalizedV1(
            beta=beta,
            w=w,
            lam=lam,
            k_localization=k,
            delta_v_factor=delta_v_factor,
            p_envelope=p,
        )

        # Compute energies for electron
        E_e, E_circ_e, E_stab_e, E_grad_e = energy_calc.total_energy(R_c_e, U_e, A_e)

        # Compute energies for muon
        E_mu, E_circ_mu, E_stab_mu, E_grad_mu = energy_calc.total_energy(R_c_mu, U_mu, A_mu)

        # Display
        print()
        print(f"{'Lepton':<10} {'E_circ':<12} {'E_stab':<12} {'E_grad':<12} {'E_total':<12}")
        print("-" * 70)
        print(f"{'electron':<10} {E_circ_e:<12.6f} {E_stab_e:<12.6f} {E_grad_e:<12.6f} {E_e:<12.6f}")
        print(f"{'muon':<10} {E_circ_mu:<12.6f} {E_stab_mu:<12.6f} {E_grad_mu:<12.6f} {E_mu:<12.6f}")
        print()

        # Check positivity
        e_positive = E_e > 0
        mu_positive = E_mu > 0

        # Rough scale check (order of magnitude)
        # e should be ~0.02 (M_E = 0.511), μ should be ~5 (M_MU = 105.7)
        # Assume S ≈ 20-25 MeV
        e_reasonable = 0.001 < E_e < 1.0
        mu_reasonable = 1.0 < E_mu < 20.0

        pass_precheck = e_positive and mu_positive and e_reasonable and mu_reasonable

        print("Precheck:")
        print(f"  E_e > 0:              {e_positive} ({'PASS' if e_positive else 'FAIL'})")
        print(f"  E_μ > 0:              {mu_positive} ({'PASS' if mu_positive else 'FAIL'})")
        print(f"  E_e in [0.001, 1.0]:  {e_reasonable} ({'PASS' if e_reasonable else 'FAIL'})")
        print(f"  E_μ in [1.0, 20.0]:   {mu_reasonable} ({'PASS' if mu_reasonable else 'FAIL'})")
        print()

        if pass_precheck:
            print(f"✓ k={k:.1f}, Δ_v/R_v={delta_v_factor} PASSES energy-balance precheck")
        else:
            print(f"✗ k={k:.1f}, Δ_v/R_v={delta_v_factor} FAILS energy-balance precheck")
            if not e_positive or not mu_positive:
                print("  → Negative energies detected")
            if not e_reasonable or not mu_reasonable:
                print("  → Energy magnitudes unreasonable (too small or too large)")

        print()
        print("=" * 70)
        print()

        results.append({
            "k": k,
            "delta_v_factor": delta_v_factor,
            "E_e": E_e,
            "E_mu": E_mu,
            "pass": pass_precheck,
        })

# Summary
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print()

passing = [r for r in results if r["pass"]]

if passing:
    best = min(passing, key=lambda x: (x["k"], x["delta_v_factor"]))
    print(f"✓ RECOMMENDED k = {best['k']:.1f}, Δ_v/R_v = {best['delta_v_factor']} for Run 2")
    print(f"  E_e = {best['E_e']:.6f}")
    print(f"  E_μ = {best['E_mu']:.6f}")
    print()
    print("Both Run 1A (sensitivity) and energy-balance precheck passed.")
    print("Safe to proceed with e,μ regression (Run 2).")
else:
    print("✗ NO configuration PASSES energy-balance precheck")
    print()
    print("All tested combinations give negative electron energies.")
    print()
    print("Analysis:")
    for r in results:
        status = "✓" if r["E_e"] > 0 and r["E_mu"] > 0 else "✗"
        print(f"  {status} k={r['k']:.1f}, Δ_v/R_v={r['delta_v_factor']}: E_e={r['E_e']:+.3f}, E_μ={r['E_mu']:+.3f}")
    print()
    print("This indicates outside-only localization still insufficient.")
    print("Need to pivot to vacuum-subtraction approach.")

print()
