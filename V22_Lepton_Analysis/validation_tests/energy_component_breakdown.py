#!/usr/bin/env python3
"""
Energy Component Breakdown

Examine E_stab, E_circ, E_grad, E_boundary for each lepton
to identify which component has the pathological scaling.

Goal: Check if any single energy term has the right ratio structure
      to explain S_τ/S_μ ≈ 1.86
"""

import numpy as np
from profile_likelihood_boundary_layer import LeptonFitter, calibrate_lambda

# Physical constants
M_E = 0.511
M_MU = 105.7
M_TAU = 1776.8

# Best fit parameters
BETA = 3.15
W = 0.020
ETA_TARGET = 0.03
R_C_REF = 0.88

print("=" * 70)
print("ENERGY COMPONENT BREAKDOWN")
print("=" * 70)
print()

# Run fit
lam = calibrate_lambda(ETA_TARGET, BETA, R_C_REF)
fitter = LeptonFitter(beta=BETA, w=W, lam=lam, sigma_model=1e-4)
result = fitter.fit(max_iter=200, seed=42)

# Extract energies
energies = result["energies"]
params = result["parameters"]

print("Best-fit parameters:")
print("-" * 70)
for lepton in ["electron", "muon", "tau"]:
    p = params[lepton]
    print(f"{lepton:8s}: R_c={p['R_c']:.4f}, U={p['U']:.4f}, A={p['A']:.4f}")
print()

# Energy components
print("Energy components (MeV):")
print("-" * 70)
print(f"{'Lepton':<10} {'E_stab':<12} {'E_circ':<12} {'E_grad':<12} {'E_total':<12}")
print("-" * 70)

energy_table = {}
for lepton in ["electron", "muon", "tau"]:
    e = energies[lepton]
    E_stab = e["E_stab"]
    E_circ = e["E_circ"]
    E_grad = e["E_grad"]
    E_total = e["E_total"]

    print(f"{lepton:<10} {E_stab:<12.6f} {E_circ:<12.6f} {E_grad:<12.6f} {E_total:<12.6f}")

    energy_table[lepton] = {
        "E_stab": E_stab,
        "E_circ": E_circ,
        "E_grad": E_grad,
        "E_total": E_total,
    }

print()

# Ratios to muon
print("Ratios to muon:")
print("-" * 70)
print(f"{'Lepton':<10} {'E_stab/E_μ':<12} {'E_circ/E_μ':<12} {'E_grad/E_μ':<12} {'E_tot/E_μ':<12}")
print("-" * 70)

E_mu = energy_table["muon"]

for lepton in ["electron", "tau"]:
    e = energy_table[lepton]

    ratio_stab = e["E_stab"] / E_mu["E_stab"] if E_mu["E_stab"] > 0 else 0
    ratio_circ = e["E_circ"] / E_mu["E_circ"] if E_mu["E_circ"] > 0 else 0
    ratio_grad = e["E_grad"] / E_mu["E_grad"] if E_mu["E_grad"] > 0 else 0
    ratio_tot = e["E_total"] / E_mu["E_total"] if E_mu["E_total"] > 0 else 0

    print(f"{lepton:<10} {ratio_stab:<12.4f} {ratio_circ:<12.4f} {ratio_grad:<12.4f} {ratio_tot:<12.4f}")

print()

# Key test: Which component ratio is closest to mass ratio?
print("Mass ratios (target):")
print("-" * 70)
m_e_ratio = M_E / M_MU
m_tau_ratio = M_TAU / M_MU
print(f"m_e/m_μ  = {m_e_ratio:.4f}")
print(f"m_τ/m_μ  = {m_tau_ratio:.4f}")
print()

# Compute scale factors S_ℓ = m_ℓ/E_ℓ
print("Scale factors S_ℓ = m_ℓ/E_ℓ:")
print("-" * 70)
S_e = M_E / energy_table["electron"]["E_total"]
S_mu = M_MU / energy_table["muon"]["E_total"]
S_tau = M_TAU / energy_table["tau"]["E_total"]

print(f"S_e   = {S_e:.4f}")
print(f"S_μ   = {S_mu:.4f}")
print(f"S_τ   = {S_tau:.4f}")
print()

print(f"S_e/S_μ  = {S_e/S_mu:.4f}  (vs m_e/m_μ = {m_e_ratio:.4f})")
print(f"S_τ/S_μ  = {S_tau/S_mu:.4f}  (vs m_τ/m_μ = {m_tau_ratio:.4f})")
print()

# Analysis
print("=" * 70)
print("ANALYSIS")
print("=" * 70)
print()

# Check if any component ratio explains S_τ/S_μ
tau_energy_ratios = {
    "E_stab": energy_table["tau"]["E_stab"] / E_mu["E_stab"] if E_mu["E_stab"] > 0 else 0,
    "E_circ": energy_table["tau"]["E_circ"] / E_mu["E_circ"] if E_mu["E_circ"] > 0 else 0,
    "E_grad": energy_table["tau"]["E_grad"] / E_mu["E_grad"] if E_mu["E_grad"] > 0 else 0,
    "E_total": energy_table["tau"]["E_total"] / E_mu["E_total"] if E_mu["E_total"] > 0 else 0,
}

target_ratio = m_tau_ratio  # m_τ/m_μ ≈ 16.8
observed_S_ratio = S_tau / S_mu  # S_τ/S_μ ≈ 1.86

print("If mass scales with single energy component, which matches best?")
print("-" * 70)
print(f"{'Component':<15} {'E_τ/E_μ':<12} {'vs m_τ/m_μ':<15} {'Error':<12}")
print("-" * 70)

for comp, ratio in tau_energy_ratios.items():
    error = abs(ratio - target_ratio) / target_ratio
    print(f"{comp:<15} {ratio:<12.4f} {target_ratio:<15.4f} {error*100:<12.1f}%")

print()
print("If we need correction factor F to get m = S·F·E, what F_τ/F_μ is needed?")
print("-" * 70)

# We have: m_τ/m_μ = (S_τ/S_μ) · (F_τ/F_μ) · (E_τ/E_μ)
# Target: m_τ/m_μ = 16.8
# Current: S_τ/S_μ ≈ 1.86, E_τ/E_μ ≈ 9.06
# So: F_τ/F_μ = (m_τ/m_μ) / [(S_τ/S_μ) · (E_τ/E_μ)]

E_ratio = tau_energy_ratios["E_total"]
required_F_ratio = target_ratio / (observed_S_ratio * E_ratio) if (observed_S_ratio * E_ratio) > 0 else 0

print(f"m_τ/m_μ          = {target_ratio:.4f}")
print(f"S_τ/S_μ          = {observed_S_ratio:.4f}")
print(f"E_τ/E_μ          = {E_ratio:.4f}")
print(f"Required F_τ/F_μ = {required_F_ratio:.4f}")
print()

# Component contributions
print("Component contributions (fraction of total energy):")
print("-" * 70)
print(f"{'Lepton':<10} {'E_stab %':<12} {'E_circ %':<12} {'E_grad %':<12}")
print("-" * 70)

for lepton in ["electron", "muon", "tau"]:
    e = energy_table[lepton]
    E_tot = e["E_total"]

    if E_tot > 0:
        frac_stab = 100 * e["E_stab"] / E_tot
        frac_circ = 100 * e["E_circ"] / E_tot
        frac_grad = 100 * e["E_grad"] / E_tot

        print(f"{lepton:<10} {frac_stab:<12.2f} {frac_circ:<12.2f} {frac_grad:<12.2f}")

print()
print("=" * 70)
print()

# Summary
print("SUMMARY:")
print("-" * 70)
print("1. If mass were proportional to E_total alone:")
print(f"   m_τ/m_μ should be {E_ratio:.2f}, but actual is {target_ratio:.2f}")
print(f"   Error: {abs(E_ratio - target_ratio)/target_ratio * 100:.1f}%")
print()

print("2. Current mapping m = S·E gives:")
print(f"   S_τ/S_μ = {observed_S_ratio:.4f} (instead of universal S)")
print(f"   This is the regime change we're trying to explain")
print()

print("3. To fix with m = S·F·E (one global S, computed F per lepton):")
print(f"   Need F_τ/F_μ ≈ {required_F_ratio:.4f}")
print(f"   Emergent-time F_t gave: ~1.0 (volume) or ~0.42 (energy)")
print(f"   → F_t doesn't match required ratio")
print()

print("4. Dominant energy component:")
dominant = {}
for lepton in ["electron", "muon", "tau"]:
    e = energy_table[lepton]
    E_tot = e["E_total"]
    if E_tot > 0:
        fractions = {
            "E_stab": e["E_stab"] / E_tot,
            "E_circ": e["E_circ"] / E_tot,
            "E_grad": e["E_grad"] / E_tot,
        }
        dominant[lepton] = max(fractions.items(), key=lambda x: x[1])
        print(f"   {lepton}: {dominant[lepton][0]} ({dominant[lepton][1]*100:.1f}%)")

print()
print("=" * 70)
