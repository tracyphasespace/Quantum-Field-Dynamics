#!/usr/bin/env python3
"""
Tau Collapse Diagnostics

Three decisive tests:
1. Per-lepton scales S_ℓ = m_ℓ/E_ℓ (regime change?)
2. Parameter bound saturation (hitting limits?)
3. Closure isolation (which ingredient causes collapse?)
"""

import numpy as np
from profile_likelihood_boundary_layer import LeptonFitter, calibrate_lambda

# Physical constants
M_E = 0.511
M_MU = 105.7
M_TAU = 1776.8

# Test point (from best fit)
BETA = 3.15
W = 0.020
ETA_TARGET = 0.03
R_C_REF = 0.88

print("=" * 70)
print("TAU COLLAPSE DIAGNOSTICS")
print("=" * 70)
print()

# ========================================================================
# DIAGNOSTIC 1: Per-Lepton Scales
# ========================================================================

print("=" * 70)
print("DIAGNOSTIC 1: Per-Lepton Scales S_ℓ = m_ℓ/E_ℓ")
print("=" * 70)
print()

# Run fit at best point
lam = calibrate_lambda(ETA_TARGET, BETA, R_C_REF)
fitter = LeptonFitter(beta=BETA, w=W, lam=lam, sigma_model=1e-4)
result = fitter.fit(max_iter=200, seed=42)

# Extract energies and parameters
E_e = result["energies"]["electron"]["E_total"]
E_mu = result["energies"]["muon"]["E_total"]
E_tau = result["energies"]["tau"]["E_total"]

params_e = result["parameters"]["electron"]
params_mu = result["parameters"]["muon"]
params_tau = result["parameters"]["tau"]

# Compute per-lepton scales
S_e = M_E / E_e if E_e > 0 else 0
S_mu = M_MU / E_mu if E_mu > 0 else 0
S_tau = M_TAU / E_tau if E_tau > 0 else 0

print(f"Best fit at β={BETA}, w={W}, λ={lam:.6f}")
print()
print("Lepton   Energy      Mass (target)   S_ℓ = m/E      Ratio to S_μ")
print("------   ------      -------------   ----------     -------------")
print(f"e        {E_e:8.4f}   {M_E:8.2f} MeV     {S_e:10.4f}     {S_e/S_mu if S_mu > 0 else 0:.4f}")
print(f"μ        {E_mu:8.4f}   {M_MU:8.2f} MeV     {S_mu:10.4f}     1.0000")
print(f"τ        {E_tau:8.4f}   {M_TAU:8.2f} MeV     {S_tau:10.4f}     {S_tau/S_mu if S_mu > 0 else 0:.4f}")
print()

if abs(S_tau/S_mu - 2.0) < 0.2:
    print("⚠ REGIME CHANGE: S_τ ≈ 2×S_μ")
    print("  → Tau requires ~2× scale factor of muon")
    print("  → Missing physics for heavy leptons")
elif abs(S_e/S_mu - 1.0) > 0.2 or abs(S_tau/S_mu - 1.0) > 0.2:
    print("⚠ NON-UNIVERSAL SCALING")
    print("  → Different leptons need different m/E factors")
else:
    print("✓ Universal scaling (S_e ≈ S_μ ≈ S_τ)")

print()

# ========================================================================
# DIAGNOSTIC 2: Parameter Bound Saturation
# ========================================================================

print("=" * 70)
print("DIAGNOSTIC 2: Parameter Bound Saturation")
print("=" * 70)
print()

# Bounds from profile_likelihood_boundary_layer.py
bounds = {
    "electron": {"R_c": (0.5, 1.5), "U": (0.01, 0.1), "A": (0.7, 1.0)},
    "muon": {"R_c": (0.05, 0.3), "U": (0.05, 0.2), "A": (0.7, 1.0)},
    "tau": {"R_c": (0.3, 0.8), "U": (0.02, 0.15), "A": (0.7, 1.0)},
}

print("Checking if parameters hit bounds (within 1%):\n")

for lepton, params in [("electron", params_e), ("muon", params_mu), ("tau", params_tau)]:
    print(f"{lepton.upper()}:")
    bound_hit = False
    for param_name in ["R_c", "U", "A"]:
        value = params[param_name]
        low, high = bounds[lepton][param_name]

        # Check if within 1% of boundary
        dist_to_low = (value - low) / (high - low)
        dist_to_high = (high - value) / (high - low)

        status = "  "
        if dist_to_low < 0.01:
            status = "⚠ AT LOWER BOUND"
            bound_hit = True
        elif dist_to_high < 0.01:
            status = "⚠ AT UPPER BOUND"
            bound_hit = True

        print(f"  {param_name:4s} = {value:.4f}  [{low:.2f}, {high:.2f}]  {status}")

    if not bound_hit:
        print("  ✓ No bounds hit")
    print()

# ========================================================================
# DIAGNOSTIC 3: Closure Isolation
# ========================================================================

print("=" * 70)
print("DIAGNOSTIC 3: Closure Isolation")
print("=" * 70)
print()
print("Testing which ingredient causes τ collapse:\n")

# Four closure configurations
configs = [
    ("(a) Baseline",       0.0,        0.0),     # λ=0, w=0
    ("(b) Gradient only",  ETA_TARGET, 0.0),     # λ>0, w=0
    ("(c) Boundary only",  0.0,        W),       # λ=0, w>0
    ("(d) Both (current)", ETA_TARGET, W),       # λ>0, w>0
]

results_closure = []

for name, eta, w_test in configs:
    lam_test = calibrate_lambda(eta, BETA, R_C_REF) if eta > 0 else 0.0

    # Note: w=0 case needs special handling (no boundary layer)
    # For now, use small w (0.001) as proxy for w=0
    w_test = max(w_test, 0.001)

    fitter_test = LeptonFitter(beta=BETA, w=w_test, lam=lam_test, sigma_model=1e-4)
    result_test = fitter_test.fit(max_iter=100, seed=42)

    E_e_test = result_test["energies"]["electron"]["E_total"]
    E_mu_test = result_test["energies"]["muon"]["E_total"]
    E_tau_test = result_test["energies"]["tau"]["E_total"]

    S_e_test = M_E / E_e_test if E_e_test > 0 else 0
    S_mu_test = M_MU / E_mu_test if E_mu_test > 0 else 0
    S_tau_test = M_TAU / E_tau_test if E_tau_test > 0 else 0

    tau_ratio = S_tau_test / S_mu_test if S_mu_test > 0 else 0

    print(f"{name:20s}  η={eta:.2f}, w={w_test:.3f}")
    print(f"  S_e={S_e_test:7.2f}, S_μ={S_mu_test:7.2f}, S_τ={S_tau_test:7.2f}")
    print(f"  S_τ/S_μ = {tau_ratio:.3f}", end="")

    if tau_ratio > 1.5:
        print("  ⚠ τ collapse")
    else:
        print("  ✓ τ OK")

    print()

    results_closure.append((name, tau_ratio))

print()
print("Summary:")
print("-" * 70)

baseline_ratio = results_closure[0][1]  # (a) baseline

for name, ratio in results_closure:
    delta = ratio - baseline_ratio
    print(f"{name:20s}  S_τ/S_μ = {ratio:.3f}  (Δ from baseline: {delta:+.3f})")

print()

# Interpretation
if results_closure[0][1] > 1.5:  # baseline bad
    print("⚠ τ collapses even in BASELINE (λ=0, w=0)")
    print("  → Issue is in circulation/stabilization, not added ingredients")
elif results_closure[1][1] > 1.5 or results_closure[3][1] > 1.5:  # gradient causes it
    print("⚠ τ collapses when GRADIENT is added")
    print("  → Gradient energy scaling wrong for high-mass leptons")
elif results_closure[2][1] > 1.5 or results_closure[3][1] > 1.5:  # boundary causes it
    print("⚠ τ collapses when BOUNDARY LAYER is added")
    print("  → Taper/boundary logic wrong for large leptons")
else:
    print("? τ collapse pattern unclear - need more investigation")

print()
print("=" * 70)
print("DIAGNOSTICS COMPLETE")
print("=" * 70)
