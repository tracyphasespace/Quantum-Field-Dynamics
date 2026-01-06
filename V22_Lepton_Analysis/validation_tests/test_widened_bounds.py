#!/usr/bin/env python3
"""
Test 1: Remove Bound Artifacts

Hypothesis: The 9× scaling is being manufactured by parameter bounds.

Current (saturating):
  μ: R_c at upper bound (0.30), U at lower bound (0.05)
  τ: U at upper bound (0.15)

This forces U_τ/U_μ ≤ 3, hence E_τ/E_μ ≤ 9 if E_circ ∝ U².

Test: Widen bounds and see if optimizer naturally moves toward
      U_τ/U_μ ≈ √16.8 ≈ 4.1, which would give E_τ/E_μ ≈ 17.
"""

import numpy as np
from profile_likelihood_boundary_layer import LeptonFitter, calibrate_lambda
import json

# Physical constants
M_E = 0.511
M_MU = 105.7
M_TAU = 1776.8

# Test parameters
BETA = 3.15
W = 0.020
ETA_TARGET = 0.03
R_C_REF = 0.88

print("=" * 70)
print("TEST 1: WIDENED BOUNDS")
print("=" * 70)
print()

# Define bound sets
bounds_original = {
    "electron": {"R_c": (0.5, 1.5), "U": (0.01, 0.1), "A": (0.7, 1.0)},
    "muon": {"R_c": (0.05, 0.30), "U": (0.05, 0.20), "A": (0.7, 1.0)},
    "tau": {"R_c": (0.30, 0.80), "U": (0.02, 0.15), "A": (0.7, 1.0)},
}

bounds_widened = {
    "electron": {"R_c": (0.5, 1.5), "U": (0.01, 0.1), "A": (0.7, 1.0)},  # Unchanged
    "muon": {"R_c": (0.05, 0.50), "U": (0.02, 0.20), "A": (0.7, 1.0)},  # R_c_max: 0.30 → 0.50, U_min: 0.05 → 0.02
    "tau": {"R_c": (0.30, 0.80), "U": (0.02, 0.25), "A": (0.7, 1.0)},   # U_max: 0.15 → 0.25
}

print("Bound comparison:")
print("-" * 70)
print(f"{'Lepton':<10} {'Parameter':<10} {'Original':<20} {'Widened':<20} {'Change':<20}")
print("-" * 70)

for lepton in ["electron", "muon", "tau"]:
    for param in ["R_c", "U", "A"]:
        orig = bounds_original[lepton][param]
        wide = bounds_widened[lepton][param]

        if orig != wide:
            change = f"✓ Modified"
        else:
            change = "  (unchanged)"

        print(f"{lepton:<10} {param:<10} {str(orig):<20} {str(wide):<20} {change}")

print()

# ========================================================================
# Test with original bounds
# ========================================================================

print("=" * 70)
print("RUN 1: ORIGINAL BOUNDS (baseline)")
print("=" * 70)
print()

# Temporarily patch LeptonFitter bounds
import profile_likelihood_boundary_layer as plb

# Save original bounds
orig_bounds_in_code = plb.LeptonFitter.__init__.__code__.co_consts

# Run fit with original bounds (already default in code)
lam = calibrate_lambda(ETA_TARGET, BETA, R_C_REF)
fitter_orig = LeptonFitter(beta=BETA, w=W, lam=lam, sigma_model=1e-4)
result_orig = fitter_orig.fit(max_iter=200, seed=42)

print("Results with ORIGINAL bounds:")
print("-" * 70)

params_orig = result_orig["parameters"]
energies_orig = result_orig["energies"]
chi2_orig = result_orig["chi2"]
S_opt_orig = result_orig["S_opt"]

print(f"χ² = {chi2_orig:.6e}")
print(f"S_opt = {S_opt_orig:.6f}")
print()

print(f"{'Lepton':<10} {'R_c':<10} {'U':<10} {'A':<10} {'E_circ':<12} {'E_total':<12}")
print("-" * 70)

for lepton in ["electron", "muon", "tau"]:
    p = params_orig[lepton]
    e = energies_orig[lepton]
    print(f"{lepton:<10} {p['R_c']:<10.4f} {p['U']:<10.4f} {p['A']:<10.4f} {e['E_circ']:<12.6f} {e['E_total']:<12.6f}")

print()

# Compute key ratios
U_mu_orig = params_orig["muon"]["U"]
U_tau_orig = params_orig["tau"]["U"]
E_circ_mu_orig = energies_orig["muon"]["E_circ"]
E_circ_tau_orig = energies_orig["tau"]["E_circ"]
E_mu_orig = energies_orig["muon"]["E_total"]
E_tau_orig = energies_orig["tau"]["E_total"]

U_ratio_orig = U_tau_orig / U_mu_orig if U_mu_orig > 0 else 0
E_circ_ratio_orig = E_circ_tau_orig / E_circ_mu_orig if E_circ_mu_orig > 0 else 0
E_ratio_orig = E_tau_orig / E_mu_orig if E_mu_orig > 0 else 0

S_mu_orig = M_MU / E_mu_orig if E_mu_orig > 0 else 0
S_tau_orig = M_TAU / E_tau_orig if E_tau_orig > 0 else 0
S_ratio_orig = S_tau_orig / S_mu_orig if S_mu_orig > 0 else 0

print("Key ratios (τ/μ):")
print(f"  U_τ/U_μ      = {U_ratio_orig:.4f}  (target: ~4.1 for E_τ/E_μ ≈ 17)")
print(f"  E_circ,τ/E_μ = {E_circ_ratio_orig:.4f}")
print(f"  E_τ/E_μ      = {E_ratio_orig:.4f}  (vs m_τ/m_μ = 16.81)")
print(f"  S_τ/S_μ      = {S_ratio_orig:.4f}  (target: ~1.0)")
print()

# Check bound saturation (original)
print("Bound saturation check (ORIGINAL):")
print("-" * 70)

for lepton in ["muon", "tau"]:
    p = params_orig[lepton]
    bounds = bounds_original[lepton]

    print(f"{lepton.upper()}:")
    for param_name in ["R_c", "U", "A"]:
        value = p[param_name]
        low, high = bounds[param_name]

        dist_to_low = (value - low) / (high - low)
        dist_to_high = (high - value) / (high - low)

        status = "  "
        if dist_to_low < 0.01:
            status = "⚠ AT LOWER BOUND"
        elif dist_to_high < 0.01:
            status = "⚠ AT UPPER BOUND"

        print(f"  {param_name:4s} = {value:.4f}  [{low:.2f}, {high:.2f}]  {status}")
    print()

# ========================================================================
# Test with widened bounds
# ========================================================================

print("=" * 70)
print("RUN 2: WIDENED BOUNDS")
print("=" * 70)
print()

# Modify bounds in LeptonFitter class (monkey patch for this test)
class LeptonFitterWideBounds(LeptonFitter):
    """LeptonFitter with widened bounds"""

    def __init__(self, beta, w, lam, sigma_model=1e-4):
        # Call parent init
        super().__init__(beta, w, lam, sigma_model)

        # Override bounds
        self.bounds_electron = [
            (bounds_widened["electron"]["R_c"][0], bounds_widened["electron"]["R_c"][1]),
            (bounds_widened["electron"]["U"][0], bounds_widened["electron"]["U"][1]),
            (bounds_widened["electron"]["A"][0], bounds_widened["electron"]["A"][1]),
        ]

        self.bounds_muon = [
            (bounds_widened["muon"]["R_c"][0], bounds_widened["muon"]["R_c"][1]),
            (bounds_widened["muon"]["U"][0], bounds_widened["muon"]["U"][1]),
            (bounds_widened["muon"]["A"][0], bounds_widened["muon"]["A"][1]),
        ]

        self.bounds_tau = [
            (bounds_widened["tau"]["R_c"][0], bounds_widened["tau"]["R_c"][1]),
            (bounds_widened["tau"]["U"][0], bounds_widened["tau"]["U"][1]),
            (bounds_widened["tau"]["A"][0], bounds_widened["tau"]["A"][1]),
        ]

# Run fit with widened bounds
fitter_wide = LeptonFitterWideBounds(beta=BETA, w=W, lam=lam, sigma_model=1e-4)
result_wide = fitter_wide.fit(max_iter=200, seed=42)

print("Results with WIDENED bounds:")
print("-" * 70)

params_wide = result_wide["parameters"]
energies_wide = result_wide["energies"]
chi2_wide = result_wide["chi2"]
S_opt_wide = result_wide["S_opt"]

print(f"χ² = {chi2_wide:.6e}")
print(f"S_opt = {S_opt_wide:.6f}")
print()

print(f"{'Lepton':<10} {'R_c':<10} {'U':<10} {'A':<10} {'E_circ':<12} {'E_total':<12}")
print("-" * 70)

for lepton in ["electron", "muon", "tau"]:
    p = params_wide[lepton]
    e = energies_wide[lepton]
    print(f"{lepton:<10} {p['R_c']:<10.4f} {p['U']:<10.4f} {p['A']:<10.4f} {e['E_circ']:<12.6f} {e['E_total']:<12.6f}")

print()

# Compute key ratios (widened)
U_mu_wide = params_wide["muon"]["U"]
U_tau_wide = params_wide["tau"]["U"]
E_circ_mu_wide = energies_wide["muon"]["E_circ"]
E_circ_tau_wide = energies_wide["tau"]["E_circ"]
E_mu_wide = energies_wide["muon"]["E_total"]
E_tau_wide = energies_wide["tau"]["E_total"]

U_ratio_wide = U_tau_wide / U_mu_wide if U_mu_wide > 0 else 0
E_circ_ratio_wide = E_circ_tau_wide / E_circ_mu_wide if E_circ_mu_wide > 0 else 0
E_ratio_wide = E_tau_wide / E_mu_wide if E_mu_wide > 0 else 0

S_mu_wide = M_MU / E_mu_wide if E_mu_wide > 0 else 0
S_tau_wide = M_TAU / E_tau_wide if E_tau_wide > 0 else 0
S_ratio_wide = S_tau_wide / S_mu_wide if S_mu_wide > 0 else 0

print("Key ratios (τ/μ):")
print(f"  U_τ/U_μ      = {U_ratio_wide:.4f}  (target: ~4.1 for E_τ/E_μ ≈ 17)")
print(f"  E_circ,τ/E_μ = {E_circ_ratio_wide:.4f}")
print(f"  E_τ/E_μ      = {E_ratio_wide:.4f}  (vs m_τ/m_μ = 16.81)")
print(f"  S_τ/S_μ      = {S_ratio_wide:.4f}  (target: ~1.0)")
print()

# Check bound saturation (widened)
print("Bound saturation check (WIDENED):")
print("-" * 70)

saturation_wide = False
for lepton in ["muon", "tau"]:
    p = params_wide[lepton]
    bounds = bounds_widened[lepton]

    print(f"{lepton.upper()}:")
    for param_name in ["R_c", "U", "A"]:
        value = p[param_name]
        low, high = bounds[param_name]

        dist_to_low = (value - low) / (high - low)
        dist_to_high = (high - value) / (high - low)

        status = "  "
        if dist_to_low < 0.01:
            status = "⚠ AT LOWER BOUND"
            saturation_wide = True
        elif dist_to_high < 0.01:
            status = "⚠ AT UPPER BOUND"
            saturation_wide = True

        print(f"  {param_name:4s} = {value:.4f}  [{low:.2f}, {high:.2f}]  {status}")
    print()

# ========================================================================
# Comparison and Decision
# ========================================================================

print("=" * 70)
print("COMPARISON: ORIGINAL vs WIDENED")
print("=" * 70)
print()

print(f"{'Metric':<20} {'Original':<15} {'Widened':<15} {'Change':<15}")
print("-" * 70)
print(f"{'χ²':<20} {chi2_orig:<15.2e} {chi2_wide:<15.2e} {chi2_wide/chi2_orig:<15.2e}")
print(f"{'S_opt':<20} {S_opt_orig:<15.4f} {S_opt_wide:<15.4f} {S_opt_wide-S_opt_orig:<+15.4f}")
print(f"{'U_τ/U_μ':<20} {U_ratio_orig:<15.4f} {U_ratio_wide:<15.4f} {U_ratio_wide-U_ratio_orig:<+15.4f}")
print(f"{'E_τ/E_μ':<20} {E_ratio_orig:<15.4f} {E_ratio_wide:<15.4f} {E_ratio_wide-E_ratio_orig:<+15.4f}")
print(f"{'S_τ/S_μ':<20} {S_ratio_orig:<15.4f} {S_ratio_wide:<15.4f} {S_ratio_wide-S_ratio_orig:<+15.4f}")
print()

# Decision criteria
print("=" * 70)
print("DECISION")
print("=" * 70)
print()

# Criterion 1: Did χ² improve dramatically?
chi2_improved = chi2_wide < 1000

# Criterion 2: Did S_τ/S_μ move toward 1.0?
S_ratio_improved = S_ratio_wide < 1.5

# Criterion 3: Did U_τ/U_μ increase toward 4.1?
U_ratio_increased = U_ratio_wide > 3.5

# Criterion 4: Are bounds still saturated?
bounds_still_saturated = saturation_wide

print(f"χ² < 1000:         {chi2_improved} ({'PASS' if chi2_improved else 'FAIL'})")
print(f"S_τ/S_μ < 1.5:     {S_ratio_improved} ({'PASS' if S_ratio_improved else 'FAIL'})")
print(f"U_τ/U_μ > 3.5:     {U_ratio_increased} ({'PASS' if U_ratio_increased else 'FAIL'})")
print(f"No saturation:     {not bounds_still_saturated} ({'PASS' if not bounds_still_saturated else 'FAIL'})")
print()

if chi2_improved and S_ratio_improved and U_ratio_increased and not bounds_still_saturated:
    print("✓ OUTCOME 1A: BOUND ARTIFACT CONFIRMED")
    print()
    print("Widening bounds resolved the τ collapse!")
    print("  → Universal scaling (S_τ/S_μ ≈ 1) restored")
    print("  → χ² is reasonable")
    print("  → No new physics needed")
    print()
    print("Next steps:")
    print("  - Document identifiability constraints")
    print("  - Publish with widened bounds")
    print("  - Revise parameter-box recommendations")
    outcome = "1A_bound_artifact"
elif not chi2_improved and not S_ratio_improved:
    print("✗ OUTCOME 1B: GENUINE REGIME CHANGE")
    print()
    print("Widening bounds did NOT resolve the τ collapse.")
    print("  → S_τ/S_μ still ≈ 1.86")
    print("  → χ² still huge")
    print("  → τ genuinely needs different physics")
    print()
    print("Next steps:")
    print("  - Proceed to Test 2 (circulation prefactor)")
    print("  - If Test 2 passes, proceed to Test 3 (two-lepton fit)")
    outcome = "1B_genuine_regime_change"
elif bounds_still_saturated:
    print("⚠ INCONCLUSIVE: STILL HITTING BOUNDS")
    print()
    print("Optimizer still saturating even with widened bounds.")
    print("  → Need to widen further")
    print("  → Or: there's a hard constraint preventing exploration")
    print()
    print("Next steps:")
    print("  - Try even wider bounds (e.g., U_max → 0.35)")
    print("  - Check for numerical instabilities at high U")
    outcome = "inconclusive_saturation"
else:
    print("~ PARTIAL IMPROVEMENT")
    print()
    print("Some metrics improved but not all criteria met.")
    print("  → May need further bound widening")
    print("  → Or: some genuine physics + some artifact")
    print()
    print("Next steps:")
    print("  - Review which criteria failed")
    print("  - Iterate with different bound choices")
    outcome = "partial_improvement"

print()
print("=" * 70)

# Save results
results_dict = {
    "original": {
        "chi2": chi2_orig,
        "S_opt": S_opt_orig,
        "U_tau_over_U_mu": U_ratio_orig,
        "E_tau_over_E_mu": E_ratio_orig,
        "S_tau_over_S_mu": S_ratio_orig,
        "parameters": {lep: {k: float(v) for k, v in params_orig[lep].items()} for lep in ["electron", "muon", "tau"]},
    },
    "widened": {
        "chi2": chi2_wide,
        "S_opt": S_opt_wide,
        "U_tau_over_U_mu": U_ratio_wide,
        "E_tau_over_E_mu": E_ratio_wide,
        "S_tau_over_S_mu": S_ratio_wide,
        "parameters": {lep: {k: float(v) for k, v in params_wide[lep].items()} for lep in ["electron", "muon", "tau"]},
    },
    "outcome": outcome,
}

with open("widened_bounds_results.json", "w") as f:
    json.dump(results_dict, f, indent=2)

print("Results saved to: widened_bounds_results.json")
print()
