#!/usr/bin/env python3
"""
T1a: Monte Carlo Feasibility Scan

Fast test: Can the model produce E_μ/E_e ~ 207 under ANY parameters?

Samples N random points in widened bounds and checks whether the
energy ratio distribution ever approaches the required mass ratio.

If E_μ/E_e never gets close to m_μ/m_e ≈ 207, the model cannot fit
(e,μ) regardless of optimizer sophistication.
"""

import numpy as np
from lepton_energy_localized_v1 import LeptonEnergyLocalizedV1
from profile_likelihood_boundary_layer import calibrate_lambda
import sys
from tqdm import tqdm
import json

# Physical constants
M_E = 0.511
M_MU = 105.7
TARGET_RATIO = M_MU / M_E  # ≈ 206.85

# Fixed configuration
beta = 3.15
w = 0.020
eta_target = 0.03
R_c_ref = 0.88
lam = calibrate_lambda(eta_target, beta, R_c_ref)

# Localization (baseline best config)
k = 1.5
delta_v = 0.5
p = 6

# Tier A widened bounds
BOUNDS = {
    "electron": {
        "R_c": (0.15, 3.00),
        "U": (0.005, 0.40),
        "A": (0.05, 0.999),
    },
    "muon": {
        "R_c": (0.02, 1.20),
        "U": (0.01, 0.60),
        "A": (0.05, 0.999),
    },
}

N_SAMPLES = 20000

print("=" * 80)
print("T1a: MONTE CARLO FEASIBILITY SCAN")
print("=" * 80)
print()
print(f"Objective: Test if E_μ/E_e can approach m_μ/m_e ≈ {TARGET_RATIO:.2f}")
print(f"Method: {N_SAMPLES:,} random samples in widened Tier-A bounds")
print()
print("Tier-A bounds:")
for lepton in ["electron", "muon"]:
    print(f"  {lepton}:")
    for param, (lo, hi) in BOUNDS[lepton].items():
        print(f"    {param}: [{lo:.4f}, {hi:.4f}]")
print()
sys.stdout.flush()

# Create energy calculator
energy_calc = LeptonEnergyLocalizedV1(
    beta=beta,
    w=w,
    lam=lam,
    k_localization=k,
    delta_v_factor=delta_v,
    p_envelope=p,
)

# Sample parameters
print("Sampling parameter space...")
sys.stdout.flush()

valid_samples = []
energy_ratios = []
chi2_values = []

for i in tqdm(range(N_SAMPLES), desc="MC sampling", unit="sample"):
    # Random parameters
    R_c_e = np.random.uniform(*BOUNDS["electron"]["R_c"])
    U_e = np.random.uniform(*BOUNDS["electron"]["U"])
    A_e = np.random.uniform(*BOUNDS["electron"]["A"])

    R_c_mu = np.random.uniform(*BOUNDS["muon"]["R_c"])
    U_mu = np.random.uniform(*BOUNDS["muon"]["U"])
    A_mu = np.random.uniform(*BOUNDS["muon"]["A"])

    # Compute energies
    try:
        E_e, _, _, _ = energy_calc.total_energy(R_c_e, U_e, A_e)
        E_mu, _, _, _ = energy_calc.total_energy(R_c_mu, U_mu, A_mu)

        # Check validity
        if E_e > 0 and E_mu > 0 and np.isfinite(E_e) and np.isfinite(E_mu):
            ratio = E_mu / E_e

            # Compute implied χ² with analytic S profiling
            energies = np.array([E_e, E_mu])
            targets = np.array([M_E, M_MU])
            sigma_model = 1e-4
            sigma_abs = sigma_model * targets
            weights = 1.0 / sigma_abs**2

            numerator = np.sum(targets * energies * weights)
            denominator = np.sum(energies**2 * weights)

            if denominator > 0:
                S_opt = numerator / denominator
                masses_model = S_opt * energies
                residuals = (masses_model - targets) / sigma_abs
                chi2 = np.sum(residuals**2)

                valid_samples.append({
                    "R_c_e": R_c_e, "U_e": U_e, "A_e": A_e,
                    "R_c_mu": R_c_mu, "U_mu": U_mu, "A_mu": A_mu,
                    "E_e": E_e, "E_mu": E_mu,
                    "ratio": ratio, "S_opt": S_opt, "chi2": chi2,
                })

                energy_ratios.append(ratio)
                chi2_values.append(chi2)

    except Exception:
        # Skip invalid samples
        continue

print()
print("=" * 80)
print("RESULTS")
print("=" * 80)
print()

n_valid = len(valid_samples)
print(f"Valid samples: {n_valid:,} / {N_SAMPLES:,} ({100*n_valid/N_SAMPLES:.1f}%)")
print()

if n_valid == 0:
    print("✗ FATAL: No valid energy configurations found in parameter space")
    print("   Model cannot produce positive energies in widened bounds.")
    sys.exit(1)

energy_ratios = np.array(energy_ratios)
chi2_values = np.array(chi2_values)

print(f"Target ratio (m_μ/m_e):  {TARGET_RATIO:.2f}")
print()

print("Energy ratio E_μ/E_e distribution:")
print(f"  Min:        {np.min(energy_ratios):.2f}")
print(f"  25th %ile:  {np.percentile(energy_ratios, 25):.2f}")
print(f"  Median:     {np.median(energy_ratios):.2f}")
print(f"  75th %ile:  {np.percentile(energy_ratios, 75):.2f}")
print(f"  Max:        {np.max(energy_ratios):.2f}")
print(f"  Mean:       {np.mean(energy_ratios):.2f}")
print(f"  Std:        {np.std(energy_ratios):.2f}")
print()

# Check if any samples approach target
threshold_10pct = TARGET_RATIO * 0.9  # Within 10% of target
threshold_50pct = TARGET_RATIO * 0.5  # Within 50% of target

n_within_10pct = np.sum(energy_ratios >= threshold_10pct)
n_within_50pct = np.sum(energy_ratios >= threshold_50pct)

print(f"Samples with E_μ/E_e ≥ 90% of target ({threshold_10pct:.1f}): {n_within_10pct}")
print(f"Samples with E_μ/E_e ≥ 50% of target ({threshold_50pct:.1f}): {n_within_50pct}")
print()

# Best sample
best_idx = np.argmin(chi2_values)
best = valid_samples[best_idx]

print("Best sample (lowest χ²):")
print(f"  χ²:       {best['chi2']:.6e}")
print(f"  S_opt:    {best['S_opt']:.4f}")
print(f"  E_μ/E_e:  {best['ratio']:.2f}")
print(f"  Electron: R_c={best['R_c_e']:.4f}, U={best['U_e']:.4f}, A={best['A_e']:.4f}")
print(f"  Muon:     R_c={best['R_c_mu']:.4f}, U={best['U_mu']:.4f}, A={best['A_mu']:.4f}")
print()

# Feasibility verdict
print("=" * 80)
print("FEASIBILITY VERDICT")
print("=" * 80)
print()

max_ratio = np.max(energy_ratios)
ratio_shortfall = (TARGET_RATIO - max_ratio) / TARGET_RATIO * 100

if max_ratio >= TARGET_RATIO * 0.9:
    print("✓ FEASIBLE: Model can produce E_μ/E_e within 10% of required ratio")
    print(f"  Maximum achieved: {max_ratio:.2f} / {TARGET_RATIO:.2f}")
    print()
    print("  Proceed to T1 (widened bounds optimization)")
elif max_ratio >= TARGET_RATIO * 0.5:
    print("~ MARGINAL: Model can reach 50-90% of required ratio")
    print(f"  Maximum achieved: {max_ratio:.2f} / {TARGET_RATIO:.2f}")
    print(f"  Shortfall: {ratio_shortfall:.1f}%")
    print()
    print("  Optimizer may still improve, but gap is large.")
    print("  Proceed to T1 cautiously; consider T4 pivots if T1 fails.")
else:
    print("✗ INFEASIBLE: Model cannot approach required E_μ/E_e ratio")
    print(f"  Maximum achieved: {max_ratio:.2f} / {TARGET_RATIO:.2f}")
    print(f"  Shortfall: {ratio_shortfall:.1f}%")
    print()
    print("  Even in widened bounds, energy ratio is far from target.")
    print("  No optimizer will fix this - functional form is inadequate for (e,μ).")
    print()
    print("  RECOMMENDATION: Skip T1 optimization, proceed directly to T4 (physics pivot).")

print()

# Save results
results = {
    "n_samples": N_SAMPLES,
    "n_valid": n_valid,
    "bounds": BOUNDS,
    "target_ratio": TARGET_RATIO,
    "ratio_stats": {
        "min": float(np.min(energy_ratios)),
        "max": float(np.max(energy_ratios)),
        "mean": float(np.mean(energy_ratios)),
        "median": float(np.median(energy_ratios)),
        "std": float(np.std(energy_ratios)),
    },
    "best_sample": {
        "chi2": float(best['chi2']),
        "S_opt": float(best['S_opt']),
        "ratio": float(best['ratio']),
        "params": {
            "electron": {"R_c": float(best['R_c_e']), "U": float(best['U_e']), "A": float(best['A_e'])},
            "muon": {"R_c": float(best['R_c_mu']), "U": float(best['U_mu']), "A": float(best['A_mu'])},
        },
    },
    "feasibility": "feasible" if max_ratio >= TARGET_RATIO * 0.9 else ("marginal" if max_ratio >= TARGET_RATIO * 0.5 else "infeasible"),
}

with open("results/V22/t1a_mc_feasibility_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("Results saved: results/V22/t1a_mc_feasibility_results.json")
print("=" * 80)
