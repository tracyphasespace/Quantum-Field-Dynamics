#!/usr/bin/env python3
"""
U_star Positivity Scan

For each (k, Δv/Rv) configuration that passes sensitivity,
find the minimal U that makes E_total > 0 for electron and muon.

E_circ ∝ U² (approximately), so:
  E_total(U) = E_circ(U₀) × (U/U₀)² - E_stab + E_grad

Solve E_total(U_star) = 0:
  U_star = U₀ × sqrt((E_stab - E_grad) / E_circ(U₀))
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

# Representative parameters (moderate core, moderate deficit)
R_c_e = 0.60
A_e = 0.75

R_c_mu = 0.18
A_mu = 0.75

# Reference U values for scaling
U0_e = 0.095
U0_mu = 0.18

print("=" * 70)
print("U_STAR POSITIVITY SCAN")
print("=" * 70)
print()

print("Representative parameters:")
print(f"  Electron: R_c={R_c_e}, A={A_e}, U₀={U0_e}")
print(f"  Muon:     R_c={R_c_mu}, A={A_mu}, U₀={U0_mu}")
print()

# Test configurations that passed sensitivity (k=1.0, 1.5 with Δv/Rv=0.25, 0.5)
configs = [
    {"k": 1.0, "delta_v_factor": 0.25},
    {"k": 1.0, "delta_v_factor": 0.5},
    {"k": 1.5, "delta_v_factor": 0.25},
    {"k": 1.5, "delta_v_factor": 0.5},
]

p = 6

print("=" * 90)
print(f"{'k':<6} {'Δv/Rv':<8} {'Lepton':<10} {'E_circ(U₀)':<12} {'E_stab':<12} {'E_grad':<12} {'U_star':<10} {'Feasible?':<10}")
print("=" * 90)

results = []

for config in configs:
    k = config["k"]
    delta_v_factor = config["delta_v_factor"]

    # Create energy calculator
    energy_calc = LeptonEnergyLocalizedV1(
        beta=beta,
        w=w,
        lam=lam,
        k_localization=k,
        delta_v_factor=delta_v_factor,
        p_envelope=p,
    )

    # Compute energies at U₀ for electron
    E_e_total, E_circ_e, E_stab_e, E_grad_e = energy_calc.total_energy(R_c_e, U0_e, A_e)

    # Compute energies at U₀ for muon
    E_mu_total, E_circ_mu, E_stab_mu, E_grad_mu = energy_calc.total_energy(R_c_mu, U0_mu, A_mu)

    # Calculate U_star for electron
    # E_total(U) = E_circ(U₀) × (U/U₀)² - E_stab + E_grad = 0
    # U_star² = U₀² × (E_stab - E_grad) / E_circ(U₀)

    numerator_e = E_stab_e - E_grad_e
    if E_circ_e > 0 and numerator_e > 0:
        U_star_e = U0_e * np.sqrt(numerator_e / E_circ_e)
    elif numerator_e <= 0:
        # Already positive at U₀
        U_star_e = U0_e
    else:
        # E_circ ≤ 0, something very wrong
        U_star_e = np.nan

    # Calculate U_star for muon
    numerator_mu = E_stab_mu - E_grad_mu
    if E_circ_mu > 0 and numerator_mu > 0:
        U_star_mu = U0_mu * np.sqrt(numerator_mu / E_circ_mu)
    elif numerator_mu <= 0:
        U_star_mu = U0_mu
    else:
        U_star_mu = np.nan

    # Check feasibility (bounds from typical fit ranges)
    U_bound_e_max = 0.10  # Typical upper bound for electron
    U_bound_mu_max = 0.20  # Typical upper bound for muon

    feasible_e = U_star_e <= U_bound_e_max if not np.isnan(U_star_e) else False
    feasible_mu = U_star_mu <= U_bound_mu_max if not np.isnan(U_star_mu) else False

    # Display
    print(f"{k:<6.1f} {delta_v_factor:<8.2f} {'electron':<10} {E_circ_e:<12.6f} {E_stab_e:<12.6f} {E_grad_e:<12.6f} {U_star_e:<10.4f} {'✓' if feasible_e else '✗':<10}")
    print(f"{'':<6} {'':<8} {'muon':<10} {E_circ_mu:<12.6f} {E_stab_mu:<12.6f} {E_grad_mu:<12.6f} {U_star_mu:<10.4f} {'✓' if feasible_mu else '✗':<10}")
    print("-" * 90)

    results.append({
        "k": k,
        "delta_v_factor": delta_v_factor,
        "U_star_e": U_star_e,
        "U_star_mu": U_star_mu,
        "feasible_e": feasible_e,
        "feasible_mu": feasible_mu,
        "E_circ_e": E_circ_e,
        "E_circ_mu": E_circ_mu,
    })

print("=" * 90)
print()

# Summary
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print()

print("U Bounds (typical):")
print(f"  Electron: U ∈ [0.01, {U_bound_e_max}]")
print(f"  Muon:     U ∈ [0.05, {U_bound_mu_max}]")
print()

# Find best configuration
feasible = [r for r in results if r["feasible_e"] and r["feasible_mu"]]

if feasible:
    # Pick smallest k (most conservative) that's feasible
    best = min(feasible, key=lambda x: (x["k"], x["delta_v_factor"]))

    print(f"✓ FEASIBLE CONFIGURATION FOUND")
    print()
    print(f"  k = {best['k']:.1f}")
    print(f"  Δv/Rv = {best['delta_v_factor']}")
    print(f"  U_star,e = {best['U_star_e']:.4f} (≤ {U_bound_e_max})")
    print(f"  U_star,μ = {best['U_star_mu']:.4f} (≤ {U_bound_mu_max})")
    print()
    print("RECOMMENDATION:")
    print(f"  • Update electron U bounds: [0.01, {max(0.10, best['U_star_e']*1.1):.3f}]")
    print(f"  • Update muon U bounds:     [0.05, {max(0.20, best['U_star_mu']*1.1):.3f}]")
    print()
    print("  • Proceed to Run 2 (e,μ regression) with these bounds")
    print(f"  • Use k = {best['k']:.1f}, Δv/Rv = {best['delta_v_factor']}, p = {p}")
    print()

    outcome = "feasible"
else:
    print("✗ NO FEASIBLE CONFIGURATION")
    print()
    print("All tested (k, Δv/Rv) require U_star beyond typical bounds.")
    print()

    # Show which configs came closest
    print("Closest configurations:")
    for r in results:
        e_status = "✓" if r["feasible_e"] else f"✗ (need {r['U_star_e']:.4f})"
        mu_status = "✓" if r["feasible_mu"] else f"✗ (need {r['U_star_mu']:.4f})"
        print(f"  k={r['k']:.1f}, Δv/Rv={r['delta_v_factor']}: e {e_status}, μ {mu_status}")
    print()

    print("PIVOT TO:")
    print("  1. Vacuum subtraction / renormalized circulation energy")
    print("  2. Redefine stabilizer to couple with circulation")
    print("  3. Explicit local excitation energy definition")

    outcome = "infeasible"

print()

# Export
import json
export_data = {
    "outcome": outcome,
    "results": [
        {
            "k": r["k"],
            "delta_v_factor": r["delta_v_factor"],
            "U_star_e": float(r["U_star_e"]),
            "U_star_mu": float(r["U_star_mu"]),
            "feasible_e": r["feasible_e"],
            "feasible_mu": r["feasible_mu"],
        }
        for r in results
    ],
}

if outcome == "feasible":
    export_data["recommended"] = {
        "k": best["k"],
        "delta_v_factor": best["delta_v_factor"],
        "p": p,
        "U_star_e": float(best["U_star_e"]),
        "U_star_mu": float(best["U_star_mu"]),
    }

with open("results/V22/u_star_positivity_scan.json", "w") as f:
    json.dump(export_data, f, indent=2)

print("Results saved: results/V22/u_star_positivity_scan.json")
print()
print("=" * 70)
