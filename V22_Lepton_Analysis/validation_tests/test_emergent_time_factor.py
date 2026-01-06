#!/usr/bin/env python3
"""
Test Emergent-Time Factor F_t

Step 1: Check if S_τ/S_μ ≈ F_t,τ/F_t,μ

Computes F_t = ⟨1/ρ(r)⟩ for each lepton configuration
and tests if this explains the scale factor discrepancy.
"""

import numpy as np
from profile_likelihood_boundary_layer import LeptonFitter, calibrate_lambda
from lepton_energy_boundary_layer import DensityBoundaryLayer

# Physical constants
M_E = 0.511
M_MU = 105.7
M_TAU = 1776.8

RHO_VAC = 1.0

def compute_time_factor(R_c, w, A, r_grid, weight_type="volume"):
    """
    Compute emergent-time factor F_t = ⟨1/ρ(r)⟩

    F_t = ∫ ρ(r)^(-1) · w(r) · r² dr / ∫ w(r) · r² dr

    Parameters
    ----------
    R_c : float
        Core radius
    w : float
        Boundary thickness
    A : float
        Amplitude
    r_grid : array
        Radial grid
    weight_type : str
        "volume" (w=1) or "energy" (w∝(Δρ)²)

    Returns
    -------
    F_t : float
        Emergent-time factor
    """
    # Create density profile
    density = DensityBoundaryLayer(R_c, w, A, rho_vac=RHO_VAC)

    # Total density
    rho = density.rho(r_grid)

    # Deficit for energy weighting
    delta_rho = density.delta_rho(r_grid)

    # Weighting function
    if weight_type == "volume":
        weight = np.ones_like(r_grid)
    elif weight_type == "energy":
        weight = delta_rho**2
    else:
        raise ValueError(f"Unknown weight_type: {weight_type}")

    # Avoid division by zero (use small epsilon where rho ≈ 0)
    rho_safe = np.maximum(rho, 1e-10)

    # Compute weighted average of 1/ρ
    integrand_num = (1.0 / rho_safe) * weight * r_grid**2
    integrand_den = weight * r_grid**2

    numerator = np.trapz(integrand_num, r_grid)
    denominator = np.trapz(integrand_den, r_grid)

    F_t = numerator / denominator if denominator > 0 else 1.0

    return F_t


def main():
    print("=" * 70)
    print("EMERGENT-TIME FACTOR TEST")
    print("=" * 70)
    print()
    print("Testing if S_τ/S_μ ≈ F_t,τ/F_t,μ")
    print()

    # Best fit parameters (from diagnostics)
    BETA = 3.15
    W = 0.020
    ETA_TARGET = 0.03
    R_C_REF = 0.88

    # Run fit
    lam = calibrate_lambda(ETA_TARGET, BETA, R_C_REF)
    fitter = LeptonFitter(beta=BETA, w=W, lam=lam, sigma_model=1e-4)
    result = fitter.fit(max_iter=200, seed=42)

    # Extract parameters and energies
    params = result["parameters"]
    energies = result["energies"]

    # Get radial grid from energy calculator (use same grid as fit)
    r_grid = fitter.energy_calc.r

    print("Best-fit parameters:")
    print("-" * 70)
    for lepton in ["electron", "muon", "tau"]:
        p = params[lepton]
        print(f"{lepton:8s}: R_c={p['R_c']:.4f}, U={p['U']:.4f}, A={p['A']:.4f}")
    print()

    # Compute F_t for each lepton (try both weighting schemes)
    results_weights = {}

    for weight_type in ["volume", "energy"]:
        print(f"Weight type: {weight_type}")
        print("-" * 70)

        F_t_values = {}
        S_values = {}

        for lepton, mass_target in [("electron", M_E), ("muon", M_MU), ("tau", M_TAU)]:
            p = params[lepton]
            E_total = energies[lepton]["E_total"]

            # Compute F_t
            F_t = compute_time_factor(p["R_c"], W, p["A"], r_grid, weight_type)

            # Compute S = m/E (observed scale factor)
            S = mass_target / E_total if E_total > 0 else 0

            F_t_values[lepton] = F_t
            S_values[lepton] = S

            print(f"  {lepton:8s}: F_t = {F_t:.6f}, S = {S:.6f}, E = {E_total:.6f}")

        print()

        # Compare ratios
        F_t_e = F_t_values["electron"]
        F_t_mu = F_t_values["muon"]
        F_t_tau = F_t_values["tau"]

        S_e = S_values["electron"]
        S_mu = S_values["muon"]
        S_tau = S_values["tau"]

        print("Ratios to muon:")
        print(f"  Electron: F_t,e/F_t,μ = {F_t_e/F_t_mu:.4f}, S_e/S_μ = {S_e/S_mu:.4f}")
        print(f"  Tau:      F_t,τ/F_t,μ = {F_t_tau/F_t_mu:.4f}, S_τ/S_μ = {S_tau/S_mu:.4f}")
        print()

        # Key test: Does F_t ratio match S ratio for tau?
        tau_F_t_ratio = F_t_tau / F_t_mu
        tau_S_ratio = S_tau / S_mu

        match_quality = abs(tau_F_t_ratio - tau_S_ratio) / tau_S_ratio

        print(f"Tau ratio match:")
        print(f"  F_t,τ/F_t,μ = {tau_F_t_ratio:.4f}")
        print(f"  S_τ/S_μ     = {tau_S_ratio:.4f}")
        print(f"  Discrepancy: {match_quality*100:.1f}%")
        print()

        if match_quality < 0.1:
            print(f"✓ EXCELLENT MATCH (<10% error)")
            print(f"  → F_t explains the scale discrepancy!")
        elif match_quality < 0.2:
            print(f"✓ GOOD MATCH (<20% error)")
            print(f"  → F_t is the dominant correction")
        elif match_quality < 0.5:
            print(f"~ PARTIAL MATCH (<50% error)")
            print(f"  → F_t helps but may need exponent p")
        else:
            print(f"✗ POOR MATCH (>{50}% error)")
            print(f"  → F_t alone insufficient")

        print()
        print("=" * 70)
        print()

        results_weights[weight_type] = {
            "F_t_ratio": tau_F_t_ratio,
            "S_ratio": tau_S_ratio,
            "match_quality": match_quality,
        }

    # Summary
    print("SUMMARY")
    print("=" * 70)
    print()
    print("Weight Type      F_t,τ/F_t,μ   S_τ/S_μ   Match Quality")
    print("-" * 70)
    for weight_type, res in results_weights.items():
        print(f"{weight_type:15s}  {res['F_t_ratio']:10.4f}  {res['S_ratio']:8.4f}   {res['match_quality']*100:5.1f}%")
    print()

    # Recommendation
    best_weight = min(results_weights.items(), key=lambda x: x[1]["match_quality"])
    print(f"Recommended weighting: {best_weight[0]}")
    print(f"  Match quality: {best_weight[1]['match_quality']*100:.1f}%")
    print()

    if best_weight[1]["match_quality"] < 0.2:
        print("✓ Ready to implement m = S · F_t · E with analytic S profiling")
    else:
        print("~ Consider adding exponent: m = S · F_t^p · E")

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
