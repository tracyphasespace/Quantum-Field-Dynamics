#!/usr/bin/env python3
"""
Profile Likelihood in β

Purpose: For each β on a dense grid, minimize χ² over all other parameters.
         This separates optimization noise from real β-identifiability.

Method:
- Fix β at each grid point
- Minimize χ² over (R_e, U_e, A_e, R_μ, U_μ, A_μ, R_τ, U_τ, A_τ, C_μ)
- Plot χ²_min(β)

If flat: model is underconstrained
If sharp minimum: multi-start spread was optimization noise
If multiple minima: real degeneracy
"""

import numpy as np
from scipy.optimize import minimize
import json
from pathlib import Path
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent))
from test_cross_lepton_fit import (
    cross_lepton_objective, LEPTON_MASSES, LEPTON_G_FACTORS,
    SIGMA_MASS_MODEL, SIGMA_G_MODEL, RHO_VAC
)
from test_all_leptons_beta_from_alpha import LeptonEnergy

def profile_likelihood_beta(beta_grid, sigma_m_model=SIGMA_MASS_MODEL,
                           sigma_g_model=SIGMA_G_MODEL,
                           num_r=100, num_theta=20, verbose=True):
    """
    Compute profile likelihood: χ²_min(β).

    For each β, minimize over all other parameters.

    Args:
        beta_grid: Array of β values to test
        sigma_m_model: Mass theory uncertainty (relative)
        sigma_g_model: g-factor theory uncertainty (absolute)
        num_r, num_theta: Grid resolution
        verbose: Print progress

    Returns:
        dict with β grid and corresponding χ²_min values
    """
    results = []

    if verbose:
        print(f"\n{'='*70}")
        print(f"PROFILE LIKELIHOOD: χ²_min(β)")
        print(f"{'='*70}")
        print(f"β grid: [{beta_grid[0]:.3f}, {beta_grid[-1]:.3f}] with {len(beta_grid)} points")
        print(f"σ_m,model = {sigma_m_model:.2e} (relative)")
        print(f"σ_g,model = {sigma_g_model:.2e} (absolute)")
        print()

    # Create energy calculators
    energy_calcs = {
        'electron': LeptonEnergy(beta=3.0, num_r=num_r, num_theta=num_theta),
        'muon': LeptonEnergy(beta=3.0, num_r=num_r, num_theta=num_theta),
        'tau': LeptonEnergy(beta=3.0, num_r=num_r, num_theta=num_theta)
    }

    # Modify objective to use custom σ values
    def objective_with_beta_fixed(params_without_beta, beta_fixed):
        """Objective function with β fixed."""
        # Insert β into parameter list
        # params = [R_e, U_e, A_e, R_μ, U_μ, A_μ, R_τ, U_τ, A_τ, C_μ]
        R_e, U_e, A_e, R_mu, U_mu, A_mu, R_tau, U_tau, A_tau, C_mu = params_without_beta

        # Physical bounds
        if beta_fixed <= 0:
            return 1e15
        for R, U, A in [(R_e, U_e, A_e), (R_mu, U_mu, A_mu), (R_tau, U_tau, A_tau)]:
            if R <= 0 or U <= 0 or A <= 0 or A > RHO_VAC:
                return 1e15

        # Update energy calculators with fixed β
        for calc in energy_calcs.values():
            calc.beta = beta_fixed

        try:
            # Compute energies
            E_e, _, _ = energy_calcs['electron'].total_energy(R_e, U_e, A_e)
            E_mu, _, _ = energy_calcs['muon'].total_energy(R_mu, U_mu, A_mu)
            E_tau, _, _ = energy_calcs['tau'].total_energy(R_tau, U_tau, A_tau)

            # Compute g-factors
            from test_cross_lepton_fit import magnetic_moment_raw
            mu_e = magnetic_moment_raw(R_e, U_e)
            mu_mu = magnetic_moment_raw(R_mu, U_mu)
            g_e_pred = C_mu * mu_e
            g_mu_pred = C_mu * mu_mu

            # Mass residuals (relative)
            mass_e_err = (E_e - LEPTON_MASSES['electron']) / LEPTON_MASSES['electron']
            mass_mu_err = (E_mu - LEPTON_MASSES['muon']) / LEPTON_MASSES['muon']
            mass_tau_err = (E_tau - LEPTON_MASSES['tau']) / LEPTON_MASSES['tau']

            # g-factor residuals (absolute)
            g_e_err = g_e_pred - LEPTON_G_FACTORS['electron']
            g_mu_err = g_mu_pred - LEPTON_G_FACTORS['muon']

            # Chi-squared
            chi2_mass = (
                (mass_e_err / sigma_m_model)**2 +
                (mass_mu_err / sigma_m_model)**2 +
                (mass_tau_err / sigma_m_model)**2
            )
            chi2_g = (
                (g_e_err / sigma_g_model)**2 +
                (g_mu_err / sigma_g_model)**2
            )

            return chi2_mass + chi2_g

        except:
            return 1e15

    # Initial guess (will be reused and refined for each β)
    params_init = [
        0.88, 0.036, 0.92,  # electron
        0.13, 0.31, 0.92,   # muon
        0.50, 1.29, 0.92,   # tau
        250.0               # C_mu
    ]

    # Bounds (10 parameters, β excluded)
    bounds = [
        (0.1, 1.0), (0.001, 3.0), (0.1, 1.0),  # electron
        (0.1, 1.0), (0.001, 3.0), (0.1, 1.0),  # muon
        (0.1, 1.0), (0.001, 3.0), (0.1, 1.0),  # tau
        (100.0, 1500.0)                         # C_mu
    ]

    for i, beta in enumerate(beta_grid):
        if verbose:
            print(f"[{i+1:3d}/{len(beta_grid)}] β = {beta:.4f} ... ", end='', flush=True)

        # Minimize over all parameters except β
        result = minimize(
            lambda p: objective_with_beta_fixed(p, beta),
            params_init,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-9}
        )

        chi2_min = result.fun
        params_opt = result.x

        # Use this as warm start for next β
        params_init = params_opt

        # Extract final observables
        R_e, U_e, A_e, R_mu, U_mu, A_mu, R_tau, U_tau, A_tau, C_mu = params_opt

        # Update energy calcs
        for calc in energy_calcs.values():
            calc.beta = beta

        E_e, _, _ = energy_calcs['electron'].total_energy(R_e, U_e, A_e)
        E_mu, _, _ = energy_calcs['muon'].total_energy(R_mu, U_mu, A_mu)
        E_tau, _, _ = energy_calcs['tau'].total_energy(R_tau, U_tau, A_tau)

        mass_e_res = abs((E_e - LEPTON_MASSES['electron']) / LEPTON_MASSES['electron'])
        mass_mu_res = abs((E_mu - LEPTON_MASSES['muon']) / LEPTON_MASSES['muon'])
        mass_tau_res = abs((E_tau - LEPTON_MASSES['tau']) / LEPTON_MASSES['tau'])

        if verbose:
            status = "✓" if result.success else "✗"
            print(f"{status} χ² = {chi2_min:8.2f}, mass_res ~ {max(mass_e_res, mass_mu_res, mass_tau_res):.2e}")

        results.append({
            'beta': float(beta),
            'chi2_min': float(chi2_min),
            'C_mu': float(C_mu),
            'optimizer_success': bool(result.success),
            'mass_residuals': {
                'electron': float(mass_e_res),
                'muon': float(mass_mu_res),
                'tau': float(mass_tau_res)
            }
        })

    return {
        'test': 'Profile Likelihood in β',
        'purpose': 'Minimize χ² over all parameters for each fixed β',
        'beta_range': [float(beta_grid[0]), float(beta_grid[-1])],
        'num_points': len(beta_grid),
        'theory_uncertainties': {
            'sigma_m_model': float(sigma_m_model),
            'sigma_g_model': float(sigma_g_model)
        },
        'timestamp': datetime.now().isoformat(),
        'results': results
    }


def analyze_profile_likelihood(data):
    """Analyze profile likelihood results."""
    results = data['results']

    betas = [r['beta'] for r in results]
    chi2s = [r['chi2_min'] for r in results]

    min_idx = np.argmin(chi2s)
    beta_min = betas[min_idx]
    chi2_min = chi2s[min_idx]

    # Check for flatness
    chi2_range = max(chi2s) - min(chi2s)
    chi2_variation = chi2_range / chi2_min if chi2_min > 0 else 0

    # Check for multiple minima (crude: look for local minima)
    local_minima = []
    for i in range(1, len(chi2s)-1):
        if chi2s[i] < chi2s[i-1] and chi2s[i] < chi2s[i+1]:
            local_minima.append((betas[i], chi2s[i]))

    print(f"\n{'='*70}")
    print(f"PROFILE LIKELIHOOD ANALYSIS")
    print(f"{'='*70}")
    print(f"\nGlobal minimum:")
    print(f"  β = {beta_min:.6f}")
    print(f"  χ²_min = {chi2_min:.2f}")
    print(f"\nLandscape:")
    print(f"  χ² range: {chi2_range:.2f}")
    print(f"  Variation: {chi2_variation:.1%}")
    print(f"  Local minima found: {len(local_minima)}")

    if len(local_minima) > 1:
        print(f"\n  Multiple local minima:")
        for beta_loc, chi2_loc in local_minima[:5]:
            print(f"    β = {beta_loc:.4f}, χ² = {chi2_loc:.2f}")

    # Interpretation
    print(f"\n{'='*70}")
    print(f"INTERPRETATION")
    print(f"{'='*70}")

    if chi2_variation < 0.1:
        print(f"\n✗ FLAT PROFILE (variation < 10%)")
        print(f"  → β is NOT identified by cross-lepton data")
        print(f"  → Need different second observable or tighter constraints")
    elif chi2_variation < 1.0:
        print(f"\n~ WEAK PREFERENCE (variation 10-100%)")
        print(f"  → Some β sensitivity, but not sharp")
        print(f"  → May improve with tighter σ_model or additional constraints")
    else:
        print(f"\n✓ SHARP MINIMUM (variation > 100%)")
        print(f"  → β IS identified by cross-lepton data")
        print(f"  → Multi-start spread was likely optimization noise")

    # Check if minimum near 3.058
    offset = abs(beta_min - 3.058)
    print(f"\nβ minimum vs expected:")
    print(f"  Observed: {beta_min:.6f}")
    print(f"  Expected: 3.058000")
    print(f"  Offset: {offset:.6f} ({offset/3.058*100:.2f}%)")

    if offset < 0.01:
        print(f"  ✓ Within 1% of expected value")
    elif offset < 0.05:
        print(f"  ~ Within 5% of expected value")
    else:
        print(f"  ✗ More than 5% offset")

    print(f"{'='*70}\n")


def save_results(data, filename='results/profile_likelihood_beta.json'):
    """Save profile likelihood results."""
    output_path = Path(__file__).parent / filename
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"✓ Results saved to {output_path}")
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Profile likelihood in β: χ²_min(β) for fixed β grid'
    )
    parser.add_argument('--beta-min', type=float, default=2.85)
    parser.add_argument('--beta-max', type=float, default=3.25)
    parser.add_argument('--num-beta', type=int, default=41,
                       help='Number of β points (41 for Δβ=0.01)')
    parser.add_argument('--sigma-m', type=float, default=SIGMA_MASS_MODEL,
                       help='Mass theory uncertainty (relative)')
    parser.add_argument('--sigma-g', type=float, default=SIGMA_G_MODEL,
                       help='g-factor theory uncertainty (absolute)')

    args = parser.parse_args()

    # Create β grid
    beta_grid = np.linspace(args.beta_min, args.beta_max, args.num_beta)

    # Run profile likelihood
    results = profile_likelihood_beta(
        beta_grid,
        sigma_m_model=args.sigma_m,
        sigma_g_model=args.sigma_g,
        verbose=True
    )

    # Analyze
    analyze_profile_likelihood(results)

    # Save
    save_results(results)

    print("\n✓ Profile likelihood scan complete")
    print("  This cleanly separates optimization noise from real β-identifiability")
