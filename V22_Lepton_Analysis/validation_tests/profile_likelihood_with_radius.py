#!/usr/bin/env python3
"""
Profile Likelihood in β with Radius Constraint

Purpose: Test whether adding RMS radius as a second geometric observable
         (independent of EM coupling) shifts β_min from ~3.15 toward 3.058.

Method:
- For each β on a dense grid:
  - Minimize χ² over (R_e,U_e,A_e, R_μ,U_μ,A_μ, R_τ,U_τ,A_τ)
  - κ is profiled out analytically via κ_opt = mean(m_ℓ·R_rms,ℓ)
  - Record χ²_min(β)

Constraints (6 total):
- 3 masses: |E_total,ℓ - m_ℓ| / (σ_m,model · m_ℓ)
- 3 radius: |m_ℓ·R_rms,ℓ - κ_opt| / (σ_κ)

Success mode: β_min shifts toward 3.058, κ stable, G_τ/G_e ~ O(1)
Failure mode: β_min remains ~3.15, tau cannot satisfy radius constraint,
              G_τ/G_e >> 1 (pathological boundary)
"""

import numpy as np
from scipy.optimize import minimize
import json
from pathlib import Path
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent))
from test_cross_lepton_fit import LEPTON_MASSES, SIGMA_MASS_MODEL, RHO_VAC
from test_all_leptons_beta_from_alpha import LeptonEnergy
from radius_constraint import (
    compute_R_rms, compute_gradient_proxy, profile_kappa_analytically
)


def profile_likelihood_with_radius(
    beta_grid,
    sigma_m_model=SIGMA_MASS_MODEL,
    sigma_r_model=0.03,  # 3% relative on κ
    num_r=100,
    num_theta=20,
    verbose=True
):
    """
    Compute profile likelihood χ²_min(β) with mass + radius constraints.

    For each β, minimize over all lepton parameters. κ is profiled analytically.

    Args:
        beta_grid: Array of β values to test
        sigma_m_model: Mass theory uncertainty (relative)
        sigma_r_model: Radius scaling uncertainty (relative, on κ)
        num_r, num_theta: Grid resolution
        verbose: Print progress

    Returns:
        dict with β grid and corresponding χ²_min values + diagnostics
    """
    results = []

    if verbose:
        print(f"\n{'='*70}")
        print(f"PROFILE LIKELIHOOD: χ²_min(β) WITH RADIUS CONSTRAINT")
        print(f"{'='*70}")
        print(f"β grid: [{beta_grid[0]:.3f}, {beta_grid[-1]:.3f}] with {len(beta_grid)} points")
        print(f"σ_m,model = {sigma_m_model:.2e} (relative)")
        print(f"σ_r,model = {sigma_r_model:.2e} (relative, on κ)")
        print(f"Constraints: 3 masses + 3 radius relations (6 total)")
        print()

    # Create energy calculators
    energy_calcs = {
        'electron': LeptonEnergy(beta=3.0, num_r=num_r, num_theta=num_theta),
        'muon': LeptonEnergy(beta=3.0, num_r=num_r, num_theta=num_theta),
        'tau': LeptonEnergy(beta=3.0, num_r=num_r, num_theta=num_theta)
    }

    def objective_with_beta_fixed(params, beta_fixed):
        """
        Objective function with β fixed, κ profiled analytically.

        Parameters (9 total):
            params = [R_e, U_e, A_e, R_μ, U_μ, A_μ, R_τ, U_τ, A_τ]

        Returns:
            χ² = χ²_mass + χ²_radius
        """
        R_e, U_e, A_e, R_mu, U_mu, A_mu, R_tau, U_tau, A_tau = params

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
            # Compute energies and R_rms for each lepton
            E_e, _, delta_rho_e, r_grid_e = energy_calcs['electron'].total_energy(
                R_e, U_e, A_e, return_profiles=True
            )
            E_mu, _, delta_rho_mu, r_grid_mu = energy_calcs['muon'].total_energy(
                R_mu, U_mu, A_mu, return_profiles=True
            )
            E_tau, _, delta_rho_tau, r_grid_tau = energy_calcs['tau'].total_energy(
                R_tau, U_tau, A_tau, return_profiles=True
            )

            # Compute RMS radii
            R_rms_e = compute_R_rms(delta_rho_e, r_grid_e)
            R_rms_mu = compute_R_rms(delta_rho_mu, r_grid_mu)
            R_rms_tau = compute_R_rms(delta_rho_tau, r_grid_tau)

            # Mass residuals (relative)
            mass_e_err = (E_e - LEPTON_MASSES['electron']) / LEPTON_MASSES['electron']
            mass_mu_err = (E_mu - LEPTON_MASSES['muon']) / LEPTON_MASSES['muon']
            mass_tau_err = (E_tau - LEPTON_MASSES['tau']) / LEPTON_MASSES['tau']

            # Profile κ analytically
            x_leptons = {
                'electron': LEPTON_MASSES['electron'] * R_rms_e,
                'muon': LEPTON_MASSES['muon'] * R_rms_mu,
                'tau': LEPTON_MASSES['tau'] * R_rms_tau
            }
            kappa_opt, sigma_kappa, chi2_radius = profile_kappa_analytically(
                x_leptons, f_sigma=sigma_r_model
            )

            # Mass χ²
            chi2_mass = (
                (mass_e_err / sigma_m_model)**2 +
                (mass_mu_err / sigma_m_model)**2 +
                (mass_tau_err / sigma_m_model)**2
            )

            # Total χ² (6 constraints: 3 masses + 3 radius relations)
            chi2_total = chi2_mass + chi2_radius

            return chi2_total

        except Exception as e:
            # Solver failure or numerical issue
            return 1e15

    # Initial guess (will be refined for each β)
    params_init = [
        0.88, 0.036, 0.92,  # electron
        0.13, 0.31, 0.92,   # muon
        0.50, 1.29, 0.92,   # tau
    ]

    # Bounds (9 parameters)
    bounds = [
        (0.1, 1.0), (0.001, 3.0), (0.1, 1.0),  # electron
        (0.1, 1.0), (0.001, 3.0), (0.1, 1.0),  # muon
        (0.1, 1.0), (0.001, 3.0), (0.1, 1.0),  # tau
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

        # Extract final observables for diagnostics
        R_e, U_e, A_e, R_mu, U_mu, A_mu, R_tau, U_tau, A_tau = params_opt

        # Update energy calcs
        for calc in energy_calcs.values():
            calc.beta = beta

        # Compute all quantities for reporting
        E_e, _, delta_rho_e, r_grid_e = energy_calcs['electron'].total_energy(
            R_e, U_e, A_e, return_profiles=True
        )
        E_mu, _, delta_rho_mu, r_grid_mu = energy_calcs['muon'].total_energy(
            R_mu, U_mu, A_mu, return_profiles=True
        )
        E_tau, _, delta_rho_tau, r_grid_tau = energy_calcs['tau'].total_energy(
            R_tau, U_tau, A_tau, return_profiles=True
        )

        # RMS radii
        R_rms_e = compute_R_rms(delta_rho_e, r_grid_e)
        R_rms_mu = compute_R_rms(delta_rho_mu, r_grid_mu)
        R_rms_tau = compute_R_rms(delta_rho_tau, r_grid_tau)

        # Gradient proxies (diagnostic)
        rho_e = RHO_VAC - delta_rho_e
        rho_mu = RHO_VAC - delta_rho_mu
        rho_tau = RHO_VAC - delta_rho_tau
        G_e = compute_gradient_proxy(rho_e, r_grid_e)
        G_mu = compute_gradient_proxy(rho_mu, r_grid_mu)
        G_tau = compute_gradient_proxy(rho_tau, r_grid_tau)

        # Profiled κ
        x_leptons = {
            'electron': LEPTON_MASSES['electron'] * R_rms_e,
            'muon': LEPTON_MASSES['muon'] * R_rms_mu,
            'tau': LEPTON_MASSES['tau'] * R_rms_tau
        }
        kappa_opt, sigma_kappa, chi2_radius = profile_kappa_analytically(
            x_leptons, f_sigma=sigma_r_model
        )

        # Mass residuals
        mass_e_res = abs((E_e - LEPTON_MASSES['electron']) / LEPTON_MASSES['electron'])
        mass_mu_res = abs((E_mu - LEPTON_MASSES['muon']) / LEPTON_MASSES['muon'])
        mass_tau_res = abs((E_tau - LEPTON_MASSES['tau']) / LEPTON_MASSES['tau'])

        if verbose:
            status = "✓" if result.success else "✗"
            print(f"{status} χ² = {chi2_min:8.2f}, κ = {kappa_opt:.4e}, "
                  f"G_τ/G_e = {G_tau/G_e:.2f}")

        results.append({
            'beta': float(beta),
            'chi2_min': float(chi2_min),
            'chi2_mass': float(chi2_min - chi2_radius),
            'chi2_radius': float(chi2_radius),
            'kappa_opt': float(kappa_opt),
            'sigma_kappa': float(sigma_kappa),
            'optimizer_success': bool(result.success),
            'mass_residuals': {
                'electron': float(mass_e_res),
                'muon': float(mass_mu_res),
                'tau': float(mass_tau_res)
            },
            'R_rms': {
                'electron': float(R_rms_e),
                'muon': float(R_rms_mu),
                'tau': float(R_rms_tau)
            },
            'gradient_proxy': {
                'electron': float(G_e),
                'muon': float(G_mu),
                'tau': float(G_tau),
                'tau_to_electron_ratio': float(G_tau / G_e) if G_e > 0 else 0.0
            },
            'x_leptons': {
                'electron': float(x_leptons['electron']),
                'muon': float(x_leptons['muon']),
                'tau': float(x_leptons['tau'])
            }
        })

    return {
        'test': 'Profile Likelihood in β with Radius Constraint',
        'purpose': 'Test if geometric constraint shifts β_min toward 3.058',
        'beta_range': [float(beta_grid[0]), float(beta_grid[-1])],
        'num_points': len(beta_grid),
        'theory_uncertainties': {
            'sigma_m_model': float(sigma_m_model),
            'sigma_r_model': float(sigma_r_model)
        },
        'constraints': '3 masses + 3 radius relations (m·R_rms = κ)',
        'timestamp': datetime.now().isoformat(),
        'results': results
    }


def analyze_profile_likelihood_with_radius(data):
    """Analyze profile likelihood results with radius constraint."""
    results = data['results']

    betas = [r['beta'] for r in results]
    chi2s = [r['chi2_min'] for r in results]
    kappas = [r['kappa_opt'] for r in results]
    G_ratios = [r['gradient_proxy']['tau_to_electron_ratio'] for r in results]

    min_idx = np.argmin(chi2s)
    beta_min = betas[min_idx]
    chi2_min = chi2s[min_idx]
    kappa_at_min = kappas[min_idx]

    # Check for flatness
    chi2_range = max(chi2s) - min(chi2s)
    chi2_variation = chi2_range / chi2_min if chi2_min > 0 else 0

    # κ stability
    kappa_mean = np.mean(kappas)
    kappa_std = np.std(kappas)
    kappa_cv = kappa_std / kappa_mean if kappa_mean > 0 else 0

    # G_τ/G_e at minimum
    G_ratio_at_min = G_ratios[min_idx]

    print(f"\n{'='*70}")
    print(f"PROFILE LIKELIHOOD ANALYSIS (WITH RADIUS)")
    print(f"{'='*70}")
    print(f"\nGlobal minimum:")
    print(f"  β = {beta_min:.6f}")
    print(f"  χ²_min = {chi2_min:.2f}")
    print(f"  κ at min = {kappa_at_min:.4e}")
    print(f"\nLandscape:")
    print(f"  χ² range: {chi2_range:.2f}")
    print(f"  Variation: {chi2_variation:.1%}")
    print(f"\nκ stability across β:")
    print(f"  Mean: {kappa_mean:.4e}")
    print(f"  Std: {kappa_std:.4e}")
    print(f"  CV: {kappa_cv:.1%}")
    print(f"\nGradient proxy at minimum:")
    print(f"  G_τ / G_e = {G_ratio_at_min:.2f}")

    # Comparison to Golden Loop
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

    # Interpretation
    print(f"\n{'='*70}")
    print(f"INTERPRETATION")
    print(f"{'='*70}")

    if chi2_variation > 1.0:
        print(f"\n✓ SHARP MINIMUM (variation > 100%)")
        print(f"  → β IS identified by mass + radius constraints")
    else:
        print(f"\n✗ WEAK PREFERENCE (variation < 100%)")
        print(f"  → Radius constraint may not be breaking degeneracy")

    if offset < 0.05:
        print(f"\n✓ β NEAR GOLDEN LOOP VALUE")
        print(f"  → Curvature/geometry constraint resolved the offset")
        print(f"  → Mechanism test: SUCCESS")
    else:
        print(f"\n✗ β STILL OFFSET FROM 3.058")
        print(f"  → Missing curvature energy is not the only discrepancy")
        print(f"  → Additional closure refinement needed")

    if G_ratio_at_min < 5.0:
        print(f"\n✓ GRADIENT PROXY REASONABLE (G_τ/G_e = {G_ratio_at_min:.2f})")
        print(f"  → Tau boundary not pathological")
    else:
        print(f"\n✗ GRADIENT PROXY LARGE (G_τ/G_e = {G_ratio_at_min:.2f})")
        print(f"  → Tau may have unphysically steep boundary")
        print(f"  → Closure missing explicit curvature penalty")

    print(f"{'='*70}\n")


def save_results(data, filename='results/profile_likelihood_with_radius.json'):
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
        description='Profile likelihood in β with radius constraint'
    )
    parser.add_argument('--beta-min', type=float, default=2.95)
    parser.add_argument('--beta-max', type=float, default=3.25)
    parser.add_argument('--num-beta', type=int, default=31,
                       help='Number of β points')
    parser.add_argument('--sigma-m', type=float, default=SIGMA_MASS_MODEL,
                       help='Mass theory uncertainty (relative)')
    parser.add_argument('--sigma-r', type=float, default=0.03,
                       help='Radius theory uncertainty (relative, on κ)')

    args = parser.parse_args()

    # Create β grid
    beta_grid = np.linspace(args.beta_min, args.beta_max, args.num_beta)

    # Run profile likelihood
    results = profile_likelihood_with_radius(
        beta_grid,
        sigma_m_model=args.sigma_m,
        sigma_r_model=args.sigma_r,
        verbose=True
    )

    # Analyze
    analyze_profile_likelihood_with_radius(results)

    # Save
    save_results(results)

    print("\n✓ Profile likelihood scan with radius constraint complete")
    print("  This tests whether geometric constraint resolves β_eff offset")
