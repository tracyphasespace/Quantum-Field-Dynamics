#!/usr/bin/env python3
"""
Profile Likelihood in β with Gradient (Curvature) Term

Purpose: THE DECISIVE TEST of whether missing curvature energy explains
         the β_eff ≈ 3.15 offset from Golden Loop β = 3.058.

Method:
- Use Family A profile: Δρ = -A(1-(r/R)²)^s with s=2 (fixed globally)
- Add gradient energy: E_grad = λ·K_grad(s)·A²R with λ=1 (fixed)
- Minimize χ² over (R_e,U_e,A_e, R_μ,U_μ,A_μ, R_τ,U_τ,A_τ) for each β
- Record χ²_min(β)

Success mode:
- β_min shifts toward 3.058 (curvature gap was the culprit)
- Landscape sharpens (E_grad ~ R breaks degeneracy)
- e/μ/τ solutions converge without pathological geometry

Failure mode:
- β_min remains ~3.15 (additional missing physics)
- Tau forces extreme parameters
- Clear diagnostic of what closure still lacks
"""

import numpy as np
from scipy.optimize import minimize
import json
from pathlib import Path
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent))
from lepton_energy_with_gradient import LeptonEnergy, K_stab, K_grad
from test_cross_lepton_fit import LEPTON_MASSES, SIGMA_MASS_MODEL, RHO_VAC


def profile_likelihood_with_gradient(
    beta_grid,
    s=2.0,
    lam=1.0,
    sigma_m_model=SIGMA_MASS_MODEL,
    num_r=100,
    num_theta=20,
    verbose=True
):
    """
    Compute profile likelihood χ²_min(β) with gradient energy.

    For each β, minimize over all lepton parameters. Shape (s) and
    gradient coefficient (λ) are fixed globally.

    Args:
        beta_grid: Array of β values to test
        s: Shape parameter (fixed, default 2.0 for C¹ smoothness)
        lam: Gradient coefficient (fixed, default 1.0)
        sigma_m_model: Mass theory uncertainty (relative)
        num_r, num_theta: Grid resolution
        verbose: Print progress

    Returns:
        dict with β grid, χ²_min values, and diagnostics
    """
    results = []

    if verbose:
        print(f"\n{'='*70}")
        print(f"PROFILE LIKELIHOOD WITH GRADIENT ENERGY")
        print(f"{'='*70}")
        print(f"Profile: Family A, Δρ = -A(1-(r/R)²)^s")
        print(f"  s = {s:.1f} (fixed globally)")
        print(f"  λ = {lam:.1f} (fixed globally)")
        print(f"  K_stab(s) = {K_stab(s):.6f}")
        print(f"  K_grad(s) = {K_grad(s):.6f}")
        print(f"\nβ grid: [{beta_grid[0]:.3f}, {beta_grid[-1]:.3f}] with {len(beta_grid)} points")
        print(f"σ_m,model = {sigma_m_model:.2e} (relative)")
        print(f"Constraints: 3 masses (6 DOF vs 10 params)")
        print()

    # Create energy calculators (s and λ fixed globally)
    energy_calcs = {
        'electron': LeptonEnergy(beta=3.0, s=s, lam=lam, num_r=num_r, num_theta=num_theta),
        'muon': LeptonEnergy(beta=3.0, s=s, lam=lam, num_r=num_r, num_theta=num_theta),
        'tau': LeptonEnergy(beta=3.0, s=s, lam=lam, num_r=num_r, num_theta=num_theta)
    }

    def objective_with_beta_fixed(params, beta_fixed):
        """
        Objective function with β, s, λ fixed.

        Parameters (9 total):
            params = [R_e, U_e, A_e, R_μ, U_μ, A_μ, R_τ, U_τ, A_τ]

        Returns:
            χ² from mass constraints
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
            # Compute energies for each lepton
            E_e, _, _, _ = energy_calcs['electron'].total_energy(R_e, U_e, A_e)
            E_mu, _, _, _ = energy_calcs['muon'].total_energy(R_mu, U_mu, A_mu)
            E_tau, _, _, _ = energy_calcs['tau'].total_energy(R_tau, U_tau, A_tau)

            # Mass residuals (relative)
            mass_e_err = (E_e - LEPTON_MASSES['electron']) / LEPTON_MASSES['electron']
            mass_mu_err = (E_mu - LEPTON_MASSES['muon']) / LEPTON_MASSES['muon']
            mass_tau_err = (E_tau - LEPTON_MASSES['tau']) / LEPTON_MASSES['tau']

            # χ² from masses
            chi2 = (
                (mass_e_err / sigma_m_model)**2 +
                (mass_mu_err / sigma_m_model)**2 +
                (mass_tau_err / sigma_m_model)**2
            )

            return chi2

        except Exception as e:
            return 1e15

    # Initial guess (will be refined for each β)
    params_init = [
        0.88, 0.036, 0.92,  # electron
        0.13, 0.31, 0.92,   # muon
        0.50, 1.29, 0.92,   # tau
    ]

    # Bounds (9 parameters)
    bounds = [
        (0.1, 1.5), (0.001, 3.0), (0.1, 1.0),  # electron
        (0.1, 1.5), (0.001, 3.0), (0.1, 1.0),  # muon
        (0.1, 1.5), (0.001, 3.0), (0.1, 1.0),  # tau
    ]

    for i, beta in enumerate(beta_grid):
        if verbose:
            print(f"[{i+1:3d}/{len(beta_grid)}] β = {beta:.4f} ... ", end='', flush=True)

        # Minimize over all parameters except β, s, λ
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
        E_e, E_circ_e, E_stab_e, E_grad_e = energy_calcs['electron'].total_energy(R_e, U_e, A_e)
        E_mu, E_circ_mu, E_stab_mu, E_grad_mu = energy_calcs['muon'].total_energy(R_mu, U_mu, A_mu)
        E_tau, E_circ_tau, E_stab_tau, E_grad_tau = energy_calcs['tau'].total_energy(R_tau, U_tau, A_tau)

        # Energy ratios (curvature vs bulk competition)
        ratio_e = E_grad_e / E_stab_e if E_stab_e > 0 else 0
        ratio_mu = E_grad_mu / E_stab_mu if E_stab_mu > 0 else 0
        ratio_tau = E_grad_tau / E_stab_tau if E_stab_tau > 0 else 0

        # Mass residuals
        mass_e_res = abs((E_e - LEPTON_MASSES['electron']) / LEPTON_MASSES['electron'])
        mass_mu_res = abs((E_mu - LEPTON_MASSES['muon']) / LEPTON_MASSES['muon'])
        mass_tau_res = abs((E_tau - LEPTON_MASSES['tau']) / LEPTON_MASSES['tau'])

        if verbose:
            status = "✓" if result.success else "✗"
            print(f"{status} χ² = {chi2_min:8.2f}, "
                  f"E_∇/E_s: e={ratio_e:.2f} μ={ratio_mu:.2f} τ={ratio_tau:.2f}")

        results.append({
            'beta': float(beta),
            'chi2_min': float(chi2_min),
            'optimizer_success': bool(result.success),
            'mass_residuals': {
                'electron': float(mass_e_res),
                'muon': float(mass_mu_res),
                'tau': float(mass_tau_res)
            },
            'parameters': {
                'electron': {'R': float(R_e), 'U': float(U_e), 'A': float(A_e)},
                'muon': {'R': float(R_mu), 'U': float(U_mu), 'A': float(A_mu)},
                'tau': {'R': float(R_tau), 'U': float(U_tau), 'A': float(A_tau)}
            },
            'energies': {
                'electron': {
                    'E_total': float(E_e),
                    'E_circ': float(E_circ_e),
                    'E_stab': float(E_stab_e),
                    'E_grad': float(E_grad_e),
                    'E_grad_over_E_stab': float(ratio_e)
                },
                'muon': {
                    'E_total': float(E_mu),
                    'E_circ': float(E_circ_mu),
                    'E_stab': float(E_stab_mu),
                    'E_grad': float(E_grad_mu),
                    'E_grad_over_E_stab': float(ratio_mu)
                },
                'tau': {
                    'E_total': float(E_tau),
                    'E_circ': float(E_circ_tau),
                    'E_stab': float(E_stab_tau),
                    'E_grad': float(E_grad_tau),
                    'E_grad_over_E_stab': float(ratio_tau)
                }
            }
        })

    return {
        'test': 'Profile Likelihood with Gradient Energy',
        'purpose': 'Test if curvature term resolves β_eff offset',
        'profile': 'Family A: Δρ = -A(1-(r/R)²)^s',
        's_fixed': float(s),
        'lambda_fixed': float(lam),
        'K_stab': float(K_stab(s)),
        'K_grad': float(K_grad(s)),
        'beta_range': [float(beta_grid[0]), float(beta_grid[-1])],
        'num_points': len(beta_grid),
        'theory_uncertainties': {
            'sigma_m_model': float(sigma_m_model)
        },
        'timestamp': datetime.now().isoformat(),
        'results': results
    }


def analyze_profile_likelihood_with_gradient(data):
    """Analyze profile likelihood results with gradient energy."""
    results = data['results']
    s = data['s_fixed']
    lam = data['lambda_fixed']

    betas = [r['beta'] for r in results]
    chi2s = [r['chi2_min'] for r in results]

    min_idx = np.argmin(chi2s)
    beta_min = betas[min_idx]
    chi2_min = chi2s[min_idx]

    # Check for flatness
    chi2_range = max(chi2s) - min(chi2s)
    chi2_variation = chi2_range / chi2_min if chi2_min > 0 else 0

    # Energy ratios at minimum
    ratios_at_min = results[min_idx]['energies']
    ratio_e = ratios_at_min['electron']['E_grad_over_E_stab']
    ratio_mu = ratios_at_min['muon']['E_grad_over_E_stab']
    ratio_tau = ratios_at_min['tau']['E_grad_over_E_stab']

    print(f"\n{'='*70}")
    print(f"PROFILE LIKELIHOOD ANALYSIS (WITH GRADIENT)")
    print(f"{'='*70}")
    print(f"\nGlobal minimum:")
    print(f"  β = {beta_min:.6f}")
    print(f"  χ²_min = {chi2_min:.2f}")
    print(f"\nLandscape:")
    print(f"  χ² range: {chi2_range:.2f}")
    print(f"  Variation: {chi2_variation:.1%}")

    print(f"\nEnergy ratios E_grad/E_stab at minimum:")
    print(f"  Electron: {ratio_e:.4f}")
    print(f"  Muon:     {ratio_mu:.4f}")
    print(f"  Tau:      {ratio_tau:.4f}")

    # Comparison to Golden Loop
    offset = abs(beta_min - 3.058)
    offset_pct = offset / 3.058 * 100

    print(f"\nβ minimum vs Golden Loop:")
    print(f"  Observed:  {beta_min:.6f}")
    print(f"  Expected:  3.058000")
    print(f"  Offset:    {offset:.6f} ({offset_pct:.2f}%)")

    # Interpretation
    print(f"\n{'='*70}")
    print(f"INTERPRETATION")
    print(f"{'='*70}")

    if chi2_variation > 1.0:
        print(f"\n✓ SHARP MINIMUM (variation > 100%)")
        print(f"  → β IS identified by gradient + mass constraints")
    else:
        print(f"\n✗ WEAK PREFERENCE (variation < 100%)")
        print(f"  → Gradient energy may not be breaking degeneracy enough")

    if offset_pct < 1.0:
        print(f"\n✓✓ EXCELLENT: β WITHIN 1% OF GOLDEN LOOP")
        print(f"  → Curvature gap WAS the primary source of β_eff offset")
        print(f"  → Mechanism test: STRONG SUCCESS")
    elif offset_pct < 3.0:
        print(f"\n✓ GOOD: β WITHIN 3% OF GOLDEN LOOP")
        print(f"  → Curvature gap accounts for most of offset")
        print(f"  → Mechanism test: SUCCESS with residual discrepancy")
    elif offset_pct < 5.0:
        print(f"\n~ PARTIAL: β IMPROVED BUT STILL 3-5% OFFSET")
        print(f"  → Curvature helps but doesn't fully resolve offset")
        print(f"  → May need additional refinements (EM response, etc.)")
    else:
        print(f"\n✗ NO IMPROVEMENT: β STILL >5% OFFSET")
        print(f"  → Curvature gap is not the main discrepancy")
        print(f"  → Different closure refinement needed")

    if all(0.1 < r < 10.0 for r in [ratio_e, ratio_mu, ratio_tau]):
        print(f"\n✓ ENERGY RATIOS REASONABLE")
        print(f"  → Gradient and bulk terms are comparable (healthy competition)")
    else:
        print(f"\n⚠ ENERGY RATIOS EXTREME")
        print(f"  → One term may be dominating (check parameter validity)")

    print(f"{'='*70}\n")

    return {
        'beta_min': beta_min,
        'offset_from_golden': offset,
        'offset_percent': offset_pct,
        'chi2_variation': chi2_variation,
        'energy_ratios': {'electron': ratio_e, 'muon': ratio_mu, 'tau': ratio_tau}
    }


def save_results(data, filename='results/profile_likelihood_with_gradient.json'):
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
        description='Profile likelihood in β with gradient energy (curvature term)'
    )
    parser.add_argument('--beta-min', type=float, default=2.95)
    parser.add_argument('--beta-max', type=float, default=3.25)
    parser.add_argument('--num-beta', type=int, default=31,
                       help='Number of β points')
    parser.add_argument('--s', type=float, default=2.0,
                       help='Shape parameter (fixed)')
    parser.add_argument('--lam', type=float, default=1.0,
                       help='Gradient coefficient (fixed)')
    parser.add_argument('--sigma-m', type=float, default=SIGMA_MASS_MODEL,
                       help='Mass theory uncertainty (relative)')

    args = parser.parse_args()

    # Create β grid
    beta_grid = np.linspace(args.beta_min, args.beta_max, args.num_beta)

    # Run profile likelihood
    results = profile_likelihood_with_gradient(
        beta_grid,
        s=args.s,
        lam=args.lam,
        sigma_m_model=args.sigma_m,
        verbose=True
    )

    # Analyze
    summary = analyze_profile_likelihood_with_gradient(results)

    # Save
    save_results(results)

    print("\n✓ Profile likelihood scan with gradient energy complete")
    print("  THE DECISIVE TEST of curvature-gap hypothesis")
