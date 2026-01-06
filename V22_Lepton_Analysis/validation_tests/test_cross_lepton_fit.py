#!/usr/bin/env python3
"""
Cross-Lepton Multi-Objective Fit

Purpose: Test if ONE shared β can simultaneously fit all three leptons (e, μ, τ)
         with shared μ-normalization constant C_μ.

This is the "real falsifiability test" - if different leptons want different β,
the "universal vacuum stiffness" hypothesis is falsified.

Parameters (11 total):
- Per-lepton: (R_ℓ, U_ℓ, A_ℓ) for ℓ ∈ {e, μ, τ} (9 params)
- Shared: β, C_μ (2 params)

Constraints (5 total, since tau g-factor is omitted):
- 3 masses: m_e, m_μ, m_τ
- 2 g-factors: g_e, g_μ (tau g-factor omitted - poorly measured)

Theory uncertainties (Tracy's guidance):
- σ_m,model = 1×10⁻⁶ (relative)
- σ_g,model = 2×10⁻⁸ (absolute)

Optimization strategy:
- Analytically eliminate C_μ (least-squares solution given other params)
- This reduces dimension and stabilizes the fit
- Multi-start from random initial guesses
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import simps
import json
from pathlib import Path
from datetime import datetime
import sys

# Import production solver classes
sys.path.insert(0, str(Path(__file__).parent))
from test_all_leptons_beta_from_alpha import (
    HillVortexStreamFunction, DensityGradient, LeptonEnergy,
    ELECTRON_MASS, MUON_TO_ELECTRON_RATIO, TAU_TO_ELECTRON_RATIO, RHO_VAC
)

# ===========================================================================
# PHYSICAL CONSTANTS AND PARAMETERS
# ===========================================================================

# Lepton masses (in units of electron mass)
LEPTON_MASSES = {
    'electron': ELECTRON_MASS,
    'muon': MUON_TO_ELECTRON_RATIO,
    'tau': TAU_TO_ELECTRON_RATIO
}

# Lepton g-factors (CODATA 2018)
G_ELECTRON = 2.00231930436256
G_MUON = 2.0023318418
G_TAU = 2.0  # Placeholder - poorly measured, will be omitted

LEPTON_G_FACTORS = {
    'electron': G_ELECTRON,
    'muon': G_MUON,
    'tau': G_TAU  # Not used in fit
}

# Theory uncertainties (Tracy's guidance, 2025-12-23)
SIGMA_MASS_MODEL = 1e-6  # Relative uncertainty
SIGMA_G_MODEL = 2e-8     # Absolute uncertainty (for e, μ)

# Magnetic moment formula constants
K_GEOMETRIC = 0.2  # Geometric factor for uniform vorticity Hill vortex
Q_CHARGE = 1.0     # Fundamental charge

# ===========================================================================
# MAGNETIC MOMENT CALCULATION
# ===========================================================================

def magnetic_moment_raw(R, U):
    """
    Raw magnetic moment from Hill vortex circulation.

    μ = k × Q × R × U

    where k ≈ 0.2 (geometric factor), Q = 1 (charge).
    """
    return K_GEOMETRIC * Q_CHARGE * R * U


def solve_C_mu_analytic(mu_e, mu_mu, g_e_obs, g_mu_obs, sigma_g):
    """
    Analytically solve for C_μ (normalization constant) via least-squares.

    Given:
        g_pred = C_μ × μ

    Minimize:
        χ²_g = Σ[(g_pred - g_obs)/σ_g]²
             = [(C_μ×μ_e - g_e)/σ_g]² + [(C_μ×μ_μ - g_μ)/σ_g]²

    Solution:
        dχ²/dC_μ = 0
        => C_μ = (g_e×μ_e + g_μ×μ_μ) / (μ_e² + μ_μ²)

    This is the weighted least-squares solution (weights cancel since σ_g is same).

    Args:
        mu_e, mu_mu: Raw magnetic moments for electron, muon
        g_e_obs, g_mu_obs: Observed g-factors
        sigma_g: g-factor uncertainty (for reference, not used in analytic solution)

    Returns:
        C_mu: Optimal normalization constant
    """
    numerator = g_e_obs * mu_e + g_mu_obs * mu_mu
    denominator = mu_e**2 + mu_mu**2

    if denominator < 1e-20:
        return 1.0  # Fallback if moments vanish

    return numerator / denominator


# ===========================================================================
# CROSS-LEPTON OBJECTIVE FUNCTION
# ===========================================================================

def cross_lepton_objective(params, energy_calcs, use_analytic_C_mu=True):
    """
    Joint objective function for all three leptons with shared β.

    Parameters (10 if C_μ analytic, else 11):
        If use_analytic_C_mu=True:
            params = [R_e, U_e, A_e, R_μ, U_μ, A_μ, R_τ, U_τ, A_τ, β]
            (C_μ computed analytically)
        Else:
            params = [R_e, U_e, A_e, R_μ, U_μ, A_μ, R_τ, U_τ, A_τ, β, C_μ]

    Constraints:
        - 3 masses (electron, muon, tau)
        - 2 g-factors (electron, muon) - tau omitted

    Chi-squared with theory uncertainties:
        χ² = Σ[(Δm/σ_m,eff)²] + Σ[(Δg/σ_g,eff)²]

    Args:
        params: Parameter vector
        energy_calcs: Dict of LeptonEnergy instances (one per lepton)
        use_analytic_C_mu: If True, solve for C_μ analytically

    Returns:
        chi_squared: Total χ² value
    """
    if use_analytic_C_mu:
        # 10 parameters
        R_e, U_e, A_e, R_mu, U_mu, A_mu, R_tau, U_tau, A_tau, beta = params
        C_mu = None  # Will be computed analytically
    else:
        # 11 parameters
        R_e, U_e, A_e, R_mu, U_mu, A_mu, R_tau, U_tau, A_tau, beta, C_mu = params

    # Physical bounds
    if beta <= 0:
        return 1e15

    for R, U, A in [(R_e, U_e, A_e), (R_mu, U_mu, A_mu), (R_tau, U_tau, A_tau)]:
        if R <= 0 or U <= 0 or A <= 0:
            return 1e15
        if A > RHO_VAC:  # Cavitation constraint
            return 1e15

    try:
        # Update energy calculators with current β
        for calc in energy_calcs.values():
            calc.beta = beta

        # Compute masses
        E_e, E_e_circ, E_e_stab = energy_calcs['electron'].total_energy(R_e, U_e, A_e)
        E_mu, E_mu_circ, E_mu_stab = energy_calcs['muon'].total_energy(R_mu, U_mu, A_mu)
        E_tau, E_tau_circ, E_tau_stab = energy_calcs['tau'].total_energy(R_tau, U_tau, A_tau)

        # Compute raw magnetic moments
        mu_e = magnetic_moment_raw(R_e, U_e)
        mu_mu = magnetic_moment_raw(R_mu, U_mu)
        # (tau moment not used in fit)

        # Analytic C_μ solution if requested
        if use_analytic_C_mu:
            C_mu = solve_C_mu_analytic(
                mu_e, mu_mu,
                LEPTON_G_FACTORS['electron'],
                LEPTON_G_FACTORS['muon'],
                SIGMA_G_MODEL
            )

        # Predicted g-factors
        g_e_pred = C_mu * mu_e
        g_mu_pred = C_mu * mu_mu

        # Mass residuals (relative)
        mass_e_err = (E_e - LEPTON_MASSES['electron']) / LEPTON_MASSES['electron']
        mass_mu_err = (E_mu - LEPTON_MASSES['muon']) / LEPTON_MASSES['muon']
        mass_tau_err = (E_tau - LEPTON_MASSES['tau']) / LEPTON_MASSES['tau']

        # g-factor residuals (absolute)
        g_e_err = g_e_pred - LEPTON_G_FACTORS['electron']
        g_mu_err = g_mu_pred - LEPTON_G_FACTORS['muon']

        # Chi-squared with theory uncertainties
        # σ_eff² = σ_exp² + σ_model²
        # For our case: σ_exp << σ_model, so σ_eff ≈ σ_model

        chi2_mass = (
            (mass_e_err / SIGMA_MASS_MODEL)**2 +
            (mass_mu_err / SIGMA_MASS_MODEL)**2 +
            (mass_tau_err / SIGMA_MASS_MODEL)**2
        )

        chi2_g = (
            (g_e_err / SIGMA_G_MODEL)**2 +
            (g_mu_err / SIGMA_G_MODEL)**2
        )

        chi2_total = chi2_mass + chi2_g

        return chi2_total

    except Exception as e:
        return 1e15


# ===========================================================================
# CROSS-LEPTON FIT
# ===========================================================================

def fit_cross_lepton(beta_initial=3.058, use_analytic_C_mu=True,
                     method='L-BFGS-B', num_r=100, num_theta=20,
                     max_iter=1000, verbose=True):
    """
    Fit all three leptons simultaneously with ONE shared β.

    Args:
        beta_initial: Initial guess for β
        use_analytic_C_mu: If True, analytically eliminate C_μ
        method: Optimization method
        num_r, num_theta: Grid resolution
        max_iter: Maximum iterations
        verbose: Print progress

    Returns:
        dict with results
    """
    # Create energy calculators (one per lepton)
    energy_calcs = {
        'electron': LeptonEnergy(beta=beta_initial, num_r=num_r, num_theta=num_theta),
        'muon': LeptonEnergy(beta=beta_initial, num_r=num_r, num_theta=num_theta),
        'tau': LeptonEnergy(beta=beta_initial, num_r=num_r, num_theta=num_theta)
    }

    # Initial guess
    # Start with approximate scaling from mass-only fits
    R_e_init = 0.44
    U_e_init = 0.024
    A_e_init = 0.90

    # Muon: roughly same R, scale U by sqrt(mass ratio)
    R_mu_init = 0.49
    U_mu_init = 0.31
    A_mu_init = 0.91

    # Tau: similar
    R_tau_init = 0.50
    U_tau_init = 1.26
    A_tau_init = 0.92

    if use_analytic_C_mu:
        # 10 parameters
        params_init = [
            R_e_init, U_e_init, A_e_init,
            R_mu_init, U_mu_init, A_mu_init,
            R_tau_init, U_tau_init, A_tau_init,
            beta_initial
        ]
        bounds = [
            (0.1, 1.0), (0.001, 3.0), (0.1, 1.0),  # electron
            (0.1, 1.0), (0.001, 3.0), (0.1, 1.0),  # muon
            (0.1, 1.0), (0.001, 3.0), (0.1, 1.0),  # tau
            (2.5, 3.5)                              # beta
        ]
    else:
        # 11 parameters
        C_mu_init = 950.0  # Approximate from calibration
        params_init = [
            R_e_init, U_e_init, A_e_init,
            R_mu_init, U_mu_init, A_mu_init,
            R_tau_init, U_tau_init, A_tau_init,
            beta_initial, C_mu_init
        ]
        bounds = [
            (0.1, 1.0), (0.001, 3.0), (0.1, 1.0),  # electron
            (0.1, 1.0), (0.001, 3.0), (0.1, 1.0),  # muon
            (0.1, 1.0), (0.001, 3.0), (0.1, 1.0),  # tau
            (2.5, 3.5),                             # beta
            (500.0, 1500.0)                         # C_mu
        ]

    if verbose:
        print(f"\nStarting cross-lepton fit:")
        print(f"  Parameters: {len(params_init)}")
        print(f"  Analytic C_μ: {use_analytic_C_mu}")
        print(f"  Method: {method}")
        print(f"  Initial β: {beta_initial:.3f}")

    # Optimize
    result = minimize(
        lambda p: cross_lepton_objective(p, energy_calcs, use_analytic_C_mu),
        params_init,
        method=method,
        bounds=bounds,
        options={'maxiter': max_iter, 'ftol': 1e-9, 'gtol': 1e-8}
    )

    params_opt = result.x
    chi2_final = result.fun

    # Extract optimized parameters
    if use_analytic_C_mu:
        R_e, U_e, A_e, R_mu, U_mu, A_mu, R_tau, U_tau, A_tau, beta_opt = params_opt

        # Compute C_μ analytically
        mu_e = magnetic_moment_raw(R_e, U_e)
        mu_mu = magnetic_moment_raw(R_mu, U_mu)
        C_mu_opt = solve_C_mu_analytic(
            mu_e, mu_mu,
            LEPTON_G_FACTORS['electron'],
            LEPTON_G_FACTORS['muon'],
            SIGMA_G_MODEL
        )
    else:
        R_e, U_e, A_e, R_mu, U_mu, A_mu, R_tau, U_tau, A_tau, beta_opt, C_mu_opt = params_opt

    # Update energy calcs with optimal β
    for calc in energy_calcs.values():
        calc.beta = beta_opt

    # Compute final observables
    E_e, _, _ = energy_calcs['electron'].total_energy(R_e, U_e, A_e)
    E_mu, _, _ = energy_calcs['muon'].total_energy(R_mu, U_mu, A_mu)
    E_tau, _, _ = energy_calcs['tau'].total_energy(R_tau, U_tau, A_tau)

    mu_e = magnetic_moment_raw(R_e, U_e)
    mu_mu = magnetic_moment_raw(R_mu, U_mu)
    mu_tau = magnetic_moment_raw(R_tau, U_tau)

    g_e_pred = C_mu_opt * mu_e
    g_mu_pred = C_mu_opt * mu_mu
    g_tau_pred = C_mu_opt * mu_tau

    # Residuals
    mass_e_res = abs((E_e - LEPTON_MASSES['electron']) / LEPTON_MASSES['electron'])
    mass_mu_res = abs((E_mu - LEPTON_MASSES['muon']) / LEPTON_MASSES['muon'])
    mass_tau_res = abs((E_tau - LEPTON_MASSES['tau']) / LEPTON_MASSES['tau'])

    g_e_res = abs(g_e_pred - LEPTON_G_FACTORS['electron'])
    g_mu_res = abs(g_mu_pred - LEPTON_G_FACTORS['muon'])

    # Chi-squared contributions
    chi2_mass_e = (mass_e_res / SIGMA_MASS_MODEL)**2
    chi2_mass_mu = (mass_mu_res / SIGMA_MASS_MODEL)**2
    chi2_mass_tau = (mass_tau_res / SIGMA_MASS_MODEL)**2
    chi2_g_e = (g_e_res / SIGMA_G_MODEL)**2
    chi2_g_mu = (g_mu_res / SIGMA_G_MODEL)**2

    if verbose:
        print(f"\n{'='*70}")
        print(f"CROSS-LEPTON FIT RESULTS")
        print(f"{'='*70}")
        print(f"\nShared parameters:")
        print(f"  β = {beta_opt:.6f} (expected: 3.058)")
        print(f"  C_μ = {C_mu_opt:.2f}")
        print(f"\nElectron:")
        print(f"  R = {R_e:.6f}, U = {U_e:.6f}, A = {A_e:.6f}")
        print(f"  Mass: {E_e:.6f} (target: {LEPTON_MASSES['electron']:.6f}, res: {mass_e_res:.2e})")
        print(f"  g: {g_e_pred:.10f} (target: {LEPTON_G_FACTORS['electron']:.10f}, res: {g_e_res:.2e})")
        print(f"\nMuon:")
        print(f"  R = {R_mu:.6f}, U = {U_mu:.6f}, A = {A_mu:.6f}")
        print(f"  Mass: {E_mu:.6f} (target: {LEPTON_MASSES['muon']:.6f}, res: {mass_mu_res:.2e})")
        print(f"  g: {g_mu_pred:.10f} (target: {LEPTON_G_FACTORS['muon']:.10f}, res: {g_mu_res:.2e})")
        print(f"\nTau:")
        print(f"  R = {R_tau:.6f}, U = {U_tau:.6f}, A = {A_tau:.6f}")
        print(f"  Mass: {E_tau:.6f} (target: {LEPTON_MASSES['tau']:.6f}, res: {mass_tau_res:.2e})")
        print(f"\nChi-squared breakdown:")
        print(f"  χ²_mass (e): {chi2_mass_e:.2f}")
        print(f"  χ²_mass (μ): {chi2_mass_mu:.2f}")
        print(f"  χ²_mass (τ): {chi2_mass_tau:.2f}")
        print(f"  χ²_g (e): {chi2_g_e:.2f}")
        print(f"  χ²_g (μ): {chi2_g_mu:.2f}")
        print(f"  χ²_total: {chi2_final:.2f}")
        print(f"\nOptimizer:")
        print(f"  Success: {result.success}")
        print(f"  Iterations: {result.nit}")
        print(f"{'='*70}\n")

    return {
        'success': bool(result.success),
        'chi2_total': float(chi2_final),
        'beta': float(beta_opt),
        'C_mu': float(C_mu_opt),
        'electron': {
            'R': float(R_e),
            'U': float(U_e),
            'amplitude': float(A_e),
            'mass': float(E_e),
            'mass_residual_relative': float(mass_e_res),
            'g_factor': float(g_e_pred),
            'g_residual': float(g_e_res),
            'chi2_mass': float(chi2_mass_e),
            'chi2_g': float(chi2_g_e)
        },
        'muon': {
            'R': float(R_mu),
            'U': float(U_mu),
            'amplitude': float(A_mu),
            'mass': float(E_mu),
            'mass_residual_relative': float(mass_mu_res),
            'g_factor': float(g_mu_pred),
            'g_residual': float(g_mu_res),
            'chi2_mass': float(chi2_mass_mu),
            'chi2_g': float(chi2_g_mu)
        },
        'tau': {
            'R': float(R_tau),
            'U': float(U_tau),
            'amplitude': float(A_tau),
            'mass': float(E_tau),
            'mass_residual_relative': float(mass_tau_res),
            'g_factor': float(g_tau_pred),
            'chi2_mass': float(chi2_mass_tau)
        },
        'optimizer': {
            'iterations': int(result.nit),
            'message': str(result.message)
        }
    }


# ===========================================================================
# MULTI-START FIT
# ===========================================================================

def multi_start_cross_lepton_fit(n_starts=5, beta_range=(2.8, 3.3),
                                 use_analytic_C_mu=True, verbose=True):
    """
    Run cross-lepton fit with multiple random initial guesses.

    This helps avoid local minima and tests robustness.

    Args:
        n_starts: Number of random starts
        beta_range: Range for random beta initialization
        use_analytic_C_mu: Use analytic C_μ elimination
        verbose: Print progress

    Returns:
        List of results (sorted by chi2)
    """
    results = []

    if verbose:
        print(f"\n{'='*70}")
        print(f"MULTI-START CROSS-LEPTON FIT ({n_starts} starts)")
        print(f"{'='*70}")

    for i in range(n_starts):
        # Random initial β
        beta_init = np.random.uniform(*beta_range)

        if verbose:
            print(f"\n--- Start {i+1}/{n_starts}: β_init = {beta_init:.3f} ---")

        result = fit_cross_lepton(
            beta_initial=beta_init,
            use_analytic_C_mu=use_analytic_C_mu,
            verbose=False
        )

        result['start_index'] = i
        result['beta_initial'] = beta_init
        results.append(result)

        if verbose:
            status = "✓" if result['success'] else "✗"
            print(f"{status} χ² = {result['chi2_total']:.2f}, β = {result['beta']:.6f}")

    # Sort by chi2
    results.sort(key=lambda r: r['chi2_total'])

    if verbose:
        print(f"\n{'='*70}")
        print(f"BEST RESULT (lowest χ²):")
        print(f"{'='*70}")
        best = results[0]
        print(f"  β = {best['beta']:.6f} (offset from 3.058: {abs(best['beta'] - 3.058):.6f})")
        print(f"  C_μ = {best['C_mu']:.2f}")
        print(f"  χ²_total = {best['chi2_total']:.2f}")
        print(f"  From initial β = {best['beta_initial']:.3f}")
        print(f"{'='*70}\n")

    return results


# ===========================================================================
# SAVE RESULTS
# ===========================================================================

def save_results(results, filename='results/cross_lepton_fit.json'):
    """Save cross-lepton fit results to JSON."""
    output_path = Path(__file__).parent / filename
    output_path.parent.mkdir(exist_ok=True)

    output_data = {
        'test': 'Cross-Lepton Multi-Objective Fit',
        'purpose': 'Test if ONE shared β fits all three leptons simultaneously',
        'theory_uncertainties': {
            'sigma_mass_model_relative': SIGMA_MASS_MODEL,
            'sigma_g_model_absolute': SIGMA_G_MODEL
        },
        'timestamp': datetime.now().isoformat(),
        'results': results if isinstance(results, list) else [results]
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"✓ Results saved to {output_path}")
    return output_path


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Cross-lepton fit: ONE shared β for all three leptons'
    )
    parser.add_argument('--beta-init', type=float, default=3.058,
                       help='Initial β value')
    parser.add_argument('--multi-start', type=int, default=0,
                       help='Number of random starts (0 = single run)')
    parser.add_argument('--no-analytic-C-mu', action='store_true',
                       help='Do not use analytic C_μ elimination (fit it numerically)')
    parser.add_argument('--num-r', type=int, default=100,
                       help='Radial grid points')
    parser.add_argument('--num-theta', type=int, default=20,
                       help='Angular grid points')

    args = parser.parse_args()

    use_analytic = not args.no_analytic_C_mu

    if args.multi_start > 0:
        # Multi-start
        results = multi_start_cross_lepton_fit(
            n_starts=args.multi_start,
            use_analytic_C_mu=use_analytic,
            verbose=True
        )
        save_results(results)
    else:
        # Single run
        result = fit_cross_lepton(
            beta_initial=args.beta_init,
            use_analytic_C_mu=use_analytic,
            num_r=args.num_r,
            num_theta=args.num_theta,
            verbose=True
        )
        save_results(result)

    print("\n✓ Cross-lepton fit complete")
    print("  Next: Analyze if β emerges uniquely or if leptons want different β")
