#!/usr/bin/env python3
"""
D-Flow Model: Compton Scale with π/2 Geometry

CORRECTED electron scale:
- R_e,flow ~ 386 fm (Compton wavelength ℏ/(m_e c))
- R_e,core = R_flow × (2/π) ≈ 246 fm (D-flow compression)

NOT:
- Classical radius (2.82 fm) - too small
- Proton radius (0.84 fm) - WRONG PARTICLE!

Test: Does correct scale + D-flow geometry → β = 3.058?
"""

import numpy as np
import emcee
import corner
import matplotlib.pyplot as plt
import json
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import functionals as func
import solvers as solv


# Experimental masses (MeV)
M_ELECTRON = 0.5110
M_MUON = 105.658
M_TAU = 1776.86

# Experimental uncertainties
SIGMA_ELECTRON = 1e-6
SIGMA_MUON = 1e-3
SIGMA_TAU = 0.12

# Model uncertainty
SIGMA_MODEL_ELECTRON = 1e-3
SIGMA_MODEL_MUON = 0.1
SIGMA_MODEL_TAU = 2.0

# Physical constants
HBARC = 197.33  # MeV·fm

# Compton wavelengths (CORRECT SCALE!)
LAMBDA_COMPTON_E = HBARC / M_ELECTRON   # ≈ 386 fm
LAMBDA_COMPTON_MU = HBARC / M_MUON      # ≈ 1.87 fm
LAMBDA_COMPTON_TAU = HBARC / M_TAU      # ≈ 0.11 fm

# D-flow compression factor
PI_OVER_2 = np.pi / 2  # ≈ 1.5708


class ComptonScaleModel:
    """
    Model with CORRECT Compton-scale radii.

    R_flow = ℏ/(mc) for each lepton
    R_core = R_flow × (2/π) from D-flow geometry

    Free parameters: (β, ξ, τ)
    """

    def __init__(self):
        # Flow radii (Compton wavelengths)
        self.R_e_flow = LAMBDA_COMPTON_E
        self.R_mu_flow = LAMBDA_COMPTON_MU
        self.R_tau_flow = LAMBDA_COMPTON_TAU

        # Core radii (D-flow compression)
        self.R_e_core = self.R_e_flow * (2/np.pi)
        self.R_mu_core = self.R_mu_flow * (2/np.pi)
        self.R_tau_core = self.R_tau_flow * (2/np.pi)

        print(f"Compton Scale Radii:")
        print(f"  Electron: R_flow = {self.R_e_flow:.1f} fm, R_core = {self.R_e_core:.1f} fm")
        print(f"  Muon:     R_flow = {self.R_mu_flow:.2f} fm, R_core = {self.R_mu_core:.2f} fm")
        print(f"  Tau:      R_flow = {self.R_tau_flow:.3f} fm, R_core = {self.R_tau_core:.3f} fm")
        print()

        # Fixed flow parameters
        self.U_e = 0.5
        self.A_e = 1.0
        self.U_mu = 0.5
        self.A_mu = 1.0
        self.U_tau = 0.5
        self.A_tau = 1.0

        # Grid (in units of R_flow)
        self.r_max = 10.0
        self.n_points = 300

        # Build profiles
        self.r_e = np.linspace(0, self.r_max * self.R_e_flow, self.n_points)
        self.r_mu = np.linspace(0, self.r_max * self.R_mu_flow, self.n_points)
        self.r_tau = np.linspace(0, self.r_max * self.R_tau_flow, self.n_points)

        self.rho_e = solv.hill_vortex_profile(self.r_e, self.R_e_flow, self.U_e, self.A_e)
        self.rho_mu = solv.hill_vortex_profile(self.r_mu, self.R_mu_flow, self.U_mu, self.A_mu)
        self.rho_tau = solv.hill_vortex_profile(self.r_tau, self.R_tau_flow, self.U_tau, self.A_tau)

    def compute_energy(self, beta, xi, tau, lepton='electron'):
        """Compute energy with Compton-scale radii."""
        if lepton == 'electron':
            r, rho = self.r_e, self.rho_e
        elif lepton == 'muon':
            r, rho = self.r_mu, self.rho_mu
        elif lepton == 'tau':
            r, rho = self.r_tau, self.rho_tau
        else:
            raise ValueError(f"Unknown lepton: {lepton}")

        # Standard energy functional
        E_total, E_grad, E_comp = func.gradient_energy_functional(rho, r, xi, beta)

        # Temporal correction
        tau_correction = 1.0 + 0.01 * (tau / beta)
        E_total_with_tau = E_total * tau_correction

        return E_total_with_tau

    def compute_masses(self, beta, xi, tau):
        """Compute predicted masses."""
        E_e = self.compute_energy(beta, xi, tau, 'electron')
        E_mu = self.compute_energy(beta, xi, tau, 'muon')
        E_tau = self.compute_energy(beta, xi, tau, 'tau')

        # Normalize to electron mass
        norm = M_ELECTRON / E_e

        m_e_pred = M_ELECTRON
        m_mu_pred = E_mu * norm
        m_tau_pred = E_tau * norm

        return m_e_pred, m_mu_pred, m_tau_pred


# Global model
model = ComptonScaleModel()


def log_prior(params):
    """Prior for (β, ξ, τ)."""
    beta, xi, tau = params

    # β prior
    beta_mean = 3.058
    beta_std = 0.15
    lp_beta = -0.5 * ((beta - beta_mean) / beta_std)**2

    # ξ prior (positive)
    if xi <= 0:
        return -np.inf
    xi_log_std = 0.5
    lp_xi = -0.5 * (np.log(xi) / xi_log_std)**2 - np.log(xi * xi_log_std * np.sqrt(2*np.pi))

    # τ prior (positive)
    if tau <= 0:
        return -np.inf
    tau_log_std = 0.5
    lp_tau = -0.5 * (np.log(tau) / tau_log_std)**2 - np.log(tau * tau_log_std * np.sqrt(2*np.pi))

    return lp_beta + lp_xi + lp_tau


def log_likelihood(params):
    """Likelihood for masses."""
    beta, xi, tau = params

    try:
        m_e_pred, m_mu_pred, m_tau_pred = model.compute_masses(beta, xi, tau)
    except Exception as e:
        return -1e10

    if not np.isfinite([m_e_pred, m_mu_pred, m_tau_pred]).all():
        return -1e10

    if m_e_pred <= 0 or m_mu_pred <= 0 or m_tau_pred <= 0:
        return -1e10

    # Combined uncertainties
    sigma_e = np.sqrt(SIGMA_ELECTRON**2 + SIGMA_MODEL_ELECTRON**2)
    sigma_mu = np.sqrt(SIGMA_MUON**2 + SIGMA_MODEL_MUON**2)
    sigma_tau = np.sqrt(SIGMA_TAU**2 + SIGMA_MODEL_TAU**2)

    # Chi-squared
    chi2_e = ((m_e_pred - M_ELECTRON) / sigma_e)**2
    chi2_mu = ((m_mu_pred - M_MUON) / sigma_mu)**2
    chi2_tau = ((m_tau_pred - M_TAU) / sigma_tau)**2

    log_L = -0.5 * (chi2_e + chi2_mu + chi2_tau)

    return log_L


def log_probability(params):
    """Posterior = prior × likelihood."""
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf

    ll = log_likelihood(params)
    if not np.isfinite(ll):
        return -np.inf

    return lp + ll


def run_compton_mcmc(
    n_walkers=24,
    n_steps=1000,
    n_burn=200,
    output_file='results/mcmc_compton_scale.h5'
):
    """Run MCMC with Compton-scale radii."""
    n_dim = 3

    print("="*70)
    print("COMPTON SCALE MCMC: Correct Electron Radius")
    print("="*70)
    print()
    print(f"R_e,flow = {LAMBDA_COMPTON_E:.1f} fm (Compton wavelength)")
    print(f"R_e,core = {LAMBDA_COMPTON_E * 2/np.pi:.1f} fm (D-flow compression)")
    print()
    print(f"Parameters: {n_dim}D (β, ξ, τ)")
    print(f"Walkers: {n_walkers}")
    print(f"Steps: {n_steps} (+ {n_burn} burn-in)")
    print()

    # Initialize walkers
    beta_init = 3.058
    xi_init = 1.0
    tau_init = 1.0

    pos = np.array([beta_init, xi_init, tau_init]) + 0.1 * np.random.randn(n_walkers, n_dim)

    # Ensure positivity
    pos[:, 1] = np.abs(pos[:, 1])
    pos[:, 2] = np.abs(pos[:, 2])

    print(f"Initial positions:")
    print(f"  β: {pos[:, 0].min():.3f} - {pos[:, 0].max():.3f}")
    print(f"  ξ: {pos[:, 1].min():.3f} - {pos[:, 1].max():.3f}")
    print(f"  τ: {pos[:, 2].min():.3f} - {pos[:, 2].max():.3f}")
    print()

    # Backend
    backend = emcee.backends.HDFBackend(output_file)
    backend.reset(n_walkers, n_dim)

    # Sampler
    sampler = emcee.EnsembleSampler(
        n_walkers, n_dim, log_probability,
        backend=backend
    )

    print(f"Running burn-in ({n_burn} steps)...")
    state = sampler.run_mcmc(pos, n_burn, progress=True)
    sampler.reset()

    print(f"\nRunning production ({n_steps} steps)...")
    sampler.run_mcmc(state, n_steps, progress=True)

    print("\n" + "="*70)
    print("MCMC COMPLETE!")
    print(f"Acceptance: {np.mean(sampler.acceptance_fraction):.3f}")
    print("="*70)
    print()

    return sampler


def analyze_compton_results(sampler, output_dir='results'):
    """Analyze Compton-scale results."""
    os.makedirs(output_dir, exist_ok=True)

    samples = sampler.get_chain(flat=True)

    print("="*70)
    print("COMPTON SCALE POSTERIOR SUMMARY")
    print("="*70)
    print()

    param_names = ['β', 'ξ', 'τ']
    results = {
        'timestamp': datetime.now().isoformat(),
        'R_e_flow_fm': float(LAMBDA_COMPTON_E),
        'R_e_core_fm': float(LAMBDA_COMPTON_E * 2/np.pi)
    }

    for i, name in enumerate(param_names):
        median = np.median(samples[:, i])
        std = np.std(samples[:, i])
        q16, q84 = np.percentile(samples[:, i], [16, 84])

        results[name] = {
            'median': float(median),
            'std': float(std),
            'q16': float(q16),
            'q84': float(q84)
        }

        print(f"{name} posterior: {median:.4f} ± {std:.4f}")
        print(f"             [{q16:.4f}, {q84:.4f}] (68% CI)")
        print()

    # Critical assessment
    beta_median = results['β']['median']
    beta_std = results['β']['std']
    xi_median = results['ξ']['median']
    beta_target = 3.058
    beta_offset = abs(beta_median - beta_target)

    # Check correlation
    beta_samples = samples[:, 0]
    xi_samples = samples[:, 1]
    corr_beta_xi = np.corrcoef(beta_samples, xi_samples)[0, 1]

    print("="*70)
    print("BREAKTHROUGH TEST: Compton Scale")
    print("="*70)
    print(f"β target (Golden Loop): {beta_target:.4f}")
    print(f"β Compton scale:        {beta_median:.4f} ± {beta_std:.4f}")
    print()
    print(f"Offset from target:     {beta_offset:.4f} ({100*beta_offset/beta_target:.2f}%)")
    print(f"β-ξ correlation:        {corr_beta_xi:.4f}")
    print()

    print(f"ξ comparison:")
    print(f"  Stage 2 (wrong scale): ξ = 25.98 ± 1.30")
    print(f"  Compton (correct):     ξ = {xi_median:.2f} ± {results['ξ']['std']:.2f}")
    print()

    if beta_offset < 0.02 and abs(corr_beta_xi) < 0.3:
        print("✅ BREAKTHROUGH! Compton scale → β = 3.058!")
        print("   Correct electron radius broke degeneracy!")
        result = "success"
    elif beta_offset < 0.05:
        print("⚠️ PARTIAL: Improved, need spin constraint")
        result = "partial"
    else:
        print("❌ Still offset - need D-flow geometry + spin")
        result = "failure"

    print("="*70)
    print()

    # Corner plot
    fig = corner.corner(
        samples,
        labels=['β', 'ξ', 'τ'],
        truths=[3.058, 1.0, 1.0],
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_fmt='.3f'
    )
    fig.suptitle(f'Compton Scale: R_e = {LAMBDA_COMPTON_E:.0f} fm',
                 fontsize=12, fontweight='bold', y=1.0)
    fig.savefig(f'{output_dir}/mcmc_compton_corner.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir}/mcmc_compton_corner.png")

    # Save results
    results['result'] = result
    results['beta_target'] = beta_target
    results['beta_offset'] = float(beta_offset)
    results['correlation_beta_xi'] = float(corr_beta_xi)

    with open(f'{output_dir}/mcmc_compton_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✓ Saved: {output_dir}/mcmc_compton_results.json")
    print()

    return results


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)

    print("\n" + "="*70)
    print("D-FLOW BREAKTHROUGH TEST")
    print("="*70)
    print("\nHypothesis: Correct Compton scale → β = 3.058")
    print("  Previous: R = 0.84 fm (proton!) → ξ collapsed")
    print("  Corrected: R = 386 fm (electron Compton) → should work!")
    print()

    # Run
    sampler = run_compton_mcmc()

    # Analyze
    results = analyze_compton_results(sampler)

    print("\n" + "="*70)
    print("COMPTON SCALE TEST COMPLETE!")
    print("="*70)
