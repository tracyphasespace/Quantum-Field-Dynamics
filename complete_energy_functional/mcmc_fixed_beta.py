#!/usr/bin/env python3
"""
MCMC with Fixed β: Test Golden Loop Hypothesis

Hypothesis: β = 3.043233053 (from α-constraint) is exact
            → Only fit (ξ, τ) to lepton masses

If this works:
  - Validates Golden Loop prediction
  - Confirms β-ξ degeneracy is real
  - Isolates ξ and τ uniquely

If this fails:
  - β ≠ 3.043233053 in reality
  - Need electromagnetic constraints (Option 2)
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

# FIXED BETA (Golden Loop prediction)
BETA_FIXED = 3.043233053


class FixedBetaModel:
    """
    Model with β FIXED at Golden Loop value.

    Free parameters: (ξ, τ) shared across leptons
    Fixed parameters: β = 3.043233053 (Golden Loop)
                      (R, U, A) per lepton from scaling
    """

    def __init__(self, beta=BETA_FIXED):
        self.beta = beta  # FIXED

        # Fixed geometry
        self.R_e = 1.0
        self.U_e = 0.5
        self.A_e = 1.0

        mass_ratio_mu = M_MUON / M_ELECTRON
        mass_ratio_tau = M_TAU / M_MUON

        self.R_mu = self.R_e * np.sqrt(mass_ratio_mu)
        self.U_mu = self.U_e
        self.A_mu = self.A_e

        self.R_tau = self.R_mu * np.sqrt(mass_ratio_tau)
        self.U_tau = self.U_e
        self.A_tau = self.A_e

        # Grid
        self.r_max = 10.0
        self.n_points = 300

        # Precompute profiles
        self.r_e = np.linspace(0, self.r_max * self.R_e, self.n_points)
        self.r_mu = np.linspace(0, self.r_max * self.R_mu, self.n_points)
        self.r_tau = np.linspace(0, self.r_max * self.R_tau, self.n_points)

        self.rho_e = solv.hill_vortex_profile(self.r_e, self.R_e, self.U_e, self.A_e)
        self.rho_mu = solv.hill_vortex_profile(self.r_mu, self.R_mu, self.U_mu, self.A_mu)
        self.rho_tau = solv.hill_vortex_profile(self.r_tau, self.R_tau, self.U_tau, self.A_tau)

    def compute_energy(self, xi, tau, lepton='electron'):
        """Compute energy with FIXED β."""
        if lepton == 'electron':
            r, rho = self.r_e, self.rho_e
        elif lepton == 'muon':
            r, rho = self.r_mu, self.rho_mu
        elif lepton == 'tau':
            r, rho = self.r_tau, self.rho_tau
        else:
            raise ValueError(f"Unknown lepton: {lepton}")

        # Static energy with fixed β
        E_total, E_grad, E_comp = func.gradient_energy_functional(rho, r, xi, self.beta)

        # Temporal correction
        tau_correction_factor = 1.0 + 0.01 * (tau / self.beta)
        E_total_with_tau = E_total * tau_correction_factor

        return E_total_with_tau

    def compute_masses(self, xi, tau):
        """Compute predicted masses with fixed β."""
        E_e = self.compute_energy(xi, tau, 'electron')
        E_mu = self.compute_energy(xi, tau, 'muon')
        E_tau = self.compute_energy(xi, tau, 'tau')

        # Normalize to electron mass
        norm = M_ELECTRON / E_e

        m_e_pred = M_ELECTRON
        m_mu_pred = E_mu * norm
        m_tau_pred = E_tau * norm

        return m_e_pred, m_mu_pred, m_tau_pred


# Global model
model = FixedBetaModel()


def log_prior(params):
    """
    Prior for (ξ, τ) with β FIXED.

    ξ ~ LogNormal(0, 0.5)   # Expect ξ ~ 1-30
    τ ~ LogNormal(0, 0.5)   # Expect τ ~ 1
    """
    xi, tau = params

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

    return lp_xi + lp_tau


def log_likelihood(params):
    """Likelihood for masses with FIXED β."""
    xi, tau = params

    try:
        m_e_pred, m_mu_pred, m_tau_pred = model.compute_masses(xi, tau)
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


def run_fixed_beta_mcmc(
    n_walkers=16,
    n_steps=1000,
    n_burn=200,
    output_file='results/mcmc_fixed_beta.h5'
):
    """
    Run MCMC with β FIXED at Golden Loop value.

    Parameters
    ----------
    n_walkers : int
        Number of walkers (recommend 8× n_dim = 16)
    n_steps : int
        Production steps
    n_burn : int
        Burn-in steps
    """
    n_dim = 2  # Only (ξ, τ)

    print("="*70)
    print("MCMC WITH FIXED β (Golden Loop Test)")
    print("="*70)
    print()
    print(f"β = {BETA_FIXED} (FIXED - from α-constraint)")
    print(f"Parameters: {n_dim}D (ξ, τ)")
    print(f"Walkers: {n_walkers}")
    print(f"Steps: {n_steps} (+ {n_burn} burn-in)")
    print()

    # Initialize walkers
    xi_init = 30.0  # Expect ξ ~ 30 from Stage 2
    tau_init = 1.0

    pos = np.array([xi_init, tau_init]) + 0.1 * np.random.randn(n_walkers, n_dim)

    # Ensure positivity
    pos[:, 0] = np.abs(pos[:, 0])
    pos[:, 1] = np.abs(pos[:, 1])

    print(f"Initial positions:")
    print(f"  ξ: {pos[:, 0].min():.3f} - {pos[:, 0].max():.3f}")
    print(f"  τ: {pos[:, 1].min():.3f} - {pos[:, 1].max():.3f}")
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


def analyze_fixed_beta_results(sampler, output_dir='results'):
    """Analyze results with fixed β."""
    os.makedirs(output_dir, exist_ok=True)

    samples = sampler.get_chain(flat=True)

    print("="*70)
    print("FIXED β POSTERIOR SUMMARY")
    print("="*70)
    print()
    print(f"β = {BETA_FIXED} (FIXED)")
    print()

    param_names = ['ξ', 'τ']
    results = {
        'timestamp': datetime.now().isoformat(),
        'beta_fixed': BETA_FIXED
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
    xi_median = results['ξ']['median']
    tau_median = results['τ']['median']

    # Compute β_eff from Stage 2
    beta_stage2 = 2.9617
    xi_stage2 = 25.979
    c_coeff = (3.15 - beta_stage2) / xi_stage2  # ≈ 0.007

    beta_eff = BETA_FIXED + c_coeff * xi_median

    print("="*70)
    print("GOLDEN LOOP HYPOTHESIS TEST")
    print("="*70)
    print(f"Hypothesis: β = {BETA_FIXED} (from α-constraint)")
    print()
    print(f"ξ fitted: {xi_median:.2f} ± {results['ξ']['std']:.2f}")
    print(f"τ fitted: {tau_median:.2f} ± {results['τ']['std']:.2f}")
    print()
    print(f"Effective β_eff = β + c·ξ")
    print(f"               = {BETA_FIXED:.3f} + {c_coeff:.4f}×{xi_median:.2f}")
    print(f"               = {beta_eff:.3f}")
    print()
    print(f"V22 target: β_eff ≈ 3.15")
    print(f"Our fit:    β_eff = {beta_eff:.3f}")
    print(f"Match:      {abs(beta_eff - 3.15) < 0.05}")
    print()

    # Test likelihood
    test_params = [xi_median, tau_median]
    log_L = log_likelihood(test_params)
    m_e, m_mu, m_tau = model.compute_masses(xi_median, tau_median)

    print(f"Log-likelihood: {log_L:.2f}")
    print()
    print(f"Predicted masses:")
    print(f"  m_e:  {m_e:.4f} MeV (obs: {M_ELECTRON:.4f})")
    print(f"  m_μ:  {m_mu:.2f} MeV (obs: {M_MUON:.2f})")
    print(f"  m_τ:  {m_tau:.2f} MeV (obs: {M_TAU:.2f})")
    print()

    chi2_e = ((m_e - M_ELECTRON) / 1e-3)**2
    chi2_mu = ((m_mu - M_MUON) / 0.1)**2
    chi2_tau = ((m_tau - M_TAU) / 2.0)**2
    chi2_total = chi2_e + chi2_mu + chi2_tau

    print(f"χ² breakdown:")
    print(f"  Electron: {chi2_e:.2e}")
    print(f"  Muon:     {chi2_mu:.2e}")
    print(f"  Tau:      {chi2_tau:.2e}")
    print(f"  Total:    {chi2_total:.2e}")
    print()

    if log_L > -10:
        print("✅ SUCCESS! β = 3.043233053 fits lepton masses perfectly!")
        print("   Golden Loop prediction VALIDATED!")
        print("   β-ξ degeneracy confirmed real.")
        result = "success"
    elif log_L > -100:
        print("⚠️ PARTIAL: Reasonable fit but not perfect")
        print("   May need fine-tuning or additional terms")
        result = "partial"
    else:
        print("❌ FAILURE: β = 3.043233053 does not fit masses")
        print("   Need electromagnetic constraints (Option 2)")
        result = "failure"

    print("="*70)
    print()

    # Corner plot
    fig = corner.corner(
        samples,
        labels=['ξ', 'τ'],
        truths=[xi_median, 1.0],
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_fmt='.3f'
    )
    fig.suptitle(f'Fixed β = {BETA_FIXED}: (ξ, τ) Posterior',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.savefig(f'{output_dir}/mcmc_fixed_beta_corner.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir}/mcmc_fixed_beta_corner.png")

    # Traces
    chain = sampler.get_chain()
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    for i, (ax, label) in enumerate(zip(axes, ['ξ', 'τ'])):
        ax.plot(chain[:, :, i], alpha=0.3, color='k', lw=0.5)
        ax.set_ylabel(label, fontsize=12)
        ax.grid(alpha=0.3)

    axes[0].set_title(f'β = {BETA_FIXED} (FIXED)', fontsize=10, color='red')
    axes[-1].set_xlabel('Step', fontsize=12)
    fig.suptitle('Fixed β Traces', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(f'{output_dir}/mcmc_fixed_beta_traces.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir}/mcmc_fixed_beta_traces.png")

    # Save results
    results['result'] = result
    results['beta_eff'] = float(beta_eff)
    results['log_likelihood'] = float(log_L)
    results['chi2_total'] = float(chi2_total)
    results['masses_predicted'] = {
        'm_e': float(m_e),
        'm_mu': float(m_mu),
        'm_tau': float(m_tau)
    }

    with open(f'{output_dir}/mcmc_fixed_beta_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✓ Saved: {output_dir}/mcmc_fixed_beta_results.json")
    print()

    return results


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)

    print("\n" + "="*70)
    print("GOLDEN LOOP HYPOTHESIS TEST")
    print("="*70)
    print(f"\nTesting: Does β = {BETA_FIXED} (Golden Loop) fit lepton masses?")
    print()

    # Run
    sampler = run_fixed_beta_mcmc()

    # Analyze
    results = analyze_fixed_beta_results(sampler)

    print("\n" + "="*70)
    print("TEST COMPLETE!")
    print("="*70)
