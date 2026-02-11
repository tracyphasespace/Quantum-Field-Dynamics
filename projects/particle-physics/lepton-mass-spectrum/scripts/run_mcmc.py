#!/usr/bin/env python3
"""
MCMC parameter estimation for QFD lepton mass model.

This script estimates the vacuum stiffness parameters (β, ξ, τ) by fitting
predicted lepton masses to experimental data using Bayesian inference.
"""

import numpy as np
import emcee
import corner
import matplotlib.pyplot as plt
import json
from datetime import datetime
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
# Add QFD root for shared constants
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..'))
from qfd.shared_constants import BETA as BETA_GOLDEN
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

# Compton wavelengths
LAMBDA_COMPTON_E = HBARC / M_ELECTRON   # ≈ 386 fm
LAMBDA_COMPTON_MU = HBARC / M_MUON      # ≈ 1.87 fm
LAMBDA_COMPTON_TAU = HBARC / M_TAU      # ≈ 0.11 fm


class LeptonMassModel:
    """
    Model for charged lepton masses using Hill vortex profiles
    at Compton-scale radii with D-flow compression geometry.
    """

    def __init__(self):
        # Flow radii (Compton wavelengths)
        self.R_e_flow = LAMBDA_COMPTON_E
        self.R_mu_flow = LAMBDA_COMPTON_MU
        self.R_tau_flow = LAMBDA_COMPTON_TAU

        # Core radii (D-flow compression factor: 2/π)
        self.R_e_core = self.R_e_flow * (2/np.pi)
        self.R_mu_core = self.R_mu_flow * (2/np.pi)
        self.R_tau_core = self.R_tau_flow * (2/np.pi)

        print(f"Model initialized with Compton-scale radii:")
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
model = LeptonMassModel()


def log_prior(params):
    """Prior for (β, ξ, τ)."""
    beta, xi, tau = params

    # β prior (centered on Golden Loop prediction)
    beta_mean = BETA_GOLDEN
    beta_std = 0.15
    lp_beta = -0.5 * ((beta - beta_mean) / beta_std)**2

    # ξ prior (positive, log-normal)
    if xi <= 0:
        return -np.inf
    xi_log_std = 0.5
    lp_xi = -0.5 * (np.log(xi) / xi_log_std)**2 - np.log(xi * xi_log_std * np.sqrt(2*np.pi))

    # τ prior (positive, log-normal)
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


def run_mcmc(
    n_walkers=24,
    n_steps=1000,
    n_burn=200,
    output_dir='../results'
):
    """Run MCMC with Compton-scale radii."""
    os.makedirs(output_dir, exist_ok=True)

    n_dim = 3

    print("="*70)
    print("MCMC Parameter Estimation: QFD Lepton Mass Model")
    print("="*70)
    print()
    print(f"Parameters: {n_dim}D (β, ξ, τ)")
    print(f"Walkers: {n_walkers}")
    print(f"Steps: {n_steps} (+ {n_burn} burn-in)")
    print()

    # Initialize walkers
    beta_init = BETA_GOLDEN
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
    output_file = os.path.join(output_dir, 'mcmc_chain.h5')
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
    print("MCMC Complete")
    print(f"Acceptance fraction: {np.mean(sampler.acceptance_fraction):.3f}")
    print("="*70)
    print()

    return sampler


def analyze_results(sampler, output_dir='../results'):
    """Analyze MCMC results."""
    os.makedirs(output_dir, exist_ok=True)

    samples = sampler.get_chain(flat=True)

    print("="*70)
    print("Posterior Summary")
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

        print(f"{name}: {median:.4f} ± {std:.4f}")
        print(f"   [{q16:.4f}, {q84:.4f}] (68% CI)")
        print()

    # Correlation analysis
    beta_samples = samples[:, 0]
    xi_samples = samples[:, 1]
    corr_beta_xi = np.corrcoef(beta_samples, xi_samples)[0, 1]
    results['correlation_beta_xi'] = float(corr_beta_xi)

    print(f"β-ξ correlation: {corr_beta_xi:.4f}")
    print()

    # Corner plot
    fig = corner.corner(
        samples,
        labels=['β', 'ξ', 'τ'],
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_fmt='.3f'
    )
    fig.savefig(os.path.join(output_dir, 'corner_plot.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/corner_plot.png")

    # Save results
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Saved: {output_dir}/results.json")
    print()

    return results


if __name__ == '__main__':
    # Run MCMC
    sampler = run_mcmc(
        n_walkers=24,
        n_steps=1000,
        n_burn=200,
        output_dir='../results'
    )

    # Analyze results
    results = analyze_results(sampler, output_dir='../results')

    print("\n" + "="*70)
    print("Analysis Complete")
    print("="*70)
