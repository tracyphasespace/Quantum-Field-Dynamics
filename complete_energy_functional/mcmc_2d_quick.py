#!/usr/bin/env python3
"""
Quick 2D MCMC: Fit only (β, ξ) with geometry fixed

Goal: Test if β → 3.043233053 when gradient term ξ is included

Strategy:
- Fix geometry (R, U, A) from V22 or physical estimates
- Fit only β and ξ (2D parameter space)
- Normalize energy to match electron mass
- Run ~1000 steps (fast - minutes not hours)
"""

import numpy as np
import emcee
import corner
import matplotlib.pyplot as plt
from multiprocessing import Pool
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

# Model uncertainty (from V22)
SIGMA_MODEL_ELECTRON = 1e-3
SIGMA_MODEL_MUON = 0.1
SIGMA_MODEL_TAU = 2.0


class QuickModel:
    """
    Simplified model with fixed geometry.

    Free parameters: (β, ξ) shared across leptons
    Fixed parameters: (R, U, A) per lepton from V22 or scaling
    """

    def __init__(self):
        # Fixed geometry (normalized units)
        # These are placeholders - should be calibrated from V22 or Koide
        self.R_e = 1.0    # Electron vortex radius (normalized)
        self.U_e = 0.5    # Electron velocity (fraction of c)
        self.A_e = 1.0    # Electron amplitude

        # Scale for heavier leptons (geometric scaling)
        mass_ratio_mu = M_MUON / M_ELECTRON
        mass_ratio_tau = M_TAU / M_MUON

        self.R_mu = self.R_e * np.sqrt(mass_ratio_mu)
        self.U_mu = self.U_e  # Keep same velocity
        self.A_mu = self.A_e  # Keep same amplitude

        self.R_tau = self.R_mu * np.sqrt(mass_ratio_tau)
        self.U_tau = self.U_e
        self.A_tau = self.A_e

        # Radial grid
        self.r_max = 10.0
        self.n_points = 300

        # Precompute density profiles (fixed)
        self.r_e = np.linspace(0, self.r_max * self.R_e, self.n_points)
        self.r_mu = np.linspace(0, self.r_max * self.R_mu, self.n_points)
        self.r_tau = np.linspace(0, self.r_max * self.R_tau, self.n_points)

        self.rho_e = solv.hill_vortex_profile(self.r_e, self.R_e, self.U_e, self.A_e)
        self.rho_mu = solv.hill_vortex_profile(self.r_mu, self.R_mu, self.U_mu, self.A_mu)
        self.rho_tau = solv.hill_vortex_profile(self.r_tau, self.R_tau, self.U_tau, self.A_tau)

    def compute_energy(self, beta, xi, lepton='electron'):
        """Compute energy for given (β, ξ)"""
        if lepton == 'electron':
            r, rho = self.r_e, self.rho_e
        elif lepton == 'muon':
            r, rho = self.r_mu, self.rho_mu
        elif lepton == 'tau':
            r, rho = self.r_tau, self.rho_tau
        else:
            raise ValueError(f"Unknown lepton: {lepton}")

        E_total, _, _ = func.gradient_energy_functional(rho, r, xi, beta)
        return E_total

    def compute_masses(self, beta, xi):
        """
        Compute predicted masses for all three leptons.

        Strategy: Compute energies, normalize to electron mass.
        """
        E_e = self.compute_energy(beta, xi, 'electron')
        E_mu = self.compute_energy(beta, xi, 'muon')
        E_tau = self.compute_energy(beta, xi, 'tau')

        # Normalize to electron mass (this handles unit conversion)
        # m_e_pred = M_ELECTRON by construction
        # Other masses scale as E_mu/E_e etc.
        norm = M_ELECTRON / E_e

        m_e_pred = M_ELECTRON
        m_mu_pred = E_mu * norm
        m_tau_pred = E_tau * norm

        return m_e_pred, m_mu_pred, m_tau_pred


# Global model instance
model = QuickModel()


def log_prior(params):
    """
    Prior for (β, ξ).

    β ~ Normal(3.043233053, 0.15)  # From α-constraint
    ξ ~ LogNormal(0, 0.5)     # Median=1, range 0.3-3
    """
    beta, xi = params

    # β prior
    beta_mean = 3.043233053
    beta_std = 0.15
    lp_beta = -0.5 * ((beta - beta_mean) / beta_std)**2

    # ξ prior (must be positive)
    if xi <= 0:
        return -np.inf

    xi_log_std = 0.5
    lp_xi = -0.5 * (np.log(xi) / xi_log_std)**2 - np.log(xi * xi_log_std * np.sqrt(2*np.pi))

    return lp_beta + lp_xi


def log_likelihood(params):
    """
    Likelihood for lepton masses.

    L(data | β, ξ) = Π_i N(m_i^obs | m_i^pred(β,ξ), σ_i)
    """
    beta, xi = params

    try:
        # Predict masses
        m_e_pred, m_mu_pred, m_tau_pred = model.compute_masses(beta, xi)

    except Exception as e:
        # Computation failed
        return -1e10

    # Check for invalid predictions
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
    """Posterior = prior × likelihood"""
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf

    ll = log_likelihood(params)
    if not np.isfinite(ll):
        return -np.inf

    return lp + ll


def run_quick_mcmc(
    n_walkers=16,
    n_steps=1000,
    n_burn=200,
    output_file='results/mcmc_2d_quick.h5'
):
    """
    Run quick 2D MCMC.

    Parameters
    ----------
    n_walkers : int
        Number of walkers (recommend 8× n_dim = 16)
    n_steps : int
        Production steps
    n_burn : int
        Burn-in steps
    """
    n_dim = 2

    print("="*70)
    print("QUICK 2D MCMC: (β, ξ) Only")
    print("="*70)
    print()
    print(f"Parameters: {n_dim}D (β, ξ)")
    print(f"Walkers: {n_walkers}")
    print(f"Steps: {n_steps} (+ {n_burn} burn-in)")
    print(f"Geometry: FIXED (R, U, A from scaling)")
    print()

    # Initialize walkers near expected values
    beta_init = 3.043233053
    xi_init = 1.0

    pos = np.array([beta_init, xi_init]) + 0.1 * np.random.randn(n_walkers, n_dim)

    # Ensure positivity for ξ
    pos[:, 1] = np.abs(pos[:, 1])

    print(f"Initial walker positions:")
    print(f"  β: {pos[:, 0].min():.3f} - {pos[:, 0].max():.3f}")
    print(f"  ξ: {pos[:, 1].min():.3f} - {pos[:, 1].max():.3f}")
    print()

    # Set up backend
    backend = emcee.backends.HDFBackend(output_file)
    backend.reset(n_walkers, n_dim)

    # Run MCMC
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
    print(f"Acceptance fraction: {np.mean(sampler.acceptance_fraction):.3f}")
    print("="*70)
    print()

    return sampler


def analyze_results(sampler, output_dir='results'):
    """Analyze and visualize results."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Get samples
    samples = sampler.get_chain(flat=True)

    print("="*70)
    print("POSTERIOR SUMMARY")
    print("="*70)
    print()

    # Statistics
    beta_samples = samples[:, 0]
    xi_samples = samples[:, 1]

    beta_median = np.median(beta_samples)
    beta_std = np.std(beta_samples)
    beta_16, beta_84 = np.percentile(beta_samples, [16, 84])

    xi_median = np.median(xi_samples)
    xi_std = np.std(xi_samples)
    xi_16, xi_84 = np.percentile(xi_samples, [16, 84])

    print(f"β posterior: {beta_median:.4f} ± {beta_std:.4f}")
    print(f"             [{beta_16:.4f}, {beta_84:.4f}] (68% CI)")
    print()
    print(f"ξ posterior: {xi_median:.4f} ± {xi_std:.4f}")
    print(f"             [{xi_16:.4f}, {xi_84:.4f}] (68% CI)")
    print()

    # Critical question
    beta_target = 3.043233053
    beta_v22 = 3.15
    beta_offset = abs(beta_median - beta_target)

    print("="*70)
    print("CRITICAL QUESTION: Did gradient term resolve β offset?")
    print("="*70)
    print(f"β Golden Loop (target): {beta_target:.4f}")
    print(f"β V22 (no gradient):    {beta_v22:.4f}")
    print(f"β posterior (with ξ):   {beta_median:.4f} ± {beta_std:.4f}")
    print()
    print(f"Offset from target:     {beta_offset:.4f} ({100*beta_offset/beta_target:.2f}%)")
    print()

    if beta_offset < 0.02:
        print("✅ SUCCESS! β converged to Golden Loop prediction!")
        print("   Gradient term RESOLVES the V22 offset!")
        result = "success"
    elif beta_offset < 0.05:
        print("⚠️ PARTIAL: β improved but still ~2-5% offset")
        print("   May need Stage 2 (temporal term)")
        result = "partial"
    else:
        print("❌ FAILURE: β offset persists")
        print("   Need Stage 2 (temporal) or Stage 3 (EM functional)")
        result = "failure"

    print("="*70)
    print()

    # Corner plot
    fig = corner.corner(
        samples,
        labels=['β', 'ξ'],
        truths=[3.043233053, 1.0],
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_fmt='.4f',
        title_kwargs={'fontsize': 12}
    )
    fig.suptitle('2D MCMC Posterior: (β, ξ)', fontsize=14, fontweight='bold')
    fig.savefig(f'{output_dir}/mcmc_2d_corner.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir}/mcmc_2d_corner.png")

    # Trace plots
    chain = sampler.get_chain()
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    for i, (ax, label, target) in enumerate(zip(axes, ['β', 'ξ'], [3.043233053, 1.0])):
        ax.plot(chain[:, :, i], alpha=0.3, color='k', lw=0.5)
        ax.axhline(target, color='r', ls='--', lw=2, label=f'Expected: {target}')
        ax.set_ylabel(label, fontsize=12)
        ax.legend()
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel('Step', fontsize=12)
    fig.suptitle('MCMC Traces', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(f'{output_dir}/mcmc_2d_traces.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir}/mcmc_2d_traces.png")

    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'result': result,
        'beta': {
            'median': float(beta_median),
            'std': float(beta_std),
            'q16': float(beta_16),
            'q84': float(beta_84)
        },
        'xi': {
            'median': float(xi_median),
            'std': float(xi_std),
            'q16': float(xi_16),
            'q84': float(xi_84)
        },
        'beta_target': beta_target,
        'beta_v22': beta_v22,
        'beta_offset': float(beta_offset),
        'beta_offset_percent': float(100 * beta_offset / beta_target)
    }

    with open(f'{output_dir}/mcmc_2d_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✓ Saved: {output_dir}/mcmc_2d_results.json")
    print()

    return results


if __name__ == '__main__':
    import os
    os.makedirs('results', exist_ok=True)

    print("\n" + "="*70)
    print("QUICK 2D MCMC TEST")
    print("="*70)
    print("\nTesting hypothesis: Does ξ term push β → 3.043233053?")
    print()

    # Run MCMC
    sampler = run_quick_mcmc(
        n_walkers=16,
        n_steps=1000,
        n_burn=200
    )

    # Analyze
    results = analyze_results(sampler)

    print("\n" + "="*70)
    print("DONE!")
    print("="*70)
