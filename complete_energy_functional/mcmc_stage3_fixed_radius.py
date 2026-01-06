#!/usr/bin/env python3
"""
Stage 3 CORRECTED: Fix R_e = 0.84 fm (Hard Length Scale)

User's insight: "To break the degeneracy, you need a Hard Length Scale."

Strategy:
- FIX R_e = 0.84 fm (experimental charge radius)
- Fit only (β, ξ, τ)
- R_e constraint provides different R-scaling for E_comp vs E_grad
- Should collapse "banana" degeneracy

Key scaling:
- E_comp ∝ β·R³  → Fixed by R_e = 0.84 fm
- E_grad ∝ ξ·R   → Fixed by R_e = 0.84 fm
- Different volume/surface scaling breaks β-ξ trade-off
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

# FIXED RADIUS (experimental charge radius - Hard Length Scale)
R_ELECTRON_FIXED = 0.84  # fm (RMS charge radius)

# Conversion
HBARC = 197.33  # MeV·fm
R_ELECTRON_NATURAL = R_ELECTRON_FIXED / HBARC  # in natural units (MeV⁻¹)


class FixedRadiusModel:
    """
    Model with R_e FIXED at experimental value.

    Free parameters: (β, ξ, τ)
    Fixed parameters: R_e = 0.84 fm (HARD LENGTH SCALE)
                      (U, A) velocity and amplitude
    """

    def __init__(self, R_e_fixed=R_ELECTRON_NATURAL):
        self.R_e = R_e_fixed  # FIXED

        # Fixed flow parameters
        self.U_e = 0.5
        self.A_e = 1.0

        self.U_mu = self.U_e
        self.A_mu = self.A_e
        self.U_tau = self.U_e
        self.A_tau = self.A_e

        # Derived radii (geometric scaling)
        mass_ratio_mu = M_MUON / M_ELECTRON
        mass_ratio_tau = M_TAU / M_MUON

        self.R_mu = self.R_e * np.sqrt(mass_ratio_mu)
        self.R_tau = self.R_mu * np.sqrt(mass_ratio_tau)

        # Grid
        self.r_max = 10.0
        self.n_points = 300

        # Precompute profiles (geometry FIXED)
        self.r_e = np.linspace(0, self.r_max * self.R_e, self.n_points)
        self.r_mu = np.linspace(0, self.r_max * self.R_mu, self.n_points)
        self.r_tau = np.linspace(0, self.r_max * self.R_tau, self.n_points)

        self.rho_e = solv.hill_vortex_profile(self.r_e, self.R_e, self.U_e, self.A_e)
        self.rho_mu = solv.hill_vortex_profile(self.r_mu, self.R_mu, self.U_mu, self.A_mu)
        self.rho_tau = solv.hill_vortex_profile(self.r_tau, self.R_tau, self.U_tau, self.A_tau)

    def compute_energy(self, beta, xi, tau, lepton='electron'):
        """Compute energy with FIXED R_e."""
        if lepton == 'electron':
            r, rho = self.r_e, self.rho_e
        elif lepton == 'muon':
            r, rho = self.r_mu, self.rho_mu
        elif lepton == 'tau':
            r, rho = self.r_tau, self.rho_tau
        else:
            raise ValueError(f"Unknown lepton: {lepton}")

        # Static energy
        E_total, E_grad, E_comp = func.gradient_energy_functional(rho, r, xi, beta)

        # Temporal correction
        tau_correction_factor = 1.0 + 0.01 * (tau / beta)
        E_total_with_tau = E_total * tau_correction_factor

        return E_total_with_tau

    def compute_masses(self, beta, xi, tau):
        """Compute predicted masses with fixed R_e."""
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
model = FixedRadiusModel()


def log_prior(params):
    """
    Prior for (β, ξ, τ) with R_e FIXED.

    β ~ Normal(3.058, 0.15)
    ξ ~ LogNormal(0, 0.5)
    τ ~ LogNormal(0, 0.5)
    """
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
    """Likelihood for masses with FIXED R_e."""
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


def run_fixed_radius_mcmc(
    n_walkers=24,
    n_steps=1000,
    n_burn=200,
    output_file='results/mcmc_stage3_fixed_radius.h5'
):
    """Run MCMC with R_e FIXED."""
    n_dim = 3  # Only (β, ξ, τ)

    print("="*70)
    print("STAGE 3 CORRECTED: R_e = 0.84 fm FIXED (Hard Length Scale)")
    print("="*70)
    print()
    print(f"R_e = {R_ELECTRON_FIXED:.3f} fm (FIXED - experimental value)")
    print(f"Parameters: {n_dim}D (β, ξ, τ)")
    print(f"Walkers: {n_walkers}")
    print(f"Steps: {n_steps} (+ {n_burn} burn-in)")
    print()
    print("Hypothesis: Fixed R_e breaks β-ξ degeneracy via different scaling:")
    print("  E_comp ∝ β·R³ (volume)")
    print("  E_grad ∝ ξ·R  (surface)")
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


def analyze_fixed_radius_results(sampler, output_dir='results'):
    """Analyze results with fixed R_e."""
    os.makedirs(output_dir, exist_ok=True)

    samples = sampler.get_chain(flat=True)

    print("="*70)
    print("FIXED R_e POSTERIOR SUMMARY")
    print("="*70)
    print()
    print(f"R_e = {R_ELECTRON_FIXED:.3f} fm (FIXED)")
    print()

    param_names = ['β', 'ξ', 'τ']
    results = {
        'timestamp': datetime.now().isoformat(),
        'R_e_fixed_fm': R_ELECTRON_FIXED
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

    print("="*70)
    print("DID FIXED R_e BREAK β-ξ DEGENERACY?")
    print("="*70)
    print(f"β target (Golden Loop): {beta_target:.4f}")
    print(f"β Stage 2 (no R):       2.9617 ± 0.1487")
    print(f"β Stage 3 (R fixed):    {beta_median:.4f} ± {beta_std:.4f}")
    print()
    print(f"Offset from target:     {beta_offset:.4f} ({100*beta_offset/beta_target:.2f}%)")
    print()

    # Check β-ξ correlation
    beta_samples = samples[:, 0]
    xi_samples = samples[:, 1]
    corr_beta_xi = np.corrcoef(beta_samples, xi_samples)[0, 1]

    print(f"β-ξ correlation coefficient: {corr_beta_xi:.4f}")
    if abs(corr_beta_xi) < 0.3:
        print("  → Degeneracy BROKEN! (weak correlation)")
    elif abs(corr_beta_xi) < 0.7:
        print("  → Partial decorrelation")
    else:
        print("  → Strong correlation persists")
    print()

    if beta_offset < 0.02 and abs(corr_beta_xi) < 0.3:
        print("✅ SUCCESS! Fixed R_e collapsed banana → β = 3.058!")
        print("   Hard length scale broke degeneracy!")
        result = "success"
    elif beta_offset < 0.05:
        print("⚠️ PARTIAL: β closer to target, correlation reduced")
        print("   May need additional constraints")
        result = "partial"
    else:
        print("❌ PERSISTS: Degeneracy not fully broken")
        print("   Check if R_e = 0.84 fm is correct value")
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
    fig.suptitle(f'Stage 3: (β, ξ, τ) with R_e = {R_ELECTRON_FIXED:.2f} fm FIXED',
                 fontsize=12, fontweight='bold', y=1.0)
    fig.savefig(f'{output_dir}/mcmc_stage3_fixed_radius_corner.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir}/mcmc_stage3_fixed_radius_corner.png")

    # Traces
    chain = sampler.get_chain()
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))

    for i, (ax, label, target) in enumerate(zip(axes, ['β', 'ξ', 'τ'], [3.058, 1.0, 1.0])):
        ax.plot(chain[:, :, i], alpha=0.3, color='k', lw=0.5)
        ax.axhline(target, color='r', ls='--', lw=2, label=f'Expected: {target}')
        ax.set_ylabel(label, fontsize=12)
        ax.legend()
        ax.grid(alpha=0.3)

    axes[0].set_title(f'R_e = {R_ELECTRON_FIXED:.2f} fm (FIXED)', fontsize=10, color='red')
    axes[-1].set_xlabel('Step', fontsize=12)
    fig.suptitle('Fixed R_e Traces', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(f'{output_dir}/mcmc_stage3_fixed_radius_traces.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir}/mcmc_stage3_fixed_radius_traces.png")

    # Save results
    results['result'] = result
    results['beta_target'] = beta_target
    results['beta_offset'] = float(beta_offset)
    results['correlation_beta_xi'] = float(corr_beta_xi)

    # Compare ξ values
    print()
    print(f"ξ comparison:")
    print(f"  Stage 2 (R variable): ξ = 25.98 ± 1.30")
    print(f"  Stage 3 (R fixed):    ξ = {xi_median:.2f} ± {results['ξ']['std']:.2f}")
    print()

    with open(f'{output_dir}/mcmc_stage3_fixed_radius_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✓ Saved: {output_dir}/mcmc_stage3_fixed_radius_results.json")
    print()

    return results


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)

    print("\n" + "="*70)
    print("STAGE 3 CORRECTED: HARD LENGTH SCALE TEST")
    print("="*70)
    print("\nHypothesis: R_e = 0.84 fm (FIXED) breaks β-ξ degeneracy")
    print("  Previous stages: R variable → degeneracy persists")
    print("  This stage: R fixed → should collapse banana")
    print()

    # Run
    sampler = run_fixed_radius_mcmc()

    # Analyze
    results = analyze_fixed_radius_results(sampler)

    print("\n" + "="*70)
    print("STAGE 3 CORRECTED COMPLETE!")
    print("="*70)
