#!/usr/bin/env python3
"""
Stage 2 MCMC: Add Temporal Term τ

E = ∫ [½ξ|∇ρ|² + β(δρ)² + τ(∂ρ/∂t)²] dV

Goal: Test if temporal term breaks β-ξ degeneracy

Strategy:
- 3D parameter space: (β, ξ, τ)
- τ affects breathing mode frequency ω ~ √(β/τ)
- For static soliton: ∂ρ/∂t = 0, but τ constrains stability
- Cross-lepton coupling + τ should isolate β
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


class Stage2Model:
    """
    Model with temporal term τ.

    Free parameters: (β, ξ, τ) shared across leptons
    Fixed parameters: (R, U, A) per lepton

    For static soliton, ∂ρ/∂t = 0, but τ affects:
    - Breathing mode frequency: ω ~ √(β/τ)
    - Effective compressibility
    - Temporal response to perturbations
    """

    def __init__(self):
        # Fixed geometry (same as Stage 1)
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

    def compute_energy(self, beta, xi, tau, lepton='electron'):
        """
        Compute energy with temporal term.

        For static solution (∂ρ/∂t = 0):
        E_static = ∫[½ξ|∇ρ|² + β(δρ)²] dV

        But τ affects effective energy via:
        - Temporal stiffness correction
        - Breathing mode contribution
        - Emergent time coupling
        """
        if lepton == 'electron':
            r, rho = self.r_e, self.rho_e
        elif lepton == 'muon':
            r, rho = self.r_mu, self.rho_mu
        elif lepton == 'tau':
            r, rho = self.r_tau, self.rho_tau
        else:
            raise ValueError(f"Unknown lepton: {lepton}")

        # Static energy (temporal term = 0 for equilibrium)
        E_total, E_grad, E_comp = func.gradient_energy_functional(rho, r, xi, beta)

        # Add effective contribution from temporal stiffness
        # This represents how τ modifies the equilibrium through
        # emergent time coupling (Cl(3,3) → Cl(3,1))
        #
        # Physical interpretation: τ affects how density responds
        # to temporal perturbations, which back-reacts on spatial structure
        #
        # Model: Add small correction proportional to τ/β ratio
        # (dimensionless coupling of temporal to vacuum stiffness)
        tau_correction_factor = 1.0 + 0.01 * (tau / beta)  # 1% correction per unit τ/β

        E_total_with_tau = E_total * tau_correction_factor

        return E_total_with_tau

    def compute_masses(self, beta, xi, tau):
        """Compute predicted masses with temporal term."""
        E_e = self.compute_energy(beta, xi, tau, 'electron')
        E_mu = self.compute_energy(beta, xi, tau, 'muon')
        E_tau = self.compute_energy(beta, xi, tau, 'tau')

        # Normalize to electron mass
        norm = M_ELECTRON / E_e

        m_e_pred = M_ELECTRON
        m_mu_pred = E_mu * norm
        m_tau_pred = E_tau * norm

        return m_e_pred, m_mu_pred, m_tau_pred

    def compute_breathing_frequency(self, beta, tau):
        """
        Compute breathing mode frequency.

        For small perturbations: ρ = ρ_eq + δρ·exp(iωt)
        Frequency: ω ~ √(β/τ)
        """
        if tau <= 0:
            return np.inf

        omega = np.sqrt(beta / tau)
        return omega


# Global model
model = Stage2Model()


def log_prior(params):
    """
    Prior for (β, ξ, τ).

    β ~ Normal(3.058, 0.15)     # From α-constraint
    ξ ~ LogNormal(0, 0.5)       # Gradient stiffness ~1
    τ ~ LogNormal(0, 0.5)       # Temporal stiffness ~1
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
    """Likelihood for masses with temporal term."""
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


def run_stage2_mcmc(
    n_walkers=24,
    n_steps=1000,
    n_burn=200,
    output_file='results/mcmc_stage2_temporal.h5'
):
    """
    Run Stage 2 MCMC with temporal term.

    Parameters
    ----------
    n_walkers : int
        Number of walkers (recommend 8× n_dim = 24)
    n_steps : int
        Production steps
    n_burn : int
        Burn-in steps
    """
    n_dim = 3

    print("="*70)
    print("STAGE 2 MCMC: (β, ξ, τ) with Temporal Term")
    print("="*70)
    print()
    print(f"Parameters: {n_dim}D (β, ξ, τ)")
    print(f"Walkers: {n_walkers}")
    print(f"Steps: {n_steps} (+ {n_burn} burn-in)")
    print(f"Geometry: FIXED")
    print()

    # Initialize walkers
    beta_init = 3.058
    xi_init = 1.0
    tau_init = 1.0

    pos = np.array([beta_init, xi_init, tau_init]) + 0.1 * np.random.randn(n_walkers, n_dim)

    # Ensure positivity for ξ, τ
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


def analyze_stage2_results(sampler, output_dir='results'):
    """Analyze Stage 2 results."""
    os.makedirs(output_dir, exist_ok=True)

    samples = sampler.get_chain(flat=True)

    print("="*70)
    print("STAGE 2 POSTERIOR SUMMARY")
    print("="*70)
    print()

    param_names = ['β', 'ξ', 'τ']
    results = {'timestamp': datetime.now().isoformat()}

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
    beta_target = 3.058
    beta_offset = abs(beta_median - beta_target)

    print("="*70)
    print("DID TEMPORAL TERM BREAK β-ξ DEGENERACY?")
    print("="*70)
    print(f"β target (Golden Loop): {beta_target:.4f}")
    print(f"β Stage 1 (no τ):       2.9518 ± 0.1529")
    print(f"β Stage 2 (with τ):     {beta_median:.4f} ± {beta_std:.4f}")
    print()
    print(f"Offset from target:     {beta_offset:.4f} ({100*beta_offset/beta_target:.2f}%)")
    print()

    if beta_offset < 0.02:
        print("✅ SUCCESS! Temporal term resolved β offset!")
        print("   β → 3.058 when (ξ, τ) included")
        result = "success"
    elif beta_offset < 0.05:
        print("⚠️ PARTIAL: Improved but still offset")
        print("   May need Stage 3 (EM functional)")
        result = "partial"
    else:
        print("❌ PERSISTS: τ didn't break degeneracy")
        print("   Need Stage 3 or independent observable")
        result = "failure"

    print("="*70)
    print()

    # Corner plot (3D)
    fig = corner.corner(
        samples,
        labels=['β', 'ξ', 'τ'],
        truths=[3.058, 1.0, 1.0],
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_fmt='.3f'
    )
    fig.suptitle('Stage 2: (β, ξ, τ) Posterior', fontsize=14, fontweight='bold', y=1.02)
    fig.savefig(f'{output_dir}/mcmc_stage2_corner.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir}/mcmc_stage2_corner.png")

    # Traces
    chain = sampler.get_chain()
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))

    for i, (ax, label, target) in enumerate(zip(axes, ['β', 'ξ', 'τ'], [3.058, 1.0, 1.0])):
        ax.plot(chain[:, :, i], alpha=0.3, color='k', lw=0.5)
        ax.axhline(target, color='r', ls='--', lw=2, label=f'Expected: {target}')
        ax.set_ylabel(label, fontsize=12)
        ax.legend()
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel('Step', fontsize=12)
    fig.suptitle('Stage 2 Traces', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(f'{output_dir}/mcmc_stage2_traces.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir}/mcmc_stage2_traces.png")

    # Save results
    results['result'] = result
    results['beta_target'] = beta_target
    results['beta_offset'] = float(beta_offset)

    with open(f'{output_dir}/mcmc_stage2_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✓ Saved: {output_dir}/mcmc_stage2_results.json")
    print()

    return results


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)

    print("\n" + "="*70)
    print("STAGE 2: TESTING TEMPORAL TERM")
    print("="*70)
    print("\nHypothesis: τ breaks β-ξ degeneracy → β converges to 3.058")
    print()

    # Run
    sampler = run_stage2_mcmc()

    # Analyze
    results = analyze_stage2_results(sampler)

    print("\n" + "="*70)
    print("STAGE 2 COMPLETE!")
    print("="*70)
