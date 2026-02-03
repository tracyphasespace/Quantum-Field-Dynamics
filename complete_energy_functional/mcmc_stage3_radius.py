#!/usr/bin/env python3
"""
Stage 3 MCMC: Add Charge Radius Constraint

Strategy: Keep (β, ξ, τ) but now FIT radius R_e as a 4th parameter.

Key insight (from user):
- E_comp ∝ β·R³ (volume)
- E_grad ∝ ξ·R  (surface)

Different scaling → adding R_e constraint breaks β-ξ degeneracy!

Observables:
1. Lepton masses: m_e, m_μ, m_τ
2. Electron charge radius: ⟨r²⟩_e^1/2 ≈ 0.84 fm

Hypothesis: R constraint will collapse "banana" → β → 3.043233053
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

# Experimental charge radius (RMS charge radius)
# Classical electron radius: r_e = e²/(4πε₀m_e c²) ≈ 2.818 fm
# But quantum corrections give effective radius ≈ 0.84 fm
R_ELECTRON_EXP = 0.84  # fm
SIGMA_RADIUS = 0.1     # fm (conservative uncertainty)

# Conversion: fm to natural units (1 fm = 1/197.33 MeV⁻¹)
HBARC = 197.33  # MeV·fm
R_ELECTRON_NATURAL = R_ELECTRON_EXP / HBARC  # in MeV⁻¹


class Stage3Model:
    """
    Model with radius R_e as FREE parameter.

    Free parameters: (β, ξ, τ, R_e) shared properties
    Fixed parameters: (U, A) velocity and amplitude
    Derived: R_μ, R_τ from scaling
    """

    def __init__(self):
        # Fixed flow parameters
        self.U_e = 0.5   # Velocity (fraction of c)
        self.A_e = 1.0   # Amplitude

        # Same for heavier leptons
        self.U_mu = self.U_e
        self.A_mu = self.A_e
        self.U_tau = self.U_e
        self.A_tau = self.A_e

        # Grid
        self.r_max = 10.0
        self.n_points = 300

    def build_profiles(self, R_e):
        """Build density profiles for given R_e."""
        # Geometric scaling for heavier leptons
        mass_ratio_mu = M_MUON / M_ELECTRON
        mass_ratio_tau = M_TAU / M_MUON

        R_mu = R_e * np.sqrt(mass_ratio_mu)
        R_tau = R_mu * np.sqrt(mass_ratio_tau)

        # Radial grids
        r_e = np.linspace(0, self.r_max * R_e, self.n_points)
        r_mu = np.linspace(0, self.r_max * R_mu, self.n_points)
        r_tau = np.linspace(0, self.r_max * R_tau, self.n_points)

        # Density profiles
        rho_e = solv.hill_vortex_profile(r_e, R_e, self.U_e, self.A_e)
        rho_mu = solv.hill_vortex_profile(r_mu, R_mu, self.U_mu, self.A_mu)
        rho_tau = solv.hill_vortex_profile(r_tau, R_tau, self.U_tau, self.A_tau)

        return {
            'electron': (r_e, rho_e, R_e),
            'muon': (r_mu, rho_mu, R_mu),
            'tau': (r_tau, rho_tau, R_tau)
        }

    def compute_charge_radius(self, r, rho):
        """
        Compute RMS charge radius.

        ⟨r²⟩ = ∫ r² ρ_charge(r) dV / ∫ ρ_charge(r) dV

        For Hill vortex, charge density ∝ density perturbation δρ
        """
        from scipy.integrate import simpson

        # Charge density (perturbation from vacuum)
        rho_charge = rho - 1.0  # δρ = ρ - ρ_vac

        # Only consider interior (where δρ > 0)
        mask = rho_charge > 0

        if not mask.any():
            return 0.0

        r_int = r[mask]
        rho_int = rho_charge[mask]

        # Normalize charge density
        Q_total = simpson(rho_int * 4 * np.pi * r_int**2, r_int)

        if Q_total <= 0:
            return 0.0

        # RMS radius
        r2_avg = simpson(rho_int * r_int**2 * 4 * np.pi * r_int**2, r_int) / Q_total
        r_rms = np.sqrt(r2_avg)

        return r_rms

    def compute_energy(self, beta, xi, tau, R_e, lepton='electron'):
        """Compute energy with variable R_e."""
        profiles = self.build_profiles(R_e)
        r, rho, R = profiles[lepton]

        # Static energy
        E_total, E_grad, E_comp = func.gradient_energy_functional(rho, r, xi, beta)

        # Temporal correction
        tau_correction_factor = 1.0 + 0.01 * (tau / beta)
        E_total_with_tau = E_total * tau_correction_factor

        return E_total_with_tau

    def compute_masses(self, beta, xi, tau, R_e):
        """Compute predicted masses with variable R_e."""
        E_e = self.compute_energy(beta, xi, tau, R_e, 'electron')
        E_mu = self.compute_energy(beta, xi, tau, R_e, 'muon')
        E_tau = self.compute_energy(beta, xi, tau, R_e, 'tau')

        # Normalize to electron mass
        norm = M_ELECTRON / E_e

        m_e_pred = M_ELECTRON
        m_mu_pred = E_mu * norm
        m_tau_pred = E_tau * norm

        return m_e_pred, m_mu_pred, m_tau_pred

    def compute_electron_radius(self, R_e):
        """Compute electron RMS charge radius."""
        profiles = self.build_profiles(R_e)
        r, rho, _ = profiles['electron']

        r_rms = self.compute_charge_radius(r, rho)

        # Convert to physical units (fm)
        r_rms_fm = r_rms * HBARC

        return r_rms_fm


# Global model
model = Stage3Model()


def log_prior(params):
    """
    Prior for (β, ξ, τ, R_e).

    β ~ Normal(3.043233053, 0.15)           # From α-constraint
    ξ ~ LogNormal(0, 0.5)             # Gradient stiffness
    τ ~ LogNormal(0, 0.5)             # Temporal stiffness
    R_e ~ Normal(R_exp, σ_R)          # Charge radius (in natural units)
    """
    beta, xi, tau, R_e = params

    # β prior
    beta_mean = 3.043233053
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

    # R_e prior (positive, centered on experimental value)
    if R_e <= 0:
        return -np.inf
    R_e_mean = R_ELECTRON_NATURAL
    R_e_std = SIGMA_RADIUS / HBARC  # Convert to natural units
    lp_R = -0.5 * ((R_e - R_e_mean) / R_e_std)**2

    return lp_beta + lp_xi + lp_tau + lp_R


def log_likelihood(params):
    """
    Likelihood for masses AND charge radius.

    Observables:
    1. m_e, m_μ, m_τ (3 masses)
    2. ⟨r²⟩_e^1/2 (electron charge radius)
    """
    beta, xi, tau, R_e = params

    try:
        # Predict masses
        m_e_pred, m_mu_pred, m_tau_pred = model.compute_masses(beta, xi, tau, R_e)

        # Predict electron charge radius
        r_rms_pred = model.compute_electron_radius(R_e)

    except Exception as e:
        return -1e10

    # Check validity
    if not np.isfinite([m_e_pred, m_mu_pred, m_tau_pred, r_rms_pred]).all():
        return -1e10

    if m_e_pred <= 0 or m_mu_pred <= 0 or m_tau_pred <= 0 or r_rms_pred <= 0:
        return -1e10

    # Combined uncertainties (masses)
    sigma_e = np.sqrt(SIGMA_ELECTRON**2 + SIGMA_MODEL_ELECTRON**2)
    sigma_mu = np.sqrt(SIGMA_MUON**2 + SIGMA_MODEL_MUON**2)
    sigma_tau = np.sqrt(SIGMA_TAU**2 + SIGMA_MODEL_TAU**2)

    # Chi-squared (masses)
    chi2_e = ((m_e_pred - M_ELECTRON) / sigma_e)**2
    chi2_mu = ((m_mu_pred - M_MUON) / sigma_mu)**2
    chi2_tau = ((m_tau_pred - M_TAU) / sigma_tau)**2

    # Chi-squared (radius)
    chi2_radius = ((r_rms_pred - R_ELECTRON_EXP) / SIGMA_RADIUS)**2

    # Total log-likelihood
    log_L = -0.5 * (chi2_e + chi2_mu + chi2_tau + chi2_radius)

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


def run_stage3_mcmc(
    n_walkers=32,
    n_steps=1000,
    n_burn=200,
    output_file='results/mcmc_stage3_radius.h5'
):
    """
    Run Stage 3 MCMC with radius constraint.

    Parameters
    ----------
    n_walkers : int
        Number of walkers (recommend 8× n_dim = 32)
    n_steps : int
        Production steps
    n_burn : int
        Burn-in steps
    """
    n_dim = 4  # (β, ξ, τ, R_e)

    print("="*70)
    print("STAGE 3 MCMC: (β, ξ, τ, R_e) with Charge Radius Constraint")
    print("="*70)
    print()
    print(f"Parameters: {n_dim}D (β, ξ, τ, R_e)")
    print(f"Walkers: {n_walkers}")
    print(f"Steps: {n_steps} (+ {n_burn} burn-in)")
    print()
    print(f"NEW OBSERVABLE: Electron charge radius")
    print(f"  R_e (exp): {R_ELECTRON_EXP:.3f} ± {SIGMA_RADIUS:.3f} fm")
    print()

    # Initialize walkers
    beta_init = 3.043233053
    xi_init = 1.0
    tau_init = 1.0
    R_e_init = R_ELECTRON_NATURAL  # Natural units

    pos = np.array([beta_init, xi_init, tau_init, R_e_init]) + \
          0.1 * np.random.randn(n_walkers, n_dim)

    # Ensure positivity
    pos[:, 1] = np.abs(pos[:, 1])  # ξ > 0
    pos[:, 2] = np.abs(pos[:, 2])  # τ > 0
    pos[:, 3] = np.abs(pos[:, 3])  # R_e > 0

    print(f"Initial positions:")
    print(f"  β:   {pos[:, 0].min():.3f} - {pos[:, 0].max():.3f}")
    print(f"  ξ:   {pos[:, 1].min():.3f} - {pos[:, 1].max():.3f}")
    print(f"  τ:   {pos[:, 2].min():.3f} - {pos[:, 2].max():.3f}")
    print(f"  R_e: {pos[:, 3].min():.4f} - {pos[:, 3].max():.4f} (nat. units)")
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


def analyze_stage3_results(sampler, output_dir='results'):
    """Analyze Stage 3 results."""
    os.makedirs(output_dir, exist_ok=True)

    samples = sampler.get_chain(flat=True)

    print("="*70)
    print("STAGE 3 POSTERIOR SUMMARY")
    print("="*70)
    print()

    param_names = ['β', 'ξ', 'τ', 'R_e']
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

        # Convert R_e to fm for display
        if name == 'R_e':
            median_fm = median * HBARC
            std_fm = std * HBARC
            q16_fm = q16 * HBARC
            q84_fm = q84 * HBARC
            print(f"{name} posterior: {median_fm:.3f} ± {std_fm:.3f} fm")
            print(f"              [{q16_fm:.3f}, {q84_fm:.3f}] (68% CI)")
            print(f"   (natural):  {median:.4f} ± {std:.4f} MeV⁻¹")
        else:
            print(f"{name} posterior: {median:.4f} ± {std:.4f}")
            print(f"             [{q16:.4f}, {q84:.4f}] (68% CI)")
        print()

    # Critical assessment
    beta_median = results['β']['median']
    beta_std = results['β']['std']
    beta_target = 3.043233053
    beta_offset = abs(beta_median - beta_target)

    print("="*70)
    print("DID RADIUS CONSTRAINT BREAK β-ξ DEGENERACY?")
    print("="*70)
    print(f"β target (Golden Loop): {beta_target:.4f}")
    print(f"β Stage 2 (no R):       2.9617 ± 0.1487")
    print(f"β Stage 3 (with R):     {beta_median:.4f} ± {beta_std:.4f}")
    print()
    print(f"Offset from target:     {beta_offset:.4f} ({100*beta_offset/beta_target:.2f}%)")
    print()

    if beta_offset < 0.02:
        print("✅ SUCCESS! Radius constraint collapsed banana → β = 3.043233053!")
        print("   Degeneracy BROKEN by hard length scale!")
        result = "success"
    elif beta_offset < 0.05:
        print("⚠️ PARTIAL: Improved but still ~2-5% offset")
        print("   May need EM functional or better R_e data")
        result = "partial"
    else:
        print("❌ PERSISTS: R constraint didn't fully break degeneracy")
        print("   Need full EM functional (Poisson solver)")
        result = "failure"

    print("="*70)
    print()

    # Corner plot (4D)
    labels = ['β', 'ξ', 'τ', 'R_e (fm)']
    samples_plot = samples.copy()
    samples_plot[:, 3] *= HBARC  # Convert R_e to fm for plotting

    fig = corner.corner(
        samples_plot,
        labels=labels,
        truths=[3.043233053, 1.0, 1.0, R_ELECTRON_EXP],
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_fmt='.3f'
    )
    fig.suptitle('Stage 3: (β, ξ, τ, R_e) Posterior with Radius Constraint',
                 fontsize=12, fontweight='bold', y=1.0)
    fig.savefig(f'{output_dir}/mcmc_stage3_corner.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir}/mcmc_stage3_corner.png")

    # Check β-ξ correlation
    beta_samples = samples[:, 0]
    xi_samples = samples[:, 1]
    corr_beta_xi = np.corrcoef(beta_samples, xi_samples)[0, 1]

    print()
    print(f"β-ξ correlation coefficient: {corr_beta_xi:.4f}")
    if abs(corr_beta_xi) < 0.3:
        print("  → Degeneracy BROKEN! (correlation near zero)")
    elif abs(corr_beta_xi) < 0.7:
        print("  → Partial decorrelation (mild correlation remains)")
    else:
        print("  → Strong correlation persists (degeneracy not broken)")
    print()

    # Save results
    results['result'] = result
    results['beta_target'] = beta_target
    results['beta_offset'] = float(beta_offset)
    results['correlation_beta_xi'] = float(corr_beta_xi)

    with open(f'{output_dir}/mcmc_stage3_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✓ Saved: {output_dir}/mcmc_stage3_results.json")
    print()

    return results


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)

    print("\n" + "="*70)
    print("STAGE 3: TESTING RADIUS CONSTRAINT")
    print("="*70)
    print("\nHypothesis: R_e constraint breaks β-ξ degeneracy")
    print("  E_comp ∝ β·R³  (volume)")
    print("  E_grad ∝ ξ·R   (surface)")
    print("  → Different R-scaling breaks degeneracy!")
    print()

    # Run
    sampler = run_stage3_mcmc()

    # Analyze
    results = analyze_stage3_results(sampler)

    print("\n" + "="*70)
    print("STAGE 3 COMPLETE!")
    print("="*70)
