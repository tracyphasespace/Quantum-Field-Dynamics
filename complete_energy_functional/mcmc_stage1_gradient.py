"""
Stage 1 MCMC: Gradient-only energy functional

E = ∫ [½ξ|∇ρ|² + β(δρ)²] dV

Fits 11 parameters:
- Shared: ξ, β
- Per-lepton: R_e, U_e, A_e, R_μ, U_μ, A_μ, R_τ, U_τ, A_τ

Goal: Test if including gradient term ξ resolves β offset from 3.15 → 3.058
"""

import numpy as np
import emcee
import corner
import matplotlib.pyplot as plt
from multiprocessing import Pool
from typing import Tuple, Dict, Optional
import json
import h5py
from datetime import datetime

from .solvers import compute_mass_from_functional
from .functionals import gradient_energy_functional


# ============================================================================
# Experimental Data
# ============================================================================

LEPTON_MASSES = {
    'electron': {'mass': 0.5110, 'sigma': 1e-6},  # MeV/c²
    'muon': {'mass': 105.658, 'sigma': 1e-3},
    'tau': {'mass': 1776.86, 'sigma': 0.12}
}

# Model uncertainty from V22 analysis
SIGMA_MODEL = {
    'electron': 1e-3,  # ~0.2%
    'muon': 0.1,       # ~0.1%
    'tau': 2.0         # ~0.1%
}


# ============================================================================
# Parameter Priors
# ============================================================================

def log_prior(params: np.ndarray) -> float:
    """
    Log prior probability for parameters.

    Parameters (11D):
    - ξ: gradient stiffness (dimensionless, ~1)
    - β: vacuum stiffness (dimensionless, ~3.058 from α-constraint)
    - R_e, U_e, A_e: electron geometry
    - R_μ, U_μ, A_μ: muon geometry
    - R_τ, U_τ, A_τ: tau geometry

    Priors:
    - ξ ~ LogNormal(μ=0, σ=0.5)  → median=1, allows 0.3-3
    - β ~ Normal(μ=3.058, σ=0.15)  → ±5% around Golden Loop
    - R ~ LogNormal  → expect Compton scale ~10^-13 m
    - U ~ Uniform(0.1, 0.9)  → fraction of c
    - A ~ LogNormal  → wide prior on amplitude
    """
    ξ, β, R_e, U_e, A_e, R_μ, U_μ, A_μ, R_τ, U_τ, A_τ = params

    lp = 0.0

    # ξ prior: LogNormal(0, 0.5)
    if ξ <= 0:
        return -np.inf
    lp += -0.5 * (np.log(ξ) / 0.5)**2 - np.log(ξ * 0.5 * np.sqrt(2*np.pi))

    # β prior: Normal(3.058, 0.15)
    lp += -0.5 * ((β - 3.058) / 0.15)**2

    # Radius priors: LogNormal around Compton scale
    for R in [R_e, R_μ, R_τ]:
        if R <= 0:
            return -np.inf
        # Center on 1e-13 m
        μ_R = np.log(1e-13)
        σ_R = 1.0  # Allow order of magnitude variation
        lp += -0.5 * ((np.log(R) - μ_R) / σ_R)**2 - np.log(R * σ_R * np.sqrt(2*np.pi))

    # Velocity priors: Uniform(0.1, 0.9)
    for U in [U_e, U_μ, U_τ]:
        if not (0.1 <= U <= 0.9):
            return -np.inf
        lp += np.log(1.0 / 0.8)  # Uniform density

    # Amplitude priors: LogNormal(0, 2.0) - wide prior
    for A in [A_e, A_μ, A_τ]:
        if A <= 0:
            return -np.inf
        lp += -0.5 * (np.log(A) / 2.0)**2 - np.log(A * 2.0 * np.sqrt(2*np.pi))

    return lp


# ============================================================================
# Likelihood Function
# ============================================================================

def log_likelihood(params: np.ndarray) -> float:
    """
    Log likelihood for lepton mass observations.

    L(data | params) = Π_i N(m_i^obs | m_i^pred(params), σ_i)

    where σ_i² = σ_exp² + σ_model²
    """
    ξ, β, R_e, U_e, A_e, R_μ, U_μ, A_μ, R_τ, U_τ, A_τ = params

    try:
        # Predict masses for each lepton
        # NOTE: This is a placeholder - needs proper unit handling
        m_e_pred = compute_mass_from_functional(ξ, β, R_e, U_e, A_e)
        m_μ_pred = compute_mass_from_functional(ξ, β, R_μ, U_μ, A_μ)
        m_τ_pred = compute_mass_from_functional(ξ, β, R_τ, U_τ, A_τ)

        # Normalize to experimental values (temporary hack for testing)
        # TODO: Implement proper unit conversion from energy → mass
        norm_e = LEPTON_MASSES['electron']['mass'] / (m_e_pred + 1e-10)
        m_e_pred *= norm_e
        m_μ_pred *= norm_e
        m_τ_pred *= norm_e

    except Exception as e:
        # Solver failed - return very low likelihood
        return -1e10

    # Observed masses and uncertainties
    m_e_obs = LEPTON_MASSES['electron']['mass']
    m_μ_obs = LEPTON_MASSES['muon']['mass']
    m_τ_obs = LEPTON_MASSES['tau']['mass']

    # Combined uncertainties
    σ_e = np.sqrt(LEPTON_MASSES['electron']['sigma']**2 + SIGMA_MODEL['electron']**2)
    σ_μ = np.sqrt(LEPTON_MASSES['muon']['sigma']**2 + SIGMA_MODEL['muon']**2)
    σ_τ = np.sqrt(LEPTON_MASSES['tau']['sigma']**2 + SIGMA_MODEL['tau']**2)

    # Log-likelihood (sum of log Gaussians)
    chi2_e = ((m_e_pred - m_e_obs) / σ_e)**2
    chi2_μ = ((m_μ_pred - m_μ_obs) / σ_μ)**2
    chi2_τ = ((m_τ_pred - m_τ_obs) / σ_τ)**2

    log_L = -0.5 * (chi2_e + chi2_μ + chi2_τ)

    return log_L


def log_probability(params: np.ndarray) -> float:
    """
    Log posterior: log P(params | data) = log P(data | params) + log P(params)
    """
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf

    ll = log_likelihood(params)
    if not np.isfinite(ll):
        return -np.inf

    return lp + ll


# ============================================================================
# MCMC Sampler
# ============================================================================

def initialize_walkers(n_walkers: int, n_dim: int) -> np.ndarray:
    """
    Initialize walker positions near expected values.

    Parameters
    ----------
    n_walkers : int
        Number of MCMC walkers
    n_dim : int
        Parameter space dimension (11 for Stage 1)

    Returns
    -------
    pos : array (n_walkers, n_dim)
        Initial walker positions
    """
    # Expected values from V22 and physics intuition
    ξ_init = 1.0
    β_init = 3.058
    R_e_init = 1e-13  # ~Compton scale
    U_e_init = 0.5
    A_e_init = 1.0

    # Scale for heavier leptons (geometric scaling)
    R_μ_init = R_e_init * np.sqrt(105.658 / 0.511)
    R_τ_init = R_μ_init * np.sqrt(1776.86 / 105.658)

    # Initial guess
    p0 = np.array([
        ξ_init, β_init,
        R_e_init, U_e_init, A_e_init,
        R_μ_init, U_e_init, A_e_init,  # Same U, A for simplicity
        R_τ_init, U_e_init, A_e_init
    ])

    # Perturb around initial guess
    pos = p0 + 0.1 * p0 * np.random.randn(n_walkers, n_dim)

    # Ensure positivity
    pos[:, [0, 1, 2, 4, 5, 7, 8, 10]] = np.abs(pos[:, [0, 1, 2, 4, 5, 7, 8, 10]])

    # Ensure velocities in range
    pos[:, [3, 6, 9]] = np.clip(pos[:, [3, 6, 9]], 0.1, 0.9)

    return pos


def run_stage1_mcmc(
    n_walkers: int = 44,
    n_steps: int = 10000,
    n_burn: int = 2000,
    n_cores: int = 8,
    output_file: str = 'stage1_chains.h5'
) -> Tuple[np.ndarray, emcee.EnsembleSampler]:
    """
    Run Stage 1 MCMC with gradient-only functional.

    Parameters
    ----------
    n_walkers : int
        Number of MCMC walkers (recommend 4× n_dim)
    n_steps : int
        Number of MCMC steps
    n_burn : int
        Burn-in steps to discard
    n_cores : int
        Number of parallel cores
    output_file : str
        HDF5 file to save chains

    Returns
    -------
    samples : array (n_samples, n_dim)
        MCMC samples after burn-in
    sampler : emcee.EnsembleSampler
        Sampler object with full chain
    """
    n_dim = 11

    print("="*70)
    print("Stage 1 MCMC: Gradient-Only Energy Functional")
    print("="*70)
    print(f"Parameters: {n_dim}D (ξ, β, R×3, U×3, A×3)")
    print(f"Walkers: {n_walkers}")
    print(f"Steps: {n_steps} (+ {n_burn} burn-in)")
    print(f"Cores: {n_cores}")
    print()

    # Initialize walkers
    pos = initialize_walkers(n_walkers, n_dim)

    # Set up sampler
    backend = emcee.backends.HDFBackend(output_file)
    backend.reset(n_walkers, n_dim)

    with Pool(n_cores) as pool:
        sampler = emcee.EnsembleSampler(
            n_walkers, n_dim, log_probability,
            backend=backend,
            pool=pool
        )

        # Run burn-in
        print(f"Running burn-in ({n_burn} steps)...")
        state = sampler.run_mcmc(pos, n_burn, progress=True)
        sampler.reset()

        # Run production
        print(f"\nRunning production ({n_steps} steps)...")
        sampler.run_mcmc(state, n_steps, progress=True)

    # Get samples
    samples = sampler.get_chain(flat=True)

    print("\n" + "="*70)
    print("MCMC Complete!")
    print(f"Acceptance fraction: {np.mean(sampler.acceptance_fraction):.3f}")
    print(f"Samples collected: {samples.shape[0]}")
    print("="*70)

    return samples, sampler


# ============================================================================
# Analysis and Visualization
# ============================================================================

def analyze_stage1_results(
    samples: np.ndarray,
    sampler: emcee.EnsembleSampler,
    output_dir: str = 'results'
) -> Dict:
    """
    Analyze MCMC results and generate diagnostic plots.

    Parameters
    ----------
    samples : array
        Flattened MCMC samples
    sampler : emcee.EnsembleSampler
        Sampler object
    output_dir : str
        Directory for output files

    Returns
    -------
    results : dict
        Summary statistics and diagnostics
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    param_names = ['ξ', 'β', 'R_e', 'U_e', 'A_e', 'R_μ', 'U_μ', 'A_μ', 'R_τ', 'U_τ', 'A_τ']

    print("\n" + "="*70)
    print("POSTERIOR SUMMARY")
    print("="*70)

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

        print(f"{name:8s}: {median:.6e} ± {std:.6e} [{q16:.6e}, {q84:.6e}]")

    # Key question: Did β shift to 3.058?
    β_median = results['β']['median']
    β_std = results['β']['std']
    β_offset = abs(β_median - 3.058)

    print("\n" + "="*70)
    print("CRITICAL QUESTION: Did gradient term resolve β offset?")
    print("="*70)
    print(f"β posterior median: {β_median:.4f} ± {β_std:.4f}")
    print(f"β Golden Loop:      3.0580")
    print(f"β V22 (ξ=0):        3.1500")
    print(f"Offset from target: {β_offset:.4f} ({100*β_offset/3.058:.2f}%)")
    print()

    if β_offset < 0.02:
        print("✓ SUCCESS: β converged to Golden Loop prediction!")
    elif β_offset < 0.05:
        print("⚠ PARTIAL: β improved but still ~2% offset")
    else:
        print("✗ FAILURE: β offset persists - need Stage 2 (temporal) or Stage 3 (EM)")

    # Corner plot
    fig = corner.corner(
        samples[:, :2],  # Just ξ and β for clarity
        labels=['ξ', 'β'],
        truths=[1.0, 3.058],
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_fmt='.4f'
    )
    fig.savefig(f'{output_dir}/stage1_corner_xi_beta.png', dpi=150)
    plt.close()

    # Trace plots
    chain = sampler.get_chain()
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    for i, (ax, name) in enumerate(zip(axes, ['ξ', 'β'])):
        ax.plot(chain[:, :, i], alpha=0.3, color='k')
        ax.set_ylabel(name)
        ax.axhline(3.058 if i == 1 else 1.0, color='r', ls='--', label='Expected')
        ax.legend()

    axes[-1].set_xlabel('Step')
    fig.tight_layout()
    fig.savefig(f'{output_dir}/stage1_traces.png', dpi=150)
    plt.close()

    # Save results
    with open(f'{output_dir}/stage1_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir}/")

    return results


# ============================================================================
# Quick Test Function
# ============================================================================

def quick_test(n_steps: int = 100):
    """
    Quick test run with minimal steps for debugging.
    """
    print("Running quick test (n_steps={})...".format(n_steps))

    samples, sampler = run_stage1_mcmc(
        n_walkers=22,  # Minimal for testing
        n_steps=n_steps,
        n_burn=20,
        n_cores=2,
        output_file='results/test_chains.h5'
    )

    analyze_stage1_results(samples, sampler, output_dir='results/test')

    print("\nQuick test complete!")


if __name__ == '__main__':
    # Full production run
    samples, sampler = run_stage1_mcmc()
    analyze_stage1_results(samples, sampler)
