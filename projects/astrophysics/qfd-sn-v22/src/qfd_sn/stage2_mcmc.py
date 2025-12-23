"""
Stage 2: Global MCMC Parameter Fitting

Fits QFD model parameters (k_J_correction, η', ξ, σ_ln_A) using MCMC.

Input:
    - Stage 1 results (filtered): CSV with ln_A, stretch, z for each SN

Output:
    - Best-fit parameters with uncertainties
    - MCMC posterior samples
    - Convergence diagnostics

Physical Model:
    ln_A_predicted(z) = (η' + ξ) × z

    Where ln_A_observed comes from Stage 1 fitting.
"""

import numpy as np
import pandas as pd
import emcee
import json
from pathlib import Path
from typing import Tuple, Dict, Any
import matplotlib.pyplot as plt
from tqdm import tqdm

from . import cosmology
from .lean_validation import constraints


def load_stage1_results(filepath: str) -> pd.DataFrame:
    """
    Load Stage 1 filtered results.

    Args:
        filepath: Path to stage1_results_filtered.csv

    Returns:
        DataFrame with columns: name, z, ln_A, ln_A_err, stretch, chi2_dof
    """
    df = pd.read_csv(filepath)

    # Handle column name variations
    if 'snid' in df.columns and 'name' not in df.columns:
        df['name'] = df['snid']

    # Ensure required columns exist
    required_base = ['z', 'ln_A']
    missing = [col for col in required_base if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # If ln_A_err missing, estimate from residuals
    if 'ln_A_err' not in df.columns:
        print("  Note: ln_A_err not in data, using constant uncertainty σ=0.2")
        df['ln_A_err'] = 0.2  # Typical per-SN uncertainty

    print(f"Loaded {len(df)} SNe from Stage 1")
    print(f"  Redshift range: {df['z'].min():.3f} to {df['z'].max():.3f}")
    print(f"  ln_A range: {df['ln_A'].min():.3f} to {df['ln_A'].max():.3f}")
    print(f"  ln_A scatter: {df['ln_A'].std():.3f}")

    return df


def log_prior(theta: np.ndarray) -> float:
    """
    Log prior probability for parameters.

    Parameters:
        theta = [k_J_correction, eta_prime, xi, sigma_ln_A]

    Priors:
        k_J_correction ~ N(51, 20²)  # Centered on successful V21 result
        eta_prime ~ N(-3, 5²)         # Weakly informative
        xi ~ N(-6, 3²)                # Centered on thermal broadening
        sigma_ln_A ~ Uniform(0.1, 3)  # Positive scatter

    Also enforces Lean constraints as hard boundaries.
    """
    k_J_corr, eta_prime, xi, sigma_ln_A = theta

    # Lean constraint hard boundaries
    k_J_total = 70.0 + k_J_corr
    if not (constraints.LeanConstraints.K_J_MIN <= k_J_total <= constraints.LeanConstraints.K_J_MAX):
        return -np.inf
    if not (constraints.LeanConstraints.ETA_PRIME_MIN <= eta_prime <= constraints.LeanConstraints.ETA_PRIME_MAX):
        return -np.inf
    if not (constraints.LeanConstraints.XI_MIN <= xi <= constraints.LeanConstraints.XI_MAX):
        return -np.inf
    if not (constraints.LeanConstraints.SIGMA_LN_A_MIN <= sigma_ln_A <= constraints.LeanConstraints.SIGMA_LN_A_MAX):
        return -np.inf

    # Gaussian priors
    log_p = 0.0

    # k_J_correction prior
    log_p += -0.5 * ((k_J_corr - 51.0) / 20.0)**2

    # eta_prime prior
    log_p += -0.5 * ((eta_prime - (-3.0)) / 5.0)**2

    # xi prior
    log_p += -0.5 * ((xi - (-6.0)) / 3.0)**2

    # sigma_ln_A uniform prior [0.1, 3.0]
    if not (0.1 <= sigma_ln_A <= 3.0):
        return -np.inf

    return log_p


def log_likelihood(theta: np.ndarray, z: np.ndarray, ln_A_obs: np.ndarray,
                   ln_A_err: np.ndarray) -> float:
    """
    Log likelihood for QFD model.

    Model:
        ln_A_predicted(z) = (η' + ξ) × z

    Likelihood:
        ln_A_obs ~ N(ln_A_predicted, √(ln_A_err² + σ²))

    Args:
        theta: [k_J_correction, eta_prime, xi, sigma_ln_A]
        z: Redshifts
        ln_A_obs: Observed ln_A from Stage 1
        ln_A_err: Uncertainties on ln_A

    Returns:
        Log likelihood value
    """
    k_J_corr, eta_prime, xi, sigma_ln_A = theta

    # Predict ln_A from QFD model
    ln_A_pred = cosmology.ln_amplitude_predicted(z, eta_prime, xi)

    # Total uncertainty (measurement + intrinsic scatter)
    sigma_total = np.sqrt(ln_A_err**2 + sigma_ln_A**2)

    # Gaussian likelihood
    chi2 = np.sum(((ln_A_obs - ln_A_pred) / sigma_total)**2)
    log_like = -0.5 * chi2 - np.sum(np.log(sigma_total)) - 0.5 * len(z) * np.log(2 * np.pi)

    return log_like


def log_probability(theta: np.ndarray, z: np.ndarray, ln_A_obs: np.ndarray,
                   ln_A_err: np.ndarray) -> float:
    """Log posterior = log prior + log likelihood."""
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, z, ln_A_obs, ln_A_err)


def run_mcmc(
    data: pd.DataFrame,
    nwalkers: int = 32,
    nsteps: int = 4000,
    nburn: int = 1000,
    progress: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run MCMC to fit QFD parameters.

    Args:
        data: Stage 1 results DataFrame
        nwalkers: Number of MCMC walkers
        nsteps: Number of MCMC steps
        nburn: Number of burn-in steps to discard
        progress: Show progress bar

    Returns:
        (samples, best_params) where:
            samples: MCMC posterior samples (nwalkers * (nsteps-nburn), 4)
            best_params: Median parameter values
    """
    # Extract data
    z = data['z'].values
    ln_A_obs = data['ln_A'].values
    ln_A_err = data['ln_A_err'].values

    # Initial guess (start near successful V21 values)
    ndim = 4
    p0_center = np.array([51.0, -0.05, -6.4, 1.6])

    # Initialize walkers in small ball around initial guess
    p0 = p0_center + 1e-2 * np.random.randn(nwalkers, ndim)

    # Ensure all walkers start in valid region
    for i in range(nwalkers):
        while not np.isfinite(log_prior(p0[i])):
            p0[i] = p0_center + 1e-2 * np.random.randn(ndim)

    # Set up MCMC sampler
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_probability,
        args=(z, ln_A_obs, ln_A_err)
    )

    print(f"\nRunning MCMC:")
    print(f"  Walkers: {nwalkers}")
    print(f"  Steps: {nsteps}")
    print(f"  Burn-in: {nburn}")
    print(f"  Parameters: k_J_correction, eta_prime, xi, sigma_ln_A")

    # Run MCMC
    if progress:
        for _ in tqdm(sampler.sample(p0, iterations=nsteps), total=nsteps, desc="MCMC"):
            pass
    else:
        sampler.run_mcmc(p0, nsteps, progress=False)

    # Get samples (discard burn-in)
    samples = sampler.get_chain(discard=nburn, flat=True)

    # Get best-fit parameters (median of posterior)
    best_params = np.median(samples, axis=0)

    # Compute acceptance fraction
    acc_frac = np.mean(sampler.acceptance_fraction)
    print(f"\nMCMC complete:")
    print(f"  Acceptance fraction: {acc_frac:.3f}")
    print(f"  Total samples: {len(samples):,}")

    # Validate best-fit against Lean constraints
    k_J_total = 70.0 + best_params[0]
    passed, results = constraints.validate_parameters(
        k_J_total=k_J_total,
        eta_prime=best_params[1],
        xi=best_params[2],
        sigma_ln_A=best_params[3]
    )

    print(f"\nLean Constraint Validation:")
    for param, (ok, msg) in results.items():
        status = "✅" if ok else "❌"
        print(f"  {status} {msg}")

    if not passed:
        print("\n⚠️  WARNING: Some Lean constraints violated!")

    return samples, best_params


def save_results(
    samples: np.ndarray,
    best_params: np.ndarray,
    data: pd.DataFrame,
    output_dir: str
) -> None:
    """
    Save MCMC results to disk.

    Args:
        samples: MCMC posterior samples
        best_params: Best-fit parameter values
        data: Input Stage 1 data
        output_dir: Directory to save results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Compute parameter statistics
    param_names = ['k_J_correction', 'eta_prime', 'xi', 'sigma_ln_A']
    param_stats = {}

    for i, name in enumerate(param_names):
        param_stats[name] = {
            'median': float(np.median(samples[:, i])),
            'mean': float(np.mean(samples[:, i])),
            'std': float(np.std(samples[:, i])),
            'percentile_16': float(np.percentile(samples[:, i], 16)),
            'percentile_84': float(np.percentile(samples[:, i], 84)),
        }

    # Validate against Lean constraints
    k_J_total = 70.0 + param_stats['k_J_correction']['median']
    lean_passed, lean_results = constraints.validate_parameters(
        k_J_total=k_J_total,
        eta_prime=param_stats['eta_prime']['median'],
        xi=param_stats['xi']['median'],
        sigma_ln_A=param_stats['sigma_ln_A']['median']
    )

    # Create summary JSON
    summary = {
        'n_sne': len(data),
        'best_fit_params': param_stats,
        'k_J_total': 70.0 + param_stats['k_J_correction']['median'],
        'mcmc_config': {
            'nwalkers': 32,
            'nsteps': 4000,
            'nburn': 1000,
            'total_samples': len(samples)
        },
        'lean_validation': {
            'all_passed': lean_passed,
            'details': {k: v[1] for k, v in lean_results.items()}
        }
    }

    # Save summary
    with open(output_path / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {output_dir}")
    print(f"  - summary.json: Parameter values and statistics")
    print(f"  - samples.npz: Full MCMC posterior samples")

    # Save samples (compressed)
    np.savez_compressed(
        output_path / 'samples.npz',
        samples=samples,
        param_names=param_names
    )

    print(f"\n{'='*60}")
    print("STAGE 2 COMPLETE")
    print(f"{'='*60}")
    print(f"\nBest-fit Parameters:")
    print(f"  k_J_total = {70.0 + param_stats['k_J_correction']['median']:.4f} ± {param_stats['k_J_correction']['std']:.4f} km/s/Mpc")
    print(f"  η' = {param_stats['eta_prime']['median']:.4f} ± {param_stats['eta_prime']['std']:.4f}")
    print(f"  ξ  = {param_stats['xi']['median']:.4f} ± {param_stats['xi']['std']:.4f}")
    print(f"  σ_ln_A = {param_stats['sigma_ln_A']['median']:.4f} ± {param_stats['sigma_ln_A']['std']:.4f}")
    print(f"\nLean Validation: {'✅ ALL PASS' if lean_passed else '❌ SOME FAILED'}")


def main():
    """Main Stage 2 execution."""
    import argparse

    parser = argparse.ArgumentParser(description='Stage 2: Global MCMC Parameter Fitting')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to stage1_results_filtered.csv')
    parser.add_argument('--output', type=str, default='results/stage2',
                       help='Output directory')
    parser.add_argument('--nwalkers', type=int, default=32,
                       help='Number of MCMC walkers')
    parser.add_argument('--nsteps', type=int, default=4000,
                       help='Number of MCMC steps')
    parser.add_argument('--nburn', type=int, default=1000,
                       help='Number of burn-in steps')

    args = parser.parse_args()

    # Load data
    print("="*60)
    print("STAGE 2: GLOBAL MCMC PARAMETER FITTING")
    print("="*60)

    data = load_stage1_results(args.input)

    # Run MCMC
    samples, best_params = run_mcmc(
        data,
        nwalkers=args.nwalkers,
        nsteps=args.nsteps,
        nburn=args.nburn
    )

    # Save results
    save_results(samples, best_params, data, args.output)


if __name__ == '__main__':
    main()
