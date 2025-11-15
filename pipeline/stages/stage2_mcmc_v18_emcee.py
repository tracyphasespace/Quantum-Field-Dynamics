#!/usr/bin/env python3
"""
Stage 2 MCMC for V18 - ln_A basis, emcee version

Drop-in replacement for NumPyro Stage 2 when you want V15-style
multi-core CPU performance and very fast runs.
"""

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import emcee
import multiprocessing as mp

import jax
import jax.numpy as jnp

# Make sure we use CPU here; GPU is not needed for this tiny model
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))

from v17_lightcurve_model import ln_A_pred
from v17_data import LightcurveLoader


def load_real_data(lightcurve_path: str, stage1_dir: str, max_sne: int = 50) -> Dict[str, np.ndarray]:
    """
    Same logic as stage2_mcmc_v18.py, but returning NumPy arrays for emcee.
    """
    print(f"Loading data for up to {max_sne} supernovae...")
    loader = LightcurveLoader(Path(lightcurve_path))
    all_lcs = loader.load()

    z_obs_list, ln_A_obs_list, A_plasma_list, beta_list = [], [], [], []

    stage1_path = Path(stage1_dir)
    if not stage1_path.exists():
        raise FileNotFoundError(f"Stage 1 directory not found: {stage1_path}")

    lambda_eff_nm = 440.0
    lambda_b_nm = 440.0

    for snid, lc_data in sorted(all_lcs.items()):
        if len(z_obs_list) >= max_sne:
            break

        persn_file = stage1_path / snid / "persn_best.npy"
        if not persn_file.exists():
            continue

        try:
            params = np.load(persn_file)
            A_plasma, beta, ln_A = params[1], params[2], params[3]

            if not np.isfinite(ln_A):
                print(f"  - Skipping SNID {snid}: invalid ln_A = {ln_A}")
                continue

            z_plasma = A_plasma * (lambda_b_nm / lambda_eff_nm) ** beta
            if z_plasma > lc_data.z:
                print(
                    f"  - Skipping SNID {snid}: z_plasma={z_plasma:.4f} > z_obs={lc_data.z:.4f}"
                )
                continue

            z_obs_list.append(lc_data.z)
            ln_A_obs_list.append(ln_A)
            A_plasma_list.append(A_plasma)
            beta_list.append(beta)

        except Exception as e:
            print(f"  - Warning: Could not load SN {snid}: {e}")

    if not z_obs_list:
        raise ValueError("No valid Stage 1 data found. Please run Stage 1 first.")

    print(f"Successfully loaded {len(z_obs_list)} supernovae.")

    return {
        "z_obs": np.array(z_obs_list, dtype=np.float64),
        "ln_A_obs": np.array(ln_A_obs_list, dtype=np.float64),
        "A_plasma": np.array(A_plasma_list, dtype=np.float64),
        "beta": np.array(beta_list, dtype=np.float64),
    }


# JIT-compiled ln_A_pred batch function for speed (CPU-XLA)
_lnA_pred_batch_jax = jax.jit(
    lambda z, k_J_corr, eta_prime, xi: ln_A_pred(z, k_J_corr, eta_prime, xi)
)


def log_prior(theta):
    """
    Priors for (k_J_correction, eta_prime, xi, sigma_ln_A)
    """
    k_J_corr, eta_prime, xi, sigma_ln_A = theta

    # k_J_correction ~ N(0, 2)
    if not (-20.0 < k_J_corr < 20.0):
        return -np.inf
    lp = -0.5 * (k_J_corr / 4.0) ** 2

    # eta_prime ~ N(0, 0.5)
    if not (-6.0 < eta_prime < 6.0):
        return -np.inf
    lp += -0.5 * (eta_prime / 1.0) ** 2

    # xi ~ N(0, 0.5)
    if not (-6.0 < xi < 6.0):
        return -np.inf
    lp += -0.5 * (xi / 1.0) ** 2

    # sigma_ln_A ~ HalfNormal(0.1) + epsilon
    if sigma_ln_A <= 0.0 or sigma_ln_A > 1.0:
        return -np.inf
    # log p(sigma) ∝ -0.5 * (sigma/0.1)^2 - log(sigma)
    lp += -0.5 * (sigma_ln_A / 0.2) ** 2 - np.log(sigma_ln_A)

    return lp


z_obs = None
ln_A_obs = None

def log_prob(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    
    k_J_corr, eta_prime, xi, sigma_ln_A = theta
    sigma = max(sigma_ln_A, 1e-3)

    ln_A_pred_np = np.array(
        _lnA_pred_batch_jax(jnp.array(z_obs), k_J_corr, eta_prime, xi)
    )
    resid = ln_A_obs - ln_A_pred_np
    ll = -0.5 * np.sum((resid / sigma) ** 2 + np.log(2.0 * np.pi * sigma**2))
    
    return lp + ll

def main():
    parser = argparse.ArgumentParser(description="V18 Stage 2 MCMC (ln_A basis, emcee)")
    parser.add_argument(
        "--lightcurves",
        type=str,
        default="pipeline/data/lightcurves_unified_v2_min3.csv",
    )
    parser.add_argument(
        "--stage1-results",
        type=str,
        default="../results/v15_clean/stage1_fullscale",
    )
    parser.add_argument(
        "--out", type=str, default="v18/results/stage2_emcee_lnA", help="Output dir"
    )
    parser.add_argument("--max-sne", type=int, default=50)
    parser.add_argument("--nwalkers", type=int, default=32)
    parser.add_argument("--nsteps", type=int, default=4000)
    parser.add_argument("--nburn", type=int, default=1000)
    parser.add_argument("--ncores", type=int, default=8)
    args = parser.parse_args()

    global z_obs, ln_A_obs
    data = load_real_data(args.lightcurves, args.stage1_results, args.max_sne)
    z_obs = data["z_obs"]
    ln_A_obs = data["ln_A_obs"]

    ndim = 4  # (k_J_correction, eta_prime, xi, sigma_ln_A)
    nwalkers = args.nwalkers

    # Initialize walkers near the prior center
    p0 = np.zeros((nwalkers, ndim))
    p0[:, 0] = np.random.normal(0.0, 0.5, size=nwalkers)   # k_J_corr
    p0[:, 1] = np.random.normal(0.0, 0.2, size=nwalkers)   # eta_prime
    p0[:, 2] = np.random.normal(0.0, 0.2, size=nwalkers)   # xi
    p0[:, 3] = np.abs(np.random.normal(0.1, 0.02, size=nwalkers))  # sigma_ln_A

    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Running emcee with {nwalkers} walkers, {args.nsteps} steps...")
    with mp.Pool(processes=args.ncores) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, pool=pool)
        sampler.run_mcmc(p0, args.nsteps, progress=True)

    # Discard burn-in and flatten
    samples = sampler.get_chain(discard=args.nburn, flat=True)
    np.savez(out_path / "samples.npz", samples=samples)

    # Summaries
    summary = {}
    param_names = ["k_J_correction", "eta_prime", "xi", "sigma_ln_A"]
    for i, name in enumerate(param_names):
        vals = samples[:, i]
        summary[name] = {
            "median": float(np.median(vals)),
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
        }

    with open(out_path / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved samples to {out_path / 'samples.npz'}")
    print(f"Saved summary to {out_path / 'summary.json'}")


if __name__ == "__main__":
    main()
