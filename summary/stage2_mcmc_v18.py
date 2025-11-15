#!/usr/bin/env python3
"""
Stage 2 MCMC for V18 - Reparameterized QFD Model

This script implements the Stage 2 MCMC fit using the reparameterized
QFD redshift model defined in the V18 blueprint.
"""

import argparse
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import json

# Add core directory to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))

from v17_qfd_model import (
    calculate_z_drag,
    calculate_z_local,
    predict_apparent_magnitude,
    C_KM_S
)
from v17_lightcurve_model import ln_A_pred, K_J_BASELINE
from v17_data import LightcurveLoader

# --- Data Loading ---
def load_real_data(lightcurve_path, stage1_dir, max_sne=50):
    """
    Loads real data for the V18 Stage 2 fit (ln_A basis).
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

    count = 0
    for snid, lc_data in sorted(all_lcs.items()):
        if len(z_obs_list) >= max_sne:
            break

        persn_file = stage1_path / snid / "persn_best.npy"
        if persn_file.exists():
            try:
                params = np.load(persn_file)
                A_plasma, beta, ln_A = params[1], params[2], params[3]
                
                if not np.isfinite(ln_A):
                    print(f"  - WARNING: Skipping SNID {snid} due to invalid ln_A value: {ln_A}")
                    continue

                z_plasma = A_plasma * (lambda_b_nm / lambda_eff_nm)**beta
                if z_plasma > lc_data.z:
                    print(f"  - WARNING: Skipping SNID {snid} because z_plasma ({z_plasma:.4f}) > z_obs ({lc_data.z:.4f})")
                    continue
                
                z_obs_list.append(lc_data.z)
                ln_A_obs_list.append(ln_A)
                A_plasma_list.append(A_plasma)
                beta_list.append(beta)
                
            except Exception as e:
                print(f"Warning: Could not load data for SN {snid}. Error: {e}")

    if not z_obs_list:
        raise ValueError("No valid Stage 1 data found. Please run Stage 1 first.")

    print(f"Successfully loaded {len(z_obs_list)} supernovae.")
    
    return {
        'z_obs': jnp.array(z_obs_list),
        'ln_A_obs': jnp.array(ln_A_obs_list),
        'A_plasma': jnp.array(A_plasma_list),
        'beta': jnp.array(beta_list),
    }


# --- NumPyro MCMC Model ---
def numpyro_model_v18(z_obs, ln_A_obs, A_plasma, beta):
    """
    Defines the MCMC model for fitting the global QFD parameters using ln_A basis.
    """
    # Priors for global parameters
    k_J_correction = numpyro.sample('k_J_correction', dist.Normal(0.0, 2.0)) # Correction to K_J_BASELINE
    eta_prime = numpyro.sample('eta_prime', dist.Normal(0.0, 0.5))
    xi = numpyro.sample('xi', dist.Normal(0.0, 0.5))
    sigma_ln_A = numpyro.sample('sigma_ln_A', dist.HalfNormal(0.1)) + 1e-3

    ln_A_predicted = ln_A_pred(z_obs, k_J_correction, eta_prime, xi)

    # Likelihood calculation
    numpyro.factor("ln_A_obs_log_prob", dist.Normal(ln_A_predicted, sigma_ln_A).log_prob(ln_A_obs).sum())


# --- Main Execution ---
def main_v18(args):
    print("--- V18 Stage 2 MCMC (ln_A Basis) ---")
    print(f"JAX local device count: {jax.local_device_count()}")
    jax.config.update("jax_enable_x64", True)
    numpyro.set_host_device_count(4)
    
    try:
        data = load_real_data(args.lightcurves, args.stage1_results, args.max_sne)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        return 1
    
    print("Setting up MCMC sampler...")
    init_params = {
        'k_J_correction': 0.0,
        'eta_prime': 0.0,
        'xi': 0.0,
        'sigma_ln_A': 0.1, 
    }
    kernel = NUTS(
        numpyro_model_v18,
        init_strategy=numpyro.infer.init_to_value(values=init_params)
    )
    mcmc = MCMC(
        kernel,
        num_warmup=args.nwarmup,
        num_samples=args.nsamples,
        num_chains=args.nchains,
        progress_bar=True
    )
    
    print(f"Running MCMC on {len(data['z_obs'])} supernovae...")
    rng_key = jax.random.PRNGKey(0)
    mcmc.run(rng_key, data['z_obs'], data['ln_A_obs'], data['A_plasma'], data['beta'])
    
    print("\n--- MCMC complete. Results: ---")
    mcmc.print_summary()
    
    samples = mcmc.get_samples()
    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)
    
    samples_np = {k: np.array(v) for k, v in samples.items()}
    np.savez(out_path / "samples.npz", **samples_np)
    print(f"\nSaved samples to: {out_path / 'samples.npz'}")

    summary = {}
    for key, value in samples.items():
        summary[key] = {
            'median': float(np.median(value)),
            'mean': float(np.mean(value)),
            'std': float(np.std(value))
        }
    
    with open(out_path / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to: {out_path / 'summary.json'}")
    print("\nDone.")
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V18 Stage 2 MCMC (ln_A Basis)")
    parser.add_argument('--lightcurves', type=str, default='pipeline/data/lightcurves_unified_v2_min3.csv', help='Path to lightcurves data file')
    parser.add_argument('--stage1-results', type=str, default='../results/v15_clean/stage1_fullscale', help='Path to Stage 1 results directory')
    parser.add_argument('--out', type=str, default='v18/results/stage2_test', help='Output directory for results')
    parser.add_argument('--max-sne', type=int, default=50, help='Maximum number of SNe to use')
    parser.add_argument('--nchains', type=int, default=4, help='Number of MCMC chains')
    parser.add_argument('--nsamples', type=int, default=2000, help='Number of samples per chain')
    parser.add_argument('--nwarmup', type=int, default=1000, help='Number of warmup steps')
    
    args = parser.parse_args()
    sys.exit(main_v18(args))