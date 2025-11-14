#!/usr/bin/env python3
"""
Stage 2 MCMC for V17 - Unambiguous QFD Model (Pure Functional, No Dicts)

This script implements the Stage 2 MCMC fit using the pure functional
QFD redshift model defined in v17/core/v17_qfd_model.py. It avoids passing
dictionaries into JAX-compiled functions.
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
    predict_apparent_magnitude
)
from v17_data import LightcurveLoader

# --- Data Loading ---
def load_real_data(lightcurve_path, stage1_dir, max_sne=50):
    """
    Loads real data for the V17 Stage 2 fit.
    """
    print(f"Loading data for up to {max_sne} supernovae...")
    loader = LightcurveLoader(Path(lightcurve_path))
    all_lcs = loader.load()
    
    z_obs_list, m_obs_list, A_plasma_list, beta_list = [], [], [], []
    
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
                
                m_obs = - (2.5 / np.log(10)) * ln_A
                if not np.isfinite(m_obs):
                    print(f"  - WARNING: Skipping SNID {snid} due to invalid m_obs value: {m_obs}")
                    continue

                z_plasma = A_plasma * (lambda_b_nm / lambda_eff_nm)**beta
                if z_plasma > lc_data.z:
                    print(f"  - WARNING: Skipping SNID {snid} because z_plasma ({z_plasma:.4f}) > z_obs ({lc_data.z:.4f})")
                    continue
                
                z_obs_list.append(lc_data.z)
                m_obs_list.append(m_obs)
                A_plasma_list.append(A_plasma)
                beta_list.append(beta)
                
            except Exception as e:
                print(f"Warning: Could not load data for SN {snid}. Error: {e}")

    if not z_obs_list:
        raise ValueError("No valid Stage 1 data found. Please run Stage 1 first.")

    print(f"Successfully loaded {len(z_obs_list)} supernovae.")
    
    return {
        'z_obs': jnp.array(z_obs_list),
        'm_obs': jnp.array(m_obs_list),
        'A_plasma': jnp.array(A_plasma_list),
        'beta': jnp.array(beta_list),
    }


# --- Inverse Solver (using pure functions) ---
@partial(jit, static_argnames=())
def find_distance_for_redshift(k_J, eta_prime, A_plasma, beta, z_obs):
    """
    Solves the inverse problem for distance D.
    """
    def residual_func(D):
        z_drag = calculate_z_drag(k_J, D)
        z_local = calculate_z_local(eta_prime, A_plasma, beta, D)
        z_pred = (1 + z_drag) * (1 + z_local) - 1
        return z_pred - z_obs

    def find_root(f, low, high, tol=1e-5, max_iter=100):
        def cond_fun(state):
            low, high, i = state
            return (jnp.abs(high - low) > tol) & (i < max_iter)
        def body_fun(state):
            low, high, i = state
            mid = (low + high) / 2.0
            f_mid = f(mid)
            f_low = f(low)
            new_low = jnp.where(jnp.sign(f_mid) == jnp.sign(f_low), mid, low)
            new_high = jnp.where(jnp.sign(f_mid) != jnp.sign(f_low), mid, high)
            return new_low, new_high, i + 1
        final_low, final_high, _ = jax.lax.while_loop(cond_fun, body_fun, (low, high, 0))
        return (final_low + final_high) / 2.0

    low, high = 1.0, 20000.0
    f_low = residual_func(low)
    f_high = residual_func(high)
    is_bracketed = (jnp.sign(f_low) != jnp.sign(f_high))
    return jnp.where(is_bracketed, find_root(residual_func, low, high), jnp.nan)


# --- NumPyro MCMC Model ---
def numpyro_model_v17(z_obs, m_obs, A_plasma, beta):
    """
    Defines the MCMC model for fitting the global QFD parameters.
    """
    k_J = numpyro.sample('k_J', dist.Normal(70.0, 10.0))
    eta_prime = numpyro.sample('eta_prime', dist.Normal(0.0, 1.0))
    xi = numpyro.sample('xi', dist.Normal(0.0, 1.0))
    sigma_m = numpyro.sample('sigma_m', dist.HalfNormal(0.2))

    def process_single_sn(z_obs_single, A_plasma_single, beta_single):
        distance = find_distance_for_redshift(k_J, eta_prime, A_plasma_single, beta_single, z_obs_single)
        z_drag = calculate_z_drag(k_J, distance)
        z_local = calculate_z_local(eta_prime, A_plasma_single, beta_single, distance)
        m_pred = predict_apparent_magnitude(xi, distance, z_drag, z_local)
        return m_pred

    m_predicted = vmap(process_single_sn)(z_obs, A_plasma, beta)

    is_nan = jnp.isnan(m_predicted)
    m_predicted_sanitized = jnp.where(is_nan, 0.0, m_predicted)
    log_likelihood = dist.Normal(m_predicted_sanitized, sigma_m).log_prob(m_obs)
    penalized_log_likelihood = jnp.where(is_nan, -1e18, log_likelihood)
    
    numpyro.factor("m_obs_log_prob", jnp.sum(penalized_log_likelihood))


# --- Main Execution ---
def main(args):
    print("--- V17 Stage 2 MCMC (Pure Functional, No Dicts) ---")
    
    try:
        data = load_real_data(args.lightcurves, args.stage1_results, args.max_sne)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        return 1
    
    print("Setting up MCMC sampler...")
    init_params = {'k_J': 70.0, 'eta_prime': 0.0, 'xi': 0.0, 'sigma_m': 0.2}
    kernel = NUTS(
        numpyro_model_v17,
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
    mcmc.run(rng_key, data['z_obs'], data['m_obs'], data['A_plasma'], data['beta'])
    
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
    parser = argparse.ArgumentParser(description="V17 Stage 2 MCMC (Pure Functional, No Dicts)")
    parser.add_argument('--lightcurves', type=str, default='v17/data/lightcurves_unified_v2_min3.csv', help='Path to lightcurves data file')
    parser.add_argument('--stage1-results', type=str, default='../results/v15_clean/stage1_fullscale', help='Path to Stage 1 results directory')
    parser.add_argument('--out', type=str, default='v17/results/stage2_test', help='Output directory for results')
    parser.add_argument('--max-sne', type=int, default=50, help='Maximum number of SNe to use')
    parser.add_argument('--nchains', type=int, default=2, help='Number of MCMC chains')
    parser.add_argument('--nsamples', type=int, default=1000, help='Number of samples per chain')
    parser.add_argument('--nwarmup', type=int, default=500, help='Number of warmup steps')
    
    args = parser.parse_args()
    sys.exit(main(args))