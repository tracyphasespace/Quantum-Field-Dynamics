#!/usr/bin/env python3
"""
Stage 2: Global MCMC Sampling with NumPyro (FULLY GPU-OPTIMIZED)

Uses NumPyro's NUTS sampler for:
- Fully GPU-native sampling (10-100× faster than emcee)
- Adaptive step size (no tuning required)
- Better convergence diagnostics
- Automatic parallelization across chains

Expected speedup: 50-100× vs emcee version

Usage:
    python stage2_mcmc_numpyro.py \
        --stage1-results results/v15_stage1_production \
        --lightcurves ../../data/unified/lightcurves_unified_v2_min3.csv \
        --out results/v15_stage2_mcmc_numpyro \
        --nchains 4 \
        --nsamples 2000 \
        --nwarmup 1000
"""

import os
import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import time
from functools import partial

# Import pipeline data structures for type-safe parameter handling
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))
from pipeline_io import PerSNParams, GlobalParams

# Import NumPyro and JAX (will use GPU by default if available)
import numpyro
import jax
import jax.numpy as jnp
from jax import vmap

jax.config.update("jax_enable_x64", True)
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import arviz as az

# Import model (from core directory)
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))
from v15_model import log_likelihood_single_sn_jax, ln_A_pred_batch

# Import summary writer
# import sys
# sys.path.insert(0, str(Path(__file__).parent.parent / 'tools'))
# from write_stage2_summary import write_stage2_summary, Standardizer

# Dummy Standardizer class (since write_stage2_summary module unavailable)
class Standardizer:
    def __init__(self, means, scales):
        self.means = means
        self.scales = scales

# Magnitude-to-natural-log conversion constant
K_MAG_PER_LN = 2.5 / np.log(10.0)  # ≈ 1.0857


def _ensure_ln_A_natural(ln_A_obs_array: np.ndarray) -> np.ndarray:
    """
    Convert Stage-1 alpha to natural-log amplitude if it's actually in magnitudes.

    Canonical definition:
    - α ≡ ln(A) (natural-log amplitude)
    - μ_obs = μ_th - K·α, where K = 2.5/ln(10) ≈ 1.0857
    - Larger distances → smaller flux → more negative α

    Stage 1 may output α in magnitude space (Δμ, positive values like 15-30).
    Stage 2 expects natural-log amplitude (negative values like -5 to -70).

    Heuristic: if median(|alpha|) > 5, treat as magnitude residuals and convert:
        α_nat = -α_mag / K

    Args:
        ln_A_obs_array: Alpha values from Stage 1

    Returns:
        Alpha in natural-log space (negative, decreasing with z)
    """
    a = np.asarray(ln_A_obs_array, dtype=float)

    if not np.isfinite(a).all():
        raise ValueError("alpha_obs contains NaN/inf")

    median_abs = np.median(np.abs(a))

    if median_abs > 5.0:
        # Looks like magnitudes (tens). Convert to natural log amplitude.
        print(f"[ALPHA CONVERSION] Detected magnitude-space alpha (median |α| = {median_abs:.1f})")
        print(f"[ALPHA CONVERSION] Converting: α_nat = -α_mag / K")
        a_nat = -a / K_MAG_PER_LN
        print(f"[ALPHA CONVERSION] Before: [{a.min():.1f}, {np.median(a):.1f}, {a.max():.1f}]")
        print(f"[ALPHA CONVERSION] After:  [{a_nat.min():.1f}, {np.median(a_nat):.1f}, {a_nat.max():.1f}]")
        return a_nat
    else:
        print(f"[ALPHA CONVERSION] Alpha already in natural-log space (median |α| = {median_abs:.1f})")
        return a


def _standardize_features(z):
    """
    Standardize basis features to zero-mean, unit-std.

    This is a pure reparameterization (no physics change) that dramatically
    reduces posterior curvature and correlation, eliminating NUTS divergences.

    Args:
        z: Redshift array (JAX or numpy)

    Returns:
        Φ: Standardized feature matrix [N, 3]
        stats: Dict with normalization constants {'m': [m1, m2, m3], 's': [s1, s2, s3]}
    """
    # Construct raw features (same as in v15_model.py)
    phi1 = jnp.log1p(z)        # ln(1+z)
    phi2 = z                    # linear
    phi3 = z / (1.0 + z)       # saturating

    # Standardize to zero-mean, unit-std (numerically stable)
    def zscore(x):
        m = jnp.mean(x)
        s = jnp.std(x) + 1e-12  # avoid division by zero
        return (x - m) / s, (m, s)

    φ1, (m1, s1) = zscore(phi1)
    φ2, (m2, s2) = zscore(phi2)
    φ3, (m3, s3) = zscore(phi3)

    Φ = jnp.stack([φ1, φ2, φ3], axis=-1)  # [N, 3]
    stats = {'m': jnp.array([m1, m2, m3]), 's': jnp.array([s1, s2, s3])}

    return Φ, stats


def _orthogonalize_basis(Φ):
    """
    Orthogonalize basis functions using QR decomposition.

    This eliminates collinearity between φ₁, φ₂, φ₃ which have correlations > 0.99
    and condition number ~ 2×10⁵. After orthogonalization, features are uncorrelated
    and the sign ambiguity is resolved.

    Args:
        Φ: Standardized feature matrix [N, 3]

    Returns:
        Q: Orthogonalized features [N, 3] (columns are orthonormal)
        R: Upper triangular transformation matrix [3, 3]
            To recover original: Φ = Q @ R
    """
    Q, R = jnp.linalg.qr(Φ)
    return Q, R


def load_stage1_results(stage1_dir, lightcurves_dict, quality_cut=2000):
    """Load all Stage 1 results"""
    results = {}
    failed = []

    for result_dir in Path(stage1_dir).iterdir():
        if not result_dir.is_dir():
            continue

        snid = result_dir.name
        metrics_file = result_dir / "metrics.json"
        persn_file = result_dir / "persn_best.npy"

        if not metrics_file.exists() or not persn_file.exists():
            continue

        try:
            # Load metrics
            with open(metrics_file) as f:
                metrics = json.load(f)

            # Load per-SN parameters
            persn_best = np.load(persn_file)

            # Get n_obs from lightcurve
            if snid not in lightcurves_dict:
                failed.append(snid)
                continue

            lc = lightcurves_dict[snid]
            n_obs = len(lc.mjd)

            # Quality filter: chi2
            # NOTE: chi2 in Stage 1 is actually -2*logL, so negative is GOOD!
            # Filter uses absolute value to catch extreme cases either way
            chi2 = metrics['chi2']
            if abs(chi2) > quality_cut:
                failed.append(snid)
                continue

            # Filter: Minimum iterations (ensure convergence, not stuck)
            # Note: With relaxed tolerance, many converge in 1-2 iters, so use very low threshold
            iters = metrics.get('iters', 0)
            if iters < 1:
                failed.append(snid)
                continue

            # CRITICAL: Filter out SNe with ln_A at boundaries (indicates failed fit)
            ln_A = persn_best[3]
            if ln_A >= 28 or ln_A <= -28:  # Near ±30 boundaries
                failed.append(snid)
                continue

            # Filter: Check other parameters not at boundaries
            # NOTE: t0 is stored as absolute MJD, not offset from peak, so we can't check boundaries
            A_plasma, beta = persn_best[1], persn_best[2]
            if A_plasma <= 0.001 or A_plasma >= 0.999:  # Near [0, 1] boundaries
                failed.append(snid)
                continue
            if beta <= 0.001 or beta >= 3.999:  # Near [0, 4] boundaries
                failed.append(snid)
                continue

            # Store result
            result = {
                'snid': snid,
                'chi2': chi2,
                'n_obs': n_obs,
                'persn_best': persn_best,
                'L_peak': metrics['L_peak'],
                'iters': metrics['iters']
            }
            results[snid] = result

        except Exception as e:
            print(f"  Warning: Failed to load {snid}: {e}")
            import traceback
            traceback.print_exc()
            failed.append(snid)
            continue

    print(f"  Loaded {len(results)} good SNe (chi2 < {quality_cut})")
    print(f"  Excluded {len(failed)} poor fits")

    return results

@partial(jax.jit, static_argnames=("cache_bust",))
def log_likelihood_ln_A_space(
    k_J: float,
    eta_prime: float,
    xi: float,
    z_batch: jnp.ndarray,  # Shape: (n_sne,)
    ln_A_obs_batch: jnp.ndarray,  # Shape: (n_sne,)
    *,
    cache_bust: int = 0,
) -> float:
    """
    Alpha-space likelihood: score globals by predicting alpha from (z; globals).

    Returns total log-likelihood (summed over all SNe).

    Uses ln_A_pred(z; k_J, eta_prime, xi) to compute residuals.
    No per-SN parameters, no lightcurve physics - just alpha prediction.

    Args:
        cache_bust: Static cache-busting token to force fresh JIT compilation
    """
    # Predict alpha from globals
    ln_A_th = ln_A_pred_batch(z_batch, k_J, eta_prime, xi)

    # Residuals
    r_ln_A = ln_A_obs_batch - ln_A_th

    # NOTE: Variance guard is enforced outside JIT (see preflight in run_alpha_space_mcmc)

    # Simple unweighted likelihood (can add sigma_ln_A later)
    logL = -0.5 * jnp.sum(r_ln_A**2)

    return logL

@jax.jit
def _log_likelihood_single_batch_jax(
    k_J: float,
    eta_prime: float,
    xi: float,
    persn_params_batch: jnp.ndarray,  # Shape: (batch_size, 4)
    L_peaks: jnp.ndarray,  # Shape: (batch_size,)
    photometries: List,  # List of JAX arrays (length = batch_size)
    redshifts: jnp.ndarray  # Shape: (batch_size,)
) -> jnp.ndarray:
    """
    JIT-compiled log-likelihood for a SINGLE batch of SNe.

    This gets compiled once per batch size, avoiding memory issues
    from compiling with all 4,727 SNe at once.

    Returns: Array of shape (batch_size,) with logL for each SN
    """
    global_params = (k_J, eta_prime, xi)

    def single_sn_logL(persn_params, L_peak, phot, z_obs):
        return log_likelihood_single_sn_jax(
            global_params, tuple(persn_params), L_peak, phot, z_obs
        )

    # Can't use vmap due to ragged arrays - use Python loop
    logLs_list = []
    for i in range(len(photometries)):
        logL = single_sn_logL(
            persn_params_batch[i],
            L_peaks[i],
            photometries[i],
            redshifts[i]
        )
        logLs_list.append(logL)

    logLs = jnp.array(logLs_list)
    return logLs


def log_likelihood_batch_jax(
    k_J: float,
    eta_prime: float,
    xi: float,
    persn_params_batch: jnp.ndarray,  # Shape: (n_sne, 4)
    L_peaks: jnp.ndarray,  # Shape: (n_sne,)
    photometries: List,  # List of JAX arrays
    redshifts: jnp.ndarray,  # Shape: (n_sne,)
    batch_size: int = 500  # Process 500 SNe at a time to avoid OOM
) -> jnp.ndarray:
    """
    Batched log-likelihood over multiple SNe.

    Splits data into batches of `batch_size` SNe to avoid memory issues
    during JAX JIT compilation. Each batch is JIT-compiled separately.

    Returns: Array of shape (n_sne,) with logL for each SN
    """
    n_sne = len(photometries)

    if n_sne <= batch_size:
        # Small enough to process in one go
        return _log_likelihood_single_batch_jax(
            k_J, eta_prime, xi,
            persn_params_batch, L_peaks, photometries, redshifts
        )

    # Process in batches
    logLs_all = []
    n_batches = (n_sne + batch_size - 1) // batch_size

    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min((batch_idx + 1) * batch_size, n_sne)

        batch_logLs = _log_likelihood_single_batch_jax(
            k_J, eta_prime, xi,
            persn_params_batch[start:end],
            L_peaks[start:end],
            photometries[start:end],
            redshifts[start:end]
        )
        logLs_all.append(batch_logLs)

    # Concatenate all batch results
    logLs = jnp.concatenate(logLs_all)
    return logLs

def numpyro_model_ln_A_space(Phi, ln_A_obs_batch, *,
                             constrain_signs='off',
                             standardizer=None,
                             R_ortho=None,
                             cache_bust: int = 0):
    """
    NumPyro model for global parameter inference using ln_A-space likelihood.

    Supports multiple sign constraint variants for A/B testing:
    - 'off': Unconstrained Normal(0,1) priors (default, current behavior)
    - 'informed': Physics-space with informed priors centered at paper values (RECOMMENDED)
    - 'alpha': Constrain standardized weights c ≤ 0 (force monotone decrease)
    - 'physics': Work in physics-space with k_J, η', ξ ≥ 0 (interpretable positivity)
    - 'ortho': Use orthogonalized basis to eliminate collinearity

    Args:
        Phi: Feature matrix [N, 3] (standardized or orthogonalized depending on variant)
        ln_A_obs_batch: Observed alpha values [N]
        constrain_signs: One of ['off', 'informed', 'alpha', 'physics', 'ortho']
        standardizer: Standardizer object (required for 'physics' and 'informed' variants)
        R_ortho: QR upper-triangular matrix (required for 'ortho' variant)
        cache_bust: Static cache-busting token to force fresh JIT compilation
    """

    if constrain_signs == 'informed':
        # Variant D: Informed priors centered at paper's best-fit values
        # Paper: k_J ≈ 10.7 km/s/Mpc, η' ≈ -8.0, ξ ≈ -7.0
        # This constrains the model to physically reasonable parameter space
        # while still allowing the data to inform the fit.
        if standardizer is None:
            raise ValueError("informed variant requires standardizer")

        # Sample physics params with informed priors
        k_J = numpyro.sample('k_J', dist.TruncatedNormal(loc=10.7, scale=3.0, low=5.0, high=20.0))
        eta_prime = numpyro.sample('eta_prime', dist.Normal(-8.0, 3.0))
        xi = numpyro.sample('xi', dist.Normal(-7.0, 3.0))

        # Convert to standardized space: c = -k_phys * scales
        # Physics model: ln_A = ln_A0 - (k_J*phi1 + eta'*phi2 + xi*phi3)
        # Standardized: ln_A = ln_A0_std + c1*φ1 + c2*φ2 + c3*φ3
        # Therefore: ci = -k_i * scale_i
        means = jnp.array(standardizer.means)
        scales = jnp.array(standardizer.scales)
        c = jnp.array([k_J, eta_prime, xi]) * scales  # FIXED: removed incorrect negative sign

        # Deterministic tracking of standardized coefficients
        numpyro.deterministic('c', c)

        # Physics-space ln_A0
        ln_A0_phys = numpyro.sample('ln_A0', dist.Normal(0.0, 5.0))

        # Convert to standardized-space offset
        # ln_A0_std = ln_A0_phys - (k_J*m1 + eta'*m2 + xi*m3)
        ln_A0_std = ln_A0_phys + jnp.dot(c, means / scales)

    elif constrain_signs == 'physics':
        # Variant B: Physics-space with positivity constraints
        if standardizer is None:
            raise ValueError("physics variant requires standardizer")

        # Sample physics params with positivity constraints
        k_J = numpyro.sample('k_J', dist.HalfNormal(20.0))
        eta_prime = numpyro.sample('eta_prime', dist.HalfNormal(10.0))
        xi = numpyro.sample('xi', dist.HalfNormal(10.0))

        # Convert to standardized space: c = -k_phys * scales
        means = jnp.array(standardizer.means)
        scales = jnp.array(standardizer.scales)
        c = jnp.array([k_J, eta_prime, xi]) * scales  # FIXED: removed incorrect negative sign

        # Deterministic tracking of standardized coefficients
        numpyro.deterministic('c', c)

        # Physics-space ln_A0
        ln_A0_phys = numpyro.sample('ln_A0', dist.Normal(0.0, 5.0))

        # Convert to standardized-space offset
        ln_A0_std = ln_A0_phys + jnp.dot(c, means / scales)

    elif constrain_signs == 'alpha':
        # Variant A: Force standardized weights c ≤ 0
        # Use transformed HalfNormal: c = -|x| where x ~ HalfNormal(1)
        c_raw = numpyro.sample('c_raw', dist.HalfNormal(1.0).expand([3]).to_event(1))
        c = -c_raw  # Now c ≤ 0
        numpyro.deterministic('c', c)

        ln_A0 = numpyro.sample('ln_A0', dist.Normal(0.0, 5.0))
        ln_A0_std = ln_A0

    elif constrain_signs == 'ortho':
        # Variant C: Use orthogonalized basis (eliminates collinearity)
        if R_ortho is None:
            raise ValueError("ortho variant requires R_ortho matrix")

        # Sample coefficients in orthogonal space
        c_ortho = numpyro.sample('c_ortho', dist.Normal(0.0, 1.0).expand([3]).to_event(1))

        # Map back to standardized-basis coefficients: c = R^(-1) @ c_ortho
        # Since Φ = Q @ R, we have: alpha = ln_A0 + c^T @ Phi = ln_A0 + c^T @ Q @ R
        # If we set c_ortho = R @ c, then: alpha = ln_A0 + c_ortho^T @ Q
        # So: c = R^(-T) @ c_ortho (inverse transpose)
        c = jnp.linalg.solve(R_ortho.T, c_ortho)
        numpyro.deterministic('c', c)

        ln_A0 = numpyro.sample('ln_A0', dist.Normal(0.0, 5.0))
        ln_A0_std = ln_A0

    else:  # constrain_signs == 'off'
        # Default: Unconstrained Normal(0, 1) priors
        c = numpyro.sample('c', dist.Normal(0.0, 1.0).expand([3]).to_event(1))
        ln_A0 = numpyro.sample('ln_A0', dist.Normal(0.0, 5.0))
        ln_A0_std = ln_A0

    # Predicted alpha: ln_A0 + sum_i c_i * φ_i
    # (For ortho variant, Phi is already Q, so this works directly)
    ln_A_th = ln_A0_std + jnp.dot(Phi, c)

    # Per-survey heteroscedastic noise
    sigma_ln_A = numpyro.sample('sigma_ln_A', dist.HalfNormal(2.0))

    # Student-t degrees of freedom for heavy-tail robustness
    nu = numpyro.sample('nu', dist.Exponential(0.1)) + 2.0

    # Student-t likelihood
    with numpyro.plate('data', Phi.shape[0]):
        numpyro.sample('ln_A_obs',
                       dist.StudentT(df=nu, loc=ln_A_th, scale=sigma_ln_A),
                       obs=ln_A_obs_batch)

def numpyro_model(persn_params_batch, L_peaks, photometries, redshifts, batch_size=500):
    """
    NumPyro model for global parameter inference (LEGACY - uses full physics).

    NOTE: Consider using numpyro_model_ln_A_space instead (simpler, faster).

    Args:
        batch_size: Number of SNe to process per batch (default 500 to avoid OOM)
    """
    # Priors (wide ranges around paper's values: k_J≈10.7, η′≈-8, ξ≈-7)
    k_J = numpyro.sample('k_J', dist.Uniform(0.1, 30))
    eta_prime = numpyro.sample('eta_prime', dist.Uniform(-15, -1))
    xi = numpyro.sample('xi', dist.Uniform(-15, 0))

    # Compute log-likelihood for all SNe (in batches to avoid OOM)
    logLs = log_likelihood_batch_jax(
        k_J, eta_prime, xi,
        persn_params_batch, L_peaks, photometries, redshifts,
        batch_size=batch_size
    )

    # Sum log-likelihoods (numpyro handles this automatically)
    total_logL = jnp.sum(logLs)

    # Factor in the total log-likelihood
    numpyro.factor('logL', total_logL)

def main():
    parser = argparse.ArgumentParser(description='Stage 2: NumPyro Global MCMC')
    parser.add_argument('--stage1-results', required=True,
                       help='Directory with Stage 1 results')
    parser.add_argument('--lightcurves', required=True,
                       help='Path to lightcurves CSV')
    parser.add_argument('--out', required=True,
                       help='Output directory')
    parser.add_argument('--nchains', type=int, default=4,
                       help='Number of MCMC chains (parallel on GPU)')
    parser.add_argument('--nsamples', type=int, default=2000,
                       help='Number of samples per chain (post-warmup)')
    parser.add_argument('--nwarmup', type=int, default=1000,
                       help='Number of warmup/burn-in steps')
    parser.add_argument('--quality-cut', type=float, default=2000,
                       help='Chi2 threshold for Stage 1 quality')
    parser.add_argument('--max-sne', type=int, default=None,
                       help='Maximum number of SNe to use (for testing)')
    parser.add_argument('--batch-size', type=int, default=500,
                       help='Batch size for likelihood computation (lower = less memory, default=500)')
    parser.add_argument('--constrain-signs',
                       choices=['off', 'informed', 'alpha', 'physics', 'ortho'],
                       default='off',
                       help='Sign constraint variant: off=unconstrained, informed=paper-centered priors (RECOMMENDED), alpha=c≤0, physics=kJ,η,ξ≥0, ortho=orthogonalized basis')
    parser.add_argument('--use-ln-a-space', action='store_true',
                       help='Use simplified ln_A-space model (100-1000x faster, recommended for production)')

    args = parser.parse_args()

    print("=" * 80)
    print("STAGE 2: NUMPYRO GLOBAL MCMC (GPU-OPTIMIZED)")
    print("=" * 80)
    print(f"Stage 1 results: {args.stage1_results}")
    print(f"Chains: {args.nchains}, Samples: {args.nsamples}, Warmup: {args.nwarmup}")
    print(f"Chain execution: Sequential on GPU (expect 'chains will be drawn sequentially' warning)")
    print()

    # Create output directory
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load lightcurves
    print("Loading lightcurves...")
    from v15_data import LightcurveLoader
    loader = LightcurveLoader(args.lightcurves)
    all_lcs = loader.load()
    print(f"  Loaded {len(all_lcs)} lightcurves")
    print()

    # Load Stage 1 results
    print("Loading Stage 1 results...")
    stage1_results = load_stage1_results(
        args.stage1_results, all_lcs, args.quality_cut
    )

    # Limit to max_sne if specified (for testing)
    if args.max_sne is not None and len(stage1_results) > args.max_sne:
        print(f"  Limiting to {args.max_sne} SNe for testing (from {len(stage1_results)} available)")
        stage1_results = dict(list(stage1_results.items())[:args.max_sne])

    if len(stage1_results) < 50:
        print(f"ERROR: Only {len(stage1_results)} good SNe, need at least 50!")
        return 1

    print()

    # Initialize at paper's values for better convergence
    init_params = {
        'k_J': 10.7,
        'eta_prime': -8.0,
        'xi': -7.0
    }

    # Choose model based on --use-ln-a-space flag
    feature_stats = None  # Will be set if using ln_A-space model

    if args.use_ln_a_space:
        # =============================================================================
        # LN_A-SPACE MODEL (FAST - recommended for production)
        # =============================================================================
        print("Preparing alpha-space data (fast model)...")

        # Extract just z and ln_A from Stage 1 results
        ln_A_obs_list = []
        redshifts_list = []
        snids = []

        for snid, result in stage1_results.items():
            if snid not in all_lcs:
                continue

            lc = all_lcs[snid]
            persn_best = result['persn_best']

            if len(persn_best) != 4:
                print(f"  WARNING: {snid} has {len(persn_best)} params, expected 4. Skipping.")
                continue

            # Load Stage 1 parameters
            persn_params = PerSNParams.from_array(persn_best)

            snids.append(snid)
            ln_A_obs_list.append(persn_params.ln_A)
            redshifts_list.append(lc.z)

        # Convert to arrays
        ln_A_obs_batch = np.array(ln_A_obs_list)
        z_batch = np.array(redshifts_list)

        # Convert from magnitude-space to natural-log space if needed
        ln_A_obs_batch_natural = _ensure_ln_A_natural(ln_A_obs_batch)

        # Standardize features for numerical stability
        Phi, feature_stats = _standardize_features(z_batch)

        print(f"  Using {len(snids)} SNe for MCMC")
        print(f"  Redshift range: [{z_batch.min():.3f}, {z_batch.max():.3f}]")
        print(f"  ln_A range: [{ln_A_obs_batch_natural.min():.1f}, {ln_A_obs_batch_natural.max():.1f}]")
        print()

        # Setup NUTS sampler (alpha-space likelihood)
        print("Setting up NUTS sampler (alpha-space likelihood - FAST)...")

        # Create standardizer object for 'physics' constraint variant
        standardizer = Standardizer(means=feature_stats['m'], scales=feature_stats['s'])

        # Bind constrain_signs and standardizer to model
        # Use 'off' mode (unconstrained) since eta' and xi are negative
        model_with_config = partial(
            numpyro_model_ln_A_space,
            constrain_signs=args.constrain_signs,
            standardizer=standardizer,
            cache_bust=0
        )

        nuts_kernel = NUTS(
            model_with_config,
            target_accept_prob=0.85,
            max_tree_depth=15,
            dense_mass=True,
            init_strategy=numpyro.infer.init_to_value(values=init_params)
        )

        mcmc = MCMC(
            nuts_kernel,
            num_warmup=args.nwarmup,
            num_samples=args.nsamples,
            num_chains=args.nchains,
            chain_method='parallel',
            progress_bar=True
        )

        print(f"  Warmup: {args.nwarmup} steps")
        print(f"  Sampling: {args.nsamples} samples × {args.nchains} chains = {args.nsamples * args.nchains} total")
        print(f"  Expected time: ~{(args.nwarmup + args.nsamples) * 0.5:.1f} seconds (alpha-space is ~100x faster)")
        print()

        # Run MCMC
        print("Running MCMC...")
        start_time = time.time()

        rng_key = jax.random.PRNGKey(0)
        mcmc.run(rng_key, Phi, ln_A_obs_batch_natural)

    else:
        # =============================================================================
        # FULL PHYSICS MODEL (SLOW - for validation only)
        # =============================================================================
        print("Preparing full lightcurve data (slow model)...")
        snids = []
        persn_params_list = []
        L_peaks_list = []
        photometries_list = []
        redshifts_list = []

        # Fixed L_peak from Stage 1
        from pipeline_io import L_PEAK_CANONICAL

        for snid, result in stage1_results.items():
            if snid not in all_lcs:
                continue

            lc = all_lcs[snid]

            # Get Stage 1 best-fit per-SN parameters
            persn_best = result['persn_best']
            if len(persn_best) != 4:
                print(f"  WARNING: {snid} has {len(persn_best)} params, expected 4. Skipping.")
                continue

            # Load Stage 1 parameters into type-safe structure
            # Stage 1 saves as: [t0, A_plasma, beta, ln_A]
            persn_params = PerSNParams.from_array(persn_best)

            # Convert to model's expected order: (t0, ln_A, A_plasma, beta)
            persn_model_tuple = persn_params.to_model_order()

            # Prepare photometry array [N_obs, 4]: mjd, wavelength_nm, flux_jy, flux_jy_err
            phot_array = np.column_stack([
                lc.mjd,
                lc.wavelength_nm,
                lc.flux_jy,
                lc.flux_err_jy,
            ])

            snids.append(snid)
            persn_params_list.append(persn_model_tuple)  # Store in model order
            L_peaks_list.append(L_PEAK_CANONICAL)
            photometries_list.append(jnp.array(phot_array))
            redshifts_list.append(lc.z)

        # Convert to JAX arrays
        persn_params_batch = jnp.array(persn_params_list)
        L_peaks = jnp.array(L_peaks_list)
        redshifts = jnp.array(redshifts_list)

        print(f"  Using {len(snids)} SNe for MCMC")
        print(f"  Redshift range: [{redshifts.min():.3f}, {redshifts.max():.3f}]")
        print(f"  Total observations: {sum(len(p) for p in photometries_list)}")
        print()

        # Setup NUTS sampler (using full physics model - paper's approach)
        print("Setting up NUTS sampler (full lightcurve physics - SLOW)...")
        print(f"  Batch size: {args.batch_size} SNe per batch (for memory efficiency)")

        # Bind batch_size to model using partial
        model_with_batch_size = partial(numpyro_model, batch_size=args.batch_size)

        nuts_kernel = NUTS(
            model_with_batch_size,
            target_accept_prob=0.85,
            max_tree_depth=15,
            dense_mass=True,
            init_strategy=numpyro.infer.init_to_value(values=init_params)
        )

        mcmc = MCMC(
            nuts_kernel,
            num_warmup=args.nwarmup,
            num_samples=args.nsamples,
            num_chains=args.nchains,
            chain_method='parallel',  # Run chains in parallel on GPU
            progress_bar=True
        )

        print(f"  Warmup: {args.nwarmup} steps")
        print(f"  Sampling: {args.nsamples} samples × {args.nchains} chains = {args.nsamples * args.nchains} total")
        print(f"  Expected time: ~{(args.nwarmup + args.nsamples) / 10:.1f} minutes")
        print()

        # Run MCMC
        print("Running MCMC...")
        start_time = time.time()

        rng_key = jax.random.PRNGKey(0)
        mcmc.run(
            rng_key,
            persn_params_batch,
            L_peaks,
            photometries_list,
            redshifts
        )

    elapsed = time.time() - start_time
    print(f"  MCMC complete in {elapsed/60:.1f} minutes")
    print()

    # Get samples
    print("Extracting samples...")
    samples = mcmc.get_samples()

    # Extract physics parameters
    print("  Extracting physics parameters...")
    if args.use_ln_a_space and args.constrain_signs == 'off':
        # ln_A-space model with 'off' mode samples standardized coefficients
        # Need to backtransform: k_phys = c / scales
        print("  Backtransforming from standardized space to physics space...")
        c_samples = np.asarray(samples['c'])  # Shape: (n_samples, 3)
        ln_A0_samples = np.asarray(samples['ln_A0'])

        # Backtransform to physics space
        # feature_stats was created in the ln_A-space branch above
        scales = np.asarray(feature_stats['s'])
        means = np.asarray(feature_stats['m'])

        k_J_samples = c_samples[:, 0] / scales[0]
        eta_prime_samples = c_samples[:, 1] / scales[1]
        xi_samples = c_samples[:, 2] / scales[2]
    else:
        # Full physics model or ln_A-space with 'physics' mode samples directly
        k_J_samples = np.asarray(samples['k_J'])
        eta_prime_samples = np.asarray(samples['eta_prime'])
        xi_samples = np.asarray(samples['xi'])

    # Enhanced diagnostics
    print()
    print("=" * 80)
    print("SAMPLE DIAGNOSTICS (Physical Parameters)")
    print("=" * 80)
    for name, arr in [('k_J', k_J_samples), ('eta_prime', eta_prime_samples),
                       ('xi', xi_samples)]:
        print(f"{name:12s}: mean={arr.mean():10.6f}, std={arr.std():10.6e}, min={arr.min():10.6f}, max={arr.max():10.6f}")
    print()

    # Print NumPyro summary
    print("MCMC parameter summary:")
    mcmc.print_summary()
    print()

    # Best-fit (median of posterior) for physical parameters
    k_J_best = float(np.median(k_J_samples))
    eta_prime_best = float(np.median(eta_prime_samples))
    xi_best = float(np.median(xi_samples))

    # Uncertainties (standard deviation) for physical parameters
    k_J_std = float(np.std(k_J_samples))
    eta_prime_std = float(np.std(eta_prime_samples))
    xi_std = float(np.std(xi_samples))

    print("=" * 80)
    print("MCMC RESULTS")
    print("=" * 80)
    print(f"Best-fit parameters (median ± std):")
    print(f"  k_J = {k_J_best:.2f} ± {k_J_std:.4f} km/s/Mpc")
    print(f"  eta' = {eta_prime_best:.4f} ± {eta_prime_std:.5f}")
    print(f"  xi = {xi_best:.2f} ± {xi_std:.4f}")
    print()

    # Check for divergences
    divergences = mcmc.get_extra_fields()['diverging']
    n_divergences = np.sum(divergences)
    if n_divergences > 0:
        print(f"⚠️  WARNING: {n_divergences} divergent transitions detected!")
        print("   Consider increasing target_accept_prob or max_tree_depth")
    else:
        print("✅ No divergences detected")
    print()

    # Model comparison metrics (WAIC/LOO) and boundary diagnostics
    print("=" * 80)
    print("MODEL COMPARISON METRICS")
    print("=" * 80)

    # Convert to ArviZ InferenceData for WAIC/LOO computation
    try:
        idata = az.from_numpyro(mcmc)

        # Compute WAIC
        waic = az.waic(idata)
        print(f"WAIC: {waic.elpd_waic:.2f} ± {waic.se:.2f}")
        print(f"  (effective number of parameters: {waic.p_waic:.1f})")

        # Compute LOO (Pareto-smoothed importance sampling leave-one-out CV)
        loo = az.loo(idata)
        print(f"LOO:  {loo.elpd_loo:.2f} ± {loo.se:.2f}")
        print(f"  (effective number of parameters: {loo.p_loo:.1f})")

        # Check for problematic observations (high Pareto k)
        high_k = np.sum(loo.pareto_k > 0.7)
        if high_k > 0:
            print(f"  ⚠️  {high_k} observations with high Pareto k (> 0.7)")
        else:
            print(f"  ✅ All Pareto k values < 0.7 (reliable LOO)")

    except Exception as e:
        print(f"⚠️  Could not compute WAIC/LOO: {e}")
        waic = None
        loo = None

    # Boundary fraction diagnostics (for constrained variants)
    print()
    print("Boundary Diagnostics:")

    if args.constrain_signs == 'physics':
        # Check how often parameters are near zero (at boundary)
        boundary_tol = 0.01
        k_J_boundary_frac = np.mean(k_J_samples < boundary_tol)
        eta_boundary_frac = np.mean(eta_prime_samples < boundary_tol)
        xi_boundary_frac = np.mean(xi_samples < boundary_tol)

        print(f"  Fraction at boundary (< {boundary_tol}):")
        print(f"    k_J:    {k_J_boundary_frac*100:.1f}%")
        print(f"    η':     {eta_boundary_frac*100:.1f}%")
        print(f"    ξ:      {xi_boundary_frac*100:.1f}%")

        if max(k_J_boundary_frac, eta_boundary_frac, xi_boundary_frac) > 0.1:
            print(f"  ⚠️  >10% of samples at boundary → constraint may be too restrictive")

    elif args.constrain_signs == 'alpha':
        # Check distribution of c values
        c0_pos_frac = np.mean(c_samples[:, 0] > 0)
        c1_pos_frac = np.mean(c_samples[:, 1] > 0)
        c2_pos_frac = np.mean(c_samples[:, 2] > 0)

        print(f"  Fraction violating c ≤ 0 constraint (numerical leakage):")
        print(f"    c[0]:   {c0_pos_frac*100:.2f}%")
        print(f"    c[1]:   {c1_pos_frac*100:.2f}%")
        print(f"    c[2]:   {c2_pos_frac*100:.2f}%")

        if max(c0_pos_frac, c1_pos_frac, c2_pos_frac) > 0.01:
            print(f"  ⚠️  Constraint leakage detected (numerical precision issue)")

    else:
        # Sign distribution diagnostics (only relevant for alternative parameterizations)
        # Commented out since current parameterization uses k_J, eta_prime, xi
        pass

    print()

    # Save results
    print("Saving results...")

    # Save samples as JSON
    samples_dict = {
        'params': ['k_J', 'eta_prime', 'xi'],
        'samples': np.column_stack([k_J_samples, eta_prime_samples, xi_samples]).tolist(),
        'mean': [float(np.mean(k_J_samples)), float(np.mean(eta_prime_samples)), float(np.mean(xi_samples))],
        'median': [k_J_best, eta_prime_best, xi_best],
        'std': [k_J_std, eta_prime_std, xi_std],
        'n_chains': args.nchains,
        'n_samples_per_chain': args.nsamples,
        'n_warmup': args.nwarmup,
        'n_divergences': int(n_divergences),
        'runtime_minutes': elapsed / 60,
        'n_snids': len(snids),
        'waic': float(waic.elpd_waic) if waic is not None else None,
        'waic_se': float(waic.se) if waic is not None else None,
        'loo': float(loo.elpd_loo) if loo is not None else None,
        'loo_se': float(loo.se) if loo is not None else None,
    }

    with open(outdir / 'samples.json', 'w') as f:
        json.dump(samples_dict, f, indent=2)
    print(f"  Saved samples to: {outdir / 'samples.json'}")

    # Save best-fit parameters
    best_fit = {
        'k_J': k_J_best,
        'eta_prime': eta_prime_best,
        'xi': xi_best,
        'k_J_std': k_J_std,
        'eta_prime_std': eta_prime_std,
        'xi_std': xi_std,
    }

    with open(outdir / 'best_fit.json', 'w') as f:
        json.dump(best_fit, f, indent=2)
    print(f"  Saved best-fit to: {outdir / 'best_fit.json'}")

    # Save raw samples as numpy arrays
    np.save(outdir / 'k_J_samples.npy', k_J_samples)
    np.save(outdir / 'eta_prime_samples.npy', eta_prime_samples)
    np.save(outdir / 'xi_samples.npy', xi_samples)
    print(f"  Saved numpy arrays to: {outdir / '*.npy'}")

    # Summary generation not needed - all results saved in samples.json and best_fit.json

    print()
    print("=" * 80)
    print("STAGE 2 COMPLETE")
    print("=" * 80)
    print()

    return 0

if __name__ == '__main__':
    exit(main())
