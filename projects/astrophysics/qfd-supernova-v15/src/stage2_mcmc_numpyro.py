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

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import time
from functools import partial

# JAX and NumPyro
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import vmap
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import arviz as az

# Import model
from v15_model import log_likelihood_single_sn_jax, alpha_pred_batch

# Import summary writer
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'tools'))
from write_stage2_summary import write_stage2_summary, Standardizer

# Magnitude-to-natural-log conversion constant
K_MAG_PER_LN = 2.5 / np.log(10.0)  # ≈ 1.0857


def _ensure_alpha_natural(alpha_obs_array: np.ndarray) -> np.ndarray:
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
        alpha_obs_array: Alpha values from Stage 1

    Returns:
        Alpha in natural-log space (negative, decreasing with z)
    """
    a = np.asarray(alpha_obs_array, dtype=float)

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

            # Quality filter
            chi2 = metrics['chi2']
            if chi2 > quality_cut:
                failed.append(snid)
                continue

            if metrics['iters'] < 5:
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
            continue

    print(f"  Loaded {len(results)} good SNe (chi2 < {quality_cut})")
    print(f"  Excluded {len(failed)} poor fits")

    return results

@partial(jax.jit, static_argnames=("cache_bust",))
def log_likelihood_alpha_space(
    k_J: float,
    eta_prime: float,
    xi: float,
    z_batch: jnp.ndarray,  # Shape: (n_sne,)
    alpha_obs_batch: jnp.ndarray,  # Shape: (n_sne,)
    *,
    cache_bust: int = 0,
) -> float:
    """
    Alpha-space likelihood: score globals by predicting alpha from (z; globals).

    Returns total log-likelihood (summed over all SNe).

    Uses alpha_pred(z; k_J, eta_prime, xi) to compute residuals.
    No per-SN parameters, no lightcurve physics - just alpha prediction.

    Args:
        cache_bust: Static cache-busting token to force fresh JIT compilation
    """
    # Predict alpha from globals
    alpha_th = alpha_pred_batch(z_batch, k_J, eta_prime, xi)

    # Residuals
    r_alpha = alpha_obs_batch - alpha_th

    # NOTE: Variance guard is enforced outside JIT (see preflight in run_alpha_space_mcmc)

    # Simple unweighted likelihood (can add sigma_alpha later)
    logL = -0.5 * jnp.sum(r_alpha**2)

    return logL

@jax.jit
def log_likelihood_batch_jax(
    k_J: float,
    eta_prime: float,
    xi: float,
    persn_params_batch: jnp.ndarray,  # Shape: (n_sne, 4)
    L_peaks: jnp.ndarray,  # Shape: (n_sne,)
    photometries: List,  # List of JAX arrays
    redshifts: jnp.ndarray  # Shape: (n_sne,)
) -> jnp.ndarray:
    """
    Vectorized log-likelihood over multiple SNe.

    Returns: Array of shape (n_sne,) with logL for each SN

    NOTE: This uses full lightcurve physics. For Stage 2, consider
    using log_likelihood_alpha_space instead (simpler, faster).
    """
    global_params = (k_J, eta_prime, xi)

    def single_sn_logL(persn_params, L_peak, phot, z_obs):
        return log_likelihood_single_sn_jax(
            global_params, tuple(persn_params), L_peak, phot, z_obs
        )

    # vmap over SNe
    logLs = vmap(single_sn_logL)(
        persn_params_batch, L_peaks, photometries, redshifts
    )

    return logLs

def numpyro_model_alpha_space(Phi, alpha_obs_batch, *,
                             constrain_signs='off',
                             standardizer=None,
                             R_ortho=None,
                             cache_bust: int = 0):
    """
    NumPyro model for global parameter inference using alpha-space likelihood.

    Supports multiple sign constraint variants for A/B testing:
    - 'off': Unconstrained Normal(0,1) priors (default, current behavior)
    - 'alpha': Constrain standardized weights c ≤ 0 (force monotone decrease)
    - 'physics': Work in physics-space with k_J, η', ξ ≥ 0 (interpretable positivity)
    - 'ortho': Use orthogonalized basis to eliminate collinearity

    Args:
        Phi: Feature matrix [N, 3] (standardized or orthogonalized depending on variant)
        alpha_obs_batch: Observed alpha values [N]
        constrain_signs: One of ['off', 'alpha', 'physics', 'ortho']
        standardizer: Standardizer object (required for 'physics' variant)
        R_ortho: QR upper-triangular matrix (required for 'ortho' variant)
        cache_bust: Static cache-busting token to force fresh JIT compilation
    """

    if constrain_signs == 'physics':
        # Variant B: Physics-space with positivity constraints
        if standardizer is None:
            raise ValueError("physics variant requires standardizer")

        # Sample physics params with positivity constraints
        k_J = numpyro.sample('k_J', dist.HalfNormal(20.0))
        eta_prime = numpyro.sample('eta_prime', dist.HalfNormal(10.0))
        xi = numpyro.sample('xi', dist.HalfNormal(10.0))

        # Convert to standardized space: c = k_phys * scales
        # (inverse of backtransform: k_phys = c / scales)
        means = jnp.array(standardizer.means)
        scales = jnp.array(standardizer.scales)
        c = jnp.array([k_J, eta_prime, xi]) * scales

        # Deterministic tracking of standardized coefficients
        numpyro.deterministic('c', c)

        # Physics-space alpha0 (includes offset correction)
        alpha0 = numpyro.sample('alpha0', dist.Normal(0.0, 5.0))

        # Convert to standardized-space offset
        alpha0_std = alpha0 + jnp.dot(c, means / scales)

    elif constrain_signs == 'alpha':
        # Variant A: Force standardized weights c ≤ 0
        # Use transformed HalfNormal: c = -|x| where x ~ HalfNormal(1)
        c_raw = numpyro.sample('c_raw', dist.HalfNormal(1.0).expand([3]).to_event(1))
        c = -c_raw  # Now c ≤ 0
        numpyro.deterministic('c', c)

        alpha0 = numpyro.sample('alpha0', dist.Normal(0.0, 5.0))
        alpha0_std = alpha0

    elif constrain_signs == 'ortho':
        # Variant C: Use orthogonalized basis (eliminates collinearity)
        if R_ortho is None:
            raise ValueError("ortho variant requires R_ortho matrix")

        # Sample coefficients in orthogonal space
        c_ortho = numpyro.sample('c_ortho', dist.Normal(0.0, 1.0).expand([3]).to_event(1))

        # Map back to standardized-basis coefficients: c = R^(-1) @ c_ortho
        # Since Φ = Q @ R, we have: alpha = alpha0 + c^T @ Phi = alpha0 + c^T @ Q @ R
        # If we set c_ortho = R @ c, then: alpha = alpha0 + c_ortho^T @ Q
        # So: c = R^(-T) @ c_ortho (inverse transpose)
        c = jnp.linalg.solve(R_ortho.T, c_ortho)
        numpyro.deterministic('c', c)

        alpha0 = numpyro.sample('alpha0', dist.Normal(0.0, 5.0))
        alpha0_std = alpha0

    else:  # constrain_signs == 'off'
        # Default: Unconstrained Normal(0, 1) priors
        c = numpyro.sample('c', dist.Normal(0.0, 1.0).expand([3]).to_event(1))
        alpha0 = numpyro.sample('alpha0', dist.Normal(0.0, 5.0))
        alpha0_std = alpha0

    # Predicted alpha: alpha0 + sum_i c_i * φ_i
    # (For ortho variant, Phi is already Q, so this works directly)
    alpha_th = alpha0_std + jnp.dot(Phi, c)

    # Per-survey heteroscedastic noise
    sigma_alpha = numpyro.sample('sigma_alpha', dist.HalfNormal(2.0))

    # Student-t degrees of freedom for heavy-tail robustness
    nu = numpyro.sample('nu', dist.Exponential(0.1)) + 2.0

    # Student-t likelihood
    with numpyro.plate('data', Phi.shape[0]):
        numpyro.sample('alpha_obs',
                       dist.StudentT(df=nu, loc=alpha_th, scale=sigma_alpha),
                       obs=alpha_obs_batch)

def numpyro_model(persn_params_batch, L_peaks, photometries, redshifts):
    """
    NumPyro model for global parameter inference (LEGACY - uses full physics).

    NOTE: Consider using numpyro_model_alpha_space instead (simpler, faster).
    """
    # Priors
    k_J = numpyro.sample('k_J', dist.Uniform(50, 90))
    eta_prime = numpyro.sample('eta_prime', dist.Uniform(0.001, 0.1))
    xi = numpyro.sample('xi', dist.Uniform(10, 50))

    # Compute log-likelihood for all SNe
    logLs = log_likelihood_batch_jax(
        k_J, eta_prime, xi,
        persn_params_batch, L_peaks, photometries, redshifts
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
    parser.add_argument('--constrain-signs',
                       choices=['off', 'alpha', 'physics', 'ortho'],
                       default='off',
                       help='Sign constraint variant: off=unconstrained, alpha=c≤0, physics=kJ,η,ξ≥0, ortho=orthogonalized basis')

    args = parser.parse_args()

    print("=" * 80)
    print("STAGE 2: NUMPYRO GLOBAL MCMC (FULLY GPU-OPTIMIZED)")
    print("=" * 80)
    print(f"Stage 1 results: {args.stage1_results}")
    print(f"Chains: {args.nchains}, Samples: {args.nsamples}, Warmup: {args.nwarmup}")
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

    if len(stage1_results) < 50:
        print(f"ERROR: Only {len(stage1_results)} good SNe, need at least 50!")
        return 1

    print()

    # FIXED: Use alpha-space likelihood (simpler, faster, no wiring bugs)
    print("Preparing alpha-space data...")
    snids = []
    z_list = []
    alpha_obs_list = []

    for snid, result in stage1_results.items():
        if snid not in all_lcs:
            continue

        lc = all_lcs[snid]

        # Extract alpha_obs from Stage 1 (persn_best order: t0, A_plasma, beta, alpha)
        persn_best = result['persn_best']
        alpha_obs = persn_best[3] if len(persn_best) == 4 else persn_best[-1]

        snids.append(snid)
        z_list.append(lc.z)
        alpha_obs_list.append(alpha_obs)

    # Convert to JAX arrays
    z_batch = jnp.array(z_list)
    alpha_obs_raw = np.array(alpha_obs_list)

    # Convert alpha to natural-log space if needed
    alpha_obs_natural = _ensure_alpha_natural(alpha_obs_raw)
    alpha_obs_batch = jnp.array(alpha_obs_natural)

    print(f"  Using {len(snids)} SNe for MCMC")
    print(f"  Redshift range: [{z_batch.min():.3f}, {z_batch.max():.3f}]")
    print(f"  Alpha range (natural-log): [{alpha_obs_batch.min():.3f}, {alpha_obs_batch.max():.3f}]")
    print()

    # Gradient sanity check (before MCMC setup)
    print("Running gradient sanity checks...")
    print("-" * 60)

    def ll_k(k):
        return log_likelihood_alpha_space(
            k, 0.01, 30.0, z_batch, alpha_obs_batch, cache_bust=0
        )

    def ll_eta(eta):
        return log_likelihood_alpha_space(
            70.0, eta, 30.0, z_batch, alpha_obs_batch, cache_bust=0
        )

    def ll_xi(xi):
        return log_likelihood_alpha_space(
            70.0, 0.01, xi, z_batch, alpha_obs_batch, cache_bust=0
        )

    grad_k = float(jax.grad(ll_k)(70.0))
    grad_eta = float(jax.grad(ll_eta)(0.01))
    grad_xi = float(jax.grad(ll_xi)(30.0))

    print(f"[GRADIENT CHECK] d logL / d k_J     @ 70.0 = {grad_k:.6e}")
    print(f"[GRADIENT CHECK] d logL / d eta'    @ 0.01 = {grad_eta:.6e}")
    print(f"[GRADIENT CHECK] d logL / d xi      @ 30.0 = {grad_xi:.6e}")

    # Check alpha stats
    print(f"[ALPHA CHECK] Range: min={alpha_obs_batch.min():.2f}, " +
          f"median={float(np.median(alpha_obs_batch)):.2f}, max={alpha_obs_batch.max():.2f}")

    # Sanity: alpha should be predominantly negative and correlate with z
    z_sorted_idx = np.argsort(z_batch)
    alpha_at_low_z = float(np.median(alpha_obs_batch[z_sorted_idx[:10]]))
    alpha_at_high_z = float(np.median(alpha_obs_batch[z_sorted_idx[-10:]]))
    print(f"[ALPHA CHECK] Median at low-z:  {alpha_at_low_z:.2f}")
    print(f"[ALPHA CHECK] Median at high-z: {alpha_at_high_z:.2f}")
    print(f"[ALPHA CHECK] Trend (should be negative): {alpha_at_high_z - alpha_at_low_z:.2f}")

    # Preflight check: compute variance of residuals with fiducial parameters
    alpha_pred_fid = alpha_pred_batch(z_batch, 70.0, 0.01, 30.0)
    r_alpha_fid = alpha_obs_batch - alpha_pred_fid
    var_r = float(jnp.var(r_alpha_fid))
    print(f"[PREFLIGHT CHECK] var(r_alpha) with fiducial params = {var_r:.3f}")
    if var_r < 1e-6:
        raise RuntimeError("WIRING BUG: var(r_alpha) ≈ 0 → check alpha_pred wiring!")
    print("-" * 60)
    print()

    # Setup NUTS sampler (using alpha-space model)
    print("Setting up NUTS sampler (alpha-space likelihood)...")
    nuts_kernel = NUTS(
        numpyro_model_alpha_space,
        target_accept_prob=0.85,  # Patch 2: Higher to reduce divergences on curved posteriors
        max_tree_depth=15,         # Patch 2: Increased for better exploration
        dense_mass=True,           # Helps with correlated dims
        init_strategy=numpyro.infer.init_to_median
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

    # --- Preflight (outside JIT): ensure residual variance > 0 to catch wiring bugs early
    print("  Preflight: checking residual variance...")
    _alpha_th = np.array(alpha_pred_batch(np.array(z_batch), 70.0, 0.01, 30.0))
    _var_resid = float(np.var(np.array(alpha_obs_batch) - _alpha_th))
    if not np.isfinite(_var_resid) or _var_resid <= 0.0:
        raise RuntimeError(f"Preflight: var(alpha_obs - alpha_pred(z; fiducials)) = {_var_resid:.6g} <= 0 — check wiring.")
    print(f"  Preflight OK: var(residuals) = {_var_resid:.6g}")

    # Gradient sanity check: ensure likelihood depends on parameters
    print("  Gradient sanity check...")
    def ll_k(k):
        return log_likelihood_alpha_space(
            k, 0.01, 30.0, z_batch, alpha_obs_batch, cache_bust=0
        )
    def ll_eta(eta):
        return log_likelihood_alpha_space(
            70.0, eta, 30.0, z_batch, alpha_obs_batch, cache_bust=0
        )
    def ll_xi(xi_val):
        return log_likelihood_alpha_space(
            70.0, 0.01, xi_val, z_batch, alpha_obs_batch, cache_bust=0
        )
    g_k = jax.grad(ll_k)(70.0)
    g_eta = jax.grad(ll_eta)(0.01)
    g_xi = jax.grad(ll_xi)(30.0)
    print(f"    d logL / d k_J      at 70   = {float(g_k):.6e}")
    print(f"    d logL / d eta_prime at 0.01 = {float(g_eta):.6e}")
    print(f"    d logL / d xi        at 30   = {float(g_xi):.6e}")
    if abs(float(g_k)) < 1e-10 or abs(float(g_eta)) < 1e-10 or abs(float(g_xi)) < 1e-10:
        print("  WARNING: Near-zero gradient detected! Parameters may not affect likelihood.")

    # Optional: clear JAX in-memory caches before run to avoid sticky traces
    jax.clear_caches()

    # Standardize features (Patch 1 from cloud.txt: reduces curvature/divergences)
    print("  Standardizing basis features...")
    Phi, stats = _standardize_features(z_batch)
    print(f"    Feature stats: m={stats['m']}, s={stats['s']}")

    # Prepare variant-specific context
    R_ortho = None
    standardizer = None

    if args.constrain_signs == 'ortho':
        print(f"  [VARIANT: {args.constrain_signs}] Orthogonalizing basis with QR decomposition...")
        Phi, R_ortho = _orthogonalize_basis(Phi)

        # Check orthogonality
        corr_ortho = np.corrcoef(np.array(Phi).T)
        print(f"    Max off-diagonal correlation: {np.max(np.abs(corr_ortho - np.eye(3))):.6f}")

    elif args.constrain_signs == 'physics':
        print(f"  [VARIANT: {args.constrain_signs}] Using physics-space priors (k_J, η', ξ ≥ 0)...")
        # Create standardizer object for back-transform
        standardizer = Standardizer(
            means=np.asarray(stats['m']),
            scales=np.asarray(stats['s'])
        )

    elif args.constrain_signs == 'alpha':
        print(f"  [VARIANT: {args.constrain_signs}] Constraining standardized weights c ≤ 0...")

    else:  # 'off'
        print(f"  [VARIANT: {args.constrain_signs}] Unconstrained priors (baseline)...")

    # Cache-bust token to force a fresh trace even in a long-lived process
    _cache_bust = int(time.time()) & 0xffff

    rng_key = jax.random.PRNGKey(0)
    mcmc.run(
        rng_key,
        Phi,  # Pass features (standardized or orthogonalized)
        alpha_obs_batch,
        constrain_signs=args.constrain_signs,
        standardizer=standardizer,
        R_ortho=R_ortho,
        cache_bust=_cache_bust,
    )

    elapsed = time.time() - start_time
    print(f"  MCMC complete in {elapsed/60:.1f} minutes")
    print()

    # Get samples
    print("Extracting samples...")
    samples = mcmc.get_samples()

    # Back-transform standardized coefficients to physical parameters
    print("  Back-transforming standardized coefficients to physical parameters...")
    c_samples = np.asarray(samples['c'])  # shape: [n_samples, 3]
    alpha0_samples_raw = np.asarray(samples['alpha0'])
    sigma_alpha_samples = np.asarray(samples['sigma_alpha'])
    nu_samples = np.asarray(samples['nu'])  # Student-t degrees of freedom

    # Convert standardization stats to numpy for back-transformation
    m = np.asarray(stats['m'])  # [m1, m2, m3]
    s = np.asarray(stats['s'])  # [s1, s2, s3]

    # Physical coefficients: k_J = c1/s1, eta_prime = c2/s2, xi = c3/s3
    k_J_samples = c_samples[:, 0] / s[0]
    eta_prime_samples = c_samples[:, 1] / s[1]
    xi_samples = c_samples[:, 2] / s[2]

    # Physical alpha0: alpha0_phys = alpha0_raw - sum(c_i * m_i / s_i)
    alpha0_samples = alpha0_samples_raw - np.dot(c_samples, m / s)

    # Enhanced diagnostics
    print()
    print("=" * 80)
    print("SAMPLE DIAGNOSTICS (Physical Parameters)")
    print("=" * 80)
    for name, arr in [('k_J', k_J_samples), ('eta_prime', eta_prime_samples),
                       ('xi', xi_samples), ('alpha0', alpha0_samples),
                       ('sigma_alpha', sigma_alpha_samples), ('nu', nu_samples)]:
        print(f"{name:12s}: mean={arr.mean():10.6f}, std={arr.std():10.6e}, min={arr.min():10.6f}, max={arr.max():10.6f}")
    print()

    # Print NumPyro summary for standardized coefficients
    print("Standardized coefficients c (from MCMC):")
    mcmc.print_summary()
    print()

    # Best-fit (median of posterior) for physical parameters
    k_J_best = float(np.median(k_J_samples))
    eta_prime_best = float(np.median(eta_prime_samples))
    xi_best = float(np.median(xi_samples))
    alpha0_best = float(np.median(alpha0_samples))
    sigma_alpha_best = float(np.median(sigma_alpha_samples))
    nu_best = float(np.median(nu_samples))

    # Uncertainties (standard deviation) for physical parameters
    k_J_std = float(np.std(k_J_samples))
    eta_prime_std = float(np.std(eta_prime_samples))
    xi_std = float(np.std(xi_samples))
    alpha0_std = float(np.std(alpha0_samples))
    sigma_alpha_std = float(np.std(sigma_alpha_samples))
    nu_std = float(np.std(nu_samples))

    print("=" * 80)
    print("MCMC RESULTS")
    print("=" * 80)
    print(f"Best-fit parameters (median):")
    print(f"  k_J = {k_J_best:.2f} ± {k_J_std:.4f}")
    print(f"  eta' = {eta_prime_best:.4f} ± {eta_prime_std:.5f}")
    print(f"  xi = {xi_best:.2f} ± {xi_std:.4f}")
    print(f"  alpha0 (zeropoint offset) = {alpha0_best:.4f} ± {alpha0_std:.5f}")
    print(f"  sigma_alpha = {sigma_alpha_best:.4f} ± {sigma_alpha_std:.5f}")
    print(f"  nu (Student-t DOF) = {nu_best:.2f} ± {nu_std:.3f}")
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
        # For unconstrained and ortho variants, report sign distribution
        c_signs = np.sign(c_samples)
        c0_neg_frac = np.mean(c_signs[:, 0] < 0)
        c1_neg_frac = np.mean(c_signs[:, 1] < 0)
        c2_neg_frac = np.mean(c_signs[:, 2] < 0)

        print(f"  Sign distribution (fraction negative):")
        print(f"    c[0]:   {c0_neg_frac*100:.1f}%")
        print(f"    c[1]:   {c1_neg_frac*100:.1f}%")
        print(f"    c[2]:   {c2_neg_frac*100:.1f}%")

    print()

    # Save results
    print("Saving results...")

    # Save samples as JSON
    samples_dict = {
        'params': ['k_J', 'eta_prime', 'xi', 'alpha0', 'sigma_alpha', 'nu'],
        'samples': np.column_stack([k_J_samples, eta_prime_samples, xi_samples, alpha0_samples, sigma_alpha_samples, nu_samples]).tolist(),
        'mean': [float(np.mean(k_J_samples)), float(np.mean(eta_prime_samples)), float(np.mean(xi_samples)), float(np.mean(alpha0_samples)), float(np.mean(sigma_alpha_samples)), float(np.mean(nu_samples))],
        'median': [k_J_best, eta_prime_best, xi_best, alpha0_best, sigma_alpha_best, nu_best],
        'std': [k_J_std, eta_prime_std, xi_std, alpha0_std, sigma_alpha_std, nu_std],
        'n_chains': args.nchains,
        'n_samples_per_chain': args.nsamples,
        'n_warmup': args.nwarmup,
        'n_divergences': int(n_divergences),
        'runtime_minutes': elapsed / 60,
        'n_snids': len(snids),
        'constrain_signs_variant': args.constrain_signs,
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
        'alpha0': alpha0_best,
        'sigma_alpha': sigma_alpha_best,
        'nu': nu_best,
        'k_J_std': k_J_std,
        'eta_prime_std': eta_prime_std,
        'xi_std': xi_std,
        'alpha0_std': alpha0_std,
        'sigma_alpha_std': sigma_alpha_std,
        'nu_std': nu_std,
        'constrain_signs_variant': args.constrain_signs
    }

    with open(outdir / 'best_fit.json', 'w') as f:
        json.dump(best_fit, f, indent=2)
    print(f"  Saved best-fit to: {outdir / 'best_fit.json'}")

    # Save raw samples as numpy arrays
    np.save(outdir / 'k_J_samples.npy', k_J_samples)
    np.save(outdir / 'eta_prime_samples.npy', eta_prime_samples)
    np.save(outdir / 'xi_samples.npy', xi_samples)
    np.save(outdir / 'sigma_alpha_samples.npy', sigma_alpha_samples)
    np.save(outdir / 'nu_samples.npy', nu_samples)
    print(f"  Saved numpy arrays to: {outdir / '*.npy'}")

    # Generate comprehensive summary JSON
    print()
    print("Generating comprehensive summary...")
    standardizer = Standardizer(
        means=np.asarray(stats['m']),
        scales=np.asarray(stats['s'])
    )

    # Prepare samples dict for summary (using standardized c and raw alpha0)
    summary_samples = {
        'c': c_samples,
        'alpha0': alpha0_samples_raw,
        'sigma_alpha': sigma_alpha_samples,
        'nu': nu_samples,
    }

    summary_path = str(outdir / 'summary.json')
    write_stage2_summary(
        summary_path,
        summary_samples,
        standardizer,
        survey_names=['DES']
    )
    print(f"  Saved comprehensive summary to: {summary_path}")

    print()
    print("=" * 80)
    print("STAGE 2 COMPLETE")
    print("=" * 80)
    print()

    return 0

if __name__ == '__main__':
    exit(main())
