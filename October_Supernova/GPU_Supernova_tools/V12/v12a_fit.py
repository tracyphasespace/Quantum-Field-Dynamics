import numpy as np
import jax
import jax.numpy as jnp
import jax.lax as lax
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.distributions import constraints
from numpyro.distributions.transforms import biject_to
import arviz as az
import os
import json
import time
from datetime import datetime

import jax.scipy.special
from jax.scipy.optimize import minimize  # <-- ADDED IMPORT
from jax import tree_util

import v12_systematics as sysw
import v12_model as vm
import v12_io as io

# Robustness constants (ported from V11)
ROBUST_SIGMA_JITTER = 1e-3
ROBUST_MIN_EFF_OBS = 6
HUGE_CHI2 = 1e9
EPS = 1e-8

# =============================================================================
# Chi-Squared Likelihood
# =============================================================================

def chi2_likelihood(params, data, global_params, use_weighted_errors):
    """
    V12: Mask-aware chi-squared calculation for a single SN's light curve.
    This is minimized to find the best-fit nuisance parameters.

    Args:
        params: [t_peak, A_0, tau_0] nuisance parameters
        data: dict with SN data (mjd, flux, errors, etc.) - includes 'mask'
        global_params: tuple (lambda_R,)
        use_weighted_errors: bool flag for survey-weighted errors

    Returns:
        chi2: scalar chi-squared value
    """
    # Per-SN (nuisance) parameters
    t_peak, A_0, tau_0 = params

    # Global (sampled) parameters
    lambda_R, = global_params  # Unpack from tuple

    # Fixed physics "knobs" (scalars for this SN)
    k_J = data['k_J']
    delta_k_J = data['delta_k_J']
    z_trans = data['z_trans']
    xi = data['xi']
    gamma_thermo = data['gamma_thermo']

    # Get the mask for this SN (ensure boolean dtype)
    mask = data['mask'].astype(bool)

    # Calculate model flux
    flux_model = vm.qfd_model_flux(
        data['mjd'], data['z'], data['wavelength_eff_nm'],
        t_peak, A_0, tau_0,
        k_J, delta_k_J, z_trans,
        lambda_R, xi, gamma_thermo
    )

    # Guard against any NaNs/Infs returned by the model in corner cases
    flux_model = jnp.nan_to_num(flux_model, nan=0.0, posinf=0.0, neginf=0.0)

    # Get flux and error from data
    flux_obs = data['flux_nu_jy']

    # Choose which error column to use based on the flag
    flux_err = lax.cond(
        use_weighted_errors,
        lambda _: data['flux_nu_jy_err_weighted'],  # Use weighted error
        lambda _: data['flux_nu_jy_err'],           # Use original error
        operand=None
    )

    # Robust sigma handling: ensure strictly positive uncertainties with jitter (V11 port)
    sigma_eff = jnp.where(
        mask,
        jnp.maximum(flux_err, 0.0) + jnp.array(ROBUST_SIGMA_JITTER, dtype=flux_err.dtype),
        1.0,
    )

    resid = (flux_obs - flux_model) / sigma_eff
    resid = jnp.where(mask, resid, 0.0)
    chi2 = jnp.sum(resid * resid)

    # Effective observation count check (V11 port)
    eff_count = jnp.sum(mask.astype(jnp.int32))
    chi2 = jnp.nan_to_num(chi2, nan=jnp.array(HUGE_CHI2, dtype=chi2.dtype),
                          posinf=jnp.array(HUGE_CHI2, dtype=chi2.dtype),
                          neginf=jnp.array(HUGE_CHI2, dtype=chi2.dtype))
    chi2 = jnp.maximum(chi2, jnp.array(EPS, dtype=chi2.dtype))
    chi2 = jnp.where(eff_count < ROBUST_MIN_EFF_OBS,
                     jnp.array(HUGE_CHI2, dtype=chi2.dtype),
                     chi2)

    return chi2


# =============================================================================
# PotentialFn version (no factor sites)
# =============================================================================

def make_per_sn(use_weighted_errors: bool, debug_chi2: bool = False):
    """
    MODIFIED: Uses jax.scipy.optimize.minimize(method='BFGS', options={'maxiter': 80}) for the
    inner optimization, which is more robust than fixed-step-size GD.

    V11 port: Added debug_chi2 parameter for per-SN chi2 diagnostics.
    """
    W = bool(use_weighted_errors)
    # This is now used as maxiter for the BFGS optimizer
    N_STEPS = 80 
    # STEP_SIZE = 1e-5  # <-- REMOVED (no longer needed)

    @jax.jit
    def per_sn(sn_slice, init_slice, lam):
        global_params = (lam,)

        def inner_objective(log_params):
            """This is the objective function for the *inner* optimization."""
            t_peak = log_params[0]
            A_0 = jnp.exp(log_params[1])
            tau_0 = jnp.exp(log_params[2])
            linear_params = jnp.array([t_peak, A_0, tau_0])
            return chi2_likelihood(linear_params, sn_slice, global_params, W)

        # --- REPLACED OPTIMIZER ---
        # Old fixed-step-size gradient descent (unstable)
        # grad_fn = jax.grad(inner_objective)
        # step = jnp.asarray(STEP_SIZE, dtype=init_slice.dtype)
        # def scan_body(log_params, _):
        #     grad = grad_fn(log_params)
        #     updated = log_params - step * grad
        #     return updated, None
        # final_log_params, _ = lax.scan(scan_body, log_init, jnp.arange(N_STEPS))
        # theta_star = lax.stop_gradient(final_log_params)
        # min_chi2 = inner_objective(theta_star)
        
        # --- NEW ROBUST OPTIMIZER ---
        # Set up initial guess in log-space
        t_peak_init, A_0_init, tau_0_init = init_slice
        log_A0_init = jnp.log(jnp.maximum(A_0_init, 1e-6))
        log_tau_init = jnp.log(jnp.maximum(tau_0_init, 1e-3))
        log_init = jnp.array([t_peak_init, log_A0_init, log_tau_init])
        
        # Run the BFGS optimizer. JAX can differentiate through this.
        result = minimize(
            inner_objective,
            log_init,
            method='BFGS',
            # Use N_STEPS as the max number of iterations
            options={'maxiter': 80} 
        )

        # Stop gradients through the optimizer path and re-evaluate chi2
        log_t_peak, log_A0, log_tau = lax.stop_gradient(result.x)
        t_peak = log_t_peak
        A_0 = jnp.exp(log_A0)
        tau_0 = jnp.exp(log_tau)
        min_chi2 = chi2_likelihood(
            jnp.array([t_peak, A_0, tau_0]),
            sn_slice,
            global_params,
            W
        )

        # V11 port: Robust NaN/inf handling with HUGE_CHI2
        min_chi2 = jnp.nan_to_num(
            min_chi2,
            nan=jnp.array(HUGE_CHI2, dtype=min_chi2.dtype),
            posinf=jnp.array(HUGE_CHI2, dtype=min_chi2.dtype),
            neginf=jnp.array(HUGE_CHI2, dtype=min_chi2.dtype),
        )
        min_chi2 = jnp.maximum(min_chi2, jnp.array(EPS, dtype=min_chi2.dtype))

        # V11 port: Check BFGS success flag
        success = getattr(result, "success", True)
        success = jnp.asarray(success, dtype=jnp.bool_)
        min_chi2 = jnp.where(
            success,
            min_chi2,
            jnp.array(HUGE_CHI2, dtype=min_chi2.dtype),
        )

        # V11 port: Diagnostic logging hook
        if debug_chi2:
            jax.debug.print("per-SN chi2: {chi2}", chi2=min_chi2)
        # --- END MODIFICATION ---

        return jnp.nan_to_num(min_chi2, nan=jnp.inf, posinf=jnp.inf, neginf=jnp.inf)

    return per_sn

def make_per_sn_vec(per_sn):
    return jax.jit(jax.vmap(per_sn, in_axes=(0, 0, None)))

def build_total_ll(batch_data, initial_nuisance, per_sn_vec, sn_microbatch):
    bd = batch_data
    num_sne = int(bd['mask'].shape[0])
    SM = int(sn_microbatch) if sn_microbatch else 0
    if SM <= 0 or num_sne <= SM:
        # no microbatch: simple sum
        @jax.jit
        def total_ll(lam):
            return jnp.sum(per_sn_vec(bd, initial_nuisance, lam))
        return total_ll

    # microbatch with static template indices
    IDX_MB = jnp.arange(SM, dtype=jnp.int32)
    n_mb = (num_sne + SM - 1) // SM

    @jax.jit
    def total_ll(lam):
        def scan_mb(acc, i):
            start = i * SM
            idx = (start + IDX_MB) % num_sne
            mb_bd   = tree_util.tree_map(
                lambda x: x[idx] if (hasattr(x, "ndim") and x.ndim >= 1) else x, bd
            )
            mb_init = initial_nuisance[idx]
            ll_mb   = jnp.sum(per_sn_vec(mb_bd, mb_init, lam))
            return acc + ll_mb, None

        (ll_sum, _), = (jax.lax.scan(scan_mb, 0.0, jnp.arange(n_mb, dtype=jnp.int32)),)
        return ll_sum

    return total_ll

def make_potential_fn(batch_data, initial_nuisance, use_weighted, sn_microbatch, debug_chi2=False):
    per_sn     = make_per_sn(use_weighted, debug_chi2=debug_chi2)
    per_sn_vec = make_per_sn_vec(per_sn) # This function is fine
    
    # This function is fine, but know it now returns sum_min_chi2
    total_ll   = build_total_ll(batch_data, initial_nuisance, per_sn_vec, sn_microbatch)

    def nlp(u_lambda_R):
        lam = jnp.exp(u_lambda_R)

        # Prior is still correct
        mu, sigma = jnp.log(60.0), 1.0
        prior = 0.5 * ((u_lambda_R - mu) / sigma)**2 + jnp.log(sigma * jnp.sqrt(2.0*jnp.pi))

        # --- MODIFIED NLP CALCULATION ---
        # total_ll(lam) now returns sum_min_chi2
        sum_min_chi2 = total_ll(lam)

        # NegLogLike = -LogLike = - (Sum over SNe of [-0.5 * min_chi2])
        num_sne = jnp.maximum(1.0, jnp.asarray(batch_data['mask'].shape[0], dtype=jnp.float32))
        neg_log_like = 0.5 * (sum_min_chi2 / num_sne)

        # nlp = NegLogPrior + NegLogLike
        # OLD: return prior - total_ll(lam)
        return (prior + neg_log_like).reshape(())

    def potential_fn(unconstrained_params):
        return nlp(unconstrained_params["u_lambda_R"])
    return potential_fn


# =============================================================================
# Batch Processing Function
# =============================================================================

def summarize_surveys(survey_list):
    """
    V12: Takes a list of survey strings from batch_data['survey']
    """
    counts = {}
    for sname in survey_list:
        if not sname:
            continue
        key = str(sname).upper()
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))

def run_batch_on_sne(batch_data, initial_nuisance_stacked, args):
    """
    V12: Runs the MCMC fitting for a batch of supernovae using vectorized data.

    Args:
        batch_data: Dict of padded JAX arrays with shape (num_sne, max_obs) or (num_sne,)
        initial_nuisance_stacked: JAX array of shape (num_sne, 3)
        args: Argument namespace from v10_config
    """
    outdir = args.outdir
    io.ensure_dir(outdir)

    num_sne = batch_data['mask'].shape[0]
    num_sne_original = int(num_sne)
    z_original = np.asarray(batch_data['z']).copy()
    survey_original = list(batch_data.get('survey', []))

    # Pad data for micro-batching if enabled
    if args.sn_microbatch > 0:
        padding_needed = (args.sn_microbatch - (num_sne % args.sn_microbatch)) % args.sn_microbatch
        if padding_needed > 0:
            print(f"  Padding {num_sne} SNe to {num_sne + padding_needed} for micro-batching.")
            # Pad batch_data
            padded_batch_data = {}
            for k, v in batch_data.items():
                if k in ['snid', 'survey']: # These are lists, not JAX arrays
                    padded_batch_data[k] = v + [None] * padding_needed # Pad with None
                elif v.ndim == 1: # Scalar per SN (e.g., z, k_J)
                    padded_batch_data[k] = jnp.pad(v, (0, padding_needed), 'constant', constant_values=0.0)
                elif v.ndim == 2: # Per-observation data (e.g., mjd, flux, mask)
                    padded_batch_data[k] = jnp.pad(v, ((0, padding_needed), (0, 0)), 'constant', constant_values=0.0)
                else:
                    padded_batch_data[k] = v # Should not happen for current data structure
            batch_data = padded_batch_data

            # Pad initial_nuisance_stacked
            initial_nuisance_stacked = jnp.pad(initial_nuisance_stacked, ((0, padding_needed), (0, 0)), 'constant', constant_values=0.0)
            num_sne = num_sne + padding_needed
        else:
            num_sne = num_sne_original
    else:
        num_sne = num_sne_original

    print(f"\nConfiguring MCMC...")
    if args.sn_microbatch > 0 and num_sne != num_sne_original:
        print(f"  Num SNe: {num_sne_original} (padded to {num_sne})")
    else:
        print(f"  Num SNe: {num_sne_original}")
    print(f"  MCMC steps: {args.n_steps}")
    print(f"  Burn-in: {args.n_burn}")
    print(f"  Weighted fit: {args.with_survey_weights}")
    print(f"  Time window (days): {args.time_window_days}")
    print(f"  Downsample factor: {args.downsample_factor}")
    print(f"  Num chains: {args.num_chains}")
    print(f"  SN microbatch: {args.sn_microbatch if args.sn_microbatch else 'all'}")
    print(f"  Target accept: {args.target_accept}")
    print(f"  Max tree depth: {args.max_tree_depth}")

    jax_batch_data = {k: v for k, v in batch_data.items() if k not in ['snid', 'survey']}

    # Build PotentialFn (no factor sites)
    potential_fn = make_potential_fn(
        batch_data=jax_batch_data,
        initial_nuisance=initial_nuisance_stacked,
        use_weighted=args.with_survey_weights,
        sn_microbatch=args.sn_microbatch
    )
    print("\nUsing NUTS sampler with differentiable inner optimization.")
    kernel = NUTS(
        potential_fn=potential_fn,
        target_accept_prob=float(args.target_accept),
        dense_mass=True,
        max_tree_depth=int(args.max_tree_depth)
    )

    num_chains = max(1, int(getattr(args, "num_chains", 1)))
    chain_method = "vectorized" if num_chains > 1 else "sequential"

    # Interpret n_steps as post-warmup draws; if you currently treat it as total,
    # keep your original and set num_samples = args.n_steps
    num_samples = int(getattr(args, "n_steps", 2000))
    num_warmup  = int(getattr(args, "n_burn", 500))
    thin        = max(1, int(getattr(args, "thin", 1)))

    mcmc = MCMC(kernel,
                num_warmup=num_warmup,
                num_samples=num_samples,
                thinning=thin,
                num_chains=num_chains,
                chain_method=chain_method,
                progress_bar=True)

    # --- Run MCMC ---
    print(f"Running MCMC sampler... (chains={num_chains}, warmup={num_warmup}, "
          f"samples={num_samples}, thin={thin}, chain_method={chain_method})")
    rng_key = jax.random.PRNGKey(args.seed)
    # V11 FIX: With potential_fn + vectorized chains, init_params must have shape (num_chains,)
    # NumPyro uses init_params to determine parameter structure since no model is provided
    base = jnp.log(60.0)
    init_params = {"u_lambda_R": jnp.full((num_chains,), base, dtype=jnp.float32)}
    mcmc.run(rng_key, init_params=init_params)


    print("\nMCMC run complete.")
    mcmc.print_summary()

    # --- Process and Save Results ---
    print("Processing and saving results...")

    # Retrieve unconstrained samples and map back to reporting space
    samples_u = mcmc.get_samples(group_by_chain=True) # Keep chains separate
    u_lambda_R_samples = samples_u["u_lambda_R"]
    lin_lambda_R_samples = jnp.exp(u_lambda_R_samples)

    # --- Post-hoc thinning of saved arrays (does not change NumPyro internal sampling) ---
    thin = max(1, int(getattr(args, 'thin', 1)))
    if thin > 1:
        try:
            lin_lambda_R_samples = lin_lambda_R_samples[:, ::thin]
            print(f"[thinning] applying thin={thin} to saved arrays")
        except Exception as e:
            print(f"[warn] thinning failed with thin={thin}: {e}")

    lin_lambda_R_samples_np = np.asarray(lin_lambda_R_samples)
    if lin_lambda_R_samples_np.ndim == 1:
        lin_lambda_R_samples_np = lin_lambda_R_samples_np[np.newaxis, :]

    # Minimal InferenceData from arrays for summary
    try:
        idata = az.from_dict(posterior={
            "lambda_R": lin_lambda_R_samples_np,
            # Keep eta_prime as derived quantity for downstream summaries
            "eta_prime": lin_lambda_R_samples_np / float(args.xi_fixed),
        })
        summary = az.summary(idata, var_names=["lambda_R", "eta_prime"])
        if {"lambda_R", "eta_prime"}.issubset(summary.index):
            lambda_R_mean = float(summary.loc["lambda_R", "mean"])
            lambda_R_std = float(summary.loc["lambda_R", "sd"])
            lambda_R_rhat = float(summary.loc["lambda_R", "r_hat"])
            lambda_R_neff = float(summary.loc["lambda_R", "ess_bulk"])
            eta_prime_mean = float(summary.loc["eta_prime", "mean"])
            eta_prime_std = float(summary.loc["eta_prime", "sd"])
            eta_prime_rhat = float(summary.loc["eta_prime", "r_hat"])
            eta_prime_neff = float(summary.loc["eta_prime", "ess_bulk"])
        else:
            raise KeyError("summary missing variables")
    except Exception:
        flat_lambda = lin_lambda_R_samples_np.reshape(-1)
        lambda_R_mean = float(np.mean(flat_lambda))
        lambda_R_std = float(np.std(flat_lambda, ddof=1)) if flat_lambda.size > 1 else 0.0
        lambda_R_rhat = float("nan")
        lambda_R_neff = float("nan")
        flat_eta = flat_lambda / float(args.xi_fixed)
        eta_prime_mean = float(np.mean(flat_eta))
        eta_prime_std = float(np.std(flat_eta, ddof=1)) if flat_eta.size > 1 else 0.0
        eta_prime_rhat = float("nan")
        eta_prime_neff = float("nan")

    # V12: Extract z values from batch_data
    zs = np.array(z_original)
    zs = zs[np.isfinite(zs)]
    median_z = float(np.median(zs)) if len(zs) > 0 else None
    c_km_s = 299792.458
    D_median = float((c_km_s * median_z) / args.k_J) if (median_z is not None and args.k_J not in (None, 0.0)) else None
    print(f"  Median z: {median_z if median_z is not None else 'NA'} ; "
          f"D_median_Mpc: {D_median if D_median is not None else 'NA'}")

    # Save best-fit JSON
    results_dict = {
        "k_J": args.k_J,
        "lambda_R": lambda_R_mean,
        "lambda_R_std": lambda_R_std,
        "lambda_R_r_hat": lambda_R_rhat,
        "lambda_R_ess_bulk": lambda_R_neff,
        "xi": args.xi_fixed,
        "eta_prime": eta_prime_mean,
        "eta_prime_std": eta_prime_std,
        "eta_prime_r_hat": eta_prime_rhat,
        "eta_prime_ess_bulk": eta_prime_neff,
        "delta_k_J": args.delta_k_J,
        "z_trans": args.z_trans,
        "gamma_thermo": args.gamma_thermo,
        "n_sne": num_sne_original,
        "n_steps": args.n_steps,
        "n_burn": args.n_burn,
        "with_survey_weights": args.with_survey_weights,
        "z_min": args.z_min,
        "z_max": args.z_max,
        "median_z": median_z,
        "D_median_Mpc": D_median,
        "time_window_days": args.time_window_days,
        "downsample_factor": args.downsample_factor,
        "num_chains": num_chains,
        "sn_microbatch": args.sn_microbatch if args.sn_microbatch else None,
        "target_accept": args.target_accept,
        "max_tree_depth": args.max_tree_depth,
        "survey_counts": summarize_surveys(survey_original),
        "tag": args.tag or None
    }

    json_path = os.path.join(outdir, "v12_best_fit.json")
    io.save_json(json_path, results_dict)

    print(f"  Best-fit parameters saved to: {json_path}")

    # Save MCMC samples
    npy_path = os.path.join(outdir, "v12_mcmc_samples_lin_lambdaR.npy")
    io.save_npy(npy_path, lin_lambda_R_samples_np)
    print(f"  MCMC samples (lambda_R) saved to: {npy_path}")

    # Save completion marker
    io.write_marker(outdir, "DONE_V12")

    batch_summary = dict(
        lambda_R=lambda_R_mean,
        lambda_R_std=lambda_R_std,
        eta_prime=eta_prime_mean,
        eta_prime_std=eta_prime_std,
        lambda_R_r_hat=lambda_R_rhat,
        lambda_R_ess_bulk=lambda_R_neff,
        eta_prime_r_hat=eta_prime_rhat,
        eta_prime_ess_bulk=eta_prime_neff,
        num_chains=num_chains,
        sn_microbatch=args.sn_microbatch if args.sn_microbatch else None,
        time_window_days=args.time_window_days,
        downsample_factor=args.downsample_factor,
        n_sne=num_sne_original,
        median_z=median_z,
        D_median_Mpc=D_median,
        survey_counts=results_dict["survey_counts"],
        target_accept=args.target_accept,
        max_tree_depth=args.max_tree_depth,
        args=vars(args) # Include all args for full context
    )

    return batch_summary

# --- Appended by patch: canonical init builder for potential_fn runs ---
import jax.numpy as jnp

def build_init_params(num_chains: int):
    """
    Build init params with correct (num_chains,) shape for potential_fn-based NUTS.
    Uses log-normal prior center: log(60).
    """
    base = jnp.log(60.0).astype(jnp.float32)
    return {"u_lambda_R": jnp.full((int(num_chains),), base, dtype=jnp.float32)}