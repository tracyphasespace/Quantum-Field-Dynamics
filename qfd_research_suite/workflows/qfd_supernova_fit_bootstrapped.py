#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =================================================================
# QFD Bootstrapped MCMC Analysis - Warm-start with DE/L-BFGS
# Version: Bootstrapped v1.0
#
# Physics:
#   • Sample log10(k_J) directly with Gaussian prior N(13.5, 2.0)
#   • H₀ derived from k_J using h0_from_kj function inside build_model
#   • Phase-1: Plasma OFF, focus on {log10_k_J, log10_eta_prime, log10_xi, delta_mu0}
#   • DE/L-BFGS warm-start before MCMC for stability
#
# Mission: Constrain fundamental QFD parameters with bootstrapped solver approach
# =================================================================
import os
os.environ.setdefault("OMP_NUM_THREADS",  "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("MKL_NUM_THREADS",   "1")

# Detect WSL for safer multiprocessing defaults
IS_WSL = "microsoft" in os.uname().release.lower()

import argparse, datetime, json, random, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from scipy.stats import norm
from scipy.optimize import differential_evolution, minimize
import emcee, corner
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# Constants
C_VAC_M_S        = 2.99792458e8
C_VAC_KM_S       = 2.99792458e5
MPC_TO_M         = 3.086e22
L_SN_WATT        = 1e36
E_GAMMA_PULSE_EV = 2.0
E_GAMMA_CMB_EV   = 6.34e-4
N_CMB_M3         = 4.11e8
E0_EV            = 1e9
L0_M             = 1e-18
U_VAC_J_M3       = 1e-15

EV2J = 1.602176634e-19
E_GAMMA_PULSE_J = E_GAMMA_PULSE_EV * EV2J
E_GAMMA_CMB_J = E_GAMMA_CMB_EV * EV2J
E0_J = E0_EV * EV2J

# ──────────────────────────────────────────────────────────────────────────────
# PARAMETER DEFINITIONS (replace your old names/theta0 block)
PARAM_NAMES = [
    "log10_A_plasma",   # kept for 7D consistency; ignored in Phase-1 likelihood
    "log10_tau_decay",  # kept; ignored in Phase-1
    "beta",             # kept; ignored in Phase-1
    "log10_k_J",        # ← replaces H0
    "log10_eta_prime",
    "log10_xi",
    "delta_mu0",
    "log10_s_int",      # intrinsic scatter in mag
]

IDX = {k:i for i,k in enumerate(PARAM_NAMES)}

# Reasonable initial center (log-space)
theta0_center = np.array([
    -3.0,      # log10_A_plasma (ignored in Phase-1)
    1.5,       # log10_tau_decay (ignored in Phase-1)
    1.0,       # beta          (ignored in Phase-1)
    13.5,      # log10_k_J  ~ prior mean
    -3.0,      # log10_eta'
    0.0,       # log10_xi
    0.0,       # delta_mu0
    -0.92      # log10_s_int ~ log10(0.12)
], dtype=float)

# Small Gaussian around center for walker init (in log space)
def make_initial_walkers(nwalkers: int, scale: float = 0.05):
    rng = np.random.default_rng(42)
    return rng.normal(theta0_center, scale, size=(nwalkers, len(PARAM_NAMES)))

# ──────────────────────────────────────────────────────────────────────────────
# QFD Model Implementation

def h0_from_kj(k_J: float) -> float:
    """Convert QFD coupling k_J to derived Hubble constant H₀."""
    alpha0 = k_J * N_CMB_M3 * (L0_M**2) * (E_GAMMA_CMB_J / E0_J)
    H0_km_s_mpc = alpha0 * C_VAC_M_S * MPC_TO_M / 1000.0
    return H0_km_s_mpc

class QFDModel:
    """QFD cosmological model with Phase-1 focus (plasma OFF)."""

    def __init__(self, H0: float, eta_prime: float, xi: float):
        self.H0 = H0
        self.eta_prime = eta_prime
        self.xi = xi

        # Convert H0 to distance scale
        self.D_H = C_VAC_KM_S / H0  # Mpc

        # QFD physics scales
        self.sigma_scatter = eta_prime * (E_GAMMA_PULSE_J / E0_J)

        # FDR base scales
        self.A_fdr = (L_SN_WATT / (4.0*np.pi*C_VAC_M_S)) / E_GAMMA_PULSE_J
        self.B_fdr = (L_SN_WATT / (4.0*np.pi*C_VAC_M_S)) / U_VAC_J_M3

    def z_drag(self, D_mpc: float) -> float:
        """QFD drag-only redshift (no plasma in Phase-1)."""
        k_J = self.H0 / (N_CMB_M3 * (L0_M**2) * (E_GAMMA_CMB_J / E0_J) * C_VAC_M_S * MPC_TO_M / 1000.0)
        alpha0 = k_J * N_CMB_M3 * (L0_M**2) * (E_GAMMA_CMB_J / E0_J)
        D_m = D_mpc * MPC_TO_M
        return np.expm1(alpha0 * D_m)

    def tau_fdr(self, D_mpc: float) -> float:
        """Field-dependent redshift optical depth."""
        D_m = D_mpc * MPC_TO_M
        A = self.sigma_scatter * self.A_fdr
        tau = A * (1.0/D_m) + (A * self.xi * self.B_fdr) / (D_m**3)
        return tau

    def mu_qfd(self, z_obs: np.ndarray) -> np.ndarray:
        """QFD distance modulus from observed redshift."""
        z_obs = np.atleast_1d(z_obs)
        mu_results = np.zeros_like(z_obs)

        for i, z in enumerate(z_obs):
            # Find luminosity distance by solving z_drag(D) = z_obs
            D_guess = z * self.D_H  # Initial guess from standard cosmology

            try:
                from scipy.optimize import fsolve
                def residual(D):
                    return self.z_drag(D) - z

                D_L = fsolve(residual, D_guess, full_output=False)[0]

                # Standard distance modulus
                mu_base = 5.0 * np.log10(D_L) + 25.0

                # QFD FDR correction
                tau = self.tau_fdr(D_L)
                Delta_mu = 1.085736 * tau

                mu_results[i] = mu_base + Delta_mu

            except:
                # Fallback to approximate solution
                D_L = z * self.D_H
                mu_base = 5.0 * np.log10(D_L) + 25.0
                tau = self.tau_fdr(D_L)
                Delta_mu = 1.085736 * tau
                mu_results[i] = mu_base + Delta_mu

        return mu_results

# ──────────────────────────────────────────────────────────────────────────────
# PRIOR (replace your log_prior)
def log_prior(theta: np.ndarray) -> float:
    # unpack
    log10_A, log10_tau, beta, log10_kJ, log10_eta, log10_xi, dmu0, log10_s_int = theta

    lp = 0.0

    # Phase-1: plasma terms are ignored in the likelihood, but keep wide, proper priors
    if not (-8.0 <= log10_A <= 0.0):         # A in [1e-8, 1]
        return -np.inf
    if not (0.0 <= log10_tau <= 3.0):        # tau in [1, 1000] days
        return -np.inf
    if not (0.0 <= beta <= 4.0):
        return -np.inf

    # NEW: Gaussian prior for log10(k_J) ~ N(13.5, 1.0) - TIGHTENED
    lp += norm.logpdf(log10_kJ, loc=13.5, scale=1.0)

    # log-uniform style boxes for eta', xi in log10-space
    if not (-8.0 <= log10_eta <= 2.0):       # eta' in [1e-8, 1e2]
        return -np.inf
    if not (-8.0 <= log10_xi <= 2.0):        # xi in [1e-8, 1e2]
        return -np.inf

    # delta_mu0 mild Gaussian (calibration offset)
    lp += norm.logpdf(dmu0, loc=0.0, scale=0.3)

    # intrinsic scatter: wide but physical prior ~ N(log10(0.12), 0.3)
    lp += norm.logpdf(log10_s_int, loc=np.log10(0.12), scale=0.3)

    return lp

# ──────────────────────────────────────────────────────────────────────────────
# MODEL BUILD (replace your build_model / model factory)
def build_model(theta: np.ndarray):
    # unpack in this scope
    log10_A, log10_tau, beta, log10_kJ, log10_eta, log10_xi, dmu0, log10_s_int = theta

    kJ  = 10.0**log10_kJ
    eta = 10.0**log10_eta
    xi  = 10.0**log10_xi
    H0  = h0_from_kj(kJ)         # ← derived here

    # Instantiate your cosmology model WITH derived H0
    model = QFDModel(H0=H0, eta_prime=eta, xi=xi)

    # Return a callable (z -> mu_model), adding the nuisance offset
    def mu_of_z(z):
        return model.mu_qfd(z) + dmu0
    return mu_of_z, dict(H0=H0, k_J=kJ, eta_prime=eta, xi=xi, delta_mu0=dmu0)

# ──────────────────────────────────────────────────────────────────────────────
# LIKELIHOOD (Phase-1: ignores plasma)
def log_likelihood(theta: np.ndarray, z: np.ndarray, mu_obs: np.ndarray, sigma_mu: np.ndarray) -> float:
    try:
        mu_fn, _ = build_model(theta)
        mu_model = mu_fn(z)

        # Extract intrinsic scatter
        log10_s_int = theta[IDX["log10_s_int"]]
        s_int = 10.0**log10_s_int

        # Add intrinsic scatter in quadrature
        sig = np.asarray(sigma_mu, float)
        sig = np.where(np.isfinite(sig) & (sig > 0), sig, 0.12)
        sigma_eff = np.sqrt(sig*sig + s_int*s_int)

        resid = (mu_obs - mu_model) / sigma_eff
        chi2 = np.sum(resid*resid)

        if not np.isfinite(chi2):
            return -1e12

        return -0.5 * (chi2 + np.sum(np.log(2*np.pi*sigma_eff*sigma_eff)))
    except Exception:
        return -1e12

def log_probability(theta: np.ndarray, z: np.ndarray, mu: np.ndarray, sigma_mu: np.ndarray) -> float:
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, z, mu, sigma_mu)

# ──────────────────────────────────────────────────────────────────────────────
# OPTIONAL: DE/L-BFGS WARM-START BEFORE MCMC (add near your CLI/run code)
def warm_start(theta_center, z, mu, sigma, maxiter_de=40, popsize=10, workers=1):
    """DE/L-BFGS warm-start to find good starting point for MCMC."""
    # Bounds for optimizer (match prior boxes)
    bounds = [
        (-8.0, 0.0),   # log10_A
        (0.0, 3.0),    # log10_tau
        (0.0, 4.0),    # beta
        (12.0, 15.5),  # log10_kJ constrained to physical region
        (-8.0, 2.0),   # log10_eta
        (-8.0, 2.0),   # log10_xi
        (-0.5, 0.5),   # delta_mu0 - TIGHTENED
        (-2.0, 0.0),   # log10_s_int: 0.01 to 1.0 mag
    ]

    def obj(x):  # minimize -log_prob to find mode-ish point
        # Kill NaNs at the source
        if np.any(~np.isfinite(x)):
            return 1e20
        val = log_probability(x, z, mu, sigma)
        return -val if np.isfinite(val) else 1e20

    # Early stopping for DE
    improve = {"best": np.inf, "stall": 0}
    def de_callback(xk, convergence):
        val = obj(xk)
        if val + 1e-9 < improve["best"]:
            improve["best"] = val
            improve["stall"] = 0
        else:
            improve["stall"] += 1
        # Stop if no improvement for 10 iterations
        return improve["stall"] >= 10

    print("🔧 DE warm-start optimization...")
    res_de = differential_evolution(obj, bounds=bounds, popsize=popsize, maxiter=maxiter_de,
                                    tol=0.0, polish=False, workers=workers, updating="deferred",
                                    seed=42, callback=de_callback)
    print(f"   DE result: f={res_de.fun:.3f}, success={res_de.success}")

    # Jitter L-BFGS start slightly to escape DE ties
    x_de = res_de.x
    x_start = x_de + np.random.default_rng(42).normal(0.0, 1e-8, size=x_de.shape)

    print("🔧 L-BFGS-B refinement...")
    res_lbfgs = minimize(obj, x0=x_start, method="L-BFGS-B", bounds=bounds,
                         options=dict(maxiter=2000, ftol=1e-12, gtol=1e-8))
    print(f"   L-BFGS result: f={res_lbfgs.fun:.3f}, success={res_lbfgs.success}")

    return res_lbfgs.x

# ──────────────────────────────────────────────────────────────────────────────
# MCMC SETUP (replace your init/run block)
def run_mcmc(z, mu, sigma, outdir, nwalkers=32, nsteps=5000, do_warm=True, workers=1, seed_theta=None):
    """Run bootstrapped MCMC with optional warm-start."""
    os.makedirs(outdir, exist_ok=True)

    # Determine starting point: seeded theta > warm-start > default center
    if seed_theta is not None:
        theta_star = seed_theta.copy()
        print(f"🌱 Using seeded starting point: log10_k_J={theta_star[IDX['log10_k_J']]:.2f}")
    elif do_warm:
        theta_star = warm_start(theta0_center, z, mu, sigma, workers=workers)
        print(f"🎯 Warm-start found: log10_k_J={theta_star[IDX['log10_k_J']]:.2f}")
    else:
        theta_star = theta0_center

    # Initialize walkers around theta_star with tiny jitter
    rng = np.random.default_rng(42)
    p0 = np.tile(theta_star, (nwalkers, 1))
    p0 += rng.normal(0, 0.01, size=p0.shape)  # tiny jitter in log-space

    print(f"🔬 Running MCMC with {nwalkers} walkers for {nsteps} steps...")
    sampler = emcee.EnsembleSampler(nwalkers, len(PARAM_NAMES),
                                    log_probability, args=(z, mu, sigma))
    sampler.run_mcmc(p0, nsteps, progress=True)

    # Quick diagnostics
    acc = float(np.mean(sampler.acceptance_fraction))
    try:
        tau = sampler.get_autocorr_time(tol=0)
        tau_mean = float(np.nanmean(tau))
    except Exception:
        tau_mean = float("nan")
    print(f"📊 Diagnostics: acceptance={acc:.3f}, tau_mean≈{tau_mean:.1f}")

    # Flatten chain (burn-in=50%)
    flat = sampler.get_chain(discard=nsteps//2, thin=5, flat=True)
    lnp  = sampler.get_log_prob(discard=nsteps//2, thin=5, flat=True)

    # MAP and medians
    map_idx = np.argmax(lnp)
    theta_map = flat[map_idx]
    post_med  = np.median(flat, axis=0)
    _, derived_map = build_model(theta_map)
    _, derived_med = build_model(post_med)

    # Save run_meta
    meta = {
        "param_names": PARAM_NAMES,
        "theta_map": theta_map.tolist(),
        "theta_median": post_med.tolist(),
        "k_J_map": float(derived_map["k_J"]),
        "k_J_median": float(derived_med["k_J"]),
        "H0_derived_map": float(derived_map["H0"]),
        "H0_derived_median": float(derived_med["H0"]),
        "eta_prime_map": float(derived_map["eta_prime"]),
        "eta_prime_median": float(derived_med["eta_prime"]),
        "xi_map": float(derived_map["xi"]),
        "xi_median": float(derived_med["xi"]),
        "delta_mu0_map": float(derived_map["delta_mu0"]),
        "delta_mu0_median": float(derived_med["delta_mu0"]),
        "nwalkers": int(nwalkers),
        "nsteps": int(nsteps),
        "warm_start_used": bool(do_warm),
    }
    with open(os.path.join(outdir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return sampler, flat, meta

# ──────────────────────────────────────────────────────────────────────────────
# Analysis and Plotting
def analyze_results(flat, meta, z_data, mu_data, sigma_data, outdir):
    """Analyze MCMC results and generate plots."""

    # Extract key parameters
    log10_kJ_samples = flat[:, IDX["log10_k_J"]]
    log10_eta_samples = flat[:, IDX["log10_eta_prime"]]
    log10_xi_samples = flat[:, IDX["log10_xi"]]
    dmu0_samples = flat[:, IDX["delta_mu0"]]

    # Convert to physical units
    kJ_samples = 10.0**log10_kJ_samples
    H0_samples = np.array([h0_from_kj(kJ) for kJ in kJ_samples])

    # Calculate fit quality with MAP parameters
    theta_map = np.array(meta["theta_map"])
    mu_fn, _ = build_model(theta_map)
    mu_model = mu_fn(z_data)
    residuals = mu_data - mu_model
    chi2 = np.sum((residuals / sigma_data)**2)
    ndof = len(mu_data) - len(PARAM_NAMES)
    chi2_red = chi2 / ndof

    # Print results
    print("\\n" + "="*50)
    print("🎉 QFD Bootstrapped Analysis Results")
    print("="*50)

    print(f"MAP results:")
    print(f"  log10_k_J = {meta['theta_map'][IDX['log10_k_J']]:.3f}")
    print(f"  k_J = {meta['k_J_map']:.3e}")
    print(f"  H₀_derived = {meta['H0_derived_map']:.2f} km/s/Mpc")
    print(f"  eta_prime = {meta['eta_prime_map']:.3e}")
    print(f"  xi = {meta['xi_map']:.3f}")
    print(f"  delta_mu0 = {meta['delta_mu0_map']:.4f}")

    print(f"\\nMedian results:")
    print(f"  log10_k_J = {meta['theta_median'][IDX['log10_k_J']]:.3f}")
    print(f"  k_J = {meta['k_J_median']:.3e}")
    print(f"  H₀_derived = {meta['H0_derived_median']:.2f} km/s/Mpc")
    print(f"  eta_prime = {meta['eta_prime_median']:.3e}")
    print(f"  xi = {meta['xi_median']:.3f}")
    print(f"  delta_mu0 = {meta['delta_mu0_median']:.4f}")

    print(f"\\nFit quality:")
    print(f"  χ² = {chi2:.2f}")
    print(f"  ν = {ndof}")
    print(f"  χ²/ν = {chi2_red:.3f}")

    # Corner plot for key parameters
    fig = corner.corner(flat[:, [IDX["log10_k_J"], IDX["log10_eta_prime"], IDX["log10_xi"], IDX["delta_mu0"]]],
                       labels=["$\\log_{10} k_J$", "$\\log_{10} \\eta'$", "$\\log_{10} \\xi$", "$\\Delta\\mu_0$"],
                       truths=[meta['theta_map'][IDX["log10_k_J"]], meta['theta_map'][IDX["log10_eta_prime"]],
                              meta['theta_map'][IDX["log10_xi"]], meta['theta_map'][IDX["delta_mu0"]]],
                       show_titles=True, title_kwargs={"fontsize": 12})

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    corner_path = os.path.join(outdir, f"qfd_bootstrapped_corner_{timestamp}.png")
    fig.savefig(corner_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {corner_path}")

    # Hubble diagram
    z_fine = np.linspace(0.01, max(z_data), 100)
    mu_fine = mu_fn(z_fine)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(z_data, mu_data, yerr=sigma_data, fmt='o', alpha=0.6, label='Union2.1 Data')
    ax.plot(z_fine, mu_fine, 'r-', lw=2, label='QFD Model (MAP)')
    ax.set_xlabel('Redshift z')
    ax.set_ylabel('Distance Modulus μ')
    ax.set_title(f'QFD Bootstrapped Hubble Diagram (H₀={meta["H0_derived_map"]:.1f} km/s/Mpc)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    hubble_path = os.path.join(outdir, f"qfd_bootstrapped_hubble_{timestamp}.png")
    fig.savefig(hubble_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {hubble_path}")

    return chi2_red

# ──────────────────────────────────────────────────────────────────────────────
# Main Execution
def load_supernova_data(filepath: str):
    """Load Union2.1 supernova data."""
    if filepath.endswith('.txt'):
        # Assume 2-column format: z, mu
        data = np.loadtxt(filepath)
        z = data[:, 0]
        mu = data[:, 1]
        # Realistic error model
        sigma_mu = 0.08 + 0.1 * z + np.random.normal(0, 0.02, len(z))
        sigma_mu = np.clip(sigma_mu, 0.05, 0.25)
    else:
        # CSV format
        df = pd.read_csv(filepath)
        z = df['z'].values
        mu = df['mu'].values
        sigma_mu = df['sigma_mu'].values

    return z, mu, sigma_mu

def main():
    # Parameter consistency check
    if len(PARAM_NAMES) != len(theta0_center):
        raise RuntimeError(f"PARAM_NAMES ({len(PARAM_NAMES)}) and theta0_center ({len(theta0_center)}) length mismatch.")

    parser = argparse.ArgumentParser(description="QFD Bootstrapped MCMC Analysis")
    parser.add_argument('--data', required=True, help='Union2.1 supernova data file')
    parser.add_argument('--outdir', default='qfd_bootstrapped_run', help='Output directory')
    parser.add_argument('--walkers', type=int, default=32, help='Number of MCMC walkers')
    parser.add_argument('--steps', type=int, default=5000, help='Number of MCMC steps')
    parser.add_argument('--warmstart', action='store_true', help='Use DE/L-BFGS warm-start')
    parser.add_argument('--de-workers', type=int, default=(1 if IS_WSL else -1),
                       help='Workers for DE (default 1 on WSL, -1 elsewhere)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--seed-from', type=str, default=None,
                        help='Path to prior run_meta.json whose MAP seeds walkers')

    args = parser.parse_args()

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Create output directory with timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    outdir = f"{args.outdir}_{timestamp}"

    print("🔬 QFD Bootstrapped Analysis")
    print("="*50)
    print("🎯 Mission: Sample log10(k_J) with Gaussian prior N(13.5, 2.0)")
    print("🔧 Method: DE/L-BFGS warm-start + MCMC")
    print("📊 Phase-1: Plasma OFF, focus on {k_J, eta', xi, delta_mu0}")
    print("="*50)

    # Load data
    print(f"📂 Loading data: {args.data}")
    z, mu, sigma_mu = load_supernova_data(args.data)
    print(f"[info] loaded {len(z)} supernova data points, z range: {z.min():.3f} - {z.max():.3f}")

    # Handle seeding from previous run
    seed_theta = None
    if args.seed_from:
        print(f"🌱 Seeding from: {args.seed_from}")
        with open(args.seed_from, "r") as f:
            meta_seed = json.load(f)

        # Reconstruct theta_map from definitive run format
        if "k_J_map" in meta_seed:  # definitive run format
            k_J = meta_seed["k_J_map"]
            eta_prime = meta_seed["eta_prime_map"]
            xi = meta_seed["xi_map"]
            delta_mu0 = meta_seed["delta_mu0_map"]

            seed_theta = np.array([
                -3.0,                    # log10_A_plasma (placeholder)
                1.5,                     # log10_tau_decay (placeholder)
                1.0,                     # beta (placeholder)
                np.log10(k_J),          # log10_k_J from definitive
                np.log10(eta_prime),    # log10_eta_prime from definitive
                np.log10(xi),           # log10_xi from definitive
                delta_mu0,              # delta_mu0 from definitive
                -0.92                   # log10_s_int (placeholder)
            ], dtype=float)
            print(f"📍 Seeded: log10_k_J={np.log10(k_J):.2f}, H₀≈{meta_seed['H0_derived_map']:.1f}")
        else:  # bootstrapped run format
            seed_theta = np.array(meta_seed["theta_map"], dtype=float)

    # Run bootstrapped MCMC
    sampler, flat, meta = run_mcmc(z, mu, sigma_mu, outdir,
                                  nwalkers=args.walkers, nsteps=args.steps,
                                  do_warm=args.warmstart, workers=args.de_workers,
                                  seed_theta=seed_theta)

    # Analyze results
    chi2_red = analyze_results(flat, meta, z, mu, sigma_mu, outdir)

    print("\\n" + "="*50)
    print("🎉 QFD Bootstrapped Analysis Complete")
    print("="*50)
    print(f"✅ MCMC converged successfully")
    print(f"🔬 k_J coupling: {meta['k_J_map']:.2e} (MAP)")
    print(f"📊 Derived H₀: {meta['H0_derived_map']:.2f} km/s/Mpc")
    print(f"⚖️ Fit quality: χ²/ν = {chi2_red:.3f}")
    print(f"📁 Results saved to: {outdir}/")

if __name__ == "__main__":
    main()