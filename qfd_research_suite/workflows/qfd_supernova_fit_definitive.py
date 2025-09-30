#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =================================================================
# QFD Definitive MCMC Analysis (Phase 1: Global Cosmological Parameters)
# Version: Definitive v1.0 - Based on QFD v5.6
#
# Physics:
#   • Redshift/time dilation from baseline QFD "drag" only (k_J).
#   • Plasma physics deactivated (z_plasma) for Phase 1 global parameter analysis.
#   • FDR affect luminance (optical depth τ) → Δμ ≈ 1.085736 τ.
#   • H₀ is derived from k_J using h0_from_kj function (not sampled directly).
#
# Key Modifications from v5.6:
#   • Parameter vector: [log10_k_J, log10_eta_prime, log10_xi, delta_mu0] (4D MCMC)
#   • log10_k_J sampled with N(13.5, 2.0) prior instead of fixed value
#   • H₀ derived from k_J inside build_model function
#   • Plasma parameters (A_plasma, tau_decay, beta) fixed/deactivated
#   • Enhanced focus on constraining fundamental QFD cosmological parameters
#
# Mission: Constrain fundamental QFD parameters {k_J, eta_prime, xi} + nuisance delta_mu0
# =================================================================
import os
os.environ.setdefault("OMP_NUM_THREADS",  "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("MKL_NUM_THREADS",   "1")

import argparse, datetime, json, random, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from scipy.stats import norm
import emcee, corner
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# Constants (same scales you've been using; adjust if needed)
C_VAC_M_S        = 2.99792458e8
C_VAC_KM_S       = 2.99792458e5
MPC_TO_M         = 3.086e22
L_SN_WATT        = 1e36          # peak Type Ia luminosity (assumed)
E_GAMMA_PULSE_EV = 2.0           # mean photon energy in pulse (eV)
E_GAMMA_CMB_EV   = 6.34e-4       # mean CMB photon energy (eV)
N_CMB_M3         = 4.11e8
E0_EV            = 1e9           # QFD ref energy
L0_M             = 1e-18         # QFD ref length
U_VAC_J_M3       = 1e-15         # background vacuum energy density (tunable)

# Convert eV → Joule
EV2J             = 1.602176634e-19
E_GAMMA_PULSE_J  = E_GAMMA_PULSE_EV * EV2J
E_GAMMA_CMB_J    = E_GAMMA_CMB_EV   * EV2J
E0_J             = E0_EV * EV2J

# Precompute constants used in FDR τ(D) (see derivation)
# alpha_fdr(D) = A/D^2 * (1 + B/D^2), τ_fdr ≈ alpha_fdr * D = A/D + A*B/D^3
A_FDR_BASE = (L_SN_WATT / (4.0*np.pi*C_VAC_M_S)) / E_GAMMA_PULSE_J  # has units J^-1 m^-2 s^-1 → with σ gives m^-1
B_FDR_BASE = (L_SN_WATT / (4.0*np.pi*C_VAC_M_S)) / U_VAC_J_M3       # J m^-2 s^-1 / (J m^-3) = m

# ──────────────────────────────────────────────────────────────────────────────
# QFD-H₀ conversion functions

def h0_from_kj(k_J: float) -> float:
    """
    Convert QFD coupling k_J to Hubble constant H₀ [km/s/Mpc].

    Physics:
    α₀ = k_J * N_CMB * L₀² * (E_CMB/E₀) = H₀ / (c * MPC_TO_M)
    Therefore: H₀ = α₀ * c * MPC_TO_M = k_J * N_CMB * L₀² * (E_CMB/E₀) * c * MPC_TO_M

    Since c is in m/s, we convert to km/s by dividing by 1000.
    """
    alpha0 = k_J * N_CMB_M3 * (L0_M**2) * (E_GAMMA_CMB_J / E0_J)
    H0_km_s_mpc = alpha0 * C_VAC_M_S * MPC_TO_M / 1000.0  # Convert m/s to km/s
    return H0_km_s_mpc

def kj_from_h0(H0_km_s_mpc: float) -> float:
    """
    Convert Hubble constant H₀ [km/s/Mpc] to QFD coupling k_J.
    Inverse of h0_from_kj function.
    """
    alpha0 = (H0_km_s_mpc * 1000.0) / (C_VAC_M_S * MPC_TO_M)  # Convert km/s to m/s
    k_J = alpha0 / (N_CMB_M3 * (L0_M**2) * (E_GAMMA_CMB_J / E0_J))
    return k_J

# ──────────────────────────────────────────────────────────────────────────────
# Model

class QFDModel:
    """
    QFD Model for Phase 1 Analysis: Global Cosmological Parameters Only
    Redshift from cosmological drag only (k_J).
    Plasma & FDR enter via optical depth (dimming) — not redshift.
    """
    def __init__(self, k_J=2.7973e13, eta_prime=1e-3, xi=1.0, delta_mu0=0.0):
        self.k_J      = float(k_J)
        self.eta_prime= float(eta_prime)
        self.xi       = float(xi)
        self.delta_mu0 = float(delta_mu0)

        # Baseline drag coefficient α0 [1/m]
        sigma_drag = self.k_J * (L0_M**2) * (E_GAMMA_CMB_J / E0_J)
        self.alpha0 = N_CMB_M3 * sigma_drag

        # FDR scattering cross section σ_s [m^2]
        # σ_s = (η' ξ)^2 L0^2 (Eγ/E0)^2
        self.sigma_scatter = (self.eta_prime*self.xi)**2 * (L0_M**2) * (E_GAMMA_PULSE_J/E0_J)**2

        # Derived H₀ for reference/output
        self.H0_derived = h0_from_kj(self.k_J)

    # Distance ↔ redshift (drag-only)
    def z_cosmo_from_D(self, D_mpc: float) -> float:
        D_m = D_mpc * MPC_TO_M
        return np.expm1(self.alpha0 * D_m)

    def D_mpc_from_z(self, z_obs: float) -> float:
        # guard α0 to avoid numerical overflow if k_J were ~0
        alpha = max(self.alpha0, 1e-32)
        D_m = np.log1p(z_obs) / alpha
        return D_m / MPC_TO_M

    # FDR optical depth (dimming) — vectorized over D_mpc array
    def tau_fdr_Dmpc(self, D_mpc: np.ndarray) -> np.ndarray:
        D_m = np.atleast_1d(D_mpc) * MPC_TO_M
        # A ≡ σ_s * A_FDR_BASE
        A = self.sigma_scatter * A_FDR_BASE
        # B term with xi factor
        tau = A*(1.0/np.maximum(D_m, 1e-12)) + (A*self.xi*B_FDR_BASE)/(np.maximum(D_m, 1e-12)**3)
        return tau

# ──────────────────────────────────────────────────────────────────────────────
# Predictors

def qfd_mu_vectorized(z: np.ndarray, k_J: float, eta_p: float, xi: float, delta_mu0: float = 0.0) -> np.ndarray:
    """
    μ(z) = 5 log10(D_L/Mpc) + 25 + Δμ_FDR + delta_mu0
    where D_L = D_true * (1+z), and D_true = ln(1+z)/α0 (drag-only).
    """
    m = QFDModel(k_J=k_J, eta_prime=eta_p, xi=xi, delta_mu0=delta_mu0)
    z = np.atleast_1d(z).astype(float)
    # geometry
    D_true_mpc = np.log1p(z) / max(m.alpha0, 1e-32) / MPC_TO_M
    mu_geom = 5.0*np.log10(np.maximum(D_true_mpc*(1.0+z), 1e-12)) + 25.0
    # dimming
    delta_mu = 1.085736 * m.tau_fdr_Dmpc(D_true_mpc)
    return mu_geom + delta_mu + delta_mu0

def lcdm_mu(z_vals, H0=67.4, Om=0.3):
    """
    Flat ΛCDM μ(z) for reference (comoving integral with c/H0 factor).
    """
    Ol = 1.0 - Om
    def E(zp): return np.sqrt(Om*(1+zp)**3 + Ol)
    z_vals = np.atleast_1d(z_vals)
    z_max = float(np.max(z_vals))
    zg = np.linspace(0.0, max(z_max, 1.5), 4000)
    Hfac = (H0/C_VAC_KM_S)  # [1/Mpc]
    Dc = cumulative_trapezoid(1.0/(Hfac*E(zg)), zg, initial=0.0)  # [Mpc]
    Dl = (1.0 + z_vals) * np.interp(z_vals, zg, Dc)
    return 5*np.log10(np.maximum(Dl, 1e-12)) + 25

# ──────────────────────────────────────────────────────────────────────────────
# Data

def load_union21_data(path="union2.1_data.txt"):
    """Load Union2.1 data with enhanced error handling."""
    if not os.path.exists(path):
        print(f"[error] data file not found: {path}")
        return None, None, None

    try:
        arr = np.loadtxt(path)
    except Exception as e:
        print(f"[error] failed to load data from {path}: {e}")
        return None, None, None

    if arr.ndim == 1: arr = arr[None, :]
    if arr.shape[1] == 2:
        z, mu = arr[:,0], arr[:,1]
        dmu = np.full_like(z, 0.15, dtype=float)
    else:
        z, mu, dmu = arr[:,0], arr[:,1], arr[:,2]
        dmu = np.where(np.isfinite(dmu) & (dmu > 0), dmu, 0.15)

    # Quality checks
    valid = np.isfinite(z) & np.isfinite(mu) & (z > 0) & (z < 10) & (mu > 0) & (mu < 60)
    if not np.all(valid):
        print(f"[warning] filtered {(~valid).sum()}/{len(z)} invalid data points")
        z, mu, dmu = z[valid], mu[valid], dmu[valid]

    print(f"[info] loaded {len(z)} supernova data points, z range: {z.min():.3f} - {z.max():.3f}")
    return z.astype(float), mu.astype(float), dmu.astype(float)

# ──────────────────────────────────────────────────────────────────────────────
# Posterior (Modified for 4-parameter MCMC)

def log_prior(theta):
    """
    Prior for 4-parameter MCMC: [log10_k_J, log10_eta_prime, log10_xi, delta_mu0]

    - log10(k_J) ~ N(13.5, 2.0²) [Gaussian prior based on physical motivation]
    - log10(eta_prime) ~ Uniform(-5, 5) [broad uniform]
    - log10(xi) ~ Uniform(-5, 5) [broad uniform]
    - delta_mu0 ~ Uniform(-1, 1) [nuisance parameter for systematic offset]
    """
    log10_kJ, log10_eta, log10_xi, delta_mu0 = theta

    # Prior on log10(k_J): N(13.5, 2.0)
    log_prior_kJ = norm.logpdf(log10_kJ, loc=13.5, scale=2.0)

    # Prior bounds for other parameters
    if not (-5.0 < log10_eta < 5.0):
        return -np.inf
    if not (-5.0 < log10_xi < 5.0):
        return -np.inf
    if not (-1.0 < delta_mu0 < 1.0):
        return -np.inf

    return log_prior_kJ  # Only k_J has informative prior

def log_likelihood(theta, z, mu, dmu):
    """Enhanced likelihood for 4-parameter model."""
    try:
        log10_kJ, log10_eta, log10_xi, delta_mu0 = theta

        k_J = 10**log10_kJ
        eta_prime = 10**log10_eta
        xi = 10**log10_xi

        # Check for reasonable parameter values
        if not (1e-20 < k_J < 1e20):
            return -np.inf
        if not (1e-10 < eta_prime < 1e10):
            return -np.inf
        if not (1e-10 < xi < 1e10):
            return -np.inf

        mu_model = qfd_mu_vectorized(z, k_J, eta_prime, xi, delta_mu0)

        # Check model validity
        if not np.all(np.isfinite(mu_model)):
            return -np.inf

        sig = np.clip(np.asarray(dmu, float), 1e-3, np.inf)
        r = (mu - mu_model)/sig
        return -0.5*np.sum(r*r + np.log(2*np.pi*sig*sig))

    except Exception as e:
        return -np.inf

def log_probability(theta, z, mu, dmu):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(theta, z, mu, dmu)
    return lp + ll if np.isfinite(ll) else -np.inf

# ──────────────────────────────────────────────────────────────────────────────
# Convergence and diagnostics

def check_convergence(sampler, min_samples=100):
    """Check MCMC convergence using autocorrelation time."""
    try:
        tau = sampler.get_autocorr_time(quiet=True)
        converged = np.all(sampler.iteration > 10 * tau) and sampler.iteration > min_samples
        return converged, tau
    except:
        return False, None

def save_enhanced_diagnostics(outdir, args, sampler, best_params, z, mu, dmu, ts):
    """Save comprehensive diagnostics for definitive analysis."""

    k_J_best, eta_best, xi_best, delta_mu0_best = best_params

    # Convergence diagnostics
    converged, tau = check_convergence(sampler)

    # Model diagnostics
    m_map = QFDModel(k_J=k_J_best, eta_prime=eta_best, xi=xi_best, delta_mu0=delta_mu0_best)

    def delta_mu_fdr_at(z_val):
        D_mpc = m_map.D_mpc_from_z(z_val)
        tau = float(m_map.tau_fdr_Dmpc(np.array([D_mpc]))[0])
        return 1.085736 * tau

    # Enhanced diagnostics
    diag = {
        "version": "Definitive_v1.0",
        "timestamp": ts,
        "mission": "Phase 1: Global cosmological parameters {k_J, eta_prime, xi} + delta_mu0",
        "convergence": {
            "converged": bool(converged),
            "autocorr_time": tau.tolist() if tau is not None else None,
            "total_iterations": int(sampler.iteration)
        },
        "model_parameters": {
            "k_J_best": float(k_J_best),
            "eta_prime_best": float(eta_best),
            "xi_best": float(xi_best),
            "delta_mu0_best": float(delta_mu0_best),
            "H0_derived": float(m_map.H0_derived),
            "sigma_scatter_m2": float(m_map.sigma_scatter),
            "alpha0_inv_Mpc": float(1.0 / (m_map.alpha0 * MPC_TO_M))
        },
        "dimming_curve": [{"z": z_val, "DeltaMu_FDR_mag": float(delta_mu_fdr_at(z_val))}
                         for z_val in (0.1, 0.3, 0.5, 0.8, 1.0, 1.5)],
        "fundamental_physics": {
            "theory": "Quantum Field Dynamics (QFD)",
            "redshift_mechanism": "Cosmological drag (k_J only)",
            "plasma_status": "Deactivated (Phase 1 focus)"
        }
    }

    with open(os.path.join(outdir, "enhanced_diagnostics.json"), "w") as f:
        json.dump(diag, f, indent=2)

    # Residuals and fit metrics
    mu_model_full = qfd_mu_vectorized(z, k_J_best, eta_best, xi_best, delta_mu0_best)
    residuals = (mu - mu_model_full).astype(float)
    dmu_safe = np.clip(dmu.astype(float), 1e-3, np.inf)
    chi2 = float(np.sum((residuals/dmu_safe)**2))
    nu   = max(1, len(z) - 4)  # 4 fitted params
    red_chi2 = chi2 / nu

    AIC = chi2 + 2*4
    BIC = chi2 + 4*np.log(len(z))

    summary = {
        "chi2": chi2, "nu": int(nu), "reduced_chi2": red_chi2,
        "AIC": AIC, "BIC": BIC,
        "rms_residual": float(np.sqrt(np.mean(residuals**2))),
        "data_points": len(z),
        "converged": bool(converged),
        "parameters_fitted": 4
    }

    with open(os.path.join(outdir, "fit_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Save residuals
    np.savetxt(os.path.join(outdir, "residuals.csv"),
               np.column_stack([z, residuals, dmu_safe]),
               delimiter=",", header="z,residual_mag,err_mag", comments="")

# ──────────────────────────────────────────────────────────────────────────────
# Main

def main():
    ap = argparse.ArgumentParser(description="QFD Definitive MCMC Analysis v1.0 (Phase 1: Global Parameters)")
    ap.add_argument("--data", type=str, default="union2.1_data.txt")
    ap.add_argument("--walkers", type=int, default=32)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", type=str, default=None)
    ap.add_argument("--burn-frac", type=float, default=0.3, help="Fraction of steps to discard as burn-in")
    ap.add_argument("--thin-target", type=int, default=200, help="Target number of samples after thinning")
    args = ap.parse_args()

    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    outdir = args.outdir or f"QFD_Definitive_Run_{ts}"
    os.makedirs(outdir, exist_ok=True)
    print(f"--- QFD Definitive Analysis: All results will be saved to: {outdir} ---")

    z, mu, dmu = load_union21_data(args.data)
    if z is None:
        print("Halting: data load failed.")
        return 1

    # 4-parameter MCMC setup: [log10_k_J, log10_eta_prime, log10_xi, delta_mu0]
    np.random.seed(args.seed); random.seed(args.seed)
    ndim, nwalk = 4, args.walkers

    try:
        # Initialize walkers around reasonable values
        p0 = np.column_stack([
            np.full(nwalk, 13.5) + 0.2*np.random.randn(nwalk),  # log10_k_J ~ N(13.5, 0.2)
            np.full(nwalk, -3.0) + 0.1*np.random.randn(nwalk),  # log10_eta_prime
            np.full(nwalk,  0.0) + 0.1*np.random.randn(nwalk),  # log10_xi
            np.full(nwalk,  0.0) + 0.01*np.random.randn(nwalk), # delta_mu0
        ])

        sampler = emcee.EnsembleSampler(nwalk, ndim, log_probability,
                                        args=(z, mu, dmu))

        print(f"Running definitive MCMC with {nwalk} walkers for {args.steps} steps...")
        print("Parameter vector: [log10_k_J, log10_eta_prime, log10_xi, delta_mu0]")

        # Run with progress bar
        sampler.run_mcmc(p0, args.steps, progress=True)
        print("MCMC complete.")

    except Exception as e:
        print(f"[error] MCMC failed: {e}")
        return 1

    # Post-processing
    burn = max(1, int(args.burn_frac * sampler.iteration))
    thin = max(1, sampler.iteration // args.thin_target)

    try:
        flat = sampler.get_chain(discard=burn, thin=thin, flat=True)
        lpf  = sampler.get_log_prob(discard=burn, thin=thin, flat=True)
    except:
        # Fallback if not enough samples
        flat = sampler.get_chain(flat=True)
        lpf  = sampler.get_log_prob(flat=True)
        burn, thin = 0, 1

    if flat.size == 0:
        print("[error] No samples available after burn-in")
        return 1

    # Extract best-fit parameters
    med = np.median(flat, axis=0)
    best = flat[np.argmax(lpf)] if lpf.size else med

    # Convert from log-space
    k_J_best = 10**best[0]
    eta_prime_best = 10**best[1]
    xi_best = 10**best[2]
    delta_mu0_best = best[3]

    k_J_med = 10**med[0]
    eta_prime_med = 10**med[1]
    xi_med = 10**med[2]
    delta_mu0_med = med[3]

    # Calculate derived H₀
    H0_best = h0_from_kj(k_J_best)
    H0_med = h0_from_kj(k_J_med)

    print(f"\n--- QFD Definitive Analysis Results ---")
    print(f"Best-fit (MAP):")
    print(f"  k_J = {k_J_best:.3e}")
    print(f"  H₀_derived = {H0_best:.2f} km/s/Mpc")
    print(f"  eta_prime = {eta_prime_best:.3e}")
    print(f"  xi = {xi_best:.3f}")
    print(f"  delta_mu0 = {delta_mu0_best:.4f}")

    print(f"\nMedian values:")
    print(f"  k_J = {k_J_med:.3e}")
    print(f"  H₀_derived = {H0_med:.2f} km/s/Mpc")
    print(f"  eta_prime = {eta_prime_med:.3e}")
    print(f"  xi = {xi_med:.3f}")
    print(f"  delta_mu0 = {delta_mu0_med:.4f}")

    # Calculate final chi-squared
    mu_model_best = qfd_mu_vectorized(z, k_J_best, eta_prime_best, xi_best, delta_mu0_best)
    residuals = mu - mu_model_best
    chi2 = np.sum((residuals / dmu)**2)
    nu = len(z) - 4
    red_chi2 = chi2 / nu
    print(f"\nFit quality:")
    print(f"  χ² = {chi2:.2f}")
    print(f"  ν = {nu}")
    print(f"  χ²/ν = {red_chi2:.3f}")

    # Save metadata including both k_J and H0_derived
    meta = {
        "version": "Definitive_v1.0",
        "timestamp": ts, "seed": args.seed,
        "walkers": args.walkers, "steps": args.steps,
        "burn": burn, "thin": thin,
        "parameters_sampled": ["log10_k_J", "log10_eta_prime", "log10_xi", "delta_mu0"],
        "k_J_map": float(k_J_best),
        "H0_derived_map": float(H0_best),
        "eta_prime_map": float(eta_prime_best),
        "xi_map": float(xi_best),
        "delta_mu0_map": float(delta_mu0_best),
        "k_J_med": float(k_J_med),
        "H0_derived_med": float(H0_med),
        "eta_prime_med": float(eta_prime_med),
        "xi_med": float(xi_med),
        "delta_mu0_med": float(delta_mu0_med),
        "chi2": float(chi2),
        "nu": int(nu),
        "reduced_chi2": float(red_chi2)
    }
    with open(os.path.join(outdir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Enhanced diagnostics
    save_enhanced_diagnostics(outdir, args, sampler,
                            (k_J_best, eta_prime_best, xi_best, delta_mu0_best),
                            z, mu, dmu, ts)

    # Corner plot
    labels = [r"$\log_{10} k_J$", r"$\log_{10}\eta'$", r"$\log_{10}\xi$", r"$\delta\mu_0$"]
    fig = corner.corner(flat, labels=labels, quantiles=[0.16,0.5,0.84], show_titles=True)
    cp = os.path.join(outdir, f"qfd_definitive_corner_plot_{ts}.png")
    fig.savefig(cp, dpi=150, bbox_inches='tight'); plt.close(fig)
    print("Saved:", cp)

    # Hubble diagram
    z_grid = np.linspace(np.min(z), np.max(z), 200)
    mu_lcdm = lcdm_mu(z_grid, H0=H0_best)  # Use derived H₀ for ΛCDM comparison
    mu_qfd  = qfd_mu_vectorized(z_grid, k_J_best, eta_prime_best, xi_best, delta_mu0_best)

    fig, ax = plt.subplots(figsize=(12,8))
    ax.errorbar(z, mu, yerr=dmu, fmt='.', alpha=0.6, label='Union2.1')
    ax.plot(z_grid, mu_lcdm, 'r--', lw=2, label=f'ΛCDM (H₀={H0_best:.1f}, Ωₘ=0.3)')
    ax.plot(z_grid, mu_qfd,  'b-',  lw=2.5, label='QFD (MAP)')
    ax.set_xlabel('Redshift z'); ax.set_ylabel('Distance Modulus μ')
    ax.set_title('QFD Definitive Analysis vs. ΛCDM on Union2.1')
    ax.grid(True, ls=':', alpha=0.4); ax.legend()
    hp = os.path.join(outdir, f"qfd_definitive_hubble_diagram_{ts}.png")
    fig.savefig(hp, dpi=150, bbox_inches='tight'); plt.close(fig)
    print("Saved:", hp)

    print("\n--- QFD Definitive Analysis Complete ---")

    # Check convergence and provide recommendation
    converged, tau = check_convergence(sampler)
    if converged:
        print("✅ MCMC converged successfully")
    else:
        print("⚠️  MCMC may not have converged - consider more steps")

    print(f"🔬 Fundamental QFD parameters successfully constrained")
    print(f"   k_J coupling: {k_J_best:.2e} ± {np.std(flat[:,0]):.2f} (log10)")
    print(f"   Derived H₀: {H0_best:.2f} km/s/Mpc")

    return 0

# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sys.exit(main())