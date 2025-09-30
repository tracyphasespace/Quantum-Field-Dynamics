#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =================================================================
# QFD Cosmological Model Fitter (μ–z; bootstrap-ready + enhanced features)
# Version: 5.6
#
# Physics:
#   • Redshift/time dilation from baseline QFD "drag" only (k_J).
#   • Plasma & FDR affect luminance (optical depth τ) → Δμ ≈ 1.085736 τ.
#
# Features v5.6:
#   • All v5.5 capabilities: diagnostics, reparameterization, scientific plots
#   • Bootstrap calibration support: --bootstrap-calibration flag
#   • Enhanced error handling and convergence checking
#   • Improved sampling with configurable burn-in and thinning
#   • Better WSL/multiprocessing compatibility
#   • Expanded diagnostic outputs for pipeline integration
#   • Support for fixed k_J mode during time-flux analysis
#
# Usage:
#   python QFD_Cosmology_Fitter_v5.6.py \
#     --data union2.1_data.txt --walkers 32 --steps 3000 --outdir ../runs_qfd
#
#   # With reparameterization for better sampling:
#   python QFD_Cosmology_Fitter_v5.6.py --reparam \
#     --data union2.1_data.txt --walkers 32 --steps 3000 --outdir ../runs_qfd
#
#   # Bootstrap calibration mode:
#   python QFD_Cosmology_Fitter_v5.6.py --bootstrap-calibration \
#     --data union2.1_data.txt --walkers 16 --steps 1000 --outdir ../runs_qfd
#
# Requires: numpy scipy matplotlib emcee corner tqdm pandas
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
KJ_DEFAULT       = 2.7973e13     # from your earlier H0≈67.4 calibration

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
# Model

class QFDModel:
    """
    Redshift from cosmological drag only (k_J).
    Plasma & FDR enter via optical depth (dimming) — not redshift.
    """
    def __init__(self, k_J=KJ_DEFAULT, eta_prime=1e-3, xi=1.0):
        self.k_J      = float(k_J)
        self.eta_prime= float(eta_prime)
        self.xi       = float(xi)

        # Baseline drag coefficient α0 [1/m]
        sigma_drag = self.k_J * (L0_M**2) * (E_GAMMA_CMB_J / E0_J)
        self.alpha0 = N_CMB_M3 * sigma_drag

        # FDR scattering cross section σ_s [m^2]
        # σ_s = (η' ξ)^2 L0^2 (Eγ/E0)^2
        self.sigma_scatter = (self.eta_prime*self.xi)**2 * (L0_M**2) * (E_GAMMA_PULSE_J/E0_J)**2

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
        # B ≡ xi * B_FDR_BASE   (already accounted via xi in sigma_scatter, but the (1+xi u_em/Uvac) piece contributes xi outside too)
        # Full derivation gives A*(1/D) + A*(xi*B_FDR_BASE)/D^3. We separate xi only once in σ_s, so keep B term as B_FDR_BASE here:
        tau = A*(1.0/np.maximum(D_m, 1e-12)) + (A*self.xi*B_FDR_BASE)/(np.maximum(D_m, 1e-12)**3)
        return tau

# ──────────────────────────────────────────────────────────────────────────────
# Predictors

def qfd_mu_vectorized(z: np.ndarray, kJ: float, eta_p: float, xi: float) -> np.ndarray:
    """
    μ(z) = 5 log10(D_L/Mpc) + 25 + Δμ_FDR
    where D_L = D_true * (1+z), and D_true = ln(1+z)/α0 (drag-only).
    """
    m = QFDModel(k_J=kJ, eta_prime=eta_p, xi=xi)
    z = np.atleast_1d(z).astype(float)
    # geometry
    D_true_mpc = np.log1p(z) / max(m.alpha0, 1e-32) / MPC_TO_M
    mu_geom = 5.0*np.log10(np.maximum(D_true_mpc*(1.0+z), 1e-12)) + 25.0
    # dimming
    delta_mu = 1.085736 * m.tau_fdr_Dmpc(D_true_mpc)
    return mu_geom + delta_mu

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
# Posterior

def log_prior(theta, reparam=False):
    """
    θ = [log10 η′, log10 ξ] or [s=log10 σ_FDR, r=log10 η′/ξ] if reparam=True
    """
    if reparam:
        s, r = theta
        # s ~ N(-36, 3²) (weakly informative)
        # r ~ Uniform(-6, 6) (broad)
        if -6.0 < r < 6.0 and -45.0 < s < -30.0:
            # Add weak Gaussian prior on s
            return -0.5 * ((s + 36.0)/3.0)**2
        return -np.inf
    else:
        log_eta, log_xi = theta
        if -5.0 < log_eta < 5.0 and -5.0 < log_xi < 5.0:
            return 0.0
        return -np.inf

def log_likelihood(theta, z, mu, dmu, kJ, reparam=False):
    """Enhanced likelihood with better error handling."""
    try:
        log_eta, log_xi = None, None
        if reparam:
            # theta = [s, r] -> (log_eta, log_xi)
            s, r = theta
            const = np.log10((L0_M**2) * (E_GAMMA_PULSE_J/E0_J)**2)
            a = (s - const)/4.0 + r/2.0
            b = (s - const)/4.0 - r/2.0
            log_eta, log_xi = a, b
        else:
            log_eta, log_xi = theta

        eta, xi = 10**log_eta, 10**log_xi

        # Check for reasonable parameter values
        if not (1e-10 < eta < 1e10 and 1e-10 < xi < 1e10):
            return -np.inf

        mu_model = qfd_mu_vectorized(z, kJ, eta, xi)

        # Check model validity
        if not np.all(np.isfinite(mu_model)):
            return -np.inf

        sig = np.clip(np.asarray(dmu, float), 1e-3, np.inf)
        r = (mu - mu_model)/sig
        return -0.5*np.sum(r*r + np.log(2*np.pi*sig*sig))

    except Exception as e:
        return -np.inf

def log_probability(theta, z, mu, dmu, kJ, reparam=False):
    lp = log_prior(theta, reparam=reparam)
    if not np.isfinite(lp): return -np.inf
    ll = log_likelihood(theta, z, mu, dmu, kJ, reparam=reparam)
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

def save_enhanced_diagnostics(outdir, args, sampler, eta_best, xi_best, z, mu, dmu, ts):
    """Save comprehensive diagnostics for pipeline integration."""

    # Convergence diagnostics
    converged, tau = check_convergence(sampler)

    # Model diagnostics
    m_map = QFDModel(k_J=args.kJ, eta_prime=eta_best, xi=xi_best)
    sigma_scatter = float(m_map.sigma_scatter)

    def D_mpc_from_z(z_val):
        return np.log1p(z_val) / max(m_map.alpha0, 1e-32) / MPC_TO_M

    def delta_mu_fdr_at(z_val):
        D_mpc = D_mpc_from_z(z_val)
        tau = float(m_map.tau_fdr_Dmpc(np.array([D_mpc]))[0])
        return 1.085736 * tau

    # Enhanced diagnostics
    diag = {
        "version": "5.6",
        "timestamp": ts,
        "convergence": {
            "converged": bool(converged),
            "autocorr_time": tau.tolist() if tau is not None else None,
            "total_iterations": int(sampler.iteration)
        },
        "model_parameters": {
            "k_J": float(args.kJ),
            "eta_prime_best": float(eta_best),
            "xi_best": float(xi_best),
            "sigma_scatter_m2": sigma_scatter,
            "alpha0_inv_Mpc": float(1.0 / (m_map.alpha0 * MPC_TO_M))
        },
        "dimming_curve": [{"z": z_val, "DeltaMu_FDR_mag": float(delta_mu_fdr_at(z_val))}
                         for z_val in (0.1, 0.3, 0.5, 0.8, 1.0, 1.5)],
        "bootstrap_ready": True
    }

    with open(os.path.join(outdir, "enhanced_diagnostics.json"), "w") as f:
        json.dump(diag, f, indent=2)

    # Residuals and fit metrics
    mu_model_full = qfd_mu_vectorized(z, args.kJ, eta_best, xi_best)
    residuals = (mu - mu_model_full).astype(float)
    dmu_safe = np.clip(dmu.astype(float), 1e-3, np.inf)
    chi2 = float(np.sum((residuals/dmu_safe)**2))
    nu   = max(1, len(z) - 2)  # 2 fitted params
    red_chi2 = chi2 / nu
    AIC = chi2 + 2*2
    BIC = chi2 + 2*np.log(len(z))

    summary = {
        "chi2": chi2, "nu": int(nu), "reduced_chi2": red_chi2,
        "AIC": AIC, "BIC": BIC,
        "rms_residual": float(np.sqrt(np.mean(residuals**2))),
        "data_points": len(z),
        "converged": bool(converged)
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
    ap = argparse.ArgumentParser(description="QFD μ–z fitter v5.6 (drag-only redshift; FDR as dimming)")
    ap.add_argument("--data", type=str, default="union2.1_data.txt")
    ap.add_argument("--walkers", type=int, default=32)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", type=str, default=None)
    ap.add_argument("--kJ", type=float, default=KJ_DEFAULT, help="Baseline drag coupling k_J")
    ap.add_argument("--reparam", action="store_true",
                    help="Sample in (s=log10 σ_FDR, r=log10 η′/ξ) instead of (log10 η′, log10 ξ)")
    ap.add_argument("--bootstrap-calibration", action="store_true",
                    help="Bootstrap calibration mode: reduced steps, enhanced diagnostics")
    ap.add_argument("--burn-frac", type=float, default=0.3, help="Fraction of steps to discard as burn-in")
    ap.add_argument("--thin-target", type=int, default=200, help="Target number of samples after thinning")
    args = ap.parse_args()

    # Adjust parameters for bootstrap mode
    if args.bootstrap_calibration:
        args.steps = min(args.steps, 1000)  # Faster for bootstrap
        args.walkers = min(args.walkers, 16)  # WSL-friendly
        print("[bootstrap mode] reduced steps and walkers for efficiency")

    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    outdir = args.outdir or f"QFD_Run_{ts}"
    os.makedirs(outdir, exist_ok=True)
    print(f"--- QFD v5.6: All results will be saved to: {outdir} ---")

    z, mu, dmu = load_union21_data(args.data)
    if z is None:
        print("Halting: data load failed.")
        return 1

    # Sampler setup
    np.random.seed(args.seed); random.seed(args.seed)
    ndim, nwalk = 2, args.walkers

    try:
        if args.reparam:
            # Initialize in (s, r) space
            p0 = np.column_stack([
                np.full(nwalk, -36.0) + 0.1*np.random.randn(nwalk),  # s = log10 σ_FDR
                np.full(nwalk,  0.0) + 0.1*np.random.randn(nwalk),   # r = log10 η′/ξ
            ])
            sampler = emcee.EnsembleSampler(nwalk, ndim, log_probability,
                                            args=(z, mu, dmu, args.kJ),
                                            kwargs={"reparam": True})
            print(f"Running MCMC with reparameterization (s, r) using {nwalk} walkers for {args.steps} steps...")
        else:
            # Initialize in (log10 η′, log10 ξ) space
            p0 = np.column_stack([
                np.full(nwalk, -3.0) + 0.1*np.random.randn(nwalk),  # log10 η′
                np.full(nwalk,  0.0) + 0.1*np.random.randn(nwalk),  # log10 ξ
            ])
            sampler = emcee.EnsembleSampler(nwalk, ndim, log_probability,
                                            args=(z, mu, dmu, args.kJ),
                                            kwargs={"reparam": False})
            print(f"Running MCMC with standard parameterization using {nwalk} walkers for {args.steps} steps...")

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

    med = np.median(flat, axis=0)
    best = flat[np.argmax(lpf)] if lpf.size else med

    # Convert parameters
    if args.reparam:
        def convert_sr_to_eta_xi(sr):
            s, r = sr
            const = np.log10((L0_M**2) * (E_GAMMA_PULSE_J/E0_J)**2)
            log_eta = (s - const)/4.0 + r/2.0
            log_xi = (s - const)/4.0 - r/2.0
            return 10**log_eta, 10**log_xi

        eta_med, xi_med = convert_sr_to_eta_xi(med)
        eta_best, xi_best = convert_sr_to_eta_xi(best)
        print(f"Best-fit (MAP): s={best[0]:.3f}, r={best[1]:.3f} → η'={eta_best:.3e}, ξ={xi_best:.3f}")
    else:
        eta_med, xi_med = 10**med[0], 10**med[1]
        eta_best, xi_best = 10**best[0], 10**best[1]
        print(f"Best-fit (MAP): η'={eta_best:.3e}, ξ={xi_best:.3f}")

    # Save metadata
    meta = {
        "version": "5.6",
        "timestamp": ts, "seed": args.seed,
        "walkers": args.walkers, "steps": args.steps,
        "burn": burn, "thin": thin,
        "k_J_used": float(args.kJ),
        "eta_prime_map": float(eta_best), "xi_map": float(xi_best),
        "eta_prime_med": float(eta_med), "xi_med": float(xi_med),
        "bootstrap_mode": args.bootstrap_calibration,
        "reparam_mode": args.reparam
    }
    with open(os.path.join(outdir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Enhanced diagnostics
    save_enhanced_diagnostics(outdir, args, sampler, eta_best, xi_best, z, mu, dmu, ts)

    # Corner plot
    if args.reparam:
        labels = [r"$s = \log_{10}\sigma_{\rm FDR}$", r"$r = \log_{10}(\eta'/\xi)$"]
    else:
        labels = [r"$\log_{10}\eta'$", r"$\log_{10}\xi$"]

    fig = corner.corner(flat, labels=labels, quantiles=[0.16,0.5,0.84], show_titles=True)
    cp = os.path.join(outdir, f"qfd_corner_plot_{ts}.png")
    fig.savefig(cp, dpi=150, bbox_inches='tight'); plt.close(fig)
    print("Saved:", cp)

    # Hubble diagram
    z_grid = np.linspace(np.min(z), np.max(z), 200)
    mu_lcdm = lcdm_mu(z_grid)
    mu_qfd  = qfd_mu_vectorized(z_grid, args.kJ, eta_best, xi_best)

    fig, ax = plt.subplots(figsize=(12,8))
    ax.errorbar(z, mu, yerr=dmu, fmt='.', alpha=0.6, label='Union2.1')
    ax.plot(z_grid, mu_lcdm, 'r--', lw=2, label=r'$\Lambda$CDM (Ω$_M$=0.3)')
    ax.plot(z_grid, mu_qfd,  'b-',  lw=2.5, label='QFD (MAP)')
    ax.set_xlabel('Redshift z'); ax.set_ylabel('Distance Modulus μ')
    ax.set_title('QFD vs. ΛCDM on Union2.1 (v5.6)')
    ax.grid(True, ls=':', alpha=0.4); ax.legend()
    hp = os.path.join(outdir, f"qfd_hubble_diagram_{ts}.png")
    fig.savefig(hp, dpi=150, bbox_inches='tight'); plt.close(fig)
    print("Saved:", hp)

    # Redshift decomposition plot
    m_map = QFDModel(k_J=args.kJ, eta_prime=eta_best, xi=xi_best)
    D_true_mpc_grid = np.log1p(z_grid)/max(m_map.alpha0,1e-32)/MPC_TO_M
    delta_mu_fdr = 1.085736 * m_map.tau_fdr_Dmpc(D_true_mpc_grid)

    fig, ax1 = plt.subplots(figsize=(11,7))
    color1 = 'tab:blue'
    ax1.plot(z_grid, z_grid, color='k', lw=1.5, label='z_cosmo(z) (drag-only)')
    ax1.set_xlabel('Observed redshift z')
    ax1.set_ylabel('z_cosmo', color='k')
    ax1.tick_params(axis='y', labelcolor='k')
    ax1.grid(True, ls=':', alpha=0.4)

    ax2 = ax1.twinx()
    color2 = 'tab:green'
    ax2.plot(z_grid, delta_mu_fdr, color=color2, lw=2.5, label='Δμ_FDR (MAP)')
    ax2.set_ylabel('Δμ_FDR [mag]', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(-2, 2))

    fig.tight_layout()
    dp = os.path.join(outdir, f"qfd_redshift_decomposition_{ts}.png")
    fig.savefig(dp, dpi=150, bbox_inches='tight'); plt.close(fig)
    print("Saved:", dp)

    print("\n--- QFD v5.6 Analysis Complete ---")

    # Check convergence and provide recommendation
    converged, tau = check_convergence(sampler)
    if converged:
        print("✅ MCMC converged successfully")
    else:
        print("⚠️  MCMC may not have converged - consider more steps")

    if args.bootstrap_calibration:
        print("🔧 Bootstrap calibration mode outputs ready for pipeline integration")

    return 0

# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sys.exit(main())