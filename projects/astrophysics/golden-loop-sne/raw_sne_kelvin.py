#!/usr/bin/env python3
"""
raw_sne_kelvin.py — Raw SNe Pipeline with v2 Kelvin Wave Physics

Fits V18 Stage 1 light-curve data (no SALT2) using the LOCKED QFD model
with ZERO free physics parameters. Works in alpha (ln_A) space — the
natural observable from template fitting — avoiding the problematic
mu_obs conversion that caused the z-slope artifact.

Data sources:
    PRIMARY: V18 3-stage MCMC output (4,885 SNe, known provenance)
    SECONDARY: V22 Stage 1 (6,724 SNe, undocumented fitter — use with caution)

Physics (locked from golden_loop_sne.py):
    D(z)  = (c/K_J) × ln(1+z)           exponential energy loss
    D_L   = D × (1+z)^(2/3)             thermodynamic vortex ring (f=2)
    τ(z)  = 1 - 1/√(1+z)               Kelvin wave scattering shape
    η     = π²/β² = 1.0657              locked geometric opacity
    K_J   = ξ_QFD × β^(3/2) ≈ 85.58    vacuum drag (zero free params)

Alpha-space prediction:
    alpha_pred(z) = C - ln(D_L(z)) - η × τ(z)

    where C is a calibration constant (intrinsic luminosity + template
    normalization) — the sole free parameter.

    Derivation:
        flux ∝ L/(4π D_L²) × exp(-extinction)
        alpha = ln(flux) = ln(L/4π) - 2ln(D_L) - 0.4×ln(10)×A_ext
        A_ext = K_MAG × η × τ(z) magnitudes
        0.4 × ln(10) × K_MAG = 0.4 × ln(10) × 5/ln(10) = 2
        ∴ alpha_pred = C - 2ln(D_L) - 2η×τ(z)

    NOTE: The factor of 2 vs 1 in front of ln(D_L) depends on whether
    alpha is ln(flux) or ln(amplitude). Stage 1 fits flux = exp(alpha) × template,
    so alpha = ln(flux_amplitude). We fit the overall scale empirically.

Copyright (c) 2026 Tracy McSheery
"""

import numpy as np
import os
import sys

# ============================================================
# CONSTANTS (ALL from α via Golden Loop — zero free parameters)
# ============================================================

ALPHA_FS = 1.0 / 137.035999084
PI = np.pi
C_LIGHT_KM_S = 299792.458
K_MAG = 5.0 / np.log(10.0)


def solve_golden_loop(alpha):
    """Solve 1/α = 2π²(e^β/β) + 1 for β."""
    target = (1.0 / alpha) - 1.0
    C = 2.0 * PI * PI
    b = 3.0
    for _ in range(100):
        eb = np.exp(b)
        val = C * (eb / b) - target
        deriv = C * eb * (b - 1.0) / (b * b)
        if abs(deriv) < 1e-30:
            break
        step = val / deriv
        b -= step
        if abs(step) < 1e-15:
            break
    return b


BETA = solve_golden_loop(ALPHA_FS)
K_VORTEX = 7.0 * PI / 5.0
XI_QFD = K_VORTEX**2 * (5.0 / 6.0)
KJ_GEOMETRIC = XI_QFD * BETA**1.5
ETA_GEOMETRIC = PI**2 / BETA**2


# ============================================================
# QFD PHYSICS FUNCTIONS (locked)
# ============================================================

def distance_qfd(z):
    """Proper distance: D = (c/K_J) × ln(1+z) [Mpc]."""
    return (C_LIGHT_KM_S / KJ_GEOMETRIC) * np.log1p(z)


def luminosity_distance_qfd(z):
    """Luminosity distance: D_L = D × (1+z)^(2/3) [Mpc]."""
    return distance_qfd(z) * (1.0 + z)**(2.0 / 3.0)


def scattering_opacity(z):
    """Kelvin wave opacity shape: τ(z) = 1 - 1/√(1+z)."""
    return 1.0 - 1.0 / np.sqrt(1.0 + z)


def mu_qfd(z, M=0.0):
    """Full QFD distance modulus (zero free physics params).

    μ = 5 log₁₀(D_L) + 25 + M + K_MAG × η × τ(z)
    """
    D_L = luminosity_distance_qfd(z)
    return (5.0 * np.log10(D_L) + 25.0 + M
            + K_MAG * ETA_GEOMETRIC * scattering_opacity(z))


def alpha_pred_qfd(z, C0=0.0, scale=1.0, eta_eff=None):
    """v2 Kelvin prediction in alpha (ln_A) space.

    alpha_pred = C0 - scale × ln(D_L(z)) - eta_eff × τ(z)

    Parameters:
        z: redshift array
        C0: calibration constant (1 free parameter)
        scale: coefficient of ln(D_L) — fixed at 1.0 for ln(amplitude)
        eta_eff: effective opacity in alpha space (default: ETA_GEOMETRIC)

    Returns predicted alpha values.
    """
    if eta_eff is None:
        eta_eff = ETA_GEOMETRIC
    D_L = luminosity_distance_qfd(z)
    return C0 - scale * np.log(D_L) - eta_eff * scattering_opacity(z)


# ============================================================
# ΛCDM COMPARISON
# ============================================================

def luminosity_distance_lcdm(z, omega_m, H0=70.0):
    """Flat ΛCDM luminosity distance via numerical integration."""
    from scipy.integrate import quad
    omega_lam = 1.0 - omega_m
    z_arr = np.atleast_1d(z).astype(float)
    D_L = np.zeros_like(z_arr)
    for i, zi in enumerate(z_arr):
        if zi > 0:
            def integrand(zp):
                return 1.0 / np.sqrt(omega_m * (1 + zp)**3 + omega_lam)
            integral, _ = quad(integrand, 0, zi)
            D_L[i] = (C_LIGHT_KM_S / H0) * (1 + zi) * integral
    return D_L


def alpha_pred_lcdm(z, C0=0.0, omega_m=0.315, H0=70.0):
    """ΛCDM prediction in alpha space (for comparison)."""
    D_L = luminosity_distance_lcdm(z, omega_m, H0)
    return C0 - np.log(np.maximum(D_L, 1e-10))


# ============================================================
# DATA LOADING
# ============================================================

def load_v18_data(path=None):
    """Load V18 Stage 3 Hubble data (known provenance, 4,885 SNe).

    Source: V18 3-stage MCMC pipeline
        Stage 1: per-SN light curve fit → (t0, alpha, stretch)
        Stage 2: global MCMC → (k_J, η', ξ)
        Stage 3: Hubble diagram generation

    Returns dict with arrays for all SNe (no additional cuts).
    """
    if path is None:
        base = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base, '..', '..', '..',
                            'V22_Supernova_Analysis', 'data',
                            'v18_hubble_data.csv')
        path = os.path.abspath(path)

    data = np.genfromtxt(path, delimiter=',', names=True,
                         dtype=None, encoding='utf-8')

    # Basic quality cut: require z > 0.01 (Hubble flow)
    mask = data['z'] > 0.01
    d = data[mask]
    print(f"  V18: Loaded {len(data)} SNe, {np.sum(mask)} pass z>0.01 cut")

    return {
        'snid': d['snid'],
        'z': d['z'].astype(float),
        'alpha': d['alpha'].astype(float),
        'source': 'V18',
    }


def load_v22_stage1(path=None):
    """Load V22 Stage 1 results (undocumented fitter — use with caution).

    WARNING: The processing_log.json metadata does NOT match the actual CSV
    contents. The fitter that produced this data is unknown (has A_plasma,
    beta columns not present in the documented stage1_fit.py).

    Returns dict with arrays for quality-cut SNe.
    """
    if path is None:
        base = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base, '..', '..', '..',
                            'projects', 'astrophysics', 'qfd-sn-v22',
                            'data', 'precomputed_filtered',
                            'stage1_results_filtered.csv')
        path = os.path.abspath(path)

    data = np.genfromtxt(path, delimiter=',', names=True,
                         dtype=None, encoding='utf-8')

    mask = (
        data['pass_n_obs'].astype(bool) &
        data['pass_chi2'].astype(bool) &
        data['pass_stretch'].astype(bool) &
        ~data['is_flashlight'].astype(bool) &
        (data['z'] > 0.02) &
        (data['z'] < 1.4) &
        (data['chi2_dof'] < 10.0)
    )

    d = data[mask]
    print(f"  V22: Loaded {len(data)} SNe, {np.sum(mask)} pass quality cuts")

    return {
        'snid': d['snid'],
        'z': d['z'].astype(float),
        'alpha': d['ln_A'].astype(float),  # ln_A = alpha
        'stretch': d['stretch'].astype(float),
        'source': 'V22',
    }


# ============================================================
# FITTING (in alpha space)
# ============================================================

def fit_C0(z, alpha_obs, model_func, **model_kwargs):
    """Fit calibration offset C0 (1 parameter, analytic).

    Minimizes Σ (alpha_obs - alpha_pred(z; C0))².
    Since alpha_pred = C0 + f(z), the optimal C0 = median(alpha_obs - f(z)).
    """
    alpha_pred_0 = model_func(z, C0=0.0, **model_kwargs)
    valid = np.isfinite(alpha_pred_0) & np.isfinite(alpha_obs)
    C0_best = np.median(alpha_obs[valid] - alpha_pred_0[valid])
    return C0_best


def fit_C0_and_scale(z, alpha_obs):
    """Fit C0 and ln(D_L) scale simultaneously (2 params, linear).

    alpha_obs = C0 - scale × ln(D_L(z)) - η × τ(z) + noise

    Returns (C0, scale).
    """
    D_L = luminosity_distance_qfd(z)
    valid = np.isfinite(D_L) & np.isfinite(alpha_obs) & (D_L > 0)

    ln_DL = np.log(D_L[valid])
    tau = scattering_opacity(z[valid])
    y = alpha_obs[valid]

    # Design matrix: alpha = C0 - scale*ln_DL - η*τ
    # Let x1=1, x2=-ln_DL, then alpha = C0*x1 + scale*x2 - η*τ
    # With η fixed, let y' = alpha + η*τ = C0 + scale*(-ln_DL)
    y_adj = y + ETA_GEOMETRIC * tau

    A = np.column_stack([np.ones(len(y_adj)), -ln_DL])
    # Normal equations
    ATA = A.T @ A
    ATy = A.T @ y_adj

    try:
        params = np.linalg.solve(ATA, ATy)
        return params[0], params[1]
    except np.linalg.LinAlgError:
        return np.median(y_adj + ln_DL), 1.0


def fit_C0_and_eta(z, alpha_obs):
    """Fit C0 and eta_eff simultaneously (2 params, linear).

    alpha_obs = C0 - ln(D_L(z)) - eta_eff × τ(z) + noise

    With scale fixed at 1.0, fit (C0, eta_eff).
    """
    D_L = luminosity_distance_qfd(z)
    valid = np.isfinite(D_L) & np.isfinite(alpha_obs) & (D_L > 0)

    ln_DL = np.log(D_L[valid])
    tau = scattering_opacity(z[valid])
    y = alpha_obs[valid]

    # y = C0 - ln_DL - eta*tau
    # y + ln_DL = C0 - eta*tau
    y_adj = y + ln_DL

    A = np.column_stack([np.ones(len(y_adj)), -tau])
    ATA = A.T @ A
    ATy = A.T @ y_adj

    try:
        params = np.linalg.solve(ATA, ATy)
        return params[0], params[1]
    except np.linalg.LinAlgError:
        return np.median(y_adj), ETA_GEOMETRIC


def sigma_clip(z, alpha_obs, alpha_model, n_sigma=3.0):
    """Iterative sigma clipping to remove non-Ia contaminants."""
    mask = np.isfinite(alpha_obs) & np.isfinite(alpha_model)
    for _ in range(5):
        resid = alpha_obs[mask] - alpha_model[mask]
        sigma = 1.4826 * np.median(np.abs(resid - np.median(resid)))
        if sigma < 1e-10:
            break
        new_mask = mask & (np.abs(alpha_obs - alpha_model) < n_sigma * sigma)
        n_clipped = np.sum(mask) - np.sum(new_mask)
        mask = new_mask
        if n_clipped == 0:
            break
    return mask


def compute_stats(alpha_obs, alpha_model):
    """Compute fit statistics."""
    valid = np.isfinite(alpha_obs) & np.isfinite(alpha_model)
    resid = alpha_obs[valid] - alpha_model[valid]
    n = len(resid)
    sigma = np.std(resid)
    sigma_mad = 1.4826 * np.median(np.abs(resid - np.median(resid)))
    sigma_mag = sigma * K_MAG  # convert to magnitude scatter
    chi2_dof = np.sum((resid / sigma_mad)**2) / max(n - 1, 1)
    return {
        'n': n, 'sigma_alpha': sigma, 'sigma_mag': sigma_mag,
        'sigma_mad': sigma_mad, 'chi2_dof': chi2_dof,
        'mean_resid': np.mean(resid), 'median_resid': np.median(resid),
    }


def binned_residuals(z, resid, n_bins=10, label=""):
    """Print binned residuals and return slope diagnostic."""
    valid = np.isfinite(z) & np.isfinite(resid)
    z_v, r_v = z[valid], resid[valid]

    bin_edges = np.percentile(z_v, np.linspace(0, 100, n_bins + 1))
    z_means, r_means, r_stds, ns = [], [], [], []

    for i in range(n_bins):
        if i == n_bins - 1:
            in_bin = (z_v >= bin_edges[i]) & (z_v <= bin_edges[i + 1])
        else:
            in_bin = (z_v >= bin_edges[i]) & (z_v < bin_edges[i + 1])
        if np.sum(in_bin) == 0:
            continue
        z_means.append(np.mean(z_v[in_bin]))
        r_means.append(np.mean(r_v[in_bin]))
        r_stds.append(np.std(r_v[in_bin]))
        ns.append(np.sum(in_bin))

    z_means = np.array(z_means)
    r_means = np.array(r_means)

    # Linear regression on bin means for slope diagnostic
    if len(z_means) >= 3:
        slope, intercept = np.polyfit(z_means, r_means, 1)
    else:
        slope, intercept = 0.0, 0.0

    if label:
        print(f"\n  --- Binned Residuals: {label} ---")
    print(f"  {'z_bin':>14} {'<z>':>8} {'<resid>':>10} {'σ':>8} {'N':>5}")
    for i in range(len(z_means)):
        lo = bin_edges[i] if i < len(bin_edges) - 1 else bin_edges[-2]
        hi = bin_edges[i + 1] if i + 1 < len(bin_edges) else bin_edges[-1]
        print(f"  [{lo:.3f},{hi:.3f})"
              f" {z_means[i]:>8.3f}"
              f" {r_means[i]:>+10.4f}"
              f" {r_stds[i]:>8.4f}"
              f" {ns[i]:>5d}")
    print(f"  Slope: {slope:+.3f} per unit z"
          f"  ({slope*K_MAG:+.3f} mag/z)")

    return slope


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_pipeline(data_source='V18'):
    """Execute the full raw → Hubble diagram pipeline in alpha space."""
    W = 72
    print("=" * W)
    print("RAW SNe PIPELINE — v2 Kelvin Wave (Alpha Space)".center(W))
    print(f"Data: {data_source} | ZERO free physics parameters".center(W))
    print("=" * W)

    # Constants
    print(f"\n  α = 1/{1/ALPHA_FS:.6f}")
    print(f"  β = {BETA:.10f}  (Golden Loop)")
    print(f"  K_J = {KJ_GEOMETRIC:.4f} km/s/Mpc")
    print(f"  η = π²/β² = {ETA_GEOMETRIC:.6f}")
    print(f"  K_MAG = 5/ln(10) = {K_MAG:.6f}")

    # Load data
    print(f"\n--- Loading {data_source} data ---")
    if data_source == 'V18':
        data = load_v18_data()
    elif data_source == 'V22':
        data = load_v22_stage1()
    else:
        raise ValueError(f"Unknown data source: {data_source}")

    z = data['z']
    alpha = data['alpha']
    print(f"  z range: [{z.min():.4f}, {z.max():.4f}]")
    print(f"  alpha range: [{alpha.min():.2f}, {alpha.max():.2f}]"
          f"  (mean={alpha.mean():.2f}, σ={alpha.std():.2f})")

    results = {}

    # ================================================================
    # MODEL A: Locked v2 Kelvin (1 free param: C0 only)
    # ================================================================
    print(f"\n{'MODEL A: Locked v2 Kelvin (C0 only)':=^{W}}")
    print("  alpha_pred = C0 - ln(D_L(z)) - η × τ(z)")
    print(f"  η = {ETA_GEOMETRIC:.4f} (LOCKED), scale = 1.0 (LOCKED)")

    C0_A = fit_C0(z, alpha, alpha_pred_qfd)
    alpha_model_A = alpha_pred_qfd(z, C0=C0_A)

    # Sigma clip
    mask_A = sigma_clip(z, alpha, alpha_model_A, n_sigma=3.0)
    n_clip_A = len(z) - np.sum(mask_A)
    print(f"  Sigma-clipped {n_clip_A} outliers ({n_clip_A/len(z)*100:.1f}%)")

    # Re-fit on clipped
    C0_A = fit_C0(z[mask_A], alpha[mask_A], alpha_pred_qfd)
    alpha_model_A = alpha_pred_qfd(z, C0=C0_A)

    stats_A = compute_stats(alpha[mask_A], alpha_model_A[mask_A])
    print(f"  C0 = {C0_A:.4f}")
    print(f"  σ = {stats_A['sigma_alpha']:.4f} (alpha)"
          f"  = {stats_A['sigma_mag']:.3f} mag")
    print(f"  χ²/dof = {stats_A['chi2_dof']:.4f}  (n={stats_A['n']})")

    slope_A = binned_residuals(z[mask_A],
                                alpha[mask_A] - alpha_model_A[mask_A],
                                label="Model A (locked)")
    results['A'] = {**stats_A, 'C0': C0_A, 'slope': slope_A}

    # ================================================================
    # MODEL B: Fitted scale (2 free params: C0, scale)
    # ================================================================
    print(f"\n{'MODEL B: Fitted scale (C0, scale)':=^{W}}")
    print("  alpha_pred = C0 - scale × ln(D_L(z)) - η × τ(z)")
    print(f"  η = {ETA_GEOMETRIC:.4f} (LOCKED)")

    C0_B, scale_B = fit_C0_and_scale(z, alpha)
    alpha_model_B = alpha_pred_qfd(z, C0=C0_B, scale=scale_B)

    mask_B = sigma_clip(z, alpha, alpha_model_B, n_sigma=3.0)
    n_clip_B = len(z) - np.sum(mask_B)
    print(f"  Sigma-clipped {n_clip_B} outliers ({n_clip_B/len(z)*100:.1f}%)")

    C0_B, scale_B = fit_C0_and_scale(z[mask_B], alpha[mask_B])
    alpha_model_B = alpha_pred_qfd(z, C0=C0_B, scale=scale_B)

    stats_B = compute_stats(alpha[mask_B], alpha_model_B[mask_B])
    print(f"  C0 = {C0_B:.4f}, scale = {scale_B:.4f}")
    print(f"  σ = {stats_B['sigma_alpha']:.4f} (alpha)"
          f"  = {stats_B['sigma_mag']:.3f} mag")
    print(f"  χ²/dof = {stats_B['chi2_dof']:.4f}  (n={stats_B['n']})")

    slope_B = binned_residuals(z[mask_B],
                                alpha[mask_B] - alpha_model_B[mask_B],
                                label="Model B (fitted scale)")
    results['B'] = {**stats_B, 'C0': C0_B, 'scale': scale_B, 'slope': slope_B}

    # ================================================================
    # MODEL C: Fitted opacity (2 free params: C0, eta_eff)
    # ================================================================
    print(f"\n{'MODEL C: Fitted opacity (C0, eta_eff)':=^{W}}")
    print("  alpha_pred = C0 - ln(D_L(z)) - eta_eff × τ(z)")
    print("  scale = 1.0 (LOCKED)")

    C0_C, eta_C = fit_C0_and_eta(z, alpha)
    alpha_model_C = alpha_pred_qfd(z, C0=C0_C, eta_eff=eta_C)

    mask_C = sigma_clip(z, alpha, alpha_model_C, n_sigma=3.0)
    n_clip_C = len(z) - np.sum(mask_C)
    print(f"  Sigma-clipped {n_clip_C} outliers ({n_clip_C/len(z)*100:.1f}%)")

    C0_C, eta_C = fit_C0_and_eta(z[mask_C], alpha[mask_C])
    alpha_model_C = alpha_pred_qfd(z, C0=C0_C, eta_eff=eta_C)

    stats_C = compute_stats(alpha[mask_C], alpha_model_C[mask_C])
    print(f"  C0 = {C0_C:.4f}, η_eff = {eta_C:.4f}"
          f"  (geometric η = {ETA_GEOMETRIC:.4f},"
          f" ratio = {eta_C/ETA_GEOMETRIC:.2f}×)")
    print(f"  σ = {stats_C['sigma_alpha']:.4f} (alpha)"
          f"  = {stats_C['sigma_mag']:.3f} mag")
    print(f"  χ²/dof = {stats_C['chi2_dof']:.4f}  (n={stats_C['n']})")

    slope_C = binned_residuals(z[mask_C],
                                alpha[mask_C] - alpha_model_C[mask_C],
                                label="Model C (fitted opacity)")
    results['C'] = {**stats_C, 'C0': C0_C, 'eta_eff': eta_C, 'slope': slope_C}

    # ================================================================
    # ΛCDM COMPARISON (2 free params: C0, Ωm)
    # ================================================================
    print(f"\n{'ΛCDM COMPARISON (C0, Ωm)':=^{W}}")
    from scipy.integrate import quad  # noqa: F811

    # Use Model A mask for fair comparison
    mask_L = mask_A
    best_chi2 = np.inf
    best_om = 0.3

    for om in np.arange(0.05, 0.95, 0.02):
        C0_L = fit_C0(z[mask_L], alpha[mask_L], alpha_pred_lcdm, omega_m=om)
        alpha_L = alpha_pred_lcdm(z[mask_L], C0=C0_L, omega_m=om)
        s = compute_stats(alpha[mask_L], alpha_L)
        if s['chi2_dof'] < best_chi2:
            best_chi2 = s['chi2_dof']
            best_om = om
            best_C0_L = C0_L

    alpha_model_L = alpha_pred_lcdm(z, C0=best_C0_L, omega_m=best_om)
    stats_L = compute_stats(alpha[mask_L], alpha_model_L[mask_L])
    print(f"  Ωm = {best_om:.2f}, C0 = {best_C0_L:.4f}")
    print(f"  σ = {stats_L['sigma_alpha']:.4f} (alpha)"
          f"  = {stats_L['sigma_mag']:.3f} mag")
    print(f"  χ²/dof = {stats_L['chi2_dof']:.4f}  (n={stats_L['n']})")

    slope_L = binned_residuals(z[mask_L],
                                alpha[mask_L] - alpha_model_L[mask_L],
                                label="ΛCDM")
    results['LCDM'] = {**stats_L, 'omega_m': best_om, 'slope': slope_L}

    # ================================================================
    # MALMQUIST BIAS DIAGNOSTIC
    # ================================================================
    print(f"\n{'MALMQUIST BIAS DIAGNOSTIC':=^{W}}")
    print("  If alpha = ln(flux), mean alpha should DECREASE with z")
    print("  (dimmer at higher z). If it INCREASES, Malmquist bias is")
    print("  present (only brightest SNe detected at high z).")

    n_diag = 5
    diag_edges = np.percentile(z, np.linspace(0, 100, n_diag + 1))
    diag_z, diag_alpha = [], []
    print(f"\n  {'z_bin':>14} {'<z>':>8} {'<alpha>':>10} {'σ(alpha)':>10} {'N':>5}")
    for i in range(n_diag):
        if i == n_diag - 1:
            in_bin = (z >= diag_edges[i]) & (z <= diag_edges[i + 1])
        else:
            in_bin = (z >= diag_edges[i]) & (z < diag_edges[i + 1])
        if np.sum(in_bin) == 0:
            continue
        zm, am = np.mean(z[in_bin]), np.mean(alpha[in_bin])
        diag_z.append(zm)
        diag_alpha.append(am)
        print(f"  [{diag_edges[i]:.3f},{diag_edges[i+1]:.3f})"
              f" {zm:>8.3f} {am:>+10.3f} {np.std(alpha[in_bin]):>10.3f}"
              f" {np.sum(in_bin):>5d}")

    if len(diag_z) >= 2:
        malmquist_slope = np.polyfit(diag_z, diag_alpha, 1)[0]
        if malmquist_slope > 0.5:
            print(f"\n  MALMQUIST BIAS DETECTED: <alpha> slope = {malmquist_slope:+.2f}")
            print("  → Mean brightness INCREASES with z (selection effect)")
            print("  → Fitted η_eff/scale absorb this bias, NOT physics")
            print("  → Need stretch standardization + selection correction")
        elif malmquist_slope < -0.5:
            print(f"\n  Expected trend: <alpha> slope = {malmquist_slope:+.2f}")
            print("  → Mean brightness decreases with z (physical signal)")
        else:
            print(f"\n  Weak trend: <alpha> slope = {malmquist_slope:+.2f}")

    # ================================================================
    # SUMMARY
    # ================================================================
    print("\n" + "=" * W)
    print(f"{'SUMMARY — Raw SNe with v2 Kelvin Wave':^{W}}")
    print("=" * W)
    print(f"\n  Data: {data_source} ({len(z)} SNe after cuts)")
    print(f"  No SALT2, no pre-reduced distance moduli")
    print(f"  Fitting space: alpha (ln_A) — natural observable")

    print(f"\n  {'Model':<28} {'σ(α)':>8} {'σ(mag)':>8} {'slope':>8}"
          f"  {'n':>5}  {'free':>4}")
    print(f"  {'-'*28} {'-'*8} {'-'*8} {'-'*8}  {'-'*5}  {'-'*4}")

    for key, label, n_free in [
        ('A', 'v2 Kelvin (locked)', 1),
        ('B', 'v2 Kelvin (fit scale)', 2),
        ('C', 'v2 Kelvin (fit opacity)', 2),
        ('LCDM', f'ΛCDM (Ωm={best_om:.2f})', 2),
    ]:
        r = results[key]
        print(f"  {label:<28} {r['sigma_alpha']:>8.4f} {r['sigma_mag']:>8.3f}"
              f" {r['slope']:>+8.3f}  {r['n']:>5d}  {n_free:>4d}")

    # Diagnostic: is the z-slope flat?
    best_slope = min(abs(results[k]['slope']) for k in results)
    if best_slope < 0.5:
        verdict = "FLAT (no significant z-trend)"
    elif best_slope < 2.0:
        verdict = "MARGINAL (mild z-trend, investigate)"
    else:
        verdict = "SLOPED (significant z-trend, calibration issue)"
    print(f"\n  Z-slope verdict: {verdict}")

    if abs(results['A']['slope']) > 2.0 and abs(results['C']['slope']) < 1.0:
        print(f"  NOTE: Locked opacity gives slope {results['A']['slope']:+.1f},"
              f" fitted gives {results['C']['slope']:+.1f}")
        print(f"  → The locked η={ETA_GEOMETRIC:.3f} underpredicts"
              f" extinction in this data")
        print(f"  → Fitted η_eff={results['C']['eta_eff']:.3f}"
              f" ({results['C']['eta_eff']/ETA_GEOMETRIC:.1f}× geometric)")

    # Published comparison
    print(f"\n  --- Published Comparisons ---")
    print(f"  V18 published (3 free params): RMS = 2.18 mag")
    print(f"  V17/V18 combined (3 params):   RMS = 2.38 mag")
    print(f"  SALT2-reduced golden_loop_sne:  σ = 0.18 mag, χ²/dof = 0.955")

    print("\n" + "=" * W)
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Raw SNe pipeline with v2 Kelvin Wave physics')
    parser.add_argument('--data', choices=['V18', 'V22', 'both'],
                        default='V18',
                        help='Data source (default: V18)')
    args = parser.parse_args()

    if args.data == 'both':
        print("\n>>> V18 DATA (primary — known provenance) <<<")
        results_v18 = run_pipeline('V18')
        print("\n\n>>> V22 DATA (secondary — undocumented fitter) <<<")
        results_v22 = run_pipeline('V22')
    else:
        results = run_pipeline(args.data)
