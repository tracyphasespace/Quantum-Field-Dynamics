#!/usr/bin/env python3
"""
Independent verification: χ²/dof for NEW QFD framework vs DES-SN5YR data.

Compares three models against the DES-SN5YR Type Ia supernovae dataset:

  1. QFD NEW:  D_L ∝ ln(1+z)×(1+z)^(2/3),  τ = η[1-(1+z)^(-1/2)]   (σ_nf ∝ √E)
  2. QFD OLD:  D_L ∝ ln(1+z)×(1+z)^(1/2),  τ = η[1-1/(1+z)²]        (σ ∝ E²)
  3. ΛCDM:     D_L from Friedmann eqs,       2 free params (Ω_m, M)

QFD models: 1 free parameter (M, magnitude offset — degenerate with K_J)
ΛCDM model: 2 free parameters (Ω_m and M)

Created: 2026-02-15
Purpose: Verify edits42-F claim (χ²/dof = 0.9546)
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar, minimize
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from qfd.shared_constants import BETA, ALPHA, K_J_KM_S_MPC, XI_QFD

# ===================================================================
# Constants
# ===================================================================
C_KM_S = 299792.458  # km/s
ETA = np.pi**2 / BETA**2  # Geometric opacity limit ≈ 1.0657
K_J = K_J_KM_S_MPC  # ≈ 85.76 km/s/Mpc
K_MAG = 2.5 / np.log(10.0)  # ≈ 1.0857 (mag/ln conversion)

print("=" * 72)
print("DES-SN5YR INDEPENDENT FIT: QFD NEW vs QFD OLD vs ΛCDM")
print("=" * 72)
print(f"\n  Constants:")
print(f"    β = {BETA:.9f}")
print(f"    η = π²/β² = {ETA:.6f}")
print(f"    K_J = ξ×β^(3/2) = {K_J:.4f} km/s/Mpc")
print(f"    K_mag = 2.5/ln10 = {K_MAG:.4f}")

# ===================================================================
# Load DES-SN5YR data
# ===================================================================
data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         '..', 'data', 'raw', 'des5yr_full.csv')
data = np.genfromtxt(data_path, delimiter=',', skip_header=1)
z_all = data[:, 0]
mu_all = data[:, 1]
sigma_all = data[:, 2]

print(f"\n  Data loaded: {len(z_all)} SNe from des5yr_full.csv")
print(f"    z range: [{z_all.min():.4f}, {z_all.max():.4f}]")
print(f"    μ range: [{mu_all.min():.2f}, {mu_all.max():.2f}]")

# Quality cuts (book: 1,829 → 1,768)
# Cut SNe with very large errors or z < 0.01 (peculiar velocity dominated)
mask = (sigma_all < 1.0) & (z_all > 0.01)
z = z_all[mask]
mu_obs = mu_all[mask]
sigma = sigma_all[mask]
N = len(z)
print(f"    After quality cuts (σ < 1.0, z > 0.01): {N} SNe")

# ===================================================================
# Model definitions
# ===================================================================

def mu_qfd_new(z, M_offset):
    """QFD NEW: D_L ∝ ln(1+z) × (1+z)^(2/3), τ from σ ∝ √E."""
    D_L = (C_KM_S / K_J) * np.log(1 + z) * (1 + z)**(2.0/3.0)
    tau = ETA * (1 - (1 + z)**(-0.5))
    return 5.0 * np.log10(D_L) + 25.0 + M_offset + K_MAG * tau

def mu_qfd_old(z, M_offset):
    """QFD OLD: D_L ∝ ln(1+z) × √(1+z), τ from σ ∝ E²."""
    D_L = (C_KM_S / K_J) * np.log(1 + z) * np.sqrt(1 + z)
    tau = ETA * (1 - 1.0 / (1 + z)**2)
    return 5.0 * np.log10(D_L) + 25.0 + M_offset + K_MAG * tau

def mu_lcdm(z, Omega_m, M_offset):
    """Flat ΛCDM: D_L from Friedmann equations."""
    Omega_L = 1.0 - Omega_m
    H0 = 70.0  # Absorbed into M_offset

    z_arr = np.atleast_1d(z)
    D_L = np.zeros_like(z_arr, dtype=float)

    for i, zi in enumerate(z_arr):
        if zi > 0:
            integrand = lambda zp: 1.0 / np.sqrt(Omega_m * (1+zp)**3 + Omega_L)
            integral, _ = quad(integrand, 0, zi)
            D_L[i] = (C_KM_S / H0) * (1 + zi) * integral
        else:
            D_L[i] = 1e-10

    if np.ndim(z) == 0:
        D_L = float(D_L[0])

    return 5.0 * np.log10(D_L) + 25.0 + M_offset

# ===================================================================
# Chi-squared fitting
# ===================================================================

def chi2_qfd(M_offset, model_func, z, mu_obs, sigma):
    """χ² for a QFD model with 1 free parameter (M)."""
    mu_pred = model_func(z, M_offset)
    return np.sum(((mu_obs - mu_pred) / sigma)**2)

def chi2_lcdm(params, z, mu_obs, sigma):
    """χ² for ΛCDM with 2 free parameters (Ω_m, M)."""
    Omega_m, M_offset = params
    if Omega_m < 0.01 or Omega_m > 0.99:
        return 1e12
    mu_pred = mu_lcdm(z, Omega_m, M_offset)
    return np.sum(((mu_obs - mu_pred) / sigma)**2)

# ===================================================================
# FIT 1: QFD NEW (1 free parameter: M)
# ===================================================================
print(f"\n{'='*72}")
print("[1] QFD NEW: σ_nf ∝ √E, D_L ∝ (1+z)^(2/3)")
print("=" * 72)

result_new = minimize_scalar(
    chi2_qfd, bounds=(-5, 5), method='bounded',
    args=(mu_qfd_new, z, mu_obs, sigma)
)
M_new = result_new.x
chi2_new = result_new.fun
dof_new = N - 1  # 1 free parameter
chi2_dof_new = chi2_new / dof_new
rms_new = np.sqrt(np.mean(((mu_obs - mu_qfd_new(z, M_new)) / sigma)**2 * sigma**2))

print(f"  Best-fit M = {M_new:.4f}")
print(f"  χ² = {chi2_new:.2f}")
print(f"  dof = {dof_new}")
print(f"  χ²/dof = {chi2_dof_new:.4f}")
print(f"  RMS residual = {rms_new:.4f} mag")

# ===================================================================
# FIT 2: QFD OLD (1 free parameter: M)
# ===================================================================
print(f"\n{'='*72}")
print("[2] QFD OLD: σ ∝ E², D_L ∝ √(1+z)")
print("=" * 72)

result_old = minimize_scalar(
    chi2_qfd, bounds=(-5, 5), method='bounded',
    args=(mu_qfd_old, z, mu_obs, sigma)
)
M_old = result_old.x
chi2_old = result_old.fun
dof_old = N - 1
chi2_dof_old = chi2_old / dof_old
rms_old = np.sqrt(np.mean(((mu_obs - mu_qfd_old(z, M_old)) / sigma)**2 * sigma**2))

print(f"  Best-fit M = {M_old:.4f}")
print(f"  χ² = {chi2_old:.2f}")
print(f"  dof = {dof_old}")
print(f"  χ²/dof = {chi2_dof_old:.4f}")
print(f"  RMS residual = {rms_old:.4f} mag")

# ===================================================================
# FIT 3: ΛCDM (2 free parameters: Ω_m, M)
# ===================================================================
print(f"\n{'='*72}")
print("[3] ΛCDM: Flat Friedmann (Ω_m + Ω_Λ = 1)")
print("=" * 72)

result_lcdm = minimize(
    chi2_lcdm, x0=[0.3, -19.4], method='Nelder-Mead',
    args=(z, mu_obs, sigma),
    options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-8}
)
Omega_m_best, M_lcdm = result_lcdm.x
chi2_lc = result_lcdm.fun
dof_lc = N - 2  # 2 free parameters
chi2_dof_lc = chi2_lc / dof_lc
rms_lc = np.sqrt(np.mean(((mu_obs - mu_lcdm(z, Omega_m_best, M_lcdm)) / sigma)**2 * sigma**2))

print(f"  Best-fit Ω_m = {Omega_m_best:.4f}")
print(f"  Best-fit M = {M_lcdm:.4f}")
print(f"  χ² = {chi2_lc:.2f}")
print(f"  dof = {dof_lc}")
print(f"  χ²/dof = {chi2_dof_lc:.4f}")
print(f"  RMS residual = {rms_lc:.4f} mag")

# ===================================================================
# COMPARISON TABLE
# ===================================================================
print(f"\n{'='*72}")
print("COMPARISON TABLE")
print("=" * 72)

print(f"\n  {'Model':<25s}  {'χ²/dof':>8s}  {'Free params':>12s}  {'RMS [mag]':>10s}")
print(f"  {'-'*25}  {'-'*8}  {'-'*12}  {'-'*10}")
print(f"  {'QFD NEW (σ∝√E, q=2/3)':<25s}  {chi2_dof_new:8.4f}  {'1 (M)':>12s}  {rms_new:10.4f}")
print(f"  {'QFD OLD (σ∝E², q=1/2)':<25s}  {chi2_dof_old:8.4f}  {'1 (M)':>12s}  {rms_old:10.4f}")
print(f"  {'ΛCDM (flat)':<25s}  {chi2_dof_lc:8.4f}  {'2 (Ω_m, M)':>12s}  {rms_lc:10.4f}")

# ===================================================================
# RESIDUAL ANALYSIS
# ===================================================================
print(f"\n{'='*72}")
print("RESIDUAL ANALYSIS (QFD NEW)")
print("=" * 72)

residuals = mu_obs - mu_qfd_new(z, M_new)
z_bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5]
print(f"\n  {'z range':<12s}  {'N':>5s}  {'Mean res':>10s}  {'σ_res':>8s}  {'χ²/N':>8s}")
print(f"  {'-'*12}  {'-'*5}  {'-'*10}  {'-'*8}  {'-'*8}")

for i in range(len(z_bins) - 1):
    mask_bin = (z >= z_bins[i]) & (z < z_bins[i+1])
    if np.sum(mask_bin) > 0:
        res_bin = residuals[mask_bin]
        sig_bin = sigma[mask_bin]
        chi2_bin = np.sum((res_bin / sig_bin)**2)
        n_bin = np.sum(mask_bin)
        print(f"  [{z_bins[i]:.1f}, {z_bins[i+1]:.1f})  {n_bin:5d}  {np.mean(res_bin):10.4f}  {np.std(res_bin):8.4f}  {chi2_bin/n_bin:8.4f}")

# ===================================================================
# DIAGNOSTIC: Shape comparison (normalized)
# ===================================================================
print(f"\n{'='*72}")
print("SHAPE DIAGNOSTIC: Where do the models diverge?")
print("=" * 72)

z_diag = np.array([0.1, 0.3, 0.5, 0.7, 1.0, 1.2])
mu_new_diag = mu_qfd_new(z_diag, M_new)
mu_old_diag = mu_qfd_old(z_diag, M_old)
mu_lc_diag = mu_lcdm(z_diag, Omega_m_best, M_lcdm)

print(f"\n  {'z':>5s}  {'μ_NEW':>8s}  {'μ_OLD':>8s}  {'μ_ΛCDM':>8s}  {'NEW-OLD':>8s}  {'NEW-ΛCDM':>9s}")
print(f"  {'-'*5}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*9}")
for i, zi in enumerate(z_diag):
    print(f"  {zi:5.1f}  {mu_new_diag[i]:8.3f}  {mu_old_diag[i]:8.3f}  {mu_lc_diag[i]:8.3f}  "
          f"{mu_new_diag[i]-mu_old_diag[i]:8.3f}  {mu_new_diag[i]-mu_lc_diag[i]:9.3f}")

# ===================================================================
# SUMMARY
# ===================================================================
print(f"\n{'='*72}")
print("SUMMARY")
print("=" * 72)

# Determine winner
if chi2_dof_new < chi2_dof_old:
    improvement = (chi2_dof_old - chi2_dof_new) / chi2_dof_old * 100
    print(f"\n  QFD NEW beats QFD OLD by {improvement:.1f}% in χ²/dof")
else:
    print(f"\n  QFD OLD beats QFD NEW (unexpected!)")

if chi2_dof_new < chi2_dof_lc:
    print(f"  QFD NEW beats ΛCDM ({chi2_dof_new:.4f} vs {chi2_dof_lc:.4f}) with FEWER free parameters")
elif abs(chi2_dof_new - chi2_dof_lc) < 0.01:
    print(f"  QFD NEW matches ΛCDM ({chi2_dof_new:.4f} vs {chi2_dof_lc:.4f}) with FEWER free parameters")
else:
    print(f"  ΛCDM beats QFD NEW ({chi2_dof_lc:.4f} vs {chi2_dof_new:.4f})")

print(f"""
  edits42-F claimed: χ²/dof = 0.9546 (QFD NEW)
  This script finds: χ²/dof = {chi2_dof_new:.4f} (QFD NEW)

  Note: The exact χ²/dof depends on:
    - Quality cuts applied (σ threshold, z range)
    - Which DES-SN5YR compilation is used
    - Whether K_J is fixed or allowed to float

  The SHAPE of the Hubble diagram is the physics test.
  The absolute offset (M, K_J) is a calibration.
""")

# ===================================================================
# FIT 4: QFD NEW with floating η (2 free params — diagnostic only)
# ===================================================================
print(f"{'='*72}")
print("[4] DIAGNOSTIC: QFD NEW with floating η (2 free params)")
print("=" * 72)

def mu_qfd_new_free_eta(z, M_offset, eta_val):
    """QFD NEW with η as a free parameter."""
    D_L = (C_KM_S / K_J) * np.log(1 + z) * (1 + z)**(2.0/3.0)
    tau = eta_val * (1 - (1 + z)**(-0.5))
    return 5.0 * np.log10(D_L) + 25.0 + M_offset + K_MAG * tau

def chi2_qfd_free_eta(params, z, mu_obs, sigma):
    """χ² for QFD NEW with 2 free parameters (M, η)."""
    M_offset, eta_val = params
    mu_pred = mu_qfd_new_free_eta(z, M_offset, eta_val)
    return np.sum(((mu_obs - mu_pred) / sigma)**2)

result_eta = minimize(
    chi2_qfd_free_eta, x0=[0.6, 1.0], method='Nelder-Mead',
    args=(z, mu_obs, sigma),
    options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-8}
)
M_eta, eta_best = result_eta.x
chi2_eta = result_eta.fun
dof_eta = N - 2
chi2_dof_eta = chi2_eta / dof_eta

print(f"  Best-fit M = {M_eta:.4f}")
print(f"  Best-fit η = {eta_best:.4f}  (derived: π²/β² = {ETA:.4f}, ratio: {eta_best/ETA:.3f})")
print(f"  χ² = {chi2_eta:.2f}")
print(f"  dof = {dof_eta}")
print(f"  χ²/dof = {chi2_dof_eta:.4f}")
print(f"  η needs to be {eta_best/ETA:.1f}× the derived value to match data!")

# ===================================================================
# FIT 5: QFD with free exponent q (2 free params — diagnostic only)
# ===================================================================
print(f"\n{'='*72}")
print("[5] DIAGNOSTIC: QFD with floating q exponent (2 free params)")
print("=" * 72)

def mu_qfd_free_q(z, M_offset, q_val):
    """QFD with free luminosity distance exponent."""
    D_L = (C_KM_S / K_J) * np.log(1 + z) * (1 + z)**q_val
    tau = ETA * (1 - (1 + z)**(-0.5))
    return 5.0 * np.log10(D_L) + 25.0 + M_offset + K_MAG * tau

def chi2_qfd_free_q(params, z, mu_obs, sigma):
    M_offset, q_val = params
    mu_pred = mu_qfd_free_q(z, M_offset, q_val)
    return np.sum(((mu_obs - mu_pred) / sigma)**2)

result_q = minimize(
    chi2_qfd_free_q, x0=[0.5, 0.667], method='Nelder-Mead',
    args=(z, mu_obs, sigma),
    options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-8}
)
M_q, q_best = result_q.x
chi2_q = result_q.fun
dof_q = N - 2
chi2_dof_q = chi2_q / dof_q

print(f"  Best-fit M = {M_q:.4f}")
print(f"  Best-fit q = {q_best:.4f}  (derived: 2/3 = {2/3:.4f})")
print(f"  χ² = {chi2_q:.2f}")
print(f"  dof = {dof_q}")
print(f"  χ²/dof = {chi2_dof_q:.4f}")

# ===================================================================
# FIT 6: QFD with free q AND free τ exponent (3 free params — diagnostic)
# ===================================================================
print(f"\n{'='*72}")
print("[6] DIAGNOSTIC: QFD with free q + free τ power (3 free params)")
print("=" * 72)

def mu_qfd_free_all(z, M_offset, q_val, tau_power):
    """QFD with free D_L exponent and free τ power law."""
    D_L = (C_KM_S / K_J) * np.log(1 + z) * (1 + z)**q_val
    tau = ETA * (1 - (1 + z)**(-tau_power))
    return 5.0 * np.log10(D_L) + 25.0 + M_offset + K_MAG * tau

def chi2_qfd_free_all(params, z, mu_obs, sigma):
    M_offset, q_val, tau_power = params
    if tau_power < 0.01 or tau_power > 5.0:
        return 1e12
    mu_pred = mu_qfd_free_all(z, M_offset, q_val, tau_power)
    return np.sum(((mu_obs - mu_pred) / sigma)**2)

result_all = minimize(
    chi2_qfd_free_all, x0=[0.5, 0.5, 0.5], method='Nelder-Mead',
    args=(z, mu_obs, sigma),
    options={'maxiter': 20000, 'xatol': 1e-8, 'fatol': 1e-8}
)
M_all, q_all, tp_all = result_all.x
chi2_all = result_all.fun
dof_all = N - 3
chi2_dof_all = chi2_all / dof_all

print(f"  Best-fit M = {M_all:.4f}")
print(f"  Best-fit q = {q_all:.4f}  (NEW claims 2/3 = {2/3:.4f}; OLD was 1/2 = 0.5)")
print(f"  Best-fit τ power = {tp_all:.4f}  (NEW claims 1/2; OLD was 2)")
print(f"  χ² = {chi2_all:.2f}")
print(f"  dof = {dof_all}")
print(f"  χ²/dof = {chi2_dof_all:.4f}")

print(f"\n{'='*72}")
print("FINAL VERDICT")
print("=" * 72)
print(f"""
  The data tells us the best (q, τ_power) pair within the QFD framework.

  Claimed: q = 2/3, τ_power = 1/2  → χ²/dof = {chi2_dof_new:.4f}
  Data prefers: q = {q_all:.4f}, τ_power = {tp_all:.4f}  → χ²/dof = {chi2_dof_all:.4f}

  {'='*60}
  The χ²/dof = 0.9546 claim is NOT confirmed.
  The QFD NEW framework as specified gives χ²/dof = {chi2_dof_new:.4f}.
  {'='*60}
""")
