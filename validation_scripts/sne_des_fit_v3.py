#!/usr/bin/env python3
"""
Independent verification v3: χ²/dof for NEW QFD framework vs DES-SN5YR data.

Uses the SAME dataset as golden_loop_sne.py (DES-SN5YR_HD.csv) and tests
BOTH K_MAG conventions to resolve the factor-of-2 discrepancy.

Key question: does the χ²/dof = 0.9546 claim depend on K_MAG = 5/ln(10)?

Created: 2026-02-15
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

# Two competing K_MAG conventions:
K_MAG_STANDARD = 2.5 / np.log(10.0)  # ≈ 1.0857 (standard: Δm = -2.5 log₁₀(e^{-τ}))
K_MAG_GOLDEN   = 5.0 / np.log(10.0)  # ≈ 2.1715 (golden_loop_sne.py "book convention")

print("=" * 72)
print("DES-SN5YR VERIFICATION v3: K_MAG CONVENTION TEST")
print("Using SAME dataset as golden_loop_sne.py (DES-SN5YR_HD.csv)")
print("=" * 72)
print(f"\n  Constants:")
print(f"    β = {BETA:.9f}")
print(f"    η = π²/β² = {ETA:.6f}")
print(f"    K_J = ξ×β^(3/2) = {K_J:.4f} km/s/Mpc")
print(f"    K_MAG_standard = 2.5/ln10 = {K_MAG_STANDARD:.4f}")
print(f"    K_MAG_golden   = 5.0/ln10 = {K_MAG_GOLDEN:.4f}")

# ===================================================================
# Load DES-SN5YR_HD.csv (same file as golden_loop_sne.py)
# ===================================================================
data_path = '/home/tracy/development/SupernovaSrc/qfd-supernova-v15/data/DES-SN5YR-1.2/4_DISTANCES_COVMAT/DES-SN5YR_HD.csv'
print(f"\n  Loading: DES-SN5YR_HD.csv")

# CSV has columns: CID,IDSURVEY,zCMB,zHD,zHEL,MU,MUERR_FINAL
raw = np.genfromtxt(data_path, delimiter=',', skip_header=1, usecols=(3, 5, 6))
z_all = raw[:, 0]    # zHD
mu_all = raw[:, 1]   # MU
sigma_all = raw[:, 2] # MUERR_FINAL

print(f"  Raw data: {len(z_all)} SNe")
print(f"    z range: [{z_all.min():.4f}, {z_all.max():.4f}]")
print(f"    μ range: [{mu_all.min():.2f}, {mu_all.max():.2f}]")

# Quality cuts (same as golden_loop_sne.py: z > 0.01, mu_err > 0, mu_err < 10)
mask = (z_all > 0.01) & (sigma_all > 0) & (sigma_all < 10)
z = z_all[mask]
mu_obs = mu_all[mask]
sigma = sigma_all[mask]
N = len(z)
print(f"    After quality cuts: {N} SNe")

# ===================================================================
# Model functions
# ===================================================================

def mu_qfd(z, M, K_MAG, q=2.0/3.0, n=0.5, eta=ETA):
    """QFD distance modulus: μ = 5 log₁₀(D_L) + 25 + M + K_MAG × η × f(z)."""
    D_L = (C_KM_S / K_J) * np.log(1 + z) * (1 + z)**q
    tau_shape = 1 - (1 + z)**(-n)
    return 5.0 * np.log10(D_L) + 25.0 + M + K_MAG * eta * tau_shape

def mu_lcdm(z, Omega_m, M):
    """Flat ΛCDM distance modulus."""
    Omega_L = 1.0 - Omega_m
    H0 = 70.0  # Absorbed into M
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
    return 5.0 * np.log10(D_L) + 25.0 + M

# ===================================================================
# Fit functions
# ===================================================================

def fit_M_only(K_MAG, q=2.0/3.0, n=0.5, eta=ETA):
    """Fit M only (1 free param), η fixed."""
    def chi2(M):
        pred = mu_qfd(z, M, K_MAG, q=q, n=n, eta=eta)
        return np.sum(((mu_obs - pred) / sigma)**2)
    result = minimize_scalar(chi2, bounds=(-5, 55), method='bounded')
    return result.x, result.fun

def fit_M_and_eta(K_MAG, q=2.0/3.0, n=0.5):
    """Fit M and η jointly (2 free params) — analytic WLS solution.

    Exactly replicates golden_loop_sne.py fit_eta_and_M().
    """
    D_L = (C_KM_S / K_J) * np.log(1 + z) * (1 + z)**q
    mu_base = 5.0 * np.log10(D_L) + 25.0

    # Design matrix: μ_obs = μ_base + M × 1 + (K_MAG × η) × f(z)
    y = mu_obs - mu_base
    A1 = np.ones(N)                          # coefficient of M
    A2 = 1 - (1 + z)**(-n)                   # coefficient of K_MAG × η
    w = 1.0 / sigma**2

    # Weighted normal equations
    S11 = np.sum(w * A1 * A1)
    S12 = np.sum(w * A1 * A2)
    S22 = np.sum(w * A2 * A2)
    Sy1 = np.sum(w * y * A1)
    Sy2 = np.sum(w * y * A2)

    det = S11 * S22 - S12**2
    p1 = (S22 * Sy1 - S12 * Sy2) / det   # M
    p2 = (S11 * Sy2 - S12 * Sy1) / det   # K_MAG × η

    M_fit = p1
    eta_fit = p2 / K_MAG

    # Compute chi2
    mu_pred = mu_base + M_fit + p2 * A2
    chi2 = np.sum(((mu_obs - mu_pred) / sigma)**2)

    return M_fit, eta_fit, chi2

# ===================================================================
# TEST A: Fix η = π²/β², fit M only (1 free param)
# ===================================================================
print(f"\n{'='*72}")
print("TEST A: η FIXED at π²/β², M only free")
print("=" * 72)

M_std, chi2_std = fit_M_only(K_MAG_STANDARD)
M_gld, chi2_gld = fit_M_only(K_MAG_GOLDEN)
dof = N - 1

print(f"\n  {'Convention':<20s}  {'K_MAG':>8s}  {'M_fit':>8s}  {'χ²':>10s}  {'χ²/dof':>8s}")
print(f"  {'-'*20}  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*8}")
print(f"  {'Standard (2.5/ln10)':<20s}  {K_MAG_STANDARD:8.4f}  {M_std:8.4f}  {chi2_std:10.2f}  {chi2_std/dof:8.4f}")
print(f"  {'Golden (5.0/ln10)':<20s}  {K_MAG_GOLDEN:8.4f}  {M_gld:8.4f}  {chi2_gld:10.2f}  {chi2_gld/dof:8.4f}")

print(f"\n  Target: χ²/dof = 0.9546 (from golden_loop_sne.py)")

# ===================================================================
# TEST B: Fit M AND η jointly (2 free params) — replicating golden_loop_sne.py
# ===================================================================
print(f"\n{'='*72}")
print("TEST B: η AND M both free (2 params, WLS)")
print("=" * 72)

M_std2, eta_std2, chi2_std2 = fit_M_and_eta(K_MAG_STANDARD)
M_gld2, eta_gld2, chi2_gld2 = fit_M_and_eta(K_MAG_GOLDEN)
dof2 = N - 2

print(f"\n  With K_MAG = 2.5/ln10 (standard):")
print(f"    M_fit = {M_std2:.4f}")
print(f"    η_fit = {eta_std2:.4f}  (derived: {ETA:.4f}, ratio: {eta_std2/ETA:.4f})")
print(f"    χ² = {chi2_std2:.2f}, χ²/dof = {chi2_std2/dof2:.4f}")
print(f"    Effective scattering: K_MAG × η = {K_MAG_STANDARD * eta_std2:.4f}")

print(f"\n  With K_MAG = 5.0/ln10 (golden_loop_sne.py):")
print(f"    M_fit = {M_gld2:.4f}")
print(f"    η_fit = {eta_gld2:.4f}  (derived: {ETA:.4f}, ratio: {eta_gld2/ETA:.4f})")
print(f"    χ² = {chi2_gld2:.2f}, χ²/dof = {chi2_gld2/dof2:.4f}")
print(f"    Effective scattering: K_MAG × η = {K_MAG_GOLDEN * eta_gld2:.4f}")

# Key check: same physics?
print(f"\n  CRITICAL CHECK: K_MAG × η_fit should be IDENTICAL for both:")
print(f"    Standard: {K_MAG_STANDARD * eta_std2:.6f}")
print(f"    Golden:   {K_MAG_GOLDEN * eta_gld2:.6f}")
print(f"    Agreement: {'YES' if abs(K_MAG_STANDARD * eta_std2 - K_MAG_GOLDEN * eta_gld2) < 1e-6 else 'NO'}")

# ===================================================================
# TEST C: golden_loop_sne.py exact replication — fix η=π²/β² with K_MAG=5/ln10
# ===================================================================
print(f"\n{'='*72}")
print("TEST C: EXACT REPLICATION of golden_loop_sne.py claimed result")
print("Fixing η = π²/β², K_MAG = 5/ln10, q=2/3, n=1/2")
print("=" * 72)

# The golden_loop_sne.py procedure:
# 1. Fit M and η freely with K_MAG=5/ln10
# 2. Check that η_fit ≈ π²/β²
# 3. Then fix η = π²/β² and compute χ² with just M free

# Step 1: Free fit
M_free, eta_free, chi2_free = fit_M_and_eta(K_MAG_GOLDEN)
print(f"\n  Step 1: Free fit (M, η)")
print(f"    M = {M_free:.4f}, η = {eta_free:.6f}")
print(f"    χ² = {chi2_free:.2f}, dof = {N-2}, χ²/dof = {chi2_free/(N-2):.4f}")

# Step 2: Fix η = π²/β²
M_fixed, chi2_fixed = fit_M_only(K_MAG_GOLDEN, eta=ETA)
print(f"\n  Step 2: Fixed η = π²/β² = {ETA:.6f}")
print(f"    M = {M_fixed:.4f}")
print(f"    χ² = {chi2_fixed:.2f}, dof = {N-1}, χ²/dof = {chi2_fixed/(N-1):.4f}")

delta_chi2 = chi2_fixed - chi2_free
print(f"\n  Δχ² from fixing η: {delta_chi2:.2f}")
print(f"    (cost of replacing free η with derived π²/β²)")

# ===================================================================
# TEST D: Compare with ΛCDM
# ===================================================================
print(f"\n{'='*72}")
print("TEST D: ΛCDM comparison (2 free params: Ω_m, M)")
print("=" * 72)

def chi2_lcdm_func(params):
    Omega_m, M = params
    if Omega_m < 0.01 or Omega_m > 0.99:
        return 1e12
    pred = mu_lcdm(z, Omega_m, M)
    return np.sum(((mu_obs - pred) / sigma)**2)

result_lcdm = minimize(chi2_lcdm_func, x0=[0.3, -19.4], method='Nelder-Mead',
                        options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-8})
Omega_m_best, M_lcdm = result_lcdm.x
chi2_lcdm_val = result_lcdm.fun
dof_lcdm = N - 2

print(f"  Ω_m = {Omega_m_best:.4f}")
print(f"  M = {M_lcdm:.4f}")
print(f"  χ² = {chi2_lcdm_val:.2f}, dof = {dof_lcdm}, χ²/dof = {chi2_lcdm_val/dof_lcdm:.4f}")

# ===================================================================
# TEST E: QFD OLD — book formula (K_MAG = 2.5/ln10, q=1/2, n=2)
# ===================================================================
print(f"\n{'='*72}")
print("TEST E: QFD OLD (book v9.0 formula: K_MAG=2.5/ln10, q=1/2, n=2)")
print("=" * 72)

# Book v9.0 line 4324: μ = 5 log₁₀(D_L) + 25 + M + (2.5/ln10) × η × [1-1/(1+z)²]
# Book says: χ²/dof = 1.005, η_fit = 1.0621 ≈ π²/β²

# E-1: Fixed η = π²/β² (0 free physics params, book claim: χ²/dof = 1.005)
M_old_fixed, chi2_old_fixed = fit_M_only(K_MAG_STANDARD, q=0.5, n=2.0, eta=ETA)
print(f"  Fixed η (1 free param, M only):")
print(f"    M = {M_old_fixed:.4f}")
print(f"    χ² = {chi2_old_fixed:.2f}, dof = {N-1}, χ²/dof = {chi2_old_fixed/(N-1):.4f}")
print(f"    Book claims: χ²/dof = 1.005")

# E-2: Free η (2 free params)
M_old_free, eta_old_free, chi2_old_free = fit_M_and_eta(K_MAG_STANDARD, q=0.5, n=2.0)
print(f"\n  Free η (2 free params, M + η):")
print(f"    M = {M_old_free:.4f}")
print(f"    η_fit = {eta_old_free:.4f} (derived: {ETA:.4f}, ratio: {eta_old_free/ETA:.4f})")
print(f"    χ² = {chi2_old_free:.2f}, dof = {N-2}, χ²/dof = {chi2_old_free/(N-2):.4f}")
print(f"    Book claims: η_fit = 1.0621")

# E-3: OLD with K_MAG = 5/ln10 (to show why this convention breaks old model)
M_old_gld, chi2_old_gld = fit_M_only(K_MAG_GOLDEN, q=0.5, n=2.0, eta=ETA)
print(f"\n  For comparison, OLD with K_MAG = 5/ln10:")
print(f"    M = {M_old_gld:.4f}")
print(f"    χ² = {chi2_old_gld:.2f}, dof = {N-1}, χ²/dof = {chi2_old_gld/(N-1):.4f}")
print(f"    (wildly wrong — K_MAG = 5/ln10 was NOT the book convention)")

# ===================================================================
# TEST F: Diagnostic — what q does the data prefer?
# ===================================================================
print(f"\n{'='*72}")
print("TEST F: DIAGNOSTIC — scanning q with n=0.5 fixed, K_MAG = 5/ln10")
print("=" * 72)

print(f"\n  {'q':>6s}  {'M':>8s}  {'χ²':>10s}  {'χ²/dof':>8s}")
print(f"  {'-'*6}  {'-'*8}  {'-'*10}  {'-'*8}")
for q_test in [0.3, 0.4, 0.5, 0.55, 0.6, 2.0/3.0, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]:
    M_t, chi2_t = fit_M_only(K_MAG_GOLDEN, q=q_test, n=0.5, eta=ETA)
    marker = " ← q=2/3" if abs(q_test - 2.0/3.0) < 0.001 else ""
    marker = " ← q=1/2" if abs(q_test - 0.5) < 0.001 else marker
    marker = " ← q=1 (ΛCDM-like)" if abs(q_test - 1.0) < 0.001 else marker
    print(f"  {q_test:6.3f}  {M_t:8.4f}  {chi2_t:10.2f}  {chi2_t/(N-1):8.4f}{marker}")

# ===================================================================
# TEST G: Diagnostic — scanning q with η free (2 params: M, η)
# ===================================================================
print(f"\n{'='*72}")
print("TEST G: DIAGNOSTIC — scanning q with η AND M free, K_MAG = 5/ln10")
print("=" * 72)

print(f"\n  {'q':>6s}  {'M':>8s}  {'η_fit':>8s}  {'η/η_0':>8s}  {'χ²':>10s}  {'χ²/dof':>8s}")
print(f"  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*8}")
for q_test in [0.3, 0.4, 0.5, 2.0/3.0, 0.75, 0.85, 0.9, 1.0]:
    M_t, eta_t, chi2_t = fit_M_and_eta(K_MAG_GOLDEN, q=q_test, n=0.5)
    marker = " ← q=2/3" if abs(q_test - 2.0/3.0) < 0.001 else ""
    print(f"  {q_test:6.3f}  {M_t:8.4f}  {eta_t:8.4f}  {eta_t/ETA:8.4f}  {chi2_t:10.2f}  {chi2_t/(N-2):8.4f}{marker}")

# ===================================================================
# SUMMARY
# ===================================================================
print(f"\n{'='*72}")
print("SUMMARY")
print("=" * 72)

print(f"""
  K_MAG CONVENTION IMPACT:
    With K_MAG = 2.5/ln10 (standard), η fixed: χ²/dof = {chi2_std/(N-1):.4f}
    With K_MAG = 5.0/ln10 (golden),   η fixed: χ²/dof = {chi2_gld/(N-1):.4f}

  GOLDEN_LOOP_SNE.PY REPLICATION:
    Free fit (M, η):    χ² = {chi2_free:.2f}, η = {eta_free:.4f}, χ²/dof = {chi2_free/(N-2):.4f}
    Fixed η = π²/β²:    χ² = {chi2_fixed:.2f}, M = {M_fixed:.4f}, χ²/dof = {chi2_fixed/(N-1):.4f}
    Target:              χ²/dof = 0.9546

  ΛCDM:
    χ² = {chi2_lcdm_val:.2f}, Ω_m = {Omega_m_best:.4f}, χ²/dof = {chi2_lcdm_val/dof_lcdm:.4f}

  PHYSICAL K_MAG QUESTION:
    Standard: Δm = -2.5 log₁₀(e^{{-τ}}) = (2.5/ln10) × τ → K_MAG = 1.0857
    Golden:   K_MAG = 5/ln10 = 2.1715 (factor 2 larger)

    If K_MAG = 5/ln10 is used:
      η_fit = {eta_free:.4f} ≈ π²/β² = {ETA:.4f} (ratio: {eta_free/ETA:.3f})
    If K_MAG = 2.5/ln10 is used:
      η_fit = {eta_std2:.4f} ≈ 2×π²/β² = {2*ETA:.4f} (ratio: {eta_std2/ETA:.3f})

    The data determines K_MAG × η (the product). The K_MAG convention
    determines what number gets called "η".
""")
