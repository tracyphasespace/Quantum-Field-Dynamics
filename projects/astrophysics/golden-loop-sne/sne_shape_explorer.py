#!/usr/bin/env python3
"""
sne_shape_explorer.py — Data-driven shape function exploration

Methodology: Let the data speak. Scan the full (q, n) landscape where:
  D_L = D × (1+z)^q           (surface brightness exponent)
  τ(z) = η × f_n(z)           (scattering opacity)
  f_n(z) = 1 - (1+z)^{-n}    (shape function, σ ∝ E^n)
  D(z) = (c/K_J) × ln(1+z)   (QFD distance)

The distance modulus is:
  μ = 5 log10(D_L/10pc) + K_MAG × η × f_n(z)
    = M + 5 log10[ln(1+z) × (1+z)^q] + K_MAG × η × f_n(z)

where M absorbs c/K_J (degenerate with absolute magnitude).

Fit parameters: M, η  (analytic WLS for each grid point)
Scan parameters: q ∈ [0, 1.5], n ∈ [0, 4]
"""

import numpy as np
import sys, os

# ─── Data loading ───────────────────────────────────────────────
DATA_PATH = "/home/tracy/development/SupernovaSrc/qfd-supernova-v15/data/DES-SN5YR-1.2/4_DISTANCES_COVMAT/DES-SN5YR_HD.csv"

def load_data():
    raw = np.genfromtxt(DATA_PATH, delimiter=',', names=True)
    z   = raw['zHD']
    mu  = raw['MU']
    err = raw['MUERR_FINAL']
    mask = (z > 0.01) & (err > 0) & (err < 10) & np.isfinite(mu)
    return z[mask], mu[mask], err[mask]

z_data, mu_data, mu_err = load_data()
w = 1.0 / mu_err**2
N = len(z_data)
print(f"Loaded {N} SNe, z ∈ [{z_data.min():.4f}, {z_data.max():.4f}]")

K_MAG = 5.0 / np.log(10.0)  # ≈ 2.17147

# ─── Model evaluation ──────────────────────────────────────────
def fit_qn(q, n):
    """For given (q, n), fit M and η analytically via WLS.

    μ_model = M + 5*log10[ln(1+z) × (1+z)^q] + K_MAG * η * f_n(z)

    Let:
      y_i = mu_data_i
      A1_i = 1  (for M)
      A2_i = K_MAG * f_n(z_i)  (for η)
      base_i = 5*log10[ln(1+z_i) × (1+z_i)^q]

    Then: y_i - base_i = M * A1_i + η * A2_i
    """
    lnz1 = np.log(1.0 + z_data)
    arg = lnz1 * (1.0 + z_data)**q

    # Guard against log10(0)
    if np.any(arg <= 0):
        return np.inf, 0.0, 0.0, np.inf

    base = 5.0 * np.log10(arg)

    # Shape function
    if n == 0:
        fn = lnz1  # limit as n→0: ln(1+z)
    else:
        fn = 1.0 - (1.0 + z_data)**(-n)

    y = mu_data - base
    A2 = K_MAG * fn

    # WLS normal equations: [A1, A2]^T W [A1, A2] [M, η]^T = [A1, A2]^T W y
    S1  = np.sum(w)
    S2  = np.sum(w * A2)
    S22 = np.sum(w * A2**2)
    Sy  = np.sum(w * y)
    S2y = np.sum(w * A2 * y)

    det = S1 * S22 - S2**2
    if abs(det) < 1e-30:
        return np.inf, 0.0, 0.0, np.inf

    M_fit   = (S22 * Sy  - S2 * S2y) / det
    eta_fit = (S1  * S2y - S2 * Sy)  / det

    resid = y - M_fit - eta_fit * A2
    chi2 = np.sum(w * resid**2)

    return chi2, M_fit, eta_fit, chi2 / (N - 2)

# ─── Grid scan ──────────────────────────────────────────────────
print("\n" + "="*70)
print("PHASE 1: Full (q, n) landscape scan")
print("="*70)

q_grid = np.linspace(0.0, 1.5, 151)
n_grid = np.linspace(0.0, 4.0, 201)

chi2_map = np.full((len(q_grid), len(n_grid)), np.inf)
eta_map  = np.full((len(q_grid), len(n_grid)), np.nan)
M_map    = np.full((len(q_grid), len(n_grid)), np.nan)

for i, q in enumerate(q_grid):
    for j, n in enumerate(n_grid):
        chi2, M, eta, chi2dof = fit_qn(q, n)
        chi2_map[i, j] = chi2
        eta_map[i, j]  = eta
        M_map[i, j]    = M

# Find global minimum
imin, jmin = np.unravel_index(np.argmin(chi2_map), chi2_map.shape)
q_best, n_best = q_grid[imin], n_grid[jmin]
chi2_best = chi2_map[imin, jmin]
eta_best = eta_map[imin, jmin]
M_best = M_map[imin, jmin]

print(f"\nGlobal minimum: q = {q_best:.4f}, n = {n_best:.4f}")
print(f"  chi2 = {chi2_best:.2f}, chi2/dof = {chi2_best/(N-2):.6f}")
print(f"  M = {M_best:.4f}, eta = {eta_best:.4f}")

# ─── Key reference points ───────────────────────────────────────
print("\n" + "-"*70)
print("Key reference points:")
print("-"*70)

refs = [
    (0.5, 2.0, "Current QFD (√(1+z), σ∝E²)"),
    (0.5, 0.0, "QFD base (√(1+z), no shape)"),
    (1.0, 2.0, "Full TD ((1+z), σ∝E²)"),
    (1.0, 0.0, "Full TD ((1+z), no shape)"),
    (0.0, 2.0, "No SB factor, σ∝E²"),
    (0.0, 0.0, "No SB factor, no shape"),
    (q_best, n_best, "GLOBAL BEST"),
]

PI = np.pi
EU = np.e
ALPHA = 1.0/137.035999084

# Solve golden loop for beta
def solve_golden_loop(alpha):
    target = (1.0/alpha) - 1.0
    C = 2.0 * PI * PI
    b = 3.0
    for _ in range(100):
        eb = np.exp(b)
        val = C * (eb/b) - target
        deriv = C * eb * (b-1.0) / (b*b)
        if abs(deriv) < 1e-30: break
        b -= val/deriv
        if abs(val/deriv) < 1e-15: break
    return b

BETA = solve_golden_loop(ALPHA)
eta_geo = PI**2 / BETA**2

print(f"\nπ²/β² = {eta_geo:.6f} (QFD geometric prediction)")
print(f"β = {BETA:.10f}\n")

fmt = "{:<35s}  q={:.3f}  n={:.3f}  chi2={:8.2f}  chi2/dof={:.4f}  eta={:7.4f}  M={:8.4f}"
for q, n, label in refs:
    c2, M, eta, c2dof = fit_qn(q, n)
    delta_eta = (eta - eta_geo)/eta_geo * 100 if abs(eta_geo) > 0 else 0
    print(fmt.format(label, q, n, c2, c2dof, eta, M))

# ─── Marginal profiles ─────────────────────────────────────────
print("\n" + "="*70)
print("PHASE 2: Marginal profiles")
print("="*70)

# Best n for each q
print("\nBest n(q) profile:")
print(f"{'q':>6s} {'n_best':>8s} {'chi2':>10s} {'chi2/dof':>10s} {'eta':>8s} {'Δeta%':>8s}")
for i, q in enumerate(q_grid[::5]):  # every 5th
    ii = list(q_grid).index(q) if q in q_grid else i*5
    if ii >= len(q_grid): continue
    jb = np.argmin(chi2_map[ii, :])
    nb = n_grid[jb]
    c2 = chi2_map[ii, jb]
    eta = eta_map[ii, jb]
    delta = (eta - eta_geo)/eta_geo * 100
    print(f"{q:6.3f} {nb:8.3f} {c2:10.2f} {c2/(N-2):10.6f} {eta:8.4f} {delta:+8.2f}%")

# Best q for each n
print("\nBest q(n) profile:")
print(f"{'n':>6s} {'q_best':>8s} {'chi2':>10s} {'chi2/dof':>10s} {'eta':>8s} {'Δeta%':>8s}")
for j, n in enumerate(n_grid[::7]):  # every 7th
    jj = list(n_grid).index(n) if n in n_grid else j*7
    if jj >= len(n_grid): continue
    ib = np.argmin(chi2_map[:, jj])
    qb = q_grid[ib]
    c2 = chi2_map[ib, jj]
    eta = eta_map[ib, jj]
    delta = (eta - eta_geo)/eta_geo * 100
    print(f"{n:6.3f} {qb:8.3f} {c2:10.2f} {c2/(N-2):10.6f} {eta:8.4f} {delta:+8.2f}%")

# ─── Δchi2 contours ────────────────────────────────────────────
print("\n" + "="*70)
print("PHASE 3: Δchi2 contours from global minimum")
print("="*70)

dchi2 = chi2_map - chi2_best
levels = [1, 4, 9, 25, 100]

for lev in levels:
    mask = dchi2 <= lev
    if np.any(mask):
        qi = q_grid[np.any(mask, axis=1)]
        ni = n_grid[np.any(mask, axis=0)]
        print(f"Δχ² ≤ {lev:3d}: q ∈ [{qi.min():.3f}, {qi.max():.3f}], "
              f"n ∈ [{ni.min():.3f}, {ni.max():.3f}]")

# ─── PHASE 4: Does η = π²/β² appear anywhere? ──────────────────
print("\n" + "="*70)
print(f"PHASE 4: Where does η = π²/β² = {eta_geo:.6f} appear?")
print("="*70)

eta_target = eta_geo
eta_diff = np.abs(eta_map - eta_target)
# Find closest approach
valid = np.isfinite(eta_diff) & (chi2_map < chi2_best + 100)
if np.any(valid):
    eta_diff_masked = np.where(valid, eta_diff, np.inf)
    ic, jc = np.unravel_index(np.argmin(eta_diff_masked), eta_diff_masked.shape)
    print(f"\nClosest approach to η = π²/β²:")
    print(f"  q = {q_grid[ic]:.4f}, n = {n_grid[jc]:.4f}")
    print(f"  η = {eta_map[ic,jc]:.6f} (target = {eta_target:.6f})")
    print(f"  Δη/η = {(eta_map[ic,jc]-eta_target)/eta_target*100:+.4f}%")
    print(f"  chi2 = {chi2_map[ic,jc]:.2f}, Δchi2 from best = {chi2_map[ic,jc]-chi2_best:.2f}")

# Trace the η = π²/β² contour
print(f"\nη = π²/β² contour (closest match per q slice, Δχ² < 100):")
print(f"{'q':>6s} {'n_match':>8s} {'chi2':>10s} {'Δchi2':>8s} {'eta':>10s}")
for i in range(0, len(q_grid), 10):
    row = eta_diff_masked[i, :]
    if np.all(np.isinf(row)): continue
    jmatch = np.argmin(row)
    if eta_diff_masked[i, jmatch] > 0.1: continue
    c2 = chi2_map[i, jmatch]
    print(f"{q_grid[i]:6.3f} {n_grid[jmatch]:8.3f} {c2:10.2f} {c2-chi2_best:+8.2f} {eta_map[i,jmatch]:10.6f}")

# ─── PHASE 5: Physical interpretation scan ──────────────────────
print("\n" + "="*70)
print("PHASE 5: Fine scan around global minimum")
print("="*70)

q_fine = np.linspace(max(0, q_best-0.2), min(1.5, q_best+0.2), 81)
n_fine = np.linspace(max(0, n_best-0.5), min(4.0, n_best+0.5), 101)

chi2_fine = np.full((len(q_fine), len(n_fine)), np.inf)
eta_fine  = np.full((len(q_fine), len(n_fine)), np.nan)

for i, q in enumerate(q_fine):
    for j, n in enumerate(n_fine):
        c2, M, eta, _ = fit_qn(q, n)
        chi2_fine[i, j] = c2
        eta_fine[i, j] = eta

imin2, jmin2 = np.unravel_index(np.argmin(chi2_fine), chi2_fine.shape)
print(f"\nRefined minimum: q = {q_fine[imin2]:.4f}, n = {n_fine[jmin2]:.4f}")
print(f"  chi2 = {chi2_fine[imin2,jmin2]:.2f}")

# ─── PHASE 6: No-scatter baseline ──────────────────────────────
print("\n" + "="*70)
print("PHASE 6: No-scatter baseline (η=0) vs best model")
print("="*70)

print(f"\n{'q':>6s} {'chi2(η=0)':>12s} {'chi2(best n)':>12s} {'Δchi2':>10s} {'n_best':>8s} {'η':>8s}")
for q in np.arange(0.0, 1.55, 0.1):
    # η = 0 → just base model
    lnz1 = np.log(1.0 + z_data)
    arg = lnz1 * (1.0 + z_data)**q
    base = 5.0 * np.log10(arg)
    y = mu_data - base
    M0 = np.sum(w * y) / np.sum(w)
    resid0 = y - M0
    chi2_0 = np.sum(w * resid0**2)

    # Best n at this q
    idx_q = np.argmin(np.abs(q_grid - q))
    jb = np.argmin(chi2_map[idx_q, :])
    chi2_bn = chi2_map[idx_q, jb]

    print(f"{q:6.2f} {chi2_0:12.2f} {chi2_bn:12.2f} {chi2_0-chi2_bn:10.2f} {n_grid[jb]:8.3f} {eta_map[idx_q,jb]:8.4f}")

# ─── PHASE 7: Check special q values ───────────────────────────
print("\n" + "="*70)
print("PHASE 7: Physics-motivated q values")
print("="*70)

special_q = [
    (0.0,   "No surface brightness"),
    (0.5,   "Current QFD: √(1+z) — energy loss only"),
    (1.0,   "ΛCDM-like: (1+z) — energy + time dilation"),
    (1.0/BETA, f"1/β = {1.0/BETA:.6f}"),
    (ALPHA*BETA, f"αβ = {ALPHA*BETA:.6f}"),
    (1.0/PI, f"1/π = {1.0/PI:.6f}"),
    (0.5 + ALPHA, f"1/2 + α = {0.5+ALPHA:.6f}"),
    (2.0/3.0, "2/3"),
    (EU/(2*PI), f"e/(2π) = {EU/(2*PI):.6f}"),
]

print(f"\n{'q':>8s} {'label':>35s} {'chi2_best_n':>12s} {'n_best':>8s} {'eta':>8s} {'Δchi2':>10s}")
for q, label in special_q:
    if q < 0 or q > 1.5: continue
    # scan n
    best_c2 = np.inf
    best_n = 0
    best_eta = 0
    for n in np.linspace(0, 4, 401):
        c2, M, eta, _ = fit_qn(q, n)
        if c2 < best_c2:
            best_c2 = c2
            best_n = n
            best_eta = eta
    print(f"{q:8.4f} {label:>35s} {best_c2:12.2f} {best_n:8.3f} {best_eta:8.4f} {best_c2-chi2_best:+10.2f}")

# ─── PHASE 8: Degeneracy direction ─────────────────────────────
print("\n" + "="*70)
print("PHASE 8: Degeneracy valley direction")
print("="*70)

# Along the valley floor, how does chi2 change?
# Find valley floor: for each q, the n that minimizes chi2
valley_q = []
valley_n = []
valley_chi2 = []
valley_eta = []

for i, q in enumerate(q_grid):
    jb = np.argmin(chi2_map[i, :])
    valley_q.append(q)
    valley_n.append(n_grid[jb])
    valley_chi2.append(chi2_map[i, jb])
    valley_eta.append(eta_map[i, jb])

valley_q = np.array(valley_q)
valley_n = np.array(valley_n)
valley_chi2 = np.array(valley_chi2)
valley_eta = np.array(valley_eta)

print(f"\nValley floor: n(q) relationship")
print(f"{'q':>6s} {'n':>8s} {'chi2':>10s} {'eta':>8s}")
for i in range(0, len(valley_q), 10):
    print(f"{valley_q[i]:6.3f} {valley_n[i]:8.3f} {valley_chi2[i]:10.2f} {valley_eta[i]:8.4f}")

# Linear fit to valley floor
mask_val = (valley_chi2 < chi2_best + 25) & np.isfinite(valley_n)
if np.sum(mask_val) > 2:
    p = np.polyfit(valley_q[mask_val], valley_n[mask_val], 1)
    print(f"\nValley floor linear fit (Δχ² < 25 region):")
    print(f"  n ≈ {p[0]:.3f} × q + {p[1]:.3f}")
    print(f"  i.e., n increases by {p[0]:.3f} per unit increase in q")

# ─── PHASE 9: Summary ──────────────────────────────────────────
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

print(f"""
Data: {N} DES-SN5YR Type Ia SNe

Global best fit:
  q = {q_best:.4f}  (surface brightness exponent)
  n = {n_best:.4f}  (scattering energy power: σ ∝ E^n)
  chi2 = {chi2_best:.2f}, chi2/dof = {chi2_best/(N-2):.6f}
  η = {eta_best:.4f}
  M = {M_best:.4f}

Reference: π²/β² = {eta_geo:.6f}

Key comparisons:
  Current QFD (q=0.5, n=2): chi2 = {fit_qn(0.5, 2.0)[0]:.2f}  (Δ = {fit_qn(0.5, 2.0)[0]-chi2_best:+.2f})
  Full TD (q=1.0, n=2):     chi2 = {fit_qn(1.0, 2.0)[0]:.2f}  (Δ = {fit_qn(1.0, 2.0)[0]-chi2_best:+.2f})
  Full TD no shape (q=1.0, n=0): chi2 = {fit_qn(1.0, 0.0)[0]:.2f}  (Δ = {fit_qn(1.0, 0.0)[0]-chi2_best:+.2f})
""")

# Compare with pure LCDM
print("ΛCDM comparison (flat, Ω_m free):")
from scipy.integrate import quad
from scipy.optimize import minimize_scalar

def mu_lcdm(z_arr, Om, M_lcdm):
    """Standard FLRW distance modulus."""
    result = np.empty_like(z_arr)
    for k, zz in enumerate(z_arr):
        def integrand(zp):
            return 1.0 / np.sqrt(Om*(1+zp)**3 + (1-Om))
        I, _ = quad(integrand, 0, zz)
        dL = (1+zz) * I  # in units of c/H0
        result[k] = 5.0 * np.log10(dL) if dL > 0 else -99
    return result + M_lcdm

def chi2_lcdm(Om):
    mu_base = mu_lcdm(z_data, Om, 0.0)
    y = mu_data - mu_base
    M_fit = np.sum(w * y) / np.sum(w)
    resid = y - M_fit
    return np.sum(w * resid**2)

res = minimize_scalar(chi2_lcdm, bounds=(0.01, 0.99), method='bounded')
print(f"  Best Ω_m = {res.x:.4f}, chi2 = {res.fun:.2f}, chi2/dof = {res.fun/(N-2):.6f}")
print(f"  QFD best - ΛCDM = {chi2_best - res.fun:+.2f}")
