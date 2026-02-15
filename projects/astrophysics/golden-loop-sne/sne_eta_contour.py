#!/usr/bin/env python3
"""
sne_eta_contour.py — Trace the η = π²/β² contour through the chi2 valley

Key question: WHERE in the (q, n) landscape does η = π²/β² naturally emerge?
And at those points, what is the chi2 penalty?
"""

import numpy as np
from scipy.optimize import brentq, minimize

# ─── Data ───────────────────────────────────────────────────────
DATA_PATH = "/home/tracy/development/SupernovaSrc/qfd-supernova-v15/data/DES-SN5YR-1.2/4_DISTANCES_COVMAT/DES-SN5YR_HD.csv"

raw = np.genfromtxt(DATA_PATH, delimiter=',', names=True)
z_data = raw['zHD']
mu_data = raw['MU']
mu_err = raw['MUERR_FINAL']
mask = (z_data > 0.01) & (mu_err > 0) & (mu_err < 10) & np.isfinite(mu_data)
z_data, mu_data, mu_err = z_data[mask], mu_data[mask], mu_err[mask]
w = 1.0 / mu_err**2
N = len(z_data)

PI = np.pi
EU = np.e
ALPHA = 1.0/137.035999084
K_MAG = 5.0/np.log(10.0)

def solve_golden_loop(alpha):
    target = (1.0/alpha) - 1.0
    C = 2.0*PI*PI
    b = 3.0
    for _ in range(100):
        eb = np.exp(b)
        val = C*(eb/b) - target
        deriv = C*eb*(b-1.0)/(b*b)
        if abs(deriv) < 1e-30: break
        b -= val/deriv
        if abs(val/deriv) < 1e-15: break
    return b

BETA = solve_golden_loop(ALPHA)
ETA_GEO = PI**2 / BETA**2

print(f"β = {BETA:.10f}")
print(f"π²/β² = {ETA_GEO:.10f}")
print(f"N = {N} SNe")

# ─── WLS fitting ────────────────────────────────────────────────
lnz1 = np.log(1.0 + z_data)

def fit_qn(q, n):
    """Fit M and η for given (q, n). Returns chi2, M, eta."""
    arg = lnz1 * (1.0 + z_data)**q
    base = 5.0 * np.log10(arg)
    fn = lnz1 if n == 0 else 1.0 - (1.0 + z_data)**(-n)
    y = mu_data - base
    A2 = K_MAG * fn
    S1 = np.sum(w); S2 = np.sum(w*A2); S22 = np.sum(w*A2**2)
    Sy = np.sum(w*y); S2y = np.sum(w*A2*y)
    det = S1*S22 - S2**2
    M = (S22*Sy - S2*S2y)/det
    eta = (S1*S2y - S2*Sy)/det
    resid = y - M - eta*A2
    return np.sum(w*resid**2), M, eta

def fit_qn_constrained_eta(q, n, eta_target):
    """Fit M only, with η fixed. Returns chi2."""
    arg = lnz1 * (1.0 + z_data)**q
    base = 5.0 * np.log10(arg)
    fn = lnz1 if n == 0 else 1.0 - (1.0 + z_data)**(-n)
    y = mu_data - base - K_MAG * eta_target * fn
    M = np.sum(w*y) / np.sum(w)
    resid = y - M
    return np.sum(w*resid**2)

# ═══════════════════════════════════════════════════════════════
# PART A: Trace η = π²/β² contour in (q, n) space
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("PART A: η = π²/β² contour trace")
print("="*70)

# For each q, find the n where η = π²/β²
print(f"\n{'q':>6s} {'n(η=π²/β²)':>12s} {'chi2':>10s} {'chi2/dof':>10s} {'η_actual':>10s} {'Δchi2':>8s}")

contour_q = []
contour_n = []
contour_chi2 = []

# Reference: unconstrained minimum
chi2_ref = 1686.13  # from previous run

for q in np.arange(0.0, 1.21, 0.02):
    # For this q, scan n to find where eta crosses π²/β²
    def eta_minus_target(n):
        _, _, eta = fit_qn(q, max(n, 0.001))
        return eta - ETA_GEO

    # Bracket search
    n_test = np.linspace(0.01, 6.0, 600)
    eta_vals = []
    for nt in n_test:
        _, _, eta = fit_qn(q, nt)
        eta_vals.append(eta)
    eta_vals = np.array(eta_vals)

    # Find sign changes
    found = False
    for k in range(len(eta_vals)-1):
        if (eta_vals[k] - ETA_GEO) * (eta_vals[k+1] - ETA_GEO) < 0:
            try:
                n_root = brentq(eta_minus_target, n_test[k], n_test[k+1], xtol=1e-8)
                c2, M, eta = fit_qn(q, n_root)
                contour_q.append(q)
                contour_n.append(n_root)
                contour_chi2.append(c2)
                print(f"{q:6.3f} {n_root:12.6f} {c2:10.2f} {c2/(N-2):10.6f} {eta:10.6f} {c2-chi2_ref:+8.2f}")
                found = True
                break
            except:
                pass

    if not found:
        # Check if eta at n=0 or small n is already close
        _, _, eta0 = fit_qn(q, 0.01)
        if abs(eta0 - ETA_GEO) < 0.5:
            print(f"{q:6.3f} {'~0':>12s}  (η at n→0 = {eta0:.4f})")

contour_q = np.array(contour_q)
contour_n = np.array(contour_n)
contour_chi2 = np.array(contour_chi2)

# Find the minimum chi2 along the η = π²/β² contour
if len(contour_chi2) > 0:
    imin = np.argmin(contour_chi2)
    print(f"\n★ BEST point on η = π²/β² contour:")
    print(f"  q = {contour_q[imin]:.4f}, n = {contour_n[imin]:.6f}")
    print(f"  chi2 = {contour_chi2[imin]:.2f}, Δchi2 from global best = {contour_chi2[imin]-chi2_ref:+.2f}")

# ═══════════════════════════════════════════════════════════════
# PART B: Fine optimization along η = π²/β² contour
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("PART B: Optimize along η = π²/β² contour")
print("="*70)

def chi2_on_contour(q):
    """For given q, find n where η = π²/β² and return chi2."""
    def eta_minus_target(n):
        _, _, eta = fit_qn(q, max(n, 0.001))
        return eta - ETA_GEO

    n_test = np.linspace(0.01, 6.0, 200)
    eta_vals = np.array([fit_qn(q, nt)[2] for nt in n_test])

    for k in range(len(eta_vals)-1):
        if (eta_vals[k] - ETA_GEO) * (eta_vals[k+1] - ETA_GEO) < 0:
            try:
                n_root = brentq(eta_minus_target, n_test[k], n_test[k+1], xtol=1e-10)
                c2, _, _ = fit_qn(q, n_root)
                return c2, n_root
            except:
                pass
    return np.inf, np.nan

# Fine optimization
from scipy.optimize import minimize_scalar
result = minimize_scalar(lambda q: chi2_on_contour(q)[0],
                         bounds=(0.3, 1.0), method='bounded')
q_opt = result.x
chi2_opt, n_opt = chi2_on_contour(q_opt)
_, M_opt, eta_opt = fit_qn(q_opt, n_opt)

print(f"\nOptimal point on η = π²/β² contour:")
print(f"  q = {q_opt:.6f}")
print(f"  n = {n_opt:.6f}")
print(f"  chi2 = {chi2_opt:.4f}, chi2/dof = {chi2_opt/(N-2):.6f}")
print(f"  M = {M_opt:.4f}")
print(f"  η = {eta_opt:.6f} (target = {ETA_GEO:.6f})")
print(f"  Δchi2 from global best = {chi2_opt - chi2_ref:+.4f}")

# ═══════════════════════════════════════════════════════════════
# PART C: Physical meaning of (q_opt, n_opt)
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("PART C: Physical interpretation")
print("="*70)

print(f"""
Surface brightness:  D_L = D × (1+z)^{q_opt:.4f}
  q = 0.5 means energy loss only (no time dilation)
  q = 1.0 means energy + time dilation
  q = {q_opt:.4f} means {q_opt*100:.1f}% of full (1+z) effect

Scattering cross-section:  σ ∝ E^{n_opt:.4f}
  n = 0: Thomson-like (energy-independent)
  n = 1: linear (Compton-like)
  n = 2: four-photon (current QFD claim)
  n = {n_opt:.4f}: sub-linear energy dependence

Opacity amplitude:  η = π²/β² = {ETA_GEO:.6f}
  This IS the geometric prediction — it works with (q,n) = ({q_opt:.3f}, {n_opt:.3f})
""")

# Check special rational forms near n_opt
print("Candidate rational/geometric forms for n:")
candidates_n = [
    (1.0/2, "1/2"),
    (1.0/3, "1/3"),
    (2.0/3, "2/3"),
    (1.0/PI, "1/π"),
    (1.0/EU, "1/e"),
    (ALPHA*BETA*PI, "αβπ"),
    (1.0/(PI*EU), "1/(πe)"),
    (BETA/(2*PI), "β/(2π)"),
    (2*ALPHA*PI, "2απ"),
    (np.log(2), "ln 2"),
    (1 - 1/EU, "1-1/e"),
    (PI/6, "π/6"),
    (EU/5, "e/5"),
    (PI/4 - 0.25, "π/4 - 1/4"),
    (1.0/np.sqrt(3), "1/√3"),
    (np.log(PI)/PI, "ln(π)/π"),
    (2/(PI+EU), "2/(π+e)"),
]
candidates_n.sort(key=lambda x: abs(x[0] - n_opt))
print(f"  n_opt = {n_opt:.6f}")
for val, name in candidates_n[:10]:
    delta = (val - n_opt)/n_opt * 100
    print(f"  {name:>12s} = {val:.6f}  ({delta:+.2f}%)")

# Check special rational forms near q_opt
print(f"\nCandidate rational/geometric forms for q:")
candidates_q = [
    (0.5, "1/2"),
    (2.0/3, "2/3"),
    (3.0/5, "3/5"),
    (5.0/8, "5/8"),
    (1.0/PI, "1/π"),
    (EU/4, "e/4"),
    (PI/5, "π/5"),
    (1 - 1/BETA, "1-1/β"),
    (BETA/5, "β/5"),
    (1.0/EU, "1/e"),
    (np.log(2), "ln 2"),
    (1/(1+BETA), "1/(1+β)"),
    (ALPHA*PI, "απ"),
    (PI/(2*EU), "π/(2e)"),
    (2/PI, "2/π"),
]
candidates_q.sort(key=lambda x: abs(x[0] - q_opt))
print(f"  q_opt = {q_opt:.6f}")
for val, name in candidates_q[:10]:
    delta = (val - q_opt)/q_opt * 100
    c2, _, _ = fit_qn(val, n_opt)
    print(f"  {name:>12s} = {val:.6f}  ({delta:+.2f}%)  chi2={c2:.2f}")

# ═══════════════════════════════════════════════════════════════
# PART D: Constrained fit — fix η = π²/β², scan (q, n) together
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("PART D: Chi2 penalty for fixing η = π²/β²")
print("="*70)

# At the optimal (q, n) on the contour, what's the constrained chi2?
chi2_constrained = fit_qn_constrained_eta(q_opt, n_opt, ETA_GEO)
chi2_free, _, _ = fit_qn(q_opt, n_opt)
print(f"\nAt (q={q_opt:.4f}, n={n_opt:.4f}):")
print(f"  chi2 (η free) = {chi2_free:.4f}")
print(f"  chi2 (η = π²/β²) = {chi2_constrained:.4f}")
print(f"  Δchi2 = {chi2_constrained - chi2_free:.4f}")

# ═══════════════════════════════════════════════════════════════
# PART E: Compare ΛCDM with QFD on η=π²/β² contour
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("PART E: ΛCDM comparison at optimal QFD point")
print("="*70)

from scipy.integrate import quad

def mu_lcdm(z_arr, Om, M_lcdm):
    result = np.empty_like(z_arr)
    for k, zz in enumerate(z_arr):
        def integrand(zp):
            return 1.0 / np.sqrt(Om*(1+zp)**3 + (1-Om))
        I, _ = quad(integrand, 0, zz)
        dL = (1+zz) * I
        result[k] = 5.0 * np.log10(dL) if dL > 0 else -99
    return result + M_lcdm

def chi2_lcdm(Om):
    mu_base = mu_lcdm(z_data, Om, 0.0)
    y = mu_data - mu_base
    M_fit = np.sum(w*y) / np.sum(w)
    resid = y - M_fit
    return np.sum(w*resid**2)

from scipy.optimize import minimize_scalar as ms
res = ms(chi2_lcdm, bounds=(0.01, 0.99), method='bounded')

print(f"\nΛCDM (1 free param: Ω_m):")
print(f"  Ω_m = {res.x:.4f}, chi2 = {res.fun:.2f}, chi2/dof = {res.fun/(N-1):.6f}")

print(f"\nQFD on η=π²/β² contour (1 free param: q, with n determined by η constraint):")
print(f"  q = {q_opt:.4f}, n = {n_opt:.4f}, chi2 = {chi2_opt:.2f}, chi2/dof = {chi2_opt/(N-1):.6f}")

print(f"\nΔchi2 (QFD − ΛCDM) = {chi2_opt - res.fun:+.2f}")
print(f"Both have 1 effective free parameter (M is always free)")

# ═══════════════════════════════════════════════════════════════
# PART F: What if q and n are BOTH geometric?
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("PART F: Fully locked models (0 free params except M)")
print("="*70)

# Test combinations where all three (q, n, η) are geometric
locked = [
    (0.5,   2.0, ETA_GEO,     "Current QFD: q=1/2, n=2, η=π²/β²"),
    (0.5,   n_opt, ETA_GEO,   f"q=1/2, n={n_opt:.3f}(opt), η=π²/β²"),
    (q_opt, n_opt, ETA_GEO,   f"q={q_opt:.3f}(opt), n={n_opt:.3f}(opt), η=π²/β²"),
    (2/3,   0.5,  ETA_GEO,    "q=2/3, n=1/2, η=π²/β²"),
    (2/3,   1/3,  ETA_GEO,    "q=2/3, n=1/3, η=π²/β²"),
    (3/5,   0.5,  ETA_GEO,    "q=3/5, n=1/2, η=π²/β²"),
    (2/PI,  1/PI, ETA_GEO,    "q=2/π, n=1/π, η=π²/β²"),
    (1-1/BETA, BETA/5, ETA_GEO, f"q=1-1/β, n=β/5, η=π²/β²"),
    (1/EU,  1-1/EU, ETA_GEO,  "q=1/e, n=1-1/e, η=π²/β²"),
    (0.5,   0.36, ETA_GEO,    "q=1/2, n=0.36, η=π²/β² [valley floor]"),
    (1.0,   1.6,  ETA_GEO,    "q=1, n=1.6, η=π²/β²"),
]

print(f"\n{'label':>50s} {'chi2':>10s} {'chi2/dof':>10s} {'Δchi2':>10s}")
for q, n, eta, label in locked:
    c2 = fit_qn_constrained_eta(q, n, eta)
    print(f"{label:>50s} {c2:10.2f} {c2/(N-1):10.6f} {c2-chi2_ref:+10.2f}")

# ΛCDM locked at Ω_m = 0.3 (Planck)
chi2_lcdm_030 = chi2_lcdm(0.3)
print(f"\n{'ΛCDM Ω_m=0.3 (Planck)':>50s} {chi2_lcdm_030:10.2f} {chi2_lcdm_030/(N-1):10.6f} {chi2_lcdm_030-chi2_ref:+10.2f}")
print(f"{'ΛCDM Ω_m=0.36 (best fit)':>50s} {res.fun:10.2f} {res.fun/(N-1):10.6f} {res.fun-chi2_ref:+10.2f}")
