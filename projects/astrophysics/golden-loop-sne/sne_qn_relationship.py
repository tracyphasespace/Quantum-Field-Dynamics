#!/usr/bin/env python3
"""
sne_qn_relationship.py — Investigate whether q and n are linked

Observation: The optimal (q=2/3, n=1/2) on the η=π²/β² contour satisfies
  q = 1/(1+n) = 1/(1+1/2) = 2/3

Is this a coincidence or physics? Test whether q = 1/(1+n) holds along
the η = π²/β² contour.

Also test: what physical constraint links q and n?
"""

import numpy as np
from scipy.optimize import brentq

DATA_PATH = "/home/tracy/development/SupernovaSrc/qfd-supernova-v15/data/DES-SN5YR-1.2/4_DISTANCES_COVMAT/DES-SN5YR_HD.csv"
raw = np.genfromtxt(DATA_PATH, delimiter=',', names=True)
z_data = raw['zHD']; mu_data = raw['MU']; mu_err = raw['MUERR_FINAL']
mask = (z_data > 0.01) & (mu_err > 0) & (mu_err < 10) & np.isfinite(mu_data)
z_data, mu_data, mu_err = z_data[mask], mu_data[mask], mu_err[mask]
w = 1.0/mu_err**2; N = len(z_data)
PI = np.pi; EU = np.e; K_MAG = 5.0/np.log(10.0)
ALPHA = 1.0/137.035999084

def solve_gl(a):
    t = (1.0/a)-1.0; C = 2*PI*PI; b = 3.0
    for _ in range(100):
        eb = np.exp(b); val = C*(eb/b)-t; d = C*eb*(b-1)/(b*b)
        if abs(d)<1e-30: break
        b -= val/d;
        if abs(val/d)<1e-15: break
    return b

BETA = solve_gl(ALPHA)
ETA_GEO = PI**2/BETA**2
lnz1 = np.log(1.0 + z_data)

def fit_qn(q, n):
    arg = lnz1 * (1.0+z_data)**q
    base = 5.0*np.log10(arg)
    fn = lnz1 if n == 0 else 1.0 - (1.0+z_data)**(-n)
    y = mu_data - base
    A2 = K_MAG * fn
    S1=np.sum(w); S2=np.sum(w*A2); S22=np.sum(w*A2**2)
    Sy=np.sum(w*y); S2y=np.sum(w*A2*y)
    det = S1*S22 - S2**2
    M = (S22*Sy - S2*S2y)/det
    eta = (S1*S2y - S2*Sy)/det
    resid = y - M - eta*A2
    return np.sum(w*resid**2), M, eta

# ══════════════════════════════════════════════════════════════
# TEST 1: Does q = 1/(1+n) hold along η = π²/β² contour?
# ══════════════════════════════════════════════════════════════
print("="*70)
print("TEST 1: q = 1/(1+n) along η = π²/β² contour")
print("="*70)

# Trace the η = π²/β² contour (from previous run)
print(f"\n{'q':>6s} {'n(η=geo)':>10s} {'1/(1+n)':>10s} {'q_pred':>10s} {'Δq/q%':>10s} {'chi2':>10s}")

for q in np.arange(0.10, 1.10, 0.05):
    def eta_minus_target(n):
        _, _, eta = fit_qn(q, max(n, 0.001))
        return eta - ETA_GEO
    n_test = np.linspace(0.01, 3.0, 300)
    eta_vals = np.array([fit_qn(q, nt)[2] for nt in n_test])
    for k in range(len(eta_vals)-1):
        if (eta_vals[k]-ETA_GEO)*(eta_vals[k+1]-ETA_GEO) < 0:
            try:
                n_root = brentq(eta_minus_target, n_test[k], n_test[k+1], xtol=1e-10)
                q_pred = 1.0/(1.0+n_root)
                delta = (q - q_pred)/q_pred * 100
                c2, _, _ = fit_qn(q, n_root)
                print(f"{q:6.3f} {n_root:10.6f} {q_pred:10.6f} {q_pred:10.6f} {delta:+10.2f}% {c2:10.2f}")
                break
            except:
                pass

# ══════════════════════════════════════════════════════════════
# TEST 2: Alternative relationships q(n)
# ══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("TEST 2: Alternative q-n relationships")
print("="*70)

# At each point on the η contour, compute various q(n) candidates
print(f"\n{'q':>6s} {'n':>8s} {'1/(1+n)':>8s} {'(2-n)/2':>8s} {'1-n/2':>8s} {'n/(2n+1)':>8s} {'2/(2+n)':>8s}")
for q in np.arange(0.10, 1.10, 0.10):
    def eta_mt(n):
        _, _, eta = fit_qn(q, max(n, 0.001))
        return eta - ETA_GEO
    n_test = np.linspace(0.01, 3.0, 300)
    eta_vals = np.array([fit_qn(q, nt)[2] for nt in n_test])
    for k in range(len(eta_vals)-1):
        if (eta_vals[k]-ETA_GEO)*(eta_vals[k+1]-ETA_GEO) < 0:
            try:
                n_root = brentq(eta_mt, n_test[k], n_test[k+1], xtol=1e-10)
                f1 = 1.0/(1+n_root)
                f2 = (2-n_root)/2
                f3 = 1-n_root/2
                f4 = n_root/(2*n_root+1)
                f5 = 2.0/(2+n_root)
                print(f"{q:6.3f} {n_root:8.4f} {f1:8.4f} {f2:8.4f} {f3:8.4f} {f4:8.4f} {f5:8.4f}")
                break
            except:
                pass

# ══════════════════════════════════════════════════════════════
# TEST 3: Constrained fit — impose q = 1/(1+n) and η = π²/β²
# ══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("TEST 3: Constrained fit with q = 1/(1+n), η = π²/β²")
print("="*70)

from scipy.optimize import minimize_scalar

def chi2_constrained(n):
    """Fix q = 1/(1+n) and η = π²/β², fit only M."""
    q = 1.0/(1.0 + n)
    arg = lnz1 * (1.0+z_data)**q
    fn = 1.0 - (1.0+z_data)**(-n) if n > 0 else lnz1
    base = 5.0*np.log10(arg)
    y = mu_data - base - K_MAG*ETA_GEO*fn
    M = np.sum(w*y)/np.sum(w)
    resid = y - M
    return np.sum(w*resid**2)

# Scan n
print(f"\n{'n':>6s} {'q=1/(1+n)':>10s} {'chi2':>10s} {'chi2/dof':>10s}")
for n in np.arange(0.1, 2.01, 0.1):
    q = 1.0/(1+n)
    c2 = chi2_constrained(n)
    print(f"{n:6.2f} {q:10.4f} {c2:10.2f} {c2/(N-1):10.6f}")

# Optimize
res = minimize_scalar(chi2_constrained, bounds=(0.01, 3.0), method='bounded')
n_opt = res.x
q_opt = 1.0/(1+n_opt)
print(f"\nOptimal: n = {n_opt:.6f}, q = {q_opt:.6f}")
print(f"  chi2 = {res.fun:.4f}, chi2/dof = {res.fun/(N-1):.6f}")
print(f"  Compare: (2/3, 1/2) gives chi2 = {chi2_constrained(0.5):.4f}")

# ══════════════════════════════════════════════════════════════
# TEST 4: What does q + n/2 = constant mean?
# ══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("TEST 4: Invariants along η = π²/β² contour")
print("="*70)

print(f"\n{'q':>6s} {'n':>8s} {'q+n':>8s} {'q+n/2':>8s} {'q*n':>8s} {'q/(1-q)':>8s} {'n/(1-n)':>8s} {'2q+n':>8s}")
for q in np.arange(0.10, 1.10, 0.05):
    def eta_mt(n):
        _, _, eta = fit_qn(q, max(n, 0.001))
        return eta - ETA_GEO
    n_test = np.linspace(0.01, 3.0, 300)
    eta_vals = np.array([fit_qn(q, nt)[2] for nt in n_test])
    for k in range(len(eta_vals)-1):
        if (eta_vals[k]-ETA_GEO)*(eta_vals[k+1]-ETA_GEO) < 0:
            try:
                n_root = brentq(eta_mt, n_test[k], n_test[k+1], xtol=1e-10)
                qn = q+n_root
                qn2 = q+n_root/2
                qxn = q*n_root
                qr = q/(1-q) if q < 0.999 else 999
                nr = n_root/(1-n_root) if n_root < 0.999 else 999
                tqn = 2*q+n_root
                print(f"{q:6.3f} {n_root:8.4f} {qn:8.4f} {qn2:8.4f} {qxn:8.4f} {qr:8.4f} {nr:8.4f} {tqn:8.4f}")
                break
            except:
                pass

# ══════════════════════════════════════════════════════════════
# TEST 5: Time dilation prediction
# ══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("TEST 5: Effective time dilation from (q=2/3, n=1/2)")
print("="*70)

# In ΛCDM: dt_obs/dt_emit = (1+z)
# In QFD: D_L = D × (1+z)^q, and scattering adds dimming
# The (1+z)^q comes from flux: F = L/(4πD_L²)
# D_L² = D² × (1+z)^{2q}
# Flux correction = (1+z)^{2q}
# Energy correction = (1+z)^1 always
# So "arrival rate" correction = (1+z)^{2q-1}
# With q=2/3: arrival rate ~ (1+z)^{1/3}

print(f"""
Flux decomposition:
  F = L / (4π D²) × (1+z)^{{-2q}}

  Energy factor: (1+z)^{{-1}} (always present)
  Remaining: (1+z)^{{-(2q-1)}} = (1+z)^{{-1/3}} for q=2/3

  This means photon arrival rate is reduced by (1+z)^{{1/3}}
  — NOT the full (1+z) of expanding cosmology

  So "effective time dilation" = (1+z)^{{1/3}} from D_L factor
  + chromatic broadening from σ ∝ E^{{1/2}} (additional)

  Total observed stretch = (1+z)^{{1/3}} × broadening(z, n=1/2)

  In standard cosmology: stretch = (1+z)^1 exactly
  In QFD (2/3, 1/2): stretch = (1+z)^{{1/3}} × F_broad(z)
""")

# What broadening function F_broad gives total (1+z)?
# (1+z)^{1/3} × F_broad = (1+z) → F_broad = (1+z)^{2/3}
print("Required broadening to match observed (1+z) stretch:")
print("  F_broad(z) = (1+z)^{2/3}")
print(f"  At z=0.5: F = {1.5**(2/3):.4f}")
print(f"  At z=1.0: F = {2.0**(2/3):.4f}")
print(f"  At z=2.0: F = {3.0**(2/3):.4f}")

# Can scattering with σ ∝ E^{1/2} produce this?
# Scattering opacity: τ(z) = η × [1 - (1+z)^{-n}] with n=1/2
# The broadening from chromatic scattering is proportional to the
# differential opacity between band edges
print("\nChromatic broadening from σ ∝ E^{1/2}:")
for z in [0.1, 0.5, 1.0, 2.0]:
    # Differential opacity between blue and red edges of a band
    # Say band spans E_blue = 1.2×E_center, E_red = 0.8×E_center
    # σ ∝ E^{1/2} → σ_blue/σ_red = (1.2/0.8)^{1/2} = 1.225
    # With n=2: σ_blue/σ_red = (1.2/0.8)^2 = 2.25 (much larger)
    tau_fn = 1.0 - (1+z)**(-0.5)
    tau_fn2 = 1.0 - (1+z)**(-2.0)
    broad_needed = (1+z)**(2/3)
    print(f"  z={z:.1f}: f_{{1/2}}(z)={tau_fn:.4f}, f_2(z)={tau_fn2:.4f}, "
          f"broadening needed={(1+z)**(2/3):.4f}")

# ══════════════════════════════════════════════════════════════
# TEST 6: q = 2/(2+n) relationship
# ══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("TEST 6: Does q = 2/(2+n) hold? (gives q=4/5 at n=1/2...)")
print("="*70)

# Check various 2-parameter relationships
# At (2/3, 1/2):
#   q = 1/(1+n): 1/1.5 = 2/3 ✓
#   q = 2n/(2n+1): 1/2 = 1/2 ✗
#   q = (1+n)/2: 3/4 ✗
#   q = 2/(2+n): 4/5 ✗
#   n = 3(1-q): 3×1/3 = 1 ✗
#   n = 1-q: 1/3 ✗
#   n = 2(1-q): 2/3 ✗
#   n·(1+n) = q·(2-q)? 1/2×3/2=3/4, 2/3×4/3=8/9 ✗
#   q + n = 7/6: hmm
#   q - n = 1/6 ✓! (2/3 - 1/2 = 1/6)
#   q·(1+n) = 1: 2/3 × 3/2 = 1 ✓!!!

print("\nRelationship q·(1+n) = 1:")
print(f"  At (2/3, 1/2): q×(1+n) = {2/3 * 1.5:.6f}")
print(f"  This is EXACTLY 1 !")
print(f"  Equivalently: q = 1/(1+n), i.e., surface brightness = 1/(1 + energy_power)")
print()
print("Physical interpretation:")
print("  If σ ∝ E^n, then the effective surface brightness goes as (1+z)^{1/(1+n)}")
print("  This links the scattering physics to the distance-luminosity relation")
print()

# Check if this holds at 2 decimal places along the contour
print("Testing q·(1+n) along η = π²/β² contour:")
print(f"{'q':>6s} {'n':>8s} {'q·(1+n)':>10s} {'Δ from 1':>10s}")
for q in np.arange(0.10, 1.10, 0.05):
    def eta_mt(n):
        _, _, eta = fit_qn(q, max(n, 0.001))
        return eta - ETA_GEO
    n_test = np.linspace(0.01, 3.0, 300)
    eta_vals = np.array([fit_qn(q, nt)[2] for nt in n_test])
    for k in range(len(eta_vals)-1):
        if (eta_vals[k]-ETA_GEO)*(eta_vals[k+1]-ETA_GEO) < 0:
            try:
                n_root = brentq(eta_mt, n_test[k], n_test[k+1], xtol=1e-10)
                prod = q*(1+n_root)
                print(f"{q:6.3f} {n_root:8.4f} {prod:10.6f} {prod-1:+10.6f}")
                break
            except:
                pass
