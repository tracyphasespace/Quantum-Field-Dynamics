#!/usr/bin/env python3
"""
compare_qfd_lcdm_mu_z.py

Rigorous model comparison between QFD (drag-only) and ΛCDM on μ-z data using
information criteria with analytically marginalized global offsets.
"""

import numpy as np
import pandas as pd
import argparse
import json
import os
from scipy.integrate import cumulative_trapezoid
from math import log10

C_KM_S = 2.99792458e5  # Speed of light in km/s

def load_union21(path):
    """Load Union2.1 data with error handling."""
    try:
        arr = np.loadtxt(path)
        if arr.ndim == 1:
            arr = arr[None,:]

        if arr.shape[1] == 2:
            z, mu = arr[:,0], arr[:,1]
            dmu = np.full_like(z, 0.15, float)  # Default error
            print(f"Loaded {len(z)} points with default errors (0.15 mag)")
        else:
            z, mu, dmu = arr[:,0], arr[:,1], arr[:,2]
            dmu = np.where(np.isfinite(dmu) & (dmu > 0), dmu, 0.15)
            print(f"Loaded {len(z)} points with provided errors")

        return z.astype(float), mu.astype(float), dmu.astype(float)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        raise

def lcdm_mu_shape(z, Om=0.3, H0=70.0):
    """ΛCDM distance modulus shape (without global offset)."""
    Ol = 1.0 - Om
    z = np.atleast_1d(z)

    # Integration grid
    z_max = max(1.5, float(z.max()))
    zg = np.linspace(0, z_max, 4000)

    # Hubble parameter
    E = np.sqrt(Om * (1 + zg)**3 + Ol)
    H_fac = H0 / C_KM_S  # 1/Mpc

    # Comoving distance
    Dc = cumulative_trapezoid(1.0 / (H_fac * E), zg, initial=0.0)  # Mpc

    # Luminosity distance
    Dl = (1 + z) * np.interp(z, zg, Dc)

    return 5 * np.log10(np.maximum(Dl, 1e-12))

def qfd_mu_shape(z):
    """QFD drag-only distance modulus shape (without global offset)."""
    z = np.atleast_1d(z)
    # QFD: D_true = ln(1+z)/α0, Dl ∝ (1+z) ln(1+z)
    shape = 5 * np.log10(np.maximum((1 + z) * np.log1p(z), 1e-30))
    return shape

def minimize_offset(mu_obs, dmu, mu_shape):
    """Analytically minimize global offset for given shape."""
    # Best offset a* such that μ_model = mu_shape + a
    w = 1.0 / np.maximum(dmu, 1e-3)**2
    a_star = np.sum(w * (mu_obs - mu_shape)) / np.sum(w)
    chi2 = np.sum(w * (mu_obs - (mu_shape + a_star))**2)
    return a_star, chi2

def scan_omega_m(z, mu, dmu, grid=None):
    """Grid search for best Ωm in ΛCDM."""
    if grid is None:
        grid = np.linspace(0.05, 0.5, 451)

    best = (None, 1e99, None)

    for Om in grid:
        shape = lcdm_mu_shape(z, Om=Om)
        _, chi2 = minimize_offset(mu, dmu, shape)

        if chi2 < best[1]:
            best = (Om, chi2, shape)

    return best  # (Om_best, chi2_best, shape_best)

def main():
    parser = argparse.ArgumentParser(
        description="Compare QFD vs ΛCDM on μ-z data using information criteria"
    )
    parser.add_argument("--data", default="union2.1_data.txt",
                       help="Union2.1 data file")
    parser.add_argument("--out", default="compare_qfd_lcdm_results.json",
                       help="Output JSON file")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")

    args = parser.parse_args()

    print("=== QFD vs ΛCDM Model Comparison on μ-z Data ===")

    # Load data
    z, mu, dmu = load_union21(args.data)
    N = len(z)
    print(f"Dataset: {N} supernovae, z = {z.min():.3f} - {z.max():.3f}")

    # QFD Analysis: 0 shape parameters (offset marginalized analytically)
    print("\n--- QFD (Drag-Only) Analysis ---")
    shape_qfd = qfd_mu_shape(z)
    offset_qfd, chi2_qfd = minimize_offset(mu, dmu, shape_qfd)
    k_qfd = 0  # No free shape parameters
    AIC_qfd = chi2_qfd + 2 * k_qfd
    BIC_qfd = chi2_qfd + k_qfd * np.log(N)

    print(f"χ² = {chi2_qfd:.2f}")
    print(f"χ²/ν = {chi2_qfd / (N - 1):.2f}")  # N-1 DOF (offset marginalized)
    print(f"Global offset = {offset_qfd:.3f} mag")
    print(f"AIC = {AIC_qfd:.2f}")
    print(f"BIC = {BIC_qfd:.2f}")

    # ΛCDM Analysis: 1 shape parameter (Ωm) + offset marginalized
    print("\n--- ΛCDM Analysis ---")
    Om_best, chi2_lcdm, shape_lcdm = scan_omega_m(z, mu, dmu)
    offset_lcdm, _ = minimize_offset(mu, dmu, shape_lcdm)
    k_lcdm = 1  # One shape parameter (Ωm)
    AIC_lcdm = chi2_lcdm + 2 * k_lcdm
    BIC_lcdm = chi2_lcdm + k_lcdm * np.log(N)

    print(f"Best Ωm = {Om_best:.3f}")
    print(f"χ² = {chi2_lcdm:.2f}")
    print(f"χ²/ν = {chi2_lcdm / (N - 2):.2f}")  # N-2 DOF (Ωm + offset)
    print(f"Global offset = {offset_lcdm:.3f} mag")
    print(f"AIC = {AIC_lcdm:.2f}")
    print(f"BIC = {BIC_lcdm:.2f}")

    # Model Comparison
    print("\n--- Model Comparison ---")
    delta_chi2 = chi2_qfd - chi2_lcdm
    delta_AIC = AIC_qfd - AIC_lcdm
    delta_BIC = BIC_qfd - BIC_lcdm

    print(f"Δχ² (QFD - ΛCDM) = {delta_chi2:.2f}")
    print(f"ΔAIC (QFD - ΛCDM) = {delta_AIC:.2f}")
    print(f"ΔBIC (QFD - ΛCDM) = {delta_BIC:.2f}")

    # Interpretation
    print("\n--- Interpretation ---")
    if abs(delta_AIC) <= 2:
        print("AIC: Models are statistically equivalent")
    elif delta_AIC < -10:
        print("AIC: Strong evidence favoring QFD")
    elif delta_AIC > 10:
        print("AIC: Strong evidence favoring ΛCDM")
    elif delta_AIC < 0:
        print("AIC: Weak evidence favoring QFD")
    else:
        print("AIC: Weak evidence favoring ΛCDM")

    if abs(delta_BIC) <= 2:
        print("BIC: Models are statistically equivalent")
    elif delta_BIC < -10:
        print("BIC: Strong evidence favoring QFD")
    elif delta_BIC > 10:
        print("BIC: Strong evidence favoring ΛCDM")
    elif delta_BIC < 0:
        print("BIC: Weak evidence favoring QFD")
    else:
        print("BIC: Weak evidence favoring ΛCDM")

    # Save results
    results = {
        "dataset": {
            "N": int(N),
            "z_range": [float(z.min()), float(z.max())],
            "file": args.data
        },
        "QFD": {
            "k_params": k_qfd,
            "chi2": float(chi2_qfd),
            "chi2_nu": float(chi2_qfd / (N - 1)),
            "AIC": float(AIC_qfd),
            "BIC": float(BIC_qfd),
            "global_offset": float(offset_qfd)
        },
        "LCDM": {
            "k_params": k_lcdm,
            "chi2": float(chi2_lcdm),
            "chi2_nu": float(chi2_lcdm / (N - 2)),
            "AIC": float(AIC_lcdm),
            "BIC": float(BIC_lcdm),
            "Omega_m": float(Om_best),
            "global_offset": float(offset_lcdm)
        },
        "comparison": {
            "delta_chi2": float(delta_chi2),
            "delta_AIC": float(delta_AIC),
            "delta_BIC": float(delta_BIC)
        }
    }

    # Output results
    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else ".", exist_ok=True)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {args.out}")

if __name__ == "__main__":
    main()