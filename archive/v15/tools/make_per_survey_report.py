#!/usr/bin/env python3
"""
Per-Survey Residual Report Generator for V15 QFD Pipeline

Generates comprehensive statistics and diagnostics for Stage 3 results:
- Global summaries (RMS, MAD, skew, trend vs z)
- Per-survey breakdowns
- Per-survey×band tables
- Z-binned statistics for plotting
- Optional ΛCDM comparison

Usage:
    python scripts/make_per_survey_report.py stage3_results.csv
    python scripts/make_per_survey_report.py stage3_results.csv --out-dir custom/path
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import argparse

K = 2.5 / np.log(10.0)  # Conversion: Δμ = -K * Δα


def robust_stats(x):
    """
    Compute robust statistics for a residual distribution

    Returns:
        dict with N, mean, std, MAD, skew, frac_gt3sigma
    """
    x = np.asarray(x)
    x = x[np.isfinite(x)]

    if x.size == 0:
        return dict(N=0, mean=np.nan, std=np.nan, mad=np.nan,
                   skew=np.nan, frac_gt3sigma=np.nan)

    mean = float(np.mean(x))
    std = float(np.std(x, ddof=1))
    # Gaussian-consistent MAD (1.4826 factor converts median absolute deviation to σ)
    mad = float(1.4826 * np.median(np.abs(x - np.median(x))))
    skew = float(stats.skew(x, bias=False)) if x.size > 2 else np.nan
    frac_gt3 = float(np.mean(np.abs(x - mean) > (3 * std))) if std > 0 else 0.0

    return dict(N=x.size, mean=mean, std=std, mad=mad,
               skew=skew, frac_gt3sigma=frac_gt3)


def bin_stats(z, y, edges):
    """
    Compute statistics in z-bins

    Args:
        z: redshift array
        y: residual array
        edges: bin edges

    Returns:
        DataFrame with z_lo, z_hi, N, mean, std per bin
    """
    recs = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (z >= a) & (z < b)
        if m.sum() == 0:
            recs.append(dict(z_lo=a, z_hi=b, N=0, mean=np.nan, std=np.nan))
        else:
            recs.append(dict(
                z_lo=a, z_hi=b,
                N=int(m.sum()),
                mean=float(np.mean(y[m])),
                std=float(np.std(y[m], ddof=1))
            ))
    return pd.DataFrame(recs)


def per_survey_band(df, resid_col):
    """
    Compute statistics per survey×band combination

    Returns:
        DataFrame with survey, band, N, mean, std, MAD, skew, frac_gt3sigma, chi2_per_obs_median
    """
    rows = []
    for (survey, band), g in df.groupby(["survey", "band"], dropna=False):
        rs = robust_stats(g[resid_col])
        rs.update(dict(
            survey=survey,
            band=band,
            chi2_per_obs_med=float(np.median(g["chi2_per_obs"]))
        ))
        rows.append(rs)

    out = pd.DataFrame(rows).sort_values(["survey", "band"])
    return out[["survey", "band", "N", "mean", "std", "mad",
               "skew", "frac_gt3sigma", "chi2_per_obs_med"]]


def slope_vs_z(df, resid_col, z_bins=np.arange(0.0, 1.55, 0.05)):
    """
    Compute trend of residuals vs redshift (unweighted OLS on bin means)

    A flat trend (slope ≈ 0) indicates no systematic offset with redshift

    Returns:
        dict with slope and slope_err
    """
    b = bin_stats(df["z"].values, df[resid_col].values, z_bins)
    b = b.dropna()

    if b["N"].sum() == 0:
        return dict(slope=np.nan, slope_err=np.nan)

    x = 0.5 * (b["z_lo"].values + b["z_hi"].values)
    y = b["mean"].values

    A = np.vstack([x, np.ones_like(x)]).T
    beta, resid, _, _ = np.linalg.lstsq(A, y, rcond=None)
    slope = float(beta[0])

    # Simple standard error
    s2 = resid[0] / max(len(y) - 2, 1) if len(resid) > 0 else np.nan
    cov = s2 * np.linalg.inv(A.T @ A)
    slope_err = float(np.sqrt(cov[0, 0])) if np.isfinite(s2) else np.nan

    return dict(slope=slope, slope_err=slope_err)


def main(input_csv, out_dir="results/v15_production/reports"):
    """
    Generate comprehensive per-survey residual reports

    Args:
        input_csv: Path to Stage 3 results CSV
        out_dir: Output directory for reports
    """
    print(f"Reading data from: {input_csv}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} observations")

    # Convert α-space residuals to μ-space for readability
    df["residual_mu"] = -K * df["residual_alpha"]

    # ========================================
    # 1. Global summaries
    # ========================================
    print("\nComputing global statistics...")
    global_alpha = robust_stats(df["residual_alpha"])
    global_mu = robust_stats(df["residual_mu"])
    trend_alpha = slope_vs_z(df, "residual_alpha")
    trend_mu = slope_vs_z(df, "residual_mu")

    global_summary = pd.DataFrame([{
        **{f"alpha_{k}": v for k, v in global_alpha.items()},
        **{f"mu_{k}": v for k, v in global_mu.items()},
        **{f"trend_alpha_{k}": v for k, v in trend_alpha.items()},
        **{f"trend_mu_{k}": v for k, v in trend_mu.items()},
    }])
    global_summary.to_csv(out / "summary_overall.csv", index=False)
    print(f"  RMS(α) = {global_alpha['std']:.4f}")
    print(f"  RMS(μ) = {global_mu['std']:.4f}")
    print(f"  Trend α vs z: {trend_alpha['slope']:.6f} ± {trend_alpha['slope_err']:.6f}")

    # ========================================
    # 2. Per-survey tables
    # ========================================
    print("\nComputing per-survey statistics...")
    by_survey = df.groupby("survey").apply(
        lambda g: pd.Series(robust_stats(g["residual_alpha"]))
    ).reset_index()
    by_survey.to_csv(out / "summary_by_survey_alpha.csv", index=False)
    print(f"  Surveys: {', '.join(df['survey'].unique())}")

    # ========================================
    # 3. Per-survey×band tables
    # ========================================
    print("\nComputing per-survey×band statistics...")
    by_survey_band = per_survey_band(df, "residual_alpha")
    by_survey_band.to_csv(out / "summary_by_survey_band_alpha.csv", index=False)

    # ========================================
    # 4. Optional ΛCDM comparison
    # ========================================
    if "residual_lcdm" in df.columns:
        print("\nComputing ΛCDM comparison...")
        lcdm_survey = df.groupby("survey").apply(
            lambda g: pd.Series(robust_stats(g["residual_lcdm"]))
        ).reset_index()
        lcdm_survey.to_csv(out / "summary_by_survey_lcdm.csv", index=False)

        global_lcdm = robust_stats(df["residual_lcdm"])
        print(f"  ΛCDM RMS: {global_lcdm['std']:.4f}")

    # ========================================
    # 5. Z-binned residuals per survey
    # ========================================
    print("\nComputing z-binned statistics...")
    zb = []
    bins = np.arange(0.0, 1.55, 0.05)
    for s, g in df.groupby("survey"):
        bs = bin_stats(g["z"].values, g["residual_alpha"].values, bins)
        bs["survey"] = s
        zb.append(bs)
    pd.concat(zb, ignore_index=True).to_csv(out / "zbin_alpha_by_survey.csv", index=False)

    # ========================================
    # Summary
    # ========================================
    print("\n" + "="*60)
    print("REPORT GENERATION COMPLETE")
    print("="*60)
    print("\nGenerated files:")
    for p in sorted(out.glob("*.csv")):
        print(f"  {p}")
    print(f"\nOutput directory: {out.absolute()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate per-survey residual reports for V15 Stage 3 results"
    )
    parser.add_argument(
        "input_csv",
        help="Path to Stage 3 results CSV"
    )
    parser.add_argument(
        "--out-dir", "-o",
        default="results/v15_production/reports",
        help="Output directory for reports (default: results/v15_production/reports)"
    )

    args = parser.parse_args()
    main(args.input_csv, args.out_dir)
