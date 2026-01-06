#!/usr/bin/env python3
"""
Post-processing analysis for T3b λ_curv sweep.

Computes critical diagnostics:
1. Penalty dominance ratio R(λ) = penalty / chi2_data
2. Separates chi2_data from total loss
3. Tracks U_e, U_mu stability vs λ
4. Identifies elbow for λ selection
"""

import re
import numpy as np
import pandas as pd
import json
from pathlib import Path

def parse_t3b_log(log_path):
    """
    Parse T3b log file and extract per-(λ, β) records.

    Log format:
    λ_curv = 1.00e-09
    ...
    β        χ²_tot       χ²_mass      χ²_g         curv         S_opt      C_g      U_e      U_μ
    1.70     9.545687e-07 9.706909e-09 4.371220e-07 1.692466e+02 0.4781     1178.44  0.0077   0.6506

    Returns DataFrame with columns:
      lam, beta, loss_total, chi2_mass, chi2_g, curv, S_opt, C_g_opt, U_e, U_mu
    """
    records = []
    current_lam = None

    with open(log_path, 'r') as f:
        for line in f:
            # Match λ_curv = X.XXe±XX
            m = re.search(r'λ_curv\s*=\s*([0-9.eE+-]+)', line)
            if m:
                current_lam = float(m.group(1))
                continue

            # Match data line: β followed by 8 floats
            # β        χ²_tot       χ²_mass      χ²_g         curv         S_opt      C_g      U_e      U_μ
            # Note: May have tqdm escape codes like [A before β value
            m = re.search(r'(\d+\.\d+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)', line)
            if m and current_lam is not None:
                beta = float(m.group(1))
                loss_total = float(m.group(2))  # This is chi2_total = chi2_data + penalty
                chi2_mass = float(m.group(3))
                chi2_g = float(m.group(4))
                curv = float(m.group(5))
                S_opt = float(m.group(6))
                C_g_opt = float(m.group(7))
                U_e = float(m.group(8))
                U_mu = float(m.group(9))

                records.append({
                    'lam': current_lam,
                    'beta': beta,
                    'loss_total': loss_total,
                    'chi2_mass': chi2_mass,
                    'chi2_g': chi2_g,
                    'curv': curv,
                    'S_opt': S_opt,
                    'C_g_opt': C_g_opt,
                    'U_e': U_e,
                    'U_mu': U_mu,
                })

    df = pd.DataFrame(records)

    # Compute derived quantities
    df['chi2_data'] = df['chi2_mass'] + df['chi2_g']
    df['penalty'] = df['lam'] * df['curv']
    df['U_ratio'] = df['U_mu'] / df['U_e']

    return df


def compute_lambda_summary(df, bounds=None):
    """
    Compute per-λ summary statistics for elbow identification.

    For each λ:
    - Find best β (minimizes loss_total)
    - Compute penalty dominance ratio R = penalty / chi2_data
    - Compute CV(S), CV(C_g)
    - Compute chi2_data range across β
    - Check bound proximity

    Parameters
    ----------
    df : DataFrame
        Output from parse_t3b_log
    bounds : dict, optional
        {"U_e": (lo, hi), "U_mu": (lo, hi), ...}
        If provided, flags parameters within 1% of bounds
    """
    if bounds is None:
        bounds = {
            'U_e': (1e-4, 0.2),
            'U_mu': (1e-3, 1.0),
        }

    summary = []

    for lam, group in df.groupby('lam'):
        # Best β (minimizes loss_total, which optimizer saw)
        idx_best = group['loss_total'].idxmin()
        best = group.loc[idx_best]

        # Penalty dominance ratio at best β
        R_penalty = best['penalty'] / best['chi2_data'] if best['chi2_data'] > 0 else np.inf

        # CV of profiled parameters
        CV_S = group['S_opt'].std() / group['S_opt'].mean() * 100
        CV_Cg = group['C_g_opt'].std() / group['C_g_opt'].mean() * 100

        # Chi2 data range (NOT loss range)
        chi2_range = group['chi2_data'].max() / group['chi2_data'].min() if group['chi2_data'].min() > 0 else np.inf

        # U stability
        U_e_range = group['U_e'].max() - group['U_e'].min()
        U_mu_range = group['U_mu'].max() - group['U_mu'].min()
        U_ratio_mean = group['U_ratio'].mean()
        U_ratio_std = group['U_ratio'].std()

        # Bound proximity (flag if within 1% of bound)
        U_e_lo, U_e_hi = bounds['U_e']
        U_mu_lo, U_mu_hi = bounds['U_mu']
        tol = 0.01

        bound_hit = False
        if best['U_e'] < U_e_lo * (1 + tol) or best['U_e'] > U_e_hi * (1 - tol):
            bound_hit = True
        if best['U_mu'] < U_mu_lo * (1 + tol) or best['U_mu'] > U_mu_hi * (1 - tol):
            bound_hit = True

        summary.append({
            'lam': lam,
            'beta_star': best['beta'],
            'chi2_data_star': best['chi2_data'],
            'curv_star': best['curv'],
            'penalty_star': best['penalty'],
            'loss_star': best['loss_total'],
            'R_penalty': R_penalty,
            'CV_S': CV_S,
            'CV_Cg': CV_Cg,
            'chi2_data_range': chi2_range,
            'U_e_star': best['U_e'],
            'U_mu_star': best['U_mu'],
            'U_ratio_star': best['U_ratio'],
            'U_e_range': U_e_range,
            'U_mu_range': U_mu_range,
            'U_ratio_mean': U_ratio_mean,
            'U_ratio_std': U_ratio_std,
            'bound_hit': bound_hit,
        })

    return pd.DataFrame(summary).sort_values('lam')


def identify_elbow(summary_df, R_max=10.0, CV_S_target=20.0):
    """
    Identify optimal λ based on elbow criteria:

    1. R_penalty < R_max (penalty not dominating)
    2. CV(S) reduction plateaus (degeneracy removed)
    3. Parameters stay interior (no bound hits)

    Returns recommended λ and diagnostic message.
    """
    # Filter: exclude penalty-dominated regimes
    viable = summary_df[summary_df['R_penalty'] < R_max].copy()

    if len(viable) == 0:
        return None, "All λ values are penalty-dominated (R > 10). Use smallest λ."

    # Filter: exclude bound-hitting
    viable = viable[~viable['bound_hit']].copy()

    if len(viable) == 0:
        return None, "All viable λ hit parameter bounds. Check bounds or model."

    # Find where CV(S) stops improving significantly
    # Use largest λ where CV(S) is still decreasing
    viable = viable.sort_values('lam')
    CV_S_vals = viable['CV_S'].values

    # Look for plateau: derivative of CV vs log(λ) near zero
    if len(viable) >= 3:
        log_lam = np.log10(viable['lam'].values + 1e-12)
        dCV_dlog_lam = np.gradient(CV_S_vals, log_lam)

        # Plateau where |slope| < 5% per decade
        plateau_mask = np.abs(dCV_dlog_lam) < 5.0
        if np.any(plateau_mask):
            # Take smallest λ in plateau region
            idx_elbow = viable.index[plateau_mask][0]
            lam_opt = viable.loc[idx_elbow, 'lam']
            msg = f"Elbow at λ={lam_opt:.1e}: CV plateau, R={viable.loc[idx_elbow, 'R_penalty']:.2f}"
            return lam_opt, msg

    # Fallback: smallest viable λ where CV(S) < target
    below_target = viable[viable['CV_S'] < CV_S_target]
    if len(below_target) > 0:
        idx_elbow = below_target.index[0]
        lam_opt = below_target.loc[idx_elbow, 'lam']
        msg = f"Selected λ={lam_opt:.1e}: CV(S)={below_target.loc[idx_elbow, 'CV_S']:.1f}% < target"
        return lam_opt, msg

    # Last resort: smallest viable λ
    idx_elbow = viable.index[0]
    lam_opt = viable.loc[idx_elbow, 'lam']
    msg = f"Selected λ={lam_opt:.1e} (smallest viable, CV not at target)"
    return lam_opt, msg


def main():
    log_path = Path("results/V22/logs/t3b_curvature_sweep.log")

    print("=" * 80)
    print("T3b λ_curv SWEEP ANALYSIS")
    print("=" * 80)
    print()

    if not log_path.exists():
        print(f"ERROR: Log file not found: {log_path}")
        return

    print(f"Parsing log: {log_path}")
    df = parse_t3b_log(log_path)

    if len(df) == 0:
        print("No data found in log. Check format or wait for run to complete.")
        return

    n_lam = df['lam'].nunique()
    n_beta = df.groupby('lam')['beta'].count().values
    print(f"  Found {n_lam} λ values")
    print(f"  β points per λ: {n_beta}")
    print()

    # Compute summary
    summary = compute_lambda_summary(df)

    print("=" * 80)
    print("PER-λ SUMMARY: PENALTY DOMINANCE & DEGENERACY")
    print("=" * 80)
    print()
    print(f"{'λ_curv':<12} {'β*':<6} {'χ²_data*':<12} {'R_penalty':<11} {'CV(S)':<8} {'CV(C_g)':<8} {'χ² range':<10} {'Bound?':<7}")
    print("-" * 95)
    for _, row in summary.iterrows():
        bound_flag = "YES" if row['bound_hit'] else "no"
        print(f"{row['lam']:<12.2e} {row['beta_star']:<6.2f} {row['chi2_data_star']:<12.3e} "
              f"{row['R_penalty']:<11.2f} {row['CV_S']:<8.1f} {row['CV_Cg']:<8.1f} "
              f"{row['chi2_data_range']:<10.1f} {bound_flag:<7}")
    print("-" * 95)
    print()

    # Penalty dominance warnings
    over_regularized = summary[summary['R_penalty'] > 10]
    if len(over_regularized) > 0:
        print("⚠ WARNING: Penalty-dominated regimes (R > 10):")
        for _, row in over_regularized.iterrows():
            print(f"  λ={row['lam']:.1e}: R={row['R_penalty']:.1f} → penalty {row['R_penalty']:.1f}× data term")
        print()

    # Elbow identification
    lam_opt, msg = identify_elbow(summary, R_max=10.0, CV_S_target=20.0)

    print("=" * 80)
    print("RECOMMENDED λ_curv SELECTION")
    print("=" * 80)
    print()
    if lam_opt is not None:
        print(f"✓ {msg}")
        print()
        row_opt = summary[summary['lam'] == lam_opt].iloc[0]
        print(f"  λ_curv = {lam_opt:.2e}")
        print(f"  Best β = {row_opt['beta_star']:.2f}")
        print(f"  χ²_data = {row_opt['chi2_data_star']:.3e}")
        print(f"  R_penalty = {row_opt['R_penalty']:.2f} (penalty/data ratio)")
        print(f"  CV(S) = {row_opt['CV_S']:.1f}%")
        print(f"  CV(C_g) = {row_opt['CV_Cg']:.1f}%")
        print(f"  χ² range = {row_opt['chi2_data_range']:.1f}×")
        print()
    else:
        print(f"✗ {msg}")
        print()

    # U hierarchy stability
    print("=" * 80)
    print("U PARAMETER STABILITY vs λ")
    print("=" * 80)
    print()
    print(f"{'λ_curv':<12} {'U_e*':<10} {'U_μ*':<10} {'U_μ/U_e':<10} {'ΔU_e':<10} {'ΔU_μ':<10}")
    print("-" * 70)
    for _, row in summary.iterrows():
        print(f"{row['lam']:<12.2e} {row['U_e_star']:<10.4f} {row['U_mu_star']:<10.4f} "
              f"{row['U_ratio_star']:<10.1f} {row['U_e_range']:<10.4f} {row['U_mu_range']:<10.4f}")
    print("-" * 70)
    print()

    # Save summary
    summary_path = Path("results/V22/t3b_lambda_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Summary saved: {summary_path}")

    # Save full data
    df_path = Path("results/V22/t3b_lambda_full_data.csv")
    df.to_csv(df_path, index=False)
    print(f"Full data saved: {df_path}")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
