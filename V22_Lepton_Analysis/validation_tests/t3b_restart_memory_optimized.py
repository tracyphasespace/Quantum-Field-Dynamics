#!/usr/bin/env python3
"""
T3b RESTART: Memory-Optimized Continuation from OOM Crash

Crash point: λ_curv = 3e-08, β = 3.30 (sweep 7/10)
Completed: λ_curv = 0, 1e-10, 3e-10, 1e-09, 3e-09, 1e-08 (6/10)

Memory optimizations:
  1. Reduce n_starts: 5 → 3 (40% fewer optimizations)
  2. Reduce workers: 8 → 4 (50% less parallel memory)
  3. Incremental saving after each lambda
  4. Clear accumulated results after saving
  5. Resume from λ = 1e-08 (verify) or 3e-08 (crash point)
"""

import numpy as np
from scipy.optimize import differential_evolution
import json
import sys
from tqdm import tqdm
import pandas as pd
import gc

from lepton_energy_localized_v1 import LeptonEnergyLocalizedV1
from lepton_energy_boundary_layer import DensityBoundaryLayer, RHO_VAC
from profile_likelihood_boundary_layer import calibrate_lambda

# Import the class from the original script
import sys
sys.path.insert(0, '/home/tracy/development/QFD_SpectralGap/V22_Lepton_Analysis/validation_tests')
from t3b_two_lepton_curvature_penalty import (
    TwoLeptonCavitationSaturated_T3b,
    M_E, M_MU, G_E, G_MU,
    W, ETA_TARGET, R_C_REF, K_LOCALIZATION, DELTA_V_FACTOR, P_ENVELOPE,
)

print("=" * 80)
print("T3b RESTART: Memory-Optimized Continuation")
print("=" * 80)
print()
print("Memory optimizations:")
print("  - n_starts: 5 → 3 (fewer multi-start optimizations)")
print("  - workers: 8 → 4 (less parallel memory)")
print("  - Incremental save after each lambda")
print("  - Aggressive garbage collection")
print()

# Full lambda grid
LAM_CURV_GRID_FULL = [0.0, 1e-10, 3e-10, 1e-9, 3e-9, 1e-8, 3e-8, 1e-7, 3e-7, 1e-6]

# Check what's already completed from CSV
try:
    df_summary = pd.read_csv("results/V22/t3b_lambda_summary.csv")
    completed_lambdas = set(df_summary['lam'].values)
    print(f"Found existing results for {len(completed_lambdas)} lambda values:")
    print(f"  {sorted(completed_lambdas)}")
    print()

    # Determine restart point
    remaining_lambdas = [lam for lam in LAM_CURV_GRID_FULL if lam not in completed_lambdas]

    if len(remaining_lambdas) == 0:
        print("All lambda values already completed! Nothing to do.")
        sys.exit(0)

    LAM_CURV_GRID = remaining_lambdas
    print(f"Remaining {len(LAM_CURV_GRID)} lambda values to process:")
    print(f"  {LAM_CURV_GRID}")
    print()

except FileNotFoundError:
    print("No existing CSV found. Starting from scratch.")
    LAM_CURV_GRID = LAM_CURV_GRID_FULL
    print(f"Processing all {len(LAM_CURV_GRID)} lambda values:")
    print(f"  {LAM_CURV_GRID}")
    print()

# Beta grid (same as original)
beta_range = (1.5, 3.4)
beta_step = 0.2
beta_grid = np.arange(beta_range[0], beta_range[1] + beta_step/2, beta_step)

# Bounds (same as original)
bounds = [
    (0.05, 3.0),     # R_c_e
    (1e-4, 0.2),     # U_e
    (0.05, 3.0),     # R_c_mu
    (1e-3, 1.0),     # U_mu
]

# Initial guess (same as original)
x_init = np.array([0.5912, 0.0081, 1.4134, 0.7182])

print(f"β grid: {len(beta_grid)} points from {beta_range[0]} to {beta_range[1]} (step {beta_step})")
print()
print("Bounds:")
for i, (name, b) in enumerate(zip(["R_c_e", "U_e", "R_c_mu", "U_mu"], bounds)):
    print(f"  {name:<8} ∈ [{b[0]:.1e}, {b[1]:.1e}]")
print()
sys.stdout.flush()

# Memory-optimized parameters
N_STARTS = 3      # Reduced from 5
WORKERS = 4       # Reduced from 8
MAXITER = 200     # Same

print("MEMORY-OPTIMIZED SETTINGS:")
print(f"  n_starts: {N_STARTS} (was 5)")
print(f"  workers:  {WORKERS} (was 8)")
print(f"  maxiter:  {MAXITER}")
print()
sys.stdout.flush()

def save_lambda_results(lam_curv, results_scan, append=True):
    """Save results for one lambda value to CSV (incremental)"""

    # Prepare data rows
    rows = []
    for entry in results_scan:
        beta = entry["beta"]
        res = entry["result"]

        chi2_data = res["chi2_mass"] + res["chi2_g"]
        penalty = res["chi2_total"] - chi2_data

        U_e = res["parameters"]["electron"]["U"]
        U_mu = res["parameters"]["muon"]["U"]

        rows.append({
            "lam": lam_curv,
            "beta": beta,
            "loss_total": res["chi2_total"],
            "chi2_mass": res["chi2_mass"],
            "chi2_g": res["chi2_g"],
            "curv": res["curv"],
            "S_opt": res["S_opt"],
            "C_g_opt": res["C_g_opt"],
            "U_e": U_e,
            "U_mu": U_mu,
            "chi2_data": chi2_data,
            "penalty": penalty,
            "U_ratio": U_mu / U_e if U_e > 0 else np.nan,
        })

    df_new = pd.DataFrame(rows)

    # Append or create
    csv_path = "results/V22/t3b_lambda_full_data.csv"
    if append:
        try:
            df_existing = pd.read_csv(csv_path)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined.to_csv(csv_path, index=False)
        except FileNotFoundError:
            df_new.to_csv(csv_path, index=False)
    else:
        df_new.to_csv(csv_path, index=False)

    # Update summary
    S_opts = df_new["S_opt"].values
    Cg_opts = df_new["C_g_opt"].values
    U_e_vals = df_new["U_e"].values
    U_mu_vals = df_new["U_mu"].values
    U_ratio_vals = df_new["U_ratio"].values
    chi2_data_vals = df_new["chi2_data"].values

    best_idx = df_new["loss_total"].idxmin()
    best_row = df_new.iloc[best_idx]

    CV_S = np.std(S_opts) / np.mean(S_opts) * 100 if len(S_opts) > 1 else 0.0
    CV_Cg = np.std(Cg_opts) / np.mean(Cg_opts) * 100 if len(Cg_opts) > 1 else 0.0

    chi2_range = chi2_data_vals.max() / chi2_data_vals.min() if chi2_data_vals.min() > 0 else np.inf

    # Penalty ratio
    penalty_star = best_row["penalty"]
    chi2_data_star = best_row["chi2_data"]
    R_penalty = penalty_star / chi2_data_star if chi2_data_star > 0 else np.inf

    summary_row = {
        "lam": lam_curv,
        "beta_star": best_row["beta"],
        "chi2_data_star": chi2_data_star,
        "curv_star": best_row["curv"],
        "penalty_star": penalty_star,
        "loss_star": best_row["loss_total"],
        "R_penalty": R_penalty,
        "CV_S": CV_S,
        "CV_Cg": CV_Cg,
        "chi2_data_range": chi2_range,
        "U_e_star": best_row["U_e"],
        "U_mu_star": best_row["U_mu"],
        "U_ratio_star": best_row["U_ratio"],
        "U_e_range": U_e_vals.max() - U_e_vals.min(),
        "U_mu_range": U_mu_vals.max() - U_mu_vals.min(),
        "U_ratio_mean": U_ratio_vals.mean(),
        "U_ratio_std": U_ratio_vals.std(),
        "bound_hit": False,  # Placeholder
    }

    df_summary_new = pd.DataFrame([summary_row])

    summary_path = "results/V22/t3b_lambda_summary.csv"
    if append:
        try:
            df_summary_existing = pd.read_csv(summary_path)
            df_summary_combined = pd.concat([df_summary_existing, df_summary_new], ignore_index=True)
            df_summary_combined.to_csv(summary_path, index=False)
        except FileNotFoundError:
            df_summary_new.to_csv(summary_path, index=False)
    else:
        df_summary_new.to_csv(summary_path, index=False)

    print(f"  ✓ Saved results for λ={lam_curv:.2e} to CSV")
    sys.stdout.flush()


# Main sweep loop
for lam_idx, lam_curv in enumerate(LAM_CURV_GRID):
    print("=" * 80)
    print(f"λ_curv = {lam_curv:.2e} ({lam_idx+1}/{len(LAM_CURV_GRID)})")
    print("=" * 80)
    print()
    sys.stdout.flush()

    results_scan = []
    x_prev = x_init.copy()
    idx_start = np.argmin(np.abs(beta_grid - 1.7))

    print(f"Running β scan...")
    print("-" * 100)
    print(f"{'β':<8} {'χ²_tot':<12} {'χ²_mass':<12} {'χ²_g':<12} {'curv':<12} {'S_opt':<10} {'C_g':<8} {'U_e':<8} {'U_μ':<8}")
    print("-" * 100)
    sys.stdout.flush()

    # Ascending from β=1.7
    for i in tqdm(range(idx_start, len(beta_grid)), desc=f"  β scan (asc, λ={lam_curv:.1e})", unit="β", leave=False):
        beta = beta_grid[i]
        lam = calibrate_lambda(ETA_TARGET, beta, R_C_REF)

        fitter = TwoLeptonCavitationSaturated_T3b(
            beta=beta, w=W, lam=lam, lam_curv=lam_curv,
            k_loc=K_LOCALIZATION, delta_v_fac=DELTA_V_FACTOR, p_env=P_ENVELOPE,
        )

        result = fitter.fit_multi_start(x_prev, bounds, maxiter=MAXITER, base_seed=42,
                                       workers=WORKERS, n_starts=N_STARTS)
        results_scan.append({"beta": float(beta), "result": result})

        print(f"{beta:<8.2f} {result['chi2_total']:<12.6e} {result['chi2_mass']:<12.6e} {result['chi2_g']:<12.6e} " +
              f"{result['curv']:<12.6e} {result['S_opt']:<10.4f} {result['C_g_opt']:<8.2f} " +
              f"{result['parameters']['electron']['U']:<8.4f} {result['parameters']['muon']['U']:<8.4f}")
        sys.stdout.flush()

        x_prev = np.array([
            result["parameters"]["electron"]["R_c"], result["parameters"]["electron"]["U"],
            result["parameters"]["muon"]["R_c"], result["parameters"]["muon"]["U"],
        ])

        # Force cleanup
        del fitter
        gc.collect()

    # Descending from β=1.5
    x_prev = x_init.copy()
    for i in tqdm(range(idx_start - 1, -1, -1), desc=f"  β scan (desc, λ={lam_curv:.1e})", unit="β", leave=False):
        beta = beta_grid[i]
        lam = calibrate_lambda(ETA_TARGET, beta, R_C_REF)

        fitter = TwoLeptonCavitationSaturated_T3b(
            beta=beta, w=W, lam=lam, lam_curv=lam_curv,
            k_loc=K_LOCALIZATION, delta_v_fac=DELTA_V_FACTOR, p_env=P_ENVELOPE,
        )

        result = fitter.fit_multi_start(x_prev, bounds, maxiter=MAXITER, base_seed=42,
                                       workers=WORKERS, n_starts=N_STARTS)
        results_scan.append({"beta": float(beta), "result": result})

        print(f"{beta:<8.2f} {result['chi2_total']:<12.6e} {result['chi2_mass']:<12.6e} {result['chi2_g']:<12.6e} " +
              f"{result['curv']:<12.6e} {result['S_opt']:<10.4f} {result['C_g_opt']:<8.2f} " +
              f"{result['parameters']['electron']['U']:<8.4f} {result['parameters']['muon']['U']:<8.4f}")
        sys.stdout.flush()

        x_prev = np.array([
            result["parameters"]["electron"]["R_c"], result["parameters"]["electron"]["U"],
            result["parameters"]["muon"]["R_c"], result["parameters"]["muon"]["U"],
        ])

        # Force cleanup
        del fitter
        gc.collect()

    print("-" * 100)
    print()

    # Sort by β
    results_scan.sort(key=lambda x: x["beta"])

    # Quick summary
    best = min(results_scan, key=lambda x: x["result"]["chi2_total"])
    S_opts = [r["result"]["S_opt"] for r in results_scan]
    Cg_opts = [r["result"]["C_g_opt"] for r in results_scan]
    CV_S = np.std(S_opts) / np.mean(S_opts) * 100
    CV_Cg = np.std(Cg_opts) / np.mean(Cg_opts) * 100

    chi2_values = [r["result"]["chi2_total"] for r in results_scan]
    chi2_range = max(chi2_values) / min(chi2_values) if min(chi2_values) > 0 else np.inf

    print(f"RESULTS (λ_curv = {lam_curv:.2e}):")
    print(f"  Best β:      {best['beta']:.2f}")
    print(f"  χ²_min:      {best['result']['chi2_total']:.6e}")
    print(f"  CV(S_opt):   {CV_S:.1f}%")
    print(f"  CV(C_g_opt): {CV_Cg:.1f}%")
    print(f"  χ² range:    {chi2_range:.1f}×")
    print()

    # INCREMENTAL SAVE
    save_lambda_results(lam_curv, results_scan, append=True)

    # Clear memory
    del results_scan
    gc.collect()

    sys.stdout.flush()

print("=" * 80)
print("RESTART COMPLETE")
print("=" * 80)
print()
print("Results saved incrementally to:")
print("  - results/V22/t3b_lambda_full_data.csv")
print("  - results/V22/t3b_lambda_summary.csv")
print()
