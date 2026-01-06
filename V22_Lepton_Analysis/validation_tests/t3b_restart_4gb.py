#!/usr/bin/env python3
"""
T3b RESTART: 4GB Memory Budget with Coarse Beta Grid

Optimizations for 4GB RAM limit:
  1. Beta grid: 10 → 7 points (30% fewer evaluations)
  2. n_starts: 5 → 2 (conservative multi-start)
  3. workers: 8 → 2 (reduced parallelism)
  4. popsize: 80 → 50 (smaller DE population)
  5. Aggressive incremental save + cleanup
  6. Monitor memory usage

Expected memory: ~3.5 GB peak (2 workers × 800MB + overhead + results)
"""

import numpy as np
from scipy.optimize import differential_evolution
import json
import sys
from tqdm import tqdm
import pandas as pd
import gc
import os
import psutil

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

def get_memory_usage_mb():
    """Get current process memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def log_memory(label=""):
    """Log memory usage"""
    mem_mb = get_memory_usage_mb()
    print(f"  [MEM] {label}: {mem_mb:.1f} MB")
    sys.stdout.flush()

print("=" * 80)
print("T3b RESTART: 4GB Memory-Constrained Run")
print("=" * 80)
print()
print("Memory optimizations for 4GB budget:")
print("  - Beta grid: 10 → 7 points (30% fewer evaluations)")
print("  - n_starts: 5 → 2 (minimal multi-start)")
print("  - workers: 8 → 2 (reduced parallel memory)")
print("  - popsize: 80 → 50 (smaller DE population)")
print("  - Incremental save + aggressive cleanup")
print()

log_memory("Initial")

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

# COARSE beta grid: 7 points instead of 10
beta_range = (1.5, 3.3)
beta_grid = np.linspace(beta_range[0], beta_range[1], 7)

print(f"COARSE β grid: {len(beta_grid)} points from {beta_range[0]} to {beta_range[1]}")
print(f"  {beta_grid}")
print()

# Bounds (same as original)
bounds = [
    (0.05, 3.0),     # R_c_e
    (1e-4, 0.2),     # U_e
    (0.05, 3.0),     # R_c_mu
    (1e-3, 1.0),     # U_mu
]

# Initial guess (same as original)
x_init = np.array([0.5912, 0.0081, 1.4134, 0.7182])

print("Bounds:")
for i, (name, b) in enumerate(zip(["R_c_e", "U_e", "R_c_mu", "U_mu"], bounds)):
    print(f"  {name:<8} ∈ [{b[0]:.1e}, {b[1]:.1e}]")
print()
sys.stdout.flush()

# AGGRESSIVE memory-saving parameters
N_STARTS = 2      # Reduced from 5 (60% reduction)
WORKERS = 2       # Reduced from 8 (75% reduction)
POPSIZE_MULT = 10 # Reduced: popsize = 10*n_params = 40 (was 80)
MAXITER = 200     # Same

print("4GB MEMORY SETTINGS:")
print(f"  n_starts:     {N_STARTS} (was 5)")
print(f"  workers:      {WORKERS} (was 8)")
print(f"  popsize:      {POPSIZE_MULT}*n_params = {POPSIZE_MULT*4} (was 80)")
print(f"  maxiter:      {MAXITER}")
print(f"  beta points:  {len(beta_grid)} (was 10)")
print()
print(f"Expected runtime per lambda: ~{len(beta_grid)*7} min = {len(beta_grid)*7/60:.1f} hr")
print(f"Total remaining: {len(LAM_CURV_GRID)} lambdas × {len(beta_grid)*7/60:.1f} hr = {len(LAM_CURV_GRID)*len(beta_grid)*7/60:.1f} hr")
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
            # Remove any existing rows for this lambda (in case of restart)
            df_existing = df_existing[df_existing['lam'] != lam_curv]
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
            # Remove any existing row for this lambda
            df_summary_existing = df_summary_existing[df_summary_existing['lam'] != lam_curv]
            df_summary_combined = pd.concat([df_summary_existing, df_summary_new], ignore_index=True)
            df_summary_combined.to_csv(summary_path, index=False)
        except FileNotFoundError:
            df_summary_new.to_csv(summary_path, index=False)
    else:
        df_summary_new.to_csv(summary_path, index=False)

    print(f"  ✓ Saved results for λ={lam_curv:.2e} to CSV")
    sys.stdout.flush()


# Custom fit function with reduced popsize
def fit_multi_start_4gb(fitter, x_prev, bounds, maxiter=200, base_seed=0, workers=2, n_starts=2):
    """Multi-start with 4GB memory budget"""
    results = []
    chi2s = []

    bound_ranges = np.array([b[1] - b[0] for b in bounds])
    jitter_std = 0.10 * bound_ranges

    for s in range(n_starts):
        seed = None if base_seed is None else int(base_seed + s)
        np.random.seed(seed)

        x0 = x_prev + np.random.normal(0, jitter_std)
        x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])

        # Reduced popsize
        n_params = len(bounds)
        popsize = max(POPSIZE_MULT * n_params, 30)

        # Seed population
        init_pop = np.zeros((popsize, n_params))
        init_pop[0, :] = x0

        # Jittered
        n_jitter = min(10, popsize - 1)
        jitter_std_local = 0.03 * bound_ranges

        for i in range(1, n_jitter + 1):
            jittered = x0 + np.random.normal(0, jitter_std_local)
            jittered = np.clip(jittered, [b[0] for b in bounds], [b[1] for b in bounds])
            init_pop[i, :] = jittered

        # Random
        for i in range(n_jitter + 1, popsize):
            for j, (lo, hi) in enumerate(bounds):
                init_pop[i, j] = np.random.uniform(lo, hi)

        result = differential_evolution(
            fitter.objective, bounds,
            maxiter=maxiter, init=init_pop, seed=seed,
            atol=1e-10, tol=1e-10, workers=workers,
            polish=True, updating='deferred',
        )

        # Extract solution and recompute diagnostics
        x_best = result.x
        R_c_e, U_e = x_best[0:2]
        R_c_mu, U_mu = x_best[2:4]
        A_e = A_mu = 1.0

        # Curvature
        leptons = {
            "e":  {"R_c": R_c_e,  "U": U_e,  "A": A_e},
            "mu": {"R_c": R_c_mu, "U": U_mu, "A": A_mu},
        }
        rhos, rho_vacs = fitter.get_density_fields_and_vac(leptons)

        from t3b_two_lepton_curvature_penalty import total_curvature_penalty_radial_nonuniform
        curv = total_curvature_penalty_radial_nonuniform(fitter.r, rhos, rho_vacs)

        # Energies and profiling
        E_e, _, _, _ = fitter.energy_calc.total_energy(R_c_e, U_e, A_e)
        E_mu, _, _, _ = fitter.energy_calc.total_energy(R_c_mu, U_mu, A_mu)
        energies = np.array([E_e, E_mu])

        weights_mass = 1.0 / fitter.sigma_mass**2
        numerator_S = np.sum(fitter.m_targets * energies * weights_mass)
        denominator_S = np.sum(energies**2 * weights_mass)
        S_opt = numerator_S / denominator_S if denominator_S > 0 else np.nan
        masses_model = S_opt * energies

        R_shell_e  = R_c_e  + fitter.w
        R_shell_mu = R_c_mu + fitter.w

        from t3b_two_lepton_curvature_penalty import compute_raw_moment
        mu_e  = compute_raw_moment(R_shell_e,  U_e)
        mu_mu = compute_raw_moment(R_shell_mu, U_mu)

        mass_ratio_e  = masses_model[0] / M_E
        mass_ratio_mu = masses_model[1] / M_E
        x_e  = mu_e  / mass_ratio_e
        x_mu = mu_mu / mass_ratio_mu
        x_values = np.array([x_e, x_mu])

        weights_g = 1.0 / fitter.sigma_g**2
        numerator_Cg = np.sum(fitter.g_targets * x_values * weights_g)
        denominator_Cg = np.sum(x_values**2 * weights_g)
        C_g_opt = numerator_Cg / denominator_Cg if denominator_Cg > 0 else np.nan

        g_model = C_g_opt * x_values

        chi2_mass = np.sum(((masses_model - fitter.m_targets) / fitter.sigma_mass)**2)
        chi2_g    = np.sum(((g_model - fitter.g_targets) / fitter.sigma_g)**2)

        r = {
            "chi2_total": float(result.fun),
            "chi2_mass": float(chi2_mass),
            "chi2_g": float(chi2_g),
            "curv": float(curv),
            "S_opt": float(S_opt),
            "C_g_opt": float(C_g_opt),
            "parameters": {
                "electron": {"R_c": float(R_c_e), "U": float(U_e), "A": float(A_e)},
                "muon":     {"R_c": float(R_c_mu), "U": float(U_mu), "A": float(A_mu)},
            },
        }

        results.append(r)
        chi2s.append(r["chi2_total"])

    chi2s = np.array(chi2s)
    best_idx = int(np.argmin(chi2s))

    best = results[best_idx]
    best["multi_start"] = {
        "n_starts": int(n_starts),
        "chi2_median": float(np.median(chi2s)),
        "chi2_all": [float(v) for v in chi2s],
        "best_index": best_idx,
    }
    return best


# Main sweep loop
for lam_idx, lam_curv in enumerate(LAM_CURV_GRID):
    print("=" * 80)
    print(f"λ_curv = {lam_curv:.2e} ({lam_idx+1}/{len(LAM_CURV_GRID)})")
    print("=" * 80)
    print()
    log_memory("Start of lambda sweep")

    results_scan = []
    x_prev = x_init.copy()
    idx_start = np.argmin(np.abs(beta_grid - 1.7))

    print(f"Running β scan (starting from β={beta_grid[idx_start]:.2f})...")
    print("-" * 100)
    print(f"{'β':<8} {'χ²_tot':<12} {'χ²_mass':<12} {'χ²_g':<12} {'curv':<12} {'S_opt':<10} {'C_g':<8} {'U_e':<8} {'U_μ':<8}")
    print("-" * 100)
    sys.stdout.flush()

    # Ascending from start point
    for i in tqdm(range(idx_start, len(beta_grid)), desc=f"  β scan (asc, λ={lam_curv:.1e})", unit="β", leave=False):
        beta = beta_grid[i]
        lam = calibrate_lambda(ETA_TARGET, beta, R_C_REF)

        fitter = TwoLeptonCavitationSaturated_T3b(
            beta=beta, w=W, lam=lam, lam_curv=lam_curv,
            k_loc=K_LOCALIZATION, delta_v_fac=DELTA_V_FACTOR, p_env=P_ENVELOPE,
        )

        result = fit_multi_start_4gb(fitter, x_prev, bounds, maxiter=MAXITER, base_seed=42,
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

    # Descending from start point - 1
    x_prev = x_init.copy()
    for i in tqdm(range(idx_start - 1, -1, -1), desc=f"  β scan (desc, λ={lam_curv:.1e})", unit="β", leave=False):
        beta = beta_grid[i]
        lam = calibrate_lambda(ETA_TARGET, beta, R_C_REF)

        fitter = TwoLeptonCavitationSaturated_T3b(
            beta=beta, w=W, lam=lam, lam_curv=lam_curv,
            k_loc=K_LOCALIZATION, delta_v_fac=DELTA_V_FACTOR, p_env=P_ENVELOPE,
        )

        result = fit_multi_start_4gb(fitter, x_prev, bounds, maxiter=MAXITER, base_seed=42,
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

    log_memory("Before sorting/analysis")

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

    log_memory("Before saving")

    # INCREMENTAL SAVE
    save_lambda_results(lam_curv, results_scan, append=True)

    log_memory("After saving")

    # Clear memory
    del results_scan
    del best
    del S_opts, Cg_opts, chi2_values
    gc.collect()

    log_memory("After cleanup")
    print()
    sys.stdout.flush()

print("=" * 80)
print("RESTART COMPLETE")
print("=" * 80)
print()
print("Results saved incrementally to:")
print("  - results/V22/t3b_lambda_full_data.csv")
print("  - results/V22/t3b_lambda_summary.csv")
print()
log_memory("Final")
