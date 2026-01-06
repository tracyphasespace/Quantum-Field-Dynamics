#!/usr/bin/env python3
"""
T3b STRATEGY 2: Three-Lepton Fit (Most Robust)

Adds tau lepton to overconstrain the system and force β identification.

Physics rationale:
  2-lepton: 4 params = 4 observables → β degenerate
  3-lepton: 6 params < 6 observables → β uniquely determined

Observables:
  Masses:    m_e = 0.511 MeV, m_μ = 105.7 MeV, m_τ = 1776.86 MeV
  g-factors: g_e = 2.00231930, g_μ = 2.00233184, g_τ = 2.00118

Configuration for 4GB / 16 threads:
  - Workers: 6 (uses 6 threads, ~3.6 GB peak)
  - n_starts: 2 (memory-conscious)
  - Lambda: [0, 1e-10, 1e-09] (minimal penalty)
  - Beta: [1.8, 1.9, ..., 2.6] (9 points)

Expected: 3 λ × 9 β × ~12 min = ~5.4 hours, ~3.6 GB memory
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

sys.path.insert(0, '/home/tracy/development/QFD_SpectralGap/V22_Lepton_Analysis/validation_tests')
from t3b_two_lepton_curvature_penalty import (
    total_curvature_penalty_radial_nonuniform,
    compute_raw_moment,
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

# Physical constants
M_E = 0.511      # MeV
M_MU = 105.7     # MeV
M_TAU = 1776.86  # MeV

# g-factor targets (PDG 2020)
G_E = 2.00231930436256
G_MU = 2.0023318414
G_TAU = 2.00118  # Less precisely known

K_GEOMETRIC = 0.2
Q_CHARGE = 1.0
A_SAT = 1.0  # Cavitation saturation

print("=" * 80)
print("T3b STRATEGY 2: Three-Lepton Fit (Overconstrained System)")
print("=" * 80)
print()
print("ADDING TAU LEPTON:")
print("  Why: Overconstrain system to force unique β")
print("  2-lepton: 4 params = 4 obs → β degenerate")
print("  3-lepton: 6 params < 6 obs → β uniquely identified")
print()
print("OBSERVABLES:")
print(f"  m_e = {M_E:.3f} MeV,  m_μ = {M_MU:.1f} MeV,  m_τ = {M_TAU:.2f} MeV")
print(f"  g_e = {G_E:.8f}, g_μ = {G_MU:.8f}, g_τ = {G_TAU:.5f}")
print()
print("MASS HIERARCHY:")
print(f"  m_μ/m_e = {M_MU/M_E:.1f},  m_τ/m_μ = {M_TAU/M_MU:.1f},  m_τ/m_e = {M_TAU/M_E:.1f}")
print()

log_memory("Initial")

# Lambda sweep: minimal penalty (curvature may not be needed!)
LAM_CURV_GRID = [0.0, 1e-10, 1e-09]

print(f"Lambda sweep: {len(LAM_CURV_GRID)} values (minimal penalty)")
print(f"  {LAM_CURV_GRID}")
print()

# Beta grid: focused on likely region
beta_grid = np.arange(1.8, 2.65, 0.1)  # 9 points

print(f"Beta grid: {len(beta_grid)} points from {beta_grid[0]:.1f} to {beta_grid[-1]:.1f}")
print(f"  {beta_grid}")
print()

# Bounds: 6 parameters (R_c_e, U_e, R_c_mu, U_mu, R_c_tau, U_tau)
bounds = [
    (0.05, 3.0),     # R_c_e
    (1e-4, 0.2),     # U_e
    (0.05, 3.0),     # R_c_mu
    (1e-3, 1.0),     # U_mu
    (0.05, 3.0),     # R_c_tau
    (0.01, 2.0),     # U_tau (allow larger for tau)
]

# Initial guess: extend from 2-lepton best fit
# Assuming tau similar to muon but scaled
x_init = np.array([
    0.5912,  # R_c_e (from 2-lepton)
    0.0086,  # U_e
    1.4134,  # R_c_mu
    0.7973,  # U_mu
    1.8,     # R_c_tau (guess: larger than mu)
    1.0,     # U_tau (guess: larger circulation)
])

print("Bounds:")
for i, (name, b) in enumerate(zip(["R_c_e", "U_e", "R_c_mu", "U_mu", "R_c_tau", "U_tau"], bounds)):
    print(f"  {name:<10} ∈ [{b[0]:.2e}, {b[1]:.2e}]")
print()

# Optimized for 4GB, 16 threads
N_STARTS = 2      # Conservative for memory
WORKERS = 6       # Use 6 of 16 threads (safe for 4GB)
POPSIZE_MULT = 8  # popsize = 8*6 = 48
MAXITER = 200

print("OPTIMIZATION SETTINGS (4GB / 16 threads):")
print(f"  n_starts:     {N_STARTS} (conservative)")
print(f"  workers:      {WORKERS} (6 of 16 threads)")
print(f"  popsize:      {POPSIZE_MULT}*n_params = {POPSIZE_MULT*6}")
print(f"  maxiter:      {MAXITER}")
print()
print(f"Expected per beta: ~12 min (more complex than 2-lepton)")
print(f"Expected per lambda: {len(beta_grid)} × 12 min = ~{len(beta_grid)*12} min = {len(beta_grid)*12/60:.1f} hr")
print(f"Total runtime: {len(LAM_CURV_GRID)} λ × {len(beta_grid)*12/60:.1f} hr = ~{len(LAM_CURV_GRID)*len(beta_grid)*12/60:.1f} hr")
print()
sys.stdout.flush()


class ThreeLeptonFitter:
    """Three-lepton fitter with curvature penalty"""

    def __init__(self, beta, w, lam, k_loc, delta_v_fac, p_env, lam_curv):
        self.beta = beta
        self.w = w
        self.lam = lam
        self.lam_curv = lam_curv

        # Targets
        self.m_targets = np.array([M_E, M_MU, M_TAU])
        self.g_targets = np.array([G_E, G_MU, G_TAU])

        # Uncertainties
        self.sigma_mass = 1e-4 * self.m_targets
        self.sigma_g = np.array([1e-3, 1e-3, 1e-2])  # Tau g-factor less precise

        # Energy calculator
        self.energy_calc = LeptonEnergyLocalizedV1(
            beta=beta, w=w, lam=lam,
            k_localization=k_loc, delta_v_factor=delta_v_fac, p_envelope=p_env,
        )

        self.r = self.energy_calc.r

    def objective(self, x):
        """
        χ² for 3 leptons with FIXED A = 1.0, free U per lepton

        Parameters (6):
          x[0] = R_c_e,   x[1] = U_e
          x[2] = R_c_mu,  x[3] = U_mu
          x[4] = R_c_tau, x[5] = U_tau

        A = 1.0 for all (cavitation saturation)
        """
        R_c_e, U_e = x[0:2]
        R_c_mu, U_mu = x[2:4]
        R_c_tau, U_tau = x[4:6]
        A_e = A_mu = A_tau = A_SAT

        # Build leptons dict
        leptons = {
            "e":   {"R_c": R_c_e,   "U": U_e,   "A": A_e},
            "mu":  {"R_c": R_c_mu,  "U": U_mu,  "A": A_mu},
            "tau": {"R_c": R_c_tau, "U": U_tau, "A": A_tau},
        }

        # Get density fields
        rhos, rho_vacs = self.get_density_fields_and_vac(leptons)

        # Curvature penalty
        curv = total_curvature_penalty_radial_nonuniform(self.r, rhos, rho_vacs, eps=1e-18)

        # Energies
        E_e,   _, _, _ = self.energy_calc.total_energy(R_c_e,   U_e,   A_e)
        E_mu,  _, _, _ = self.energy_calc.total_energy(R_c_mu,  U_mu,  A_mu)
        E_tau, _, _, _ = self.energy_calc.total_energy(R_c_tau, U_tau, A_tau)
        energies = np.array([E_e, E_mu, E_tau])

        if np.any(energies <= 0) or np.any(~np.isfinite(energies)):
            return 1e12

        # Profile S over 3 masses
        weights_mass = 1.0 / self.sigma_mass**2
        numerator_S = np.sum(self.m_targets * energies * weights_mass)
        denominator_S = np.sum(energies**2 * weights_mass)
        if denominator_S <= 0:
            return 1e12

        S_opt = numerator_S / denominator_S
        masses_model = S_opt * energies

        # Magnetic moments
        R_shell_e   = R_c_e   + self.w
        R_shell_mu  = R_c_mu  + self.w
        R_shell_tau = R_c_tau + self.w

        mu_e   = compute_raw_moment(R_shell_e,   U_e)
        mu_mu  = compute_raw_moment(R_shell_mu,  U_mu)
        mu_tau = compute_raw_moment(R_shell_tau, U_tau)

        # Raw g-proxies
        mass_ratio_e   = masses_model[0] / M_E
        mass_ratio_mu  = masses_model[1] / M_E
        mass_ratio_tau = masses_model[2] / M_E

        x_e   = mu_e   / mass_ratio_e
        x_mu  = mu_mu  / mass_ratio_mu
        x_tau = mu_tau / mass_ratio_tau
        x_values = np.array([x_e, x_mu, x_tau])

        if np.any(x_values <= 0) or np.any(~np.isfinite(x_values)):
            return 1e12

        # Profile C_g over 3 leptons
        weights_g = 1.0 / self.sigma_g**2
        numerator_Cg = np.sum(self.g_targets * x_values * weights_g)
        denominator_Cg = np.sum(x_values**2 * weights_g)
        if denominator_Cg <= 0:
            return 1e12

        C_g_opt = numerator_Cg / denominator_Cg
        g_model = C_g_opt * x_values

        # χ² mass + g
        chi2_mass = np.sum(((masses_model - self.m_targets) / self.sigma_mass)**2)
        chi2_g    = np.sum(((g_model - self.g_targets) / self.sigma_g)**2)

        # Total: chi2 + curvature penalty
        return chi2_mass + chi2_g + self.lam_curv * curv

    def get_density_fields_and_vac(self, leptons: dict):
        """Build per-lepton density fields"""
        r = self.r
        Nr = r.shape[0]

        rhos = {}
        rho_vacs = {}

        rho_vac_arr = np.full((Nr,), RHO_VAC, dtype=float)

        for name, p in leptons.items():
            R_c = float(p["R_c"])
            A = float(p["A"])

            density = DensityBoundaryLayer(R_c=R_c, w=self.w, amplitude=A, rho_vac=RHO_VAC)
            delta = density.delta_rho(r)
            rho = rho_vac_arr + delta

            rhos[name] = rho
            rho_vacs[name] = rho_vac_arr

        return rhos, rho_vacs


def fit_three_lepton(fitter, x_prev, bounds, maxiter=200, base_seed=0, workers=6, n_starts=2):
    """Multi-start optimization for 3-lepton fit"""
    results = []
    chi2s = []

    bound_ranges = np.array([b[1] - b[0] for b in bounds])
    jitter_std = 0.10 * bound_ranges

    for s in range(n_starts):
        seed = None if base_seed is None else int(base_seed + s)
        np.random.seed(seed)

        x0 = x_prev + np.random.normal(0, jitter_std)
        x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])

        # Population
        n_params = len(bounds)
        popsize = max(POPSIZE_MULT * n_params, 40)

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

        # Extract solution
        x_best = result.x
        R_c_e, U_e = x_best[0:2]
        R_c_mu, U_mu = x_best[2:4]
        R_c_tau, U_tau = x_best[4:6]
        A_e = A_mu = A_tau = A_SAT

        # Curvature
        leptons = {
            "e":   {"R_c": R_c_e,   "U": U_e,   "A": A_e},
            "mu":  {"R_c": R_c_mu,  "U": U_mu,  "A": A_mu},
            "tau": {"R_c": R_c_tau, "U": U_tau, "A": A_tau},
        }
        rhos, rho_vacs = fitter.get_density_fields_and_vac(leptons)
        curv = total_curvature_penalty_radial_nonuniform(fitter.r, rhos, rho_vacs)

        # Energies and profiling
        E_e, _, _, _ = fitter.energy_calc.total_energy(R_c_e, U_e, A_e)
        E_mu, _, _, _ = fitter.energy_calc.total_energy(R_c_mu, U_mu, A_mu)
        E_tau, _, _, _ = fitter.energy_calc.total_energy(R_c_tau, U_tau, A_tau)
        energies = np.array([E_e, E_mu, E_tau])

        weights_mass = 1.0 / fitter.sigma_mass**2
        numerator_S = np.sum(fitter.m_targets * energies * weights_mass)
        denominator_S = np.sum(energies**2 * weights_mass)
        S_opt = numerator_S / denominator_S if denominator_S > 0 else np.nan
        masses_model = S_opt * energies

        R_shell_e   = R_c_e   + fitter.w
        R_shell_mu  = R_c_mu  + fitter.w
        R_shell_tau = R_c_tau + fitter.w

        mu_e   = compute_raw_moment(R_shell_e,   U_e)
        mu_mu  = compute_raw_moment(R_shell_mu,  U_mu)
        mu_tau = compute_raw_moment(R_shell_tau, U_tau)

        mass_ratio_e   = masses_model[0] / M_E
        mass_ratio_mu  = masses_model[1] / M_E
        mass_ratio_tau = masses_model[2] / M_E

        x_e   = mu_e   / mass_ratio_e
        x_mu  = mu_mu  / mass_ratio_mu
        x_tau = mu_tau / mass_ratio_tau
        x_values = np.array([x_e, x_mu, x_tau])

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
                "tau":      {"R_c": float(R_c_tau), "U": float(U_tau), "A": float(A_tau)},
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


def save_lambda_results(lam_curv, results_scan, append=True):
    """Save results to CSV"""
    rows = []
    for entry in results_scan:
        beta = entry["beta"]
        res = entry["result"]

        chi2_data = res["chi2_mass"] + res["chi2_g"]
        penalty = res["chi2_total"] - chi2_data

        U_e = res["parameters"]["electron"]["U"]
        U_mu = res["parameters"]["muon"]["U"]
        U_tau = res["parameters"]["tau"]["U"]

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
            "U_tau": U_tau,
            "chi2_data": chi2_data,
            "penalty": penalty,
            "U_mu_over_U_e": U_mu / U_e if U_e > 0 else np.nan,
            "U_tau_over_U_mu": U_tau / U_mu if U_mu > 0 else np.nan,
        })

    df_new = pd.DataFrame(rows)

    csv_path = "results/V22/t3b_three_lepton_full_data.csv"
    if append:
        try:
            df_existing = pd.read_csv(csv_path)
            df_existing = df_existing[df_existing['lam'] != lam_curv]
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined.to_csv(csv_path, index=False)
        except FileNotFoundError:
            df_new.to_csv(csv_path, index=False)
    else:
        df_new.to_csv(csv_path, index=False)

    # Summary
    S_opts = df_new["S_opt"].values
    Cg_opts = df_new["C_g_opt"].values
    chi2_data_vals = df_new["chi2_data"].values

    best_idx = df_new["loss_total"].idxmin()
    best_row = df_new.iloc[best_idx]

    CV_S = np.std(S_opts) / np.mean(S_opts) * 100 if len(S_opts) > 1 else 0.0
    CV_Cg = np.std(Cg_opts) / np.mean(Cg_opts) * 100 if len(Cg_opts) > 1 else 0.0

    chi2_range = chi2_data_vals.max() / chi2_data_vals.min() if chi2_data_vals.min() > 0 else np.inf

    summary_row = {
        "lam": lam_curv,
        "beta_star": best_row["beta"],
        "chi2_data_star": best_row["chi2_data"],
        "loss_star": best_row["loss_total"],
        "CV_S": CV_S,
        "CV_Cg": CV_Cg,
        "chi2_data_range": chi2_range,
        "U_e_star": best_row["U_e"],
        "U_mu_star": best_row["U_mu"],
        "U_tau_star": best_row["U_tau"],
    }

    df_summary_new = pd.DataFrame([summary_row])

    summary_path = "results/V22/t3b_three_lepton_summary.csv"
    if append:
        try:
            df_summary_existing = pd.read_csv(summary_path)
            df_summary_existing = df_summary_existing[df_summary_existing['lam'] != lam_curv]
            df_summary_combined = pd.concat([df_summary_existing, df_summary_new], ignore_index=True)
            df_summary_combined.to_csv(summary_path, index=False)
        except FileNotFoundError:
            df_summary_new.to_csv(summary_path, index=False)
    else:
        df_summary_new.to_csv(summary_path, index=False)

    print(f"  ✓ Saved results for λ={lam_curv:.2e}")
    sys.stdout.flush()


# Main loop
for lam_idx, lam_curv in enumerate(LAM_CURV_GRID):
    print("=" * 80)
    print(f"λ_curv = {lam_curv:.2e} ({lam_idx+1}/{len(LAM_CURV_GRID)})")
    print("=" * 80)
    print()
    log_memory("Start of lambda")

    results_scan = []
    x_prev = x_init.copy()
    idx_start = len(beta_grid) // 2

    print(f"Running β scan (starting from β={beta_grid[idx_start]:.2f})...")
    print("-" * 120)
    print(f"{'β':<8} {'χ²_tot':<12} {'χ²_m':<12} {'χ²_g':<12} {'S':<10} {'C_g':<8} {'U_e':<8} {'U_μ':<8} {'U_τ':<8}")
    print("-" * 120)
    sys.stdout.flush()

    # Ascending
    for i in tqdm(range(idx_start, len(beta_grid)), desc=f"  β asc", unit="β", leave=False):
        beta = beta_grid[i]
        lam = calibrate_lambda(ETA_TARGET, beta, R_C_REF)

        fitter = ThreeLeptonFitter(
            beta=beta, w=W, lam=lam, lam_curv=lam_curv,
            k_loc=K_LOCALIZATION, delta_v_fac=DELTA_V_FACTOR, p_env=P_ENVELOPE,
        )

        result = fit_three_lepton(fitter, x_prev, bounds, maxiter=MAXITER,
                                 base_seed=42, workers=WORKERS, n_starts=N_STARTS)
        results_scan.append({"beta": float(beta), "result": result})

        print(f"{beta:<8.2f} {result['chi2_total']:<12.2e} {result['chi2_mass']:<12.2e} {result['chi2_g']:<12.2e} "
              f"{result['S_opt']:<10.4f} {result['C_g_opt']:<8.2f} "
              f"{result['parameters']['electron']['U']:<8.4f} "
              f"{result['parameters']['muon']['U']:<8.4f} "
              f"{result['parameters']['tau']['U']:<8.4f}")
        sys.stdout.flush()

        x_prev = np.array([
            result["parameters"]["electron"]["R_c"], result["parameters"]["electron"]["U"],
            result["parameters"]["muon"]["R_c"], result["parameters"]["muon"]["U"],
            result["parameters"]["tau"]["R_c"], result["parameters"]["tau"]["U"],
        ])

        del fitter
        gc.collect()

    # Descending
    x_prev = x_init.copy()
    for i in tqdm(range(idx_start - 1, -1, -1), desc=f"  β desc", unit="β", leave=False):
        beta = beta_grid[i]
        lam = calibrate_lambda(ETA_TARGET, beta, R_C_REF)

        fitter = ThreeLeptonFitter(
            beta=beta, w=W, lam=lam, lam_curv=lam_curv,
            k_loc=K_LOCALIZATION, delta_v_fac=DELTA_V_FACTOR, p_env=P_ENVELOPE,
        )

        result = fit_three_lepton(fitter, x_prev, bounds, maxiter=MAXITER,
                                 base_seed=42, workers=WORKERS, n_starts=N_STARTS)
        results_scan.append({"beta": float(beta), "result": result})

        print(f"{beta:<8.2f} {result['chi2_total']:<12.2e} {result['chi2_mass']:<12.2e} {result['chi2_g']:<12.2e} "
              f"{result['S_opt']:<10.4f} {result['C_g_opt']:<8.2f} "
              f"{result['parameters']['electron']['U']:<8.4f} "
              f"{result['parameters']['muon']['U']:<8.4f} "
              f"{result['parameters']['tau']['U']:<8.4f}")
        sys.stdout.flush()

        x_prev = np.array([
            result["parameters"]["electron"]["R_c"], result["parameters"]["electron"]["U"],
            result["parameters"]["muon"]["R_c"], result["parameters"]["muon"]["U"],
            result["parameters"]["tau"]["R_c"], result["parameters"]["tau"]["U"],
        ])

        del fitter
        gc.collect()

    print("-" * 120)
    print()

    # Analysis
    results_scan.sort(key=lambda x: x["beta"])
    best = min(results_scan, key=lambda x: x["result"]["chi2_total"])

    S_opts = [r["result"]["S_opt"] for r in results_scan]
    Cg_opts = [r["result"]["C_g_opt"] for r in results_scan]
    CV_S = np.std(S_opts) / np.mean(S_opts) * 100 if len(S_opts) > 1 else 0.0
    CV_Cg = np.std(Cg_opts) / np.mean(Cg_opts) * 100 if len(Cg_opts) > 1 else 0.0

    print(f"RESULTS (λ = {lam_curv:.2e}):")
    print(f"  Best β:      {best['beta']:.2f}")
    print(f"  χ²_min:      {best['result']['chi2_total']:.2e}")
    print(f"  CV(S):       {CV_S:.1f}%  {'← β IDENTIFIED!' if CV_S < 20 else '← marginal' if CV_S < 40 else '← unidentified'}")
    print(f"  CV(C_g):     {CV_Cg:.1f}%")
    print()

    save_lambda_results(lam_curv, results_scan, append=True)

    del results_scan
    gc.collect()
    log_memory("After cleanup")
    print()

print("=" * 80)
print("THREE-LEPTON FIT COMPLETE!")
print("=" * 80)
print()
print("Results: results/V22/t3b_three_lepton_*.csv")
log_memory("Final")
