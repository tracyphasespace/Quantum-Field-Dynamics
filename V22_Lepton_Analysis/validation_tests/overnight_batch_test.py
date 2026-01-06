#!/usr/bin/env python3
"""
Overnight Batch Test: Sequential Localization Configuration Sweep

Tests 6 configurations with different (k, Δv/Rv, p) combinations to find
best localization parameters for e,μ regression.

Configuration:
  - 6 workers (3 GB → 4.5 GB estimated)
  - maxiter=50 (faster iteration, ~30-40 min per config)
  - Total runtime: ~3-4 hours for 6 configs
  - Sequential execution with full logging
"""

import numpy as np
from scipy.optimize import differential_evolution
from lepton_energy_localized_v1 import LeptonEnergyLocalizedV1
from profile_likelihood_boundary_layer import calibrate_lambda
import json
import sys
from tqdm import tqdm
from datetime import datetime
import traceback

# Physical constants
M_E = 0.511
M_MU = 105.7

# Fixed parameters
W = 0.020
ETA_TARGET = 0.03
R_C_REF = 0.88
BETA_RANGE = (3.00, 3.20)
N_BETA = 11  # Reduced from 21 for faster sweep

# Test configurations: (k, Δv/Rv, p, description)
CONFIGS = [
    (1.0, 0.5, 6, "Strong localization, moderate falloff"),
    (2.0, 0.5, 6, "Weak localization, moderate falloff"),
    (1.5, 0.25, 6, "Medium localization, narrow falloff"),
    (1.5, 0.75, 6, "Medium localization, wide falloff"),
    (1.5, 0.5, 4, "Medium localization, soft envelope"),
    (1.5, 0.5, 8, "Medium localization, sharp envelope"),
]


class TwoLeptonFitterCorrected:
    """Fit electron + muon with corrected sign convention"""

    def __init__(self, beta, w, lam, k_loc, delta_v_fac, p_env, sigma_model=1e-4):
        self.beta = beta
        self.w = w
        self.lam = lam
        self.sigma_model = sigma_model

        # Targets (e, μ only)
        self.m_targets = np.array([M_E, M_MU])

        # Create energy calculator with corrected signs
        self.energy_calc = LeptonEnergyLocalizedV1(
            beta=beta,
            w=w,
            lam=lam,
            k_localization=k_loc,
            delta_v_factor=delta_v_fac,
            p_envelope=p_env,
        )

    def objective(self, x):
        """χ² objective with global S profiling"""
        R_c_e, U_e, A_e = x[0:3]
        R_c_mu, U_mu, A_mu = x[3:6]

        # Compute energies (corrected signs: all add)
        E_e, _, _, _ = self.energy_calc.total_energy(R_c_e, U_e, A_e)
        E_mu, _, _, _ = self.energy_calc.total_energy(R_c_mu, U_mu, A_mu)

        energies = np.array([E_e, E_mu])

        # Check for invalid energies
        if np.any(energies <= 0) or np.any(~np.isfinite(energies)):
            return 1e12

        # Analytic S profiling
        sigma_abs = self.sigma_model * self.m_targets
        weights = 1.0 / sigma_abs**2

        numerator = np.sum(self.m_targets * energies * weights)
        denominator = np.sum(energies**2 * weights)

        if denominator > 0:
            S_opt = numerator / denominator
        else:
            return 1e12

        # χ²
        masses_model = S_opt * energies
        residuals = (masses_model - self.m_targets) / sigma_abs
        chi2 = np.sum(residuals**2)

        return chi2

    def fit(self, max_iter=50, seed=None, workers=6):
        """Run fit with specified workers"""
        if seed is not None:
            np.random.seed(seed)

        # Bounds
        bounds = [
            (0.5, 1.5),    # R_c_e
            (0.01, 0.10),  # U_e
            (0.70, 1.0),   # A_e
            (0.05, 0.30),  # R_c_mu
            (0.05, 0.20),  # U_mu
            (0.70, 1.0),   # A_mu
        ]

        result = differential_evolution(
            self.objective,
            bounds,
            maxiter=max_iter,
            seed=seed,
            atol=1e-8,
            tol=1e-8,
            workers=workers,
        )

        # Extract parameters
        x_best = result.x
        R_c_e, U_e, A_e = x_best[0:3]
        R_c_mu, U_mu, A_mu = x_best[3:6]

        # Compute final energies
        E_e, E_circ_e, E_stab_e, E_grad_e = self.energy_calc.total_energy(R_c_e, U_e, A_e)
        E_mu, E_circ_mu, E_stab_mu, E_grad_mu = self.energy_calc.total_energy(R_c_mu, U_mu, A_mu)

        # Get diagnostics
        R_e = R_c_e + self.w
        R_mu = R_c_mu + self.w
        _, F_inner_e, _, _, _ = self.energy_calc.circulation_energy_with_diagnostics(R_e, U_e, A_e, R_c_e)
        _, F_inner_mu, _, _, _ = self.energy_calc.circulation_energy_with_diagnostics(R_mu, U_mu, A_mu, R_c_mu)

        energies = np.array([E_e, E_mu])

        # Compute S_opt
        sigma_abs = self.sigma_model * self.m_targets
        weights = 1.0 / sigma_abs**2
        numerator = np.sum(self.m_targets * energies * weights)
        denominator = np.sum(energies**2 * weights)
        S_opt = numerator / denominator if denominator > 0 else np.nan

        masses_model = S_opt * energies

        # Check bound hits
        tol = 1e-6
        bounds_hit = {
            "electron": {
                "R_c": "lower" if abs(R_c_e - bounds[0][0]) < tol else ("upper" if abs(R_c_e - bounds[0][1]) < tol else "none"),
                "U": "lower" if abs(U_e - bounds[1][0]) < tol else ("upper" if abs(U_e - bounds[1][1]) < tol else "none"),
                "A": "lower" if abs(A_e - bounds[2][0]) < tol else ("upper" if abs(A_e - bounds[2][1]) < tol else "none"),
            },
            "muon": {
                "R_c": "lower" if abs(R_c_mu - bounds[3][0]) < tol else ("upper" if abs(R_c_mu - bounds[3][1]) < tol else "none"),
                "U": "lower" if abs(U_mu - bounds[4][0]) < tol else ("upper" if abs(U_mu - bounds[4][1]) < tol else "none"),
                "A": "lower" if abs(A_mu - bounds[5][0]) < tol else ("upper" if abs(A_mu - bounds[5][1]) < tol else "none"),
            },
        }

        return {
            "chi2": result.fun,
            "S_opt": S_opt,
            "parameters": {
                "electron": {"R_c": R_c_e, "U": U_e, "A": A_e},
                "muon": {"R_c": R_c_mu, "U": U_mu, "A": A_mu},
            },
            "energies": {
                "electron": {"E_total": E_e, "E_circ": E_circ_e, "E_stab": E_stab_e, "E_grad": E_grad_e},
                "muon": {"E_total": E_mu, "E_circ": E_circ_mu, "E_stab": E_stab_mu, "E_grad": E_grad_mu},
            },
            "diagnostics": {
                "electron": {"F_inner": F_inner_e},
                "muon": {"F_inner": F_inner_mu},
            },
            "bounds_hit": bounds_hit,
            "masses_model": masses_model,
            "masses_target": self.m_targets,
        }


def run_config_sweep(k_loc, delta_v_fac, p_env, description, config_idx):
    """Run β sweep for a single configuration"""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("=" * 80)
    print(f"CONFIG {config_idx + 1}/{len(CONFIGS)}: {description}")
    print("=" * 80)
    print(f"Started: {timestamp}")
    print(f"  k = {k_loc}")
    print(f"  Δv/Rv = {delta_v_fac}")
    print(f"  p = {p_env}")
    print()
    sys.stdout.flush()

    beta_grid = np.linspace(*BETA_RANGE, N_BETA)
    results_scan = []

    print(f"β scan: [{BETA_RANGE[0]}, {BETA_RANGE[1]}] with {N_BETA} points")
    print("-" * 80)
    print(f"{'β':<10} {'χ²':<15} {'S_opt':<12}")
    print("-" * 80)
    sys.stdout.flush()

    try:
        for beta in tqdm(beta_grid, desc=f"Config {config_idx+1}", unit="β"):
            lam = calibrate_lambda(ETA_TARGET, beta, R_C_REF)

            fitter = TwoLeptonFitterCorrected(
                beta=beta,
                w=W,
                lam=lam,
                k_loc=k_loc,
                delta_v_fac=delta_v_fac,
                p_env=p_env,
                sigma_model=1e-4,
            )

            result = fitter.fit(max_iter=50, seed=42, workers=6)

            chi2 = result["chi2"]
            S_opt = result["S_opt"]

            results_scan.append({
                "beta": beta,
                "chi2": chi2,
                "S_opt": S_opt,
                "result": result,
            })

            print(f"{beta:<10.4f} {chi2:<15.6e} {S_opt:<12.4f}")
            sys.stdout.flush()

        print("-" * 80)
        print()

        # Find minimum
        best = min(results_scan, key=lambda x: x["chi2"])
        beta_min = best["beta"]
        chi2_min = best["chi2"]

        print("RESULTS:")
        print(f"  β_min = {beta_min:.4f}")
        print(f"  χ²_min = {chi2_min:.6e}")
        print(f"  S_opt = {best['S_opt']:.4f}")
        print()

        # Acceptance criteria
        no_pathology = chi2_min < 1e6
        S_positive = best["S_opt"] > 0

        result_best = best["result"]
        e_hits = sum(1 for v in result_best["bounds_hit"]["electron"].values() if v != "none")
        mu_hits = sum(1 for v in result_best["bounds_hit"]["muon"].values() if v != "none")
        not_all_bounds = e_hits <= 1 and mu_hits <= 1

        beta_interior = BETA_RANGE[0] + 0.02 < beta_min < BETA_RANGE[1] - 0.02

        if no_pathology and S_positive and not_all_bounds and beta_interior:
            outcome = "PASS"
        elif no_pathology and S_positive:
            outcome = "SOFT_PASS"
        else:
            outcome = "FAIL"

        print(f"Outcome: {outcome}")
        print()
        sys.stdout.flush()

        # Save results
        results_dict = {
            "config_idx": config_idx,
            "description": description,
            "timestamp": timestamp,
            "k_localization": k_loc,
            "delta_v_factor": delta_v_fac,
            "p_envelope": p_env,
            "beta_min": beta_min,
            "chi2_min": chi2_min,
            "S_opt": best["S_opt"],
            "outcome": outcome,
            "best_fit": result_best,
            "all_beta_results": results_scan,
        }

        filename = f"results/V22/overnight_config{config_idx+1}_k{k_loc}_dv{delta_v_fac}_p{p_env}.json"
        with open(filename, "w") as f:
            json.dump(results_dict, f, indent=2)

        print(f"Saved: {filename}")
        print()

        return results_dict

    except Exception as e:
        print(f"ERROR in config {config_idx + 1}:")
        print(traceback.format_exc())
        sys.stdout.flush()
        return {
            "config_idx": config_idx,
            "description": description,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


# ========================================================================
# Main Batch Execution
# ========================================================================

if __name__ == "__main__":
    start_time = datetime.now()

    print("=" * 80)
    print("OVERNIGHT BATCH TEST: Localization Configuration Sweep")
    print("=" * 80)
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Configurations: {len(CONFIGS)}")
    print(f"Workers: 6")
    print(f"β points per config: {N_BETA}")
    print(f"maxiter: 50")
    print()
    print("Estimated runtime: 3-4 hours total")
    print("=" * 80)
    print()
    sys.stdout.flush()

    all_results = []

    for idx, (k, dv, p, desc) in enumerate(CONFIGS):
        result = run_config_sweep(k, dv, p, desc, idx)
        all_results.append(result)

    # Summary
    end_time = datetime.now()
    duration = end_time - start_time

    print("=" * 80)
    print("OVERNIGHT BATCH COMPLETE")
    print("=" * 80)
    print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total duration: {duration}")
    print()
    print("SUMMARY:")
    print("-" * 80)
    print(f"{'Config':<8} {'k':<6} {'Δv/Rv':<8} {'p':<4} {'χ²_min':<15} {'S_opt':<10} {'Outcome':<10}")
    print("-" * 80)

    for res in all_results:
        if "error" not in res:
            print(f"{res['config_idx']+1:<8} {res['k_localization']:<6} {res['delta_v_factor']:<8} "
                  f"{res['p_envelope']:<4} {res['chi2_min']:<15.6e} {res['S_opt']:<10.4f} {res['outcome']:<10}")
        else:
            print(f"{res['config_idx']+1:<8} ERROR: {res['error']}")

    print("-" * 80)
    print()

    # Find best config
    valid_results = [r for r in all_results if "error" not in r]
    if valid_results:
        best_overall = min(valid_results, key=lambda x: x["chi2_min"])
        print("BEST CONFIGURATION:")
        print(f"  Config {best_overall['config_idx'] + 1}: {best_overall['description']}")
        print(f"  k = {best_overall['k_localization']}, Δv/Rv = {best_overall['delta_v_factor']}, p = {best_overall['p_envelope']}")
        print(f"  χ²_min = {best_overall['chi2_min']:.6e}")
        print(f"  S_opt = {best_overall['S_opt']:.4f}")
        print(f"  β_min = {best_overall['beta_min']:.4f}")
        print(f"  Outcome: {best_overall['outcome']}")
        print()

    # Save summary
    summary = {
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration_seconds": duration.total_seconds(),
        "configurations_tested": len(CONFIGS),
        "results": all_results,
    }

    with open("results/V22/overnight_batch_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("Summary saved: results/V22/overnight_batch_summary.json")
    print("=" * 80)
    sys.stdout.flush()
