#!/usr/bin/env python3
"""
T1c: Coarse β Scan (Stage 2a)

Purpose: Map χ²(β) landscape with coarse resolution to locate minimum region.

Strategy:
  - β ∈ [1.5, 3.4] at 0.2 steps → 11 points
  - Warm start: initialize each β from previous β optimum + jitter
  - Widened bounds (same as T1b)
  - Track β trajectory and parameter evolution

Next: Stage 2b will refine ±0.2 around coarse minimum with 0.02-0.05 spacing.
"""

import numpy as np
from scipy.optimize import differential_evolution
from lepton_energy_localized_v1 import LeptonEnergyLocalizedV1
from profile_likelihood_boundary_layer import calibrate_lambda
import json
import sys
from tqdm import tqdm

# Physical constants
M_E = 0.511
M_MU = 105.7
TARGET_RATIO = M_MU / M_E

# Configuration
W = 0.020
ETA_TARGET = 0.03
R_C_REF = 0.88
K_LOCALIZATION = 1.5
DELTA_V_FACTOR = 0.5
P_ENVELOPE = 6

# T1b best point (β=3.15, χ²≈10⁻⁸)
T1B_BEST = {
    "beta": 3.15,
    "chi2": 8.143891e-08,
    "params": {
        "electron": {"R_c": 0.202593, "U": 0.474143, "A": 0.545308},
        "muon": {"R_c": 1.313947, "U": 0.457582, "A": 0.649694},
    },
}

print("=" * 80)
print("T1c: COARSE β SCAN (Stage 2a)")
print("=" * 80)
print()
print("Objective: Locate β_min region via coarse scan")
print()
print("Warm start from T1b best (β=3.15):")
print(f"  χ² = {T1B_BEST['chi2']:.6e}")
print(f"  Electron: R_c={T1B_BEST['params']['electron']['R_c']:.4f}, U={T1B_BEST['params']['electron']['U']:.4f}, A={T1B_BEST['params']['electron']['A']:.4f}")
print(f"  Muon:     R_c={T1B_BEST['params']['muon']['R_c']:.4f}, U={T1B_BEST['params']['muon']['U']:.4f}, A={T1B_BEST['params']['muon']['A']:.4f}")
print()
sys.stdout.flush()


class TwoLeptonFitterWarmStart:
    """Fit with warm-start initialization from previous β"""

    def __init__(self, beta, w, lam, k_loc, delta_v_fac, p_env, sigma_model=1e-4):
        self.beta = beta
        self.w = w
        self.lam = lam
        self.sigma_model = sigma_model

        self.m_targets = np.array([M_E, M_MU])

        self.energy_calc = LeptonEnergyLocalizedV1(
            beta=beta,
            w=w,
            lam=lam,
            k_localization=k_loc,
            delta_v_factor=delta_v_fac,
            p_envelope=p_env,
        )

    def objective(self, x):
        """χ² objective"""
        R_c_e, U_e, A_e = x[0:3]
        R_c_mu, U_mu, A_mu = x[3:6]

        E_e, _, _, _ = self.energy_calc.total_energy(R_c_e, U_e, A_e)
        E_mu, _, _, _ = self.energy_calc.total_energy(R_c_mu, U_mu, A_mu)

        energies = np.array([E_e, E_mu])

        if np.any(energies <= 0) or np.any(~np.isfinite(energies)):
            return 1e12

        sigma_abs = self.sigma_model * self.m_targets
        weights = 1.0 / sigma_abs**2

        numerator = np.sum(self.m_targets * energies * weights)
        denominator = np.sum(energies**2 * weights)

        if denominator > 0:
            S_opt = numerator / denominator
        else:
            return 1e12

        masses_model = S_opt * energies
        residuals = (masses_model - self.m_targets) / sigma_abs
        chi2 = np.sum(residuals**2)

        return chi2

    def fit_warm_start(self, x_prev, bounds, maxiter=200, seed=None, workers=8):
        """
        Run DE with warm start from previous β optimum

        Population:
          - 1 copy of previous optimum
          - 15 jittered neighbors (3% std)
          - Remaining: random uniform
        """
        if seed is not None:
            np.random.seed(seed)

        n_params = len(bounds)
        popsize = max(15 * n_params, 100)

        # Seed population
        init_pop = np.zeros((popsize, n_params))

        # First: exact previous optimum
        init_pop[0, :] = x_prev

        # Next 15: jittered
        n_jitter = min(15, popsize - 1)
        bound_ranges = np.array([b[1] - b[0] for b in bounds])
        jitter_std = 0.03 * bound_ranges

        for i in range(1, n_jitter + 1):
            jittered = x_prev + np.random.normal(0, jitter_std)
            jittered = np.clip(jittered, [b[0] for b in bounds], [b[1] for b in bounds])
            init_pop[i, :] = jittered

        # Remaining: random
        for i in range(n_jitter + 1, popsize):
            for j, (lo, hi) in enumerate(bounds):
                init_pop[i, j] = np.random.uniform(lo, hi)

        result = differential_evolution(
            self.objective,
            bounds,
            maxiter=maxiter,
            init=init_pop,
            seed=seed,
            atol=1e-10,
            tol=1e-10,
            workers=workers,
            polish=True,
            updating='deferred',
        )

        x_best = result.x
        R_c_e, U_e, A_e = x_best[0:3]
        R_c_mu, U_mu, A_mu = x_best[3:6]

        # Compute diagnostics
        E_e, E_circ_e, E_stab_e, E_grad_e = self.energy_calc.total_energy(R_c_e, U_e, A_e)
        E_mu, E_circ_mu, E_stab_mu, E_grad_mu = self.energy_calc.total_energy(R_c_mu, U_mu, A_mu)

        energies = np.array([E_e, E_mu])

        sigma_abs = self.sigma_model * self.m_targets
        weights = 1.0 / sigma_abs**2
        numerator = np.sum(self.m_targets * energies * weights)
        denominator = np.sum(energies**2 * weights)
        S_opt = numerator / denominator if denominator > 0 else np.nan

        ratio = E_mu / E_e

        # Bound hits
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
            "chi2": float(result.fun),
            "S_opt": float(S_opt),
            "ratio": float(ratio),
            "parameters": {
                "electron": {"R_c": float(R_c_e), "U": float(U_e), "A": float(A_e)},
                "muon": {"R_c": float(R_c_mu), "U": float(U_mu), "A": float(A_mu)},
            },
            "energies": {
                "electron": {"E_total": float(E_e), "E_circ": float(E_circ_e), "E_stab": float(E_stab_e), "E_grad": float(E_grad_e)},
                "muon": {"E_total": float(E_mu), "E_circ": float(E_circ_mu), "E_stab": float(E_stab_mu), "E_grad": float(E_grad_mu)},
            },
            "bounds_hit": bounds_hit,
        }


# ========================================================================
# Coarse β scan
# ========================================================================

beta_range = (1.5, 3.4)
beta_step = 0.2
beta_grid = np.arange(beta_range[0], beta_range[1] + beta_step/2, beta_step)

print(f"β grid: {len(beta_grid)} points from {beta_range[0]} to {beta_range[1]} (step {beta_step})")
print(f"  {beta_grid}")
print()

# Widened bounds (same as T1b)
bounds = [
    (0.05, 3.0),
    (0.005, 0.6),
    (0.05, 0.999),
    (0.02, 1.5),
    (0.01, 0.9),
    (0.05, 0.999),
]

# Initialize from T1b best (β=3.15)
# Find closest β in grid
idx_start = np.argmin(np.abs(beta_grid - T1B_BEST["beta"]))
beta_start = beta_grid[idx_start]

print(f"Starting from β = {beta_start:.2f} (closest to T1b β = {T1B_BEST['beta']:.2f})")
print()

# Initialize x_prev
x_prev = np.array([
    T1B_BEST["params"]["electron"]["R_c"],
    T1B_BEST["params"]["electron"]["U"],
    T1B_BEST["params"]["electron"]["A"],
    T1B_BEST["params"]["muon"]["R_c"],
    T1B_BEST["params"]["muon"]["U"],
    T1B_BEST["params"]["muon"]["A"],
])

results_scan = []

print("Running coarse β scan...")
print("-" * 80)
print(f"{'β':<10} {'χ²':<15} {'S_opt':<12} {'E_μ/E_e':<12} {'Bound hits':<12}")
print("-" * 80)
sys.stdout.flush()

# Scan outward from start point in both directions
# First: β >= beta_start (ascending)
for i in range(idx_start, len(beta_grid)):
    beta = beta_grid[i]
    lam = calibrate_lambda(ETA_TARGET, beta, R_C_REF)

    fitter = TwoLeptonFitterWarmStart(
        beta=beta,
        w=W,
        lam=lam,
        k_loc=K_LOCALIZATION,
        delta_v_fac=DELTA_V_FACTOR,
        p_env=P_ENVELOPE,
        sigma_model=1e-4,
    )

    result = fitter.fit_warm_start(x_prev, bounds, maxiter=200, seed=42, workers=8)

    chi2 = result["chi2"]
    S_opt = result["S_opt"]
    ratio = result["ratio"]

    e_hits = sum(1 for v in result["bounds_hit"]["electron"].values() if v != "none")
    mu_hits = sum(1 for v in result["bounds_hit"]["muon"].values() if v != "none")
    total_hits = e_hits + mu_hits

    results_scan.append({
        "beta": float(beta),
        "result": result,
    })

    print(f"{beta:<10.2f} {chi2:<15.6e} {S_opt:<12.4f} {ratio:<12.4f} {total_hits:<12d}")
    sys.stdout.flush()

    # Update x_prev for next iteration
    x_prev = np.array([
        result["parameters"]["electron"]["R_c"],
        result["parameters"]["electron"]["U"],
        result["parameters"]["electron"]["A"],
        result["parameters"]["muon"]["R_c"],
        result["parameters"]["muon"]["U"],
        result["parameters"]["muon"]["A"],
    ])

# Second: β < beta_start (descending)
# Reset x_prev to T1b best
x_prev = np.array([
    T1B_BEST["params"]["electron"]["R_c"],
    T1B_BEST["params"]["electron"]["U"],
    T1B_BEST["params"]["electron"]["A"],
    T1B_BEST["params"]["muon"]["R_c"],
    T1B_BEST["params"]["muon"]["U"],
    T1B_BEST["params"]["muon"]["A"],
])

for i in range(idx_start - 1, -1, -1):
    beta = beta_grid[i]
    lam = calibrate_lambda(ETA_TARGET, beta, R_C_REF)

    fitter = TwoLeptonFitterWarmStart(
        beta=beta,
        w=W,
        lam=lam,
        k_loc=K_LOCALIZATION,
        delta_v_fac=DELTA_V_FACTOR,
        p_env=P_ENVELOPE,
        sigma_model=1e-4,
    )

    result = fitter.fit_warm_start(x_prev, bounds, maxiter=200, seed=42, workers=8)

    chi2 = result["chi2"]
    S_opt = result["S_opt"]
    ratio = result["ratio"]

    e_hits = sum(1 for v in result["bounds_hit"]["electron"].values() if v != "none")
    mu_hits = sum(1 for v in result["bounds_hit"]["muon"].values() if v != "none")
    total_hits = e_hits + mu_hits

    results_scan.append({
        "beta": float(beta),
        "result": result,
    })

    print(f"{beta:<10.2f} {chi2:<15.6e} {S_opt:<12.4f} {ratio:<12.4f} {total_hits:<12d}")
    sys.stdout.flush()

    # Update x_prev for next iteration
    x_prev = np.array([
        result["parameters"]["electron"]["R_c"],
        result["parameters"]["electron"]["U"],
        result["parameters"]["electron"]["A"],
        result["parameters"]["muon"]["R_c"],
        result["parameters"]["muon"]["U"],
        result["parameters"]["muon"]["A"],
    ])

print("-" * 80)
print()

# Sort results by β
results_scan.sort(key=lambda x: x["beta"])

# Find minimum
best = min(results_scan, key=lambda x: x["result"]["chi2"])
beta_min = best["beta"]
chi2_min = best["result"]["chi2"]

print("=" * 80)
print("COARSE SCAN RESULTS")
print("=" * 80)
print()

print(f"Best β (coarse): {beta_min:.2f}")
print(f"  χ²_min = {chi2_min:.6e}")
print(f"  S_opt = {best['result']['S_opt']:.4f} MeV")
print(f"  E_μ/E_e = {best['result']['ratio']:.4f} (target: {TARGET_RATIO:.4f})")
print()

print("Best-fit parameters:")
for lepton in ["electron", "muon"]:
    p = best["result"]["parameters"][lepton]
    print(f"  {lepton:8s}: R_c={p['R_c']:.6f}, U={p['U']:.6f}, A={p['A']:.6f}")
print()

# Check if minimum is interior
beta_interior = beta_range[0] + beta_step/2 < beta_min < beta_range[1] - beta_step/2

print(f"β minimum interior to scan range: {beta_interior}")
if beta_interior:
    print(f"  β_min = {beta_min:.2f} ∈ ({beta_range[0]}, {beta_range[1]})")
    print()
    print("NEXT: Stage 2b (refine ±0.2 around β={:.2f})".format(beta_min))
    refine_range = (max(beta_range[0], beta_min - 0.2), min(beta_range[1], beta_min + 0.2))
    print(f"  Suggested refinement: β ∈ [{refine_range[0]:.2f}, {refine_range[1]:.2f}] at 0.02 spacing")
else:
    print(f"  β_min = {beta_min:.2f} at edge of scan range")
    print()
    print("CAUTION: Minimum may lie outside scanned range.")
    if beta_min <= beta_range[0] + beta_step/2:
        print(f"  Consider extending scan to lower β (< {beta_range[0]})")
    else:
        print(f"  Consider extending scan to higher β (> {beta_range[1]})")

print()

# Save
results_dict = {
    "k_localization": K_LOCALIZATION,
    "delta_v_factor": DELTA_V_FACTOR,
    "p_envelope": P_ENVELOPE,
    "beta_range": list(beta_range),
    "beta_step": beta_step,
    "beta_grid": beta_grid.tolist(),
    "beta_min": beta_min,
    "chi2_min": chi2_min,
    "beta_interior": beta_interior,
    "scan_results": results_scan,
    "best_fit": best["result"],
}

with open("results/V22/t1c_coarse_beta_scan_results.json", "w") as f:
    json.dump(results_dict, f, indent=2)

print("Results saved: results/V22/t1c_coarse_beta_scan_results.json")
print()
print("=" * 80)
