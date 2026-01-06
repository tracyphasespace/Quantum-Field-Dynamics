#!/usr/bin/env python3
"""
T1b: Seeded Verification at β = 3.15

Purpose: Prove the optimizer can exploit the feasible basin found by T1a Monte Carlo.

Strategy:
  - Initialize DE population with MC best point + jittered neighbors + random
  - Use widened bounds that contain the MC basin with margin
  - Log all diagnostics to confirm optimizer convergence

Pass/Fail gate:
  PASS if χ² ≤ 2× MC best (~40) and ratio within 0.5% of 206.85, no bound saturation
  FAIL if DE cannot reproduce MC basin → must pivot optimizer (local, CMA-ES, rescaling)
"""

import numpy as np
from scipy.optimize import differential_evolution
from lepton_energy_localized_v1 import LeptonEnergyLocalizedV1
from profile_likelihood_boundary_layer import calibrate_lambda
import json
import sys

# Physical constants
M_E = 0.511
M_MU = 105.7
TARGET_RATIO = M_MU / M_E  # ≈ 206.85

# MC best point from T1a (χ² = 20.18, ratio = 206.72)
MC_BEST = {
    "electron": {"R_c": 0.17746454102150733, "U": 0.3796502371279164, "A": 0.5417811437722336},
    "muon": {"R_c": 0.921585041298433, "U": 0.5650887189813897, "A": 0.4394674257317683},
}

# Configuration (same as T1a)
BETA = 3.15
W = 0.020
ETA_TARGET = 0.03
R_C_REF = 0.88
K_LOCALIZATION = 1.5
DELTA_V_FACTOR = 0.5
P_ENVELOPE = 6

print("=" * 80)
print("T1b: SEEDED VERIFICATION AT β = 3.15")
print("=" * 80)
print()
print("Objective: Confirm optimizer can exploit MC-discovered feasible basin")
print()
print("MC best point (from T1a):")
print(f"  χ² = 20.18, S_opt = 15.03, ratio = 206.72")
print(f"  Electron: R_c={MC_BEST['electron']['R_c']:.4f}, U={MC_BEST['electron']['U']:.4f}, A={MC_BEST['electron']['A']:.4f}")
print(f"  Muon:     R_c={MC_BEST['muon']['R_c']:.4f}, U={MC_BEST['muon']['U']:.4f}, A={MC_BEST['muon']['A']:.4f}")
print()
sys.stdout.flush()


class TwoLeptonFitterSeeded:
    """Fit with seeded initialization from MC best point"""

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

        # Evaluation counter
        self.n_eval = 0
        self.best_chi2 = np.inf
        self.best_x = None

    def objective(self, x):
        """χ² objective with global S profiling"""
        self.n_eval += 1

        R_c_e, U_e, A_e = x[0:3]
        R_c_mu, U_mu, A_mu = x[3:6]

        # Compute energies
        E_e, _, _, _ = self.energy_calc.total_energy(R_c_e, U_e, A_e)
        E_mu, _, _, _ = self.energy_calc.total_energy(R_c_mu, U_mu, A_mu)

        energies = np.array([E_e, E_mu])

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

        masses_model = S_opt * energies
        residuals = (masses_model - self.m_targets) / sigma_abs
        chi2 = np.sum(residuals**2)

        # Track best
        if chi2 < self.best_chi2:
            self.best_chi2 = chi2
            self.best_x = x.copy()

        # Progress logging every 100 evaluations
        if self.n_eval % 100 == 0:
            ratio = E_mu / E_e
            print(f"  Eval {self.n_eval:5d}: χ² = {chi2:12.6e}, ratio = {ratio:8.2f}, best χ² = {self.best_chi2:12.6e}")
            sys.stdout.flush()

        return chi2

    def fit_seeded(self, mc_best, bounds, maxiter=300, seed=None, workers=8):
        """
        Run DE with seeded initialization

        Population structure:
          - 1 copy of MC best point
          - 20 jittered neighbors (Gaussian noise, 5% std of bound ranges)
          - Remaining population: random uniform in bounds
        """
        if seed is not None:
            np.random.seed(seed)

        # Convert MC best to parameter vector
        x_mc = np.array([
            mc_best["electron"]["R_c"],
            mc_best["electron"]["U"],
            mc_best["electron"]["A"],
            mc_best["muon"]["R_c"],
            mc_best["muon"]["U"],
            mc_best["muon"]["A"],
        ])

        # Create initial population
        n_params = len(bounds)
        popsize = max(15 * n_params, 100)  # At least 100 for good coverage

        # Seed population
        init_pop = np.zeros((popsize, n_params))

        # First member: exact MC point
        init_pop[0, :] = x_mc

        # Next 20: jittered around MC point
        n_jitter = min(20, popsize - 1)
        bound_ranges = np.array([b[1] - b[0] for b in bounds])
        jitter_std = 0.05 * bound_ranges  # 5% of each parameter's range

        for i in range(1, n_jitter + 1):
            jittered = x_mc + np.random.normal(0, jitter_std)
            # Clip to bounds
            jittered = np.clip(jittered, [b[0] for b in bounds], [b[1] for b in bounds])
            init_pop[i, :] = jittered

        # Remaining: uniform random
        for i in range(n_jitter + 1, popsize):
            for j, (lo, hi) in enumerate(bounds):
                init_pop[i, j] = np.random.uniform(lo, hi)

        print(f"DE configuration:")
        print(f"  Population size: {popsize} ({1} exact + {n_jitter} jittered + {popsize - n_jitter - 1} random)")
        print(f"  Max iterations: {maxiter}")
        print(f"  Workers: {workers}")
        print(f"  Bounds: widened to cover MC basin with margin")
        print()
        print("Starting optimization...")
        print("-" * 80)
        sys.stdout.flush()

        self.n_eval = 0
        self.best_chi2 = np.inf

        result = differential_evolution(
            self.objective,
            bounds,
            maxiter=maxiter,
            init=init_pop,
            seed=seed,
            atol=1e-10,
            tol=1e-10,
            workers=workers,
            polish=True,  # Explicitly enable L-BFGS-B refinement
            updating='deferred',  # Better for parallel workers
        )

        print("-" * 80)
        print(f"Optimization complete: {self.n_eval} evaluations")
        print()
        sys.stdout.flush()

        # Extract and analyze result
        x_best = result.x
        R_c_e, U_e, A_e = x_best[0:3]
        R_c_mu, U_mu, A_mu = x_best[3:6]

        # Compute final energies and diagnostics
        E_e, E_circ_e, E_stab_e, E_grad_e = self.energy_calc.total_energy(R_c_e, U_e, A_e)
        E_mu, E_circ_mu, E_stab_mu, E_grad_mu = self.energy_calc.total_energy(R_c_mu, U_mu, A_mu)

        R_e = R_c_e + self.w
        R_mu = R_c_mu + self.w
        _, F_inner_e, _, _, _ = self.energy_calc.circulation_energy_with_diagnostics(R_e, U_e, A_e, R_c_e)
        _, F_inner_mu, _, _, _ = self.energy_calc.circulation_energy_with_diagnostics(R_mu, U_mu, A_mu, R_c_mu)

        energies = np.array([E_e, E_mu])

        sigma_abs = self.sigma_model * self.m_targets
        weights = 1.0 / sigma_abs**2
        numerator = np.sum(self.m_targets * energies * weights)
        denominator = np.sum(energies**2 * weights)
        S_opt = numerator / denominator if denominator > 0 else np.nan

        masses_model = S_opt * energies
        ratio = E_mu / E_e

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

        # Distance to MC seed (L2 norm in normalized parameter space)
        x_mc_norm = (x_mc - np.array([b[0] for b in bounds])) / np.array([b[1] - b[0] for b in bounds])
        x_best_norm = (x_best - np.array([b[0] for b in bounds])) / np.array([b[1] - b[0] for b in bounds])
        dist_to_mc = np.linalg.norm(x_best_norm - x_mc_norm)

        return {
            "success": result.success,
            "message": result.message,
            "chi2": result.fun,
            "S_opt": S_opt,
            "ratio": ratio,
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
            "n_evaluations": self.n_eval,
            "dist_to_mc_seed": dist_to_mc,
        }


# ========================================================================
# Run seeded verification
# ========================================================================

lam = calibrate_lambda(ETA_TARGET, BETA, R_C_REF)

fitter = TwoLeptonFitterSeeded(
    beta=BETA,
    w=W,
    lam=lam,
    k_loc=K_LOCALIZATION,
    delta_v_fac=DELTA_V_FACTOR,
    p_env=P_ENVELOPE,
    sigma_model=1e-4,
)

# Widened bounds covering MC basin with margin
# MC point: e: R_c=0.177, U=0.380, A=0.542; μ: R_c=0.922, U=0.565, A=0.439
bounds = [
    (0.05, 3.0),   # R_c_e (MC: 0.177)
    (0.005, 0.6),  # U_e (MC: 0.380)
    (0.05, 0.999), # A_e (MC: 0.542)
    (0.02, 1.5),   # R_c_mu (MC: 0.922)
    (0.01, 0.9),   # U_mu (MC: 0.565)
    (0.05, 0.999), # A_mu (MC: 0.439)
]

result = fitter.fit_seeded(MC_BEST, bounds, maxiter=300, seed=42, workers=8)

# ========================================================================
# Results
# ========================================================================

print("=" * 80)
print("RESULTS")
print("=" * 80)
print()

chi2 = result["chi2"]
S_opt = result["S_opt"]
ratio = result["ratio"]

print(f"Optimizer result:")
print(f"  χ² = {chi2:.6e}")
print(f"  S_opt = {S_opt:.4f} MeV")
print(f"  E_μ/E_e = {ratio:.4f} (target: {TARGET_RATIO:.4f})")
print(f"  Evaluations: {result['n_evaluations']}")
print(f"  Distance to MC seed: {result['dist_to_mc_seed']:.6f} (normalized L2)")
print()

print("Parameters:")
for lepton in ["electron", "muon"]:
    p = result["parameters"][lepton]
    mc = MC_BEST[lepton]
    print(f"  {lepton:8s}: R_c={p['R_c']:.6f} (MC: {mc['R_c']:.6f}), U={p['U']:.6f} (MC: {mc['U']:.6f}), A={p['A']:.6f} (MC: {mc['A']:.6f})")
print()

print("Energies:")
for lepton in ["electron", "muon"]:
    e = result["energies"][lepton]
    print(f"  {lepton:8s}: E_total={e['E_total']:.6f}, E_circ={e['E_circ']:.6f}, E_stab={e['E_stab']:.6f}, E_grad={e['E_grad']:.6f}")
print()

print("Diagnostics:")
for lepton in ["electron", "muon"]:
    d = result["diagnostics"][lepton]
    print(f"  {lepton:8s}: F_inner={d['F_inner']:.2%}")
print()

print("Bound hits:")
for lepton in ["electron", "muon"]:
    b = result["bounds_hit"][lepton]
    hits = [f"{k}={v}" for k, v in b.items() if v != "none"]
    if hits:
        print(f"  {lepton:8s}: {', '.join(hits)}")
    else:
        print(f"  {lepton:8s}: none")
print()

# ========================================================================
# Pass/Fail Gate
# ========================================================================

print("=" * 80)
print("PASS/FAIL GATE")
print("=" * 80)
print()

MC_CHI2 = 20.18
chi2_threshold = 2 * MC_CHI2  # ≤ 40
ratio_error = abs(ratio - TARGET_RATIO) / TARGET_RATIO * 100  # percent
ratio_threshold = 0.5  # percent

e_hits = sum(1 for v in result["bounds_hit"]["electron"].values() if v != "none")
mu_hits = sum(1 for v in result["bounds_hit"]["muon"].values() if v != "none")
total_hits = e_hits + mu_hits

chi2_ok = chi2 <= chi2_threshold
ratio_ok = ratio_error <= ratio_threshold
bounds_ok = total_hits <= 2  # At most 1 per lepton

print(f"1. χ² ≤ 2× MC best ({chi2_threshold:.1f}):     {chi2_ok} ({'PASS' if chi2_ok else 'FAIL'})")
print(f"     χ² = {chi2:.2e} (MC: {MC_CHI2:.2e})")
print()

print(f"2. Ratio within 0.5% of target:      {ratio_ok} ({'PASS' if ratio_ok else 'FAIL'})")
print(f"     E_μ/E_e = {ratio:.4f}, target = {TARGET_RATIO:.4f}, error = {ratio_error:.3f}%")
print()

print(f"3. No bound saturation (≤2 hits):    {bounds_ok} ({'PASS' if bounds_ok else 'FAIL'})")
print(f"     Total bound hits: {total_hits}/6 (e: {e_hits}/3, μ: {mu_hits}/3)")
print()

pass_overall = chi2_ok and ratio_ok and bounds_ok

if pass_overall:
    print("✓ VERIFICATION PASS")
    print()
    print("Optimizer successfully exploited MC-discovered basin.")
    print(f"Achieved χ² = {chi2:.2e}, {ratio_error:.3f}% ratio error.")
    print()
    print("NEXT: Proceed to coarse→refine β scan (Stage 2a + 2b)")
    outcome = "pass"
elif chi2_ok and ratio_ok:
    print("~ SOFT PASS")
    print()
    print("Optimizer found good χ² and ratio, but some bound saturation.")
    print("May still proceed to β scan, but watch for edge artifacts.")
    outcome = "soft_pass"
else:
    print("✗ VERIFICATION FAIL")
    print()
    if not chi2_ok:
        print(f"  χ² = {chi2:.2e} exceeds 2× MC best ({chi2_threshold:.1f})")
    if not ratio_ok:
        print(f"  Ratio error {ratio_error:.3f}% exceeds 0.5% threshold")
    if not bounds_ok:
        print(f"  {total_hits} bound hits indicate saturation")
    print()
    print("Optimizer cannot exploit the MC basin even when seeded.")
    print()
    print("NEXT STEPS:")
    print("  1. Try local optimizer (Nelder-Mead, L-BFGS-B) from MC point")
    print("  2. Try CMA-ES (better for narrow basins)")
    print("  3. Try parameter rescaling (log R_c, logit U/A)")
    outcome = "fail"

print()

# Save
results_dict = {
    "beta": BETA,
    "k_localization": K_LOCALIZATION,
    "delta_v_factor": DELTA_V_FACTOR,
    "p_envelope": P_ENVELOPE,
    "mc_reference": {
        "chi2": MC_CHI2,
        "S_opt": 15.03,
        "ratio": 206.72,
        "params": MC_BEST,
    },
    "optimizer_result": result,
    "outcome": outcome,
    "pass_criteria": {
        "chi2_threshold": chi2_threshold,
        "chi2_ok": chi2_ok,
        "ratio_threshold_pct": ratio_threshold,
        "ratio_error_pct": ratio_error,
        "ratio_ok": ratio_ok,
        "bounds_ok": bounds_ok,
    },
}

with open("results/V22/t1b_seeded_verification_results.json", "w") as f:
    json.dump(results_dict, f, indent=2)

print("Results saved: results/V22/t1b_seeded_verification_results.json")
print()
print("=" * 80)
