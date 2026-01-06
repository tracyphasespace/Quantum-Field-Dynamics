#!/usr/bin/env python3
"""
T3a Single-β Test: Validate 3-lepton + universal U implementation

Test β = 3.0 with n_starts=5 to confirm:
  1. No syntax/runtime errors
  2. Multi-start converges to same basin
  3. χ² values reasonable
  4. g_τ prediction reasonable
"""

import numpy as np
from scipy.optimize import differential_evolution
from lepton_energy_localized_v1 import LeptonEnergyLocalizedV1
from profile_likelihood_boundary_layer import calibrate_lambda
import json
import sys

# Physical constants (MeV)
M_E   = 0.511
M_MU  = 105.7
M_TAU = 1776.86

# g-factor targets
G_E  = 2.00231930436256
G_MU = 2.0023318414

# Magnetic moment proxy
K_GEOMETRIC = 0.2
Q_CHARGE = 1.0

# Configuration
W = 0.020
ETA_TARGET = 0.03
R_C_REF = 0.88
K_LOCALIZATION = 1.5
DELTA_V_FACTOR = 0.5
P_ENVELOPE = 6
A_SAT = 1.0

print("=" * 80)
print("T3a SINGLE-β TEST: β = 3.0, n_starts = 5")
print("=" * 80)
print()
print("Objective: Validate implementation before full scan")
print()
sys.stdout.flush()


def compute_raw_moment(R_shell, U):
    return K_GEOMETRIC * Q_CHARGE * R_shell * U


class ThreeLeptonUniversalU:
    def __init__(self, beta, w, lam, k_loc, delta_v_fac, p_env):
        self.beta = beta
        self.w = w
        self.lam = lam
        self.m_targets = np.array([M_E, M_MU, M_TAU])
        self.g_targets = np.array([G_E, G_MU])
        self.sigma_mass = np.array([1e-4 * M_E, 1e-4 * M_MU, 0.12])
        self.sigma_g = 1e-3
        self.energy_calc = LeptonEnergyLocalizedV1(
            beta=beta, w=w, lam=lam,
            k_localization=k_loc, delta_v_factor=delta_v_fac, p_envelope=p_env,
        )

    def objective(self, x):
        R_c_e, R_c_mu, R_c_tau, U_univ = x
        U_e = U_mu = U_tau = U_univ
        A_e = A_mu = A_tau = A_SAT

        E_e,  _, _, _ = self.energy_calc.total_energy(R_c_e,  U_e,  A_e)
        E_mu, _, _, _ = self.energy_calc.total_energy(R_c_mu, U_mu, A_mu)
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

        # g-proxies
        mass_ratio_e   = masses_model[0] / M_E
        mass_ratio_mu  = masses_model[1] / M_E
        mass_ratio_tau = masses_model[2] / M_E

        x_e   = mu_e   / mass_ratio_e
        x_mu  = mu_mu  / mass_ratio_mu
        x_tau = mu_tau / mass_ratio_tau

        x_em = np.array([x_e, x_mu])
        if np.any(x_em <= 0) or np.any(~np.isfinite(x_em)) or (not np.isfinite(x_tau)):
            return 1e12

        # Profile C_g using e, μ only
        weights_g = 1.0 / self.sigma_g**2
        numerator_Cg = np.sum(self.g_targets * x_em * weights_g)
        denominator_Cg = np.sum(x_em**2 * weights_g)
        if denominator_Cg <= 0:
            return 1e12
        C_g_opt = numerator_Cg / denominator_Cg

        g_model_em = C_g_opt * x_em

        # χ²
        residuals_mass = (masses_model - self.m_targets) / self.sigma_mass
        chi2_mass = np.sum(residuals_mass**2)
        residuals_g = (g_model_em - self.g_targets) / self.sigma_g
        chi2_g = np.sum(residuals_g**2)

        return chi2_mass + chi2_g

    def fit_warm_start(self, x_prev, bounds, maxiter=200, seed=None, workers=8):
        if seed is not None:
            np.random.seed(seed)

        n_params = len(bounds)
        popsize = max(15 * n_params, 80)
        init_pop = np.zeros((popsize, n_params))
        init_pop[0, :] = x_prev

        n_jitter = min(15, popsize - 1)
        bound_ranges = np.array([b[1] - b[0] for b in bounds])
        jitter_std = 0.03 * bound_ranges

        for i in range(1, n_jitter + 1):
            jittered = x_prev + np.random.normal(0, jitter_std)
            jittered = np.clip(jittered, [b[0] for b in bounds], [b[1] for b in bounds])
            init_pop[i, :] = jittered

        for i in range(n_jitter + 1, popsize):
            for j, (lo, hi) in enumerate(bounds):
                init_pop[i, j] = np.random.uniform(lo, hi)

        result = differential_evolution(
            self.objective, bounds,
            maxiter=maxiter, init=init_pop, seed=seed,
            atol=1e-10, tol=1e-10, workers=workers,
            polish=True, updating='deferred',
        )

        x_best = result.x
        R_c_e, R_c_mu, R_c_tau, U_univ = x_best
        U_e = U_mu = U_tau = U_univ
        A_e = A_mu = A_tau = A_SAT

        # Recompute diagnostics
        E_e, _, _, _ = self.energy_calc.total_energy(R_c_e, U_e, A_e)
        E_mu, _, _, _ = self.energy_calc.total_energy(R_c_mu, U_mu, A_mu)
        E_tau, _, _, _ = self.energy_calc.total_energy(R_c_tau, U_tau, A_tau)
        energies = np.array([E_e, E_mu, E_tau])

        weights_mass = 1.0 / self.sigma_mass**2
        numerator_S = np.sum(self.m_targets * energies * weights_mass)
        denominator_S = np.sum(energies**2 * weights_mass)
        S_opt = numerator_S / denominator_S if denominator_S > 0 else np.nan
        masses_model = S_opt * energies

        R_shell_e   = R_c_e   + self.w
        R_shell_mu  = R_c_mu  + self.w
        R_shell_tau = R_c_tau + self.w

        mu_e   = compute_raw_moment(R_shell_e,   U_e)
        mu_mu  = compute_raw_moment(R_shell_mu,  U_mu)
        mu_tau = compute_raw_moment(R_shell_tau, U_tau)

        mass_ratio_e   = masses_model[0] / M_E
        mass_ratio_mu  = masses_model[1] / M_E
        mass_ratio_tau = masses_model[2] / M_E

        x_e   = mu_e   / mass_ratio_e
        x_mu  = mu_mu  / mass_ratio_mu
        x_tau = mu_tau / mass_ratio_tau
        x_em = np.array([x_e, x_mu])

        weights_g = 1.0 / self.sigma_g**2
        numerator_Cg = np.sum(self.g_targets * x_em * weights_g)
        denominator_Cg = np.sum(x_em**2 * weights_g)
        C_g_opt = numerator_Cg / denominator_Cg if denominator_Cg > 0 else np.nan
        g_tau_pred = C_g_opt * x_tau

        g_model_em = C_g_opt * x_em
        residuals_mass = (masses_model - self.m_targets) / self.sigma_mass
        chi2_mass = np.sum(residuals_mass**2)
        residuals_g = (g_model_em - self.g_targets) / self.sigma_g
        chi2_g = np.sum(residuals_g**2)

        return {
            "chi2_total": float(result.fun),
            "chi2_mass": float(chi2_mass),
            "chi2_g": float(chi2_g),
            "S_opt": float(S_opt),
            "C_g_opt": float(C_g_opt),
            "g_tau_pred": float(g_tau_pred),
            "parameters": {
                "R_c_e": float(R_c_e), "R_c_mu": float(R_c_mu), "R_c_tau": float(R_c_tau),
                "U_univ": float(U_univ),
            },
            "masses_model": [float(masses_model[0]), float(masses_model[1]), float(masses_model[2])],
        }

    def fit_multi_start(self, x_prev, bounds, maxiter=200, base_seed=0, workers=8, n_starts=5):
        results = []
        chi2s = []

        bound_ranges = np.array([b[1] - b[0] for b in bounds])
        jitter_std = 0.10 * bound_ranges

        for s in range(n_starts):
            seed = None if base_seed is None else int(base_seed + s)
            np.random.seed(seed)

            x0 = x_prev + np.random.normal(0, jitter_std)
            x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])

            r = self.fit_warm_start(x0, bounds, maxiter=maxiter, seed=seed, workers=workers)
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
        return best, results


# ========================================================================
# Single-β test
# ========================================================================

beta_test = 3.0
lam = calibrate_lambda(ETA_TARGET, beta_test, R_C_REF)

bounds = [
    (0.05, 3.0),   # R_c_e
    (0.05, 3.0),   # R_c_mu
    (0.05, 3.0),   # R_c_tau
    (0.05, 0.9),   # U_univ
]

x_init = np.array([0.25, 1.2, 0.6, 0.45])

print(f"Testing β = {beta_test:.2f}")
print(f"Initial guess: R_c = [{x_init[0]:.2f}, {x_init[1]:.2f}, {x_init[2]:.2f}], U = {x_init[3]:.2f}")
print()
print("Running 5-seed multi-start...")
sys.stdout.flush()

fitter = ThreeLeptonUniversalU(
    beta=beta_test, w=W, lam=lam,
    k_loc=K_LOCALIZATION, delta_v_fac=DELTA_V_FACTOR, p_env=P_ENVELOPE,
)

best, all_results = fitter.fit_multi_start(x_init, bounds, maxiter=200, base_seed=42, workers=8, n_starts=5)

print()
print("=" * 80)
print("MULTI-START RESULTS")
print("=" * 80)
print()
print(f"{'Seed':<6} {'χ²_total':<15} {'S_opt':<10} {'C_g_opt':<10} {'g_τ_pred':<10}")
print("-" * 60)
for i, r in enumerate(all_results):
    print(f"{i:<6} {r['chi2_total']:<15.6e} {r['S_opt']:<10.4f} {r['C_g_opt']:<10.2f} {r['g_tau_pred']:<10.4f}")
print("-" * 60)
print()

chi2_all = best["multi_start"]["chi2_all"]
chi2_spread = max(chi2_all) / min(chi2_all) if min(chi2_all) > 0 else np.inf

print(f"STABILITY ANALYSIS:")
print(f"  Best χ²:    {best['chi2_total']:.6e}")
print(f"  Median χ²:  {best['multi_start']['chi2_median']:.6e}")
print(f"  χ² spread:  {chi2_spread:.2f}× (min to max across seeds)")
print()

if chi2_spread < 10:
    print("✓ STABLE: All seeds converged to same basin")
elif chi2_spread < 100:
    print("~ PARTIAL: Some basin variation (acceptable)")
else:
    print("✗ UNSTABLE: Large basin-hopping (investigate)")
print()

print(f"BEST-FIT PARAMETERS:")
print(f"  R_c,e   = {best['parameters']['R_c_e']:.4f} fm")
print(f"  R_c,μ   = {best['parameters']['R_c_mu']:.4f} fm")
print(f"  R_c,τ   = {best['parameters']['R_c_tau']:.4f} fm")
print(f"  U_univ  = {best['parameters']['U_univ']:.4f}")
print()
print(f"  S_opt   = {best['S_opt']:.4f} MeV")
print(f"  C_g_opt = {best['C_g_opt']:.2f}")
print()

print(f"MASS FIT:")
print(f"  m_e:  {best['masses_model'][0]:.6f} MeV (target: {M_E:.6f})")
print(f"  m_μ:  {best['masses_model'][1]:.6f} MeV (target: {M_MU:.6f})")
print(f"  m_τ:  {best['masses_model'][2]:.2f} MeV (target: {M_TAU:.2f})")
print()

print(f"g-FACTOR PREDICTION:")
print(f"  g_τ_pred = {best['g_tau_pred']:.6f} (SM ≈ 2.002)")
if 1.8 < best['g_tau_pred'] < 2.3:
    print("  → Reasonable range ✓")
else:
    print("  → Outside expected range (investigate)")
print()

# Save test results
with open("results/V22/t3a_single_beta_test_results.json", "w") as f:
    json.dump({
        "beta": float(beta_test),
        "best_fit": best,
        "all_seeds": all_results,
        "stability": {
            "chi2_spread": float(chi2_spread),
            "stable": bool(chi2_spread < 10),
        },
    }, f, indent=2)

print("Test results saved: results/V22/t3a_single_beta_test_results.json")
print("=" * 80)

if chi2_spread < 10 and 1.8 < best['g_tau_pred'] < 2.3:
    print()
    print("✓ TEST PASSED: Proceed with full β scan")
else:
    print()
    print("⚠ REVIEW NEEDED: Check stability or g_τ before full scan")
