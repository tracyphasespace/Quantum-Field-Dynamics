#!/usr/bin/env python3
"""
T3a: Three-Lepton Fit with Universal Boundary Speed

Critical constraints:
  1. A = 1.0 (cavitation saturation) for ALL leptons
  2. U_e = U_μ = U_τ = U_univ (universal boundary speed)

Rationale:
  - T2d refined scan showed β UNIDENTIFIED despite cavitation (Δβ ≈ 33)
  - 4 DOF (R_c, U per lepton) + 2 profiled scales vs 4 observables → exactly solvable
  - Universal U adds τ mass constraint WITHOUT adding U DOF
  - Tests hypothesis: rim circulation speed is universal lepton property

Observable set:
  - Masses: m_e, m_μ, m_τ (3 hard constraints)
  - g-factors: g_e, g_μ (2 hard constraints, profiled C_g)
  - g_τ: PREDICTED (not used in χ²) - falsifiable without using theory as data

DOF: 4 parameters
  - R_c,e, R_c,μ, R_c,τ (3 core radii)
  - U_univ (1 shared boundary speed)

Expected outcome:
  PASS: β identifiable via curvature analysis (Δβ finite, reasonable)
  FAIL: β still flat → need additional restriction or stability constraint
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
M_TAU = 1776.86  # PDG 2024

# g-factor targets (dimensionless)
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

# CAVITATION SATURATION
A_SAT = 1.0  # ρ_min = ρ_vac - A = 0

print("=" * 80)
print("T3a: THREE-LEPTON FIT WITH UNIVERSAL BOUNDARY SPEED")
print("=" * 80)
print()
print("CRITICAL CONSTRAINTS:")
print(f"  A_e = A_μ = A_τ = {A_SAT} (cavitation saturation)")
print(f"  U_e = U_μ = U_τ = U_univ (universal boundary speed)")
print()
print("DOF: 4 parameters")
print("  Free: R_c,e, R_c,μ, R_c,τ, U_univ")
print()
print("Observable set:")
print(f"  Masses:    m_e = {M_E:.3f} MeV, m_μ = {M_MU:.1f} MeV, m_τ = {M_TAU:.2f} MeV")
print(f"  g-factors: g_e = {G_E:.10f}, g_μ = {G_MU:.10f}")
print(f"  g_τ:       PREDICTED (not in χ²)")
print()
print("Profiling:")
print("  S_opt:   profiled from 3 mass terms (e, μ, τ)")
print("  C_g_opt: profiled from 2 g terms (e, μ only)")
print()
print("Multi-start: 5 seeds per β (branch control)")
print()
sys.stdout.flush()


def compute_raw_moment(R_shell, U):
    """Raw magnetic moment: μ = k × Q × R_shell × U"""
    return K_GEOMETRIC * Q_CHARGE * R_shell * U


class ThreeLeptonUniversalU:
    """Fit e, μ, τ with A=1.0 and universal U"""

    def __init__(self, beta, w, lam, k_loc, delta_v_fac, p_env):
        self.beta = beta
        self.w = w
        self.lam = lam

        # Targets
        self.m_targets = np.array([M_E, M_MU, M_TAU])
        self.g_targets = np.array([G_E, G_MU])  # e, μ only

        # Uncertainties
        self.sigma_mass = np.array([1e-4 * M_E, 1e-4 * M_MU, 0.12])  # MeV
        self.sigma_g = 1e-3

        # Energy calculator
        self.energy_calc = LeptonEnergyLocalizedV1(
            beta=beta,
            w=w,
            lam=lam,
            k_localization=k_loc,
            delta_v_factor=delta_v_fac,
            p_envelope=p_env,
        )

    def objective(self, x):
        """
        χ² with FIXED A = 1.0, universal U, and dual profiling (S, C_g)

        Parameters (4):
          x[0] = R_c_e
          x[1] = R_c_mu
          x[2] = R_c_tau
          x[3] = U_univ

        A = 1.0 for all leptons (cavitation saturation)
        U is universal across leptons (structural restriction)
        """
        R_c_e, R_c_mu, R_c_tau, U_univ = x
        U_e = U_mu = U_tau = U_univ

        # FIXED amplitude at cavitation limit
        A_e = A_mu = A_tau = A_SAT

        # Energies
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

        # Magnetic moments (raw proxy)
        R_shell_e   = R_c_e   + self.w
        R_shell_mu  = R_c_mu  + self.w
        R_shell_tau = R_c_tau + self.w

        mu_e   = compute_raw_moment(R_shell_e,   U_e)
        mu_mu  = compute_raw_moment(R_shell_mu,  U_mu)
        mu_tau = compute_raw_moment(R_shell_tau, U_tau)

        # Raw g-proxies x = μ / (m_l/m_e)
        mass_ratio_e   = masses_model[0] / M_E
        mass_ratio_mu  = masses_model[1] / M_E
        mass_ratio_tau = masses_model[2] / M_E

        x_e   = mu_e   / mass_ratio_e
        x_mu  = mu_mu  / mass_ratio_mu
        x_tau = mu_tau / mass_ratio_tau  # predicted only

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
        g_tau_pred = C_g_opt * x_tau

        # χ²
        residuals_mass = (masses_model - self.m_targets) / self.sigma_mass
        chi2_mass = np.sum(residuals_mass**2)

        residuals_g = (g_model_em - self.g_targets) / self.sigma_g
        chi2_g = np.sum(residuals_g**2)

        chi2_total = chi2_mass + chi2_g
        return chi2_total

    def fit_warm_start(self, x_prev, bounds, maxiter=200, seed=None, workers=8):
        """Fit with warm start (4 parameters: R_c_e, R_c_mu, R_c_tau, U_univ)"""
        if seed is not None:
            np.random.seed(seed)

        n_params = len(bounds)
        popsize = max(15 * n_params, 80)

        # Seed population
        init_pop = np.zeros((popsize, n_params))
        init_pop[0, :] = x_prev

        # Jittered
        n_jitter = min(15, popsize - 1)
        bound_ranges = np.array([b[1] - b[0] for b in bounds])
        jitter_std = 0.03 * bound_ranges

        for i in range(1, n_jitter + 1):
            jittered = x_prev + np.random.normal(0, jitter_std)
            jittered = np.clip(jittered, [b[0] for b in bounds], [b[1] for b in bounds])
            init_pop[i, :] = jittered

        # Random
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

        # Extract solution
        x_best = result.x
        R_c_e, R_c_mu, R_c_tau, U_univ = x_best
        U_e = U_mu = U_tau = U_univ

        # FIXED amplitudes
        A_e = A_mu = A_tau = A_SAT

        # Recompute diagnostics
        E_e, E_circ_e, E_stab_e, E_grad_e = self.energy_calc.total_energy(R_c_e, U_e, A_e)
        E_mu, E_circ_mu, E_stab_mu, E_grad_mu = self.energy_calc.total_energy(R_c_mu, U_mu, A_mu)
        E_tau, E_circ_tau, E_stab_tau, E_grad_tau = self.energy_calc.total_energy(R_c_tau, U_tau, A_tau)

        energies = np.array([E_e, E_mu, E_tau])

        # S profiling
        weights_mass = 1.0 / self.sigma_mass**2
        numerator_S = np.sum(self.m_targets * energies * weights_mass)
        denominator_S = np.sum(energies**2 * weights_mass)
        S_opt = numerator_S / denominator_S if denominator_S > 0 else np.nan

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

        # C_g profiling (e, μ only)
        weights_g = 1.0 / self.sigma_g**2
        numerator_Cg = np.sum(self.g_targets * x_em * weights_g)
        denominator_Cg = np.sum(x_em**2 * weights_g)
        C_g_opt = numerator_Cg / denominator_Cg if denominator_Cg > 0 else np.nan

        g_model_em = C_g_opt * x_em
        g_tau_pred = C_g_opt * x_tau

        g_e = g_model_em[0]
        g_mu = g_model_em[1]

        # χ²
        residuals_mass = (masses_model - self.m_targets) / self.sigma_mass
        chi2_mass = np.sum(residuals_mass**2)

        residuals_g = (g_model_em - self.g_targets) / self.sigma_g
        chi2_g = np.sum(residuals_g**2)

        # Helper for bound hits
        def hit(val, lo, hi, tol=1e-6):
            if abs(val - lo) < tol: return "lower"
            if abs(val - hi) < tol: return "upper"
            return "none"

        bounds_hit = {
            "electron": {"R_c": hit(R_c_e, bounds[0][0], bounds[0][1]), "U": hit(U_univ, bounds[3][0], bounds[3][1])},
            "muon":     {"R_c": hit(R_c_mu, bounds[1][0], bounds[1][1]), "U": hit(U_univ, bounds[3][0], bounds[3][1])},
            "tau":      {"R_c": hit(R_c_tau, bounds[2][0], bounds[2][1]), "U": hit(U_univ, bounds[3][0], bounds[3][1])},
        }

        return {
            "chi2_total": float(result.fun),
            "chi2_mass": float(chi2_mass),
            "chi2_g": float(chi2_g),
            "S_opt": float(S_opt),
            "C_g_opt": float(C_g_opt),
            "g_tau_pred": float(g_tau_pred),
            "parameters": {
                "electron": {"R_c": float(R_c_e), "U": float(U_univ), "A": float(A_e)},
                "muon":     {"R_c": float(R_c_mu), "U": float(U_univ), "A": float(A_mu)},
                "tau":      {"R_c": float(R_c_tau), "U": float(U_univ), "A": float(A_tau)},
            },
            "energies": {
                "electron": {"E_total": float(E_e), "E_circ": float(E_circ_e), "E_stab": float(E_stab_e), "E_grad": float(E_grad_e)},
                "muon":     {"E_total": float(E_mu), "E_circ": float(E_circ_mu), "E_stab": float(E_stab_mu), "E_grad": float(E_grad_mu)},
                "tau":      {"E_total": float(E_tau), "E_circ": float(E_circ_tau), "E_stab": float(E_stab_tau), "E_grad": float(E_grad_tau)},
            },
            "masses_model": {
                "electron": float(masses_model[0]),
                "muon": float(masses_model[1]),
                "tau": float(masses_model[2])
            },
            "g_model": {"electron": float(g_e), "muon": float(g_mu)},
            "g_residuals": {"electron": float(g_e - G_E), "muon": float(g_mu - G_MU)},
            "bounds_hit": bounds_hit,
        }

    def fit_multi_start(self, x_prev, bounds, maxiter=200, base_seed=0, workers=8, n_starts=5):
        """Run DE n_starts times with different seeds / jitters; return best + median χ²."""
        results = []
        chi2s = []

        # Build a slightly larger jitter for multi-start diversification
        bound_ranges = np.array([b[1] - b[0] for b in bounds])
        jitter_std = 0.10 * bound_ranges  # stronger than warm-start jitter

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
        return best


# ========================================================================
# β scan with 3 leptons, universal U, multi-start
# ========================================================================

beta_range = (1.5, 3.4)
beta_step = 0.2
beta_grid = np.arange(beta_range[0], beta_range[1] + beta_step/2, beta_step)

print(f"β grid: {len(beta_grid)} points")
print(f"  {beta_grid}")
print()

# Bounds (4 parameters: R_c_e, R_c_mu, R_c_tau, U_univ)
bounds = [
    (0.05, 3.0),   # R_c_e
    (0.05, 3.0),   # R_c_mu
    (0.05, 3.0),   # R_c_tau
    (0.05, 0.9),   # U_univ
]

# Initialize [R_c_e, R_c_mu, R_c_tau, U_univ]
x_init = np.array([0.25, 1.2, 0.6, 0.45])

idx_start = np.argmin(np.abs(beta_grid - 3.0))
beta_start = beta_grid[idx_start]

print(f"Starting from β = {beta_start:.2f}")
print()

results_scan = []

print("Running β scan with 3 leptons, universal U, multi-start (n=5)...")
print("-" * 140)
print(f"{'β':<8} {'χ²_tot':<12} {'χ²_med':<12} {'χ²_mass':<12} {'χ²_g':<12} {'S_opt':<10} {'C_g':<8} {'R_c_e':<8} {'R_c_μ':<8} {'R_c_τ':<8} {'U':<8} {'g_τ':<8}")
print("-" * 140)
sys.stdout.flush()

x_prev = x_init.copy()

# Ascending
for i in range(idx_start, len(beta_grid)):
    beta = beta_grid[i]
    lam = calibrate_lambda(ETA_TARGET, beta, R_C_REF)

    fitter = ThreeLeptonUniversalU(
        beta=beta, w=W, lam=lam,
        k_loc=K_LOCALIZATION, delta_v_fac=DELTA_V_FACTOR, p_env=P_ENVELOPE,
    )

    result = fitter.fit_multi_start(x_prev, bounds, maxiter=200, base_seed=42, workers=8, n_starts=5)

    results_scan.append({"beta": float(beta), "result": result})

    chi2_med = result["multi_start"]["chi2_median"]
    print(f"{beta:<8.2f} {result['chi2_total']:<12.6e} {chi2_med:<12.6e} {result['chi2_mass']:<12.6e} {result['chi2_g']:<12.6e} " +
          f"{result['S_opt']:<10.4f} {result['C_g_opt']:<8.2f} " +
          f"{result['parameters']['electron']['R_c']:<8.4f} {result['parameters']['muon']['R_c']:<8.4f} {result['parameters']['tau']['R_c']:<8.4f} " +
          f"{result['parameters']['electron']['U']:<8.4f} {result['g_tau_pred']:<8.4f}")
    sys.stdout.flush()

    x_prev = np.array([
        result["parameters"]["electron"]["R_c"],
        result["parameters"]["muon"]["R_c"],
        result["parameters"]["tau"]["R_c"],
        result["parameters"]["electron"]["U"],  # U_univ
    ])

# Descending
x_prev = x_init.copy()

for i in range(idx_start - 1, -1, -1):
    beta = beta_grid[i]
    lam = calibrate_lambda(ETA_TARGET, beta, R_C_REF)

    fitter = ThreeLeptonUniversalU(
        beta=beta, w=W, lam=lam,
        k_loc=K_LOCALIZATION, delta_v_fac=DELTA_V_FACTOR, p_env=P_ENVELOPE,
    )

    result = fitter.fit_multi_start(x_prev, bounds, maxiter=200, base_seed=42, workers=8, n_starts=5)

    results_scan.append({"beta": float(beta), "result": result})

    chi2_med = result["multi_start"]["chi2_median"]
    print(f"{beta:<8.2f} {result['chi2_total']:<12.6e} {chi2_med:<12.6e} {result['chi2_mass']:<12.6e} {result['chi2_g']:<12.6e} " +
          f"{result['S_opt']:<10.4f} {result['C_g_opt']:<8.2f} " +
          f"{result['parameters']['electron']['R_c']:<8.4f} {result['parameters']['muon']['R_c']:<8.4f} {result['parameters']['tau']['R_c']:<8.4f} " +
          f"{result['parameters']['electron']['U']:<8.4f} {result['g_tau_pred']:<8.4f}")
    sys.stdout.flush()

    x_prev = np.array([
        result["parameters"]["electron"]["R_c"],
        result["parameters"]["muon"]["R_c"],
        result["parameters"]["tau"]["R_c"],
        result["parameters"]["electron"]["U"],  # U_univ
    ])

print("-" * 140)
print()

# Sort by β
results_scan.sort(key=lambda x: x["beta"])

# Find minimum
best = min(results_scan, key=lambda x: x["result"]["chi2_total"])

print("=" * 80)
print("RESULTS")
print("=" * 80)
print()
print(f"Best β: {best['beta']:.2f}")
print(f"  χ²_total = {best['result']['chi2_total']:.6e}")
print(f"  χ²_median (multi-start) = {best['result']['multi_start']['chi2_median']:.6e}")
print(f"  S_opt    = {best['result']['S_opt']:.4f} MeV")
print(f"  C_g_opt  = {best['result']['C_g_opt']:.4f}")
print(f"  g_τ_pred = {best['result']['g_tau_pred']:.4f}")
print()

# CV statistics
S_opts = [r["result"]["S_opt"] for r in results_scan]
Cg_opts = [r["result"]["C_g_opt"] for r in results_scan]

CV_S = np.std(S_opts) / np.mean(S_opts) * 100
CV_Cg = np.std(Cg_opts) / np.mean(Cg_opts) * 100

print(f"COEFFICIENT OF VARIATION:")
print(f"  CV(S_opt):   {CV_S:.1f}% (T2d: 60%)")
print(f"  CV(C_g_opt): {CV_Cg:.1f}% (T2d: 22%)")
print()

chi2_values = [r["result"]["chi2_total"] for r in results_scan]
chi2_range = max(chi2_values) / min(chi2_values) if min(chi2_values) > 0 else np.inf

print(f"χ² variation: {min(chi2_values):.6e} to {max(chi2_values):.6e} ({chi2_range:.1f}×)")
print()

# Multi-start stability check
chi2_spreads = [max(r["result"]["multi_start"]["chi2_all"]) / min(r["result"]["multi_start"]["chi2_all"])
                for r in results_scan if min(r["result"]["multi_start"]["chi2_all"]) > 0]
avg_spread = np.mean(chi2_spreads) if chi2_spreads else np.inf

print(f"Multi-start stability:")
print(f"  Avg χ² spread across seeds: {avg_spread:.1f}×")
if avg_spread < 10:
    print(f"  → STABLE basins (good)")
else:
    print(f"  → Basin-hopping still present (caution)")
print()

# Assessment
print("=" * 80)
print("IDENTIFIABILITY ASSESSMENT")
print("=" * 80)
print()

if CV_S < 30 and CV_Cg < 20 and avg_spread < 10:
    print("✓ PASS: Universal U constraint stabilized fit - β IDENTIFIABLE")
    print(f"  CV improvements: S ({CV_S:.1f}% vs 60%), C_g ({CV_Cg:.1f}% vs 22%)")
    print(f"  Multi-start stability achieved")
elif CV_S < 50 and CV_Cg < 30:
    print("~ PARTIAL: Improvement but β weakly constrained")
    print(f"  Recommend: refined β scan + curvature analysis")
else:
    print("✗ FAIL: β remains unidentified even with universal U")
    print()
    print("NEXT: Add stability/curvature constraint or tighten bounds")

print()

# Save
results_dict = {
    "constraints": {
        "cavitation_amplitude": A_SAT,
        "universal_U": True,
    },
    "beta_grid": beta_grid.tolist(),
    "CV_S": float(CV_S),
    "CV_Cg": float(CV_Cg),
    "chi2_range": float(chi2_range),
    "multi_start_stability": float(avg_spread),
    "scan_results": results_scan,
    "best_fit": best["result"],
}

with open("results/V22/t3a_three_lepton_universal_U_results.json", "w") as f:
    json.dump(results_dict, f, indent=2)

print("Results saved: results/V22/t3a_three_lepton_universal_U_results.json")
print("=" * 80)
