#!/usr/bin/env python3
"""
T2d: Refined β Scan Around Minimum

Purpose: Quantify β identifiability with fine resolution around apparent minimum

T2c coarse scan found β_min ≈ 1.70 with Δβ = 0.20
This scan refines to:
  - β ∈ [1.55, 1.95]
  - Δβ = 0.02 (21 points)
  - Compute finite-difference curvature d²χ²/dβ² at minimum

Deliverables:
  - χ²(β) profile with fine resolution
  - S_opt(β), C_g_opt(β) evolution
  - Numerical curvature at minimum
  - Confidence width estimate
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
G_E = 2.00231930436256
G_MU = 2.0023318414

# Configuration
W = 0.020
ETA_TARGET = 0.03
R_C_REF = 0.88
K_LOCALIZATION = 1.5
DELTA_V_FACTOR = 0.5
P_ENVELOPE = 6
A_SAT = 1.0

K_GEOMETRIC = 0.2
Q_CHARGE = 1.0

print("=" * 80)
print("T2d: REFINED β SCAN AROUND MINIMUM")
print("=" * 80)
print()
print("Objective: Quantify β identifiability with fine resolution")
print()
print("Configuration:")
print(f"  Cavitation: A = {A_SAT} (fixed)")
print(f"  β range: [1.55, 1.95], step = 0.02 (21 points)")
print(f"  DOF: 4 (R_c, U for e,μ)")
print()
sys.stdout.flush()


def compute_raw_moment(R_shell, U):
    return K_GEOMETRIC * Q_CHARGE * R_shell * U


class TwoLeptonCavitationSaturated:
    """Same as T2c"""

    def __init__(self, beta, w, lam, k_loc, delta_v_fac, p_env):
        self.beta = beta
        self.w = w
        self.lam = lam
        self.m_targets = np.array([M_E, M_MU])
        self.g_targets = np.array([G_E, G_MU])
        self.sigma_mass = 1e-4 * self.m_targets
        self.sigma_g = 1e-3
        self.energy_calc = LeptonEnergyLocalizedV1(
            beta=beta, w=w, lam=lam,
            k_localization=k_loc, delta_v_factor=delta_v_fac, p_envelope=p_env,
        )

    def objective(self, x):
        R_c_e, U_e = x[0:2]
        R_c_mu, U_mu = x[2:4]
        A_e = A_SAT
        A_mu = A_SAT

        E_e, _, _, _ = self.energy_calc.total_energy(R_c_e, U_e, A_e)
        E_mu, _, _, _ = self.energy_calc.total_energy(R_c_mu, U_mu, A_mu)
        energies = np.array([E_e, E_mu])

        if np.any(energies <= 0) or np.any(~np.isfinite(energies)):
            return 1e12

        # Profile S
        weights_mass = 1.0 / self.sigma_mass**2
        numerator_S = np.sum(self.m_targets * energies * weights_mass)
        denominator_S = np.sum(energies**2 * weights_mass)
        if denominator_S <= 0:
            return 1e12
        S_opt = numerator_S / denominator_S
        masses_model = S_opt * energies

        # Magnetic moments
        R_shell_e = R_c_e + self.w
        R_shell_mu = R_c_mu + self.w
        mu_e = compute_raw_moment(R_shell_e, U_e)
        mu_mu = compute_raw_moment(R_shell_mu, U_mu)

        # Profile C_g
        mass_ratio_e = masses_model[0] / M_E
        mass_ratio_mu = masses_model[1] / M_E
        x_e = mu_e / mass_ratio_e
        x_mu = mu_mu / mass_ratio_mu
        x_values = np.array([x_e, x_mu])

        if np.any(x_values <= 0) or np.any(~np.isfinite(x_values)):
            return 1e12

        weights_g = 1.0 / self.sigma_g**2
        numerator_Cg = np.sum(self.g_targets * x_values * weights_g)
        denominator_Cg = np.sum(x_values**2 * weights_g)
        if denominator_Cg <= 0:
            return 1e12
        C_g_opt = numerator_Cg / denominator_Cg
        g_model = C_g_opt * x_values

        # χ²
        residuals_mass = (masses_model - self.m_targets) / self.sigma_mass
        chi2_mass = np.sum(residuals_mass**2)
        residuals_g = (g_model - self.g_targets) / self.sigma_g
        chi2_g = np.sum(residuals_g**2)
        chi2_total = chi2_mass + chi2_g

        return chi2_total

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
        R_c_e, U_e = x_best[0:2]
        R_c_mu, U_mu = x_best[2:4]
        A_e = A_SAT
        A_mu = A_SAT

        E_e, _, _, _ = self.energy_calc.total_energy(R_c_e, U_e, A_e)
        E_mu, _, _, _ = self.energy_calc.total_energy(R_c_mu, U_mu, A_mu)
        energies = np.array([E_e, E_mu])

        # S profiling
        weights_mass = 1.0 / self.sigma_mass**2
        numerator_S = np.sum(self.m_targets * energies * weights_mass)
        denominator_S = np.sum(energies**2 * weights_mass)
        S_opt = numerator_S / denominator_S if denominator_S > 0 else np.nan
        masses_model = S_opt * energies

        # Magnetic moments
        R_shell_e = R_c_e + self.w
        R_shell_mu = R_c_mu + self.w
        mu_e = compute_raw_moment(R_shell_e, U_e)
        mu_mu = compute_raw_moment(R_shell_mu, U_mu)

        # C_g profiling
        mass_ratio_e = masses_model[0] / M_E
        mass_ratio_mu = masses_model[1] / M_E
        x_e = mu_e / mass_ratio_e
        x_mu = mu_mu / mass_ratio_mu
        x_values = np.array([x_e, x_mu])

        weights_g = 1.0 / self.sigma_g**2
        numerator_Cg = np.sum(self.g_targets * x_values * weights_g)
        denominator_Cg = np.sum(x_values**2 * weights_g)
        C_g_opt = numerator_Cg / denominator_Cg if denominator_Cg > 0 else np.nan

        g_model = C_g_opt * x_values

        # χ²
        residuals_mass = (masses_model - self.m_targets) / self.sigma_mass
        chi2_mass = np.sum(residuals_mass**2)
        residuals_g = (g_model - self.g_targets) / self.sigma_g
        chi2_g = np.sum(residuals_g**2)

        return {
            "chi2_total": float(result.fun),
            "chi2_mass": float(chi2_mass),
            "chi2_g": float(chi2_g),
            "S_opt": float(S_opt),
            "C_g_opt": float(C_g_opt),
            "parameters": {
                "electron": {"R_c": float(R_c_e), "U": float(U_e)},
                "muon": {"R_c": float(R_c_mu), "U": float(U_mu)},
            },
        }


# ========================================================================
# Refined β scan
# ========================================================================

beta_min = 1.55
beta_max = 1.95
beta_step = 0.02
beta_grid = np.arange(beta_min, beta_max + beta_step/2, beta_step)

print(f"β grid: {len(beta_grid)} points from {beta_min} to {beta_max} (step {beta_step})")
print()

bounds = [
    (0.05, 3.0),   # R_c_e
    (0.005, 0.6),  # U_e
    (0.02, 1.5),   # R_c_mu
    (0.01, 0.9),   # U_mu
]

# Start from T2c best at β=1.70
x_init = np.array([0.5912, 0.0081, 1.4134, 0.7182])

results_scan = []

print("Running refined scan...")
print("-" * 80)
print(f"{'β':<8} {'χ²_total':<15} {'S_opt':<12} {'C_g_opt':<12}")
print("-" * 80)
sys.stdout.flush()

x_prev = x_init.copy()

for beta in beta_grid:
    lam = calibrate_lambda(ETA_TARGET, beta, R_C_REF)

    fitter = TwoLeptonCavitationSaturated(
        beta=beta, w=W, lam=lam,
        k_loc=K_LOCALIZATION, delta_v_fac=DELTA_V_FACTOR, p_env=P_ENVELOPE,
    )

    result = fitter.fit_warm_start(x_prev, bounds, maxiter=200, seed=42, workers=8)

    results_scan.append({"beta": float(beta), "result": result})

    print(f"{beta:<8.2f} {result['chi2_total']:<15.6e} {result['S_opt']:<12.4f} {result['C_g_opt']:<12.2f}")
    sys.stdout.flush()

    x_prev = np.array([
        result["parameters"]["electron"]["R_c"], result["parameters"]["electron"]["U"],
        result["parameters"]["muon"]["R_c"], result["parameters"]["muon"]["U"],
    ])

print("-" * 80)
print()

# Find minimum
chi2_values = np.array([r["result"]["chi2_total"] for r in results_scan])
idx_min = np.argmin(chi2_values)
beta_best = results_scan[idx_min]["beta"]
chi2_best = chi2_values[idx_min]

print("=" * 80)
print("REFINED MINIMUM")
print("=" * 80)
print()
print(f"β_min = {beta_best:.2f}")
print(f"χ²_min = {chi2_best:.6e}")
print(f"S_opt = {results_scan[idx_min]['result']['S_opt']:.4f} MeV")
print(f"C_g_opt = {results_scan[idx_min]['result']['C_g_opt']:.2f}")
print()

# Compute finite-difference curvature
if idx_min > 0 and idx_min < len(beta_grid) - 1:
    chi2_prev = chi2_values[idx_min - 1]
    chi2_next = chi2_values[idx_min + 1]

    # Second derivative: f''(x) ≈ [f(x+h) - 2f(x) + f(x-h)] / h²
    d2chi2_dbeta2 = (chi2_next - 2*chi2_best + chi2_prev) / beta_step**2

    print(f"CURVATURE (finite difference):")
    print(f"  d²χ²/dβ² ≈ {d2chi2_dbeta2:.6e}")
    print()

    # Confidence width (Δβ where χ² increases by 1)
    # Approximation: χ²(β) ≈ χ²_min + ½ d²χ²/dβ² (β - β_min)²
    # Δχ² = 1 → ½ d²χ²/dβ² Δβ² = 1 → Δβ = √(2 / d²χ²/dβ²)
    if d2chi2_dbeta2 > 0:
        delta_beta_1sigma = np.sqrt(2.0 / d2chi2_dbeta2)
        print(f"  Confidence width (Δχ² = 1): Δβ ≈ {delta_beta_1sigma:.4f}")
        print(f"  β ∈ [{beta_best - delta_beta_1sigma:.2f}, {beta_best + delta_beta_1sigma:.2f}] (68% CL, approx)")
    else:
        print(f"  WARNING: Curvature ≤ 0, minimum may not be well-defined")
else:
    print("WARNING: Minimum at edge of scan range, cannot compute curvature")

print()

# CV statistics
S_opts = [r["result"]["S_opt"] for r in results_scan]
Cg_opts = [r["result"]["C_g_opt"] for r in results_scan]

CV_S = np.std(S_opts) / np.mean(S_opts) * 100
CV_Cg = np.std(Cg_opts) / np.mean(Cg_opts) * 100

print(f"CV OVER REFINED RANGE:")
print(f"  CV(S_opt):   {CV_S:.1f}%")
print(f"  CV(C_g_opt): {CV_Cg:.1f}%")
print()

# Save
results_dict = {
    "beta_grid": beta_grid.tolist(),
    "beta_min": float(beta_best),
    "chi2_min": float(chi2_best),
    "curvature_d2chi2_dbeta2": float(d2chi2_dbeta2) if idx_min > 0 and idx_min < len(beta_grid) - 1 else None,
    "confidence_width_delta_beta": float(delta_beta_1sigma) if (idx_min > 0 and idx_min < len(beta_grid) - 1 and d2chi2_dbeta2 > 0) else None,
    "CV_S": float(CV_S),
    "CV_Cg": float(CV_Cg),
    "scan_results": results_scan,
}

with open("results/V22/t2d_refined_beta_scan_results.json", "w") as f:
    json.dump(results_dict, f, indent=2)

print("Results saved: results/V22/t2d_refined_beta_scan_results.json")
print("=" * 80)
