#!/usr/bin/env python3
"""
T2b: Mass + g-factor with PROFILED g-normalization

Critical improvement over T2':
  - Profile C_g (g-normalization) analytically, just like S_opt
  - This separates "bad calibration" from "bad physics"

If χ²_g collapses → problem was units/calibration portability
If χ²_g stays huge → shape mismatch (relative scaling wrong)

Procedure:
  1. Compute energies E_e, E_μ
  2. Profile S from mass terms: S_opt = Σ[m·E/σ²] / Σ[E²/σ²]
  3. Compute model masses: m_model = S_opt × E
  4. Compute raw g-proxies: x_ℓ = μ_ℓ / (m_model / m_e)
     where μ_ℓ = k × Q × R_shell × U
  5. Profile C_g from g terms: C_g_opt = Σ[g_target·x/σ²] / Σ[x²/σ²]
  6. Compute model g-factors: g_model = C_g_opt × x_ℓ
  7. χ² = χ²_mass + χ²_g
"""

import numpy as np
from scipy.optimize import differential_evolution
from lepton_energy_localized_v1 import LeptonEnergyLocalizedV1
from profile_likelihood_boundary_layer import calibrate_lambda
import json
import sys

# Physical constants
M_E = 0.511  # MeV
M_MU = 105.7  # MeV

# g-factor targets
G_E = 2.00231930436256
G_MU = 2.0023318414

# Magnetic moment proxy parameters
K_GEOMETRIC = 0.2
Q_CHARGE = 1.0

# Configuration
W = 0.020
ETA_TARGET = 0.03
R_C_REF = 0.88
K_LOCALIZATION = 1.5
DELTA_V_FACTOR = 0.5
P_ENVELOPE = 6

print("=" * 80)
print("T2b: MASS + g-FACTOR WITH PROFILED NORMALIZATION")
print("=" * 80)
print()
print("Objective: Separate calibration mismatch from structural physics conflict")
print()
print("Key change from T2':")
print("  - C_g (g-normalization) is PROFILED, not fixed at 948")
print("  - Profile: C_g_opt = Σ[g_target × x / σ²] / Σ[x² / σ²]")
print("  - where x_ℓ = (R_shell × U) / (m_model / m_e)")
print()
print("Observable set:")
print(f"  Masses:    m_e = {M_E:.3f} MeV, m_μ = {M_MU:.1f} MeV")
print(f"  g-factors: g_e = {G_E:.10f}, g_μ = {G_MU:.10f}")
print()
print("Weighting:")
print(f"  σ_m / m = 1e-4 (0.01%)")
print(f"  σ_g = 1e-3 (absolute)")
print()
sys.stdout.flush()


def compute_raw_moment(R_shell, U):
    """Raw magnetic moment proxy: μ = k × Q × R_shell × U"""
    mu = K_GEOMETRIC * Q_CHARGE * R_shell * U
    return mu


class TwoLeptonMassGFitterProfiled:
    """Fit e,μ with mass + g-factor, profiling BOTH S and C_g"""

    def __init__(self, beta, w, lam, k_loc, delta_v_fac, p_env):
        self.beta = beta
        self.w = w
        self.lam = lam

        # Targets
        self.m_targets = np.array([M_E, M_MU])
        self.g_targets = np.array([G_E, G_MU])

        # Uncertainties
        self.sigma_mass = 1e-4 * self.m_targets
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
        Combined mass + g-factor χ² with DUAL profiling (S and C_g)

        Parameters (6):
          x[0:3] = R_c, U, A for electron
          x[3:6] = R_c, U, A for muon

        Procedure:
          1. Compute E_e, E_μ
          2. Profile S from mass: S_opt = Σ[m·E/σ_m²] / Σ[E²/σ_m²]
          3. Compute m_model = S_opt × E
          4. Compute raw g-proxies: x_ℓ = μ_ℓ / (m_model / m_e)
          5. Profile C_g from g: C_g_opt = Σ[g_target·x/σ_g²] / Σ[x²/σ_g²]
          6. Compute g_model = C_g_opt × x
          7. χ² = Σ[(m_model - m_target)²/σ_m²] + Σ[(g_model - g_target)²/σ_g²]
        """
        R_c_e, U_e, A_e = x[0:3]
        R_c_mu, U_mu, A_mu = x[3:6]

        # Energies
        E_e, _, _, _ = self.energy_calc.total_energy(R_c_e, U_e, A_e)
        E_mu, _, _, _ = self.energy_calc.total_energy(R_c_mu, U_mu, A_mu)

        energies = np.array([E_e, E_mu])

        if np.any(energies <= 0) or np.any(~np.isfinite(energies)):
            return 1e12

        # Profile S (from mass terms)
        weights_mass = 1.0 / self.sigma_mass**2
        numerator_S = np.sum(self.m_targets * energies * weights_mass)
        denominator_S = np.sum(energies**2 * weights_mass)

        if denominator_S <= 0:
            return 1e12

        S_opt = numerator_S / denominator_S
        masses_model = S_opt * energies

        # Raw magnetic moments (use R_shell)
        R_shell_e = R_c_e + self.w
        R_shell_mu = R_c_mu + self.w

        mu_e = compute_raw_moment(R_shell_e, U_e)
        mu_mu = compute_raw_moment(R_shell_mu, U_mu)

        # Raw g-proxies (before normalization)
        mass_ratio_e = masses_model[0] / M_E  # Should be ≈ 1
        mass_ratio_mu = masses_model[1] / M_E  # Should be ≈ 206.85

        x_e = mu_e / mass_ratio_e
        x_mu = mu_mu / mass_ratio_mu

        x_values = np.array([x_e, x_mu])

        if np.any(x_values <= 0) or np.any(~np.isfinite(x_values)):
            return 1e12

        # Profile C_g (from g terms)
        weights_g = 1.0 / self.sigma_g**2
        numerator_Cg = np.sum(self.g_targets * x_values * weights_g)
        denominator_Cg = np.sum(x_values**2 * weights_g)

        if denominator_Cg <= 0:
            return 1e12

        C_g_opt = numerator_Cg / denominator_Cg

        # Model g-factors (after profiling)
        g_model = C_g_opt * x_values

        # χ²_mass
        residuals_mass = (masses_model - self.m_targets) / self.sigma_mass
        chi2_mass = np.sum(residuals_mass**2)

        # χ²_g
        residuals_g = (g_model - self.g_targets) / self.sigma_g
        chi2_g = np.sum(residuals_g**2)

        # Total
        chi2_total = chi2_mass + chi2_g

        return chi2_total

    def fit_warm_start(self, x_prev, bounds, maxiter=200, seed=None, workers=8):
        """Fit with warm start"""
        if seed is not None:
            np.random.seed(seed)

        n_params = len(bounds)
        popsize = max(15 * n_params, 100)

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

        # Extract solution and recompute all diagnostics
        x_best = result.x
        R_c_e, U_e, A_e = x_best[0:3]
        R_c_mu, U_mu, A_mu = x_best[3:6]

        E_e, E_circ_e, E_stab_e, E_grad_e = self.energy_calc.total_energy(R_c_e, U_e, A_e)
        E_mu, E_circ_mu, E_stab_mu, E_grad_mu = self.energy_calc.total_energy(R_c_mu, U_mu, A_mu)

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

        # Raw g-proxies
        mass_ratio_e = masses_model[0] / M_E
        mass_ratio_mu = masses_model[1] / M_E

        x_e = mu_e / mass_ratio_e
        x_mu = mu_mu / mass_ratio_mu

        x_values = np.array([x_e, x_mu])

        # C_g profiling
        weights_g = 1.0 / self.sigma_g**2
        numerator_Cg = np.sum(self.g_targets * x_values * weights_g)
        denominator_Cg = np.sum(x_values**2 * weights_g)
        C_g_opt = numerator_Cg / denominator_Cg if denominator_Cg > 0 else np.nan

        # Model g-factors
        g_model = C_g_opt * x_values
        g_e = g_model[0]
        g_mu = g_model[1]

        # χ² components
        residuals_mass = (masses_model - self.m_targets) / self.sigma_mass
        chi2_mass = np.sum(residuals_mass**2)

        residuals_g = (g_model - self.g_targets) / self.sigma_g
        chi2_g = np.sum(residuals_g**2)

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
            "chi2_total": float(result.fun),
            "chi2_mass": float(chi2_mass),
            "chi2_g": float(chi2_g),
            "S_opt": float(S_opt),
            "C_g_opt": float(C_g_opt),  # NEW: profiled g-normalization
            "parameters": {
                "electron": {"R_c": float(R_c_e), "U": float(U_e), "A": float(A_e)},
                "muon": {"R_c": float(R_c_mu), "U": float(U_mu), "A": float(A_mu)},
            },
            "energies": {
                "electron": {"E_total": float(E_e), "E_circ": float(E_circ_e), "E_stab": float(E_stab_e), "E_grad": float(E_grad_e)},
                "muon": {"E_total": float(E_mu), "E_circ": float(E_circ_mu), "E_stab": float(E_stab_mu), "E_grad": float(E_grad_mu)},
            },
            "masses_model": {"electron": float(masses_model[0]), "muon": float(masses_model[1])},
            "masses_target": {"electron": float(M_E), "muon": float(M_MU)},
            "magnetic_moments": {"electron": float(mu_e), "muon": float(mu_mu)},
            "g_proxy_raw": {"electron": float(x_e), "muon": float(x_mu)},  # Before C_g scaling
            "g_model": {"electron": float(g_e), "muon": float(g_mu)},  # After C_g scaling
            "g_target": {"electron": float(G_E), "muon": float(G_MU)},
            "g_residuals": {"electron": float(g_e - G_E), "muon": float(g_mu - G_MU)},
            "R_shell": {"electron": float(R_shell_e), "muon": float(R_shell_mu)},
            "mass_ratios": {"electron": float(mass_ratio_e), "muon": float(mass_ratio_mu)},
            "bounds_hit": bounds_hit,
        }


# ========================================================================
# Coarse β scan with DUAL profiling
# ========================================================================

beta_range = (1.5, 3.4)
beta_step = 0.2
beta_grid = np.arange(beta_range[0], beta_range[1] + beta_step/2, beta_step)

print(f"β grid: {len(beta_grid)} points")
print(f"  {beta_grid}")
print()

bounds = [
    (0.05, 3.0),
    (0.005, 0.6),
    (0.05, 0.999),
    (0.02, 1.5),
    (0.01, 0.9),
    (0.05, 0.999),
]

# Start from T1c best
x_init = np.array([0.233068, 0.387321, 0.517347, 1.280013, 0.507202, 0.273778])
idx_start = np.argmin(np.abs(beta_grid - 3.3))
beta_start = beta_grid[idx_start]

print(f"Starting from β = {beta_start:.2f}")
print()

results_scan = []

print("Running β scan with DUAL profiling (S and C_g)...")
print("-" * 140)
print(f"{'β':<8} {'χ²_total':<13} {'χ²_mass':<13} {'χ²_g':<13} {'S_opt':<10} {'C_g_opt':<10} {'g_e_res':<12} {'g_μ_res':<12} {'Hits':<6}")
print("-" * 140)
sys.stdout.flush()

x_prev = x_init.copy()

# Ascending scan
for i in range(idx_start, len(beta_grid)):
    beta = beta_grid[i]
    lam = calibrate_lambda(ETA_TARGET, beta, R_C_REF)

    fitter = TwoLeptonMassGFitterProfiled(
        beta=beta, w=W, lam=lam,
        k_loc=K_LOCALIZATION, delta_v_fac=DELTA_V_FACTOR, p_env=P_ENVELOPE,
    )

    result = fitter.fit_warm_start(x_prev, bounds, maxiter=200, seed=42, workers=8)

    chi2_total = result["chi2_total"]
    chi2_mass = result["chi2_mass"]
    chi2_g = result["chi2_g"]
    S_opt = result["S_opt"]
    C_g_opt = result["C_g_opt"]
    g_e_res = result["g_residuals"]["electron"]
    g_mu_res = result["g_residuals"]["muon"]

    total_hits = sum(1 for v in result["bounds_hit"]["electron"].values() if v != "none") + \
                 sum(1 for v in result["bounds_hit"]["muon"].values() if v != "none")

    results_scan.append({"beta": float(beta), "result": result})

    print(f"{beta:<8.2f} {chi2_total:<13.6e} {chi2_mass:<13.6e} {chi2_g:<13.6e} {S_opt:<10.4f} {C_g_opt:<10.2f} {g_e_res:<+12.6f} {g_mu_res:<+12.6f} {total_hits:<6d}")
    sys.stdout.flush()

    x_prev = np.array([
        result["parameters"]["electron"]["R_c"], result["parameters"]["electron"]["U"], result["parameters"]["electron"]["A"],
        result["parameters"]["muon"]["R_c"], result["parameters"]["muon"]["U"], result["parameters"]["muon"]["A"],
    ])

# Descending scan
x_prev = x_init.copy()

for i in range(idx_start - 1, -1, -1):
    beta = beta_grid[i]
    lam = calibrate_lambda(ETA_TARGET, beta, R_C_REF)

    fitter = TwoLeptonMassGFitterProfiled(
        beta=beta, w=W, lam=lam,
        k_loc=K_LOCALIZATION, delta_v_fac=DELTA_V_FACTOR, p_env=P_ENVELOPE,
    )

    result = fitter.fit_warm_start(x_prev, bounds, maxiter=200, seed=42, workers=8)

    chi2_total = result["chi2_total"]
    chi2_mass = result["chi2_mass"]
    chi2_g = result["chi2_g"]
    S_opt = result["S_opt"]
    C_g_opt = result["C_g_opt"]
    g_e_res = result["g_residuals"]["electron"]
    g_mu_res = result["g_residuals"]["muon"]

    total_hits = sum(1 for v in result["bounds_hit"]["electron"].values() if v != "none") + \
                 sum(1 for v in result["bounds_hit"]["muon"].values() if v != "none")

    results_scan.append({"beta": float(beta), "result": result})

    print(f"{beta:<8.2f} {chi2_total:<13.6e} {chi2_mass:<13.6e} {chi2_g:<13.6e} {S_opt:<10.4f} {C_g_opt:<10.2f} {g_e_res:<+12.6f} {g_mu_res:<+12.6f} {total_hits:<6d}")
    sys.stdout.flush()

    x_prev = np.array([
        result["parameters"]["electron"]["R_c"], result["parameters"]["electron"]["U"], result["parameters"]["electron"]["A"],
        result["parameters"]["muon"]["R_c"], result["parameters"]["muon"]["U"], result["parameters"]["muon"]["A"],
    ])

print("-" * 140)
print()

# Sort by β
results_scan.sort(key=lambda x: x["beta"])

# Find minimum
best = min(results_scan, key=lambda x: x["result"]["chi2_total"])
beta_min = best["beta"]

print("=" * 80)
print("RESULTS")
print("=" * 80)
print()

print(f"Best β: {beta_min:.2f}")
print(f"  χ²_total = {best['result']['chi2_total']:.6e}")
print(f"  χ²_mass  = {best['result']['chi2_mass']:.6e}")
print(f"  χ²_g     = {best['result']['chi2_g']:.6e}")
print(f"  S_opt    = {best['result']['S_opt']:.4f} MeV")
print(f"  C_g_opt  = {best['result']['C_g_opt']:.4f} (profiled normalization, cf. fixed 948)")
print()

print("Predicted vs Target:")
for lep in ["electron", "muon"]:
    print(f"  {lep}:")
    print(f"    mass: {best['result']['masses_model'][lep]:.6f} (target: {best['result']['masses_target'][lep]:.6f})")
    print(f"    g:    {best['result']['g_model'][lep]:.10f} (target: {best['result']['g_target'][lep]:.10f}, Δ={best['result']['g_residuals'][lep]:+.6e})")
print()

# Diagnostic
chi2_values = [r["result"]["chi2_total"] for r in results_scan]
chi2_range = max(chi2_values) / min(chi2_values) if min(chi2_values) > 0 else np.inf

print(f"χ² variation: {min(chi2_values):.6e} to {max(chi2_values):.6e} ({chi2_range:.2f}×)")
print()

if chi2_range < 10:
    print("⚠ χ² still flat (< 10× variation) - g-proxy may not be independent")
elif chi2_range < 100:
    print("~ Moderate χ² variation (10-100×) - partial degeneracy break")
else:
    print("✓ Strong χ² variation (> 100×) - degeneracy broken")

print()

# Save
results_dict = {
    "beta_grid": beta_grid.tolist(),
    "beta_min": beta_min,
    "chi2_range": float(chi2_range),
    "scan_results": results_scan,
    "best_fit": best["result"],
}

with open("results/V22/t2b_profiled_normalization_results.json", "w") as f:
    json.dump(results_dict, f, indent=2)

print("Results saved: results/V22/t2b_profiled_normalization_results.json")
print("=" * 80)
