#!/usr/bin/env python3
"""
T2c: Mass + g-factor with CAVITATION-SATURATED Amplitude

Critical constraint: FIX A = 1.0 (cavitation limit ρ_min = 0) for ALL leptons

Rationale:
  - T2b showed CV(S)=157%, CV(C_g)=52% → β unidentified
  - A is the free compensator absorbing β variation
  - Cavitation saturation is physically principled (QFD cavitation limit)
  - Reduces DOF from 3 to 2 per lepton (R_c, U only)

Expected outcome:
  PASS: CV(S), CV(C_g) drop sharply, β becomes identifiable
  FAIL: β still unidentified → need tau (Option 3)

Density profile:
  ρ(r) = ρ_vac + Δρ(r)
  Δρ(r) = -A(1-(r/R_outer)²)²·T(r)

Cavitation condition:
  ρ_min = ρ(0) = ρ_vac - A = 0
  → A_sat = ρ_vac = 1.0
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

# g-factor targets
G_E = 2.00231930436256
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
print("T2c: CAVITATION-SATURATED (A = 1.0)")
print("=" * 80)
print()
print("CRITICAL CONSTRAINT:")
print(f"  A_electron = A_muon = {A_SAT} (cavitation limit ρ_min = 0)")
print()
print("DOF per lepton: 2 (R_c, U only) — removed A as free parameter")
print()
print("Observable set:")
print(f"  Masses:    m_e = {M_E:.3f} MeV, m_μ = {M_MU:.1f} MeV")
print(f"  g-factors: g_e = {G_E:.10f}, g_μ = {G_MU:.10f}")
print()
print("Profiling:")
print("  S_opt:   profiled from mass terms")
print("  C_g_opt: profiled from g terms")
print()
sys.stdout.flush()


def compute_raw_moment(R_shell, U):
    """Raw magnetic moment: μ = k × Q × R_shell × U"""
    return K_GEOMETRIC * Q_CHARGE * R_shell * U


class TwoLeptonCavitationSaturated:
    """Fit e,μ with cavitation-saturated A = 1.0"""

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
        χ² with FIXED A = 1.0 and dual profiling (S, C_g)

        Parameters (4):
          x[0:2] = R_c, U for electron
          x[2:4] = R_c, U for muon

        A = 1.0 for both leptons (cavitation saturation)
        """
        R_c_e, U_e = x[0:2]
        R_c_mu, U_mu = x[2:4]

        # FIXED amplitude at cavitation limit
        A_e = A_SAT
        A_mu = A_SAT

        # Energies
        E_e, _, _, _ = self.energy_calc.total_energy(R_c_e, U_e, A_e)
        E_mu, _, _, _ = self.energy_calc.total_energy(R_c_mu, U_mu, A_mu)

        energies = np.array([E_e, E_mu])

        if np.any(energies <= 0) or np.any(~np.isfinite(energies)):
            return 1e12

        # Profile S (from mass)
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

        # Raw g-proxies
        mass_ratio_e = masses_model[0] / M_E
        mass_ratio_mu = masses_model[1] / M_E

        x_e = mu_e / mass_ratio_e
        x_mu = mu_mu / mass_ratio_mu

        x_values = np.array([x_e, x_mu])

        if np.any(x_values <= 0) or np.any(~np.isfinite(x_values)):
            return 1e12

        # Profile C_g (from g)
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
        """Fit with warm start (4 parameters: R_c, U for e,μ)"""
        if seed is not None:
            np.random.seed(seed)

        n_params = len(bounds)
        popsize = max(15 * n_params, 80)  # Smaller pop for 4 DOF

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
        R_c_e, U_e = x_best[0:2]
        R_c_mu, U_mu = x_best[2:4]

        # FIXED amplitudes
        A_e = A_SAT
        A_mu = A_SAT

        # Recompute diagnostics
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

        # g-proxies
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

        g_model = C_g_opt * x_values
        g_e = g_model[0]
        g_mu = g_model[1]

        # χ²
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
            },
            "muon": {
                "R_c": "lower" if abs(R_c_mu - bounds[2][0]) < tol else ("upper" if abs(R_c_mu - bounds[2][1]) < tol else "none"),
                "U": "lower" if abs(U_mu - bounds[3][0]) < tol else ("upper" if abs(U_mu - bounds[3][1]) < tol else "none"),
            },
        }

        return {
            "chi2_total": float(result.fun),
            "chi2_mass": float(chi2_mass),
            "chi2_g": float(chi2_g),
            "S_opt": float(S_opt),
            "C_g_opt": float(C_g_opt),
            "parameters": {
                "electron": {"R_c": float(R_c_e), "U": float(U_e), "A": float(A_e)},
                "muon": {"R_c": float(R_c_mu), "U": float(U_mu), "A": float(A_mu)},
            },
            "energies": {
                "electron": {"E_total": float(E_e), "E_circ": float(E_circ_e), "E_stab": float(E_stab_e), "E_grad": float(E_grad_e)},
                "muon": {"E_total": float(E_mu), "E_circ": float(E_circ_mu), "E_stab": float(E_stab_mu), "E_grad": float(E_grad_mu)},
            },
            "masses_model": {"electron": float(masses_model[0]), "muon": float(masses_model[1])},
            "g_model": {"electron": float(g_e), "muon": float(g_mu)},
            "g_residuals": {"electron": float(g_e - G_E), "muon": float(g_mu - G_MU)},
            "bounds_hit": bounds_hit,
        }


# ========================================================================
# β scan with cavitation saturation
# ========================================================================

beta_range = (1.5, 3.4)
beta_step = 0.2
beta_grid = np.arange(beta_range[0], beta_range[1] + beta_step/2, beta_step)

print(f"β grid: {len(beta_grid)} points")
print(f"  {beta_grid}")
print()

# Bounds (4 parameters: R_c, U for electron and muon)
# Using same ranges as before, just removing A
bounds = [
    (0.05, 3.0),   # R_c_e
    (0.005, 0.6),  # U_e
    (0.02, 1.5),   # R_c_mu
    (0.01, 0.9),   # U_mu
]

# Start from T2b best at β=2.1, extract R_c, U only
x_init = np.array([0.5501, 0.0069, 0.9716, 0.8225])
idx_start = np.argmin(np.abs(beta_grid - 2.1))
beta_start = beta_grid[idx_start]

print(f"Starting from β = {beta_start:.2f}")
print()

results_scan = []

print("Running β scan with CAVITATION SATURATION (A = 1.0)...")
print("-" * 120)
print(f"{'β':<8} {'χ²_total':<13} {'χ²_mass':<13} {'χ²_g':<13} {'S_opt':<10} {'C_g_opt':<10} {'R_c_e':<8} {'U_e':<8} {'R_c_μ':<8} {'U_μ':<8}")
print("-" * 120)
sys.stdout.flush()

x_prev = x_init.copy()

# Ascending
for i in range(idx_start, len(beta_grid)):
    beta = beta_grid[i]
    lam = calibrate_lambda(ETA_TARGET, beta, R_C_REF)

    fitter = TwoLeptonCavitationSaturated(
        beta=beta, w=W, lam=lam,
        k_loc=K_LOCALIZATION, delta_v_fac=DELTA_V_FACTOR, p_env=P_ENVELOPE,
    )

    result = fitter.fit_warm_start(x_prev, bounds, maxiter=200, seed=42, workers=8)

    results_scan.append({"beta": float(beta), "result": result})

    print(f"{beta:<8.2f} {result['chi2_total']:<13.6e} {result['chi2_mass']:<13.6e} {result['chi2_g']:<13.6e} " +
          f"{result['S_opt']:<10.4f} {result['C_g_opt']:<10.2f} " +
          f"{result['parameters']['electron']['R_c']:<8.4f} {result['parameters']['electron']['U']:<8.4f} " +
          f"{result['parameters']['muon']['R_c']:<8.4f} {result['parameters']['muon']['U']:<8.4f}")
    sys.stdout.flush()

    x_prev = np.array([
        result["parameters"]["electron"]["R_c"], result["parameters"]["electron"]["U"],
        result["parameters"]["muon"]["R_c"], result["parameters"]["muon"]["U"],
    ])

# Descending
x_prev = x_init.copy()

for i in range(idx_start - 1, -1, -1):
    beta = beta_grid[i]
    lam = calibrate_lambda(ETA_TARGET, beta, R_C_REF)

    fitter = TwoLeptonCavitationSaturated(
        beta=beta, w=W, lam=lam,
        k_loc=K_LOCALIZATION, delta_v_fac=DELTA_V_FACTOR, p_env=P_ENVELOPE,
    )

    result = fitter.fit_warm_start(x_prev, bounds, maxiter=200, seed=42, workers=8)

    results_scan.append({"beta": float(beta), "result": result})

    print(f"{beta:<8.2f} {result['chi2_total']:<13.6e} {result['chi2_mass']:<13.6e} {result['chi2_g']:<13.6e} " +
          f"{result['S_opt']:<10.4f} {result['C_g_opt']:<10.2f} " +
          f"{result['parameters']['electron']['R_c']:<8.4f} {result['parameters']['electron']['U']:<8.4f} " +
          f"{result['parameters']['muon']['R_c']:<8.4f} {result['parameters']['muon']['U']:<8.4f}")
    sys.stdout.flush()

    x_prev = np.array([
        result["parameters"]["electron"]["R_c"], result["parameters"]["electron"]["U"],
        result["parameters"]["muon"]["R_c"], result["parameters"]["muon"]["U"],
    ])

print("-" * 120)
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
print(f"  S_opt    = {best['result']['S_opt']:.4f} MeV")
print(f"  C_g_opt  = {best['result']['C_g_opt']:.4f}")
print()

# CV statistics
S_opts = [r["result"]["S_opt"] for r in results_scan]
Cg_opts = [r["result"]["C_g_opt"] for r in results_scan]

CV_S = np.std(S_opts) / np.mean(S_opts) * 100
CV_Cg = np.std(Cg_opts) / np.mean(Cg_opts) * 100

print(f"COEFFICIENT OF VARIATION:")
print(f"  CV(S_opt):   {CV_S:.1f}% (T2b: 157%)")
print(f"  CV(C_g_opt): {CV_Cg:.1f}% (T2b: 52%)")
print()

chi2_values = [r["result"]["chi2_total"] for r in results_scan]
chi2_range = max(chi2_values) / min(chi2_values) if min(chi2_values) > 0 else np.inf

print(f"χ² variation: {min(chi2_values):.6e} to {max(chi2_values):.6e} ({chi2_range:.1f}×)")
print()

# Assessment
print("=" * 80)
print("IDENTIFIABILITY ASSESSMENT")
print("=" * 80)
print()

if CV_S < 50 and CV_Cg < 30:
    print("✓ PASS: Profiled constants stabilized - β IDENTIFIABLE")
    print(f"  CV reductions: S ({CV_S:.1f}% vs 157%), C_g ({CV_Cg:.1f}% vs 52%)")
elif CV_S < 100 and CV_Cg < 40:
    print("~ PARTIAL: Profiled constants improved but still variable")
    print(f"  β WEAKLY CONSTRAINED")
else:
    print("✗ FAIL: Profiled constants still highly variable")
    print(f"  β REMAINS UNIDENTIFIED even with cavitation saturation")
    print()
    print("NEXT: Add tau (Option 3) - light sector alone insufficient")

print()

# Save
results_dict = {
    "cavitation_amplitude": A_SAT,
    "beta_grid": beta_grid.tolist(),
    "CV_S": float(CV_S),
    "CV_Cg": float(CV_Cg),
    "chi2_range": float(chi2_range),
    "scan_results": results_scan,
    "best_fit": best["result"],
}

with open("results/V22/t2c_cavitation_saturated_results.json", "w") as f:
    json.dump(results_dict, f, indent=2)

print("Results saved: results/V22/t2c_cavitation_saturated_results.json")
print("=" * 80)
