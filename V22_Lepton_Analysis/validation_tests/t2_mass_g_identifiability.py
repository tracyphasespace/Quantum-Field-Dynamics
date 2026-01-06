#!/usr/bin/env python3
"""
T2': Mass + g-factor Identifiability Test

Purpose: Break β degeneracy by adding magnetic moment constraints.

Configuration:
  - Observables: (m_e, m_μ, g_e, g_μ) — 4 constraints
  - Parameters: (R_c, U, A) × 2 leptons — 6 DOF
  - S_opt: Analytically profiled from MASS terms only
  - Weights: σ_m = 1e-4 × m, σ_g = 1e-3 (absolute)

Magnetic moment proxy:
  μ = k × Q × R_shell × U
  where R_shell = R_c + w (vortex boundary, not core radius)
  g = (948.0 × μ) / (m_model / m_e)

Expected outcome:
  - β becomes identifiable with interior minimum, OR
  - Degeneracy persists (g proxy not independent of mass constraint), OR
  - Conflict appears (proxy inconsistent with energy functional)
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

# g-factor targets (QED-accurate values for identifiability)
G_E = 2.00231930436256  # Electron g-factor
G_MU = 2.0023318414     # Muon g-factor (user noted: be careful with normalization)

# Magnetic moment proxy parameters
K_GEOMETRIC = 0.2       # Hill vortex geometric factor
Q_CHARGE = 1.0          # Fundamental charge
G_NORMALIZATION = 948.0 # Calibrated from β=3.058 electron solution

# Configuration (same as T1c best)
W = 0.020
ETA_TARGET = 0.03
R_C_REF = 0.88
K_LOCALIZATION = 1.5
DELTA_V_FACTOR = 0.5
P_ENVELOPE = 6

print("=" * 80)
print("T2': MASS + g-FACTOR IDENTIFIABILITY TEST")
print("=" * 80)
print()
print("Objective: Break β degeneracy with magnetic moment constraints")
print()
print("Observable set:")
print(f"  Masses:    m_e = {M_E:.3f} MeV, m_μ = {M_MU:.1f} MeV")
print(f"  g-factors: g_e = {G_E:.10f}, g_μ = {G_MU:.10f}")
print()
print("Weighting:")
print(f"  σ_m / m = 1e-4 (0.01%)")
print(f"  σ_g = 1e-3 (absolute, ~0.05% relative)")
print()
print("Magnetic moment proxy:")
print(f"  μ = {K_GEOMETRIC} × {Q_CHARGE} × R_shell × U")
print(f"  g = {G_NORMALIZATION} × μ / (m_model / m_e)")
print(f"  where R_shell = R_c + w = R_c + {W}")
print()
sys.stdout.flush()


def compute_magnetic_moment(R_shell, U):
    """
    Magnetic moment proxy for Hill vortex circulation.

    Formula: μ = k × Q × R_shell × U

    CRITICAL: Use R_shell (vortex boundary), not R_c (core radius).
    """
    mu = K_GEOMETRIC * Q_CHARGE * R_shell * U
    return mu


def compute_g_factor(mu, mass_model, mass_electron):
    """
    Convert magnetic moment to g-factor.

    Formula: g = normalization × μ / (m_model / m_e)

    CRITICAL: Use MODEL mass (S × E), not target mass.
    """
    mass_ratio = mass_model / mass_electron
    g = G_NORMALIZATION * mu / mass_ratio
    return g


class TwoLeptonMassGFitter:
    """Fit e,μ with mass + g-factor constraints"""

    def __init__(self, beta, w, lam, k_loc, delta_v_fac, p_env):
        self.beta = beta
        self.w = w
        self.lam = lam

        # Targets
        self.m_targets = np.array([M_E, M_MU])
        self.g_targets = np.array([G_E, G_MU])

        # Uncertainties
        self.sigma_mass = 1e-4 * self.m_targets  # 0.01% relative
        self.sigma_g = 1e-3                       # 0.05% relative (absolute)

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
        Combined mass + g-factor χ²

        Parameters (6):
          x[0:3] = R_c, U, A for electron
          x[3:6] = R_c, U, A for muon

        Procedure:
          1. Compute energies E_e, E_μ
          2. Profile S from mass terms only: S_opt = Σ[m·E/σ²] / Σ[E²/σ²]
          3. Compute model masses: m_model = S_opt × E
          4. Compute magnetic moments: μ = k × Q × R_shell × U
          5. Compute g-factors: g = 948 × μ / (m_model / m_e)
          6. χ² = χ²_mass + χ²_g
        """
        R_c_e, U_e, A_e = x[0:3]
        R_c_mu, U_mu, A_mu = x[3:6]

        # Compute energies
        E_e, _, _, _ = self.energy_calc.total_energy(R_c_e, U_e, A_e)
        E_mu, _, _, _ = self.energy_calc.total_energy(R_c_mu, U_mu, A_mu)

        energies = np.array([E_e, E_mu])

        if np.any(energies <= 0) or np.any(~np.isfinite(energies)):
            return 1e12

        # Analytic S profiling (from mass terms only)
        weights_mass = 1.0 / self.sigma_mass**2
        numerator = np.sum(self.m_targets * energies * weights_mass)
        denominator = np.sum(energies**2 * weights_mass)

        if denominator <= 0:
            return 1e12

        S_opt = numerator / denominator

        # Model masses (after profiling)
        masses_model = S_opt * energies

        # χ²_mass (2 terms: e, μ)
        residuals_mass = (masses_model - self.m_targets) / self.sigma_mass
        chi2_mass = np.sum(residuals_mass**2)

        # Magnetic moments (use R_shell, not R_c)
        R_shell_e = R_c_e + self.w
        R_shell_mu = R_c_mu + self.w

        mu_e = compute_magnetic_moment(R_shell_e, U_e)
        mu_mu = compute_magnetic_moment(R_shell_mu, U_mu)

        # g-factors (use MODEL masses, not targets)
        g_e = compute_g_factor(mu_e, masses_model[0], M_E)
        g_mu = compute_g_factor(mu_mu, masses_model[1], M_E)

        g_model = np.array([g_e, g_mu])

        # χ²_g (2 terms: e, μ)
        residuals_g = (g_model - self.g_targets) / self.sigma_g
        chi2_g = np.sum(residuals_g**2)

        # Total χ²
        chi2_total = chi2_mass + chi2_g

        return chi2_total

    def fit_warm_start(self, x_prev, bounds, maxiter=200, seed=None, workers=8):
        """Fit with warm start from previous β"""
        if seed is not None:
            np.random.seed(seed)

        n_params = len(bounds)
        popsize = max(15 * n_params, 100)

        # Seed population
        init_pop = np.zeros((popsize, n_params))
        init_pop[0, :] = x_prev

        # Jittered neighbors
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
        R_c_e, U_e, A_e = x_best[0:3]
        R_c_mu, U_mu, A_mu = x_best[3:6]

        # Recompute all observables at solution
        E_e, E_circ_e, E_stab_e, E_grad_e = self.energy_calc.total_energy(R_c_e, U_e, A_e)
        E_mu, E_circ_mu, E_stab_mu, E_grad_mu = self.energy_calc.total_energy(R_c_mu, U_mu, A_mu)

        energies = np.array([E_e, E_mu])

        # S profiling
        weights_mass = 1.0 / self.sigma_mass**2
        numerator = np.sum(self.m_targets * energies * weights_mass)
        denominator = np.sum(energies**2 * weights_mass)
        S_opt = numerator / denominator if denominator > 0 else np.nan

        # Model masses
        masses_model = S_opt * energies

        # Magnetic moments and g-factors
        R_shell_e = R_c_e + self.w
        R_shell_mu = R_c_mu + self.w

        mu_e = compute_magnetic_moment(R_shell_e, U_e)
        mu_mu = compute_magnetic_moment(R_shell_mu, U_mu)

        g_e = compute_g_factor(mu_e, masses_model[0], M_E)
        g_mu = compute_g_factor(mu_mu, masses_model[1], M_E)

        g_model = np.array([g_e, g_mu])

        # Individual χ² components
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
            "g_model": {"electron": float(g_e), "muon": float(g_mu)},
            "g_target": {"electron": float(G_E), "muon": float(G_MU)},
            "magnetic_moments": {"electron": float(mu_e), "muon": float(mu_mu)},
            "R_shell": {"electron": float(R_shell_e), "muon": float(R_shell_mu)},
            "bounds_hit": bounds_hit,
        }


# ========================================================================
# Coarse β scan with mass + g constraints
# ========================================================================

beta_range = (1.5, 3.4)
beta_step = 0.2
beta_grid = np.arange(beta_range[0], beta_range[1] + beta_step/2, beta_step)

print(f"β grid: {len(beta_grid)} points from {beta_range[0]} to {beta_range[1]} (step {beta_step})")
print(f"  {beta_grid}")
print()

# Widened bounds (same as T1c)
bounds = [
    (0.05, 3.0),
    (0.005, 0.6),
    (0.05, 0.999),
    (0.02, 1.5),
    (0.01, 0.9),
    (0.05, 0.999),
]

# Start from T1c best at β=3.3 (closest to β=3.3 in grid)
# T1c best params at β=3.30
x_init = np.array([0.233068, 0.387321, 0.517347, 1.280013, 0.507202, 0.273778])

# Find closest β in grid to 3.3
idx_start = np.argmin(np.abs(beta_grid - 3.3))
beta_start = beta_grid[idx_start]

print(f"Starting from β = {beta_start:.2f} (warm start from T1c best at β=3.30)")
print()

results_scan = []

print("Running β scan with mass + g-factor constraints...")
print("-" * 120)
print(f"{'β':<10} {'χ²_total':<15} {'χ²_mass':<15} {'χ²_g':<15} {'S_opt':<12} {'Bound hits':<12}")
print("-" * 120)
sys.stdout.flush()

x_prev = x_init.copy()

# Scan ascending from start
for i in range(idx_start, len(beta_grid)):
    beta = beta_grid[i]
    lam = calibrate_lambda(ETA_TARGET, beta, R_C_REF)

    fitter = TwoLeptonMassGFitter(
        beta=beta,
        w=W,
        lam=lam,
        k_loc=K_LOCALIZATION,
        delta_v_fac=DELTA_V_FACTOR,
        p_env=P_ENVELOPE,
    )

    result = fitter.fit_warm_start(x_prev, bounds, maxiter=200, seed=42, workers=8)

    chi2_total = result["chi2_total"]
    chi2_mass = result["chi2_mass"]
    chi2_g = result["chi2_g"]
    S_opt = result["S_opt"]

    e_hits = sum(1 for v in result["bounds_hit"]["electron"].values() if v != "none")
    mu_hits = sum(1 for v in result["bounds_hit"]["muon"].values() if v != "none")
    total_hits = e_hits + mu_hits

    results_scan.append({"beta": float(beta), "result": result})

    print(f"{beta:<10.2f} {chi2_total:<15.6e} {chi2_mass:<15.6e} {chi2_g:<15.6e} {S_opt:<12.4f} {total_hits:<12d}")
    sys.stdout.flush()

    # Update for next iteration
    x_prev = np.array([
        result["parameters"]["electron"]["R_c"],
        result["parameters"]["electron"]["U"],
        result["parameters"]["electron"]["A"],
        result["parameters"]["muon"]["R_c"],
        result["parameters"]["muon"]["U"],
        result["parameters"]["muon"]["A"],
    ])

# Scan descending from start
x_prev = x_init.copy()

for i in range(idx_start - 1, -1, -1):
    beta = beta_grid[i]
    lam = calibrate_lambda(ETA_TARGET, beta, R_C_REF)

    fitter = TwoLeptonMassGFitter(
        beta=beta,
        w=W,
        lam=lam,
        k_loc=K_LOCALIZATION,
        delta_v_fac=DELTA_V_FACTOR,
        p_env=P_ENVELOPE,
    )

    result = fitter.fit_warm_start(x_prev, bounds, maxiter=200, seed=42, workers=8)

    chi2_total = result["chi2_total"]
    chi2_mass = result["chi2_mass"]
    chi2_g = result["chi2_g"]
    S_opt = result["S_opt"]

    e_hits = sum(1 for v in result["bounds_hit"]["electron"].values() if v != "none")
    mu_hits = sum(1 for v in result["bounds_hit"]["muon"].values() if v != "none")
    total_hits = e_hits + mu_hits

    results_scan.append({"beta": float(beta), "result": result})

    print(f"{beta:<10.2f} {chi2_total:<15.6e} {chi2_mass:<15.6e} {chi2_g:<15.6e} {S_opt:<12.4f} {total_hits:<12d}")
    sys.stdout.flush()

    # Update for next iteration
    x_prev = np.array([
        result["parameters"]["electron"]["R_c"],
        result["parameters"]["electron"]["U"],
        result["parameters"]["electron"]["A"],
        result["parameters"]["muon"]["R_c"],
        result["parameters"]["muon"]["U"],
        result["parameters"]["muon"]["A"],
    ])

print("-" * 120)
print()

# Sort by β
results_scan.sort(key=lambda x: x["beta"])

# Find minimum
best = min(results_scan, key=lambda x: x["result"]["chi2_total"])
beta_min = best["beta"]
chi2_min = best["result"]["chi2_total"]

print("=" * 80)
print("RESULTS")
print("=" * 80)
print()

print(f"Best β: {beta_min:.2f}")
print(f"  χ²_total = {chi2_min:.6e}")
print(f"  χ²_mass  = {best['result']['chi2_mass']:.6e}")
print(f"  χ²_g     = {best['result']['chi2_g']:.6e}")
print(f"  S_opt    = {best['result']['S_opt']:.4f} MeV")
print()

print("Best-fit parameters:")
for lepton in ["electron", "muon"]:
    p = best["result"]["parameters"][lepton]
    print(f"  {lepton:8s}: R_c={p['R_c']:.6f}, U={p['U']:.6f}, A={p['A']:.6f}")
print()

print("Model vs Target:")
for lepton in ["electron", "muon"]:
    m_model = best["result"]["masses_model"][lepton]
    m_target = best["result"]["masses_target"][lepton]
    g_model = best["result"]["g_model"][lepton]
    g_target = best["result"]["g_target"][lepton]

    print(f"  {lepton:8s}:")
    print(f"    mass:     {m_model:.6f} (target: {m_target:.6f}, Δ={(m_model-m_target)/m_target*100:+.6f}%)")
    print(f"    g-factor: {g_model:.10f} (target: {g_target:.10f}, Δ={(g_model-g_target)/g_target*100:+.6f}%)")
print()

# Check if β is interior
beta_interior = beta_range[0] + beta_step/2 < beta_min < beta_range[1] - beta_step/2

print(f"β minimum interior to scan range: {beta_interior}")
if beta_interior:
    print(f"  β_min = {beta_min:.2f} ∈ ({beta_range[0]}, {beta_range[1]})")
    print()
    print("✓ DEGENERACY BROKEN: β is identifiable with mass + g constraints")
else:
    print(f"  β_min = {beta_min:.2f} at edge of scan range")
    if beta_min <= beta_range[0] + beta_step/2:
        print("  Consider extending to lower β")
    else:
        print("  Consider extending to higher β")
    print()
    print("~ PARTIAL BREAK: Degeneracy reduced but β minimum at boundary")

print()

# Diagnostic: χ² range across β
chi2_values = [r["result"]["chi2_total"] for r in results_scan]
chi2_min_all = min(chi2_values)
chi2_max_all = max(chi2_values)
chi2_range = chi2_max_all / chi2_min_all if chi2_min_all > 0 else np.inf

print(f"χ² variation across β:")
print(f"  Min: {chi2_min_all:.6e}")
print(f"  Max: {chi2_max_all:.6e}")
print(f"  Range: {chi2_range:.2f}×")
print()

if chi2_range < 10:
    print("⚠ WARNING: χ² nearly flat across β (variation < 10×)")
    print("  g-factor proxy may not be independent of mass constraint")
    print("  Consider adding radius constraint or tightening σ_g")
elif chi2_range < 100:
    print("~ MODERATE: χ² shows some β preference (10-100× variation)")
    print("  Degeneracy partially broken")
else:
    print("✓ STRONG: χ² shows clear β preference (>100× variation)")
    print("  Degeneracy effectively broken")

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
    "chi2_range": float(chi2_range),
    "sigma_mass_relative": 1e-4,
    "sigma_g_absolute": 1e-3,
    "scan_results": results_scan,
    "best_fit": best["result"],
}

with open("results/V22/t2_mass_g_identifiability_results.json", "w") as f:
    json.dump(results_dict, f, indent=2)

print("Results saved: results/V22/t2_mass_g_identifiability_results.json")
print()
print("=" * 80)
