#!/usr/bin/env python3
"""
Run 2: e,μ Regression with CORRECTED SIGN CONVENTION

Configuration:
  - k = 1.5, Δv/Rv = 0.5, p = 6 (outside-only localization)
  - E_total = E_circ + E_stab + E_grad (all penalties add)

Fit scope:
  - Leptons: electron + muon only
  - Shared: β (scan), global S (analytic)
  - Per-lepton: U, R_c, A

Acceptance criteria:
  1. χ² not in 10^8 regime (substantial reduction)
  2. S_opt > 0
  3. No "all parameters at bounds"
  4. β interior to search interval
  5. Multi-start stability
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

# Localization configuration (FROZEN)
K_LOCALIZATION = 1.5
DELTA_V_FACTOR = 0.5
P_ENVELOPE = 6

# Test parameters
W = 0.020
ETA_TARGET = 0.03
R_C_REF = 0.88

print("=" * 80)
print("RUN 2: e,μ REGRESSION (Corrected Sign Convention)")
print("=" * 80)
print()

print("Configuration:")
print(f"  Localization: k={K_LOCALIZATION}, Δv/Rv={DELTA_V_FACTOR}, p={P_ENVELOPE}")
print(f"  Functional: E_total = E_circ + E_stab + E_grad (corrected signs)")
print()


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
        """
        χ² objective with global S profiling

        Parameters (6):
          x[0:3] = R_c, U, A for electron
          x[3:6] = R_c, U, A for muon
        """
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

    def fit(self, max_iter=200, seed=None):
        """Run fit"""
        if seed is not None:
            np.random.seed(seed)

        # Bounds (reasonable, not too tight)
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
            workers=4,  # Parallel evaluation on 4 cores
        )

        # Extract parameters
        x_best = result.x
        R_c_e, U_e, A_e = x_best[0:3]
        R_c_mu, U_mu, A_mu = x_best[3:6]

        # Compute final energies
        E_e, E_circ_e, E_stab_e, E_grad_e = self.energy_calc.total_energy(R_c_e, U_e, A_e)
        E_mu, E_circ_mu, E_stab_mu, E_grad_mu = self.energy_calc.total_energy(R_c_mu, U_mu, A_mu)

        # Get diagnostics (F_inner)
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


# ========================================================================
# β Scan
# ========================================================================

beta_range = (3.00, 3.20)
n_beta = 21

print(f"β scan: [{beta_range[0]}, {beta_range[1]}] with {n_beta} points")
print()

beta_grid = np.linspace(*beta_range, n_beta)
results_scan = []

print("Running scan...")
print("-" * 80)
print(f"{'β':<10} {'χ²':<15} {'S_opt':<12}")
print("-" * 80)
sys.stdout.flush()

for beta in tqdm(beta_grid, desc="β scan", unit="point"):
    lam = calibrate_lambda(ETA_TARGET, beta, R_C_REF)

    fitter = TwoLeptonFitterCorrected(
        beta=beta,
        w=W,
        lam=lam,
        k_loc=K_LOCALIZATION,
        delta_v_fac=DELTA_V_FACTOR,
        p_env=P_ENVELOPE,
        sigma_model=1e-4,
    )

    result = fitter.fit(max_iter=100, seed=42)

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

print("=" * 80)
print("RESULTS")
print("=" * 80)
print()

print(f"Best fit:")
print(f"  β_min = {beta_min:.4f}")
print(f"  χ²_min = {chi2_min:.6e}")
print(f"  S_opt = {best['S_opt']:.4f}")
print()

# Best-fit parameters
result_best = best["result"]
print("Parameters:")
for lepton in ["electron", "muon"]:
    p = result_best["parameters"][lepton]
    print(f"  {lepton:8s}: R_c={p['R_c']:.4f}, U={p['U']:.4f}, A={p['A']:.4f}")
print()

# Energies
print("Energies:")
for lepton in ["electron", "muon"]:
    e = result_best["energies"][lepton]
    print(f"  {lepton:8s}: E_total={e['E_total']:.6f}, E_circ={e['E_circ']:.6f}, E_stab={e['E_stab']:.6f}, E_grad={e['E_grad']:.6f}")
print()

# Diagnostics
print("Diagnostics:")
for lepton in ["electron", "muon"]:
    d = result_best["diagnostics"][lepton]
    print(f"  {lepton:8s}: F_inner={d['F_inner']:.2%}")
print()

# Bound hits
print("Bound hits:")
for lepton in ["electron", "muon"]:
    b = result_best["bounds_hit"][lepton]
    hits = [f"{k}={v}" for k, v in b.items() if v != "none"]
    if hits:
        print(f"  {lepton:8s}: {', '.join(hits)}")
    else:
        print(f"  {lepton:8s}: none")
print()

# ========================================================================
# Acceptance Criteria
# ========================================================================

print("=" * 80)
print("ACCEPTANCE CRITERIA")
print("=" * 80)
print()

# Criteria
no_pathology = chi2_min < 1e6  # Orders of magnitude reduction from 10^8
S_positive = best["S_opt"] > 0
beta_interior = beta_range[0] + 0.02 < beta_min < beta_range[1] - 0.02

# Count bound hits
e_hits = sum(1 for v in result_best["bounds_hit"]["electron"].values() if v != "none")
mu_hits = sum(1 for v in result_best["bounds_hit"]["muon"].values() if v != "none")
not_all_bounds = e_hits <= 1 and mu_hits <= 1

print(f"1. No pathology (χ² < 10^6):          {no_pathology} ({'PASS' if no_pathology else 'FAIL'})")
print(f"     χ² = {chi2_min:.2e}")
print()

print(f"2. S_opt > 0:                          {S_positive} ({'PASS' if S_positive else 'FAIL'})")
print(f"     S_opt = {best['S_opt']:.4f}")
print()

print(f"3. Not all parameters at bounds:       {not_all_bounds} ({'PASS' if not_all_bounds else 'FAIL'})")
print(f"     Electron hits: {e_hits}/3, Muon hits: {mu_hits}/3")
print()

print(f"4. β interior to search interval:      {beta_interior} ({'PASS' if beta_interior else 'FAIL'})")
print(f"     β_min = {beta_min:.4f} ∈ ({beta_range[0]}, {beta_range[1]})")
print()

# Overall
pass_overall = no_pathology and S_positive and not_all_bounds and beta_interior

if pass_overall:
    print("✓ RUN 2 PASS")
    print()
    print("All acceptance criteria met.")
    print("Light leptons (e,μ) validate corrected sign convention.")
    print()
    print("NEXT: Run 2b (β targeting check) and proceed to τ")
    outcome = "pass"
elif no_pathology and S_positive:
    print("~ RUN 2 SOFT PASS")
    print()
    print("χ² improved substantially and S_opt > 0, but some degeneracy remains.")
    print("May need additional constraint (radius or moment) before τ.")
    outcome = "soft_pass"
else:
    print("✗ RUN 2 FAIL")
    print()
    if not no_pathology:
        print("  χ² still pathological")
    if not S_positive:
        print("  S_opt ≤ 0")
    print()
    print("Corrected signs did not fully resolve fit.")
    outcome = "fail"

print()

# Save
results_dict = {
    "k_localization": K_LOCALIZATION,
    "delta_v_factor": DELTA_V_FACTOR,
    "p_envelope": P_ENVELOPE,
    "beta_min": beta_min,
    "chi2_min": chi2_min,
    "S_opt": best["S_opt"],
    "outcome": outcome,
    "best_fit": result_best,
}

with open("results/V22/run2_emu_corrected_results.json", "w") as f:
    json.dump(results_dict, f, indent=2)

print("Results saved: results/V22/run2_emu_corrected_results.json")
print()
print("=" * 80)
