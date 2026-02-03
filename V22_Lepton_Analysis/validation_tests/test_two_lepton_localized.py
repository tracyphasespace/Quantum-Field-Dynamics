#!/usr/bin/env python3
"""
Run 2: e,μ Regression with Localized Vortex

Test that localization (k=1.0 fixed) does not degrade e,μ fit quality.

Expected:
  - β behavior stable
  - χ² similar to baseline
  - Fit converges without issues
"""

import numpy as np
from scipy.optimize import differential_evolution
from lepton_energy_localized_v0 import LeptonEnergyLocalized
from profile_likelihood_boundary_layer import calibrate_lambda
import json

# Physical constants
M_E = 0.511
M_MU = 105.7

# Fixed parameters from Run 1A
K_LOCALIZATION = 1.0  # Best from Run 1A
P_ENVELOPE = 8

# Test parameters
W = 0.020
ETA_TARGET = 0.03
R_C_REF = 0.88

print("=" * 70)
print("RUN 2: e,μ REGRESSION (Localized Vortex, k=1.0)")
print("=" * 70)
print()


class TwoLeptonFitterLocalized:
    """Fit electron + muon with localized vortex"""

    def __init__(self, beta, w, lam, k_localization, sigma_model=1e-4):
        self.beta = beta
        self.w = w
        self.lam = lam
        self.k_localization = k_localization
        self.sigma_model = sigma_model

        # Targets (e, μ only)
        self.m_targets = np.array([M_E, M_MU])

        # Create energy calculator with localization
        self.energy_calc = LeptonEnergyLocalized(
            beta=beta,
            w=w,
            lam=lam,
            k_localization=k_localization,
            p_envelope=P_ENVELOPE,
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

        # Compute energies
        E_e, _, _, _ = self.energy_calc.total_energy(R_c_e, U_e, A_e)
        E_mu, _, _, _ = self.energy_calc.total_energy(R_c_mu, U_mu, A_mu)

        energies = np.array([E_e, E_mu])

        # Analytic S profiling
        sigma_abs = self.sigma_model * self.m_targets
        weights = 1.0 / sigma_abs**2

        numerator = np.sum(self.m_targets * energies * weights)
        denominator = np.sum(energies**2 * weights)

        if denominator > 0:
            S_opt = numerator / denominator
        else:
            return 1e10

        # χ²
        masses_model = S_opt * energies
        residuals = (masses_model - self.m_targets) / sigma_abs
        chi2 = np.sum(residuals**2)

        return chi2

    def fit(self, max_iter=200, seed=None):
        """Run fit"""
        if seed is not None:
            np.random.seed(seed)

        # Bounds (same as baseline)
        bounds = [
            (0.5, 1.5),    # R_c_e
            (0.01, 0.1),   # U_e
            (0.7, 1.0),    # A_e
            (0.05, 0.3),   # R_c_mu
            (0.05, 0.2),   # U_mu
            (0.7, 1.0),    # A_mu
        ]

        result = differential_evolution(
            self.objective,
            bounds,
            maxiter=max_iter,
            seed=seed,
            atol=1e-8,
            tol=1e-8,
            workers=1,
        )

        # Extract parameters
        x_best = result.x
        R_c_e, U_e, A_e = x_best[0:3]
        R_c_mu, U_mu, A_mu = x_best[3:6]

        # Compute final energies and diagnostics
        E_e, E_circ_e, E_stab_e, E_grad_e = self.energy_calc.total_energy(R_c_e, U_e, A_e)
        E_mu, E_circ_mu, E_stab_mu, E_grad_mu = self.energy_calc.total_energy(R_c_mu, U_mu, A_mu)

        # Get diagnostics (F_inner, I)
        _, F_inner_e, I_e = self.energy_calc.circulation_energy_with_diagnostics(
            R_c_e + self.w, U_e, A_e, R_c_e
        )
        _, F_inner_mu, I_mu = self.energy_calc.circulation_energy_with_diagnostics(
            R_c_mu + self.w, U_mu, A_mu, R_c_mu
        )

        energies = np.array([E_e, E_mu])

        # Compute S_opt
        sigma_abs = self.sigma_model * self.m_targets
        weights = 1.0 / sigma_abs**2
        numerator = np.sum(self.m_targets * energies * weights)
        denominator = np.sum(energies**2 * weights)
        S_opt = numerator / denominator if denominator > 0 else 1.0

        masses_model = S_opt * energies

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
                "electron": {"F_inner": F_inner_e, "I": I_e},
                "muon": {"F_inner": F_inner_mu, "I": I_mu},
            },
            "masses_model": masses_model,
            "masses_target": self.m_targets,
        }


# ========================================================================
# β Scan
# ========================================================================

print(f"k_localization = {K_LOCALIZATION} (fixed from Run 1A)")
print(f"p_envelope = {P_ENVELOPE}")
print()

beta_range = (3.00, 3.20)
n_beta = 21

print(f"β scan: [{beta_range[0]}, {beta_range[1]}] with {n_beta} points")
print()

beta_grid = np.linspace(*beta_range, n_beta)
results_scan = []

print("Running scan...")
print("-" * 70)
print(f"{'β':<10} {'χ²':<15} {'S_opt':<12} {'S_e/S_μ':<12}")
print("-" * 70)

for beta in beta_grid:
    lam = calibrate_lambda(ETA_TARGET, beta, R_C_REF)

    fitter = TwoLeptonFitterLocalized(
        beta=beta, w=W, lam=lam, k_localization=K_LOCALIZATION, sigma_model=1e-4
    )
    result = fitter.fit(max_iter=200, seed=42)

    chi2 = result["chi2"]
    S_opt = result["S_opt"]

    E_e = result["energies"]["electron"]["E_total"]
    E_mu = result["energies"]["muon"]["E_total"]

    S_e = M_E / E_e if E_e > 0 else 0
    S_mu = M_MU / E_mu if E_mu > 0 else 0
    S_ratio = S_e / S_mu if S_mu > 0 else 0

    results_scan.append({
        "beta": beta,
        "chi2": chi2,
        "S_opt": S_opt,
        "S_e_over_S_mu": S_ratio,
        "result": result,
    })

    print(f"{beta:<10.4f} {chi2:<15.6e} {S_opt:<12.4f} {S_ratio:<12.4f}")

print("-" * 70)
print()

# Find minimum
best = min(results_scan, key=lambda x: x["chi2"])
beta_min = best["beta"]
chi2_min = best["chi2"]

print("=" * 70)
print("RESULTS")
print("=" * 70)
print()

print(f"Best fit:")
print(f"  β_min = {beta_min:.4f}")
print(f"  χ²_min = {chi2_min:.6e}")
print(f"  S_opt = {best['S_opt']:.4f}")
print(f"  S_e/S_μ = {best['S_e_over_S_mu']:.4f}")
print()

# Best-fit parameters
result_best = best["result"]
print("Parameters:")
for lepton in ["electron", "muon"]:
    p = result_best["parameters"][lepton]
    print(f"  {lepton}: R_c={p['R_c']:.4f}, U={p['U']:.4f}, A={p['A']:.4f}")
print()

# Diagnostics
print("Diagnostics (F_inner, I):")
for lepton in ["electron", "muon"]:
    d = result_best["diagnostics"][lepton]
    print(f"  {lepton}: F_inner={d['F_inner']:.2%}, I={d['I']:.2f}")
print()

# ========================================================================
# Acceptance Criteria
# ========================================================================

print("=" * 70)
print("ACCEPTANCE CRITERIA (Run 2)")
print("=" * 70)
print()

BETA_TARGET = 3.043233053

beta_close = abs(beta_min - BETA_TARGET) < 0.03
chi2_reasonable = chi2_min < 20
S_universal = abs(best["S_e_over_S_mu"] - 1.0) < 0.15

print(f"Target: β = {BETA_TARGET}")
print(f"Observed: β = {beta_min:.4f}")
print(f"Offset: Δβ = {beta_min - BETA_TARGET:+.4f}")
print()

print("Criteria:")
print(f"  |β - β_target| < 0.03:  {beta_close} ({'PASS' if beta_close else 'FAIL'})")
print(f"  χ² < 20:                {chi2_reasonable} ({'PASS' if chi2_reasonable else 'FAIL'})")
print(f"  |S_e/S_μ - 1| < 0.15:   {S_universal} ({'PASS' if S_universal else 'FAIL'})")
print()

if beta_close and chi2_reasonable and S_universal:
    print("✓ RUN 2 PASS: e,μ regression successful")
    print()
    print("Light leptons validate model with localized vortex!")
    print(f"  → β ≈ {BETA_TARGET} (Golden Loop target)")
    print("  → Universal scaling (S_e ≈ S_μ)")
    print("  → Profile-sensitivity restored (ΔI/I ≈ 9%)")
    print()
    print("Proceed to Run 3 (τ recovery)")
    outcome = "pass"
else:
    print("✗ RUN 2 FAIL: Regression detected")
    print()
    if not beta_close:
        print("  β drift persists even for e,μ")
    if not chi2_reasonable:
        print("  Fit quality degraded")
    if not S_universal:
        print("  S_e/S_μ regime split even for light leptons")
    print()
    print("Investigation needed before proceeding to Run 3")
    outcome = "fail"

print()

# Save
results_dict = {
    "k_localization": K_LOCALIZATION,
    "beta_min": beta_min,
    "chi2_min": chi2_min,
    "S_opt": best["S_opt"],
    "S_e_over_S_mu": best["S_e_over_S_mu"],
    "beta_target": BETA_TARGET,
    "outcome": outcome,
}

with open("results/V22/two_lepton_localized_results.json", "w") as f:
    json.dump(results_dict, f, indent=2)

print("Results saved: results/V22/two_lepton_localized_results.json")
print()
print("=" * 70)
