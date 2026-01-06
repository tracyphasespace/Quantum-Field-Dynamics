#!/usr/bin/env python3
"""
Test 3: Two-Lepton Fit (e, μ Only)

Goal: Determine if τ is the sole driver of cross-lepton β drift.

Hypothesis: If τ is the outlier, fitting only (e, μ) should recover
            β ≈ 3.058 (Golden Loop target from α).

Expected outcomes:
  A) β_eff(e,μ) ≈ 3.058 ± 0.02 → τ is outlier, light leptons validate model
  B) β_eff(e,μ) ≈ 3.15+ → universal issue, model wrong even for light leptons
"""

import numpy as np
from profile_likelihood_boundary_layer import LeptonFitter, calibrate_lambda
import json

# Physical constants
M_E = 0.511
M_MU = 105.7
M_TAU = 1776.8

# Test parameters
W = 0.020
ETA_TARGET = 0.03
R_C_REF = 0.88  # Electron reference for calibration

print("=" * 70)
print("TEST 3: TWO-LEPTON FIT (e, μ ONLY)")
print("=" * 70)
print()
print("Testing if τ is the outlier driving β drift")
print()


class LeptonFitterTwoLepton(LeptonFitter):
    """Fit only electron + muon (exclude tau)"""

    def __init__(self, beta, w, lam, sigma_model=1e-4):
        # Call parent __init__ first
        super().__init__(beta, w, lam, sigma_model)

        # Override targets for e, μ only
        self.m_targets = np.array([M_E, M_MU])
        self.leptons_active = ["electron", "muon"]

    def objective(self, x):
        """
        Modified objective: only fit electron + muon

        Parameters (6 total):
          x[0:3] = R_c, U, A for electron
          x[3:6] = R_c, U, A for muon
        """
        # Unpack parameters (only 6, not 9)
        R_c_e, U_e, A_e = x[0:3]
        R_c_mu, U_mu, A_mu = x[3:6]

        # Compute energies for electron and muon
        E_e, E_circ_e, E_stab_e, E_grad_e = self.energy_calc.total_energy(R_c_e, U_e, A_e)
        E_mu, E_circ_mu, E_stab_mu, E_grad_mu = self.energy_calc.total_energy(R_c_mu, U_mu, A_mu)

        energies = np.array([E_e, E_mu])

        # Absolute uncertainties (relative sigma_model × target masses)
        sigma_abs = self.sigma_model * self.m_targets
        weights = 1.0 / sigma_abs**2

        # Analytic S profiling (same formula, only 2 leptons)
        numerator = np.sum(self.m_targets * energies * weights)
        denominator = np.sum(energies**2 * weights)

        if denominator > 0:
            S_opt = numerator / denominator
        else:
            return 1e10

        # Compute model masses
        masses_model = S_opt * energies

        # Residuals
        residuals = (masses_model - self.m_targets) / sigma_abs

        # χ²
        chi2 = np.sum(residuals**2)

        return chi2

    def fit(self, max_iter=200, seed=None):
        """Run differential evolution fit (only 6 parameters)"""
        from scipy.optimize import differential_evolution

        if seed is not None:
            np.random.seed(seed)

        # Bounds for electron and muon only (6 parameters)
        bounds_all = [
            (0.5, 1.5),    # R_c_e (electron)
            (0.01, 0.1),   # U_e
            (0.7, 1.0),    # A_e
            (0.05, 0.3),   # R_c_mu (muon)
            (0.05, 0.2),   # U_mu
            (0.7, 1.0),    # A_mu
        ]

        result = differential_evolution(
            self.objective,
            bounds_all,
            maxiter=max_iter,
            seed=seed,
            atol=1e-8,
            tol=1e-8,
            workers=1,
        )

        # Extract best parameters
        x_best = result.x
        R_c_e, U_e, A_e = x_best[0:3]
        R_c_mu, U_mu, A_mu = x_best[3:6]

        # Compute final energies
        E_e, E_circ_e, E_stab_e, E_grad_e = self.energy_calc.total_energy(R_c_e, U_e, A_e)
        E_mu, E_circ_mu, E_stab_mu, E_grad_mu = self.energy_calc.total_energy(R_c_mu, U_mu, A_mu)

        energies = np.array([E_e, E_mu])

        # Compute optimal S
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
            "masses_model": masses_model,
            "masses_target": self.m_targets,
        }


# ========================================================================
# β Scan (e, μ only)
# ========================================================================

print("=" * 70)
print("β SCAN: ELECTRON + MUON ONLY")
print("=" * 70)
print()

# Scan range (cover theoretical target β ≈ 3.058 and current best β ≈ 3.15)
beta_range = (3.00, 3.20)
n_beta = 21

print(f"β range: [{beta_range[0]}, {beta_range[1]}] with {n_beta} points")
print(f"w = {W} (fixed)")
print(f"η_target = {ETA_TARGET} (for λ calibration)")
print()

beta_grid = np.linspace(*beta_range, n_beta)
results_scan = []

print("Running β scan...")
print("-" * 70)
print(f"{'β':<10} {'χ²':<15} {'S_opt':<12} {'S_e/S_μ':<12}")
print("-" * 70)

for i, beta in enumerate(beta_grid):
    # Calibrate λ for this β
    lam = calibrate_lambda(ETA_TARGET, beta, R_C_REF)

    # Fit with two leptons only
    fitter = LeptonFitterTwoLepton(beta=beta, w=W, lam=lam, sigma_model=1e-4)
    result = fitter.fit(max_iter=200, seed=42)

    # Extract metrics
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
        "parameters": result["parameters"],
        "energies": result["energies"],
    })

    print(f"{beta:<10.4f} {chi2:<15.6e} {S_opt:<12.4f} {S_ratio:<12.4f}")

print("-" * 70)
print()

# Find minimum χ²
best_result = min(results_scan, key=lambda x: x["chi2"])
beta_min = best_result["beta"]
chi2_min = best_result["chi2"]
S_opt_min = best_result["S_opt"]
S_ratio_min = best_result["S_e_over_S_mu"]

print("=" * 70)
print("RESULTS")
print("=" * 70)
print()

print(f"Best fit (e, μ only):")
print(f"  β_min = {beta_min:.4f}")
print(f"  χ²_min = {chi2_min:.6e}")
print(f"  S_opt = {S_opt_min:.4f}")
print(f"  S_e/S_μ = {S_ratio_min:.4f}")
print()

# Display best-fit parameters
print("Best-fit parameters:")
print("-" * 70)
params_best = best_result["parameters"]
energies_best = best_result["energies"]

for lepton in ["electron", "muon"]:
    p = params_best[lepton]
    e = energies_best[lepton]
    print(f"{lepton:8s}: R_c={p['R_c']:.4f}, U={p['U']:.4f}, A={p['A']:.4f}, E={e['E_total']:.6f}")
print()

# ========================================================================
# Decision Criteria
# ========================================================================

print("=" * 70)
print("DECISION")
print("=" * 70)
print()

# Expected β from Golden Loop
BETA_TARGET = 3.058

# Criteria
beta_close_to_target = abs(beta_min - BETA_TARGET) < 0.03
chi2_reasonable = chi2_min < 20
S_ratio_universal = abs(S_ratio_min - 1.0) < 0.15

print(f"Theoretical target: β = {BETA_TARGET}")
print(f"Observed minimum:   β = {beta_min:.4f}")
print(f"Offset:             Δβ = {beta_min - BETA_TARGET:+.4f}")
print()

print("Decision criteria:")
print(f"  |β_min - β_target| < 0.03:  {beta_close_to_target} ({'PASS' if beta_close_to_target else 'FAIL'})")
print(f"  χ² < 20:                    {chi2_reasonable} ({'PASS' if chi2_reasonable else 'FAIL'})")
print(f"  |S_e/S_μ - 1.0| < 0.15:     {S_ratio_universal} ({'PASS' if S_ratio_universal else 'FAIL'})")
print()

if beta_close_to_target and chi2_reasonable and S_ratio_universal:
    print("✓ OUTCOME A: τ IS OUTLIER")
    print()
    print("Light leptons (e, μ) validate the model!")
    print("  → β_min ≈ β_target (Golden Loop)")
    print("  → Universal scaling (S_e ≈ S_μ)")
    print("  → χ² reasonable")
    print()
    print("Interpretation:")
    print("  - Hill vortex model works for light leptons (m < 200 MeV)")
    print("  - τ (m_τ ≈ 2 m_proton) requires model extension")
    print("  - Consistent with hadronic mass-scale transition")
    print()
    print("Publishable narrative:")
    print("  'Electron and muon masses confirm β = 3.058 ± 0.02'")
    print("  'Tau exhibits systematic 46% energy deficit'")
    print("  'Deviation consistent with charge-circulation → ballast transition'")
    print()
    print("Next steps:")
    print("  1. Publish: (e,μ) validation + τ anomaly quantified")
    print("  2. Implement Option A or B (localize vortex / align shell)")
    print("  3. Re-test with extended model")
    outcome = "A_tau_outlier"

elif not beta_close_to_target:
    print("✗ OUTCOME B: UNIVERSAL ISSUE")
    print()
    print("Even (e, μ) alone prefer β ≈ {:.3f}".format(beta_min))
    print("  → Issue is not τ-specific")
    print("  → Model/mapping wrong even for light leptons")
    print()
    print("Interpretation:")
    print("  - Circulation energy functional may be incorrect")
    print("  - EM proxy calibration may be off")
    print("  - Or: constraint system incomplete (need radius, g-2, etc.)")
    print()
    print("Next steps:")
    print("  1. Review Hill vortex energy derivation")
    print("  2. Check circulation_energy() implementation")
    print("  3. Add observables beyond mass (magnetic moment, radius)")
    outcome = "B_universal_issue"

else:
    print("~ PARTIAL VALIDATION")
    print()
    print("Some criteria pass, others fail.")
    print("  → Need further investigation")
    print()
    if not chi2_reasonable:
        print("  χ² still high → fit quality issue")
    if not S_ratio_universal:
        print("  S_e/S_μ ≠ 1 → even e,μ show regime split")
    print()
    outcome = "partial"

print()
print("=" * 70)

# Save results
results_dict = {
    "beta_min": beta_min,
    "chi2_min": chi2_min,
    "S_opt": S_opt_min,
    "S_e_over_S_mu": S_ratio_min,
    "beta_target": BETA_TARGET,
    "beta_offset": beta_min - BETA_TARGET,
    "outcome": outcome,
    "scan": [
        {
            "beta": float(r["beta"]),
            "chi2": float(r["chi2"]),
            "S_opt": float(r["S_opt"]),
            "S_e_over_S_mu": float(r["S_e_over_S_mu"]),
        }
        for r in results_scan
    ],
}

with open("two_lepton_fit_results.json", "w") as f:
    json.dump(results_dict, f, indent=2)

print("Results saved to: two_lepton_fit_results.json")
print()

# Create simple plot if matplotlib available
try:
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    betas = [r["beta"] for r in results_scan]
    chi2s = [r["chi2"] for r in results_scan]
    S_ratios = [r["S_e_over_S_mu"] for r in results_scan]

    # χ² vs β
    ax1.plot(betas, chi2s, 'o-', linewidth=2, markersize=6)
    ax1.axvline(BETA_TARGET, color='red', linestyle='--', linewidth=2, label=f'β_target = {BETA_TARGET}')
    ax1.axvline(beta_min, color='green', linestyle='--', linewidth=2, label=f'β_min = {beta_min:.4f}')
    ax1.set_xlabel('β', fontsize=12)
    ax1.set_ylabel('χ²', fontsize=12)
    ax1.set_title('Two-Lepton Fit (e, μ only): χ² vs β', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # S_e/S_μ vs β
    ax2.plot(betas, S_ratios, 'o-', linewidth=2, markersize=6, color='purple')
    ax2.axhline(1.0, color='black', linestyle='--', linewidth=1, label='S_e/S_μ = 1 (universal)')
    ax2.axvline(BETA_TARGET, color='red', linestyle='--', linewidth=2, alpha=0.5)
    ax2.axvline(beta_min, color='green', linestyle='--', linewidth=2, alpha=0.5)
    ax2.set_xlabel('β', fontsize=12)
    ax2.set_ylabel('S_e/S_μ', fontsize=12)
    ax2.set_title('Scale Ratio vs β', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('two_lepton_fit_scan.png', dpi=150, bbox_inches='tight')
    print("Plot saved to: two_lepton_fit_scan.png")
    print()

except ImportError:
    print("(matplotlib not available, skipping plot)")
    print()
