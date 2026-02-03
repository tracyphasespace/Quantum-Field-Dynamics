#!/usr/bin/env python3
"""
3-Lepton Fit with β FIXED by α Constraint

Based on Appendix Z.17:
  α^(-1) ≈ π² × exp(β) × (c2/c1)

With c2/c1 = 0.6522 (V22 nuclear) and α^(-1) = 137.036:
  → β_crit = 3.043233053

This script fixes β and fits only:
  - R_c_e, U_e (electron)
  - R_c_μ, U_μ (muon)
  - R_c_τ, U_τ (tau)
  - S, C_g (global)

Total: 8 parameters to fit 6 observables (overconstrained)
"""

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
import time
import os
import gc

# ============================================================================
# FIXED β FROM α CONSTRAINT
# ============================================================================
BETA_FIXED = 3.043233053  # From Appendix Z.17

# ============================================================================
# PHYSICAL CONSTANTS
# ============================================================================
M_E = 0.511      # MeV
M_MU = 105.7     # MeV
M_TAU = 1776.86  # MeV

G_E = 2.00231930436256
G_MU = 2.0023318414
G_TAU = 2.00118

# ============================================================================
# OPTIMIZATION SETTINGS (Memory-optimized for 4GB)
# ============================================================================
N_STARTS = 3      # Multiple starts for robustness
WORKERS = 6       # 6 of 16 threads
POPSIZE_MULT = 8  # popsize = 8*6 = 48
MAXITER = 200

print("=" * 80)
print("3-LEPTON FIT WITH FIXED β FROM α CONSTRAINT")
print("=" * 80)
print(f"\nβ = {BETA_FIXED:.6f} (FIXED by α, not fitted)")
print(f"  Source: Appendix Z.17, α^(-1) ≈ π² × exp(β) × (c2/c1)")
print(f"  Nuclear c2/c1 = 0.6522")
print(f"  α^(-1) = 137.036")
print()

print("Leptons:")
print(f"  Electron: m = {M_E:.3f} MeV, g = {G_E:.8f}")
print(f"  Muon:     m = {M_MU:.1f} MeV, g = {G_MU:.10f}")
print(f"  Tau:      m = {M_TAU:.2f} MeV, g = {G_TAU:.5f}")
print()

print("Fitting 8 parameters to 6 observables (overconstrained):")
print("  R_c_e, U_e, R_c_μ, U_μ, R_c_τ, U_τ, S, C_g")
print()

# ============================================================================
# 3-LEPTON FITTER CLASS
# ============================================================================

class ThreeLeptonFitter:
    def __init__(self, beta_fixed):
        self.beta = beta_fixed
        self.m_e = M_E
        self.m_mu = M_MU
        self.m_tau = M_TAU
        self.g_e = G_E
        self.g_mu = G_MU
        self.g_tau = G_TAU

    def mass_formula(self, R_c, U, S, beta):
        """QFD mass formula"""
        A = 1.0  # Cavitation saturation
        numerator = (1 + U)**beta - 1
        denominator = R_c**2 * (1 - np.exp(-S * A))
        return numerator / denominator if denominator > 1e-30 else 1e30

    def g_factor_formula(self, U, C_g):
        """QFD g-factor formula"""
        return 2.0 * (1 + U * C_g)

    def objective(self, params):
        """
        Minimize χ² for all three leptons

        params = [R_c_e, U_e, R_c_mu, U_mu, R_c_tau, U_tau, S, C_g]
        """
        R_c_e, U_e, R_c_mu, U_mu, R_c_tau, U_tau, S, C_g = params

        # Sanity checks
        if S <= 0 or C_g <= 0:
            return 1e20
        if U_e <= 0 or U_mu <= 0 or U_tau <= 0:
            return 1e20

        # Calculate masses
        m_e_pred = self.mass_formula(R_c_e, U_e, S, self.beta)
        m_mu_pred = self.mass_formula(R_c_mu, U_mu, S, self.beta)
        m_tau_pred = self.mass_formula(R_c_tau, U_tau, S, self.beta)

        # Calculate g-factors
        g_e_pred = self.g_factor_formula(U_e, C_g)
        g_mu_pred = self.g_factor_formula(U_mu, C_g)
        g_tau_pred = self.g_factor_formula(U_tau, C_g)

        # Chi-squared for masses
        chi2_m_e = ((m_e_pred - self.m_e) / self.m_e)**2
        chi2_m_mu = ((m_mu_pred - self.m_mu) / self.m_mu)**2
        chi2_m_tau = ((m_tau_pred - self.m_tau) / self.m_tau)**2
        chi2_mass = chi2_m_e + chi2_m_mu + chi2_m_tau

        # Chi-squared for g-factors (using g-2 precision)
        chi2_g_e = ((g_e_pred - self.g_e) / 1e-6)**2
        chi2_g_mu = ((g_mu_pred - self.g_mu) / 1e-6)**2
        chi2_g_tau = ((g_tau_pred - self.g_tau) / 1e-4)**2  # Lower precision for tau
        chi2_g = chi2_g_e + chi2_g_mu + chi2_g_tau

        # Total
        chi2_total = chi2_mass + chi2_g

        return chi2_total

    def fit(self, n_starts=3, workers=4, popsize=48, maxiter=200):
        """
        Fit all parameters using differential evolution
        """
        # Bounds: [R_c_e, U_e, R_c_mu, U_mu, R_c_tau, U_tau, S, C_g]
        bounds = [
            (0.05, 3.0),     # R_c_e
            (1e-4, 0.2),     # U_e
            (0.05, 3.0),     # R_c_mu
            (1e-3, 1.0),     # U_mu
            (0.05, 3.0),     # R_c_tau
            (0.01, 2.0),     # U_tau
            (0.1, 20.0),     # S (allow wider range)
            (100, 20000),    # C_g (allow wider range)
        ]

        print(f"Bounds:")
        param_names = ['R_c_e', 'U_e', 'R_c_mu', 'U_mu', 'R_c_tau', 'U_tau', 'S', 'C_g']
        for name, (lo, hi) in zip(param_names, bounds):
            print(f"  {name:8s} ∈ [{lo:.2e}, {hi:.2e}]")
        print()

        best_result = None
        best_loss = np.inf

        for start in range(n_starts):
            print(f"Start {start+1}/{n_starts}...")

            result = differential_evolution(
                self.objective,
                bounds,
                workers=workers,
                updating='deferred',
                popsize=popsize,
                maxiter=maxiter,
                seed=start,
                disp=False,
                atol=1e-12,
                tol=1e-12
            )

            if result.fun < best_loss:
                best_loss = result.fun
                best_result = result

            print(f"  Loss = {result.fun:.3e}")
            gc.collect()

        return best_result

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("=" * 80)
    print("OPTIMIZATION SETTINGS")
    print("=" * 80)
    print(f"  n_starts:     {N_STARTS}")
    print(f"  workers:      {WORKERS}")
    print(f"  popsize:      {POPSIZE_MULT}*8 = {POPSIZE_MULT * 8}")
    print(f"  maxiter:      {MAXITER}")
    print()

    start_time = time.time()

    # Create fitter
    fitter = ThreeLeptonFitter(beta_fixed=BETA_FIXED)

    # Fit
    print("=" * 80)
    print("FITTING...")
    print("=" * 80)

    result = fitter.fit(
        n_starts=N_STARTS,
        workers=WORKERS,
        popsize=POPSIZE_MULT * 8,
        maxiter=MAXITER
    )

    elapsed = time.time() - start_time

    # Extract results
    R_c_e, U_e, R_c_mu, U_mu, R_c_tau, U_tau, S, C_g = result.x

    # Calculate predicted values
    m_e_pred = fitter.mass_formula(R_c_e, U_e, S, BETA_FIXED)
    m_mu_pred = fitter.mass_formula(R_c_mu, U_mu, S, BETA_FIXED)
    m_tau_pred = fitter.mass_formula(R_c_tau, U_tau, S, BETA_FIXED)

    g_e_pred = fitter.g_factor_formula(U_e, C_g)
    g_mu_pred = fitter.g_factor_formula(U_mu, C_g)
    g_tau_pred = fitter.g_factor_formula(U_tau, C_g)

    # Calculate chi-squared components
    chi2_m_e = ((m_e_pred - M_E) / M_E)**2
    chi2_m_mu = ((m_mu_pred - M_MU) / M_MU)**2
    chi2_m_tau = ((m_tau_pred - M_TAU) / M_TAU)**2
    chi2_mass = chi2_m_e + chi2_m_mu + chi2_m_tau

    chi2_g_e = ((g_e_pred - G_E) / 1e-6)**2
    chi2_g_mu = ((g_mu_pred - G_MU) / 1e-6)**2
    chi2_g_tau = ((g_tau_pred - G_TAU) / 1e-4)**2
    chi2_g = chi2_g_e + chi2_g_mu + chi2_g_tau

    # Print results
    print("\n" + "=" * 80)
    print("RESULTS: β FIXED BY α CONSTRAINT")
    print("=" * 80)

    print(f"\nβ = {BETA_FIXED:.6f} (FIXED)")
    print(f"\nFitted parameters:")
    print(f"  S   = {S:.6f}")
    print(f"  C_g = {C_g:.2f}")
    print()
    print(f"  R_c_e = {R_c_e:.6f},  U_e = {U_e:.8f}  ({100*U_e:.4f}%)")
    print(f"  R_c_μ = {R_c_mu:.6f},  U_μ = {U_mu:.8f}  ({100*U_mu:.4f}%)")
    print(f"  R_c_τ = {R_c_tau:.6f},  U_τ = {U_tau:.8f}  ({100*U_tau:.2f}%)")

    print(f"\nχ²_total = {result.fun:.3e}")
    print(f"  χ²_mass = {chi2_mass:.3e}")
    print(f"  χ²_g    = {chi2_g:.3e}")

    print(f"\nMass fits:")
    print(f"  e:  {m_e_pred:.6f} MeV (obs: {M_E:.6f}, error: {100*(m_e_pred-M_E)/M_E:.2e}%)")
    print(f"  μ:  {m_mu_pred:.6f} MeV (obs: {M_MU:.6f}, error: {100*(m_mu_pred-M_MU)/M_MU:.2e}%)")
    print(f"  τ:  {m_tau_pred:.6f} MeV (obs: {M_TAU:.6f}, error: {100*(m_tau_pred-M_TAU)/M_TAU:.2e}%)")

    print(f"\ng-factor fits:")
    print(f"  g_e:  {g_e_pred:.10f} (obs: {G_E:.10f}, Δ: {(g_e_pred-G_E)*1e6:.2f} × 10⁻⁶)")
    print(f"  g_μ:  {g_mu_pred:.10f} (obs: {G_MU:.10f}, Δ: {(g_mu_pred-G_MU)*1e6:.2f} × 10⁻⁶)")
    print(f"  g_τ:  {g_tau_pred:.10f} (obs: {G_TAU:.10f}, Δ: {(g_tau_pred-G_TAU)*1e4:.2f} × 10⁻⁴)")

    print(f"\nRuntime: {elapsed:.1f} seconds")

    # Save results
    os.makedirs('results/V22', exist_ok=True)

    results_dict = {
        'beta': BETA_FIXED,
        'S': S,
        'C_g': C_g,
        'R_c_e': R_c_e,
        'U_e': U_e,
        'R_c_mu': R_c_mu,
        'U_mu': U_mu,
        'R_c_tau': R_c_tau,
        'U_tau': U_tau,
        'chi2_total': result.fun,
        'chi2_mass': chi2_mass,
        'chi2_g': chi2_g,
        'm_e_pred': m_e_pred,
        'm_mu_pred': m_mu_pred,
        'm_tau_pred': m_tau_pred,
        'g_e_pred': g_e_pred,
        'g_mu_pred': g_mu_pred,
        'g_tau_pred': g_tau_pred,
        'runtime_sec': elapsed
    }

    df = pd.DataFrame([results_dict])
    output_file = 'results/V22/t3b_fixed_beta_alpha.csv'
    df.to_csv(output_file, index=False)

    print(f"\n✓ Results saved to {output_file}")
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print("""
This fit uses the theoretically-correct β value from the α constraint,
rather than treating β as a free parameter.

Compare these results to:
1. V22 nuclear fit (c2/c1, S values)
2. Previous free-β lepton fits
3. Check if U_τ > 1.0 issue is resolved

If S and C_g match nuclear sector values, this supports β universality!
""")

if __name__ == "__main__":
    main()
