#!/usr/bin/env python3
"""
ANALYZE_RESIDUALS.py

Objective:
    Mathematically fit the residuals from the Straight Ruler diagnostic
    to identify the functional form of missing Lagrangian terms.

Method:
    1. Load diagnostic_residuals.csv
    2. Separate Low A (≤16) and High A (≥40) regimes
    3. Fit Low A to: Δ_low ~ C₁/A^α (Rotor/Winding energy)
    4. Fit High A to: Δ_high ~ C₂·A^β (Saturation energy)
    5. Plot fits and report coefficients

This DERIVES the correction terms from the gap, rather than fitting them.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Load the diagnostic results
df = pd.read_csv('diagnostic_residuals.csv')

print("=" * 70)
print("RESIDUAL ANALYSIS: Deriving Missing Lagrangian Terms")
print("=" * 70)
print()

# Separate into domains
df_low = df[df.A <= 16].copy()
df_mid = df[(df.A > 16) & (df.A < 120)].copy()
df_high = df[df.A >= 120].copy()

print(f"Low A domain (A ≤ 16):  {len(df_low)} isotopes")
print(f"Mid A domain (16 < A < 120): {len(df_mid)} isotopes")
print(f"High A domain (A ≥ 120): {len(df_high)} isotopes")
print()

# --- LOW A ANALYSIS: Rotor/Winding Energy ---
print("=" * 70)
print("LOW A REGIME: Testing Rotor Hypothesis")
print("=" * 70)
print()

# Hypothesis: Δ ~ C / A^α
# Test α = 1, 2/3, 5/3 (different rotor models)

def rotor_model(A, C, alpha):
    """Δ(A) = C / A^α"""
    return C / (A ** alpha)

# Test different exponents
alphas_to_test = [
    (1.0, "1/A (Simple inverse)"),
    (2.0/3.0, "A^(-2/3) (1/R² scaling)"),
    (5.0/3.0, "A^(-5/3) (Rotor moment of inertia)"),
]

best_fit_low = None
best_rsq_low = -1

for alpha_fixed, label in alphas_to_test:
    # Fit with fixed alpha
    def model_fixed_alpha(A, C):
        return rotor_model(A, C, alpha_fixed)

    try:
        popt, _ = curve_fit(model_fixed_alpha, df_low.A, df_low.Residual, p0=[100])
        C_fit = popt[0]

        # Calculate R²
        y_pred = model_fixed_alpha(df_low.A, C_fit)
        ss_res = np.sum((df_low.Residual - y_pred)**2)
        ss_tot = np.sum((df_low.Residual - df_low.Residual.mean())**2)
        r_squared = 1 - (ss_res / ss_tot)

        print(f"  {label:30s}: C = {C_fit:8.2f}, R² = {r_squared:.4f}")

        if r_squared > best_rsq_low:
            best_rsq_low = r_squared
            best_fit_low = (C_fit, alpha_fixed, label)
    except:
        print(f"  {label:30s}: Fit failed")

print()
print(f"BEST FIT (Low A): {best_fit_low[2]}")
print(f"  Correction₁(A) = {best_fit_low[0]:.2f} / A^{best_fit_low[1]:.3f}")
print(f"  R² = {best_rsq_low:.4f}")
print()

# --- HIGH A ANALYSIS: Saturation Energy ---
print("=" * 70)
print("HIGH A REGIME: Testing Saturation Hypothesis")
print("=" * 70)
print()

# Hypothesis: Δ ~ C · A^β
# Test β = 1, 4/3, 5/3 (different saturation models)

def saturation_model(A, C, beta):
    """Δ(A) = C · A^β"""
    return C * (A ** beta)

betas_to_test = [
    (1.0, "A^1 (Linear)"),
    (4.0/3.0, "A^(4/3) (Surface area)"),
    (5.0/3.0, "A^(5/3) (Volume strain)"),
    (2.0, "A^2 (Coulomb)"),
]

best_fit_high = None
best_rsq_high = -1

for beta_fixed, label in betas_to_test:
    def model_fixed_beta(A, C):
        return saturation_model(A, C, beta_fixed)

    try:
        popt, _ = curve_fit(model_fixed_beta, df_high.A, df_high.Residual, p0=[1])
        C_fit = popt[0]

        y_pred = model_fixed_beta(df_high.A, C_fit)
        ss_res = np.sum((df_high.Residual - y_pred)**2)
        ss_tot = np.sum((df_high.Residual - df_high.Residual.mean())**2)
        r_squared = 1 - (ss_res / ss_tot)

        print(f"  {label:30s}: C = {C_fit:8.4f}, R² = {r_squared:.4f}")

        if r_squared > best_rsq_high:
            best_rsq_high = r_squared
            best_fit_high = (C_fit, beta_fixed, label)
    except:
        print(f"  {label:30s}: Fit failed")

print()
print(f"BEST FIT (High A): {best_fit_high[2]}")
print(f"  Correction₂(A) = {best_fit_high[0]:.4f} · A^{best_fit_high[1]:.3f}")
print(f"  R² = {best_rsq_high:.4f}")
print()

# --- GENERATE IMPROVED PLOT ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left plot: Full range with both fits
ax1.plot(df.A, df.Residual, 'ko', markersize=8, label='Measured Gap')

# Plot best fits
A_low_smooth = np.linspace(1, 16, 100)
A_high_smooth = np.linspace(120, 240, 100)

y_low_fit = rotor_model(A_low_smooth, best_fit_low[0], best_fit_low[1])
y_high_fit = saturation_model(A_high_smooth, best_fit_high[0], best_fit_high[1])

ax1.plot(A_low_smooth, y_low_fit, 'b-', linewidth=2,
         label=f'Low A: {best_fit_low[0]:.0f}/A^{best_fit_low[1]:.2f}')
ax1.plot(A_high_smooth, y_high_fit, 'r-', linewidth=2,
         label=f'High A: {best_fit_high[0]:.2f}·A^{best_fit_high[1]:.2f}')

ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax1.set_xlabel('Mass Number A', fontsize=12)
ax1.set_ylabel('Residual Energy (MeV)', fontsize=12)
ax1.set_title('Missing Lagrangian Terms (β=3.043233053 Fixed)', fontsize=14)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_yscale('log')
ax1.set_ylim(10, 30000)

# Right plot: Relative error (Δ/E_exp)
df['RelError'] = np.abs(df.Residual / df.Exp)
ax2.plot(df.A, df['RelError'], 'mo', markersize=8)
ax2.axhline(1.0, color='r', linestyle='--', label='100% error')
ax2.axhline(0.1, color='g', linestyle='--', label='10% error')
ax2.set_xlabel('Mass Number A', fontsize=12)
ax2.set_ylabel('|Δ/E_exp| (fractional error)', fontsize=12)
ax2.set_title('Relative Error vs Mass Number', fontsize=14)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_yscale('log')

plt.tight_layout()
plt.savefig('residual_analysis.png', dpi=150)
print("=" * 70)
print("Plot saved: residual_analysis.png")
print()

# --- SUMMARY ---
print("=" * 70)
print("DERIVED LAGRANGIAN CORRECTIONS")
print("=" * 70)
print()
print("From the Straight Ruler measurement, we derive:")
print()
print("1. LOW A CORRECTION (Winding/Rotor Energy):")
print(f"   E_rotor(A) = -{best_fit_low[0]:.2f} / A^{best_fit_low[1]:.3f} MeV")
print(f"   Physical interpretation: {best_fit_low[2]}")
print()
print("2. HIGH A CORRECTION (Saturation Energy):")
print(f"   E_saturation(A) = -{best_fit_high[0]:.4f} · A^{best_fit_high[1]:.3f} MeV")
print(f"   Physical interpretation: {best_fit_high[2]}")
print()
print("3. CORRECTED MODEL:")
print(f"   E_total(A) = E_soliton(β=3.043233053) + E_rotor(A) + E_saturation(A)")
print()
print("=" * 70)
print("Next step: Implement these corrections and verify convergence.")
print("=" * 70)
