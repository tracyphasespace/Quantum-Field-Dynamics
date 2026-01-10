#!/usr/bin/env python3
"""
ANALYZE_RESIDUALS_V2.py

More flexible fitting approach that handles the actual residual patterns.
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
df_high = df[df.A >= 120].copy()

print("Data Summary:")
print(f"  Low A (≤16):  {len(df_low)} points, Δ range: {df_low.Residual.min():.1f} to {df_low.Residual.max():.1f} MeV")
print(f"  High A (≥120): {len(df_high)} points, Δ range: {df_high.Residual.min():.1f} to {df_high.Residual.max():.1f} MeV")
print()

# --- LOW A ANALYSIS ---
print("=" * 70)
print("LOW A REGIME (A ≤ 16)")
print("=" * 70)
print()

# Try general power law: Δ = C · A^α (allowing free α)
def power_law_general(A, C, alpha):
    return C * (A ** alpha)

# Fit with both parameters free
try:
    popt_low, pcov_low = curve_fit(power_law_general, df_low.A, df_low.Residual,
                                     p0=[50, 0.5], bounds=([0, -5], [1000, 5]))
    C_low, alpha_low = popt_low

    y_pred_low = power_law_general(df_low.A, C_low, alpha_low)
    ss_res = np.sum((df_low.Residual - y_pred_low)**2)
    ss_tot = np.sum((df_low.Residual - df_low.Residual.mean())**2)
    r_squared_low = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    print(f"Best fit: Δ_low(A) = {C_low:.2f} · A^{alpha_low:.3f}")
    print(f"R² = {r_squared_low:.4f}")

    # Check if it's closer to inverse or direct power
    if alpha_low < 0:
        print(f"Interpretation: Inverse scaling ~ 1/A^{abs(alpha_low):.2f} (Rotor-like)")
    else:
        print(f"Interpretation: Direct scaling ~ A^{alpha_low:.2f} (Unexpected!)")
except Exception as e:
    print(f"Fit failed: {e}")
    C_low, alpha_low = 50, 0.5
    r_squared_low = -1

print()

# --- HIGH A ANALYSIS ---
print("=" * 70)
print("HIGH A REGIME (A ≥ 120)")
print("=" * 70)
print()

# For high A, try: Δ = C · A^β
try:
    popt_high, pcov_high = curve_fit(power_law_general, df_high.A, df_high.Residual,
                                      p0=[1, 2], bounds=([0, 0], [1000, 5]))
    C_high, beta_high = popt_high

    y_pred_high = power_law_general(df_high.A, C_high, beta_high)
    ss_res = np.sum((df_high.Residual - y_pred_high)**2)
    ss_tot = np.sum((df_high.Residual - df_high.Residual.mean())**2)
    r_squared_high = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    print(f"Best fit: Δ_high(A) = {C_high:.4f} · A^{beta_high:.3f}")
    print(f"R² = {r_squared_high:.4f}")

    if 1.5 < beta_high < 2.5:
        print(f"Interpretation: Quadratic-ish ~ A² (Coulomb-like)")
    elif 1.0 < beta_high < 1.5:
        print(f"Interpretation: Linear-ish ~ A (Surface tension)")
    else:
        print(f"Interpretation: Strong nonlinearity")
except Exception as e:
    print(f"Fit failed: {e}")
    C_high, beta_high = 1, 2
    r_squared_high = -1

print()

# --- VISUAL ANALYSIS ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Top left: Log-log plot (reveals power laws)
ax = axes[0, 0]
ax.loglog(df.A, np.abs(df.Residual), 'ko', markersize=8, label='|Residual|')
A_smooth = np.linspace(1, 240, 500)
if r_squared_low > -0.5:
    y_low = power_law_general(A_smooth[A_smooth <= 16], C_low, alpha_low)
    ax.loglog(A_smooth[A_smooth <= 16], y_low, 'b-', linewidth=2, label=f'Low A fit: A^{alpha_low:.2f}')
if r_squared_high > -0.5:
    y_high = power_law_general(A_smooth[A_smooth >= 120], C_high, beta_high)
    ax.loglog(A_smooth[A_smooth >= 120], y_high, 'r-', linewidth=2, label=f'High A fit: A^{beta_high:.2f}')
ax.set_xlabel('A (log scale)')
ax.set_ylabel('|Residual| (log scale)')
ax.set_title('Power Law Analysis')
ax.legend()
ax.grid(True, alpha=0.3, which='both')

# Top right: Linear scale
ax = axes[0, 1]
ax.plot(df.A, df.Residual, 'ko-', markersize=8)
ax.axhline(0, color='gray', linestyle='--')
ax.set_xlabel('Mass Number A')
ax.set_ylabel('Residual (MeV)')
ax.set_title('Raw Residuals vs A')
ax.grid(True, alpha=0.3)

# Bottom left: Residual per nucleon
ax = axes[1, 0]
df['Residual_per_A'] = df.Residual / df.A
ax.plot(df.A, df['Residual_per_A'], 'mo-', markersize=8)
ax.axhline(0, color='gray', linestyle='--')
ax.set_xlabel('Mass Number A')
ax.set_ylabel('Residual per nucleon (MeV/A)')
ax.set_title('Missing Energy per Nucleon')
ax.grid(True, alpha=0.3)

# Bottom right: Test specific scalings
ax = axes[1, 1]
# Test if residual follows Z²/A scaling (Coulomb-like)
df['Z2_over_A'] = (df.Z ** 2) / df.A
ax.plot(df['Z2_over_A'], df.Residual, 'go', markersize=8)
ax.set_xlabel('Z²/A')
ax.set_ylabel('Residual (MeV)')
ax.set_title('Coulomb Scaling Test')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('residual_analysis_v2.png', dpi=150)

print("=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
print()
print("Key Observations:")
print(f"  1. Low A follows: Δ ~ {C_low:.1f} · A^{alpha_low:.2f} (R²={r_squared_low:.3f})")
print(f"  2. High A follows: Δ ~ {C_high:.2f} · A^{beta_high:.2f} (R²={r_squared_high:.3f})")
print()
print("If R² < 0: The model doesn't fit the data well.")
print("This suggests the residuals DON'T follow simple power laws.")
print("Possible reasons:")
print("  - Multiple competing effects at each A")
print("  - Shell effects dominating over smooth trends")
print("  - The 'compressed branch' problem affecting all A differently")
print()
print("Plots saved: residual_analysis_v2.png")
