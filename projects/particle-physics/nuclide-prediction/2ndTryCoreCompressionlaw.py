from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# -------------------------------------------------------
# 1. Load nuclide data and include ALL isotopes
# -------------------------------------------------------
try:
    # Build a path relative to this script file for reliability
    script_dir = Path(__file__).resolve().parent
    # The parser now generates 'NuMass.csv' as per the spec.
    csv_path = script_dir / "NuMass.csv"
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    raise SystemExit(f"Error: NuMass.csv not found. Please run the parser first. Location: {csv_path}")

# Include ALL isotopes (both stable and unstable)
all_isotopes = df.copy()
Q_obs = all_isotopes["Q"].to_numpy()
A_obs = all_isotopes["A"].to_numpy()

print(f"Total isotopes in dataset: {len(all_isotopes)}")
stable_count = len(df[df['Stable'] == 1])
unstable_count = len(df[df['Stable'] == 0])
print(f"Stable isotopes: {stable_count} ({100*stable_count/len(all_isotopes):.1f}%)")
print(f"Unstable isotopes: {unstable_count} ({100*unstable_count/len(all_isotopes):.1f}%)")

# -------------------------------------------------------
# 2. Define and fit the two-term Core Compression model
# -------------------------------------------------------
def model(A, c1, c2):
    """The two-term model for the Core Compression Law: Q = c1*A^(2/3) + c2*A"""
    return c1 * (A ** (2/3)) + c2 * A

# Fit the model to find the best coefficients for ALL isotopes
popt, pcov = curve_fit(model, A_obs, Q_obs)
c1_fit, c2_fit = popt
print(f"\nBest-fit coefficients for all isotopes:")
print(f"  c1 = {c1_fit:.6f}")
print(f"  c2 = {c2_fit:.6f}")
print(f"Model: Q = {c1_fit:.6f} × A^(2/3) + {c2_fit:.6f} × A")

# Calculate the predicted Q for each isotope using the model
Q_pred = model(A_obs, c1_fit, c2_fit)

# Calculate comprehensive statistics
ss_res = np.sum((Q_obs - Q_pred) ** 2)
ss_tot = np.sum((Q_obs - np.mean(Q_obs)) ** 2)
r_squared = 1 - (ss_res / ss_tot)
rmse = np.sqrt(np.mean((Q_obs - Q_pred) ** 2))
max_abs_residual = np.max(np.abs(Q_obs - Q_pred))
residuals = Q_obs - Q_pred

print(f"\nStatistics for all {len(all_isotopes)} isotopes:")
print(f"  R-squared: {r_squared:.8f}")
print(f"  RMSE: {rmse:.4f}")
print(f"  Max |residual|: {max_abs_residual:.4f}")
print(f"  Std deviation of residuals: {np.std(residuals):.4f}")

# -------------------------------------------------------
# 3. Plot the fit for all isotopes (Q vs A)
# -------------------------------------------------------
plt.figure(figsize=(14, 8))

# Separate stable and unstable for different colors
stable_mask = all_isotopes["Stable"] == 1
unstable_mask = all_isotopes["Stable"] == 0

plt.scatter(A_obs[stable_mask], Q_obs[stable_mask], 
           c="blue", s=15, alpha=0.7, label=f"Stable isotopes ({np.sum(stable_mask)})")
plt.scatter(A_obs[unstable_mask], Q_obs[unstable_mask], 
           c="red", s=15, alpha=0.5, label=f"Unstable isotopes ({np.sum(unstable_mask)})")

# Plot the fitted curve
A_fit = np.linspace(1, max(A_obs), 500)
Q_fit = model(A_fit, c1_fit, c2_fit)
plt.plot(A_fit, Q_fit, "black", linewidth=2, 
         label=f"Core Compression Law (R² = {r_squared:.6f})")

plt.xlabel("Mass Number A")
plt.ylabel("Charge Number Q")
plt.title(f"Core Compression Law: All Known Isotopes")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()

# -------------------------------------------------------
# 4. Calculate and plot the residuals
# -------------------------------------------------------
magic_numbers_Q = [2, 8, 20, 28, 50, 82, 126]  # Proton magic numbers

plt.figure(figsize=(14, 8))

# Plot residuals with different colors for stable/unstable
plt.scatter(Q_obs[stable_mask], residuals[stable_mask], 
           c="blue", s=20, alpha=0.7, label="Stable isotopes")
plt.scatter(Q_obs[unstable_mask], residuals[unstable_mask], 
           c="red", s=20, alpha=0.5, label="Unstable isotopes")

# Draw a horizontal line at zero for reference
plt.axhline(0, color='black', linestyle='--', linewidth=1, label="Perfect Fit")

# Draw vertical lines for the magic numbers
for i, mn in enumerate(magic_numbers_Q):
    if mn <= max(Q_obs):  # Only show magic numbers in our data range
        if i == 0:
            plt.axvline(mn, color='green', linestyle=':', linewidth=1.5, alpha=0.7, 
                       label="Proton Magic Numbers")
        else:
            plt.axvline(mn, color='green', linestyle=':', linewidth=1.5, alpha=0.7)

plt.xlabel("Charge Number Q")
plt.ylabel("Residual (Q_observed - Q_predicted)")
plt.title("Residuals: Shell Effects in Core Compression Law")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()

# -------------------------------------------------------
# 5. Core Compression Law: Density Impact Ratio Analysis
# -------------------------------------------------------
def density_impact_ratio(A, c1, c2):
    """Core-to-surface contribution ratio: (3c₂/2c₁) × A^(1/3)"""
    return (3 * c2 / (2 * c1)) * (A ** (1/3))

# Compute ratio for each observed isotope
ratios = density_impact_ratio(A_obs, c1_fit, c2_fit)

plt.figure(figsize=(14, 8))

# Plot ratios with different colors for stable/unstable
plt.scatter(A_obs[stable_mask], ratios[stable_mask], 
           c="blue", s=20, alpha=0.7, label="Stable isotopes")
plt.scatter(A_obs[unstable_mask], ratios[unstable_mask], 
           c="red", s=20, alpha=0.5, label="Unstable isotopes")

# Overlay smooth theoretical curve
A_fit_ratio = np.linspace(1, max(A_obs), 300)
ratio_fit = density_impact_ratio(A_fit_ratio, c1_fit, c2_fit)
plt.plot(A_fit_ratio, ratio_fit, "black", linewidth=2, 
         label=f"Ratio(A) = (3c₂/2c₁)·A^(1/3)")

plt.xlabel("Mass Number A")
plt.ylabel("Density Impact Ratio (Core / Surface)")
plt.title("Core Compression Law: All Known Nuclides")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()

# -------------------------------------------------------
# 6. Statistical analysis by isotope type
# -------------------------------------------------------
# Fit separate models for comparison
stable_data = all_isotopes[all_isotopes["Stable"] == 1]
if len(stable_data) > 0:
    Q_stable = stable_data["Q"].to_numpy()
    A_stable = stable_data["A"].to_numpy()
    popt_stable, _ = curve_fit(model, A_stable, Q_stable)
    c1_stable, c2_stable = popt_stable
    
    Q_pred_stable = model(A_stable, c1_stable, c2_stable)
    r_squared_stable = 1 - np.sum((Q_stable - Q_pred_stable)**2) / np.sum((Q_stable - np.mean(Q_stable))**2)
    rmse_stable = np.sqrt(np.mean((Q_stable - Q_pred_stable)**2))
    
    print(f"\nStable isotopes only ({len(stable_data)} isotopes):")
    print(f"  Coefficients: c1 = {c1_stable:.6f}, c2 = {c2_stable:.6f}")
    print(f"  R-squared: {r_squared_stable:.8f}")
    print(f"  RMSE: {rmse_stable:.4f}")

# Statistics for unstable isotopes using the all-isotope model
unstable_residuals = residuals[unstable_mask]
ss_res_unstable = np.sum(unstable_residuals ** 2)
ss_tot_unstable = np.sum((Q_obs[unstable_mask] - np.mean(Q_obs[unstable_mask])) ** 2)
r_squared_unstable = 1 - ss_res_unstable / ss_tot_unstable
rmse_unstable = np.sqrt(np.mean(unstable_residuals ** 2))

print(f"\nUnstable isotopes ({np.sum(unstable_mask)} isotopes, using all-isotope fit):")
print(f"  R-squared: {r_squared_unstable:.8f}")
print(f"  RMSE: {rmse_unstable:.4f}")

# -------------------------------------------------------
# 7. Final summary
# -------------------------------------------------------
print("\n" + "="*80)
print("CORE COMPRESSION LAW - Preliminary FINDINGS")
print("="*80)
print("This analysis reveals demonstrable order in nuclear structure:")
print(f"•  {len(all_isotopes)} known isotopes follow the same law (R² = {r_squared:.6f})")
print(f"• Even {unstable_count} unstable isotopes fit  well (R² = {r_squared_unstable:.6f})")
print("• Traditional physics: usually only stable isotopes are predictable")
print("• Our Model: ALL isotopes follow universal scaling")
print()
print("Physical interpretation:")
print(f"• c1 term ({c1_fit:.6f}): Surface effects ∝ A^(2/3)")
print(f"• c2 term ({c2_fit:.6f}): Core compression ∝ A")
print("• The ratio plot shows universal core/surface scaling")
print()
print("This is useful now but a higher order model is being pursued.")
print("="*80)