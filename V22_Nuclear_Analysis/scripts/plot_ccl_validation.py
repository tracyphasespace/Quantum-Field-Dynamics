#!/usr/bin/env python3
"""
Create validation plots for V22 Core Compression Law analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path

# Load data
data = pd.read_csv("/home/tracy/development/QFD_SpectralGap/data/raw/ame2020_ccl.csv")
if 'target' in data.columns:
    data['Z'] = data['target']

A = data['A'].values
Z_obs = data['Z'].values

# Load V22 results
results_path = Path("/home/tracy/development/QFD_SpectralGap/V22_Nuclear_Analysis/results/v22_ccl_best_fit.json")
with open(results_path) as f:
    results = json.load(f)

c1 = results['best_fit']['c1']
c2 = results['best_fit']['c2']
r_squared = results['best_fit']['r_squared']

# Predictions
Z_pred = c1 * np.power(A, 2.0/3.0) + c2 * A
residuals = Z_obs - Z_pred

# Create figure
fig = plt.figure(figsize=(14, 10))

# Panel 1: Core Compression Law
ax1 = plt.subplot(2, 2, 1)
ax1.scatter(A, Z_obs, s=5, alpha=0.3, color='gray', label='AME2020 data (2,550 nuclides)')
A_model = np.linspace(A.min(), A.max(), 200)
Z_model = c1 * np.power(A_model, 2.0/3.0) + c2 * A_model
ax1.plot(A_model, Z_model, 'r-', linewidth=2, label=f'V22: Z = {c1:.3f}·A^(2/3) + {c2:.3f}·A')
ax1.set_xlabel('Mass Number A', fontsize=12)
ax1.set_ylabel('Charge Number Z', fontsize=12)
ax1.set_title(f'Core Compression Law (R² = {r_squared:.4f})', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Panel 2: Residuals vs A
ax2 = plt.subplot(2, 2, 2)
ax2.scatter(A, residuals, s=5, alpha=0.5, color='blue')
ax2.axhline(0, color='black', linestyle='--', linewidth=1)
ax2.set_xlabel('Mass Number A', fontsize=12)
ax2.set_ylabel('Residual (Z_obs - Z_pred)', fontsize=12)
ax2.set_title('Residuals vs Mass Number', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Panel 3: Residuals vs Z
ax3 = plt.subplot(2, 2, 3)
ax3.scatter(Z_obs, residuals, s=5, alpha=0.5, color='green')
ax3.axhline(0, color='black', linestyle='--', linewidth=1)
ax3.set_xlabel('Observed Charge Z', fontsize=12)
ax3.set_ylabel('Residual (Z_obs - Z_pred)', fontsize=12)
ax3.set_title('Residuals vs Charge', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Panel 4: Histogram of residuals
ax4 = plt.subplot(2, 2, 4)
ax4.hist(residuals, bins=50, color='purple', alpha=0.7, edgecolor='black')
ax4.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero residual')
ax4.set_xlabel('Residual (Z_obs - Z_pred)', fontsize=12)
ax4.set_ylabel('Count', fontsize=12)
ax4.set_title(f'Residual Distribution (σ = {np.std(residuals):.2f})', fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()

# Save
output_dir = Path("/home/tracy/development/QFD_SpectralGap/V22_Nuclear_Analysis/results")
output_dir.mkdir(exist_ok=True, parents=True)
plt.savefig(output_dir / "ccl_validation_plots.png", dpi=300, bbox_inches='tight')
print(f"✅ Saved validation plots to: {output_dir / 'ccl_validation_plots.png'}")

# Statistics
print("\n" + "=" * 60)
print("CORE COMPRESSION LAW STATISTICS")
print("=" * 60)
print(f"Number of nuclides: {len(A)}")
print(f"Parameters: c1 = {c1:.6f}, c2 = {c2:.6f}")
print(f"R² = {r_squared:.6f}")
print(f"Mean residual = {np.mean(residuals):.4f}")
print(f"Std residual = {np.std(residuals):.4f}")
print(f"Max |residual| = {np.abs(residuals).max():.2f}")
print("=" * 60)
