#!/usr/bin/env python3
"""
Test different fitting approaches for charge-poor track.

Question: Why does charge-poor have c₁=0?
- Is it the bounds constraint clamping negative c₁ to zero?
- Or is the data genuinely linear (no A^(2/3) term)?

Test by removing bounds and visualizing the data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Phase 1 reference parameters
C1_REF = 0.496296
C2_REF = 0.323671


def backbone(A, c1, c2):
    """Q(A) = c1*A^(2/3) + c2*A"""
    return c1 * (A ** (2/3)) + c2 * A


def classify_track(Z, A, threshold=2.5):
    """Classify nucleus into charge regime"""
    Q_ref = backbone(A, C1_REF, C2_REF)
    deviation = Z - Q_ref

    if deviation > threshold:
        return 'charge_rich'
    elif deviation < -threshold:
        return 'charge_poor'
    else:
        return 'charge_nominal'


print("=" * 80)
print("Charge-Poor Track Investigation")
print("=" * 80)
print()

# Load data
df = pd.read_csv("../NuMass.csv")
print(f"Loaded {len(df)} isotopes")

# Classify
df['track'] = df.apply(lambda row: classify_track(row['Q'], row['A'], threshold=2.5), axis=1)

# Extract charge-poor data
charge_poor = df[df['track'] == 'charge_poor'].copy()
A_poor = charge_poor['A'].values
Q_poor = charge_poor['Q'].values

print(f"Charge-poor isotopes: {len(charge_poor)}")
print()

# ============================================================================
# Test 1: Fit WITH bounds (current approach)
# ============================================================================
print("=" * 80)
print("TEST 1: Fit with bounds c₁ ≥ 0 (current)")
print("=" * 80)
print()

try:
    popt_bounded, _ = curve_fit(backbone, A_poor, Q_poor,
                                p0=[C1_REF, C2_REF],
                                bounds=([0, 0], [2, 1]))
    c1_bounded, c2_bounded = popt_bounded

    Q_pred_bounded = backbone(A_poor, c1_bounded, c2_bounded)
    rmse_bounded = np.sqrt(np.mean((Q_poor - Q_pred_bounded) ** 2))

    print(f"c₁ = {c1_bounded:.6f}")
    print(f"c₂ = {c2_bounded:.6f}")
    print(f"RMSE = {rmse_bounded:.4f} Z")

    if abs(c1_bounded) < 1e-6:
        print("⚠️  c₁ ≈ 0 - hit lower bound constraint!")
    print()

except Exception as e:
    print(f"Fit failed: {e}")
    print()
    c1_bounded, c2_bounded, rmse_bounded = 0, 0, 999

# ============================================================================
# Test 2: Fit WITHOUT bounds
# ============================================================================
print("=" * 80)
print("TEST 2: Fit without bounds (allow negative c₁)")
print("=" * 80)
print()

try:
    popt_unbounded, _ = curve_fit(backbone, A_poor, Q_poor,
                                  p0=[C1_REF, C2_REF])
    c1_unbounded, c2_unbounded = popt_unbounded

    Q_pred_unbounded = backbone(A_poor, c1_unbounded, c2_unbounded)
    rmse_unbounded = np.sqrt(np.mean((Q_poor - Q_pred_unbounded) ** 2))

    print(f"c₁ = {c1_unbounded:.6f}")
    print(f"c₂ = {c2_unbounded:.6f}")
    print(f"RMSE = {rmse_unbounded:.4f} Z")

    if c1_unbounded < 0:
        print("✓ c₁ is NEGATIVE - this is the natural fit!")
    print()

except Exception as e:
    print(f"Fit failed: {e}")
    print()
    c1_unbounded, c2_unbounded, rmse_unbounded = 0, 0, 999

# ============================================================================
# Test 3: Pure linear (Q = c₂·A)
# ============================================================================
print("=" * 80)
print("TEST 3: Pure linear Q = c₂·A (force c₁=0)")
print("=" * 80)
print()

def linear_only(A, c2):
    return c2 * A

popt_linear, _ = curve_fit(linear_only, A_poor, Q_poor, p0=[C2_REF])
c2_linear = popt_linear[0]

Q_pred_linear = linear_only(A_poor, c2_linear)
rmse_linear = np.sqrt(np.mean((Q_poor - Q_pred_linear) ** 2))

print(f"c₁ = 0.000000 (fixed)")
print(f"c₂ = {c2_linear:.6f}")
print(f"RMSE = {rmse_linear:.4f} Z")
print()

# ============================================================================
# Test 4: Pure surface (Q = c₁·A^(2/3))
# ============================================================================
print("=" * 80)
print("TEST 4: Pure surface Q = c₁·A^(2/3) (force c₂=0)")
print("=" * 80)
print()

def surface_only(A, c1):
    return c1 * (A ** (2/3))

popt_surface, _ = curve_fit(surface_only, A_poor, Q_poor, p0=[C1_REF])
c1_surface = popt_surface[0]

Q_pred_surface = surface_only(A_poor, c1_surface)
rmse_surface = np.sqrt(np.mean((Q_poor - Q_pred_surface) ** 2))

print(f"c₁ = {c1_surface:.6f}")
print(f"c₂ = 0.000000 (fixed)")
print(f"RMSE = {rmse_surface:.4f} Z")
print()

# ============================================================================
# Comparison
# ============================================================================
print("=" * 80)
print("COMPARISON")
print("=" * 80)
print()

results = [
    ("Bounded (c₁≥0)", c1_bounded, c2_bounded, rmse_bounded),
    ("Unbounded", c1_unbounded, c2_unbounded, rmse_unbounded),
    ("Pure Linear", 0.0, c2_linear, rmse_linear),
    ("Pure Surface", c1_surface, 0.0, rmse_surface),
]

print(f"{'Model':<20} {'c₁':>10} {'c₂':>10} {'RMSE':>10} {'ΔvsUnbounded':>15}")
print("-" * 80)
for name, c1, c2, rmse in results:
    delta = rmse - rmse_unbounded
    marker = "←" if abs(delta) < 0.01 else ""
    print(f"{name:<20} {c1:10.6f} {c2:10.6f} {rmse:10.4f} {delta:+14.4f} {marker}")
print()

if rmse_unbounded < rmse_bounded:
    improvement = rmse_bounded - rmse_unbounded
    pct = 100 * improvement / rmse_bounded
    print(f"✓ Unbounded is BETTER by {improvement:.4f} Z ({pct:.2f}% improvement)")
else:
    print(f"⚠️ Bounded is actually better (unexpected)")
print()

# ============================================================================
# Visualization
# ============================================================================
print("=" * 80)
print("GENERATING VISUALIZATION")
print("=" * 80)
print()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Left: Data + different fits
A_range = np.linspace(A_poor.min(), A_poor.max(), 500)

ax1.scatter(A_poor, Q_poor, s=3, alpha=0.4, c='black', label='Data')

# Bounded fit
Q_bounded = backbone(A_range, c1_bounded, c2_bounded)
ax1.plot(A_range, Q_bounded, 'r-', linewidth=2,
         label=f'Bounded (c₁≥0): RMSE={rmse_bounded:.3f}')

# Unbounded fit
Q_unbounded = backbone(A_range, c1_unbounded, c2_unbounded)
ax1.plot(A_range, Q_unbounded, 'b--', linewidth=2,
         label=f'Unbounded: RMSE={rmse_unbounded:.3f}')

# Linear only
Q_linear = linear_only(A_range, c2_linear)
ax1.plot(A_range, Q_linear, 'g:', linewidth=2,
         label=f'Linear only: RMSE={rmse_linear:.3f}')

ax1.set_xlabel('Mass Number A', fontsize=12)
ax1.set_ylabel('Charge Z', fontsize=12)
ax1.set_title('Charge-Poor Track: Fit Comparison', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Right: Residuals
residuals_bounded = Q_poor - backbone(A_poor, c1_bounded, c2_bounded)
residuals_unbounded = Q_poor - backbone(A_poor, c1_unbounded, c2_unbounded)

ax2.scatter(A_poor, residuals_bounded, s=3, alpha=0.5, c='red', label='Bounded')
ax2.scatter(A_poor, residuals_unbounded, s=3, alpha=0.5, c='blue', label='Unbounded')
ax2.axhline(0, color='black', linestyle='--', linewidth=1)

ax2.set_xlabel('Mass Number A', fontsize=12)
ax2.set_ylabel('Residual (Z_actual - Z_pred)', fontsize=12)
ax2.set_title('Residuals: Bounded vs Unbounded', fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('charge_poor_investigation.png', dpi=150)
print("✓ Saved: charge_poor_investigation.png")
print()

# ============================================================================
# Physical Interpretation
# ============================================================================
print("=" * 80)
print("PHYSICAL INTERPRETATION")
print("=" * 80)
print()

if c1_unbounded < 0:
    print("FINDING: c₁ is naturally NEGATIVE for charge-poor track")
    print()
    print("Interpretation in QFD Soliton Picture:")
    print(f"  Q = {c1_unbounded:.4f}·A^(2/3) + {c2_unbounded:.4f}·A")
    print()
    print("  Negative surface term (c₁ < 0):")
    print("    - Boundary curvature opposes charge accumulation")
    print("    - Charge-deficit soliton fields have inverted surface tension")
    print("    - Volume term dominates (c₂·A positive and large)")
    print()
    print("  Physical meaning:")
    print("    - Charge-poor = low charge density distribution")
    print("    - Surface effects REDUCE charge (not increase)")
    print("    - Bulk packing dominates charge scaling")
    print()
elif abs(c1_unbounded) < 0.01:
    print("FINDING: c₁ ≈ 0 even without constraint")
    print()
    print("Interpretation:")
    print("  - Charge-poor nuclei genuinely have minimal surface contribution")
    print("  - Pure volume scaling Q ≈ 0.385·A")
    print("  - Different geometric regime than charge-rich/nominal")
    print()
else:
    print(f"FINDING: c₁ = {c1_unbounded:.4f} (positive)")
    print()
    print("Interpretation:")
    print("  - Standard scaling preserved")
    print("  - Bounded constraint was not the issue")
    print()

# ============================================================================
# Recommendation
# ============================================================================
print("=" * 80)
print("RECOMMENDATION")
print("=" * 80)
print()

if rmse_unbounded < rmse_bounded - 0.05:
    print("✓ REMOVE BOUNDS CONSTRAINT")
    print(f"  - Improvement: {rmse_bounded - rmse_unbounded:.4f} Z")
    print(f"  - Allow negative c₁ for charge-poor track")
    print(f"  - Update three_track_ccl.py to use unbounded fit")
    print()
    print("  Code change:")
    print("    popt, _ = curve_fit(backbone, A_track, Q_track,")
    print("                        p0=[C1_REF, C2_REF])")
    print("    # Remove: bounds=([0, 0], [2, 1])")
elif abs(rmse_unbounded - rmse_bounded) < 0.05:
    print("⚠️ MINIMAL DIFFERENCE")
    print(f"  - Bounded: {rmse_bounded:.4f} Z")
    print(f"  - Unbounded: {rmse_unbounded:.4f} Z")
    print(f"  - Constraint doesn't significantly affect fit")
    print()
    if abs(c1_unbounded) < 0.01:
        print("  → Charge-poor genuinely has c₁ ≈ 0")
        print("  → Consider pure linear model for this track")
else:
    print("⚠️ BOUNDED IS BETTER (unexpected!)")
    print(f"  - Keep current constraint")

print()
print("=" * 80)
print("DONE")
print("=" * 80)
