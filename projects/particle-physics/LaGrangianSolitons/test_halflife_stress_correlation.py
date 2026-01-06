#!/usr/bin/env python3
"""
HALF-LIFE vs GEOMETRIC STRESS CORRELATION TEST
================================================================================
Test hypothesis: t_1/2 ∝ exp(-k × σ^n) where σ = |N| is geometric stress

Key insight: Z and A are INTEGERS (topological winding numbers)
- Z = electric charge (quantized)
- A = baryon number (quantized)
- Residuals exist but are small (mass defect, binding energy)

================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from scipy.optimize import curve_fit

# 15-Path Model Parameters
c1_0 = 0.970454
c2_0 = 0.234920
c3_0 = -1.928732
dc1 = -0.021538
dc2 = 0.001730
dc3 = -0.540530

def calculate_N_continuous(A, Z):
    """Calculate geometric coordinate N(A,Z) - A and Z are integers!"""
    if A < 1:
        return 0
    A_term = A**(2/3)
    Z_0 = c1_0 * A_term + c2_0 * A + c3_0
    dZ = dc1 * A_term + dc2 * A + dc3
    if abs(dZ) < 1e-10:
        return 0
    return (Z - Z_0) / dZ

def halflife_to_seconds(value, unit):
    """Convert half-life to seconds"""
    conversions = {
        'years': 365.25 * 24 * 3600,
        'days': 24 * 3600,
        'hours': 3600,
        'minutes': 60,
        'seconds': 1,
        'ms': 1e-3,
        'μs': 1e-6,
        'ns': 1e-9,
    }
    return value * conversions.get(unit, 1)

# Comprehensive radioactive isotope database with half-lives
# Format: (Name, Z, A, Half-life value, Half-life unit, Decay mode)
radioactive_database = [
    # Very short-lived
    ("Be-8", 4, 8, 8.19e-17, 'seconds', "α"),

    # Short-lived
    ("F-18", 9, 18, 109.8, 'minutes', "β+"),
    ("O-15", 8, 15, 122.2, 'seconds', "β+"),
    ("N-13", 7, 13, 9.97, 'minutes', "β+"),
    ("C-11", 6, 11, 20.4, 'minutes', "β+"),

    # Intermediate (days to years)
    ("I-131", 53, 131, 8.02, 'days', "β-"),
    ("P-32", 15, 32, 14.3, 'days', "β-"),
    ("S-35", 16, 35, 87.5, 'days', "β-"),
    ("Ca-45", 20, 45, 163, 'days', "β-"),
    ("Fe-55", 26, 55, 2.74, 'years', "EC"),
    ("Co-60", 27, 60, 5.27, 'years', "β-"),
    ("H-3", 1, 3, 12.3, 'years', "β-"),
    ("Kr-85", 36, 85, 10.8, 'years', "β-"),

    # Long-lived (thousands to millions of years)
    ("C-14", 6, 14, 5730, 'years', "β-"),
    ("Cl-36", 17, 36, 3.01e5, 'years', "β-"),
    ("Sr-90", 38, 90, 28.8, 'years', "β-"),
    ("Cs-137", 55, 137, 30.2, 'years', "β-"),
    ("Ra-226", 88, 226, 1600, 'years', "α"),
    ("Pu-239", 94, 239, 24110, 'years', "α"),
    ("Am-241", 95, 241, 432.2, 'years', "α"),
    ("Np-237", 93, 237, 2.14e6, 'years', "α"),

    # Very long-lived (billions of years)
    ("K-40", 19, 40, 1.25e9, 'years', "β-/EC"),
    ("U-235", 92, 235, 7.04e8, 'years', "α"),
    ("U-238", 92, 238, 4.47e9, 'years', "α"),
    ("Th-232", 90, 232, 1.41e10, 'years', "α"),

    # Additional isotopes across stress range
    ("Na-22", 11, 22, 2.60, 'years', "β+"),
    ("Na-24", 11, 24, 15.0, 'hours', "β-"),
    ("Ar-37", 18, 37, 35.0, 'days', "EC"),
    ("Mn-54", 25, 54, 312, 'days', "EC"),
    ("Co-57", 27, 57, 271.8, 'days', "EC"),
    ("Zn-65", 30, 65, 244.3, 'days', "EC"),
    ("Se-75", 34, 75, 119.8, 'days', "EC"),
    ("Tc-99", 43, 99, 2.11e5, 'years', "β-"),
    ("Ru-106", 44, 106, 373.6, 'days', "β-"),
    ("Ag-110m", 47, 110, 249.8, 'days', "β-"),
    ("Cd-109", 48, 109, 462.0, 'days', "EC"),
    ("I-129", 53, 129, 1.57e7, 'years', "β-"),
    ("Pm-147", 61, 147, 2.62, 'years', "β-"),
    ("Sm-151", 62, 151, 90, 'years', "β-"),
]

print("="*80)
print("HALF-LIFE vs GEOMETRIC STRESS CORRELATION")
print("="*80)
print()
print(f"Testing {len(radioactive_database)} radioactive isotopes")
print()

# Calculate stress coordinates and convert half-lives
data = []
for name, Z, A, t_half_val, t_half_unit, decay_mode in radioactive_database:
    # Z and A are INTEGERS (topological winding numbers)
    assert isinstance(Z, int) and isinstance(A, int), f"{name}: Z and A must be integers!"

    N_coord = calculate_N_continuous(A, Z)
    stress = abs(N_coord)
    t_half_seconds = halflife_to_seconds(t_half_val, t_half_unit)

    data.append({
        'name': name,
        'Z': Z,
        'A': A,
        'N': N_coord,
        'stress': stress,
        't_half_s': t_half_seconds,
        't_half_val': t_half_val,
        't_half_unit': t_half_unit,
        'decay': decay_mode
    })

# Convert to arrays
names = [d['name'] for d in data]
Z_vals = np.array([d['Z'] for d in data])
A_vals = np.array([d['A'] for d in data])
N_vals = np.array([d['N'] for d in data])
stress_vals = np.array([d['stress'] for d in data])
t_half_vals = np.array([d['t_half_s'] for d in data])
log_t_half = np.log10(t_half_vals)

print("Stress range:")
print(f"  Min: {np.min(stress_vals):.3f}")
print(f"  Max: {np.max(stress_vals):.3f}")
print()

print("Half-life range:")
print(f"  Min: {np.min(t_half_vals):.2e} seconds ({names[np.argmin(t_half_vals)]})")
print(f"  Max: {np.max(t_half_vals):.2e} seconds ({names[np.argmax(t_half_vals)]})")
print(f"  Span: {np.log10(np.max(t_half_vals)/np.min(t_half_vals)):.1f} orders of magnitude")
print()

# Test correlations
print("="*80)
print("CORRELATION TESTS")
print("="*80)
print()

# Test 1: Linear in stress
corr_linear, p_linear = pearsonr(stress_vals, log_t_half)
print(f"Test 1: log(t_1/2) vs σ (linear)")
print(f"  Pearson r = {corr_linear:.4f}")
print(f"  p-value = {p_linear:.2e}")
print(f"  Interpretation: {'Significant' if p_linear < 0.05 else 'Not significant'}")
print()

# Test 2: Quadratic in stress
corr_quad, p_quad = pearsonr(stress_vals**2, log_t_half)
print(f"Test 2: log(t_1/2) vs σ² (quadratic)")
print(f"  Pearson r = {corr_quad:.4f}")
print(f"  p-value = {p_quad:.2e}")
print(f"  Interpretation: {'Significant' if p_quad < 0.05 else 'Not significant'}")
print()

# Test 3: Inverse relationship
corr_inv, p_inv = pearsonr(stress_vals, -log_t_half)
print(f"Test 3: -log(t_1/2) vs σ (inverse)")
print(f"  Pearson r = {corr_inv:.4f}")
print(f"  p-value = {p_inv:.2e}")
print(f"  Interpretation: {'Significant' if p_inv < 0.05 else 'Not significant'}")
print()

# Test 4: Spearman (non-parametric)
corr_spear, p_spear = spearmanr(stress_vals, log_t_half)
print(f"Test 4: Spearman rank correlation")
print(f"  Spearman ρ = {corr_spear:.4f}")
print(f"  p-value = {p_spear:.2e}")
print(f"  Interpretation: {'Significant' if p_spear < 0.05 else 'Not significant'}")
print()

# Fit models
print("="*80)
print("MODEL FITTING")
print("="*80)
print()

# Model 1: log(t) = a - b*σ
def model_linear(sigma, a, b):
    return a - b * sigma

try:
    popt_lin, pcov_lin = curve_fit(model_linear, stress_vals, log_t_half)
    residuals_lin = log_t_half - model_linear(stress_vals, *popt_lin)
    rmse_lin = np.sqrt(np.mean(residuals_lin**2))

    print(f"Model 1: log(t_1/2) = a - b×σ")
    print(f"  a = {popt_lin[0]:.3f} ± {np.sqrt(pcov_lin[0,0]):.3f}")
    print(f"  b = {popt_lin[1]:.3f} ± {np.sqrt(pcov_lin[1,1]):.3f}")
    print(f"  RMSE = {rmse_lin:.3f} log10(seconds)")
    print()
except:
    popt_lin = None
    print("Model 1: Failed to fit")
    print()

# Model 2: log(t) = a - b*σ²
def model_quad(sigma, a, b):
    return a - b * sigma**2

try:
    popt_quad, pcov_quad = curve_fit(model_quad, stress_vals, log_t_half)
    residuals_quad = log_t_half - model_quad(stress_vals, *popt_quad)
    rmse_quad = np.sqrt(np.mean(residuals_quad**2))

    print(f"Model 2: log(t_1/2) = a - b×σ²")
    print(f"  a = {popt_quad[0]:.3f} ± {np.sqrt(pcov_quad[0,0]):.3f}")
    print(f"  b = {popt_quad[1]:.3f} ± {np.sqrt(pcov_quad[1,1]):.3f}")
    print(f"  RMSE = {rmse_quad:.3f} log10(seconds)")
    print()
except:
    popt_quad = None
    print("Model 2: Failed to fit")
    print()

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Panel A: Half-life vs Stress (linear)
ax1 = axes[0, 0]
scatter = ax1.scatter(stress_vals, log_t_half, c=A_vals, s=80,
                     cmap='viridis', edgecolors='black', linewidth=1, alpha=0.7)
if popt_lin is not None:
    sigma_fit = np.linspace(0, np.max(stress_vals)*1.1, 100)
    ax1.plot(sigma_fit, model_linear(sigma_fit, *popt_lin), 'r--',
            linewidth=2, label=f'Linear fit: r={corr_linear:.3f}')

# Annotate outliers
for i, (name, stress, logt) in enumerate(zip(names, stress_vals, log_t_half)):
    if stress > 2.5 or abs(logt) > 15:
        ax1.annotate(name, (stress, logt), fontsize=7, alpha=0.7)

ax1.set_xlabel('Geometric Stress σ = |N|', fontsize=12, fontweight='bold')
ax1.set_ylabel('log₁₀(Half-life [seconds])', fontsize=12, fontweight='bold')
ax1.set_title('(A) Half-Life vs Geometric Stress (Linear)', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(alpha=0.3)
cbar1 = plt.colorbar(scatter, ax=ax1)
cbar1.set_label('Mass Number A', fontsize=10)

# Panel B: Half-life vs Stress² (quadratic)
ax2 = axes[0, 1]
scatter2 = ax2.scatter(stress_vals**2, log_t_half, c=Z_vals, s=80,
                      cmap='plasma', edgecolors='black', linewidth=1, alpha=0.7)
if popt_quad is not None:
    sigma_sq_fit = np.linspace(0, np.max(stress_vals**2)*1.1, 100)
    ax2.plot(sigma_sq_fit, model_quad(np.sqrt(sigma_sq_fit), *popt_quad), 'r--',
            linewidth=2, label=f'Quadratic fit: r={corr_quad:.3f}')

ax2.set_xlabel('Geometric Stress² σ²', fontsize=12, fontweight='bold')
ax2.set_ylabel('log₁₀(Half-life [seconds])', fontsize=12, fontweight='bold')
ax2.set_title('(B) Half-Life vs Geometric Stress² (Quadratic)', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(alpha=0.3)
cbar2 = plt.colorbar(scatter2, ax=ax2)
cbar2.set_label('Charge Z', fontsize=10)

# Panel C: Residuals (Linear model)
ax3 = axes[1, 0]
if popt_lin is not None:
    ax3.scatter(stress_vals, residuals_lin, c=log_t_half, s=60,
               cmap='coolwarm', edgecolors='black', linewidth=1, alpha=0.7)
    ax3.axhline(0, color='red', linestyle='--', linewidth=2)
    ax3.axhline(rmse_lin, color='orange', linestyle=':', linewidth=1, label=f'±RMSE = ±{rmse_lin:.2f}')
    ax3.axhline(-rmse_lin, color='orange', linestyle=':', linewidth=1)

ax3.set_xlabel('Geometric Stress σ', fontsize=12, fontweight='bold')
ax3.set_ylabel('Residual log₁₀(t_1/2)', fontsize=12, fontweight='bold')
ax3.set_title('(C) Residuals from Linear Model', fontsize=13, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(alpha=0.3)

# Panel D: Half-life vs N coordinate (signed)
ax4 = axes[1, 1]
colors_decay = ['blue' if 'α' in d['decay'] else 'green' if 'β+' in d['decay'] or 'EC' in d['decay'] else 'red'
                for d in data]
for i, (N, logt, color, name) in enumerate(zip(N_vals, log_t_half, colors_decay, names)):
    ax4.scatter(N, logt, c=color, s=60, edgecolors='black', linewidth=1, alpha=0.7)

ax4.axvline(0, color='gold', linestyle='-', linewidth=3, alpha=0.5, label='Ground State (N=0)')
ax4.set_xlabel('Geometric Coordinate N (signed)', fontsize=12, fontweight='bold')
ax4.set_ylabel('log₁₀(Half-life [seconds])', fontsize=12, fontweight='bold')
ax4.set_title('(D) Half-Life vs Signed Coordinate N', fontsize=13, fontweight='bold')
ax4.grid(alpha=0.3)

# Legend for decay modes
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='red', label='β⁻ decay'),
                  Patch(facecolor='green', label='β⁺/EC decay'),
                  Patch(facecolor='blue', label='α decay')]
ax4.legend(handles=legend_elements, loc='upper left', fontsize=9)

plt.suptitle('HALF-LIFE vs GEOMETRIC STRESS CORRELATION TEST\n' +
             'Topological Quantization: Z and A are Integers (Winding Numbers)',
             fontsize=15, fontweight='bold', y=0.998)

plt.tight_layout()
plt.savefig('halflife_stress_correlation.png', dpi=200, bbox_inches='tight')
plt.savefig('halflife_stress_correlation.pdf', bbox_inches='tight')

print("="*80)
print("INTERPRETATION")
print("="*80)
print()

if abs(corr_linear) > 0.3 and p_linear < 0.05:
    print("✓ SIGNIFICANT LINEAR CORRELATION FOUND")
    print(f"  As stress increases, half-life {'decreases' if corr_linear < 0 else 'increases'}")
    print(f"  Correlation strength: r = {corr_linear:.3f}")
elif abs(corr_quad) > 0.3 and p_quad < 0.05:
    print("✓ SIGNIFICANT QUADRATIC CORRELATION FOUND")
    print(f"  Quadratic relationship: r = {corr_quad:.3f}")
else:
    print("⚠ WEAK OR NO SIGNIFICANT CORRELATION")
    print(f"  Linear: r = {corr_linear:.3f} (p = {p_linear:.3f})")
    print(f"  Quadratic: r = {corr_quad:.3f} (p = {p_quad:.3f})")
    print()
    print("Possible reasons:")
    print("  1. Decay mode matters (α vs β different physics)")
    print("  2. Other factors dominate (Q-value, barrier penetration)")
    print("  3. Stress is necessary but not sufficient for half-life")

print()
print("="*80)
print("TOPOLOGICAL QUANTIZATION NOTE")
print("="*80)
print()
print("All isotopes have INTEGER Z and A (topological winding numbers):")
print(f"  Z ∈ {{1, 2, 3, ..., 94}} - Electric charge quantization")
print(f"  A ∈ {{1, 2, 3, ..., 241}} - Baryon number quantization")
print()
print("The continuous field N(A,Z) is calculated from these discrete values.")
print("Residuals in atomic mass (mass defect) are ~0.1% - negligible for N calculation.")
print()
print("="*80)

print("\nFigures saved:")
print("  - halflife_stress_correlation.png (200 DPI)")
print("  - halflife_stress_correlation.pdf (vector)")
print()
