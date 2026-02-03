#!/usr/bin/env python3
"""
COMPREHENSIVE NUCLIDE SPACE EXPLORATION
===========================================================================
Map the success and failure regimes of QFD soliton stability predictions.

GOALS:
1. Test on 100+ experimental nuclides (light to heavy)
2. Explore shielding factor space (0.5 to 1.0)
3. Identify where model succeeds vs fails
4. Map failure modes by (A, Z, Z/A)
5. Understand systematic trends

===========================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

# ============================================================================
# FUNDAMENTAL CONSTANTS
# ============================================================================

alpha_fine   = 1.0 / 137.036
beta_vacuum  = 1.0 / 3.043233053
lambda_time  = 0.42
M_proton     = 938.272

V_0 = M_proton * (1 - (alpha_fine**2) / beta_vacuum)
beta_nuclear = M_proton * beta_vacuum / 2

E_volume  = V_0 * (1 - lambda_time / (12 * np.pi))
E_surface = beta_nuclear / 15
a_sym = (beta_vacuum * M_proton) / 15

hbar_c = 197.327
r_0 = 1.2
a_c_base = alpha_fine * hbar_c / r_0

# ============================================================================
# EXPANDED NUCLIDE DATABASE
# ============================================================================

# Comprehensive test set spanning the nuclear chart
nuclides = [
    # Hydrogen
    ("H-1",   1, 1), ("H-2",   1, 2), ("H-3",   1, 3),
    # Helium
    ("He-3",  2, 3), ("He-4",  2, 4),
    # Lithium
    ("Li-6",  3, 6), ("Li-7",  3, 7),
    # Beryllium
    ("Be-9",  4, 9),
    # Boron
    ("B-10",  5, 10), ("B-11",  5, 11),
    # Carbon
    ("C-12",  6, 12), ("C-13",  6, 13), ("C-14",  6, 14),
    # Nitrogen
    ("N-14",  7, 14), ("N-15",  7, 15),
    # Oxygen
    ("O-16",  8, 16), ("O-17",  8, 17), ("O-18",  8, 18),
    # Fluorine
    ("F-19",  9, 19),
    # Neon
    ("Ne-20", 10, 20), ("Ne-21", 10, 21), ("Ne-22", 10, 22),
    # Sodium
    ("Na-23", 11, 23),
    # Magnesium
    ("Mg-24", 12, 24), ("Mg-25", 12, 25), ("Mg-26", 12, 26),
    # Aluminum
    ("Al-27", 13, 27),
    # Silicon
    ("Si-28", 14, 28), ("Si-29", 14, 29), ("Si-30", 14, 30),
    # Phosphorus
    ("P-31",  15, 31),
    # Sulfur
    ("S-32",  16, 32), ("S-33",  16, 33), ("S-34",  16, 34), ("S-36",  16, 36),
    # Chlorine
    ("Cl-35", 17, 35), ("Cl-37", 17, 37),
    # Argon
    ("Ar-36", 18, 36), ("Ar-38", 18, 38), ("Ar-40", 18, 40),
    # Potassium
    ("K-39",  19, 39), ("K-40",  19, 40), ("K-41",  19, 41),
    # Calcium
    ("Ca-40", 20, 40), ("Ca-42", 20, 42), ("Ca-43", 20, 43), ("Ca-44", 20, 44), ("Ca-48", 20, 48),
    # Scandium
    ("Sc-45", 21, 45),
    # Titanium
    ("Ti-46", 22, 46), ("Ti-47", 22, 47), ("Ti-48", 22, 48), ("Ti-49", 22, 49), ("Ti-50", 22, 50),
    # Vanadium
    ("V-50",  23, 50), ("V-51",  23, 51),
    # Chromium
    ("Cr-50", 24, 50), ("Cr-52", 24, 52), ("Cr-53", 24, 53), ("Cr-54", 24, 54),
    # Manganese
    ("Mn-55", 25, 55),
    # Iron
    ("Fe-54", 26, 54), ("Fe-56", 26, 56), ("Fe-57", 26, 57), ("Fe-58", 26, 58),
    # Cobalt
    ("Co-59", 27, 59),
    # Nickel
    ("Ni-58", 28, 58), ("Ni-60", 28, 60), ("Ni-61", 28, 61), ("Ni-62", 28, 62), ("Ni-64", 28, 64),
    # Copper
    ("Cu-63", 29, 63), ("Cu-65", 29, 65),
    # Zinc
    ("Zn-64", 30, 64), ("Zn-66", 30, 66), ("Zn-67", 30, 67), ("Zn-68", 30, 68), ("Zn-70", 30, 70),
    # Gallium
    ("Ga-69", 31, 69), ("Ga-71", 31, 71),
    # Germanium
    ("Ge-70", 32, 70), ("Ge-72", 32, 72), ("Ge-73", 32, 73), ("Ge-74", 32, 74), ("Ge-76", 32, 76),
    # Arsenic
    ("As-75", 33, 75),
    # Selenium
    ("Se-74", 34, 74), ("Se-76", 34, 76), ("Se-77", 34, 77), ("Se-78", 34, 78), ("Se-80", 34, 80), ("Se-82", 34, 82),
    # Bromine
    ("Br-79", 35, 79), ("Br-81", 35, 81),
    # Krypton
    ("Kr-78", 36, 78), ("Kr-80", 36, 80), ("Kr-82", 36, 82), ("Kr-83", 36, 83), ("Kr-84", 36, 84), ("Kr-86", 36, 86),
    # Rubidium
    ("Rb-85", 37, 85), ("Rb-87", 37, 87),
    # Strontium
    ("Sr-84", 38, 84), ("Sr-86", 38, 86), ("Sr-87", 38, 87), ("Sr-88", 38, 88),
    # Yttrium
    ("Y-89",  39, 89),
    # Zirconium
    ("Zr-90", 40, 90), ("Zr-91", 40, 91), ("Zr-92", 40, 92), ("Zr-94", 40, 94), ("Zr-96", 40, 96),
    # Molybdenum
    ("Mo-92", 42, 92), ("Mo-94", 42, 94), ("Mo-95", 42, 95), ("Mo-96", 42, 96), ("Mo-97", 42, 97), ("Mo-98", 42, 98), ("Mo-100", 42, 100),
    # Silver
    ("Ag-107", 47, 107), ("Ag-109", 47, 109),
    # Cadmium
    ("Cd-106", 48, 106), ("Cd-108", 48, 108), ("Cd-110", 48, 110), ("Cd-111", 48, 111), ("Cd-112", 48, 112), ("Cd-113", 48, 113), ("Cd-114", 48, 114), ("Cd-116", 48, 116),
    # Tin
    ("Sn-112", 50, 112), ("Sn-114", 50, 114), ("Sn-115", 50, 115), ("Sn-116", 50, 116), ("Sn-117", 50, 117), ("Sn-118", 50, 118), ("Sn-119", 50, 119), ("Sn-120", 50, 120), ("Sn-122", 50, 122), ("Sn-124", 50, 124),
    # Xenon
    ("Xe-124", 54, 124), ("Xe-126", 54, 126), ("Xe-128", 54, 128), ("Xe-129", 54, 129), ("Xe-130", 54, 130), ("Xe-131", 54, 131), ("Xe-132", 54, 132), ("Xe-134", 54, 134), ("Xe-136", 54, 136),
    # Barium
    ("Ba-130", 56, 130), ("Ba-132", 56, 132), ("Ba-134", 56, 134), ("Ba-135", 56, 135), ("Ba-136", 56, 136), ("Ba-137", 56, 137), ("Ba-138", 56, 138),
    # Lead
    ("Pb-204", 82, 204), ("Pb-206", 82, 206), ("Pb-207", 82, 207), ("Pb-208", 82, 208),
    # Bismuth
    ("Bi-209", 83, 209),
    # Thorium
    ("Th-232", 90, 232),
    # Uranium
    ("U-235", 92, 235), ("U-238", 92, 238),
]

print(f"Testing on {len(nuclides)} nuclides (Z=1 to Z=92)")
print()

# ============================================================================
# SHIELDING FACTOR EXPLORATION
# ============================================================================

def test_shielding_factor(shield_factor, nuclides_list):
    """Test a specific shielding factor on all nuclides."""
    a_c = a_c_base * shield_factor

    def total_energy(A, Z):
        q = Z / A if A > 0 else 0
        E_bulk = E_volume * A
        E_surf = E_surface * (A ** (2/3))
        E_asym = a_sym * A * ((1 - 2*q)**2) if A > 0 else 0
        E_disp = a_c * (Z**2) / (A ** (1/3)) if A > 0 else 0
        return E_bulk + E_surf + E_asym + E_disp

    def find_stable_Z(A):
        if A == 1:
            return 1  # Only possibility
        if A == 2:
            return 1  # H-2 is most stable A=2 configuration
        result = minimize_scalar(
            lambda Z: total_energy(A, Z),
            bounds=(1, A-1),
            method='bounded'
        )
        return int(np.round(result.x))

    results = []
    for name, Z_exp, A in nuclides_list:
        Z_pred = find_stable_Z(A)
        Delta_Z = Z_pred - Z_exp
        q_exp = Z_exp / A
        q_pred = Z_pred / A
        results.append({
            'name': name,
            'A': A,
            'Z_exp': Z_exp,
            'Z_pred': Z_pred,
            'Delta_Z': Delta_Z,
            'q_exp': q_exp,
            'q_pred': q_pred,
        })

    return results

# ============================================================================
# EXPLORE MULTIPLE SHIELDING FACTORS
# ============================================================================

print("="*85)
print("SHIELDING FACTOR EXPLORATION")
print("="*85)
print()

shield_factors = [0.50, 0.55, 0.60, 0.65, 0.70, 5/7, 0.75, 0.80, 0.90, 1.00]

print(f"{'Shield':<8} {'a_c (MeV)':<12} {'Mean|ΔZ|':<10} {'Med|ΔZ|':<10} {'Max|ΔZ|':<10} {'Exact%':<8}")
print("-"*85)

best_results = None
best_factor = None
best_mean_error = float('inf')

for factor in shield_factors:
    results = test_shielding_factor(factor, nuclides)
    errors = [abs(r['Delta_Z']) for r in results]
    mean_err = np.mean(errors)
    med_err = np.median(errors)
    max_err = np.max(errors)
    exact_pct = 100 * sum(e == 0 for e in errors) / len(errors)
    a_c = a_c_base * factor

    if mean_err < best_mean_error:
        best_mean_error = mean_err
        best_factor = factor
        best_results = results

    print(f"{factor:<8.3f} {a_c:<12.3f} {mean_err:<10.3f} {med_err:<10.1f} {max_err:<10.0f} {exact_pct:<8.1f}%")

print("="*85)
print(f"\nOptimal shielding factor: {best_factor:.3f}")
print(f"  a_c = {a_c_base * best_factor:.3f} MeV")
print(f"  Mean |ΔZ| = {best_mean_error:.3f} charges")
print()

# ============================================================================
# DETAILED ANALYSIS OF OPTIMAL FACTOR
# ============================================================================

print("="*85)
print(f"DETAILED RESULTS (Shield factor = {best_factor:.3f})")
print("="*85)
print()

# Categorize by mass region
light = [r for r in best_results if r['A'] < 40]
medium = [r for r in best_results if 40 <= r['A'] < 100]
heavy = [r for r in best_results if 100 <= r['A'] < 200]
superheavy = [r for r in best_results if r['A'] >= 200]

print("PERFORMANCE BY MASS REGION:")
print("-"*85)
for name, group in [("Light (A<40)", light), ("Medium (40≤A<100)", medium),
                     ("Heavy (100≤A<200)", heavy), ("Superheavy (A≥200)", superheavy)]:
    if len(group) > 0:
        errors = [abs(r['Delta_Z']) for r in group]
        exact = sum(e == 0 for e in errors)
        print(f"{name:<20} N={len(group):<4} Mean|ΔZ|={np.mean(errors):.2f}  "
              f"Median={np.median(errors):.0f}  Max={np.max(errors):.0f}  Exact={exact}/{len(group)} ({100*exact/len(group):.0f}%)")

print()
print("PERFORMANCE BY CHARGE REGION:")
print("-"*85)
for z_range, z_name in [(range(1,11), "Z=1-10"), (range(11,21), "Z=11-20"),
                         (range(21,31), "Z=21-30"), (range(31,51), "Z=31-50"),
                         (range(51,100), "Z>50")]:
    group = [r for r in best_results if r['Z_exp'] in z_range]
    if len(group) > 0:
        errors = [abs(r['Delta_Z']) for r in group]
        exact = sum(e == 0 for e in errors)
        print(f"{z_name:<20} N={len(group):<4} Mean|ΔZ|={np.mean(errors):.2f}  "
              f"Median={np.median(errors):.0f}  Max={np.max(errors):.0f}  Exact={exact}/{len(group)} ({100*exact/len(group):.0f}%)")

print()
print("PERFORMANCE BY CHARGE FRACTION:")
print("-"*85)
for q_min, q_max, q_name in [(0.45, 0.51, "q≈0.5 (symmetric)"), (0.40, 0.45, "0.40<q<0.45"),
                               (0.35, 0.40, "0.35<q<0.40"), (0.0, 0.35, "q<0.35 (very asymmetric)")]:
    group = [r for r in best_results if q_min <= r['q_exp'] < q_max]
    if len(group) > 0:
        errors = [abs(r['Delta_Z']) for r in group]
        exact = sum(e == 0 for e in errors)
        print(f"{q_name:<25} N={len(group):<4} Mean|ΔZ|={np.mean(errors):.2f}  "
              f"Median={np.median(errors):.0f}  Max={np.max(errors):.0f}  Exact={exact}/{len(group)} ({100*exact/len(group):.0f}%)")

# ============================================================================
# IDENTIFY WORST FAILURES
# ============================================================================

print()
print("="*85)
print("TOP 15 FAILURES (Largest |ΔZ|)")
print("="*85)
print(f"{'Nuclide':<10} {'A':>4} {'Z_exp':>6} {'Z_pred':>6} {'ΔZ':>6} {'q_exp':>7} {'q_pred':>7}")
print("-"*85)

sorted_by_error = sorted(best_results, key=lambda r: abs(r['Delta_Z']), reverse=True)
for r in sorted_by_error[:15]:
    print(f"{r['name']:<10} {r['A']:>4} {r['Z_exp']:>6} {r['Z_pred']:>6} {r['Delta_Z']:>+6} "
          f"{r['q_exp']:>7.4f} {r['q_pred']:>7.4f}")

# ============================================================================
# IDENTIFY BEST SUCCESSES
# ============================================================================

print()
print("="*85)
print("PERFECT PREDICTIONS (ΔZ = 0)")
print("="*85)

perfect = [r for r in best_results if r['Delta_Z'] == 0]
print(f"Total exact matches: {len(perfect)}/{len(best_results)} ({100*len(perfect)/len(best_results):.1f}%)")
print()
print(f"{'Nuclide':<10} {'A':>4} {'Z':>4} {'q':>7}")
print("-"*85)
for r in perfect[:20]:  # Show first 20
    print(f"{r['name']:<10} {r['A']:>4} {r['Z_exp']:>4} {r['q_exp']:>7.4f}")
if len(perfect) > 20:
    print(f"... and {len(perfect)-20} more")

# ============================================================================
# VISUALIZATION
# ============================================================================

print()
print("Generating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: ΔZ vs A
ax = axes[0, 0]
A_vals = [r['A'] for r in best_results]
DZ_vals = [r['Delta_Z'] for r in best_results]
ax.scatter(A_vals, DZ_vals, alpha=0.6, s=20)
ax.axhline(0, color='k', linestyle='--', alpha=0.3)
ax.set_xlabel('Baryon Number A', fontsize=11)
ax.set_ylabel('Charge Error ΔZ', fontsize=11)
ax.set_title(f'Prediction Error vs Mass (Shield={best_factor:.3f})', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# Plot 2: ΔZ vs Z
ax = axes[0, 1]
Z_vals = [r['Z_exp'] for r in best_results]
ax.scatter(Z_vals, DZ_vals, alpha=0.6, s=20, color='orange')
ax.axhline(0, color='k', linestyle='--', alpha=0.3)
ax.set_xlabel('Charge Z', fontsize=11)
ax.set_ylabel('Charge Error ΔZ', fontsize=11)
ax.set_title('Prediction Error vs Charge', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# Plot 3: ΔZ vs q
ax = axes[1, 0]
q_vals = [r['q_exp'] for r in best_results]
ax.scatter(q_vals, DZ_vals, alpha=0.6, s=20, color='green')
ax.axhline(0, color='k', linestyle='--', alpha=0.3)
ax.set_xlabel('Charge Fraction q = Z/A', fontsize=11)
ax.set_ylabel('Charge Error ΔZ', fontsize=11)
ax.set_title('Prediction Error vs Charge Fraction', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# Plot 4: Error distribution histogram
ax = axes[1, 1]
errors_abs = [abs(r['Delta_Z']) for r in best_results]
ax.hist(errors_abs, bins=range(0, int(max(errors_abs))+2), alpha=0.7, edgecolor='black')
ax.set_xlabel('|ΔZ| (charges)', fontsize=11)
ax.set_ylabel('Count', fontsize=11)
ax.set_title('Error Distribution', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('nuclide_space_exploration.png', dpi=150, bbox_inches='tight')
print("Saved: nuclide_space_exploration.png")

print()
print("="*85)
print("EXPLORATION COMPLETE")
print("="*85)
