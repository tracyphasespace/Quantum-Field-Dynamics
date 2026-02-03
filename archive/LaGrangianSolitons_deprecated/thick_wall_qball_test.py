#!/usr/bin/env python3
"""
THICK-WALL Q-BALL TEST - LINEAR CURVATURE SCALING
===========================================================================
User's Derivation:

For a thick-walled soliton with dielectric profile φ(r) = φ₀(R/r):

E_grad = ∫ (∇φ)²/2 dV
       = ∫ (φ₀²/r²) × 4πr² dr  (from R_core to R_total)
       = 4πφ₀² × R
       ~ A^(1/3)   ← LINEAR SCALING

Compare:
- Thin wall: E_surf ~ A^(2/3) (surface area, harsh penalty for heavy nuclei)
- Thick wall: E_curv ~ A^(1/3) (gradient volume, lower penalty for heavy nuclei)

Hypothesis:
Replacing A^(2/3) with A^(1/3) will fix heavy nuclei where we over-penalized
surface formation, changing the balance against E_vac ~ Z²/A^(1/3).

Test:
1. Baseline: E_surf ~ A^(2/3) (thin wall) → 176/285 (61.8%)
2. Thick wall: E_curv ~ A^(1/3) (gradient energy) → ?/285
3. Analyze which nuclei are fixed (expect heavy nuclei, A > 100)
===========================================================================
"""

import numpy as np
from collections import Counter, defaultdict

# QFD Constants
alpha_fine = 1.0 / 137.036
beta_vacuum = 1.0 / 3.043233053
M_proton = 938.272
KAPPA_E = 0.0001
SHIELD_FACTOR = 0.52
DELTA_PAIRING = 11.0

# Derived Constants
V_0 = M_proton * (1 - (alpha_fine**2) / beta_vacuum)
beta_nuclear = M_proton * beta_vacuum / 2

# THIN WALL (baseline)
E_surface_coeff = beta_nuclear / 15  # From C(6,2) = 15 projection

# THICK WALL (to be optimized)
# E_curvature_coeff will be fitted to match energy scale
# Physical expectation: Should be larger than E_surface_coeff to compensate for lower power

a_sym_base = (beta_vacuum * M_proton) / 15
a_disp = (alpha_fine * 197.327 / 1.2) * SHIELD_FACTOR

def qfd_energy_thin_wall(A, Z):
    """Baseline: Thin-wall (A^(2/3)) surface energy."""
    N = A - Z
    if Z <= 0 or N < 0:
        return 1e12

    q = Z / A
    lambda_time = KAPPA_E * Z

    E_volume_coeff = V_0 * (1 - lambda_time / (12 * np.pi))
    E_bulk = E_volume_coeff * A
    E_surf = E_surface_coeff * (A**(2/3))  # THIN WALL
    E_asym = a_sym_base * A * ((1 - 2*q)**2)
    E_vac = a_disp * (Z**2) / (A**(1/3))

    # Pairing energy
    E_pair = 0
    if Z % 2 == 0 and N % 2 == 0:
        E_pair = -DELTA_PAIRING / np.sqrt(A)
    elif Z % 2 == 1 and N % 2 == 1:
        E_pair = +DELTA_PAIRING / np.sqrt(A)

    return E_bulk + E_surf + E_asym + E_vac + E_pair

def qfd_energy_thick_wall(A, Z, curvature_coeff):
    """Thick-wall: Linear (A^(1/3)) gradient/curvature energy."""
    N = A - Z
    if Z <= 0 or N < 0:
        return 1e12

    q = Z / A
    lambda_time = KAPPA_E * Z

    E_volume_coeff = V_0 * (1 - lambda_time / (12 * np.pi))
    E_bulk = E_volume_coeff * A
    E_curv = curvature_coeff * (A**(1/3))  # THICK WALL (linear scaling)
    E_asym = a_sym_base * A * ((1 - 2*q)**2)
    E_vac = a_disp * (Z**2) / (A**(1/3))

    # Pairing energy
    E_pair = 0
    if Z % 2 == 0 and N % 2 == 0:
        E_pair = -DELTA_PAIRING / np.sqrt(A)
    elif Z % 2 == 1 and N % 2 == 1:
        E_pair = +DELTA_PAIRING / np.sqrt(A)

    return E_bulk + E_curv + E_asym + E_vac + E_pair

def find_stable_Z(A, model='thin', curvature_coeff=None):
    """Find Z that minimizes energy."""
    best_Z, best_E = 1, 1e12

    for Z in range(1, A):
        if model == 'thin':
            E = qfd_energy_thin_wall(A, Z)
        else:  # thick
            E = qfd_energy_thick_wall(A, Z, curvature_coeff)

        if E < best_E:
            best_E, best_Z = E, Z

    return best_Z

# Load data
with open('qfd_optimized_suite.py', 'r') as f:
    content = f.read()
start = content.find('test_nuclides = [')
end = content.find(']', start) + 1
test_nuclides = eval(content[start:end].replace('test_nuclides = ', ''))

print("="*95)
print("THICK-WALL Q-BALL TEST - LINEAR CURVATURE SCALING")
print("="*95)
print()

# ============================================================================
# BASELINE: THIN WALL
# ============================================================================
print("="*95)
print("BASELINE: THIN WALL (A^(2/3) SURFACE ENERGY)")
print("="*95)
print()

correct_thin = 0
errors_thin = []

for name, Z_exp, A in test_nuclides:
    Z_pred = find_stable_Z(A, model='thin')
    if Z_pred == Z_exp:
        correct_thin += 1
    else:
        errors_thin.append({
            'name': name,
            'A': A,
            'Z_exp': Z_exp,
            'Z_pred': Z_pred,
            'error': Z_pred - Z_exp,
        })

print(f"Thin wall (baseline): {correct_thin}/285 ({100*correct_thin/285:.1f}%)")
print()

# Error by mass region
print(f"Errors by mass region (thin wall):")
light_errors = [e for e in errors_thin if e['A'] < 60]
medium_errors = [e for e in errors_thin if 60 <= e['A'] < 140]
heavy_errors = [e for e in errors_thin if e['A'] >= 140]

print(f"  Light (A < 60):    {len(light_errors)} errors")
print(f"  Medium (60≤A<140): {len(medium_errors)} errors")
print(f"  Heavy (A ≥ 140):   {len(heavy_errors)} errors")
print()

# ============================================================================
# OPTIMIZE CURVATURE COEFFICIENT
# ============================================================================
print("="*95)
print("OPTIMIZE THICK-WALL CURVATURE COEFFICIENT")
print("="*95)
print()

print("Testing different curvature coefficients (E_curv = coeff × A^(1/3))...")
print()

# Physical expectation: E_curv_coeff should be larger than E_surface_coeff (10.23 MeV)
# because A^(1/3) is smaller than A^(2/3), so we need a larger coefficient to get similar energy magnitude

test_coeffs = np.linspace(10, 100, 19)  # Test range from 10 to 100 MeV

print(f"{'E_curv_coeff':<15} {'Correct':<12} {'Success %':<12} {'vs Thin':<15} {'Marker'}\"")
print("-"*95)

best_coeff = E_surface_coeff
best_correct = correct_thin

for coeff in test_coeffs:
    correct = 0

    for name, Z_exp, A in test_nuclides:
        Z_pred = find_stable_Z(A, model='thick', curvature_coeff=coeff)
        if Z_pred == Z_exp:
            correct += 1

    delta = correct - correct_thin
    pct = 100 * correct / 285

    marker = ""
    if correct > best_correct:
        marker = "★★★"
        best_correct = correct
        best_coeff = coeff
    elif correct > correct_thin:
        marker = "★"

    print(f"{coeff:<15.1f} {correct:<12} {pct:<12.1f} {delta:+d}  {marker:<10}")

print()
print(f"Optimal curvature coefficient: {best_coeff:.1f} MeV")
print(f"Thin wall baseline: {correct_thin}/285 ({100*correct_thin/285:.1f}%)")
print(f"Thick wall optimized: {best_correct}/285 ({100*best_correct/285:.1f}%)")
print(f"Improvement: {best_correct - correct_thin:+d} matches")
print()

# ============================================================================
# ANALYZE WHICH NUCLEI ARE FIXED
# ============================================================================
print("="*95)
print(f"NUCLEI FIXED BY THICK WALL (E_curv = {best_coeff:.1f} × A^(1/3))")
print("="*95)
print()

fixed_by_thick = []
broken_by_thick = []

for name, Z_exp, A in test_nuclides:
    Z_pred_thin = find_stable_Z(A, model='thin')
    Z_pred_thick = find_stable_Z(A, model='thick', curvature_coeff=best_coeff)

    if Z_pred_thin != Z_exp and Z_pred_thick == Z_exp:
        fixed_by_thick.append({
            'name': name,
            'A': A,
            'Z': Z_exp,
            'N': A - Z_exp,
            'mod_4': A % 4,
        })
    elif Z_pred_thin == Z_exp and Z_pred_thick != Z_exp:
        broken_by_thick.append({
            'name': name,
            'A': A,
            'Z': Z_exp,
            'Z_pred_thick': Z_pred_thick,
        })

print(f"Nuclei fixed: {len(fixed_by_thick)}")
print(f"Nuclei broken: {len(broken_by_thick)}")
print()

if fixed_by_thick:
    print(f"{'Nuclide':<12} {'A':<6} {'Z':<6} {'N':<6} {'A mod 4':<10} {'Mass Region'}\"")
    print("-"*95)

    for n in sorted(fixed_by_thick, key=lambda x: x['A']):
        mass_region = "Light" if n['A'] < 60 else "Medium" if n['A'] < 140 else "Heavy"
        marker = "★★★" if n['A'] >= 140 else ""

        print(f"{n['name']:<12} {n['A']:<6} {n['Z']:<6} {n['N']:<6} {n['mod_4']:<10} {mass_region:<15} {marker}")

    print()

    # Statistics
    light_fixed = sum(1 for n in fixed_by_thick if n['A'] < 60)
    medium_fixed = sum(1 for n in fixed_by_thick if 60 <= n['A'] < 140)
    heavy_fixed = sum(1 for n in fixed_by_thick if n['A'] >= 140)

    print(f"Fixed by mass region:")
    print(f"  Light (A < 60):    {light_fixed}/{len(fixed_by_thick)} ({100*light_fixed/len(fixed_by_thick):.1f}%)")
    print(f"  Medium (60≤A<140): {medium_fixed}/{len(fixed_by_thick)} ({100*medium_fixed/len(fixed_by_thick):.1f}%)")
    print(f"  Heavy (A ≥ 140):   {heavy_fixed}/{len(fixed_by_thick)} ({100*heavy_fixed/len(fixed_by_thick):.1f}%)")
    print()

if broken_by_thick:
    print(f"Nuclei broken by thick wall:")
    for n in sorted(broken_by_thick, key=lambda x: x['A'])[:20]:
        print(f"  {n['name']}: Z_exp={n['Z']}, Z_pred_thick={n['Z_pred_thick']}")
    if len(broken_by_thick) > 20:
        print(f"  ... and {len(broken_by_thick) - 20} more")
    print()

# ============================================================================
# COMPARE ENERGY SCALES
# ============================================================================
print("="*95)
print("ENERGY SCALE COMPARISON (THIN VS THICK)")
print("="*95)
print()

print(f"For representative nuclei, compare surface/curvature energy magnitudes:")
print()

print(f"{'Nuclide':<12} {'A':<6} {'E_surf (thin)':<18} {'E_curv (thick)':<18} {'Ratio (thick/thin)'}\"")
print("-"*95)

test_nuclei = [
    ("Ca-40", 20, 40),
    ("Fe-56", 26, 56),
    ("Zr-90", 40, 90),
    ("Sn-120", 50, 120),
    ("Pb-208", 82, 208),
]

for name, Z, A in test_nuclei:
    E_surf_thin = E_surface_coeff * (A**(2/3))
    E_curv_thick = best_coeff * (A**(1/3))
    ratio = E_curv_thick / E_surf_thin

    marker = "★" if A >= 140 else ""

    print(f"{name:<12} {A:<6} {E_surf_thin:<18.2f} {E_curv_thick:<18.2f} {ratio:<18.3f} {marker}")

print()

print(f"Observations:")
print(f"  • Thin wall (A^(2/3)): Energy increases rapidly with A")
print(f"  • Thick wall (A^(1/3)): Energy increases slowly with A")
print(f"  • For heavy nuclei (A~200), thick wall has LOWER surface penalty")
print(f"  • This reduces over-penalization of large nuclei")
print()

# ============================================================================
# HEAVY NUCLEI FOCUS
# ============================================================================
print("="*95)
print("HEAVY NUCLEI PERFORMANCE (A ≥ 140)")
print("="*95)
print()

heavy_nuclei = [(name, Z, A) for name, Z, A in test_nuclides if A >= 140]

print(f"Heavy nuclei count: {len(heavy_nuclei)}")
print()

# Thin wall performance
correct_thin_heavy = 0
for name, Z_exp, A in heavy_nuclei:
    Z_pred = find_stable_Z(A, model='thin')
    if Z_pred == Z_exp:
        correct_thin_heavy += 1

# Thick wall performance
correct_thick_heavy = 0
for name, Z_exp, A in heavy_nuclei:
    Z_pred = find_stable_Z(A, model='thick', curvature_coeff=best_coeff)
    if Z_pred == Z_exp:
        correct_thick_heavy += 1

print(f"Heavy nuclei (A ≥ 140) performance:")
print(f"  Thin wall:  {correct_thin_heavy}/{len(heavy_nuclei)} ({100*correct_thin_heavy/len(heavy_nuclei):.1f}%)")
print(f"  Thick wall: {correct_thick_heavy}/{len(heavy_nuclei)} ({100*correct_thick_heavy/len(heavy_nuclei):.1f}%)")
print(f"  Improvement: {correct_thick_heavy - correct_thin_heavy:+d} matches")
print()

# ============================================================================
# SUMMARY
# ============================================================================
print("="*95)
print("SUMMARY: THICK-WALL Q-BALL HYPOTHESIS TEST")
print("="*95)
print()

print(f"HYPOTHESIS: Thick-walled dielectric gradient (E ~ A^(1/3)) reduces")
print(f"surface penalty for heavy nuclei, fixing the 'over-penalization' error.")
print()

print(f"RESULTS:")
print(f"  Thin wall (baseline):      {correct_thin}/285 ({100*correct_thin/285:.1f}%)")
print(f"  Thick wall (optimized):    {best_correct}/285 ({100*best_correct/285:.1f}%)")
print(f"  Improvement:               {best_correct - correct_thin:+d} matches ({100*(best_correct - correct_thin)/285:+.1f}%)")
print()

if best_correct > correct_thin:
    print(f"★★★ THICK WALL IMPROVES PREDICTIONS!")
    print()
    print(f"Optimal curvature coefficient: E_curv = {best_coeff:.1f} MeV × A^(1/3)")
    print()
    print(f"Physical interpretation:")
    print(f"  • Gradient energy ∫ (∇φ)² dV with φ(r) ~ R/r")
    print(f"  • Linear scaling from volume integration: ∫ (1/r²) r² dr = ∫ dr ~ R")
    print(f"  • Reduces penalty for large nuclei (heavy elements)")
    print(f"  • Fixes balance against Coulomb E_vac ~ Z²/A^(1/3)")
    print()

    if heavy_fixed > 0:
        print(f"★ Heavy nuclei (A ≥ 140) improved by {correct_thick_heavy - correct_thin_heavy:+d} matches")
        print(f"  → Confirms hypothesis: thick wall corrects over-penalization")
    else:
        print(f"Note: Heavy nuclei not specifically improved")
        print(f"  → Improvement distributed across mass regions")

elif best_correct == correct_thin:
    print(f"✗ Thick wall shows NO improvement over thin wall")
    print(f"  → A^(1/3) scaling does not change predictions")
    print(f"  → The geometric limit may be fundamental, not from scaling law")

else:
    print(f"✗ Thick wall WORSENS predictions!")
    print(f"  → A^(2/3) (thin wall) is physically correct")
    print(f"  → Surface tension dominates over gradient energy")

print()

# Target assessment
target_185 = int(0.65 * 285)
print(f"Progress toward 65% geometric limit (185/285):")
print(f"  Thin wall:  {correct_thin}/185 ({100*correct_thin/target_185:.1f}%)")
print(f"  Thick wall: {best_correct}/185 ({100*best_correct/target_185:.1f}%)")
print(f"  Remaining:  {target_185 - best_correct} matches to reach 65%")
print()

print("="*95)
print("PHYSICAL SCALING LAW:")
print("="*95)
print()
print("Thin wall:  E_surf ~ σ × A^(2/3)   (surface tension × area)")
print("Thick wall: E_curv ~ κ × A^(1/3)   (gradient stiffness × radius)")
print()
print("The thick-wall model treats the proton atmosphere as a VOLUME of")
print("refractive vacuum with profile φ(r) = φ₀(R/r), not a surface.")
print()
print("This is Q-ball physics (topological soliton with long-range tail),")
print("not liquid-drop model (thin shell with surface tension).")
print()
print("="*95)
