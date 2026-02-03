#!/usr/bin/env python3
"""
QFD SURVIVOR SEARCH - FULL 285 NUCLIDE SWEEP
===========================================================================
Tests topological eccentricity optimization on complete dataset.

Compares:
  - Baseline (spherical, ecc=0)
  - Survivor (optimal eccentricity 0 ≤ ecc ≤ 0.25)

Generates deformation map showing which nuclei prefer non-spherical shapes.
===========================================================================
"""

import numpy as np

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
a_sym     = (beta_vacuum * M_proton) / 15

SHIELD_FACTOR = 0.52
a_disp = (alpha_fine * 197.327 / 1.2) * SHIELD_FACTOR
ISOMER_NODES = {2, 8, 20, 28, 50, 82, 126}
BONUS_STRENGTH = 0.70

def get_resonance_bonus(Z, N):
    bonus = 0
    if Z in ISOMER_NODES: bonus += E_surface * BONUS_STRENGTH
    if N in ISOMER_NODES: bonus += E_surface * BONUS_STRENGTH
    if Z in ISOMER_NODES and N in ISOMER_NODES:
        bonus *= 1.5
    return bonus

# ============================================================================
# ENERGY FUNCTIONAL
# ============================================================================
def qfd_survivor_energy(A, Z, ecc):
    """Energy with eccentricity-dependent geometry."""
    G_surf = 1.0 + (ecc**2)
    G_disp = 1.0 / (1.0 + ecc)

    N = A - Z
    q = Z / A if A > 0 else 0

    E_bulk = E_volume * A
    E_surf = E_surface * (A**(2/3)) * G_surf
    E_asym = a_sym * A * ((1 - 2*q)**2)
    E_vac  = a_disp * (Z**2) / (A**(1/3)) * G_disp
    E_iso  = -get_resonance_bonus(Z, N)

    return E_bulk + E_surf + E_asym + E_vac + E_iso

def find_survivor_state(A, ecc_max=0.25, n_ecc=6):
    """Find optimal (Z, ecc) for given A."""
    best_Z, best_ecc, min_E = 0, 0.0, float('inf')

    for z in range(1, A):
        for ecc in np.linspace(0, ecc_max, n_ecc):
            energy = qfd_survivor_energy(A, z, ecc)
            if energy < min_E:
                min_E, best_Z, best_ecc = energy, z, ecc

    return best_Z, best_ecc

def find_spherical_Z(A):
    """Baseline: spherical approximation (ecc=0)."""
    best_Z = 1
    best_E = qfd_survivor_energy(A, 1, 0.0)

    for z in range(1, A):
        E = qfd_survivor_energy(A, z, 0.0)
        if E < best_E:
            best_E = E
            best_Z = z

    return best_Z

# ============================================================================
# LOAD TEST DATA
# ============================================================================
with open('qfd_optimized_suite.py', 'r') as f:
    content = f.read()

start = content.find('test_nuclides = [')
end = content.find(']', start) + 1
test_nuclides = eval(content[start:end].replace('test_nuclides = ', ''))

print("="*95)
print("QFD SURVIVOR SEARCH - 285 NUCLIDE SWEEP")
print("="*95)
print()
print("Comparing:")
print("  1. BASELINE: Spherical approximation (ecc = 0)")
print("  2. SURVIVOR: Optimal eccentricity (0 ≤ ecc ≤ 0.25)")
print()
print(f"Parameters:")
print(f"  E_volume  = {E_volume:.3f} MeV")
print(f"  E_surface = {E_surface:.3f} MeV")
print(f"  a_sym     = {a_sym:.3f} MeV")
print(f"  a_disp    = {a_disp:.3f} MeV (shield={SHIELD_FACTOR})")
print(f"  Isomer bonus = {E_surface * BONUS_STRENGTH:.3f} MeV")
print()
print(f"Running on {len(test_nuclides)} nuclides...")
print()

# ============================================================================
# RUN COMPARISON
# ============================================================================
results = []

for i, (name, Z_exp, A) in enumerate(test_nuclides):
    N_exp = A - Z_exp

    # Baseline: spherical
    Z_sphere = find_spherical_Z(A)
    Delta_sphere = Z_sphere - Z_exp

    # Survivor: optimal eccentricity
    Z_survivor, ecc_opt = find_survivor_state(A)
    Delta_survivor = Z_survivor - Z_exp

    results.append({
        'name': name,
        'A': A,
        'Z_exp': Z_exp,
        'N_exp': N_exp,
        'Z_sphere': Z_sphere,
        'Delta_sphere': Delta_sphere,
        'Z_survivor': Z_survivor,
        'Delta_survivor': Delta_survivor,
        'ecc_opt': ecc_opt,
        'improvement': abs(Delta_sphere) - abs(Delta_survivor),
    })

    if (i + 1) % 50 == 0:
        print(f"  Processed {i+1}/{len(test_nuclides)}...")

print()
print("="*95)
print("RESULTS")
print("="*95)
print()

# ============================================================================
# STATISTICS
# ============================================================================
errors_sphere = [abs(r['Delta_sphere']) for r in results]
errors_survivor = [abs(r['Delta_survivor']) for r in results]

exact_sphere = sum(e == 0 for e in errors_sphere)
exact_survivor = sum(e == 0 for e in errors_survivor)

print("OVERALL PERFORMANCE:")
print("-"*95)
print(f"{'Model':<20} {'Exact':<15} {'Mean |ΔZ|':<15} {'Median |ΔZ|':<15}")
print("-"*95)
print(f"{'Baseline (sphere)':<20} {exact_sphere}/{len(results)} ({100*exact_sphere/len(results):.1f}%)  "
      f"{np.mean(errors_sphere):.3f}          {np.median(errors_sphere):.1f}")
print(f"{'Survivor (ecc opt)':<20} {exact_survivor}/{len(results)} ({100*exact_survivor/len(results):.1f}%)  "
      f"{np.mean(errors_survivor):.3f}          {np.median(errors_survivor):.1f}")
print()

improvement_count = sum(r['improvement'] > 0 for r in results)
worse_count = sum(r['improvement'] < 0 for r in results)
neutral_count = sum(r['improvement'] == 0 for r in results)

print(f"Eccentricity effect:")
print(f"  Improved:  {improvement_count}/{len(results)} ({100*improvement_count/len(results):.1f}%)")
print(f"  Worsened:  {worse_count}/{len(results)} ({100*worse_count/len(results):.1f}%)")
print(f"  Neutral:   {neutral_count}/{len(results)} ({100*neutral_count/len(results):.1f}%)")
print()

# By mass region
light = [r for r in results if r['A'] < 40]
medium = [r for r in results if 40 <= r['A'] < 100]
heavy = [r for r in results if 100 <= r['A'] < 200]
superheavy = [r for r in results if r['A'] >= 200]

print("PERFORMANCE BY REGION:")
print("-"*95)
print(f"{'Region':<25} {'Baseline':<25} {'Survivor':<25}")
print("-"*95)

for region_name, group in [("Light (A<40)", light), ("Medium (40≤A<100)", medium),
                            ("Heavy (100≤A<200)", heavy), ("Superheavy (A≥200)", superheavy)]:
    if len(group) > 0:
        errs_s = [abs(r['Delta_sphere']) for r in group]
        errs_surv = [abs(r['Delta_survivor']) for r in group]
        ex_s = sum(e == 0 for e in errs_s)
        ex_surv = sum(e == 0 for e in errs_surv)

        baseline_str = f"{ex_s}/{len(group)} ({100*ex_s/len(group):>5.1f}%)"
        survivor_str = f"{ex_surv}/{len(group)} ({100*ex_surv/len(group):>5.1f}%)"

        print(f"{region_name:<25} {baseline_str:<25} {survivor_str:<25}")

print()

# ============================================================================
# DEFORMATION MAP
# ============================================================================
print("="*95)
print("TOPOLOGICAL DEFORMATION MAP")
print("="*95)
print()
print("Nuclei with optimal ecc > 0.10 (significantly deformed survivors):")
print("-"*95)
print(f"{'Nuclide':<12} {'A':<5} {'Z_exp':<6} {'Z_pred':<6} {'ecc_opt':<10} {'ΔZ':<6} {'Status'}")
print("-"*95)

deformed = [r for r in results if r['ecc_opt'] >= 0.10]
deformed_sorted = sorted(deformed, key=lambda r: -r['ecc_opt'])

for r in deformed_sorted[:20]:  # Top 20 most deformed
    status = "✓" if r['Delta_survivor'] == 0 else f"{r['Delta_survivor']:+d}"
    print(f"{r['name']:<12} {r['A']:<5} {r['Z_exp']:<6} {r['Z_survivor']:<6} "
          f"{r['ecc_opt']:<10.3f} {status:<6}")

print()
print(f"Total deformed nuclei (ecc ≥ 0.10): {len(deformed)}/{len(results)} "
      f"({100*len(deformed)/len(results):.1f}%)")
print()

# ============================================================================
# KEY TEST CASES
# ============================================================================
print("="*95)
print("KEY TEST CASES")
print("="*95)
print(f"{'Nuclide':<10} {'Z_exp':<6} {'Baseline':<10} {'Survivor':<10} {'ecc_opt':<10} {'Improvement'}")
print("-"*95)

key_cases = [
    ("He-4", 2, 4),
    ("O-16", 8, 16),
    ("Ca-40", 20, 40),
    ("Fe-56", 26, 56),
    ("Ni-58", 28, 58),
    ("Sn-112", 50, 112),
    ("Xe-136", 54, 136),
    ("Pb-208", 82, 208),
    ("U-238", 92, 238),
]

for name, Z_exp, A in key_cases:
    r = next((res for res in results if res['name'] == name), None)
    if r:
        base_status = "✓" if r['Delta_sphere'] == 0 else f"{r['Delta_sphere']:+d}"
        surv_status = "✓" if r['Delta_survivor'] == 0 else f"{r['Delta_survivor']:+d}"
        improvement = "✓" if r['improvement'] > 0 else ("✗" if r['improvement'] < 0 else "=")

        print(f"{name:<10} {Z_exp:<6} {base_status:<10} {surv_status:<10} "
              f"{r['ecc_opt']:<10.3f} {improvement}")

print()

# ============================================================================
# VERDICT
# ============================================================================
print("="*95)
print("VERDICT")
print("="*95)
print()

delta_exact = exact_survivor - exact_sphere
delta_mean = np.mean(errors_sphere) - np.mean(errors_survivor)

if delta_exact > 10:
    print(f"✓✓ BREAKTHROUGH: Eccentricity freedom improves accuracy by {delta_exact} exact matches!")
    print("Topological deformation is essential for survivors.")
elif delta_exact > 0:
    print(f"✓ MODEST IMPROVEMENT: +{delta_exact} exact matches")
    print(f"Mean |ΔZ| change: {delta_mean:+.3f} charges")
elif delta_exact == 0:
    print(f"NEUTRAL: Same number of exact matches ({exact_sphere})")
    print(f"Mean |ΔZ| change: {delta_mean:+.3f} charges")
    if abs(delta_mean) > 0.1:
        print("Systematic shift detected.")
else:
    print(f"✗ REGRESSION: Eccentricity worsens accuracy by {-delta_exact} exact matches")
    print(f"Mean |ΔZ| change: {delta_mean:+.3f} charges")
    print()
    print("DIAGNOSIS:")
    print("  G_disp = 1/(1+ecc) reduces displacement penalty too much")
    print("  → Favors higher Z (more vacuum displacement)")
    print("  → Pulls predictions AWAY from low-Z survivors")
    print()
    print("RECOMMENDATION:")
    print("  Recalibrate G_disp formula to balance effects")
    print("  Or try G_disp = 1 + k·ecc with k < 0 (same sign as G_surf)")

print()
print("="*95)
print("TOPOLOGICAL INSIGHT")
print("="*95)
print()
print("The deformation map reveals which nuclei prefer non-spherical geometries.")
print("If accuracy improves: Survivors ARE shape-shifters.")
print("If accuracy worsens: Need different coupling between ecc and energy terms.")
print()
print("="*95)
