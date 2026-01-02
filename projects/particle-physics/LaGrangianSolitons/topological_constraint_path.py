#!/usr/bin/env python3
"""
TOPOLOGICAL CONSTRAINT PATH - Sequential Vortex Addition Only
===========================================================================
User's specification: Soliton can only change winding by ±1 topological step.

Key difference from previous implementation:
  - Previous: Test all Z ∈ [1, A], penalty if Z ≠ prev_Z
  - This: Test ONLY Z ∈ {prev_z-1, prev_z, prev_z+1}, penalty if Z ≠ prev_Z

Physical interpretation:
  - Vortex winding number changes by discrete topological events
  - Can't jump from Z=20 → Z=30 (would need to add 10 vortices sequentially)
  - Must pass through all intermediate winding numbers

Topological constraint: ΔZ ∈ {-1, 0, +1} per mass step
===========================================================================
"""

import numpy as np

# ============================================================================
# FUNDAMENTAL CONSTANTS
# ============================================================================
alpha_fine   = 1.0 / 137.036
beta_vacuum  = 1.0 / 3.058231
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

# USER'S SPECIFICATION: κ = 0.15 × E_surface
KAPPA = 0.15

def get_resonance_bonus(Z, N):
    bonus = 0
    if Z in ISOMER_NODES: bonus += E_surface * BONUS_STRENGTH
    if N in ISOMER_NODES: bonus += E_surface * BONUS_STRENGTH
    if Z in ISOMER_NODES and N in ISOMER_NODES:
        bonus *= 1.5
    return bonus

def qfd_base_energy(A, Z):
    N = A - Z
    if N < 0 or Z < 1:
        return float('inf')
    q = Z / A
    E_bulk = E_volume * A
    E_surf = E_surface * (A**(2/3))
    E_asym = a_sym * A * ((1 - 2*q)**2)
    E_vac  = a_disp * (Z**2) / (A**(1/3))
    E_iso  = -get_resonance_bonus(Z, N)
    return E_bulk + E_surf + E_asym + E_vac + E_iso

def build_topologically_constrained_path(A_max):
    """
    Build path with STRICT topological constraint.

    At each A, only test Z ∈ {prev_z-1, prev_z, prev_z+1}
    (Can't jump arbitrarily - must add/remove vortices one at a time)
    """
    path = {1: 1}  # H-1: A=1, Z=1

    for A in range(2, A_max + 1):
        prev_z = path[A - 1]

        # TOPOLOGICAL CONSTRAINT: Only ±1 or stay
        # Allow Z-1 to handle isotope chains (Z constant, N increases)
        candidates = [prev_z - 1, prev_z, prev_z + 1]

        best_Z = prev_z
        best_E = float('inf')

        for Z_test in candidates:
            if Z_test < 1 or Z_test >= A:  # Physical bounds
                continue

            E_base = qfd_base_energy(A, Z_test)

            # Phase-slip penalty if Z changes
            if Z_test != prev_z:
                E_total = E_base + KAPPA * E_surface
            else:
                E_total = E_base

            if E_total < best_E:
                best_E = E_total
                best_Z = Z_test

        path[A] = best_Z

    return path

# ============================================================================
# LOAD TEST DATA
# ============================================================================
with open('qfd_optimized_suite.py', 'r') as f:
    content = f.read()

start = content.find('test_nuclides = [')
end = content.find(']', start) + 1
test_nuclides = eval(content[start:end].replace('test_nuclides = ', ''))

print("="*95)
print("TOPOLOGICAL CONSTRAINT PATH - Sequential Vortex Addition")
print("="*95)
print()
print("Constraint: ΔZ ∈ {-1, 0, +1} per mass step (strict topology)")
print(f"κ = {KAPPA:.2f} × E_surface = {KAPPA * E_surface:.3f} MeV (phase-slip barrier)")
print()

# ============================================================================
# BUILD PATH
# ============================================================================
A_max = max(A for _, _, A in test_nuclides)
topo_path = build_topologically_constrained_path(A_max)

print(f"Building path from A=1 to A={A_max}...")
print()

# ============================================================================
# EVALUATE
# ============================================================================
results_topo = []
results_static = []

# Static baseline for comparison
def static_solver(A):
    best_Z = 1
    best_E = qfd_base_energy(A, 1)
    for Z in range(1, A):
        E = qfd_base_energy(A, Z)
        if E < best_E:
            best_E = E
            best_Z = Z
    return best_Z

for name, Z_exp, A in test_nuclides:
    # Topological path
    Z_topo = topo_path.get(A, 1)
    Delta_topo = Z_topo - Z_exp

    # Static
    Z_stat = static_solver(A)
    Delta_stat = Z_stat - Z_exp

    results_topo.append({
        'name': name,
        'A': A,
        'Z_exp': Z_exp,
        'Z_pred': Z_topo,
        'Delta_Z': Delta_topo,
    })

    results_static.append({
        'name': name,
        'A': A,
        'Z_exp': Z_exp,
        'Z_pred': Z_stat,
        'Delta_Z': Delta_stat,
    })

# ============================================================================
# STATISTICS
# ============================================================================
print("="*95)
print("RESULTS")
print("="*95)
print()

errors_topo = [abs(r['Delta_Z']) for r in results_topo]
errors_stat = [abs(r['Delta_Z']) for r in results_static]

exact_topo = sum(e == 0 for e in errors_topo)
exact_stat = sum(e == 0 for e in errors_stat)

print(f"{'Model':<30} {'Exact':<20} {'Mean |ΔZ|':<15} {'Median |ΔZ|'}")
print("-"*95)
print(f"{'Static (no constraint)':<30} {exact_stat}/{len(results_static)} ({100*exact_stat/len(results_static):.1f}%)  "
      f"{np.mean(errors_stat):<15.3f} {np.median(errors_stat):.1f}")
print(f"{'Topological (ΔZ ∈ {{-1,0,+1}})':<30} {exact_topo}/{len(results_topo)} ({100*exact_topo/len(results_topo):.1f}%)  "
      f"{np.mean(errors_topo):<15.3f} {np.median(errors_topo):.1f}")
print()

improvement = exact_topo - exact_stat
if improvement > 0:
    print(f"✓ IMPROVEMENT: +{improvement} exact matches")
    print(f"  Topological constraint helps!")
elif improvement < 0:
    print(f"✗ REGRESSION: {improvement} exact matches")
    print(f"  Topological constraint hurts accuracy")
else:
    print(f"= NEUTRAL: Same accuracy")
    print(f"  Constraint doesn't affect predictions")

print()

# By region
for model_name, results in [("Static", results_static), ("Topological", results_topo)]:
    light = [r for r in results if r['A'] < 40]
    medium = [r for r in results if 40 <= r['A'] < 100]
    heavy = [r for r in results if 100 <= r['A'] < 200]
    superheavy = [r for r in results if r['A'] >= 200]

    print(f"{model_name.upper()} - BY REGION:")
    print("-"*95)
    for region_name, group in [("Light (A<40)", light), ("Medium (40≤A<100)", medium),
                                ("Heavy (100≤A<200)", heavy), ("Superheavy (A≥200)", superheavy)]:
        if len(group) > 0:
            errs = [abs(r['Delta_Z']) for r in group]
            ex = sum(e == 0 for e in errs)
            print(f"  {region_name:<25} {ex}/{len(group)} ({100*ex/len(group):>5.1f}%)  "
                  f"Mean|ΔZ|={np.mean(errs):.2f}")
    print()

# ============================================================================
# KEY TEST CASES
# ============================================================================
print("="*95)
print("KEY TEST CASES")
print("="*95)
print(f"{'Nuclide':<12} {'A':<5} {'Z_exp':<8} {'Static':<10} {'Topo':<10} {'Improvement'}")
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
    r_topo = next((r for r in results_topo if r['name'] == name), None)
    r_stat = next((r for r in results_static if r['name'] == name), None)

    if r_topo and r_stat:
        stat_str = "✓" if r_stat['Delta_Z'] == 0 else f"{r_stat['Delta_Z']:+d}"
        topo_str = "✓" if r_topo['Delta_Z'] == 0 else f"{r_topo['Delta_Z']:+d}"

        improvement = ""
        if abs(r_topo['Delta_Z']) < abs(r_stat['Delta_Z']):
            improvement = "✓ Better"
        elif abs(r_topo['Delta_Z']) > abs(r_stat['Delta_Z']):
            improvement = "✗ Worse"
        else:
            improvement = "= Same"

        print(f"{name:<12} {A:<5} {Z_exp:<8} {stat_str:<10} {topo_str:<10} {improvement}")

print()

# ============================================================================
# PATH ANALYSIS
# ============================================================================
print("="*95)
print("TOPOLOGICAL PATH STRUCTURE")
print("="*95)
print()

# Find transitions
transitions = []
for A in range(2, A_max + 1):
    Z_curr = topo_path[A]
    Z_prev = topo_path[A - 1]
    dZ = Z_curr - Z_prev
    if dZ != 0:
        transitions.append((A, Z_prev, Z_curr, dZ))

print(f"Total steps: {A_max - 1}")
print(f"Z changes: {len(transitions)}")
print(f"Z constant: {A_max - 1 - len(transitions)}")
print()

# Count transition types
dZ_plus1 = sum(1 for _, _, _, dZ in transitions if dZ == 1)
dZ_minus1 = sum(1 for _, _, _, dZ in transitions if dZ == -1)
dZ_other = len(transitions) - dZ_plus1 - dZ_minus1

print(f"Transition breakdown:")
print(f"  ΔZ = +1: {dZ_plus1} ({100*dZ_plus1/len(transitions) if len(transitions)>0 else 0:.1f}%)")
print(f"  ΔZ = -1: {dZ_minus1} ({100*dZ_minus1/len(transitions) if len(transitions)>0 else 0:.1f}%)")
print(f"  ΔZ = other: {dZ_other}")
print()

# Show final Z at high A
print("Final Z values (high A):")
for A in [200, 210, 220, 230, 238]:
    if A in topo_path:
        Z_topo = topo_path[A]
        Z_stat = static_solver(A)
        print(f"  A={A}: Z_topo={Z_topo}, Z_static={Z_stat}")

print()
print("="*95)
print("VERDICT")
print("="*95)
print()

if exact_topo > exact_stat:
    print("✓ TOPOLOGICAL CONSTRAINT IMPROVES PREDICTIONS")
    print()
    print("Physical interpretation:")
    print("  - Soliton winding number changes sequentially (ΔZ = ±1)")
    print("  - Path-dependent buildup with phase-slip barrier")
    print("  - Nuclei inherit topological structure from predecessors")
elif exact_topo == exact_stat and KAPPA > 0:
    print("= CONSTRAINT EQUIVALENT TO STATIC (with κ penalty)")
    print()
    print("Physical interpretation:")
    print("  - κ barrier exactly compensates for constraint")
    print("  - Net effect: same as unconstrained optimization")
else:
    print("✗ TOPOLOGICAL CONSTRAINT WORSENS PREDICTIONS")
    print()
    print("Physical interpretation:")
    print("  - Nuclei are NOT path-dependent")
    print("  - Each nucleus finds global Z minimum independently")
    print("  - No topological memory from sequential buildup")
    print()
    print("Conclusion:")
    print("  - Stability valley represents TRUE equilibrium states")
    print("  - Solitons can reconfigure winding number freely")
    print("  - Berry phase / hysteresis model doesn't apply")

print()
print("="*95)
