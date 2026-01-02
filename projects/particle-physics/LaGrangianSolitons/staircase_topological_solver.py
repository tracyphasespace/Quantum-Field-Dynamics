#!/usr/bin/env python3
"""
QFD STAIRCASE SOLVER - TOPOLOGICAL PHASE HYSTERESIS
===========================================================================
Implements Berry phase / topological tension term that creates MEMORY.

Key physics:
  - Soliton winding number (Z) is topologically locked
  - Changing Z requires overcoming phase-slip barrier: E_slip ≈ κ·E_surface
  - Z inherits from previous mass number: Z(A) depends on Z(A-1)
  - Discrete "snap" transitions, not smooth optimization

This models nuclear buildup as a PATH-DEPENDENT process:
  H-1 → H-2 → He-3 → He-4 → ... → U-238

At each step, Z can only change if ΔE > κ (phase-slip threshold).
===========================================================================
"""

import numpy as np
import matplotlib.pyplot as plt

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

# TOPOLOGICAL PHASE STIFFNESS (Berry phase barrier)
# Fraction of surface energy required to "slip" winding number
KAPPA = 0.25  # Phase-slip barrier strength

def get_resonance_bonus(Z, N):
    bonus = 0
    if Z in ISOMER_NODES: bonus += E_surface * BONUS_STRENGTH
    if N in ISOMER_NODES: bonus += E_surface * BONUS_STRENGTH
    if Z in ISOMER_NODES and N in ISOMER_NODES:
        bonus *= 1.5
    return bonus

# ============================================================================
# ENERGY FUNCTIONAL WITH TOPOLOGICAL TENSION
# ============================================================================
def qfd_base_energy(A, Z):
    """Base soliton energy (no phase-slip term)."""
    N = A - Z
    q = Z / A if A > 0 else 0

    E_bulk = E_volume * A
    E_surf = E_surface * (A**(2/3))
    E_asym = a_sym * A * ((1 - 2*q)**2)
    E_vac  = a_disp * (Z**2) / (A**(1/3))
    E_iso  = -get_resonance_bonus(Z, N)

    return E_bulk + E_surf + E_asym + E_vac + E_iso

def qfd_staircase_energy(A, Z, Z_prev):
    """
    Energy with topological phase-slip barrier.

    If Z ≠ Z_prev, soliton must overcome Berry phase barrier
    to re-align its winding number with the vacuum lattice.

    Parameters:
      A: Mass number
      Z: Proposed proton number
      Z_prev: Previous stable proton number (topological memory)

    Returns:
      Total energy including phase-slip penalty
    """
    E_base = qfd_base_energy(A, Z)

    # Topological tension: Cost to change winding number
    if Z != Z_prev:
        # Phase-slip barrier ≈ κ × E_surface
        # Physical interpretation: Must break phase-lock with vacuum
        E_phase_slip = KAPPA * E_surface * abs(Z - Z_prev)
        return E_base + E_phase_slip

    return E_base

# ============================================================================
# PATH-DEPENDENT STAIRCASE SOLVER
# ============================================================================
def build_stability_path(A_max=250):
    """
    Build nucleus sequentially from A=1 to A_max.

    At each step, Z inherits from previous A and can only jump
    if energy gain exceeds topological phase-slip barrier.

    Returns:
      path: List of (A, Z_stable) tuples following the staircase
    """
    path = []

    # Initialize: A=1 is Hydrogen (Z=1)
    Z_current = 1
    path.append((1, Z_current))

    # Build up mass number sequentially
    for A in range(2, A_max + 1):
        # Test all possible Z values
        best_Z = Z_current  # Default: stay at current winding number
        best_E = qfd_staircase_energy(A, Z_current, Z_current)

        for Z_test in range(1, A):
            E_test = qfd_staircase_energy(A, Z_test, Z_current)

            if E_test < best_E:
                best_E = E_test
                best_Z = Z_test

        # Update topological memory
        Z_current = best_Z
        path.append((A, Z_current))

    return path

# ============================================================================
# COMPARISON: STAIRCASE vs STATIC
# ============================================================================
def static_solver(A):
    """Static optimization (no memory) - current baseline."""
    best_Z = 1
    best_E = qfd_base_energy(A, 1)

    for Z in range(1, A):
        E = qfd_base_energy(A, Z)
        if E < best_E:
            best_E = E
            best_Z = Z

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
print("QFD STAIRCASE SOLVER - TOPOLOGICAL PHASE HYSTERESIS")
print("="*95)
print()
print("Berry Phase Term: ℒ_phase = κ ∫ (∇θ - eA)² dV")
print()
print(f"Parameters:")
print(f"  E_volume  = {E_volume:.3f} MeV")
print(f"  E_surface = {E_surface:.3f} MeV")
print(f"  a_sym     = {a_sym:.3f} MeV")
print(f"  a_disp    = {a_disp:.3f} MeV")
print(f"  κ (phase stiffness) = {KAPPA:.3f} × E_surface = {KAPPA * E_surface:.3f} MeV")
print()
print("Building stability path from H-1 to heavy nuclei...")
print()

# ============================================================================
# BUILD STAIRCASE PATH
# ============================================================================
A_max = max(A for _, _, A in test_nuclides)
staircase_path = build_stability_path(A_max)

# Convert to lookup dictionary
staircase_Z = {A: Z for A, Z in staircase_path}

# ============================================================================
# EVALUATE ON TEST SET
# ============================================================================
results_staircase = []
results_static = []

for name, Z_exp, A in test_nuclides:
    N_exp = A - Z_exp

    # Staircase prediction (path-dependent)
    Z_stair = staircase_Z.get(A, 1)
    Delta_stair = Z_stair - Z_exp

    # Static prediction (independent optimization)
    Z_stat = static_solver(A)
    Delta_stat = Z_stat - Z_exp

    results_staircase.append({
        'name': name,
        'A': A,
        'Z_exp': Z_exp,
        'Z_pred': Z_stair,
        'Delta_Z': Delta_stair,
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
print("RESULTS - STAIRCASE vs STATIC")
print("="*95)
print()

errors_stair = [abs(r['Delta_Z']) for r in results_staircase]
errors_stat = [abs(r['Delta_Z']) for r in results_static]

exact_stair = sum(e == 0 for e in errors_stair)
exact_stat = sum(e == 0 for e in errors_stat)

print(f"{'Model':<25} {'Exact':<20} {'Mean |ΔZ|':<15} {'Median |ΔZ|'}")
print("-"*95)
print(f"{'Static (baseline)':<25} {exact_stat}/{len(results_static)} ({100*exact_stat/len(results_static):.1f}%)  "
      f"{np.mean(errors_stat):<15.3f} {np.median(errors_stat):.1f}")
print(f"{'Staircase (Berry phase)':<25} {exact_stair}/{len(results_staircase)} ({100*exact_stair/len(results_staircase):.1f}%)  "
      f"{np.mean(errors_stair):<15.3f} {np.median(errors_stair):.1f}")
print()

improvement = exact_stair - exact_stat
if improvement > 0:
    print(f"✓ IMPROVEMENT: +{improvement} exact matches ({improvement/len(results_static)*100:.1f} percentage points)")
elif improvement < 0:
    print(f"✗ REGRESSION: {improvement} exact matches")
else:
    print(f"= NEUTRAL: Same accuracy")

print()

# By mass region
for model_name, results in [("Static", results_static), ("Staircase", results_staircase)]:
    light = [r for r in results if r['A'] < 40]
    medium = [r for r in results if 40 <= r['A'] < 100]
    heavy = [r for r in results if 100 <= r['A'] < 200]
    superheavy = [r for r in results if r['A'] >= 200]

    print(f"{model_name.upper()} - PERFORMANCE BY REGION:")
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
print(f"{'Nuclide':<12} {'A':<5} {'Z_exp':<8} {'Static':<10} {'Staircase':<10} {'Improvement'}")
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
    r_stair = next((r for r in results_staircase if r['name'] == name), None)
    r_stat = next((r for r in results_static if r['name'] == name), None)

    if r_stair and r_stat:
        stat_str = "✓" if r_stat['Delta_Z'] == 0 else f"{r_stat['Delta_Z']:+d}"
        stair_str = "✓" if r_stair['Delta_Z'] == 0 else f"{r_stair['Delta_Z']:+d}"

        improvement = ""
        if abs(r_stair['Delta_Z']) < abs(r_stat['Delta_Z']):
            improvement = "✓ Better"
        elif abs(r_stair['Delta_Z']) > abs(r_stat['Delta_Z']):
            improvement = "✗ Worse"
        else:
            improvement = "= Same"

        print(f"{name:<12} {A:<5} {Z_exp:<8} {stat_str:<10} {stair_str:<10} {improvement}")

print()

# ============================================================================
# PHASE-LOCK ANALYSIS
# ============================================================================
print("="*95)
print("TOPOLOGICAL PHASE-LOCK ANALYSIS")
print("="*95)
print()

# Count how many transitions occur
transitions = []
for i in range(1, len(staircase_path)):
    A_prev, Z_prev = staircase_path[i-1]
    A_curr, Z_curr = staircase_path[i]

    if Z_curr != Z_prev:
        transitions.append((A_curr, Z_prev, Z_curr, Z_curr - Z_prev))

print(f"Total mass steps: {len(staircase_path) - 1}")
print(f"Transitions (Z changes): {len(transitions)}")
print(f"Phase-locked steps (Z constant): {len(staircase_path) - 1 - len(transitions)}")
print(f"Average lock length: {(len(staircase_path) - 1) / len(transitions) if len(transitions) > 0 else 0:.1f} steps")
print()

print("First 20 transitions:")
print(f"{'A':<6} {'Z_prev':<8} {'Z_new':<8} {'ΔZ':<6} {'Type'}")
print("-"*95)
for A, Z_prev, Z_new, dZ in transitions[:20]:
    transition_type = "Magic" if Z_new in ISOMER_NODES else "Regular"
    print(f"{A:<6} {Z_prev:<8} {Z_new:<8} {dZ:+<6} {transition_type}")

print()

# ============================================================================
# VISUALIZATION
# ============================================================================
print("="*95)
print("GENERATING STAIRCASE VISUALIZATION")
print("="*95)
print()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Stability paths comparison
ax = axes[0, 0]

# Staircase path
A_stair = [A for A, Z in staircase_path]
Z_stair = [Z for A, Z in staircase_path]
ax.plot(A_stair, Z_stair, 'b-', linewidth=2, label='Staircase (Berry phase)', alpha=0.7)

# Static path
A_range = range(1, A_max + 1)
Z_static_path = [static_solver(A) for A in A_range]
ax.plot(A_range, Z_static_path, 'r--', linewidth=1.5, label='Static (no memory)', alpha=0.7)

# Experimental data
A_exp = [A for _, _, A in test_nuclides]
Z_exp_all = [Z for Z, _, A in test_nuclides]
ax.scatter(A_exp, Z_exp_all, c='green', s=20, alpha=0.6, label='Experimental', zorder=3)

ax.set_xlabel('Mass Number A')
ax.set_ylabel('Proton Number Z')
ax.set_title('Stability Valley: Staircase vs Static')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Transition points
ax = axes[0, 1]

trans_A = [A for A, _, _, _ in transitions]
trans_dZ = [dZ for _, _, _, dZ in transitions]

ax.scatter(trans_A, trans_dZ, c='blue', s=40, alpha=0.6)
ax.axhline(0, color='black', linestyle='--', linewidth=0.5)
ax.set_xlabel('Mass Number A')
ax.set_ylabel('ΔZ (transition jump)')
ax.set_title('Phase-Slip Events (Z transitions)')
ax.grid(True, alpha=0.3)

# Plot 3: Error comparison
ax = axes[1, 0]

A_test = [r['A'] for r in results_static]
errors_static_plot = [r['Delta_Z'] for r in results_static]
errors_stair_plot = [r['Delta_Z'] for r in results_staircase]

ax.scatter(A_test, errors_static_plot, c='red', s=30, alpha=0.5, label='Static')
ax.scatter(A_test, errors_stair_plot, c='blue', s=30, alpha=0.5, label='Staircase')
ax.axhline(0, color='green', linestyle='--', linewidth=1.5, label='Exact')
ax.set_xlabel('Mass Number A')
ax.set_ylabel('ΔZ (Prediction Error)')
ax.set_title('Prediction Errors')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Improvement histogram
ax = axes[1, 1]

improvements_all = [abs(r_stat['Delta_Z']) - abs(r_stair['Delta_Z'])
                    for r_stat, r_stair in zip(results_static, results_staircase)]

ax.hist(improvements_all, bins=30, color='purple', alpha=0.7, edgecolor='black')
ax.axvline(0, color='red', linestyle='--', linewidth=2, label='No change')
ax.set_xlabel('Improvement (|ΔZ_static| - |ΔZ_staircase|)')
ax.set_ylabel('Count')
ax.set_title('Distribution of Improvements')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('STAIRCASE_TOPOLOGICAL_SOLVER.png', dpi=150, bbox_inches='tight')
print("Saved: STAIRCASE_TOPOLOGICAL_SOLVER.png")
print()

print("="*95)
print("INTERPRETATION")
print("="*95)
print()
print("The Berry phase term introduces TOPOLOGICAL MEMORY:")
print("  - Z is phase-locked to previous configuration")
print("  - Only jumps when ΔE > κ·E_surface (phase-slip barrier)")
print("  - Creates discrete 'staircase' transitions, not smooth optimization")
print()
print("If staircase improves accuracy:")
print("  → Nuclei build sequentially, maintaining topological phase coherence")
print("  → Survivors are phase-locked configurations")
print()
print("If staircase matches static:")
print("  → Phase-slip barrier too weak (κ too small)")
print("  → Need to calibrate κ from experimental transition data")
print()
print("="*95)
