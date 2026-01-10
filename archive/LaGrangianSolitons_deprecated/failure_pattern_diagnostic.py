#!/usr/bin/env python3
"""
FAILURE PATTERN DIAGNOSTIC - RESONANCE ANALYSIS
===========================================================================
For each nuclide, calculate the CORRECTION needed to match experiment.

Goal: Find patterns in required corrections:
  - Do magic number nuclei need ecc ≈ 0? (resonant = no correction)
  - Are corrections systematic vs random?
  - Do failures cluster near/far from magic numbers?

Method:
  For each nuclide, find minimal eccentricity ecc* such that:
    Z_pred(A, ecc*) = Z_exp

  If ecc* ≈ 0 at magic numbers → Resonant (model already correct)
  If ecc* systematic → Missing physics has a pattern
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

def get_resonance_bonus(Z, N):
    bonus = 0
    if Z in ISOMER_NODES: bonus += E_surface * BONUS_STRENGTH
    if N in ISOMER_NODES: bonus += E_surface * BONUS_STRENGTH
    if Z in ISOMER_NODES and N in ISOMER_NODES:
        bonus *= 1.5
    return bonus

# ============================================================================
# ENERGY FUNCTIONAL WITH DIAGNOSTIC PARAMETERS
# ============================================================================
def qfd_energy_with_correction(A, Z, ecc_surf=0, ecc_disp=0):
    """
    Energy with separate surface and displacement corrections.

    This allows us to diagnose WHICH term needs adjustment.
    """
    G_surf = 1.0 + ecc_surf**2
    G_disp = 1.0 + ecc_disp**2

    N = A - Z
    q = Z / A if A > 0 else 0

    E_bulk = E_volume * A
    E_surf = E_surface * (A**(2/3)) * G_surf
    E_asym = a_sym * A * ((1 - 2*q)**2)
    E_vac  = a_disp * (Z**2) / (A**(1/3)) * G_disp
    E_iso  = -get_resonance_bonus(Z, N)

    return E_bulk + E_surf + E_asym + E_vac + E_iso

def find_required_correction(A, Z_exp, correction_type='combined'):
    """
    Find minimal correction needed to make Z_pred = Z_exp.

    Returns:
      ecc_required: Eccentricity adjustment needed
      None if no correction can fix it (would need ecc > 0.5)
    """
    # Test range of corrections
    ecc_range = np.linspace(-0.5, 0.5, 100)

    for ecc in ecc_range:
        if correction_type == 'combined':
            # Both terms adjusted equally
            best_Z = find_stable_Z(A, ecc, ecc)
        elif correction_type == 'surface':
            # Only surface term adjusted
            best_Z = find_stable_Z(A, ecc, 0)
        elif correction_type == 'displacement':
            # Only displacement term adjusted
            best_Z = find_stable_Z(A, 0, ecc)

        if best_Z == Z_exp:
            return ecc

    return None  # No correction in range works

def find_stable_Z(A, ecc_surf, ecc_disp):
    """Find optimal Z with given corrections."""
    best_Z = 1
    best_E = qfd_energy_with_correction(A, 1, ecc_surf, ecc_disp)

    for Z in range(1, A):
        E = qfd_energy_with_correction(A, Z, ecc_surf, ecc_disp)
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
print("FAILURE PATTERN DIAGNOSTIC - RESONANCE ANALYSIS")
print("="*95)
print()
print("For each nuclide, finding MINIMAL CORRECTION needed to match experiment")
print()
print("Key question: Do magic number nuclei need ecc ≈ 0? (resonant = model correct)")
print()

# ============================================================================
# RUN DIAGNOSTIC
# ============================================================================
results = []

print(f"Analyzing {len(test_nuclides)} nuclides...")
print()

for i, (name, Z_exp, A) in enumerate(test_nuclides):
    N_exp = A - Z_exp

    # Baseline prediction
    Z_baseline = find_stable_Z(A, 0, 0)
    Delta_baseline = Z_baseline - Z_exp

    # Find required correction
    ecc_required = find_required_correction(A, Z_exp, correction_type='combined')

    # Check if magic
    Z_magic = Z_exp in ISOMER_NODES
    N_magic = N_exp in ISOMER_NODES
    doubly_magic = Z_magic and N_magic

    results.append({
        'name': name,
        'A': A,
        'Z_exp': Z_exp,
        'N_exp': N_exp,
        'Z_baseline': Z_baseline,
        'Delta_baseline': Delta_baseline,
        'ecc_required': ecc_required,
        'Z_magic': Z_magic,
        'N_magic': N_magic,
        'doubly_magic': doubly_magic,
    })

    if (i + 1) % 50 == 0:
        print(f"  Processed {i+1}/{len(test_nuclides)}...")

print()

# ============================================================================
# PATTERN ANALYSIS
# ============================================================================
print("="*95)
print("RESONANCE PATTERN ANALYSIS")
print("="*95)
print()

# Separate by magic character
survivors = [r for r in results if r['Delta_baseline'] == 0]
failures = [r for r in results if r['Delta_baseline'] != 0]

magic_survivors = [r for r in survivors if r['Z_magic'] or r['N_magic']]
nonmagic_survivors = [r for r in survivors if not (r['Z_magic'] or r['N_magic'])]

magic_failures = [r for r in failures if r['Z_magic'] or r['N_magic']]
nonmagic_failures = [r for r in failures if not (r['Z_magic'] or r['N_magic'])]

print("SURVIVAL BY MAGIC CHARACTER:")
print("-"*95)
print(f"{'Category':<30} {'Count':<10} {'Survivors':<15} {'Survival %'}")
print("-"*95)

total_magic = len([r for r in results if r['Z_magic'] or r['N_magic']])
total_nonmagic = len([r for r in results if not (r['Z_magic'] or r['N_magic'])])

surv_magic = len(magic_survivors)
surv_nonmagic = len(nonmagic_survivors)

print(f"{'Magic (Z or N in nodes)':<30} {total_magic:<10} {surv_magic:<15} "
      f"{100*surv_magic/total_magic if total_magic > 0 else 0:.1f}%")
print(f"{'Non-magic':<30} {total_nonmagic:<10} {surv_nonmagic:<15} "
      f"{100*surv_nonmagic/total_nonmagic if total_nonmagic > 0 else 0:.1f}%")

print()

# Corrections needed
print("CORRECTIONS NEEDED (for failures only):")
print("-"*95)

failures_with_correction = [r for r in failures if r['ecc_required'] is not None]
failures_no_correction = [r for r in failures if r['ecc_required'] is None]

print(f"Failures fixable with ecc < 0.5:   {len(failures_with_correction)}/{len(failures)}")
print(f"Failures needing ecc > 0.5:        {len(failures_no_correction)}/{len(failures)}")
print()

if len(failures_with_correction) > 0:
    ecc_values = [abs(r['ecc_required']) for r in failures_with_correction]

    print(f"Correction statistics (|ecc| for fixable failures):")
    print(f"  Mean:    {np.mean(ecc_values):.3f}")
    print(f"  Median:  {np.median(ecc_values):.3f}")
    print(f"  Min:     {np.min(ecc_values):.3f}")
    print(f"  Max:     {np.max(ecc_values):.3f}")
    print()

# Magic vs non-magic corrections
magic_fail_ecc = [abs(r['ecc_required']) for r in failures_with_correction
                   if (r['Z_magic'] or r['N_magic'])]
nonmagic_fail_ecc = [abs(r['ecc_required']) for r in failures_with_correction
                      if not (r['Z_magic'] or r['N_magic'])]

if len(magic_fail_ecc) > 0 and len(nonmagic_fail_ecc) > 0:
    print("CORRECTION BY MAGIC CHARACTER:")
    print("-"*95)
    print(f"Magic failures need:     mean |ecc| = {np.mean(magic_fail_ecc):.3f}")
    print(f"Non-magic failures need: mean |ecc| = {np.mean(nonmagic_fail_ecc):.3f}")
    print()

    if np.mean(magic_fail_ecc) < np.mean(nonmagic_fail_ecc):
        print("✓ Magic failures need SMALLER corrections (closer to resonance!)")
    else:
        print("✗ Magic failures need LARGER corrections (not resonant)")
    print()

# ============================================================================
# VISUALIZATION
# ============================================================================
print("="*95)
print("GENERATING DIAGNOSTIC PLOTS")
print("="*95)
print()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Correction vs Mass Number
ax = axes[0, 0]
for r in results:
    if r['Delta_baseline'] == 0:
        # Survivor - no correction needed
        ax.scatter(r['A'], 0, c='green', s=30, alpha=0.6, marker='o')
    elif r['ecc_required'] is not None:
        # Failure - correction available
        color = 'blue' if (r['Z_magic'] or r['N_magic']) else 'orange'
        ax.scatter(r['A'], r['ecc_required'], c=color, s=30, alpha=0.6)
    else:
        # Failure - no correction works
        ax.scatter(r['A'], 0.5, c='red', s=30, alpha=0.6, marker='x')

ax.axhline(0, color='black', linestyle='--', linewidth=0.5, label='No correction (resonant)')
ax.set_xlabel('Mass Number A')
ax.set_ylabel('Required Correction (ecc)')
ax.set_title('Correction Needed vs Mass Number')
ax.legend(['Resonance', 'Survivor', 'Magic failure', 'Non-magic failure', 'Unfixable'])
ax.grid(True, alpha=0.3)

# Plot 2: Correction vs Distance to Nearest Magic Number (Z)
ax = axes[0, 1]
for r in results:
    # Distance to nearest magic Z
    dist_Z = min(abs(r['Z_exp'] - m) for m in ISOMER_NODES)

    if r['Delta_baseline'] == 0:
        ax.scatter(dist_Z, 0, c='green', s=30, alpha=0.6)
    elif r['ecc_required'] is not None:
        color = 'blue' if (r['Z_magic'] or r['N_magic']) else 'orange'
        ax.scatter(dist_Z, abs(r['ecc_required']), c=color, s=30, alpha=0.6)

ax.axhline(0, color='black', linestyle='--', linewidth=0.5)
ax.set_xlabel('Distance to Nearest Magic Z')
ax.set_ylabel('|Required Correction|')
ax.set_title('Correction vs Proximity to Magic Number')
ax.grid(True, alpha=0.3)

# Plot 3: N-Z plane with correction magnitude
ax = axes[1, 0]
for r in results:
    if r['Delta_baseline'] == 0:
        size = 30
        color = 'green'
    elif r['ecc_required'] is not None:
        size = abs(r['ecc_required']) * 200 + 10
        color = 'red'
    else:
        size = 50
        color = 'black'

    ax.scatter(r['Z_exp'], r['N_exp'], c=color, s=size, alpha=0.5)

# Mark magic numbers
for m in ISOMER_NODES:
    if m < 140:
        ax.axvline(m, color='blue', linestyle=':', alpha=0.3, linewidth=0.5)
        ax.axhline(m, color='blue', linestyle=':', alpha=0.3, linewidth=0.5)

ax.set_xlabel('Proton Number Z')
ax.set_ylabel('Neutron Number N')
ax.set_title('N-Z Plane (size = correction needed)')
ax.grid(True, alpha=0.2)

# Plot 4: Histogram of corrections
ax = axes[1, 1]

if len(failures_with_correction) > 0:
    ecc_all = [r['ecc_required'] for r in failures_with_correction]

    ax.hist(ecc_all, bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero correction (resonant)')
    ax.set_xlabel('Required Correction (ecc)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Corrections Needed')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('FAILURE_PATTERN_DIAGNOSTIC.png', dpi=150, bbox_inches='tight')
print("Saved: FAILURE_PATTERN_DIAGNOSTIC.png")
print()

# ============================================================================
# SPECIFIC EXAMPLES
# ============================================================================
print("="*95)
print("SPECIFIC EXAMPLES - RESONANCE CHECK")
print("="*95)
print()
print(f"{'Nuclide':<12} {'Z':<5} {'N':<5} {'Magic?':<12} {'Baseline':<10} {'ecc_req':<10} {'Resonant?'}")
print("-"*95)

key_cases = [
    ("He-4", 2, 4),
    ("O-16", 8, 16),
    ("Ca-40", 20, 40),
    ("Ni-58", 28, 58),
    ("Sn-120", 50, 120),
    ("Pb-208", 82, 208),
    ("Fe-56", 26, 56),
    ("Xe-136", 54, 136),
]

for name, Z_exp, A in key_cases:
    r = next((res for res in results if res['name'] == name), None)
    if r:
        magic_str = "Doubly" if r['doubly_magic'] else ("Z" if r['Z_magic'] else ("N" if r['N_magic'] else "No"))
        baseline_str = "✓" if r['Delta_baseline'] == 0 else f"{r['Delta_baseline']:+d}"
        ecc_str = "0.000" if r['ecc_required'] is None or abs(r['ecc_required']) < 0.001 else f"{r['ecc_required']:+.3f}"
        resonant = "✓ YES" if abs(r['ecc_required'] or 0) < 0.05 else "✗ NO"

        print(f"{name:<12} {r['Z_exp']:<5} {r['N_exp']:<5} {magic_str:<12} {baseline_str:<10} "
              f"{ecc_str:<10} {resonant}")

print()

# ============================================================================
# INTERPRETATION
# ============================================================================
print("="*95)
print("INTERPRETATION - RESONANCE HYPOTHESIS")
print("="*95)
print()

# Count near-resonant failures
near_resonant = [r for r in failures_with_correction if abs(r['ecc_required']) < 0.1]
far_from_resonant = [r for r in failures_with_correction if abs(r['ecc_required']) >= 0.1]

print(f"Failures near resonance (|ecc| < 0.1):  {len(near_resonant)}/{len(failures_with_correction)}")
print(f"Failures far from resonance (|ecc| ≥ 0.1): {len(far_from_resonant)}/{len(failures_with_correction)}")
print()

if len(magic_fail_ecc) > 0 and np.mean(magic_fail_ecc) < 0.1:
    print("✓✓ RESONANCE CONFIRMED:")
    print("   Magic number failures need small corrections (≈0)")
    print("   → Model is CLOSE to correct physics at magic numbers")
    print("   → Missing term is small perturbation near resonances")
elif len(magic_fail_ecc) > 0:
    print("✗ RESONANCE NOT FOUND:")
    print("   Magic number failures need large corrections")
    print("   → Model fundamentally wrong even at magic numbers")
else:
    print("? INSUFFICIENT DATA:")
    print("   Not enough magic number failures to test resonance hypothesis")

print()
print("="*95)
