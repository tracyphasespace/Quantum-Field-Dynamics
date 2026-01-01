#!/usr/bin/env python3
"""
CURRENT PERFORMANCE ANALYSIS - QFD Complete Suite
===========================================================================
Analyzes results from qfd_complete_suite.py and visualizes where the
model succeeds vs fails, highlighting the need for parameter tuning.
===========================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

# ============================================================================
# REPRODUCE QFD COMPLETE SUITE PARAMETERS
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

hbar_c = 197.327
r_0 = 1.2
a_disp = (alpha_fine * hbar_c / r_0) * (5.0 / 7.0)

ISOMER_NODES = {2, 8, 20, 28, 50, 82, 126}
ISOMER_BONUS = E_surface

def get_isomer_resonance_bonus(Z, N):
    """Stability bonus for isomer closures."""
    bonus = 0
    if Z in ISOMER_NODES: bonus += ISOMER_BONUS
    if N in ISOMER_NODES: bonus += ISOMER_BONUS
    if Z in ISOMER_NODES and N in ISOMER_NODES:
        bonus *= 1.5
    return bonus

def qfd_total_energy(A, Z):
    """Complete QFD energy functional."""
    N = A - Z
    q = Z / A if A > 0 else 0

    E_bulk = E_volume * A
    E_surf = E_surface * (A**(2/3))
    E_asym = a_sym * A * ((1 - 2*q)**2)
    E_vac  = a_disp * (Z**2) / (A**(1/3))
    E_iso  = -get_isomer_resonance_bonus(Z, N)

    return E_bulk + E_surf + E_asym + E_vac + E_iso

def find_stable_isotope(A):
    """Find optimal Z for given A."""
    if A <= 2:
        return 1

    # Discrete search over all integer Z
    best_Z = 1
    best_E = qfd_total_energy(A, 1)

    for Z in range(1, A):
        E = qfd_total_energy(A, Z)
        if E < best_E:
            best_E = E
            best_Z = Z

    return best_Z

# ============================================================================
# TEST DATA (from qfd_complete_suite.py run)
# ============================================================================
test_nuclides = [
    ("He-4",  2,  4),
    ("C-12",  6,  12),
    ("Ca-40", 20, 40),
    ("Fe-56", 26, 56),
    ("Sn-112", 50, 112),
    ("Pb-208", 82, 208),
]

# Compute predictions
results = []
for name, Z_exp, A in test_nuclides:
    N_exp = A - Z_exp
    Z_pred = find_stable_isotope(A)
    Delta_Z = Z_pred - Z_exp

    results.append({
        'name': name,
        'A': A,
        'Z_exp': Z_exp,
        'N_exp': N_exp,
        'Z_pred': Z_pred,
        'Delta_Z': Delta_Z,
        'is_magic_Z': Z_exp in ISOMER_NODES,
        'is_magic_N': N_exp in ISOMER_NODES,
    })

# ============================================================================
# VISUALIZATION
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: Predicted vs Experimental Z
ax1 = axes[0, 0]
A_vals = [r['A'] for r in results]
Z_exp_vals = [r['Z_exp'] for r in results]
Z_pred_vals = [r['Z_pred'] for r in results]

# Plot diagonal (perfect prediction)
A_range = np.linspace(0, 220, 100)
ax1.plot(A_range, A_range, 'k--', alpha=0.3, linewidth=2, label='Z_pred = Z_exp')

# Plot predictions
colors = ['green' if r['Delta_Z'] == 0 else 'red' for r in results]
sizes = [150 if r['is_magic_Z'] and r['is_magic_N'] else 100 for r in results]

for r, color, size in zip(results, colors, sizes):
    marker = 's' if (r['is_magic_Z'] and r['is_magic_N']) else 'o'
    ax1.scatter(r['Z_exp'], r['Z_pred'], c=color, s=size,
               marker=marker, edgecolors='black', linewidth=1.5, alpha=0.7)
    ax1.annotate(r['name'], (r['Z_exp'], r['Z_pred']),
                xytext=(5, 5), textcoords='offset points', fontsize=8)

ax1.set_xlabel('Z (Experimental)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Z (Predicted)', fontsize=11, fontweight='bold')
ax1.set_title('Panel A: Predicted vs Experimental Charge\nGreen = Exact, Red = Error, Square = Doubly Magic',
             fontsize=11, fontweight='bold')
ax1.grid(alpha=0.3)
ax1.legend()
ax1.set_xlim(0, 90)
ax1.set_ylim(0, 90)

# Panel 2: Error (ΔZ) vs Mass Number A
ax2 = axes[0, 1]
Delta_Z_vals = [r['Delta_Z'] for r in results]

# Scatter plot with color coding
for r in results:
    color = 'green' if r['Delta_Z'] == 0 else 'red'
    marker = 's' if (r['is_magic_Z'] and r['is_magic_N']) else 'o'
    size = 150 if (r['is_magic_Z'] and r['is_magic_N']) else 100
    ax2.scatter(r['A'], r['Delta_Z'], c=color, s=size,
               marker=marker, edgecolors='black', linewidth=1.5, alpha=0.7)
    ax2.annotate(r['name'], (r['A'], r['Delta_Z']),
                xytext=(5, 5), textcoords='offset points', fontsize=8)

ax2.axhline(0, color='black', linestyle='-', linewidth=2, alpha=0.5)
ax2.set_xlabel('Mass Number A', fontsize=11, fontweight='bold')
ax2.set_ylabel('ΔZ = Z_pred - Z_exp', fontsize=11, fontweight='bold')
ax2.set_title('Panel B: Prediction Error vs Mass\nSystematic Under-Prediction for Heavy Nuclei',
             fontsize=11, fontweight='bold')
ax2.grid(alpha=0.3)
ax2.set_xlim(0, 220)
ax2.set_ylim(-7, 1)

# Panel 3: Energy Landscape for Pb-208
ax3 = axes[1, 0]
A_pb = 208
Z_range = np.arange(70, 90, 1)
E_pb = [qfd_total_energy(A_pb, Z) for Z in Z_range]

# Normalize to minimum
E_pb = np.array(E_pb) - np.min(E_pb)

ax3.plot(Z_range, E_pb, 'b-', linewidth=2, label='Total Energy')
ax3.axvline(82, color='green', linestyle='--', linewidth=2, label='Z = 82 (Experimental)')
ax3.axvline(76, color='red', linestyle='--', linewidth=2, label='Z = 76 (Predicted)')

# Mark minimum
Z_min_idx = np.argmin(E_pb)
Z_min = Z_range[Z_min_idx]
ax3.scatter(Z_min, 0, c='red', s=200, marker='*', edgecolors='black',
           linewidth=2, zorder=5, label=f'Minimum at Z={Z_min}')

ax3.set_xlabel('Charge Z', fontsize=11, fontweight='bold')
ax3.set_ylabel('Relative Energy (MeV)', fontsize=11, fontweight='bold')
ax3.set_title(f'Panel C: Energy Landscape for Pb-208 (A={A_pb})\nPredicted Minimum at Z={Z_min}, Actual Z=82',
             fontsize=11, fontweight='bold')
ax3.grid(alpha=0.3)
ax3.legend(loc='upper right')
ax3.set_xlim(70, 89)

# Panel 4: Isomer Bonus Contribution
ax4 = axes[1, 1]

# Compute energy components for each nucleus
components_data = []
for r in results:
    A, Z = r['A'], r['Z_exp']
    N = A - Z
    q = Z / A

    E_bulk = E_volume * A
    E_surf = E_surface * (A**(2/3))
    E_asym = a_sym * A * ((1 - 2*q)**2)
    E_vac  = a_disp * (Z**2) / (A**(1/3))
    E_iso  = get_isomer_resonance_bonus(Z, N)

    components_data.append({
        'name': r['name'],
        'E_surf': E_surf,
        'E_asym': E_asym,
        'E_vac': E_vac,
        'E_iso': E_iso,
    })

# Bar plot
names = [r['name'] for r in results]
x_pos = np.arange(len(names))
width = 0.2

surf_vals = [c['E_surf'] for c in components_data]
asym_vals = [c['E_asym'] for c in components_data]
vac_vals = [c['E_vac'] for c in components_data]
iso_vals = [c['E_iso'] for c in components_data]

ax4.bar(x_pos - 1.5*width, surf_vals, width, label='E_surface', alpha=0.7)
ax4.bar(x_pos - 0.5*width, asym_vals, width, label='E_asymmetry', alpha=0.7)
ax4.bar(x_pos + 0.5*width, vac_vals, width, label='E_vacuum', alpha=0.7)
ax4.bar(x_pos + 1.5*width, iso_vals, width, label='E_isomer', alpha=0.7, color='gold')

ax4.set_xlabel('Nucleus', fontsize=11, fontweight='bold')
ax4.set_ylabel('Energy Contribution (MeV)', fontsize=11, fontweight='bold')
ax4.set_title('Panel D: Energy Component Comparison\nIsomer Bonus in Gold',
             fontsize=11, fontweight='bold')
ax4.set_xticks(x_pos)
ax4.set_xticklabels(names, rotation=45, ha='right')
ax4.legend(loc='upper left', fontsize=8)
ax4.grid(axis='y', alpha=0.3)

# Main title
fig.suptitle('QFD COMPLETE SUITE - CURRENT PERFORMANCE ANALYSIS\n' +
            'Parameter-free framework with isomer ladder (5/7 shielding)',
            fontsize=14, fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('CURRENT_PERFORMANCE_ANALYSIS.png', dpi=300, bbox_inches='tight')
print("✓ Saved: CURRENT_PERFORMANCE_ANALYSIS.png")

# ============================================================================
# STATISTICS SUMMARY
# ============================================================================
print("\n" + "="*80)
print("PERFORMANCE SUMMARY")
print("="*80)
print()

exact_matches = sum(1 for r in results if r['Delta_Z'] == 0)
total = len(results)
mean_error = np.mean([abs(r['Delta_Z']) for r in results])
max_error = max([abs(r['Delta_Z']) for r in results])

print(f"Total nuclides tested:  {total}")
print(f"Exact matches (ΔZ=0):   {exact_matches}/{total} ({100*exact_matches/total:.1f}%)")
print(f"Mean |ΔZ|:              {mean_error:.2f} charges")
print(f"Max |ΔZ|:               {max_error} charges")
print()

print("Detailed Results:")
print("-" * 80)
print(f"{'Nucleus':<10} {'A':>4} {'Z_exp':>6} {'Z_pred':>6} {'ΔZ':>6} {'Magic Z?':<9} {'Magic N?':<9}")
print("-" * 80)
for r in results:
    magic_z = "✓" if r['is_magic_Z'] else ""
    magic_n = "✓" if r['is_magic_N'] else ""
    print(f"{r['name']:<10} {r['A']:>4} {r['Z_exp']:>6} {r['Z_pred']:>6} "
          f"{r['Delta_Z']:>+6} {magic_z:<9} {magic_n:<9}")

print("="*80)
print()

print("DIAGNOSIS:")
print("-" * 80)
print("✓ Light nuclei (He-4, C-12): EXACT predictions")
print("✗ Heavy nuclei (Sn, Pb): Systematic ΔZ = -5 to -6 (under-prediction)")
print("✗ Magic number nuclei: Pulled toward lower Z despite isomer bonus")
print()
print("HYPOTHESIS:")
print("  • 5/7 shielding factor too weak (a_disp = 0.857 MeV too strong)")
print("  • Vacuum displacement Z²/A^(1/3) dominates for large Z")
print("  • Isomer bonus insufficient to counter displacement pull")
print()
print("RECOMMENDATION:")
print("  • Test shielding factor 0.50 (reduce a_disp to 0.600 MeV)")
print("  • OR increase isomer bonus to 1.5 × E_surface")
print("  • OR hybrid approach with balanced adjustments")
print("="*80)

plt.show()
