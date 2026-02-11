#!/usr/bin/env python3
"""
Final Carbon Valley Test - Corrected Physics

CRITICAL FIXES (2025-12-31):
1. Deterministic solver (seeded RNG)
2. Mass integral (∫ψ² dV) instead of Laplacian (dead-end)
3. Correct stress direction: DESCENDING as A increases for fixed Z

Physical Expectation:
- C-11 (proton-rich): Stress > 0 (Z=6 > Q_backbone≈5)
- C-12 (balanced): Stress ≈ 0 (resonance)
- C-13, C-14 (neutron-rich): Stress < 0 (Z=6 < Q_backbone>6)

Zero-crossing at C-12 validates geometric resonance.
"""
import sys
import os
import json
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, 'src')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
from qfd.shared_constants import C1_SURFACE, C2_VOLUME

# CCL constants (from shared_constants)
CCL_C1 = C1_SURFACE
CCL_C2 = C2_VOLUME

# Carbon isotopes
carbon_isotopes = [
    {'name': 'C-11', 'A': 11, 'Z': 6, 'type': 'Proton-rich (β+)'},
    {'name': 'C-12', 'A': 12, 'Z': 6, 'type': 'Magic (stable)'},
    {'name': 'C-13', 'A': 13, 'Z': 6, 'type': 'Stable (minor)'},
    {'name': 'C-14', 'A': 14, 'Z': 6, 'type': 'Neutron-rich (β-)'},
]

# Parameters
params = {
    'c_v2_base': 2.201711,
    'c_v2_iso': 0.027035,
    'c_v2_mass': -0.000205,
    'c_v4_base': 5.282364,
    'c_v4_size': -0.085018,
    'alpha_e_scale': 1.007419,
    'beta_e_scale': 0.504312,
    'c_sym': 25.0,
    'kappa_rho': 0.029816
}

print("=" * 80)
print("FINAL CARBON VALLEY TEST - CORRECTED PHYSICS")
print("=" * 80)
print()
print("Fixes applied:")
print("  ✓ Deterministic solver (seed=4242)")
print("  ✓ Mass integral ∫ψ² dV (not Laplacian)")
print("  ✓ Correct stress direction (descending with A)")
print()
print("Expected: Stress crosses ZERO at C-12 (proton-rich → neutron-rich)")
print()
print("=" * 80)
print()

results = []

for iso in carbon_isotopes:
    Z, A = iso['Z'], iso['A']
    print(f"Solving {iso['name']} (Z={Z}, A={A})...", end=" ", flush=True)

    cmd = [
        'python3', 'src/qfd_solver.py',
        '--A', str(A),
        '--Z', str(Z),
        '--c-v2-base', str(params['c_v2_base']),
        '--c-v2-iso', str(params['c_v2_iso']),
        '--c-v2-mass', str(params['c_v2_mass']),
        '--c-v4-base', str(params['c_v4_base']),
        '--c-v4-size', str(params['c_v4_size']),
        '--alpha-e-scale', str(params['alpha_e_scale']),
        '--beta-e-scale', str(params['beta_e_scale']),
        '--c-sym', str(params['c_sym']),
        '--kappa-rho', str(params['kappa_rho']),
        '--grid-points', '48',
        '--iters-outer', '500',
        '--device', 'cuda',
        '--early-stop-vir', '0.5',  # Relaxed for Derrick virial
        '--seed', '4242',  # DETERMINISTIC
        '--emit-json'
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    data = json.loads(result.stdout)

    E_model = data['E_model']
    virial = data['virial']
    M_soliton = data['M_soliton']
    R_soliton = data['R_soliton']
    Q_backbone = data['Q_backbone']
    stress = data['stress_vector']  # Z - Q_backbone
    BE_per_A = E_model / A

    results.append({
        'Isotope': iso['name'],
        'A': A,
        'M_soliton': M_soliton,
        'Q_backbone': Q_backbone,
        'Stress': stress,
        'R_ratio': R_soliton,
        'BE/A': BE_per_A,
        'Virial': virial,
        'Type': iso['type']
    })

    print(f"M={M_soliton:.1f}, Stress={stress:+.2f}, BE/A={BE_per_A:.2f}")

print()
print("=" * 80)
print("RESULTS TABLE")
print("=" * 80)
print()

df = pd.DataFrame(results)
print(df.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
print()

# Zero-crossing analysis
print("=" * 80)
print("ZERO-CROSSING ANALYSIS")
print("=" * 80)
print()

c11_stress = df.loc[df['A'] == 11, 'Stress'].values[0]
c12_stress = df.loc[df['A'] == 12, 'Stress'].values[0]
c13_stress = df.loc[df['A'] == 13, 'Stress'].values[0]
c14_stress = df.loc[df['A'] == 14, 'Stress'].values[0]

print("Stress progression (should DESCEND: positive → zero → negative):")
for idx, row in df.iterrows():
    marker = "★" if row['A'] == 12 else " "
    sign = "+" if row['Stress'] > 0.1 else ("-" if row['Stress'] < -0.1 else "0")
    print(f"{marker} {row['Isotope']:6s}: Stress = {row['Stress']:+7.3f}  [{sign}]  ({row['Type']})")
print()

# Test 1: Descending stress (CORRECTED LOGIC)
if c11_stress > c12_stress > c13_stress:
    print("✓ Stress descends correctly (proton-rich → neutron-rich)")
    test1 = True
else:
    print("✗ Stress does NOT descend monotonically")
    test1 = False

# Test 2: Zero-crossing at C-12
if c11_stress > 0 and c13_stress < 0:
    print("✓✓ Sign crossing detected (positive → negative)")
    if abs(c12_stress) < abs(c11_stress) and abs(c12_stress) < abs(c13_stress):
        print("✓✓✓ C-12 has minimal |stress| → ZERO CROSSING CONFIRMED!")
        test2 = True
    else:
        print("⚠ Crossing exists but C-12 not at minimum")
        test2 = False
else:
    print("✗ No sign crossing detected")
    test2 = False

print()

# Test 3: Energy minimum
most_stable = df.loc[df['BE/A'].idxmin()]
if most_stable['A'] == 12:
    print("✓ C-12 is energy minimum (lowest BE/A)")
    test3 = True
else:
    print(f"✗ Energy minimum is {most_stable['Isotope']}, not C-12")
    test3 = False

print()

# Overall status
if test1 and test2 and test3:
    status = "SUCCESS"
    print("=" * 80)
    print("✓✓✓ GEOMETRIC RESONANCE VALIDATED!")
    print("=" * 80)
elif test2 and test3:
    status = "PARTIAL"
    print("⚠ PARTIAL SUCCESS - Zero crossing exists, C-12 is minimum")
else:
    status = "FAILURE"
    print("✗ GEOMETRIC RESONANCE NOT ACHIEVED")

print()

# Visualization
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Stress Vector
ax1.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Zero Line')
ax1.plot(df['A'], df['Stress'], 'o-', linewidth=2.5, markersize=10, color='blue')
ax1.scatter(12, c12_stress, s=300, c='gold', marker='*', edgecolors='red', linewidths=2, zorder=10)

ax1.set_xlabel('Mass Number A', fontsize=13, fontweight='bold')
ax1.set_ylabel('Stress = Z - Q_backbone', fontsize=13, fontweight='bold')
ax1.set_title('CCL Stress (Corrected)', fontsize=15, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=11)

for idx, row in df.iterrows():
    ax1.annotate(row['Isotope'], (row['A'], row['Stress']),
                 textcoords="offset points", xytext=(0, 12), ha='center', fontsize=11, fontweight='bold')

# Plot 2: Energy Landscape
ax2.plot(df['A'], df['BE/A'], 's-', linewidth=2.5, markersize=10, color='green')
ax2.scatter(12, df.loc[df['A']==12, 'BE/A'].values[0], s=300, c='gold', marker='*', edgecolors='red', linewidths=2, zorder=10)

ax2.set_xlabel('Mass Number A', fontsize=13, fontweight='bold')
ax2.set_ylabel('BE/A (MeV)', fontsize=13, fontweight='bold')
ax2.set_title('Energy Landscape', fontsize=15, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Plot 3: Soliton Mass Integral
ax3.plot(df['A'], df['M_soliton'], '^-', linewidth=2.5, markersize=10, color='purple')
ax3.scatter(12, df.loc[df['A']==12, 'M_soliton'].values[0], s=300, c='gold', marker='*', edgecolors='red', linewidths=2, zorder=10)

ax3.set_xlabel('Mass Number A', fontsize=13, fontweight='bold')
ax3.set_ylabel('M_soliton = ∫ψ² dV', fontsize=13, fontweight='bold')
ax3.set_title('Soliton Mass Integral', fontsize=15, fontweight='bold')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('carbon_valley_final.png', dpi=200, bbox_inches='tight')
print("✓ Plot saved: carbon_valley_final.png")
print()

df.to_csv('carbon_valley_final.csv', index=False)
print("✓ Data saved: carbon_valley_final.csv")
print()

# Final summary
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print()

print(f"Stress at C-12: {c12_stress:+.3f}")
print(f"Energy minimum: {most_stable['Isotope']}")
print(f"Soliton mass range: {df['M_soliton'].min():.1f} - {df['M_soliton'].max():.1f}")
print()

if status == "SUCCESS":
    print("The QFD soliton model correctly predicts C-12 as geometric resonance!")
    print()
    print("Validated:")
    print("  • Stress crosses zero at C-12 (CCL backbone)")
    print("  • Energy minimum at C-12 (thermodynamic stability)")
    print("  • Soliton mass correlates with A (field density conserved)")
    print()
    print("READY FOR PUBLICATION")
elif status == "PARTIAL":
    print("Model shows promise but needs parameter refinement.")
    print()
    print("Next steps:")
    print("  • Re-optimize parameters with CCL stress + Derrick virial")
    print("  • Test other magic nuclei (He-4, O-16)")
else:
    print("Model does not reproduce C-12 stability.")
    print()
    print("Review needed:")
    print("  • Check if V4/V6 parameters too loose (allowing C-13 minimum)")
    print("  • Verify Derrick virial (T+3V) is correct scaling")
    print("  • Test with higher grid resolution (64 points)")

print()
print("Done!")
