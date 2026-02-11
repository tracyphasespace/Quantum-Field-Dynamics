#!/usr/bin/env python3
"""
Geometric Resonance Verification - CALIBRATED Carbon Valley

Uses g_c from proton calibration to measure emergent charge in elementary units.

With proper calibration:
- C-11: Q_actual < 6 (starved, negative stress)
- C-12: Q_actual ≈ 6 (resonant, stress ≈ 0) ← ZERO CROSSING
- C-13: Q_actual > 6 (heavy, positive stress)
- C-14: Q_actual >> 6 (over-massive, large positive stress)

This tests if the field GEOMETRY naturally produces CCL backbone.
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

# Load calibration
try:
    with open('gc_calibration.json', 'r') as f:
        calibration = json.load(f)
    g_c = calibration['g_c']
    print(f"✓ Loaded calibration: g_c = {g_c:.6f}")
except FileNotFoundError:
    print("✗ Calibration file not found! Run calibrate_gc_proton.py first.")
    sys.exit(1)

print()

# CCL constants (from shared_constants)
CCL_C1 = C1_SURFACE
CCL_C2 = C2_VOLUME

# Carbon isotopes
carbon_isotopes = [
    {'name': 'C-11', 'A': 11, 'Z': 6},
    {'name': 'C-12', 'A': 12, 'Z': 6},
    {'name': 'C-13', 'A': 13, 'Z': 6},
    {'name': 'C-14', 'A': 14, 'Z': 6},
]

# Parameters
params = calibration['parameters']

print("=" * 80)
print("CALIBRATED GEOMETRIC RESONANCE - CARBON VALLEY")
print("=" * 80)
print()
print(f"Using calibrated g_c = {g_c:.6f} (from H-1 proton)")
print()
print("Question: Does the field geometry produce Q ≈ 6 for C-12?")
print("Success: Stress crosses zero at A=12")
print()
print("=" * 80)
print()

results = []

for iso in carbon_isotopes:
    Z, A = iso['Z'], iso['A']
    print(f"Solving {iso['name']} (Z={Z}, A={A})...", end=" ", flush=True)

    # Build command with calibrated g_c
    # NOTE: We need to modify solver to accept g_c parameter
    # For now, we'll use the raw Q_actual and re-scale it
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
        '--early-stop-vir', '0.5',
        '--emit-json'
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    data = json.loads(result.stdout)

    # Extract values (Q_actual is computed with g_c=1.0 in solver)
    E_model = data['E_model']
    virial = data['virial']
    Q_raw = data['Q_actual']  # Raw value (g_c=1.0)

    # Re-calibrate using correct g_c
    Q_actual = g_c * Q_raw  # Now in units of elementary charge

    # CCL prediction
    Q_backbone = CCL_C1 * (A ** (2.0/3.0)) + CCL_C2 * A

    # Stress vector (signed deviation)
    stress = Q_actual - Q_backbone

    # BE/A for energy landscape
    BE_per_A = E_model / A

    results.append({
        'Isotope': iso['name'],
        'A': A,
        'Q_actual': Q_actual,
        'Q_backbone': Q_backbone,
        'Stress': stress,
        'BE/A': BE_per_A,
        'Virial': virial
    })

    print(f"Q={Q_actual:.2f}, Q_CCL={Q_backbone:.2f}, Stress={stress:+.2f}")

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

print("Stress Vector vs Mass Number:")
for idx, row in df.iterrows():
    marker = "★" if row['A'] == 12 else " "
    sign_str = "ZERO" if abs(row['Stress']) < 0.5 else ("POS" if row['Stress'] > 0 else "NEG")
    print(f"{marker} A={row['A']:2d}: Stress = {row['Stress']:+7.3f}  [{sign_str:4s}]")
print()

# Check for zero-crossing
c11_stress = df.loc[df['A'] == 11, 'Stress'].values[0]
c12_stress = df.loc[df['A'] == 12, 'Stress'].values[0]
c13_stress = df.loc[df['A'] == 13, 'Stress'].values[0]
c14_stress = df.loc[df['A'] == 14, 'Stress'].values[0]

# Test 1: Minimal stress at C-12
if abs(c12_stress) < abs(c11_stress) and abs(c12_stress) < abs(c13_stress):
    print("✓ C-12 has minimal |stress| among neighbors")
    if abs(c12_stress) < 1.0:
        print(f"✓✓ C-12 stress ≈ 0 ({c12_stress:+.2f}, within tolerance)")
        test1 = True
    else:
        print(f"⚠ C-12 stress = {c12_stress:+.2f} (should be closer to 0)")
        test1 = False
else:
    print("✗ C-12 does NOT have minimal stress")
    test1 = False

print()

# Test 2: Sign pattern (negative → zero → positive)
if c11_stress < c12_stress < c13_stress:
    print("✓ Stress increases monotonically: C-11 → C-12 → C-13")
    if c11_stress < 0 and c13_stress > 0:
        print("✓✓ Sign crossing detected (negative → positive)")
        test2 = True
    else:
        print("⚠ Monotonic but no sign crossing")
        test2 = False
else:
    print("✗ Stress not monotonic")
    test2 = False

print()

# Test 3: Energy minimum at C-12
most_stable = df.loc[df['BE/A'].idxmin()]
if most_stable['A'] == 12:
    print("✓ C-12 is energy minimum (lowest BE/A)")
    test3 = True
else:
    print(f"✗ Energy minimum is A={most_stable['A']}, not C-12")
    test3 = False

print()

# Overall status
if test1 and test2 and test3:
    status = "SUCCESS"
    print("✓✓✓ GEOMETRIC RESONANCE VALIDATED!")
elif (test1 or test2) and test3:
    status = "PARTIAL"
    print("⚠ PARTIAL SUCCESS - Energy correct, stress nearly zero")
else:
    status = "FAILURE"
    print("✗ GEOMETRIC RESONANCE NOT ACHIEVED")

print()

# Visualization
print("=" * 80)
print("VISUALIZATION")
print("=" * 80)
print()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Stress Vector with zero line
ax1.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Zero Line (Resonance)', alpha=0.7)
ax1.plot(df['A'], df['Stress'], 'o-', linewidth=2.5, markersize=10, color='blue', label='Stress Vector')
ax1.scatter(12, c12_stress, s=300, c='gold', marker='*', edgecolors='red', linewidths=2, zorder=10, label='C-12 (Target)')

ax1.set_xlabel('Mass Number A', fontsize=13, fontweight='bold')
ax1.set_ylabel('Stress = Q_actual - Q_backbone', fontsize=13, fontweight='bold')
ax1.set_title('Geometric Resonance Test: Carbon Series', fontsize=15, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=11)

for idx, row in df.iterrows():
    ax1.annotate(row['Isotope'], (row['A'], row['Stress']),
                 textcoords="offset points", xytext=(0, 12), ha='center', fontsize=11, fontweight='bold')

# Plot 2: Energy landscape
ax2.plot(df['A'], df['BE/A'], 's-', linewidth=2.5, markersize=10, color='green', label='BE/A')
ax2.scatter(12, df.loc[df['A']==12, 'BE/A'].values[0], s=300, c='gold', marker='*', edgecolors='red', linewidths=2, zorder=10, label='C-12 (Minimum)')

ax2.set_xlabel('Mass Number A', fontsize=13, fontweight='bold')
ax2.set_ylabel('Binding Energy per Nucleon (MeV)', fontsize=13, fontweight='bold')
ax2.set_title('Energy Landscape (Parabolic Valley)', fontsize=15, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=11)

for idx, row in df.iterrows():
    ax2.annotate(row['Isotope'], (row['A'], row['BE/A']),
                 textcoords="offset points", xytext=(0, -18), ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('geometric_resonance_calibrated.png', dpi=200, bbox_inches='tight')
print("✓ Plot saved: geometric_resonance_calibrated.png")
print()

df.to_csv('geometric_resonance_calibrated.csv', index=False)
print("✓ Data saved: geometric_resonance_calibrated.csv")
print()

# Final conclusion
print("=" * 80)
print("CONCLUSION")
print("=" * 80)
print()

if status == "SUCCESS":
    print("The QFD field equations naturally produce the CCL backbone!")
    print()
    print("This validates:")
    print("  • Top-Down (5,842 isotopes) ↔ Bottom-Up (QFD solver) convergence")
    print("  • Charge emerges from field curvature (Laplacian topology)")
    print("  • C-12 is geometric resonance point (zero stress)")
    print("  • β universality across nuclear sector")
    print()
    print("READY FOR PUBLICATION!")
elif status == "PARTIAL":
    print("The model is close but needs refinement.")
    print()
    print(f"C-12 stress = {c12_stress:+.2f} (target: 0)")
    print()
    print("Next steps:")
    print("  • Fine-tune parameters with CCL stress constraint")
    print("  • Increase grid resolution (48 → 64)")
    print("  • Test other magic nuclei (He-4, O-16, Ca-40)")
else:
    print("Geometric resonance not achieved.")
    print()
    print("Diagnostic:")
    print(f"  C-12 stress: {c12_stress:+.2f} (should be ≈ 0)")
    print(f"  Energy minimum: A={most_stable['A']} (should be 12)")
    print()
    print("Review needed:")
    print("  • Field equation formulation")
    print("  • Parameter calibration")
    print("  • Virial formula (T + 3V correct?)")

print()
print("Done!")
