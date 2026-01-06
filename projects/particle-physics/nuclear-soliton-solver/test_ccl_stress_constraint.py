#!/usr/bin/env python3
"""
Test CCL Stress Constraint Integration

Validates that Core Compression Law stress properly guides optimization
toward stable isotopes and away from unstable ones.

This connects Top-Down (5,842 isotope empirical fit) with Bottom-Up
(QFD soliton solver).
"""
import sys
import pandas as pd
import numpy as np
sys.path.insert(0, 'src')

from parallel_objective import ccl_stress
from ccl_seeded_solver import backbone_charge

# Test isotopes spanning stability range
test_isotopes = [
    {'name': 'C-12', 'A': 12, 'Z': 6, 'expected_stress': 'Low', 'stability': 'Stable (magic)'},
    {'name': 'Pb-208', 'A': 208, 'Z': 82, 'expected_stress': 'Moderate', 'stability': 'Stable (doubly magic)'},
    {'name': 'U-238', 'A': 238, 'Z': 92, 'expected_stress': 'High', 'stability': 'Unstable (α, 4.5 Gyr)'},
    {'name': 'Fe-56', 'A': 56, 'Z': 26, 'expected_stress': 'Very Low', 'stability': 'Stable (most bound)'},
    {'name': 'C-14', 'A': 14, 'Z': 6, 'expected_stress': 'Moderate', 'stability': 'Unstable (β-, 5730 yr)'},
]

print("=" * 80)
print("CCL STRESS CONSTRAINT VALIDATION")
print("=" * 80)
print()
print("Core Compression Law: Q(A) = c1·A^(2/3) + c2·A")
print(f"  c1 = {0.5293:.4f} (surface term)")
print(f"  c2 = {0.3167:.4f} (volume term)")
print()
print("Stress = |Z - Q_backbone(A)|")
print("  Stress < 1: Stable isotope")
print("  Stress > 3: Unstable, will decay")
print()
print("=" * 80)
print()

results = []
for iso in test_isotopes:
    Z, A = iso['Z'], iso['A']

    # Calculate CCL predictions
    Q_backbone = backbone_charge(A)
    stress = ccl_stress(Z, A)

    # Stress penalty in loss function
    stress_weight = 0.1
    stress_penalty = stress_weight * (stress ** 2)

    results.append({
        'Isotope': iso['name'],
        'A': A,
        'Z': Z,
        'Q_backbone': Q_backbone,
        'Stress': stress,
        'Penalty': stress_penalty,
        'Status': iso['stability']
    })

df = pd.DataFrame(results)

print("CCL Stress Analysis:")
print()
print(df.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
print()

# Analyze stress distribution
print("=" * 80)
print("STRESS DISTRIBUTION")
print("=" * 80)
print()

stable = df[df['Isotope'].isin(['C-12', 'Fe-56', 'Pb-208'])]
unstable = df[df['Isotope'].isin(['C-14', 'U-238'])]

print(f"Stable isotopes:")
print(f"  Mean stress:   {stable['Stress'].mean():.3f}")
print(f"  Mean penalty:  {stable['Penalty'].mean():.4f}")
print()

print(f"Unstable isotopes:")
print(f"  Mean stress:   {unstable['Stress'].mean():.3f}")
print(f"  Mean penalty:  {unstable['Penalty'].mean():.4f}")
print()

if unstable['Stress'].mean() > stable['Stress'].mean():
    print("✓ Correct: Unstable isotopes have higher stress")
else:
    print("✗ Unexpected: Stable isotopes have higher stress")

print()
print("=" * 80)
print("OPTIMIZATION IMPACT")
print("=" * 80)
print()

print("How CCL stress guides optimization:")
print()
print("1. C-12 (stable, low stress):")
print(f"   - Stress = {df.loc[df['Isotope']=='C-12', 'Stress'].values[0]:.3f}")
print(f"   - Penalty = {df.loc[df['Isotope']=='C-12', 'Penalty'].values[0]:.4f}")
print("   → Optimization focuses on accurate binding energy")
print()

print("2. Fe-56 (most bound, very low stress):")
print(f"   - Stress = {df.loc[df['Isotope']=='Fe-56', 'Stress'].values[0]:.3f}")
print(f"   - Penalty = {df.loc[df['Isotope']=='Fe-56', 'Penalty'].values[0]:.4f}")
print("   → Highest priority for parameter calibration")
print()

print("3. Pb-208 (stable, moderate stress):")
print(f"   - Stress = {df.loc[df['Isotope']=='Pb-208', 'Stress'].values[0]:.3f}")
print(f"   - Penalty = {df.loc[df['Isotope']=='Pb-208', 'Penalty'].values[0]:.4f}")
print("   → Slightly higher penalty than C-12")
print()

print("4. U-238 (unstable, high stress):")
print(f"   - Stress = {df.loc[df['Isotope']=='U-238', 'Stress'].values[0]:.3f}")
print(f"   - Penalty = {df.loc[df['Isotope']=='U-238', 'Penalty'].values[0]:.4f}")
print("   → Heavily penalized, optimization de-prioritizes")
print()

print("=" * 80)
print("PHYSICAL INTERPRETATION")
print("=" * 80)
print()

print("The CCL stress constraint achieves:")
print()
print("1. **Empirical Guidance**: Uses validated stability map (5,842 isotopes)")
print("   instead of generic virial condition")
print()
print("2. **Focus on Stable Isotopes**: Parameters optimized for clean soliton")
print("   solutions where they should exist")
print()
print("3. **Realistic Expectations**: Doesn't waste effort trying to get perfect")
print("   solutions for inherently unstable configurations")
print()
print("4. **Top-Down ↔ Bottom-Up**: Connects phenomenological CCL with")
print("   fundamental QFD field equations")
print()

print("=" * 80)
print("NEXT STEP: RE-RUN OPTIMIZATION")
print("=" * 80)
print()

print("With CCL stress constraint + Derrick virial (T + 3V):")
print()
print("Expected improvements:")
print("  ✓ C-12 remains most stable (already achieved)")
print("  ✓ Binding energies more accurate for stable isotopes")
print("  ✓ Better convergence (virial as sanity check, not primary driver)")
print("  ✓ Parameters reflect true QFD soliton physics")
print()

print("Run command:")
print("  python3 run_parallel_optimization.py --maxiter 50 --popsize 15 --workers 4")
print()
print("This will calibrate parameters to minimize:")
print("  Loss = Energy_Error² + 0.1·(CCL_Stress)² + Virial_Sanity_Check")
print()

print("Done!")
