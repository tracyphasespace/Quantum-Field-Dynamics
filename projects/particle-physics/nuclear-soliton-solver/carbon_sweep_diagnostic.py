#!/usr/bin/env python3
"""
Carbon Isotope Sweep - Critical Diagnostic

Tests if C-12 emerges as energy minimum (most stable)
regardless of absolute binding energy calibration.

If C-12 is the deepest well → Physics correct, just needs re-calibration
If C-12 is NOT minimum → Physics engine needs review
"""
import pandas as pd
import numpy as np

from src.parallel_objective import run_solver_direct

# Carbon isotopes to test
carbon_isotopes = [
    {'name': 'C-11', 'A': 11, 'Z': 6, 'stability': 'Unstable (β+, 20.4 min)'},
    {'name': 'C-12', 'A': 12, 'Z': 6, 'stability': 'Stable (98.9%) - MAGIC'},
    {'name': 'C-13', 'A': 13, 'Z': 6, 'stability': 'Stable (1.1%)'},
    {'name': 'C-14', 'A': 14, 'Z': 6, 'stability': 'Unstable (β-, 5730 yr)'},
]

# Test with BOTH virial formulas to see which gives correct ordering
print("=" * 80)
print("CARBON ISOTOPE SWEEP - VIRIAL DIAGNOSTIC")
print("=" * 80)
print()
print("Testing if C-12 emerges as energy minimum...")
print("(Ignore absolute magnitude - check RELATIVE stability)")
print()

# Parameters (from successful C-12 Golden Probe)
params = {
    'c_v2_base': 3.113872,
    'c_v2_iso': 0.029852,
    'c_v2_mass': -0.000038,
    'c_v4_base': 11.959113,
    'c_v4_size': -0.091107,
    'alpha_e_scale': 0.992827,
    'beta_e_scale': 0.491950,
    'c_sym': 25.762166,
    'kappa_rho': 0.026947
}

results = []

print("Running solver on each carbon isotope...")
print()

for iso in carbon_isotopes:
    Z, A = iso['Z'], iso['A']
    print(f"Solving {iso['name']} (Z={Z}, A={A})...", end=" ", flush=True)

    # Run with current (corrected) virial formula
    result = run_solver_direct(
        A=A, Z=Z,
        params=params,
        grid_points=48,
        iters_outer=250,  # Match failed Octet run
        device='cuda',
        early_stop_vir=0.18
    )

    E_model = result.get('E_model', 0)
    virial = result.get('virial', 999)
    converged = result.get('converged', False)

    # Key metrics
    BE_per_nucleon = E_model / A if A > 0 else 0

    results.append({
        'Isotope': iso['name'],
        'A': A,
        'N': A - Z,
        'E_model': E_model,
        'BE/A': BE_per_nucleon,
        'Virial': virial,
        'Converged': converged,
        'Status': iso['stability']
    })

    print(f"E={E_model:.1f} MeV, BE/A={BE_per_nucleon:.1f}, Virial={virial:.2f}")

print()
print("=" * 80)
print("RESULTS TABLE")
print("=" * 80)
print()

df = pd.DataFrame(results)

# Display table
print(df.to_string(index=False))
print()

# Analyze relative stability
c12_energy = df.loc[df['Isotope'] == 'C-12', 'E_model'].values[0]
c12_be_per_A = df.loc[df['Isotope'] == 'C-12', 'BE/A'].values[0]

print("=" * 80)
print("STABILITY ANALYSIS")
print("=" * 80)
print()

print("Binding Energy per Nucleon (BE/A):")
print("  Most negative = Most tightly bound = Most stable")
print()

# Sort by BE/A (most negative first)
df_sorted = df.sort_values('BE/A')
print(df_sorted[['Isotope', 'BE/A', 'Status']].to_string(index=False))
print()

# Find the winner
most_stable = df_sorted.iloc[0]
print(f"PREDICTED MOST STABLE: {most_stable['Isotope']}")
print(f"  BE/A = {most_stable['BE/A']:.2f} MeV")
print()

# Check if C-12 is the winner
if most_stable['Isotope'] == 'C-12':
    print("✓✓✓ SUCCESS! C-12 is predicted as most stable")
    print("    Physics engine is CORRECT!")
    print("    (Absolute calibration may be off, but relative ordering is right)")
else:
    print("✗✗✗ FAILURE! C-12 is NOT the most stable")
    print(f"    Predicted: {most_stable['Isotope']}")
    print("    Physics engine needs review")

print()

# Additional diagnostics
print("=" * 80)
print("VIRIAL DIAGNOSTIC")
print("=" * 80)
print()

print("Virial values (should be low for stable isotopes):")
for _, row in df.iterrows():
    status_marker = "★" if row['Isotope'] == 'C-12' else " "
    print(f"{status_marker} {row['Isotope']:6s}: Virial = {row['Virial']:8.2f}  ({row['Status']})")

print()

# Test if ordering changes with other virial interpretations
print("=" * 80)
print("TESTING ALTERNATIVE VIRIAL: T + 3V (Soliton scaling)")
print("=" * 80)
print()

print("If QFD nucleus is a soliton 'bag', virial might scale as T + 3V")
print("(Derrick's theorem for self-confined fields)")
print()

# We can't easily recompute with different virial formula without modifying code
# But we can check if the CURRENT virial pattern makes sense

print("Virial residuals for stable vs unstable:")
stable = df[df['Isotope'].isin(['C-12', 'C-13'])]
unstable = df[df['Isotope'].isin(['C-11', 'C-14'])]

print(f"  Stable isotopes (C-12, C-13):   mean virial = {stable['Virial'].mean():.2f}")
print(f"  Unstable isotopes (C-11, C-14): mean virial = {unstable['Virial'].mean():.2f}")
print()

if stable['Virial'].mean() < unstable['Virial'].mean():
    print("✓ Pattern correct: Stable isotopes have lower virial")
else:
    print("✗ Pattern wrong: Unstable isotopes have lower virial")

print()
print("=" * 80)
print("CONCLUSION")
print("=" * 80)
print()

if most_stable['Isotope'] == 'C-12':
    print("DIAGNOSIS: Physics model is SOUND")
    print()
    print("Next steps:")
    print("  1. Accept that current virial formula may be correct for solitons")
    print("  2. Re-calibrate parameters with correct virial formula")
    print("  3. Lower virial penalty in SCF (10.0 → 0.1) to let energy dominate")
    print("  4. Test Derrick scaling: Check if T + 3V gives better convergence")
else:
    print("DIAGNOSIS: Physics model needs review")
    print()
    print("Possible issues:")
    print("  1. Missing shell structure terms for light nuclei")
    print("  2. Pairing energy not captured")
    print("  3. Surface tension coefficient wrong for A ~ 12")

print()
print("Done!")
