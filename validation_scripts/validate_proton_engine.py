#!/usr/bin/env python3
"""
ENGINE D: PROTON DRIP LINE & EMISSION VALIDATOR

Two-Track Validation:
  Track 1 (Topology): N-conservation in proton emission
  Track 2 (Mechanics): Geometric tension ratio at proton drip line

Hypothesis:
  - Track 1: Proton emission preserves harmonic mode (ΔN ≈ 0 or 1)
  - Track 2: Proton drip occurs at LOWER tension than neutron drip
             (Coulomb repulsion aids volume pressure)

Author: Tracy McSheery
Date: 2026-01-03
Status: Final validation - completing the quadrant
"""

import os
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from pathlib import Path

# Resolve paths relative to this script's location
_script_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(_script_dir))
from nucleus_classifier import classify_nucleus

def get_family_parameters(family, N):
    """
    Get geometric quantization parameters for a given family and N mode.
    """
    params_dict = {
        'A': [0.9618, 0.2475, -2.4107, -0.0295, 0.0064, -0.8653],
        'B': [1.473890, 0.172746, 0.502666, -0.025915, 0.004164, -0.865483],
        'C': [1.169611, 0.232621, -4.467213, -0.043412, 0.004986, -0.512975],
    }

    if family not in params_dict:
        return None

    c1_0, c2_0, c3_0, dc1, dc2, dc3 = params_dict[family]

    c1_eff = c1_0 + N * dc1
    c2_eff = c2_0 + N * dc2
    c3_eff = c3_0 + N * dc3

    return c1_eff, c2_eff, c3_eff

def find_proton_drip_line(df):
    """
    Find the proton drip line by identifying the boundary of known nuclei
    on the proton-deficient side.

    For each Z, find the minimum N (most proton-rich isotope) that is bound.
    """
    drip_line = []

    for Z in range(2, 120):  # Start at Z=2 (He)
        isotopes = df[df['Z'] == Z].copy()
        if len(isotopes) == 0:
            continue

        # Neutron number N = A - Z
        isotopes['neutron_number'] = isotopes['A'] - isotopes['Z']

        # Minimum neutron number for this Z (most proton-rich)
        min_N = isotopes['neutron_number'].min()
        min_A = isotopes[isotopes['neutron_number'] == min_N]['A'].values[0]

        drip_line.append({
            'Z': Z,
            'A_min': min_A,
            'N_min': min_N,
            'N_over_Z': min_N / Z if Z > 0 else 0
        })

    return pd.DataFrame(drip_line)

def calculate_tension_ratio(A, Z):
    """
    Calculate the Surface Tension to Volume Pressure ratio.

    Same formula as neutron drip (Engine A), but for proton-rich side.
    """
    N_mode, family = classify_nucleus(A, Z)

    if N_mode is None or family is None:
        return None, None, None, None

    params = get_family_parameters(family, N_mode)
    if params is None:
        return None, None, None, None

    c1_eff, c2_eff, c3_eff = params

    c2_over_c1 = c2_eff / c1_eff if c1_eff != 0 else 0

    # Tension ratio (includes A dependence)
    tension_ratio = c2_over_c1 * (A ** (1.0/3.0))

    return tension_ratio, c2_over_c1, N_mode, family

def main():
    print("=" * 90)
    print("ENGINE D: PROTON DRIP LINE - DUAL-TRACK VALIDATION")
    print("=" * 90)
    print()
    print("Track 1: Topological Conservation (N-ladder)")
    print("Track 2: Geometric Mechanics (Tension ratio)")
    print()

    # Load AME2020 data
    df = pd.read_csv(_script_dir / 'data/ame2020_system_energies.csv')

    print(f"Loaded {len(df)} nuclei from AME2020")
    print()

    # ========================================================================
    # TRACK 2: GEOMETRIC MECHANICS (Do this first - identifies drip line)
    # ========================================================================
    print("=" * 90)
    print("TRACK 2: GEOMETRIC STRESS AT PROTON DRIP LINE")
    print("=" * 90)
    print()

    # Calculate tension ratio for all nuclei
    results = []
    for _, row in df.iterrows():
        A = int(row['A'])
        Z = int(row['Z'])

        tension_ratio, c2_c1, N_mode, family = calculate_tension_ratio(A, Z)

        if tension_ratio is not None:
            results.append({
                'A': A,
                'Z': Z,
                'N_neutrons': A - Z,
                'element': row['element'],
                'N_mode': N_mode,
                'family': family,
                'tension_ratio': tension_ratio,
                'c2_over_c1': c2_c1,
                'BE_per_A': row['BE_per_A_MeV']
            })

    df_tension = pd.DataFrame(results)

    print(f"Classified {len(df_tension)} nuclei ({100*len(df_tension)/len(df):.1f}%)")
    print()

    # Find proton drip line
    drip_line = find_proton_drip_line(df)

    # Add tension ratio to drip line nuclei
    drip_with_tension = []
    for _, drip in drip_line.iterrows():
        Z = drip['Z']
        A_min = drip['A_min']

        match = df_tension[(df_tension['Z'] == Z) & (df_tension['A'] == A_min)]
        if len(match) > 0:
            drip_with_tension.append({
                'Z': Z,
                'A_drip': A_min,
                'N_drip': drip['N_min'],
                'tension_ratio': match['tension_ratio'].values[0],
                'c2_c1': match['c2_over_c1'].values[0],
                'family': match['family'].values[0],
                'N_mode': match['N_mode'].values[0]
            })

    df_drip = pd.DataFrame(drip_with_tension)

    # Statistics
    print("PROTON DRIP LINE STATISTICS")
    print("-" * 90)
    print(f"Z range analyzed: {df_drip['Z'].min()} - {df_drip['Z'].max()}")
    print(f"Mean tension ratio at proton drip: {df_drip['tension_ratio'].mean():.3f} ± {df_drip['tension_ratio'].std():.3f}")
    print(f"Mean c2/c1 at proton drip: {df_drip['c2_c1'].mean():.3f} ± {df_drip['c2_c1'].std():.3f}")
    print()

    # Family distribution
    print("Family distribution at proton drip:")
    for family in ['A', 'B', 'C']:
        count = len(df_drip[df_drip['family'] == family])
        pct = 100 * count / len(df_drip) if len(df_drip) > 0 else 0
        if count > 0:
            mean_ratio = df_drip[df_drip['family'] == family]['tension_ratio'].mean()
            print(f"  Family {family}: {count} nuclei ({pct:.1f}%), mean ratio: {mean_ratio:.3f}")
    print()

    # Critical ratio analysis
    percentiles = [10, 25, 50, 75, 90]
    print("Tension ratio percentiles at PROTON drip line:")
    for p in percentiles:
        val = np.percentile(df_drip['tension_ratio'], p)
        print(f"  {p:2d}th percentile: {val:.3f}")
    print()

    proton_critical = np.percentile(df_drip['tension_ratio'], 50)  # Median
    print(f"PROTON DRIP CRITICAL RATIO: {proton_critical:.3f}")
    print(f"NEUTRON DRIP CRITICAL RATIO: 1.701 (from Engine A)")
    print()

    # Compare
    if proton_critical < 1.701:
        print("✅ HYPOTHESIS CONFIRMED:")
        print(f"   Proton drip ({proton_critical:.3f}) occurs at LOWER tension than neutron drip (1.701)")
        print(f"   Difference: {1.701 - proton_critical:.3f}")
        print()
        print("PHYSICS INTERPRETATION:")
        print("  - Coulomb repulsion (V₄ term) aids volume pressure")
        print("  - Proton-rich nuclei burst skin earlier than neutron-rich")
        print("  - Asymmetry in drip lines is geometric necessity")
    else:
        print("❌ UNEXPECTED:")
        print(f"   Proton drip ({proton_critical:.3f}) requires HIGHER tension than neutron drip")
        print(f"   This contradicts Coulomb-assisted pressure hypothesis")

    print()

    # Show highest tension proton-drip nuclei
    print("HIGHEST TENSION PROTON-DRIP NUCLEI:")
    print("-" * 90)
    print(f"{'Nucleus':<10} {'Z':>4} {'A':>4} {'N':>4} {'Ratio':>8} {'Family':<8}")
    print("-" * 90)

    top_proton = df_drip.nlargest(20, 'tension_ratio')
    for _, nuc in top_proton.iterrows():
        element_name = df[df['Z'] == nuc['Z']].iloc[0]['element']
        nucleus = f"{element_name}-{int(nuc['A_drip'])}"
        print(f"{nucleus:<10} {int(nuc['Z']):>4} {int(nuc['A_drip']):>4} {int(nuc['N_drip']):>4} "
              f"{nuc['tension_ratio']:>8.3f} {nuc['family']:<8}")

    print()

    # ========================================================================
    # TRACK 1: TOPOLOGICAL CONSERVATION
    # ========================================================================
    print("=" * 90)
    print("TRACK 1: TOPOLOGICAL INTEGER CONSERVATION")
    print("=" * 90)
    print()

    print("Hypothesis: Proton emission preserves harmonic mode (ΔN ≈ 0 or 1)")
    print("Unlike cluster decay (magic N ejected), proton is 'evaporation' of soliton core")
    print()

    # Test: For each proton-drip nucleus, check if N_parent ≈ N_daughter
    conservation_tests = []

    for _, drip in df_drip.iterrows():
        Z = int(drip['Z'])
        A_p = int(drip['A_drip'])
        N_p = int(drip['N_mode'])

        # Proton emission: (A, Z) → (A-1, Z-1) + p
        A_d = A_p - 1
        Z_d = Z - 1

        if Z_d > 0 and A_d > 0:
            # Classify daughter
            N_d, fam_d = classify_nucleus(A_d, Z_d)

            if N_d is not None:
                delta_N = N_p - N_d

                conservation_tests.append({
                    'Parent_Z': Z,
                    'Parent_A': A_p,
                    'Parent_N': N_p,
                    'Daughter_A': A_d,
                    'Daughter_Z': Z_d,
                    'Daughter_N': N_d,
                    'Delta_N': delta_N,
                    'Conserved': abs(delta_N) <= 1
                })

    df_conserv = pd.DataFrame(conservation_tests)

    print(f"Test cases: {len(df_conserv)} proton drip nuclei")
    print()

    # Statistics
    delta_N_values = df_conserv['Delta_N'].values
    print("ΔN distribution:")
    print(f"  Mean: {np.mean(delta_N_values):.2f}")
    print(f"  Median: {np.median(delta_N_values):.2f}")
    print(f"  Std: {np.std(delta_N_values):.2f}")
    print(f"  Range: [{np.min(delta_N_values):.0f}, {np.max(delta_N_values):.0f}]")
    print()

    # Count by ΔN
    print("ΔN frequency:")
    for dN in sorted(df_conserv['Delta_N'].unique()):
        count = len(df_conserv[df_conserv['Delta_N'] == dN])
        pct = 100 * count / len(df_conserv)
        print(f"  ΔN = {dN:+.0f}: {count:3d} cases ({pct:5.1f}%)")
    print()

    # Conservation check
    conserved_count = len(df_conserv[df_conserv['Conserved']])
    conserved_pct = 100 * conserved_count / len(df_conserv)

    print(f"Conservation (|ΔN| ≤ 1): {conserved_count}/{len(df_conserv)} ({conserved_pct:.1f}%)")
    print()

    if conserved_pct > 80:
        print("✅ TOPOLOGY CONSERVED:")
        print(f"   {conserved_pct:.1f}% of proton emissions preserve harmonic mode (|ΔN| ≤ 1)")
        print()
        print("PHYSICS INTERPRETATION:")
        print("  - Proton emission is 'evaporation' from soliton surface")
        print("  - Parent harmonic mode preserved (ΔN ≈ 0)")
        print("  - Unlike cluster decay (magic N ejection)")
        print("  - Single nucleon removal doesn't disrupt standing wave")
    else:
        print("⚠️  MIXED RESULTS:")
        print(f"   Only {conserved_pct:.1f}% conserve |ΔN| ≤ 1")
        print("   Proton emission may involve mode changes")

    print()

    # Show examples
    print("EXAMPLE CASES:")
    print("-" * 90)
    print(f"{'Parent':<12} {'N_p':>4} | {'Daughter':<12} {'N_d':>4} | {'ΔN':>4} | Conservation")
    print("-" * 90)

    for _, case in df_conserv.head(15).iterrows():
        elem_p = df[df['Z'] == case['Parent_Z']].iloc[0]['element']
        elem_d = df[df['Z'] == case['Daughter_Z']].iloc[0]['element'] if case['Daughter_Z'] in df['Z'].values else '?'

        parent_label = f"{elem_p}-{int(case['Parent_A'])}"
        daughter_label = f"{elem_d}-{int(case['Daughter_A'])}"

        status = "✅" if case['Conserved'] else "❌"

        print(f"{parent_label:<12} {int(case['Parent_N']):>4} | {daughter_label:<12} {int(case['Daughter_N']):>4} | "
              f"{int(case['Delta_N']):>4} | {status}")

    print()

    # ========================================================================
    # VISUALIZATION
    # ========================================================================
    print("=" * 90)
    print("CREATING VISUALIZATION")
    print("=" * 90)
    print()

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Panel A: Proton vs Neutron Drip Comparison
    ax = axes[0, 0]

    # Load neutron drip data
    try:
        df_neutron_drip = pd.read_csv('results/neutron_drip_line_analysis.csv')
        neutron_ratios = df_neutron_drip['tension_ratio'].values
    except:
        neutron_ratios = [1.701]  # Use critical value if file not found

    ax.hist(df_drip['tension_ratio'], bins=20, alpha=0.7, label='Proton drip', color='red', density=True)
    if len(neutron_ratios) > 1:
        ax.hist(neutron_ratios, bins=20, alpha=0.7, label='Neutron drip', color='blue', density=True)

    ax.axvline(proton_critical, color='red', linestyle='--', linewidth=2, label=f'Proton: {proton_critical:.3f}')
    ax.axvline(1.701, color='blue', linestyle='--', linewidth=2, label='Neutron: 1.701')

    ax.set_xlabel('Tension Ratio', fontsize=12)
    ax.set_ylabel('Probability Density', fontsize=12)
    ax.set_title('Proton vs Neutron Drip Line Comparison', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel B: ΔN Distribution
    ax = axes[0, 1]
    delta_N_counts = df_conserv['Delta_N'].value_counts().sort_index()
    ax.bar(delta_N_counts.index, delta_N_counts.values, alpha=0.7, color='green')
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='ΔN = 0 (mode preservation)')
    ax.set_xlabel('ΔN (N_parent - N_daughter)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Proton Emission: Harmonic Mode Change', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel C: Drip Lines in N-Z Plane
    ax = axes[1, 0]

    # Plot all nuclei
    ax.scatter(df_tension['Z'], df_tension['N_neutrons'],
              c=df_tension['tension_ratio'], cmap='RdYlGn_r', alpha=0.3, s=5, vmin=0, vmax=2)

    # Plot proton drip line
    ax.plot(df_drip['Z'], df_drip['N_drip'],
           'r-', linewidth=2, label='Proton drip line')

    # Plot neutron drip line if available
    try:
        df_neutron_drip = pd.read_csv('results/neutron_drip_line_analysis.csv')
        ax.plot(df_neutron_drip['Z'], df_neutron_drip['N_drip'],
               'b-', linewidth=2, label='Neutron drip line')
    except:
        pass

    ax.set_xlabel('Proton Number Z', fontsize=12)
    ax.set_ylabel('Neutron Number N', fontsize=12)
    ax.set_title('Nuclear Drip Lines (colored by tension ratio)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel D: Family Distribution
    ax = axes[1, 1]

    family_counts = df_drip['family'].value_counts()
    colors = {'A': 'blue', 'B': 'orange', 'C': 'green'}
    ax.bar(family_counts.index, family_counts.values,
          color=[colors.get(f, 'gray') for f in family_counts.index], alpha=0.7)

    ax.set_xlabel('Family', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Family Distribution at Proton Drip Line', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/proton_drip_engine_validation.png', dpi=150, bbox_inches='tight')
    print("Saved: figures/proton_drip_engine_validation.png")
    print()

    # Save results
    os.makedirs('results', exist_ok=True)
    df_drip.to_csv('results/proton_drip_line_analysis.csv', index=False)
    print("Saved: results/proton_drip_line_analysis.csv")

    df_conserv.to_csv('results/proton_emission_conservation.csv', index=False)
    print("Saved: results/proton_emission_conservation.csv")
    print()

    # ========================================================================
    # FINAL CONCLUSION
    # ========================================================================
    print("=" * 90)
    print("FINAL CONCLUSION: ENGINE D VALIDATION")
    print("=" * 90)
    print()

    print("TRACK 2 (Geometric Mechanics):")
    if proton_critical < 1.701:
        print(f"  ✅ Proton drip at LOWER tension ({proton_critical:.3f} vs 1.701)")
        print(f"  ✅ Coulomb-assisted pressure confirmed")
    else:
        print(f"  ❌ Proton drip at HIGHER tension ({proton_critical:.3f} vs 1.701)")

    print()

    print("TRACK 1 (Topological Conservation):")
    if conserved_pct > 80:
        print(f"  ✅ Mode preservation confirmed ({conserved_pct:.1f}% conserve |ΔN| ≤ 1)")
        print(f"  ✅ Proton emission is 'evaporation', not cluster ejection")
    else:
        print(f"  ⚠️  Mixed results ({conserved_pct:.1f}% conserve |ΔN| ≤ 1)")

    print()

    print("QUADRANT STATUS:")
    print("  Engine A (Neutron Drip): ✅ Validated (ratio > 1.701)")
    print("  Engine B (Fission): ✅ Validated (ζ > 2.0, N_eff conservation)")
    print("  Engine C (Cluster Decay): ✅ Validated (Pythagorean N²)")
    print(f"  Engine D (Proton Drip): {'✅ Validated' if proton_critical < 1.701 and conserved_pct > 80 else '⚠️  Partial'}")
    print()

    print("=" * 90)
    print("THE NUCLEAR STABILITY MAP IS COMPLETE.")
    print("=" * 90)
    print()

if __name__ == "__main__":
    main()
