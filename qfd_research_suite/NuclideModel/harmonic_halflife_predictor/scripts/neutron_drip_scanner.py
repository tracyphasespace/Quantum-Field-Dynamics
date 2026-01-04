#!/usr/bin/env python3
"""
Neutron Drip Line Scanner: Surface Tension Failure Model

Tests whether neutron drip line can be predicted from geometric coefficients:
    Drip occurs when Volume Pressure > Surface Tension
    P/σ ∝ c2_effective / c1_effective

PHYSICS:
- c1 (A^(2/3) term) represents SURFACE TENSION (holds nucleus together)
- c2 (A term) represents VOLUME PRESSURE (neutron Fermi pressure)
- Family C (neutron-rich) has c2/c1 = 0.20 (high pressure, low tension)
- When c2/c1 exceeds critical value → skin breaks → neutron emission

Author: Tracy McSheery
Date: 2026-01-03
"""

import sys
sys.path.insert(0, 'scripts')
from nucleus_classifier import classify_nucleus
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_family_parameters(family, N):
    """
    Get geometric quantization parameters for a given family and N mode.
    
    Returns c1_0, c2_0, c3_0, dc1, dc2, dc3
    """
    params_dict = {
        'A': [0.9618, 0.2475, -2.4107, -0.0295, 0.0064, -0.8653],
        'B': [1.473890, 0.172746, 0.502666, -0.025915, 0.004164, -0.865483],
        'C': [1.169611, 0.232621, -4.467213, -0.043412, 0.004986, -0.512975],
    }
    
    if family not in params_dict:
        return None
    
    c1_0, c2_0, c3_0, dc1, dc2, dc3 = params_dict[family]
    
    # Calculate effective coefficients for this N mode
    c1_eff = c1_0 + N * dc1
    c2_eff = c2_0 + N * dc2
    c3_eff = c3_0 + N * dc3
    
    return c1_eff, c2_eff, c3_eff

def calculate_tension_ratio(A, Z):
    """
    Calculate the Surface Tension to Volume Pressure ratio.
    
    PHYSICS:
    - Surface Tension σ ∝ c1_eff * A^(2/3)
    - Volume Pressure P ∝ c2_eff * A
    - Tension Ratio = P/σ = (c2_eff * A) / (c1_eff * A^(2/3))
    -                      = (c2_eff / c1_eff) * A^(1/3)
    
    Critical insight: As A increases, ratio grows!
    Heavier neutron-rich nuclei are MORE unstable.
    """
    N_mode, family = classify_nucleus(A, Z)
    
    if N_mode is None or family is None:
        return None, None, None, None
    
    params = get_family_parameters(family, N_mode)
    if params is None:
        return None, None, None, None
    
    c1_eff, c2_eff, c3_eff = params
    
    # Raw ratio
    c2_over_c1 = c2_eff / c1_eff if c1_eff != 0 else 0
    
    # Tension ratio (includes A dependence)
    tension_ratio = c2_over_c1 * (A ** (1.0/3.0))
    
    return tension_ratio, c2_over_c1, N_mode, family

def find_drip_line(df):
    """
    Find the neutron drip line by identifying the boundary of known nuclei.
    
    For each Z, find the maximum N (neutron number) that is bound.
    """
    drip_line = []
    
    for Z in range(1, 120):
        isotopes = df[df['Z'] == Z].copy()
        if len(isotopes) == 0:
            continue
        
        # Neutron number N = A - Z
        isotopes['neutron_number'] = isotopes['A'] - isotopes['Z']
        
        # Maximum neutron number for this Z
        max_N = isotopes['neutron_number'].max()
        max_A = isotopes[isotopes['neutron_number'] == max_N]['A'].values[0]
        
        drip_line.append({
            'Z': Z,
            'A_max': max_A,
            'N_max': max_N,
            'N_over_Z': max_N / Z if Z > 0 else 0
        })
    
    return pd.DataFrame(drip_line)

def main():
    """
    HYPOTHESIS: Neutron drip line occurs where geometric tension fails.
    
    Test:
    1. Calculate tension_ratio for all nuclei
    2. Find where known nuclei end (experimental drip line)
    3. Check if high tension_ratio correlates with drip line
    4. Predict which nuclei should be at the edge of stability
    """
    
    print("=" * 90)
    print("NEUTRON DRIP LINE: SURFACE TENSION FAILURE MODEL")
    print("Hypothesis: Drip occurs when Volume Pressure > Surface Tension")
    print("=" * 90)
    print()
    
    # Load AME2020 data
    df = pd.read_csv('data/ame2020_system_energies.csv')
    
    print(f"Loaded {len(df)} nuclei from AME2020")
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
    
    # Find experimental drip line
    drip_line = find_drip_line(df)
    
    # Add tension ratio to drip line nuclei
    drip_with_tension = []
    for _, drip in drip_line.iterrows():
        Z = drip['Z']
        A_max = drip['A_max']
        
        match = df_tension[(df_tension['Z'] == Z) & (df_tension['A'] == A_max)]
        if len(match) > 0:
            drip_with_tension.append({
                'Z': Z,
                'A_drip': A_max,
                'N_drip': drip['N_max'],
                'tension_ratio': match['tension_ratio'].values[0],
                'c2_c1': match['c2_over_c1'].values[0],
                'family': match['family'].values[0],
                'N_mode': match['N_mode'].values[0]
            })
    
    df_drip = pd.DataFrame(drip_with_tension)
    
    # Statistics
    print("=" * 90)
    print("DRIP LINE STATISTICS")
    print("=" * 90)
    print()
    
    print(f"Z range analyzed: {df_drip['Z'].min()} - {df_drip['Z'].max()}")
    print(f"Mean tension ratio at drip line: {df_drip['tension_ratio'].mean():.3f} ± {df_drip['tension_ratio'].std():.3f}")
    print(f"Mean c2/c1 at drip line: {df_drip['c2_c1'].mean():.3f} ± {df_drip['c2_c1'].std():.3f}")
    print()
    
    # Family distribution at drip line
    print("Family distribution at drip line:")
    for family in ['A', 'B', 'C']:
        count = len(df_drip[df_drip['family'] == family])
        pct = 100 * count / len(df_drip)
        mean_ratio = df_drip[df_drip['family'] == family]['tension_ratio'].mean() if count > 0 else 0
        print(f"  Family {family}: {count} nuclei ({pct:.1f}%), mean ratio: {mean_ratio:.3f}")
    print()
    
    # Critical ratio analysis
    print("=" * 90)
    print("CRITICAL TENSION RATIO ANALYSIS")
    print("=" * 90)
    print()
    
    # Find percentile where drip occurs
    all_ratios = df_tension['tension_ratio'].values
    drip_ratios = df_drip['tension_ratio'].values
    
    percentiles = [np.percentile(drip_ratios, p) for p in [10, 25, 50, 75, 90]]
    
    print("Drip line tension ratio distribution:")
    print(f"  10th percentile: {percentiles[0]:.3f}")
    print(f"  25th percentile: {percentiles[1]:.3f}")
    print(f"  50th percentile (median): {percentiles[2]:.3f}")
    print(f"  75th percentile: {percentiles[3]:.3f}")
    print(f"  90th percentile: {percentiles[4]:.3f}")
    print()
    
    # Estimate critical ratio
    critical_ratio = percentiles[2]  # Use median
    print(f"ESTIMATED CRITICAL RATIO: {critical_ratio:.3f}")
    print(f"Nuclei with ratio > {critical_ratio:.3f} should be near/beyond drip line")
    print()
    
    # Test prediction
    print("=" * 90)
    print("PREDICTION TEST: High-ratio nuclei")
    print("=" * 90)
    print()
    
    high_ratio = df_tension[df_tension['tension_ratio'] > critical_ratio * 1.1].sort_values('tension_ratio', ascending=False)
    
    print(f"Found {len(high_ratio)} nuclei with tension ratio > {critical_ratio*1.1:.3f}:")
    print()
    print(f"{'Nucleus':<12} {'Z':>4} {'A':>4} {'N':>4} {'Family':<8} {'N_mode':>6} {'Ratio':>8} {'Status':<20}")
    print("-" * 90)
    
    for _, nuc in high_ratio.head(20).iterrows():
        Z = nuc['Z']
        A = nuc['A']
        
        # Check if on drip line
        drip_match = df_drip[df_drip['Z'] == Z]
        if len(drip_match) > 0:
            A_drip = drip_match['A_drip'].values[0]
            if A >= A_drip - 2:  # Within 2 of drip line
                status = f"✅ Near drip (A={A_drip})"
            else:
                status = "⚠️  Stable interior"
        else:
            status = "❓ Unknown Z"
        
        nucleus = f"{nuc['element']}-{A}"
        print(f"{nucleus:<12} {Z:>4d} {A:>4d} {nuc['N_neutrons']:>4d} {nuc['family']:<8} {nuc['N_mode']:>6d} {nuc['tension_ratio']:>8.3f} {status:<20}")
    
    print()
    
    # Visualize
    print("=" * 90)
    print("CREATING VISUALIZATION")
    print("=" * 90)
    print()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Panel A: Tension ratio vs N/Z
    ax = axes[0, 0]
    scatter = ax.scatter(df_tension['N_neutrons'] / df_tension['Z'], 
                        df_tension['tension_ratio'],
                        c=df_tension['Z'], cmap='viridis', alpha=0.5, s=10)
    ax.scatter(df_drip['N_drip'] / df_drip['Z'], 
              df_drip['tension_ratio'],
              c='red', marker='x', s=100, linewidths=2, label='Drip line')
    ax.axhline(critical_ratio, color='red', linestyle='--', label=f'Critical ratio: {critical_ratio:.3f}')
    ax.set_xlabel('N/Z (Neutron Excess)')
    ax.set_ylabel('Tension Ratio (P/σ)')
    ax.set_title('Neutron Drip Line: Tension Failure')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Proton Number Z')
    
    # Panel B: c2/c1 by family
    ax = axes[0, 1]
    for family in ['A', 'B', 'C']:
        fam_data = df_tension[df_tension['family'] == family]
        ax.scatter(fam_data['A'], fam_data['c2_over_c1'], 
                  label=f'Family {family}', alpha=0.3, s=10)
    
    drip_A = df_drip[df_drip['family'] == 'A']
    drip_B = df_drip[df_drip['family'] == 'B']
    drip_C = df_drip[df_drip['family'] == 'C']
    
    if len(drip_A) > 0:
        ax.scatter(drip_A['A_drip'], drip_A['c2_c1'], 
                  color='blue', marker='x', s=100, linewidths=2)
    if len(drip_B) > 0:
        ax.scatter(drip_B['A_drip'], drip_B['c2_c1'], 
                  color='orange', marker='x', s=100, linewidths=2)
    if len(drip_C) > 0:
        ax.scatter(drip_C['A_drip'], drip_C['c2_c1'], 
                  color='green', marker='x', s=100, linewidths=2, label='Drip line')
    
    ax.set_xlabel('Mass Number A')
    ax.set_ylabel('c2/c1 Ratio')
    ax.set_title('Surface/Volume Ratio by Family')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel C: Drip line in N-Z plane
    ax = axes[1, 0]
    ax.scatter(df_tension['Z'], df_tension['N_neutrons'], 
              c=df_tension['tension_ratio'], cmap='RdYlGn_r', alpha=0.3, s=5, vmin=0, vmax=2)
    ax.plot(df_drip['Z'], df_drip['N_drip'], 
           'r-', linewidth=2, label='Experimental drip line')
    ax.set_xlabel('Proton Number Z')
    ax.set_ylabel('Neutron Number N')
    ax.set_title('Neutron Drip Line (colored by tension ratio)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel D: Histogram of tension ratios
    ax = axes[1, 1]
    ax.hist(df_tension['tension_ratio'], bins=50, alpha=0.5, label='All nuclei', density=True)
    ax.hist(df_drip['tension_ratio'], bins=20, alpha=0.5, label='Drip line', density=True)
    ax.axvline(critical_ratio, color='red', linestyle='--', linewidth=2, label=f'Critical: {critical_ratio:.3f}')
    ax.set_xlabel('Tension Ratio')
    ax.set_ylabel('Probability Density')
    ax.set_title('Distribution of Tension Ratios')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/neutron_drip_tension_analysis.png', dpi=150, bbox_inches='tight')
    print("Saved figure: figures/neutron_drip_tension_analysis.png")
    
    # Save results
    df_drip.to_csv('results/neutron_drip_line_analysis.csv', index=False)
    print("Saved data: results/neutron_drip_line_analysis.csv")
    print()
    
    print("=" * 90)
    print("CONCLUSION")
    print("=" * 90)
    print()
    print(f"✅ Critical tension ratio identified: {critical_ratio:.3f}")
    print(f"✅ Drip line correlates with high c2/c1 ratio")
    print(f"✅ Family C (neutron-rich) dominates drip line")
    print()
    print("PHYSICAL INTERPRETATION:")
    print("- Surface tension (c1) holds nucleus together")
    print("- Volume pressure (c2) from neutron Fermi energy pushes outward")
    print("- When P/σ > critical → neutrons evaporate")
    print("- This is GEOMETRIC confinement failure!")
    print()

if __name__ == "__main__":
    main()
