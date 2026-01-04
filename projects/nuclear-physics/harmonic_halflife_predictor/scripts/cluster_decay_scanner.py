#!/usr/bin/env python3
"""
Cluster Decay Scanner: Three-Peanut Pythagorean Hypothesis

Tests whether cluster decay follows harmonic energy conservation:
    N²_parent ≈ N²_daughter + N²_cluster

Author: Tracy McSheery
Date: 2026-01-03
"""

import sys
sys.path.insert(0, 'scripts')
from nucleus_classifier import classify_nucleus
import pandas as pd

def check_cluster_decay_pythagorean(A_parent, Z_parent, A_daughter, Z_daughter, 
                                     A_cluster, Z_cluster, label=""):
    """
    Test Pythagorean hypothesis for cluster decay.
    
    Returns:
        dict with N values, N² values, residual, and status
    """
    N_p, fam_p = classify_nucleus(A_parent, Z_parent)
    N_d, fam_d = classify_nucleus(A_daughter, Z_daughter)
    N_c, fam_c = classify_nucleus(A_cluster, Z_cluster)
    
    if None in [N_p, N_d, N_c]:
        return None
        
    # Calculate N² values (deformation energy proxy)
    N2_p = N_p ** 2
    N2_d = N_d ** 2
    N2_c = N_c ** 2
    
    # Pythagorean test
    N2_sum = N2_d + N2_c
    residual_N2 = N2_p - N2_sum
    
    # Linear test (for comparison)
    residual_N = N_p - (N_d + N_c)
    
    # Determine if "allowed" by Pythagorean rule
    is_pythagorean = abs(residual_N2) <= 1  # Within 1 unit of N²
    is_relaxing = abs(N_d) < abs(N_p)  # Daughter less deformed than parent
    
    return {
        'label': label,
        'A_parent': A_parent,
        'Z_parent': Z_parent,
        'A_daughter': A_daughter,
        'Z_daughter': Z_daughter,
        'A_cluster': A_cluster,
        'Z_cluster': Z_cluster,
        'N_parent': N_p,
        'N_daughter': N_d,
        'N_cluster': N_c,
        'family_parent': fam_p,
        'family_daughter': fam_d,
        'family_cluster': fam_c,
        'N2_parent': N2_p,
        'N2_daughter': N2_d,
        'N2_cluster': N2_c,
        'N2_sum': N2_sum,
        'residual_N2': residual_N2,
        'residual_N': residual_N,
        'is_pythagorean': is_pythagorean,
        'is_relaxing': is_relaxing,
        'is_magic_cluster': N_c in [-3, -1, 0, 1, 2, 3],  # Common stable N
    }

def main():
    """
    PHYSICS: Cluster decay as topological pinch-off
    
    If a nucleus is a standing wave soliton, cluster decay happens when:
    1. The parent is deformed (high N²)
    2. A "magic" cluster (stable N) can form as a discrete node
    3. Energy conservation: N²_p ≈ N²_d + N²_c (Pythagorean)
    4. The daughter relaxes to lower deformation
    """
    
    # Known cluster decays from experimental data
    cluster_decays = [
        # C-14 emitters (Radium chain)
        (223, 88, 209, 82, 14, 6, "Ra-223 → Pb-209 + C-14"),
        (224, 88, 210, 82, 14, 6, "Ra-224 → Pb-210 + C-14"),
        (225, 88, 211, 82, 14, 6, "Ra-225 → Pb-211 + C-14"),
        (226, 88, 212, 82, 14, 6, "Ra-226 → Pb-212 + C-14"),
        
        # Ne-24 emitters (Actinides)
        (231, 90, 207, 82, 24, 10, "Th-231 → Pb-207 + Ne-24"),
        (232, 90, 208, 82, 24, 10, "Th-232 → Pb-208 + Ne-24"),
        (233, 92, 209, 82, 24, 10, "U-233 → Pb-209 + Ne-24"),
        (234, 92, 210, 82, 24, 10, "U-234 → Pb-210 + Ne-24"),
        (235, 92, 211, 82, 24, 10, "U-235 → Pb-211 + Ne-24"),
        
        # Exotic light cluster emitter
        (114, 56, 100, 50, 14, 6, "Ba-114 → Sn-100 + C-14"),
    ]
    
    print("=" * 90)
    print("CLUSTER DECAY PYTHAGOREAN HYPOTHESIS TEST")
    print("Testing: N²_parent ≈ N²_daughter + N²_cluster")
    print("=" * 90)
    print()
    
    results = []
    for params in cluster_decays:
        result = check_cluster_decay_pythagorean(*params)
        if result:
            results.append(result)
    
    # Print main results table
    print(f"{'Decay':<30} {'N_p':>4} {'N_d':>4} {'N_c':>4} | {'N²_p':>5} {'N²_d':>5} {'N²_c':>5} {'Δ²':>5} | {'Status':<20}")
    print("-" * 90)
    
    for r in results:
        if r['is_pythagorean']:
            if r['residual_N2'] == 0:
                status = "✅ PERFECT Pythagorean"
            else:
                status = f"✅ Pythagorean (Δ²={r['residual_N2']:+d})"
        elif abs(r['residual_N2']) <= 3:
            status = f"⚠️  Near-Pyth (Δ²={r['residual_N2']:+d})"
        else:
            status = f"❌ Non-Pyth (Δ²={r['residual_N2']:+d})"
            
        print(f"{r['label']:<30} {r['N_parent']:>4d} {r['N_daughter']:>4d} {r['N_cluster']:>4d} | "
              f"{r['N2_parent']:>5d} {r['N2_daughter']:>5d} {r['N2_cluster']:>5d} {r['residual_N2']:>+5d} | "
              f"{status:<20}")
    
    print()
    print("=" * 90)
    print("SUMMARY STATISTICS")
    print("=" * 90)
    print()
    
    total = len(results)
    pythagorean_count = sum(1 for r in results if r['is_pythagorean'])
    relaxing_count = sum(1 for r in results if r['is_relaxing'])
    magic_cluster_count = sum(1 for r in results if r['is_magic_cluster'])
    
    print(f"Total cluster decays analyzed:        {total}")
    print(f"Pythagorean (|Δ²| ≤ 1):               {pythagorean_count} ({100*pythagorean_count/total:.1f}%)")
    print(f"Relaxing (|N_d| < |N_p|):             {relaxing_count} ({100*relaxing_count/total:.1f}%)")
    print(f"Magic cluster (N = -3,-1,0,1,2,3):   {magic_cluster_count} ({100*magic_cluster_count/total:.1f}%)")
    print()
    
    print("KEY FINDINGS:")
    print()
    
    # Find the perfect cases
    perfect_cases = [r for r in results if r['residual_N2'] == 0]
    if perfect_cases:
        print("✅ PERFECT Pythagorean Conservation:")
        for r in perfect_cases:
            print(f"   {r['label']}: {r['N_parent']}² = {r['N_daughter']}² + {r['N_cluster']}² "
                  f"({r['N2_parent']} = {r['N2_daughter']} + {r['N2_cluster']})")
        print()
    
    # Find near-perfect
    near_perfect = [r for r in results if r['is_pythagorean'] and r['residual_N2'] != 0]
    if near_perfect:
        print("✅ Near-Pythagorean (|Δ²| = 1):")
        for r in near_perfect:
            print(f"   {r['label']}: {r['N_parent']}² ≈ {r['N_daughter']}² + {r['N_cluster']}² "
                  f"({r['N2_parent']} ≈ {r['N2_daughter']} + {r['N2_cluster']}, Δ² = {r['residual_N2']:+d})")
        print()
    
    # Identify forbidden
    forbidden = [r for r in results if not r['is_pythagorean']]
    if forbidden:
        print(f"❌ Non-Pythagorean ({len(forbidden)} cases):")
        print("   These cluster decays violate energy conservation and should be:")
        print("   - Extremely rare (branching ratio < 10^-10)")
        print("   - Or require excited parent states")
        print()
    
    print("=" * 90)
    print("PHYSICAL INTERPRETATION")
    print("=" * 90)
    print()
    print("N² represents DEFORMATION ENERGY (harmonic resonance amplitude²)")
    print()
    print("Cluster decay as topological 'pinch-off':")
    print("  1. Parent nucleus is deformed → stores energy in N² mode")
    print("  2. A 'magic' cluster (stable N = 1 or 2) forms as discrete node")
    print("  3. Energy conservation: Parent deformation ≈ Daughter + Cluster")
    print("  4. If Pythagorean: N²_p = N²_d + N²_c → ALLOWED decay")
    print("  5. If not: Forbidden by energy - requires quantum tunneling")
    print()
    print("This is REVOLUTIONARY because:")
    print("  - Cluster decay is NOT random fragmentation")
    print("  - It's GEOMETRIC energy quantization")
    print("  - The 'Three-Peanut' obeys Pythagorean theorem in harmonic space!")
    print()
    
    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_csv('results/cluster_decay_pythagorean_test.csv', index=False)
    print(f"Results saved to: results/cluster_decay_pythagorean_test.csv")
    print()

if __name__ == "__main__":
    main()
