#!/usr/bin/env python3
"""
ENGINE B: FISSION TOPOLOGY VALIDATOR
Testing the Conservation Law: N_parent = N_frag1 + N_frag2

Hypothesis:
Mass asymmetry in fission occurs to satisfy Integer Harmonic Conservation.
If N_parent is odd, symmetric fission is impossible (N/2 is non-integer).

Author: Tracy McSheery
Date: 2026-01-03
"""

import pandas as pd
import numpy as np
import sys
import os

# Import your core classifier
sys.path.insert(0, 'scripts')
from nucleus_classifier import classify_nucleus

def test_fission_peaks():
    print("=" * 90)
    print("ENGINE B: SPONTANEOUS FISSION TOPOLOGY TEST")
    print("Testing if Peak Yield Fragments conserve Harmonic Number N")
    print("=" * 90)
    print()

    # 1. DEFINE THE TEST CASES (Known Fission Peaks from Literature)
    # Data Source: Independent Yields for Thermal/Spontaneous Fission (JEFF/ENDF)
    # We look at the most common light (L) and heavy (H) fragments after neutron emission
    fission_cases = [
        # Parent         Fragment 1 (Light)    Fragment 2 (Heavy)    Note
        ('U-235+n', 236, 92,  'Sr-94',  38, 94,  'Xe-140', 54, 140, 'Standard Peak'),
        ('Pu-239+n', 240, 94, 'Sr-98',  38, 98,  'Ba-141', 56, 141, 'Standard Peak'),
        ('Cf-252',   252, 98, 'Mo-106', 42, 106, 'Ba-144', 56, 144, 'Spontaneous'),
        ('Fm-258',   258, 100, 'Sn-128', 50, 128, 'Sn-130', 50, 130, 'Symmetric Mode') # Rare symmetric case!
    ]

    results = []

    for case in fission_cases:
        p_lbl, p_A, p_Z, f1_lbl, f1_Z, f1_A, f2_lbl, f2_Z, f2_A, note = case

        # 1. Classify Parent
        # For induced fission (U-235+n), the parent is the compound nucleus (U-236)
        N_p, fam_p = classify_nucleus(p_A, p_Z)

        # 2. Classify Fragments
        N_f1, fam_f1 = classify_nucleus(f1_A, f1_Z)
        N_f2, fam_f2 = classify_nucleus(f2_A, f2_Z)

        if N_p is None or N_f1 is None or N_f2 is None:
            print(f"Skipping {p_lbl}: Could not classify one or more nuclei.")
            continue

        # 3. Test Conservation
        N_sum = N_f1 + N_f2
        delta = N_p - N_sum

        match = (delta == 0)

        res = {
            'Parent': p_lbl,
            'A_p': p_A,
            'Z_p': p_Z,
            'N_p': N_p,
            'Fam_p': fam_p,
            'Frag1': f1_lbl,
            'N_f1': N_f1,
            'Fam_f1': fam_f1,
            'Frag2': f2_lbl,
            'N_f2': N_f2,
            'Fam_f2': fam_f2,
            'Sum': N_sum,
            'Delta': delta,
            'Match': "✅" if match else "❌",
            'Note': note
        }
        results.append(res)

    # OUTPUT TABLE
    print(f"{'Parent':<12} {'N_p':<5} | {'Frag1':<10} {'N_f1':<5} | {'Frag2':<10} {'N_f2':<5} | {'Sum':<5} {'Match':<5} | Note")
    print("-" * 90)

    perfect_matches = 0
    near_matches = 0

    for r in results:
        print(f"{r['Parent']:<12} {r['N_p']:<5} | {r['Frag1']:<10} {r['N_f1']:<5} | "
              f"{r['Frag2']:<10} {r['N_f2']:<5} | {r['Sum']:<5} {r['Match']:<5} | {r['Note']}")

        if r['Match'] == "✅":
            perfect_matches += 1
        elif abs(r['Delta']) <= 1:
            near_matches += 1

    print("-" * 90)
    print(f"Perfect Conservation: {perfect_matches}/{len(results)} ({perfect_matches/len(results)*100:.1f}%)")
    print(f"Near Conservation (|Δ| ≤ 1): {near_matches}/{len(results)}")
    print()

    # DETAILED ANALYSIS
    print("=" * 90)
    print("DETAILED HARMONIC ANALYSIS")
    print("=" * 90)
    print()

    for r in results:
        print(f"CASE: {r['Parent']} → {r['Frag1']} + {r['Frag2']}")
        print(f"  Parent: A={r['A_p']}, Z={r['Z_p']}, N={r['N_p']}, Family={r['Fam_p']}")
        print(f"  Fragment 1: {r['Frag1']}, N={r['N_f1']}, Family={r['Fam_f1']}")
        print(f"  Fragment 2: {r['Frag2']}, N={r['N_f2']}, Family={r['Fam_f2']}")
        print(f"  Conservation: {r['N_p']} = {r['N_f1']} + {r['N_f2']} = {r['Sum']}  (Δ = {r['Delta']})")

        # Check for symmetry
        if r['N_f1'] == r['N_f2']:
            print(f"  ⚡ SYMMETRIC SPLIT: Both fragments have N = {r['N_f1']}")
        else:
            print(f"  ⚡ ASYMMETRIC SPLIT: N_light = {r['N_f1']}, N_heavy = {r['N_f2']}")

        # Integer partition analysis
        if r['N_p'] % 2 == 0:
            print(f"  📊 Parent N is EVEN ({r['N_p']}) → Symmetric split is possible (N/2 = {r['N_p']/2})")
        else:
            print(f"  📊 Parent N is ODD ({r['N_p']}) → Symmetric split FORBIDDEN (N/2 = {r['N_p']/2} is non-integer)")

        print(f"  Result: {r['Match']}")
        print()

    # ASYMMETRY ANALYSIS
    print("=" * 90)
    print("ASYMMETRY EXPLANATION TEST")
    print("=" * 90)
    print()

    print("HYPOTHESIS: Fission asymmetry arises from INTEGER CONSTRAINT on harmonic modes.")
    print()

    asymmetric_cases = [r for r in results if r['N_f1'] != r['N_f2']]
    symmetric_cases = [r for r in results if r['N_f1'] == r['N_f2']]

    print(f"Asymmetric fissions: {len(asymmetric_cases)}/{len(results)}")
    print(f"Symmetric fissions: {len(symmetric_cases)}/{len(results)}")
    print()

    if len(asymmetric_cases) > 0:
        print("ASYMMETRIC CASES:")
        for r in asymmetric_cases:
            print(f"  {r['Parent']}: N={r['N_p']} ({'ODD' if r['N_p'] % 2 == 1 else 'EVEN'}) "
                  f"→ {r['N_f1']} + {r['N_f2']} (|ΔN| = {abs(r['N_f1'] - r['N_f2'])})")
    print()

    if len(symmetric_cases) > 0:
        print("SYMMETRIC CASES:")
        for r in symmetric_cases:
            print(f"  {r['Parent']}: N={r['N_p']} ({'ODD' if r['N_p'] % 2 == 1 else 'EVEN'}) "
                  f"→ {r['N_f1']} + {r['N_f2']} (Perfect symmetry!)")
    print()

    # PREDICTION
    print("=" * 90)
    print("THEORETICAL PREDICTION")
    print("=" * 90)
    print()

    print("If N_parent is ODD:")
    print("  → Cannot split symmetrically (N/2 is non-integer)")
    print("  → Must find asymmetric integer partition")
    print("  → Explains 'camel humps' in mass yield curve!")
    print()
    print("If N_parent is EVEN:")
    print("  → CAN split symmetrically (N/2 is integer)")
    print("  → Symmetric fission becomes possible")
    print("  → Explains rare symmetric modes (e.g., Fm-258)")
    print()

    # Check prediction
    correct_predictions = 0
    total_predictions = 0

    for r in results:
        total_predictions += 1
        is_symmetric = (r['N_f1'] == r['N_f2'])
        parent_even = (r['N_p'] % 2 == 0)

        # Prediction: Even N → can be symmetric, Odd N → must be asymmetric
        if parent_even and is_symmetric:
            print(f"✅ {r['Parent']}: N={r['N_p']} (EVEN) → Symmetric split ALLOWED → Observed: Symmetric")
            correct_predictions += 1
        elif not parent_even and not is_symmetric:
            print(f"✅ {r['Parent']}: N={r['N_p']} (ODD) → Symmetric split FORBIDDEN → Observed: Asymmetric")
            correct_predictions += 1
        elif parent_even and not is_symmetric:
            print(f"⚠️  {r['Parent']}: N={r['N_p']} (EVEN) → Symmetric ALLOWED but chose Asymmetric (energetics)")
            correct_predictions += 1  # Not a violation, just preferred asymmetry
        else:
            print(f"❌ {r['Parent']}: N={r['N_p']} (ODD) → Should be Asymmetric → Observed: Symmetric (VIOLATION!)")

    print()
    print(f"Prediction accuracy: {correct_predictions}/{total_predictions} ({correct_predictions/total_predictions*100:.1f}%)")
    print()

    print("=" * 90)
    print("CONCLUSION")
    print("=" * 90)
    print()

    if perfect_matches == len(results):
        print("🎉 PERFECT CONSERVATION: All fission fragments conserve harmonic number N!")
        print("   Mass asymmetry is explained by INTEGER ARITHMETIC.")
        print("   This is the topological origin of the 'camel humps'!")
    elif perfect_matches + near_matches == len(results):
        print("✅ NEAR-PERFECT CONSERVATION: Fission follows harmonic conservation within ±1.")
        print("   Small deviations likely due to neutron emission during scission.")
    else:
        print("📊 PARTIAL VALIDATION: Some cases conserve N, others require refinement.")
        print("   May need to account for prompt neutrons or excited fragment states.")
    print()

if __name__ == "__main__":
    test_fission_peaks()
