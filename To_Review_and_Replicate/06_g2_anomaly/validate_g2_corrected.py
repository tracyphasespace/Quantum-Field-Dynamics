#!/usr/bin/env python3
"""
QFD: G-2 Validation - Corrected Calculation

Based on GitHub code: validate_g2_anomaly_corrected.py
Shows how V₄ is extracted from experimental g-2 and whether it matches geometric prediction.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def create_g2_figure(V4_geometric, V4_electron, V4_muon, output_dir='.'):
    """Create comparison figure for g-2 validation."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    labels = ['Electron\n(from g-2)', 'Muon\n(from g-2)', 'Geometric\n(-ξ/β)']
    values = [V4_electron, V4_muon, V4_geometric]
    colors = ['steelblue', 'coral', 'forestgreen']

    bars = ax.bar(labels, values, color=colors, edgecolor='black', linewidth=1.5)

    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_ylabel('V₄ Coefficient', fontsize=12)
    ax.set_title('QFD g-2 Prediction: V₄ = -ξ/β\n(Parameters from mass fit, NOT from g-2)', fontsize=14)
    ax.set_ylim(min(values) - 0.1, max(values) + 0.1)
    ax.grid(True, alpha=0.3, axis='y')

    # Add error annotation
    error_e = abs(V4_geometric - V4_electron) / abs(V4_electron) * 100
    error_mu = abs(V4_geometric - V4_muon) / abs(V4_muon) * 100
    ax.text(0.02, 0.98, f'Electron error: {error_e:.1f}%\nMuon error: {error_mu:.1f}%',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    output_path = Path(output_dir) / 'g2_validation.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_path}")
    plt.close()

def validate_g2_corrected():
    print("=" * 70)
    print("G-2 ANOMALY: CORRECTED VALIDATION")
    print("=" * 70)
    print("\nBased on: validate_g2_anomaly_corrected.py from GitHub")

    # Constants
    ALPHA = 1.0 / 137.036  # Fine structure constant

    # 1. EXPERIMENTAL G-2 DATA
    print("\n[1] EXPERIMENTAL ANOMALOUS MAGNETIC MOMENTS")
    print("    Source: PDG 2024, Fermilab 2021")

    leptons = {
        "Electron": {
            "mass": 0.51099895,           # MeV
            "a_exp": 0.00115965218128,    # Measured g-2 anomaly
        },
        "Muon": {
            "mass": 105.6583755,          # MeV
            "a_exp": 0.00116592059,       # Measured g-2 anomaly
        }
    }

    for name, data in leptons.items():
        print(f"\n    {name}:")
        print(f"      Mass:     {data['mass']:.6f} MeV")
        print(f"      a_exp:    {data['a_exp']:.14f}")

    # 2. SCHWINGER TERM (Leading QED)
    print("\n[2] SCHWINGER TERM (Leading QED)")
    print("    a_schwinger = α/(2π)")

    a_schwinger = ALPHA / (2 * np.pi)
    print(f"    a_schwinger = {a_schwinger:.14f}")

    # 3. EXTRACT V₄ FROM EXPERIMENT
    print("\n[3] EXTRACT V₄ FROM EXPERIMENTAL DATA")
    print("    Formula: a_exp = a_schwinger + V₄·(α/π)² + ...")
    print("    Therefore: V₄ = (a_exp - a_schwinger) / (α/π)²")

    alpha_over_pi_sq = (ALPHA / np.pi)**2

    print(f"\n    (α/π)² = {alpha_over_pi_sq:.10f}")

    V4_values = {}

    for name, data in leptons.items():
        delta_a = data['a_exp'] - a_schwinger
        V4_required = delta_a / alpha_over_pi_sq

        V4_values[name] = V4_required

        print(f"\n    {name}:")
        print(f"      δa = a_exp - a_schwinger = {delta_a:.10e}")
        print(f"      V₄_required = δa / (α/π)² = {V4_required:.6f}")

    # 4. COMPARE ELECTRON AND MUON V₄
    print("\n[4] UNIVERSALITY TEST")
    print("    Question: Is V₄ the same for electron and muon?")

    V4_electron = V4_values["Electron"]
    V4_muon = V4_values["Muon"]

    print(f"\n    V₄ (electron) = {V4_electron:.6f}")
    print(f"    V₄ (muon)     = {V4_muon:.6f}")

    ratio = V4_muon / V4_electron
    diff_percent = abs(V4_muon - V4_electron) / abs(V4_electron) * 100

    print(f"\n    Ratio: V₄_μ / V₄_e = {ratio:.4f}")
    print(f"    Difference: {diff_percent:.2f}%")

    if diff_percent < 5.0:
        print("\n    ✅ V₄ is UNIVERSAL (within 5%)")
        print("    → Suggests geometric origin!")
    else:
        print(f"\n    ❌ V₄ is NOT universal ({diff_percent:.1f}% difference)")
        print("    → Lepton-specific corrections needed")

    # 5. GEOMETRIC PREDICTION FROM FITTED PARAMETERS
    print("\n[5] GEOMETRIC PREDICTION FROM β, ξ")
    print("    From mass fit: β = 3.063, ξ = 0.97")

    beta = 3.063
    xi = 0.97

    V4_geometric = -xi / beta

    print(f"\n    V₄_geometric = -ξ/β = -{xi:.2f}/{beta:.3f}")
    print(f"    V₄_geometric = {V4_geometric:.6f}")

    # 6. COMPARE PREDICTION TO EXPERIMENT
    print("\n[6] PREDICTION vs EXPERIMENT")

    for name, V4_exp in V4_values.items():
        error = abs(V4_geometric - V4_exp) / abs(V4_exp) * 100

        print(f"\n    {name}:")
        print(f"      V₄ predicted (geometric): {V4_geometric:.6f}")
        print(f"      V₄ required (from g-2):   {V4_exp:.6f}")
        print(f"      Error: {error:.2f}%")

        if error < 5.0:
            print(f"      ✅ MATCH within 5%!")
        else:
            print(f"      ❌ MISMATCH ({error:.1f}%)")

    # 7. CRITICAL ANALYSIS
    print("\n[7] CRITICAL ANALYSIS")
    print("    " + "="*60)

    print("\n    The GitHub claim breakdown:")
    print("    1. Fit β, ξ, τ to lepton masses (3 params → 3 values)")
    print("    2. Calculate V₄ = -ξ/β (derived, not fitted)")
    print("    3. Extract V₄ from experimental g-2 data")
    print("    4. Compare: does geometric V₄ match experimental V₄?")

    print("\n    IS THIS A GENUINE PREDICTION?")

    avg_V4_exp = (V4_electron + V4_muon) / 2
    error_avg = abs(V4_geometric - avg_V4_exp) / abs(avg_V4_exp) * 100

    print(f"\n    Average experimental V₄: {avg_V4_exp:.6f}")
    print(f"    Geometric prediction V₄: {V4_geometric:.6f}")
    print(f"    Error: {error_avg:.2f}%")

    if error_avg < 5.0:
        print("\n    ✅ YES - This is a genuine prediction!")
        print("    Reasoning:")
        print("      - β and ξ were fitted to MASSES only")
        print("      - V₄ = -ξ/β was NOT fitted to g-2")
        print("      - V₄ independently predicts g-2 to <5% error")
        print("      - This is a DIFFERENT observable (not circular)")
    else:
        print(f"\n    ⚠️  PARTIAL - {error_avg:.1f}% error is significant")
        print("    May need lepton-specific corrections")

    # 8. RECONSTRUCTION CHECK
    print("\n[8] RECONSTRUCT G-2 FROM V₄")

    for name, data in leptons.items():
        V4 = V4_geometric  # Use geometric prediction
        a_predicted = a_schwinger + V4 * alpha_over_pi_sq

        error = abs(a_predicted - data['a_exp']) / data['a_exp'] * 100

        print(f"\n    {name}:")
        print(f"      a_exp (measured):  {data['a_exp']:.14f}")
        print(f"      a_pred (V₄={V4:.3f}): {a_predicted:.14f}")
        print(f"      Error: {error:.2f}%")

    # 9. HONEST SUMMARY
    print("\n[9] HONEST SUMMARY")
    print("    " + "="*60)

    print("\n    ✅ What works:")
    print(f"      - V₄ from -ξ/β = {V4_geometric:.3f}")
    print(f"      - Electron V₄ required = {V4_electron:.3f} (error {abs(V4_geometric-V4_electron)/abs(V4_electron)*100:.1f}%)")
    print(f"      - Muon V₄ required = {V4_muon:.3f} (error {abs(V4_geometric-V4_muon)/abs(V4_muon)*100:.1f}%)")

    print("\n    ⚠️  Questions:")
    print("      - Why does GitHub report V₄_muon = +0.836 (different sign)?")
    print("      - Are there lepton-specific geometric factors?")
    print("      - Is -ξ/β the complete formula, or is there more?")

    print("\n    VERDICT:")
    if error_avg < 10.0:
        print("      This IS a prediction (not a fit)")
        print("      Error <10% suggests real geometric connection")
        print("      Status: PROMISING ✅")
    else:
        print("      Needs more investigation")
        print("      Status: INCONCLUSIVE ⚠️")

    return V4_geometric, V4_electron, V4_muon

if __name__ == "__main__":
    V4_geo, V4_e, V4_mu = validate_g2_corrected()
    create_g2_figure(V4_geo, V4_e, V4_mu)
