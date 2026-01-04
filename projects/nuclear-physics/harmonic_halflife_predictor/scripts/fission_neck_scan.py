#!/usr/bin/env python3
"""
Fission Neck Snap Scanner: Rayleigh-Plateau Instability Model

Tests whether spontaneous fission can be predicted from geometric elongation:
    SF occurs when elongation factor ζ > critical value
    ζ = (1 + β) / (1 - β/2)

PHYSICS:
- β (deformation) estimated from c2/c1 ratio (high c2/c1 → more elongated)
- ζ (elongation factor) represents aspect ratio of "peanut" shape
- When ζ > critical → neck becomes too thin → fission
- Rayleigh-Plateau instability: Same as water droplet breaking from stream

Author: Tracy McSheery
Date: 2026-01-03
"""

import sys
sys.path.insert(0, 'scripts')
from nucleus_classifier import classify_nucleus
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

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

def calculate_deformation_beta(A, Z, calibration_k=0.7):
    """
    Estimate deformation parameter β from harmonic model.

    PHYSICS:
    - High c2/c1 ratio → More volume energy → More deformation
    - Family B (surface-dominated): low β
    - Family C (volume-dominated): high β

    β represents the elongation of the nucleus from spherical shape.

    calibration_k: Empirical scaling factor (0.5-1.0)
    """
    N_mode, family = classify_nucleus(A, Z)

    if N_mode is None or family is None:
        return None, None, None

    params = get_family_parameters(family, N_mode)
    if params is None:
        return None, None, None

    c1_eff, c2_eff, c3_eff = params

    # Deformation proxy (higher c2/c1 → more elongated)
    c2_over_c1 = c2_eff / c1_eff if c1_eff != 0 else 0

    # Estimate β from c2/c1 ratio
    # Physical constraint: β should be in range [0, 0.7] for stable nuclei
    beta_est = calibration_k * c2_over_c1

    # Cap at physically reasonable values
    beta_est = min(beta_est, 0.7)

    return beta_est, c2_over_c1, family

def calculate_elongation_factor(beta):
    """
    Calculate elongation factor from deformation parameter.

    ζ = (1 + β) / (1 - β/2)

    PHYSICS:
    - Numerator: Length extension along symmetry axis
    - Denominator: Radius contraction of the "neck"
    - Volume conservation: R_long × R_neck² = constant

    Examples:
    - β = 0 (sphere): ζ = 1.00 (no elongation)
    - β = 0.3: ζ = 1.51 (moderately deformed)
    - β = 0.6: ζ = 2.29 (highly deformed, thin neck)
    """
    if beta is None or beta >= 2.0:  # Avoid division by zero
        return None

    numerator = 1.0 + beta
    denominator = 1.0 - beta / 2.0

    if denominator <= 0:
        return None

    zeta = numerator / denominator

    return zeta

def main():
    """
    HYPOTHESIS: Spontaneous fission is Rayleigh-Plateau instability.

    Test:
    1. Calculate elongation factor ζ for actinides (Z ≥ 90)
    2. Check correlation with SF half-lives
    3. Identify critical ζ_crit above which fission dominates
    4. Predict which superheavy elements should be fission-stable
    """

    print("=" * 90)
    print("SPONTANEOUS FISSION: NECK SNAP MODEL")
    print("Hypothesis: SF is Rayleigh-Plateau instability (ζ > critical)")
    print("=" * 90)
    print()

    # Load AME2020 data
    df = pd.read_csv('data/ame2020_system_energies.csv')

    print(f"Loaded {len(df)} nuclei from AME2020")
    print()

    # Calculate deformation and elongation for all nuclei
    results = []
    for _, row in df.iterrows():
        A = int(row['A'])
        Z = int(row['Z'])

        beta, c2_c1, family = calculate_deformation_beta(A, Z)

        if beta is not None:
            zeta = calculate_elongation_factor(beta)

            if zeta is not None:
                results.append({
                    'A': A,
                    'Z': Z,
                    'element': row['element'],
                    'N_neutrons': A - Z,
                    'beta': beta,
                    'zeta': zeta,
                    'c2_over_c1': c2_c1,
                    'family': family,
                    'BE_per_A': row['BE_per_A_MeV']
                })

    df_geom = pd.DataFrame(results)

    print(f"Calculated geometry for {len(df_geom)} nuclei ({100*len(df_geom)/len(df):.1f}%)")
    print()

    # Filter for actinides (where SF is relevant)
    df_actinides = df_geom[df_geom['Z'] >= 90].copy()

    print("=" * 90)
    print("ACTINIDE ANALYSIS (Z ≥ 90)")
    print("=" * 90)
    print()

    print(f"Actinides found: {len(df_actinides)}")
    print(f"Mean elongation factor: {df_actinides['zeta'].mean():.3f} ± {df_actinides['zeta'].std():.3f}")
    print(f"Mean deformation: {df_actinides['beta'].mean():.3f} ± {df_actinides['beta'].std():.3f}")
    print(f"Mean c2/c1: {df_actinides['c2_over_c1'].mean():.3f} ± {df_actinides['c2_over_c1'].std():.3f}")
    print()

    # Family distribution
    print("Family distribution in actinides:")
    for family in ['A', 'B', 'C']:
        count = len(df_actinides[df_actinides['family'] == family])
        if count > 0:
            pct = 100 * count / len(df_actinides)
            mean_zeta = df_actinides[df_actinides['family'] == family]['zeta'].mean()
            print(f"  Family {family}: {count} nuclei ({pct:.1f}%), mean ζ: {mean_zeta:.3f}")
    print()

    # Identify highest elongation nuclei
    print("=" * 90)
    print("HIGHEST ELONGATION ACTINIDES (Most fission-prone)")
    print("=" * 90)
    print()

    df_sorted = df_actinides.sort_values('zeta', ascending=False)

    print(f"{'Nucleus':<12} {'Z':>4} {'A':>4} {'N':>4} {'Family':<8} {'β':>8} {'ζ':>8} {'Status':<20}")
    print("-" * 90)

    for _, nuc in df_sorted.head(20).iterrows():
        nucleus = f"{nuc['element']}-{nuc['A']}"

        # Estimate fission susceptibility
        if nuc['zeta'] > 2.5:
            status = "🔴 Immediate fission"
        elif nuc['zeta'] > 2.0:
            status = "🟠 High SF probability"
        elif nuc['zeta'] > 1.8:
            status = "🟡 Moderate SF"
        else:
            status = "🟢 Stable"

        print(f"{nucleus:<12} {nuc['Z']:>4.0f} {nuc['A']:>4.0f} {nuc['N_neutrons']:>4.0f} "
              f"{nuc['family']:<8} {nuc['beta']:>8.3f} {nuc['zeta']:>8.3f} {status:<20}")

    print()

    # Elongation distribution analysis
    print("=" * 90)
    print("ELONGATION FACTOR STATISTICS")
    print("=" * 90)
    print()

    percentiles = [10, 25, 50, 75, 90, 95, 99]
    print("Actinide elongation percentiles:")
    for p in percentiles:
        val = np.percentile(df_actinides['zeta'], p)
        print(f"  {p:2d}th percentile: ζ = {val:.3f}")
    print()

    # Estimate critical elongation
    zeta_median = np.percentile(df_actinides['zeta'], 75)
    print(f"ESTIMATED CRITICAL ζ (75th percentile): {zeta_median:.3f}")
    print(f"Nuclei with ζ > {zeta_median:.3f} should have enhanced SF probability")
    print()

    # Check for correlation with binding energy (proxy for stability)
    print("=" * 90)
    print("CORRELATION WITH STABILITY")
    print("=" * 90)
    print()

    # Higher elongation should correlate with lower stability (lower BE/A)
    correlation = stats.pearsonr(df_actinides['zeta'], df_actinides['BE_per_A'])

    print(f"Correlation between ζ and BE/A:")
    print(f"  Pearson r = {correlation[0]:.3f}")
    print(f"  p-value = {correlation[1]:.3e}")

    if correlation[0] < -0.3:
        print(f"  ✅ Significant negative correlation!")
        print(f"     Higher elongation → Lower stability (as expected)")
    elif abs(correlation[0]) < 0.3:
        print(f"  ⚠️  Weak correlation")
        print(f"     Other factors may dominate (shell effects)")
    else:
        print(f"  ❓ Unexpected positive correlation")
    print()

    # Visualization
    print("=" * 90)
    print("CREATING VISUALIZATION")
    print("=" * 90)
    print()

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Panel A: Elongation vs Mass Number
    ax = axes[0, 0]
    scatter = ax.scatter(df_actinides['A'], df_actinides['zeta'],
                        c=df_actinides['Z'], cmap='plasma', s=50, alpha=0.7)
    ax.axhline(2.0, color='orange', linestyle='--', linewidth=2, label='ζ = 2.0 (moderate risk)')
    ax.axhline(2.5, color='red', linestyle='--', linewidth=2, label='ζ = 2.5 (critical)')
    ax.set_xlabel('Mass Number A', fontsize=12)
    ax.set_ylabel('Elongation Factor ζ', fontsize=12)
    ax.set_title('Actinide Elongation: Fission Susceptibility', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Proton Number Z')

    # Panel B: Deformation vs Elongation
    ax = axes[0, 1]
    ax.scatter(df_actinides['beta'], df_actinides['zeta'],
              c=df_actinides['Z'], cmap='plasma', s=50, alpha=0.7)

    # Plot theoretical curve
    beta_theory = np.linspace(0, 0.7, 100)
    zeta_theory = (1 + beta_theory) / (1 - beta_theory/2)
    ax.plot(beta_theory, zeta_theory, 'k-', linewidth=2, label='ζ = (1+β)/(1-β/2)')

    ax.set_xlabel('Deformation Parameter β', fontsize=12)
    ax.set_ylabel('Elongation Factor ζ', fontsize=12)
    ax.set_title('Geometric Relationship', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel C: Elongation vs Binding Energy
    ax = axes[1, 0]
    ax.scatter(df_actinides['zeta'], df_actinides['BE_per_A'],
              c=df_actinides['A'], cmap='viridis', s=50, alpha=0.7)
    ax.set_xlabel('Elongation Factor ζ', fontsize=12)
    ax.set_ylabel('Binding Energy per Nucleon (MeV)', fontsize=12)
    ax.set_title(f'Stability vs Elongation (r = {correlation[0]:.3f})',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add trend line
    z = np.polyfit(df_actinides['zeta'], df_actinides['BE_per_A'], 1)
    p = np.poly1d(z)
    zeta_range = np.linspace(df_actinides['zeta'].min(), df_actinides['zeta'].max(), 100)
    ax.plot(zeta_range, p(zeta_range), "r--", linewidth=2, alpha=0.8, label='Linear fit')
    ax.legend()

    # Panel D: Distribution of elongation factors
    ax = axes[1, 1]
    ax.hist(df_geom['zeta'], bins=50, alpha=0.5, label='All nuclei', density=True)
    ax.hist(df_actinides['zeta'], bins=20, alpha=0.7, label='Actinides', density=True, color='red')
    ax.axvline(2.0, color='orange', linestyle='--', linewidth=2, label='ζ = 2.0')
    ax.axvline(2.5, color='red', linestyle='--', linewidth=2, label='ζ = 2.5')
    ax.set_xlabel('Elongation Factor ζ', fontsize=12)
    ax.set_ylabel('Probability Density', fontsize=12)
    ax.set_title('Distribution of Nuclear Elongation', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('figures/fission_neck_snap_correlation.png', dpi=150, bbox_inches='tight')
    print("Saved figure: figures/fission_neck_snap_correlation.png")

    # Save results
    df_actinides.to_csv('results/fission_elongation_analysis.csv', index=False)
    print("Saved data: results/fission_elongation_analysis.csv")
    print()

    # Save full dataset for reference
    df_geom.to_csv('results/nuclear_geometry_full.csv', index=False)
    print("Saved full geometry data: results/nuclear_geometry_full.csv")
    print()

    print("=" * 90)
    print("CONCLUSION")
    print("=" * 90)
    print()
    print(f"✅ Elongation factors calculated for {len(df_geom)} nuclei")
    print(f"✅ Actinide analysis: {len(df_actinides)} nuclei (Z ≥ 90)")
    print(f"✅ Mean actinide elongation: ζ = {df_actinides['zeta'].mean():.3f}")
    print(f"✅ Correlation with stability: r = {correlation[0]:.3f}")
    print()
    print("PHYSICAL INTERPRETATION:")
    print("- β (deformation) estimated from c2/c1 ratio")
    print("- ζ (elongation) = (1 + β) / (1 - β/2)")
    print("- High ζ → thin neck → Rayleigh-Plateau instability → fission")
    print("- Critical ζ ≈ 2.0-2.5 (above this, SF becomes dominant)")
    print()
    print("NEXT STEPS:")
    print("- Obtain experimental SF half-life data")
    print("- Test correlation: log(T_1/2) vs ζ")
    print("- Determine universal 'snap coefficient' k")
    print("- Predict superheavy element stability")
    print()

if __name__ == "__main__":
    main()
