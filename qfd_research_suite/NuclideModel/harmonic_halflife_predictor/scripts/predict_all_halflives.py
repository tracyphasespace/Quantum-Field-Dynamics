#!/usr/bin/env python3
"""
PREDICT HALF-LIVES FOR ALL ISOTOPES IN AME2020

Using harmonic resonance model + empirical correlations:
  - Alpha: log(t) = a + b/√Q + c*|ΔN|
  - Beta-: log(t) = a + b*log(Q) + c*|ΔN|
  - Beta+: log(t) = a + b*log(Q) + c*|ΔN|

Strategy:
1. For each nucleus, calculate Q-values for all decay modes
2. Determine which modes are energetically allowed
3. Predict half-life for each allowed mode
4. Select fastest decay (shortest t_1/2) as primary mode
5. Save predictions to CSV
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# Load AME2020 data
df_ame = pd.read_csv('./data/ame2020_system_energies.csv')
df_ame = df_ame[(df_ame['Z'] > 0) & (df_ame['A'] > 0)].copy()

# Create lookup
nucleus_dict = {}
for _, row in df_ame.iterrows():
    nucleus_dict[(int(row['A']), int(row['Z']))] = row

# 3-Family Model
params_A = [0.9618, 0.2475, -2.4107, -0.0295, 0.0064, -0.8653]
params_B = [1.473890, 0.172746, 0.502666, -0.025915, 0.004164, -0.865483]
params_C = [1.169611, 0.232621, -4.467213, -0.043412, 0.004986, -0.512975]

def classify_nucleus(A, Z):
    """Classify nucleus using 3-family model."""
    for params, N_min, N_max, family in [
        (params_A, -3, 3, 'A'),
        (params_B, -3, 3, 'B'),
        (params_C, 4, 10, 'C')
    ]:
        c1_0, c2_0, c3_0, dc1, dc2, dc3 = params
        for N in range(N_min, N_max+1):
            c1 = c1_0 + N * dc1
            c2 = c2_0 + N * dc2
            c3 = c3_0 + N * dc3
            Z_pred = c1 * (A**(2.0/3.0)) + c2 * A + c3
            if int(round(Z_pred)) == Z:
                return N, family
    return None, None

# ============================================================================
# FIT REGRESSION MODELS (from experimental data)
# ============================================================================

print("="*100)
print("FITTING PREDICTION MODELS FROM EXPERIMENTAL DATA")
print("="*100)
print()

# Load experimental half-life data
df_exp = pd.read_csv('harmonic_halflife_results.csv')

# Fit Alpha model
alpha_exp = df_exp[df_exp['mode'] == 'alpha'].copy()
if len(alpha_exp) > 10:
    X_alpha = np.column_stack([1.0/np.sqrt(alpha_exp['Q_MeV']),
                               alpha_exp['abs_delta_N']])
    y_alpha = alpha_exp['log_halflife']

    def alpha_model(X, a, b, c):
        return a + b * X[:, 0] + c * X[:, 1]

    alpha_params, _ = curve_fit(alpha_model, X_alpha, y_alpha)
    print(f"ALPHA MODEL: log(t) = {alpha_params[0]:.2f} + {alpha_params[1]:.2f}/√Q + {alpha_params[2]:.2f}*|ΔN|")
else:
    alpha_params = [-24.14, 67.05, 2.56]  # From previous analysis
    print(f"ALPHA MODEL: Using default parameters")

# Fit Beta- model
beta_minus_exp = df_exp[df_exp['mode'] == 'beta-'].copy()
if len(beta_minus_exp) > 5:
    # Simple model: log(t) ~ -log(Q^5) (Fermi golden rule approximation)
    # Plus harmonic correction
    X_beta_minus = np.column_stack([np.log10(beta_minus_exp['Q_MeV']),
                                    beta_minus_exp['abs_delta_N']])
    y_beta_minus = beta_minus_exp['log_halflife']

    def beta_model(X, a, b, c):
        return a + b * X[:, 0] + c * X[:, 1]

    try:
        beta_minus_params, _ = curve_fit(beta_model, X_beta_minus, y_beta_minus)
        print(f"BETA- MODEL: log(t) = {beta_minus_params[0]:.2f} + {beta_minus_params[1]:.2f}*log(Q) + {beta_minus_params[2]:.2f}*|ΔN|")
    except:
        beta_minus_params = [12.0, -5.0, 1.0]
        print(f"BETA- MODEL: Using default parameters (fit failed)")
else:
    beta_minus_params = [12.0, -5.0, 1.0]
    print(f"BETA- MODEL: Using default parameters")

# Fit Beta+ model
# IMPORTANT: All experimental Beta+ have |ΔN|=1, so we cannot constrain the |ΔN| coefficient
# Use simplified 2-parameter model: log(t) = a + b*log(Q)
beta_plus_exp = df_exp[df_exp['mode'] == 'beta+'].copy()
if len(beta_plus_exp) > 5:
    X_beta_plus_simple = np.log10(beta_plus_exp['Q_MeV'].values)
    y_beta_plus = beta_plus_exp['log_halflife'].values

    def beta_plus_simple_model(x, a, b):
        return a + b * x

    try:
        beta_plus_params_2d, _ = curve_fit(beta_plus_simple_model, X_beta_plus_simple, y_beta_plus)
        # Convert to 3-parameter format [a, b, c] with c=0
        beta_plus_params = [beta_plus_params_2d[0], beta_plus_params_2d[1], 0.0]
        print(f"BETA+ MODEL: log(t) = {beta_plus_params[0]:.2f} + {beta_plus_params[1]:.2f}*log(Q) + {beta_plus_params[2]:.2f}*|ΔN| (ΔN=0, no data)")
    except:
        beta_plus_params = [11.39, -23.12, 0.0]
        print(f"BETA+ MODEL: Using default parameters (fit failed)")
else:
    beta_plus_params = [11.39, -23.12, 0.0]
    print(f"BETA+ MODEL: Using default parameters")

print()

# ============================================================================
# PREDICT FOR ALL NUCLEI
# ============================================================================

print("="*100)
print("PREDICTING HALF-LIVES FOR ALL NUCLEI")
print("="*100)
print()

predictions = []
count = 0

for _, parent in df_ame.iterrows():
    A_p, Z_p = int(parent['A']), int(parent['Z'])
    N_p, fam_p = classify_nucleus(A_p, Z_p)

    if N_p is None:
        continue  # Skip unclassified

    count += 1
    if count % 500 == 0:
        print(f"Processed {count}/3557 nuclei...")

    decay_modes = []

    # --- ALPHA DECAY ---
    A_d, Z_d = A_p - 4, Z_p - 2
    if Z_d >= 1 and (A_d, Z_d) in nucleus_dict:
        daughter = nucleus_dict[(A_d, Z_d)]
        N_d, fam_d = classify_nucleus(A_d, Z_d)

        if N_d is not None and pd.notna(parent['BE_per_A_MeV']) and pd.notna(daughter['BE_per_A_MeV']):
            BE_parent = parent['BE_per_A_MeV'] * A_p
            BE_daughter = daughter['BE_per_A_MeV'] * A_d
            BE_alpha = 28.296
            Q_alpha = BE_daughter + BE_alpha - BE_parent  # Correct sign!

            if Q_alpha > 0.1:  # Energetically allowed
                delta_N = abs(N_d - N_p)
                log_t = alpha_params[0] + alpha_params[1]/np.sqrt(Q_alpha) + alpha_params[2]*delta_N

                decay_modes.append({
                    'mode': 'alpha',
                    'Q': Q_alpha,
                    'delta_N': N_d - N_p,
                    'abs_delta_N': delta_N,
                    'log_halflife': log_t,
                    'halflife_sec': 10**log_t,
                    'daughter_A': A_d,
                    'daughter_Z': Z_d,
                    'daughter_N': N_d
                })

    # --- BETA- DECAY ---
    A_d, Z_d = A_p, Z_p + 1
    if (A_d, Z_d) in nucleus_dict:
        daughter = nucleus_dict[(A_d, Z_d)]
        N_d, fam_d = classify_nucleus(A_d, Z_d)

        if N_d is not None:
            Q_beta = parent['mass_excess_MeV'] - daughter['mass_excess_MeV']

            if Q_beta > 0.01:
                delta_N = abs(N_d - N_p)
                log_t = beta_minus_params[0] + beta_minus_params[1]*np.log10(Q_beta) + beta_minus_params[2]*delta_N

                decay_modes.append({
                    'mode': 'beta-',
                    'Q': Q_beta,
                    'delta_N': N_d - N_p,
                    'abs_delta_N': delta_N,
                    'log_halflife': log_t,
                    'halflife_sec': 10**log_t,
                    'daughter_A': A_d,
                    'daughter_Z': Z_d,
                    'daughter_N': N_d
                })

    # --- BETA+ DECAY ---
    A_d, Z_d = A_p, Z_p - 1
    if Z_d > 0 and (A_d, Z_d) in nucleus_dict:
        daughter = nucleus_dict[(A_d, Z_d)]
        N_d, fam_d = classify_nucleus(A_d, Z_d)

        if N_d is not None:
            Q_beta_plus = parent['mass_excess_MeV'] - daughter['mass_excess_MeV'] - 2*0.511

            if Q_beta_plus > 0.01:
                delta_N = abs(N_d - N_p)
                log_t = beta_plus_params[0] + beta_plus_params[1]*np.log10(Q_beta_plus) + beta_plus_params[2]*delta_N

                decay_modes.append({
                    'mode': 'beta+',
                    'Q': Q_beta_plus,
                    'delta_N': N_d - N_p,
                    'abs_delta_N': delta_N,
                    'log_halflife': log_t,
                    'halflife_sec': 10**log_t,
                    'daughter_A': A_d,
                    'daughter_Z': Z_d,
                    'daughter_N': N_d
                })

    # Select fastest decay mode (shortest half-life)
    if len(decay_modes) > 0:
        fastest = min(decay_modes, key=lambda x: x['halflife_sec'])

        predictions.append({
            'A': A_p,
            'Z': Z_p,
            'element': parent['element'],
            'N_mode': N_p,
            'family': fam_p,
            'BE_per_A': parent['BE_per_A_MeV'],
            'primary_decay': fastest['mode'],
            'Q_MeV': fastest['Q'],
            'delta_N': fastest['delta_N'],
            'abs_delta_N': fastest['abs_delta_N'],
            'daughter_A': fastest['daughter_A'],
            'daughter_Z': fastest['daughter_Z'],
            'daughter_N': fastest['daughter_N'],
            'predicted_log_halflife': fastest['log_halflife'],
            'predicted_halflife_sec': fastest['halflife_sec'],
            'predicted_halflife_years': fastest['halflife_sec'] / (365.25 * 24 * 3600),
            'num_decay_modes': len(decay_modes)
        })
    else:
        # Stable nucleus (no energetically allowed decays)
        predictions.append({
            'A': A_p,
            'Z': Z_p,
            'element': parent['element'],
            'N_mode': N_p,
            'family': fam_p,
            'BE_per_A': parent['BE_per_A_MeV'],
            'primary_decay': 'stable',
            'Q_MeV': 0,
            'delta_N': 0,
            'abs_delta_N': 0,
            'daughter_A': A_p,
            'daughter_Z': Z_p,
            'daughter_N': N_p,
            'predicted_log_halflife': np.inf,
            'predicted_halflife_sec': np.inf,
            'predicted_halflife_years': np.inf,
            'num_decay_modes': 0
        })

df_predictions = pd.DataFrame(predictions)

print(f"\nCompleted predictions for {len(df_predictions)} nuclei")
print()

# ============================================================================
# STATISTICS
# ============================================================================

print("="*100)
print("PREDICTION STATISTICS")
print("="*100)
print()

print("Decay Mode Distribution:")
for mode in ['stable', 'alpha', 'beta-', 'beta+']:
    count = (df_predictions['primary_decay'] == mode).sum()
    pct = 100 * count / len(df_predictions)
    print(f"  {mode:<10}: {count:>4} ({pct:>5.1f}%)")

print()

# Half-life ranges
for mode in ['alpha', 'beta-', 'beta+']:
    subset = df_predictions[df_predictions['primary_decay'] == mode]
    if len(subset) > 0:
        print(f"{mode.upper()} half-lives:")
        print(f"  Range: 10^{subset['predicted_log_halflife'].min():.1f} to 10^{subset['predicted_log_halflife'].max():.1f} sec")
        print(f"  Median: 10^{subset['predicted_log_halflife'].median():.1f} sec = {subset['predicted_halflife_years'].median():.2e} years")
        print()

# Selection rule statistics
print("Selection Rule Distribution:")
for abs_dn in sorted(df_predictions['abs_delta_N'].unique()):
    if abs_dn == 0:
        continue
    count = (df_predictions['abs_delta_N'] == abs_dn).sum()
    pct = 100 * count / len(df_predictions[df_predictions['abs_delta_N'] > 0])
    status = "Allowed" if abs_dn <= 1 else "Forbidden"
    print(f"  |ΔN| = {abs_dn:.0f}: {count:>4} ({pct:>5.1f}%) - {status}")

print()

# ============================================================================
# SAVE TO CSV
# ============================================================================

output_file = 'predicted_halflives_all_isotopes.csv'
df_predictions.to_csv(output_file, index=False)

print(f"Saved predictions to: {output_file}")
print()

# ============================================================================
# CREATE SUMMARY MD FILE
# ============================================================================

md_file = 'predicted_halflives_summary.md'

with open(md_file, 'w') as f:
    f.write("# Predicted Half-Lives for All AME2020 Isotopes\n\n")
    f.write(f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d')}\n\n")
    f.write(f"**Total isotopes:** {len(df_predictions)}\n\n")

    f.write("## Prediction Models\n\n")
    f.write("### Alpha Decay\n")
    f.write("```\n")
    f.write(f"log₁₀(t₁/₂) = {alpha_params[0]:.2f} + {alpha_params[1]:.2f}/√Q + {alpha_params[2]:.2f}*|ΔN|\n")
    f.write("```\n\n")

    f.write("### Beta- Decay\n")
    f.write("```\n")
    f.write(f"log₁₀(t₁/₂) = {beta_minus_params[0]:.2f} + {beta_minus_params[1]:.2f}*log(Q) + {beta_minus_params[2]:.2f}*|ΔN|\n")
    f.write("```\n\n")

    f.write("### Beta+ Decay\n")
    f.write("```\n")
    f.write(f"log₁₀(t₁/₂) = {beta_plus_params[0]:.2f} + {beta_plus_params[1]:.2f}*log(Q) + {beta_plus_params[2]:.2f}*|ΔN|\n")
    f.write("```\n\n")

    f.write("## Decay Mode Distribution\n\n")
    f.write("| Mode | Count | Percentage |\n")
    f.write("|------|-------|------------|\n")
    for mode in ['stable', 'alpha', 'beta-', 'beta+']:
        count = (df_predictions['primary_decay'] == mode).sum()
        pct = 100 * count / len(df_predictions)
        f.write(f"| {mode} | {count} | {pct:.1f}% |\n")

    f.write("\n## Examples\n\n")
    f.write("### Very Long-Lived Alpha Emitters\n\n")
    long_alpha = df_predictions[df_predictions['primary_decay'] == 'alpha'].nlargest(5, 'predicted_halflife_years')
    f.write("| Isotope | Q (MeV) | ΔN | Predicted t₁/₂ (years) |\n")
    f.write("|---------|---------|----|-----------------------|\n")
    for _, row in long_alpha.iterrows():
        f.write(f"| {row['element']}-{row['A']} | {row['Q_MeV']:.2f} | {row['delta_N']:.0f} | {row['predicted_halflife_years']:.2e} |\n")

    f.write("\n### Fast Beta- Emitters\n\n")
    fast_beta = df_predictions[df_predictions['primary_decay'] == 'beta-'].nsmallest(5, 'predicted_halflife_sec')
    f.write("| Isotope | Q (MeV) | ΔN | Predicted t₁/₂ (sec) |\n")
    f.write("|---------|---------|----|-----------------------|\n")
    for _, row in fast_beta.iterrows():
        f.write(f"| {row['element']}-{row['A']} | {row['Q_MeV']:.2f} | {row['delta_N']:.0f} | {row['predicted_halflife_sec']:.2e} |\n")

    f.write("\n### Predicted Stable Isotopes\n\n")
    stable = df_predictions[df_predictions['primary_decay'] == 'stable']
    f.write(f"Total: {len(stable)} isotopes\n\n")
    f.write(f"Examples: {', '.join([f'{row.element}-{row.A}' for _, row in stable.head(20).iterrows()])}\n\n")

    f.write("## Full Dataset\n\n")
    f.write(f"See `{output_file}` for complete predictions.\n")

print(f"Saved summary to: {md_file}")
print()

print("="*100)
print("PREDICTION COMPLETE")
print("="*100)
