#!/usr/bin/env python3
"""
COMPREHENSIVE DECAY ANALYSIS: All Possible Transitions in AME2020

For every nucleus, calculate:
1. Alpha decay product (A-4, Z-2) - if it exists and is energetically allowed
2. Beta- decay product (A, Z+1) - if it exists
3. Beta+ decay product (A, Z-1) - if it exists
4. Compute ΔN for each transition
5. Test selection rules on hundreds of transitions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from collections import Counter

# Load data - resolve path relative to this script's location
from pathlib import Path
_script_dir = Path(__file__).parent.resolve()
df = pd.read_csv(_script_dir / '../data/ame2020_system_energies.csv')
df = df[(df['Z'] > 0) & (df['A'] > 0)].copy()

# Create lookup dictionary for fast access
nucleus_dict = {}
for _, row in df.iterrows():
    nucleus_dict[(int(row['A']), int(row['Z']))] = row

# 3-Family Model Parameters
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

print("="*100)
print("COMPREHENSIVE DECAY TRANSITION ANALYSIS")
print("="*100)
print()

print("Classifying all nuclei...")
df['N_mode'], df['family'] = zip(*df.apply(
    lambda r: classify_nucleus(int(r['A']), int(r['Z'])), axis=1))

classified_count = df['N_mode'].notna().sum()
print(f"Classified: {classified_count}/3557 ({100*classified_count/len(df):.2f}%)")
print()

# ============================================================================
# GENERATE ALL POSSIBLE DECAY TRANSITIONS
# ============================================================================

print("Generating all possible decay transitions...")
print()

alpha_transitions = []
beta_minus_transitions = []
beta_plus_transitions = []

for _, parent in df.iterrows():
    A_p, Z_p = int(parent['A']), int(parent['Z'])
    N_p, fam_p = parent['N_mode'], parent['family']

    if N_p is None:
        continue  # Skip unclassified parents

    # 1. Alpha decay: (A,Z) → (A-4, Z-2)
    A_d, Z_d = A_p - 4, Z_p - 2
    if (A_d, Z_d) in nucleus_dict and Z_d >= 1:
        daughter = nucleus_dict[(A_d, Z_d)]
        N_d, fam_d = classify_nucleus(A_d, Z_d)

        if N_d is not None:
            # Calculate Q-value using binding energies
            # Q = BE(daughter) + BE(He-4) - BE(parent)
            BE_parent = parent['BE_per_A_MeV'] * A_p if pd.notna(parent['BE_per_A_MeV']) else 0
            BE_daughter = daughter['BE_per_A_MeV'] * A_d if pd.notna(daughter['BE_per_A_MeV']) else 0
            BE_alpha = 28.296  # He-4 total binding energy in MeV

            Q_alpha = BE_parent - BE_daughter - BE_alpha

            # Only include if energetically allowed (Q > 0)
            if Q_alpha > 0:
                alpha_transitions.append({
                    'A_p': A_p, 'Z_p': Z_p, 'N_p': N_p, 'fam_p': fam_p,
                    'A_d': A_d, 'Z_d': Z_d, 'N_d': N_d, 'fam_d': fam_d,
                    'delta_N': N_d - N_p,
                    'Q': Q_alpha,
                    'BE_p': parent['BE_per_A_MeV'],
                    'BE_d': daughter['BE_per_A_MeV']
                })

    # 2. Beta- decay: (A,Z) → (A, Z+1)
    A_d, Z_d = A_p, Z_p + 1
    if (A_d, Z_d) in nucleus_dict:
        daughter = nucleus_dict[(A_d, Z_d)]
        N_d, fam_d = classify_nucleus(A_d, Z_d)

        if N_d is not None:
            Q_beta = (parent['mass_excess_MeV'] - daughter['mass_excess_MeV'])

            if Q_beta > 0:
                beta_minus_transitions.append({
                    'A_p': A_p, 'Z_p': Z_p, 'N_p': N_p, 'fam_p': fam_p,
                    'A_d': A_d, 'Z_d': Z_d, 'N_d': N_d, 'fam_d': fam_d,
                    'delta_N': N_d - N_p,
                    'Q': Q_beta,
                    'BE_p': parent['BE_per_A_MeV'],
                    'BE_d': daughter['BE_per_A_MeV']
                })

    # 3. Beta+ decay: (A,Z) → (A, Z-1)
    A_d, Z_d = A_p, Z_p - 1
    if Z_d > 0 and (A_d, Z_d) in nucleus_dict:
        daughter = nucleus_dict[(A_d, Z_d)]
        N_d, fam_d = classify_nucleus(A_d, Z_d)

        if N_d is not None:
            Q_beta_plus = (parent['mass_excess_MeV'] - daughter['mass_excess_MeV'] -
                          2*0.511)  # Two electron masses

            if Q_beta_plus > 0:
                beta_plus_transitions.append({
                    'A_p': A_p, 'Z_p': Z_p, 'N_p': N_p, 'fam_p': fam_p,
                    'A_d': A_d, 'Z_d': Z_d, 'N_d': N_d, 'fam_d': fam_d,
                    'delta_N': N_d - N_p,
                    'Q': Q_beta_plus,
                    'BE_p': parent['BE_per_A_MeV'],
                    'BE_d': daughter['BE_per_A_MeV']
                })

df_alpha = pd.DataFrame(alpha_transitions)
df_beta_minus = pd.DataFrame(beta_minus_transitions)
df_beta_plus = pd.DataFrame(beta_plus_transitions)

print(f"Alpha transitions (Q>0): {len(df_alpha)}")
print(f"Beta- transitions (Q>0): {len(df_beta_minus)}")
print(f"Beta+ transitions (Q>0): {len(df_beta_plus)}")
print(f"Total transitions: {len(df_alpha) + len(df_beta_minus) + len(df_beta_plus)}")
print()

# ============================================================================
# ANALYZE ALPHA DECAY
# ============================================================================

print("="*100)
print("ALPHA DECAY TRANSITIONS (n = {})".format(len(df_alpha)))
print("="*100)
print()

if len(df_alpha) > 0:
    print("ΔN Distribution:")
    from collections import Counter
    dn_counts = Counter(df_alpha['delta_N'])

    print(f"{'ΔN':<6} {'Count':<10} {'%':<10} {'Mean Q (MeV)':<15}")
    print("-"*100)

    for dn in sorted(dn_counts.keys()):
        count = dn_counts[dn]
        pct = 100 * count / len(df_alpha)
        subset = df_alpha[df_alpha['delta_N'] == dn]
        mean_Q = subset['Q'].mean()

        print(f"{dn:<6} {count:<10} {pct:<10.1f} {mean_Q:<15.2f}")

    print()

    # Statistics
    print(f"Mean ΔN: {df_alpha['delta_N'].mean():.2f} ± {df_alpha['delta_N'].std():.2f}")
    print(f"Median ΔN: {df_alpha['delta_N'].median():.0f}")
    print(f"Mode ΔN: {df_alpha['delta_N'].mode().values[0]:.0f}")
    print()

# ============================================================================
# ANALYZE BETA- DECAY
# ============================================================================

print("="*100)
print("BETA- DECAY TRANSITIONS (n = {})".format(len(df_beta_minus)))
print("="*100)
print()

if len(df_beta_minus) > 0:
    print("ΔN Distribution:")
    dn_counts = Counter(df_beta_minus['delta_N'])

    print(f"{'ΔN':<6} {'Count':<10} {'%':<10} {'Prediction'}")
    print("-"*100)

    for dn in sorted(dn_counts.keys()):
        count = dn_counts[dn]
        pct = 100 * count / len(df_beta_minus)

        prediction = "✓ Correct (ΔN<0)" if dn < 0 else ("○ Same mode" if dn == 0 else "✗ Wrong (ΔN>0)")

        print(f"{dn:<6} {count:<10} {pct:<10.1f} {prediction}")

    print()

    # Validation
    negative = (df_beta_minus['delta_N'] < 0).sum()
    zero = (df_beta_minus['delta_N'] == 0).sum()
    positive = (df_beta_minus['delta_N'] > 0).sum()

    print(f"ΔN < 0 (predicted): {negative}/{len(df_beta_minus)} ({100*negative/len(df_beta_minus):.1f}%)")
    print(f"ΔN = 0: {zero}/{len(df_beta_minus)} ({100*zero/len(df_beta_minus):.1f}%)")
    print(f"ΔN > 0 (wrong): {positive}/{len(df_beta_minus)} ({100*positive/len(df_beta_minus):.1f}%)")
    print()

    if negative / len(df_beta_minus) > 0.8:
        print("★★★★★ EXCELLENT! >80% follow predicted ΔN < 0")
    elif negative / len(df_beta_minus) > 0.6:
        print("★★★ GOOD! Majority follow predicted ΔN < 0")
    else:
        print("→ Prediction needs refinement")

print()

# ============================================================================
# ANALYZE BETA+ DECAY
# ============================================================================

print("="*100)
print("BETA+ DECAY TRANSITIONS (n = {})".format(len(df_beta_plus)))
print("="*100)
print()

if len(df_beta_plus) > 0:
    print("ΔN Distribution:")
    dn_counts = Counter(df_beta_plus['delta_N'])

    print(f"{'ΔN':<6} {'Count':<10} {'%':<10} {'Prediction'}")
    print("-"*100)

    for dn in sorted(dn_counts.keys()):
        count = dn_counts[dn]
        pct = 100 * count / len(df_beta_plus)

        prediction = "✓ Correct (ΔN>0)" if dn > 0 else ("○ Same mode" if dn == 0 else "✗ Wrong (ΔN<0)")

        print(f"{dn:<6} {count:<10} {pct:<10.1f} {prediction}")

    print()

    # Validation
    positive = (df_beta_plus['delta_N'] > 0).sum()
    zero = (df_beta_plus['delta_N'] == 0).sum()
    negative = (df_beta_plus['delta_N'] < 0).sum()

    print(f"ΔN > 0 (predicted): {positive}/{len(df_beta_plus)} ({100*positive/len(df_beta_plus):.1f}%)")
    print(f"ΔN = 0: {zero}/{len(df_beta_plus)} ({100*zero/len(df_beta_plus):.1f}%)")
    print(f"ΔN < 0 (wrong): {negative}/{len(df_beta_plus)} ({100*negative/len(df_beta_plus):.1f}%)")

print()

# ============================================================================
# SELECTION RULES
# ============================================================================

print("="*100)
print("HARMONIC SELECTION RULES (All Transitions)")
print("="*100)
print()

# Combine all transitions
all_transitions = []
if len(df_alpha) > 0:
    df_alpha['mode'] = 'alpha'
    all_transitions.append(df_alpha)
if len(df_beta_minus) > 0:
    df_beta_minus['mode'] = 'beta-'
    all_transitions.append(df_beta_minus)
if len(df_beta_plus) > 0:
    df_beta_plus['mode'] = 'beta+'
    all_transitions.append(df_beta_plus)

if len(all_transitions) > 0:
    df_all = pd.concat(all_transitions, ignore_index=True)
    df_all['abs_delta_N'] = df_all['delta_N'].abs()

    print(f"Total energetically allowed transitions: {len(df_all)}")
    print()

    print(f"|ΔN| Distribution:")
    print(f"{'|ΔN|':<8} {'Count':<10} {'%':<10} {'Mean Q (MeV)':<15} {'Status'}")
    print("-"*100)

    for abs_dn in sorted(df_all['abs_delta_N'].unique()):
        subset = df_all[df_all['abs_delta_N'] == abs_dn]
        count = len(subset)
        pct = 100 * count / len(df_all)
        mean_Q = subset['Q'].mean()

        status = "Allowed" if abs_dn <= 1 else "Forbidden"

        print(f"{abs_dn:<8.0f} {count:<10} {pct:<10.1f} {mean_Q:<15.2f} {status}")

    print()

    # Statistical test
    allowed = df_all[df_all['abs_delta_N'] <= 1]
    forbidden = df_all[df_all['abs_delta_N'] > 1]

    print(f"Selection Rule Analysis:")
    print(f"  Allowed (|ΔN|≤1):   {len(allowed)}/{len(df_all)} ({100*len(allowed)/len(df_all):.1f}%)")
    print(f"  Forbidden (|ΔN|>1): {len(forbidden)}/{len(df_all)} ({100*len(forbidden)/len(df_all):.1f}%)")
    print()

    print(f"  Mean Q (allowed):   {allowed['Q'].mean():.2f} MeV")
    print(f"  Mean Q (forbidden): {forbidden['Q'].mean():.2f} MeV")
    print()

    if len(forbidden) > 0 and forbidden['Q'].mean() < allowed['Q'].mean():
        print("  ★★★ SELECTION RULE: Forbidden transitions have LOWER Q-values")
        print("      (Requires more energy → less likely)")

print()

# ============================================================================
# VISUALIZATION
# ============================================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Plot 1: Alpha decay ΔN distribution
ax = axes[0, 0]
if len(df_alpha) > 0:
    dn_vals = df_alpha['delta_N'].dropna().values
    if len(dn_vals) > 0:
        bins = np.arange(np.floor(dn_vals.min())-0.5, np.ceil(dn_vals.max())+1.5, 1)
        ax.hist(dn_vals, bins=bins, alpha=0.7, color='red', edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='ΔN=0')
    ax.set_xlabel('ΔN (daughter - parent)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'Alpha Decay: ΔN Distribution (n={len(df_alpha)})', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

# Plot 2: Beta- decay ΔN distribution
ax = axes[0, 1]
if len(df_beta_minus) > 0:
    dn_vals = df_beta_minus['delta_N'].dropna().values
    if len(dn_vals) > 0:
        bins = np.arange(np.floor(dn_vals.min())-0.5, np.ceil(dn_vals.max())+1.5, 1)
        ax.hist(dn_vals, bins=bins, alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='ΔN=0')
    ax.set_xlabel('ΔN (daughter - parent)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'Beta- Decay: ΔN Distribution (n={len(df_beta_minus)})\nPrediction: ΔN<0', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

# Plot 3: Beta+ decay ΔN distribution
ax = axes[0, 2]
if len(df_beta_plus) > 0:
    dn_vals = df_beta_plus['delta_N'].dropna().values
    if len(dn_vals) > 0:
        bins = np.arange(np.floor(dn_vals.min())-0.5, np.ceil(dn_vals.max())+1.5, 1)
        ax.hist(dn_vals, bins=bins, alpha=0.7, color='green', edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='ΔN=0')
    ax.set_xlabel('ΔN (daughter - parent)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'Beta+ Decay: ΔN Distribution (n={len(df_beta_plus)})\nPrediction: ΔN>0', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

# Plot 4: Selection rule - Q-value vs |ΔN|
ax = axes[1, 0]
if len(all_transitions) > 0:
    for mode, color in [('alpha', 'red'), ('beta-', 'blue'), ('beta+', 'green')]:
        subset = df_all[df_all['mode'] == mode]
        if len(subset) > 0:
            ax.scatter(subset['abs_delta_N'], subset['Q'],
                      alpha=0.5, s=20, c=color, label=mode)

    ax.set_xlabel('|ΔN| (absolute mode change)', fontsize=12)
    ax.set_ylabel('Q-value (MeV)', fontsize=12)
    ax.set_title('Selection Rule: Q-value vs |ΔN|', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

# Plot 5: Transition matrix (heat map)
ax = axes[1, 1]
if len(df_all) > 0:
    # Create transition count matrix
    N_range = range(-3, 11)
    transition_matrix = np.zeros((len(N_range), len(N_range)))

    for _, trans in df_all.iterrows():
        if pd.isna(trans['N_p']) or pd.isna(trans['N_d']):
            continue
            
        N_p, N_d = int(trans['N_p']), int(trans['N_d'])
        if N_p in N_range and N_d in N_range:
            i = N_range.index(N_p)
            j = N_range.index(N_d)
            transition_matrix[i, j] += 1

    im = ax.imshow(transition_matrix, cmap='YlOrRd', aspect='auto', origin='lower')
    ax.set_xticks(range(len(N_range)))
    ax.set_yticks(range(len(N_range)))
    ax.set_xticklabels(N_range)
    ax.set_yticklabels(N_range)
    ax.set_xlabel('Daughter Mode N', fontsize=12)
    ax.set_ylabel('Parent Mode N', fontsize=12)
    ax.set_title('Transition Matrix: Parent → Daughter', fontsize=14)
    plt.colorbar(im, ax=ax, label='Count')

# Plot 6: Summary statistics
ax = axes[1, 2]
ax.axis('off')

summary_text = f"""
COMPREHENSIVE DECAY ANALYSIS
{'='*40}

Total Transitions: {len(df_all) if len(all_transitions) > 0 else 0}

Alpha:  {len(df_alpha)}
Beta-:  {len(df_beta_minus)}
Beta+:  {len(df_beta_plus)}

{'='*40}
PREDICTIONS:

Beta- (ΔN<0): {100*(df_beta_minus['delta_N']<0).sum()/len(df_beta_minus):.1f}% ✓
Beta+ (ΔN>0): {100*(df_beta_plus['delta_N']>0).sum()/len(df_beta_plus):.1f}% ✓

Alpha mean ΔN: {df_alpha['delta_N'].mean():.2f}

{'='*40}
SELECTION RULE:

Allowed (|ΔN|≤1): {100*len(allowed)/len(df_all):.1f}%
Forbidden (|ΔN|>1): {100*len(forbidden)/len(df_all):.1f}%

Mean Q (allowed): {allowed['Q'].mean():.2f} MeV
Mean Q (forbidden): {forbidden['Q'].mean():.2f} MeV
"""

ax.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
       verticalalignment='center', transform=ax.transAxes)

plt.tight_layout()
plt.savefig('comprehensive_decay_analysis.png', dpi=150, bbox_inches='tight')
print("Saved: comprehensive_decay_analysis.png")
print()

print("="*100)
print("ANALYSIS COMPLETE")
print("="*100)
