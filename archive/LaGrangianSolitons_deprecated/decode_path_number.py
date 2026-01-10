#!/usr/bin/env python3
"""
DECODE PATH NUMBER N - FIND PHYSICAL CORRELATE
===========================================================================
We have 7 paths (N = -3 to +3) that classify all 285 nuclei perfectly.

Question: What determines which nucleus belongs to which path?

Test correlations:
1. N vs A (mass dependence)
2. N vs Z (charge dependence)
3. N vs N_neutron (neutron excess)
4. N vs q = Z/A (charge fraction)
5. N vs A mod 4 (topology)
6. N vs parity (even-even, odd-odd, odd-A)
7. N vs (N-Z) (isospin)

Goal: Find formula to predict N without testing all 7 paths.
===========================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from collections import defaultdict

# Load QFD constants and path model
c1_0 = 0.961752
c2_0 = 0.247527
c3_0 = -2.410727
delta_c1 = -0.029498
delta_c2 = 0.006412
delta_c3 = -0.865252

def predict_Z_path_N(A, N):
    """Predict Z using path N."""
    c1 = c1_0 + N * delta_c1
    c2 = c2_0 + N * delta_c2
    c3 = c3_0 + N * delta_c3
    Z_pred = c1 * (A**(2/3)) + c2 * A + c3
    return int(round(Z_pred))

# Load data
with open('qfd_optimized_suite.py', 'r') as f:
    content = f.read()
start = content.find('test_nuclides = [')
end = content.find(']', start) + 1
test_nuclides = eval(content[start:end].replace('test_nuclides = ', ''))

# Classify all nuclei
data = []
for name, Z_exp, A in test_nuclides:
    # Find which path this nucleus belongs to
    for N in range(-3, 4):
        Z_pred = predict_Z_path_N(A, N)
        if Z_pred == Z_exp:
            N_neutron = A - Z_exp
            data.append({
                'name': name,
                'A': A,
                'Z': Z_exp,
                'N_neutron': N_neutron,
                'N_path': N,
                'q': Z_exp / A,
                'N_minus_Z': N_neutron - Z_exp,
                'mod4': A % 4,
                'parity': 'even-even' if (Z_exp % 2 == 0 and N_neutron % 2 == 0) else
                          'odd-odd' if (Z_exp % 2 == 1 and N_neutron % 2 == 1) else 'odd-A',
            })
            break

print("="*95)
print("DECODE PATH NUMBER N - CORRELATION ANALYSIS")
print("="*95)
print()

# ============================================================================
# CORRELATION TESTS
# ============================================================================
print("="*95)
print("CORRELATION WITH NUCLEAR PROPERTIES")
print("="*95)
print()

N_path_values = [d['N_path'] for d in data]

correlations = []

# Test various properties
properties = [
    ('A (mass)', [d['A'] for d in data]),
    ('Z (protons)', [d['Z'] for d in data]),
    ('N (neutrons)', [d['N_neutron'] for d in data]),
    ('q = Z/A (charge fraction)', [d['q'] for d in data]),
    ('N - Z (neutron excess)', [d['N_minus_Z'] for d in data]),
    ('A mod 4', [d['mod4'] for d in data]),
]

print(f"{'Property':<30} {'Correlation r':<15} {'Significance'}")
print("-"*95)

for prop_name, prop_values in properties:
    r, p_value = pearsonr(N_path_values, prop_values)
    
    marker = ""
    if abs(r) > 0.7:
        marker = "★★★ STRONG"
    elif abs(r) > 0.5:
        marker = "★★ MODERATE"
    elif abs(r) > 0.3:
        marker = "★ WEAK"
    
    print(f"{prop_name:<30} {r:<15.4f} {marker}")

print()

# ============================================================================
# PATH DISTRIBUTION BY PROPERTIES
# ============================================================================
print("="*95)
print("PATH DISTRIBUTION BY A MOD 4")
print("="*95)
print()

print(f"{'A mod 4':<12} {'Path N distribution'}")
print("-"*95)

for mod in range(4):
    mod_data = [d for d in data if d['mod4'] == mod]
    path_dist = defaultdict(int)
    for d in mod_data:
        path_dist[d['N_path']] += 1
    
    dist_str = ', '.join([f"N={N}:{count}" for N, count in sorted(path_dist.items())])
    print(f"{mod:<12} {dist_str}")

print()

# ============================================================================
# PATH DISTRIBUTION BY PARITY
# ============================================================================
print("="*95)
print("PATH DISTRIBUTION BY PARITY")
print("="*95)
print()

print(f"{'Parity':<15} {'Path N distribution (top 3)'}")
print("-"*95)

for parity in ['even-even', 'odd-odd', 'odd-A']:
    parity_data = [d for d in data if d['parity'] == parity]
    path_dist = defaultdict(int)
    for d in parity_data:
        path_dist[d['N_path']] += 1
    
    # Get top 3 paths
    top_paths = sorted(path_dist.items(), key=lambda x: x[1], reverse=True)[:3]
    dist_str = ', '.join([f"N={N}:{count}" for N, count in top_paths])
    print(f"{parity:<15} {dist_str}")

print()

# ============================================================================
# MEAN VALUES BY PATH
# ============================================================================
print("="*95)
print("MEAN NUCLEAR PROPERTIES BY PATH")
print("="*95)
print()

print(f"{'Path N':<10} {'<A>':<10} {'<Z>':<10} {'<N-Z>':<10} {'<q>':<10} {'Dominant parity'}")
print("-"*95)

for N in range(-3, 4):
    path_data = [d for d in data if d['N_path'] == N]
    
    if len(path_data) == 0:
        continue
    
    mean_A = np.mean([d['A'] for d in path_data])
    mean_Z = np.mean([d['Z'] for d in path_data])
    mean_NZ = np.mean([d['N_minus_Z'] for d in path_data])
    mean_q = np.mean([d['q'] for d in path_data])
    
    # Dominant parity
    parity_counts = defaultdict(int)
    for d in path_data:
        parity_counts[d['parity']] += 1
    dominant_parity = max(parity_counts.items(), key=lambda x: x[1])[0]
    
    print(f"{N:<10} {mean_A:<10.1f} {mean_Z:<10.1f} {mean_NZ:<10.1f} {mean_q:<10.4f} {dominant_parity}")

print()

# ============================================================================
# PREDICTIVE MODEL: CAN WE PREDICT N FROM (A, Z)?
# ============================================================================
print("="*95)
print("PREDICTIVE MODEL: N FROM (A, Z)")
print("="*95)
print()

# Try simple linear regression
from sklearn.linear_model import LinearRegression

X = np.array([[d['A'], d['Z'], d['N_neutron'], d['q'], d['N_minus_Z']] for d in data])
y = np.array([d['N_path'] for d in data])

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)
y_pred_rounded = np.round(y_pred).astype(int)

# Clip to valid range
y_pred_rounded = np.clip(y_pred_rounded, -3, 3)

accuracy = np.mean(y_pred_rounded == y)

print(f"Linear regression model:")
print(f"  N ≈ {model.intercept_:.4f}")
print(f"      + {model.coef_[0]:.6f} × A")
print(f"      + {model.coef_[1]:.6f} × Z")
print(f"      + {model.coef_[2]:.6f} × N_neutron")
print(f"      + {model.coef_[3]:.6f} × q")
print(f"      + {model.coef_[4]:.6f} × (N-Z)")
print()
print(f"Accuracy predicting N: {accuracy:.1%} ({int(accuracy*285)}/285)")
print()

if accuracy > 0.95:
    print("★★★ EXCELLENT! N is predictable from (A, Z)")
    print("    Can use formula instead of testing all 7 paths")
elif accuracy > 0.8:
    print("★★ GOOD! N is mostly predictable")
    print("   Can narrow search to ±1 around predicted N")
else:
    print("→ N has complex dependence")
    print("  Need to test all 7 paths")

print()

# ============================================================================
# VISUALIZATIONS
# ============================================================================
print("="*95)
print("CREATING VISUALIZATIONS...")
print("="*95)
print()

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Plot 1: N vs A
ax1 = axes[0, 0]
for N in range(-3, 4):
    path_data = [d for d in data if d['N_path'] == N]
    A_vals = [d['A'] for d in path_data]
    N_vals = [N] * len(A_vals)
    ax1.scatter(A_vals, N_vals, alpha=0.6, s=30, label=f'N={N}')

ax1.set_xlabel('Mass Number A', fontsize=12)
ax1.set_ylabel('Path Number N', fontsize=12)
ax1.set_title('Path Assignment vs Mass', fontsize=14, fontweight='bold')
ax1.set_yticks(range(-3, 4))
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=8)

# Plot 2: N vs q = Z/A
ax2 = axes[0, 1]
for N in range(-3, 4):
    path_data = [d for d in data if d['N_path'] == N]
    q_vals = [d['q'] for d in path_data]
    N_vals = [N] * len(q_vals)
    ax2.scatter(q_vals, N_vals, alpha=0.6, s=30, label=f'N={N}')

ax2.set_xlabel('Charge Fraction q = Z/A', fontsize=12)
ax2.set_ylabel('Path Number N', fontsize=12)
ax2.set_title('Path vs Charge Fraction', fontsize=14, fontweight='bold')
ax2.set_yticks(range(-3, 4))
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=8)

# Plot 3: N vs (N-Z)
ax3 = axes[0, 2]
for N in range(-3, 4):
    path_data = [d for d in data if d['N_path'] == N]
    NZ_vals = [d['N_minus_Z'] for d in path_data]
    N_vals = [N] * len(NZ_vals)
    ax3.scatter(NZ_vals, N_vals, alpha=0.6, s=30, label=f'N={N}')

ax3.set_xlabel('Neutron Excess (N-Z)', fontsize=12)
ax3.set_ylabel('Path Number N', fontsize=12)
ax3.set_title('Path vs Neutron Excess', fontsize=14, fontweight='bold')
ax3.set_yticks(range(-3, 4))
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=8)

# Plot 4: Path population distribution
ax4 = axes[1, 0]
path_counts = [len([d for d in data if d['N_path'] == N]) for N in range(-3, 4)]
ax4.bar(range(-3, 4), path_counts, color='steelblue', edgecolor='black')
ax4.set_xlabel('Path Number N', fontsize=12)
ax4.set_ylabel('Population', fontsize=12)
ax4.set_title('Path Population Distribution', fontsize=14, fontweight='bold')
ax4.set_xticks(range(-3, 4))
ax4.grid(True, alpha=0.3, axis='y')

# Plot 5: Predicted vs Actual N
ax5 = axes[1, 1]
ax5.scatter(y, y_pred, alpha=0.5, s=20)
ax5.plot([-3, 3], [-3, 3], 'r--', linewidth=2, label='Perfect prediction')
ax5.set_xlabel('Actual Path N', fontsize=12)
ax5.set_ylabel('Predicted N (regression)', fontsize=12)
ax5.set_title(f'Path Predictability (R²={accuracy:.3f})', fontsize=14, fontweight='bold')
ax5.set_xticks(range(-3, 4))
ax5.set_yticks(range(-3, 4))
ax5.legend()
ax5.grid(True, alpha=0.3)

# Plot 6: c1/c2 ratio evolution
ax6 = axes[1, 2]
N_vals = range(-3, 4)
ratios = [(c1_0 + N*delta_c1)/(c2_0 + N*delta_c2) for N in N_vals]
path_pops = [len([d for d in data if d['N_path'] == N]) for N in N_vals]

ax6.bar(N_vals, ratios, width=0.6, alpha=0.7, color='green', edgecolor='black')
ax6.scatter(N_vals, ratios, s=np.array(path_pops)*3, c='red', alpha=0.6, zorder=3)
ax6.set_xlabel('Path Number N', fontsize=12)
ax6.set_ylabel('Envelope/Core Ratio (c1/c2)', fontsize=12)
ax6.set_title('Geometric Ratio vs Path', fontsize=14, fontweight='bold')
ax6.set_xticks(range(-3, 4))
ax6.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('path_number_decode.png', dpi=150, bbox_inches='tight')
print("Saved: path_number_decode.png")
print()

print("="*95)
