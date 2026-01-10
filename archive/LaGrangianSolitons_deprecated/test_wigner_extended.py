#!/usr/bin/env python3
"""
EXTENDED WIGNER TEST - INCLUDING NEGATIVE W (STRONGER PENALTY)
===========================================================================
Previous test: W=0 was optimal (no improvement)

Now test:
- Positive W: Attractive Wigner energy (np pairing bonus)
- Negative W: Stronger penalty than current (repulsive)
- W = -11.0: Equivalent to current model

Also test A-dependent vs 1/A vs 1/√A scaling
===========================================================================
"""

import numpy as np
from collections import Counter

# QFD Constants
alpha_fine = 1.0 / 137.036
beta_vacuum = 1.0 / 3.058231
M_proton = 938.272
KAPPA_E = 0.0001
SHIELD_FACTOR = 0.52
DELTA_PAIRING = 11.0

# Derived
V_0 = M_proton * (1 - (alpha_fine**2) / beta_vacuum)
beta_nuclear = M_proton * beta_vacuum / 2
E_surface_coeff = beta_nuclear / 15
a_sym_base = (beta_vacuum * M_proton) / 15
a_disp = (alpha_fine * 197.327 / 1.2) * SHIELD_FACTOR

def qfd_energy_wigner_general(A, Z, W, scaling='A'):
    """General Wigner energy with different scaling options."""
    N = A - Z
    if Z <= 0 or N < 0:
        return 1e12

    q = Z / A
    lambda_time = KAPPA_E * Z

    E_volume_coeff = V_0 * (1 - lambda_time / (12 * np.pi))
    E_bulk = E_volume_coeff * A
    E_surf = E_surface_coeff * (A**(2/3))
    E_asym = a_sym_base * A * ((1 - 2*q)**2)
    E_vac = a_disp * (Z**2) / (A**(1/3))

    # Pairing energy with different scalings
    E_pair = 0
    if Z % 2 == 0 and N % 2 == 0:
        E_pair = -DELTA_PAIRING / np.sqrt(A)
    elif Z % 2 == 1 and N % 2 == 1:
        # Different scaling options for odd-odd
        if scaling == 'A':
            E_pair = -W / A
        elif scaling == 'sqrtA':
            E_pair = -W / np.sqrt(A)
        elif scaling == 'A23':
            E_pair = -W / (A**(2/3))
        else:
            E_pair = -W / A

    return E_bulk + E_surf + E_asym + E_vac + E_pair

def find_stable_Z(A, energy_func):
    """Find Z with minimum energy."""
    best_Z, best_E = 1, energy_func(A, 1)
    for Z in range(1, A):
        E = energy_func(A, Z)
        if E < best_E:
            best_E, best_Z = E, Z
    return best_Z

# Load data
with open('qfd_optimized_suite.py', 'r') as f:
    content = f.read()
start = content.find('test_nuclides = [')
end = content.find(']', start) + 1
test_nuclides = eval(content[start:end].replace('test_nuclides = ', ''))

print("="*95)
print("EXTENDED WIGNER TEST - ALL SCALINGS")
print("="*95)
print()

# Test different scalings
scalings_to_test = ['A', 'sqrtA', 'A23']
scaling_names = {
    'A': '1/A',
    'sqrtA': '1/√A',
    'A23': '1/A^(2/3)'
}

best_overall = None
best_overall_matches = 175

for scaling in scalings_to_test:
    print(f"Testing scaling: E_pair = -W × {scaling_names[scaling]}")
    print("-"*95)
    
    # Test W from -50 to +50 MeV
    W_values = np.linspace(-50, 50, 101)
    results = []
    
    for W in W_values:
        correct = 0
        for name, Z_exp, A in test_nuclides:
            Z_pred = find_stable_Z(A, lambda A, Z: qfd_energy_wigner_general(A, Z, W, scaling))
            if Z_pred == Z_exp:
                correct += 1
        results.append((W, correct))
    
    results.sort(key=lambda x: x[1], reverse=True)
    
    # Show top 5
    for i, (W, matches) in enumerate(results[:5]):
        delta = matches - 175
        marker = "★★★" if delta >= 5 else "★★" if delta >= 3 else "★" if delta > 0 else ""
        print(f"  W={W:+6.1f}: {matches}/285  ({delta:+d})  {marker}")
    
    if results[0][1] > best_overall_matches:
        best_overall = (results[0][0], scaling, results[0][1])
        best_overall_matches = results[0][1]
    
    print()

print("="*95)
print("BEST RESULT ACROSS ALL SCALINGS")
print("="*95)
print()

if best_overall is None:
    print("No improvement found. Current model is optimal.")
    print()
    print("Conclusion:")
    print("  - Original pairing (+11/√A penalty for odd-odd) is best")
    print("  - Wigner energy (attractive np pairing) doesn't help")
    print("  - Odd-odd failures are due to OTHER physics (not pairing)")
    print()
else:
    W_best, scaling_best, matches_best = best_overall
    print(f"Best configuration:")
    print(f"  Scaling: {scaling_names[scaling_best]}")
    print(f"  W = {W_best:+.1f} MeV")
    print(f"  Matches: {matches_best}/285 ({100*matches_best/285:.1f}%)")
    print(f"  Improvement: {matches_best - 175:+d}")
    print()
    
    # Detailed analysis with best parameters
    print("="*95)
    print("DETAILED ANALYSIS WITH BEST PARAMETERS")
    print("="*95)
    print()
    
    predictions = []
    for name, Z_exp, A in test_nuclides:
        N_exp = A - Z_exp
        Z_pred = find_stable_Z(A, lambda A, Z: qfd_energy_wigner_general(A, Z, W_best, scaling_best))
        error = Z_pred - Z_exp
        
        predictions.append({
            'name': name,
            'Z_exp': Z_exp,
            'Z_pred': Z_pred,
            'error': error,
            'parity': 'even-even' if (Z_exp % 2 == 0 and N_exp % 2 == 0) else
                      'odd-odd' if (Z_exp % 2 == 1 and N_exp % 2 == 1) else 'odd-A',
        })
    
    # By parity
    print("Performance by parity:")
    print(f"{'Parity':<15} {'Baseline':<12} {'New model':<12} {'Change'}")
    print("-"*95)
    
    for parity in ['even-even', 'odd-odd', 'odd-A']:
        parity_preds = [p for p in predictions if p['parity'] == parity]
        new_correct = len([p for p in parity_preds if p['error'] == 0])
        
        # Baseline (from previous test)
        baseline_correct = {
            'even-even': int(0.59 * 166),  # 59% of 166
            'odd-odd': 2,  # 2/9
            'odd-A': int(0.682 * 110),  # 68.2% of 110
        }
        
        total = len(parity_preds)
        change = new_correct - baseline_correct[parity]
        
        marker = "★★★" if change >= 3 else "★★" if change >= 2 else "★" if change > 0 else ""
        print(f"{parity:<15} {baseline_correct[parity]:<12} {new_correct:<12} {change:+d}  {marker}")
    
    print()

print("="*95)
print("PHYSICAL INTERPRETATION")
print("="*95)
print()

print("Key insights:")
print()
print("1. If W ≈ 0: Odd-odd nuclei have NO pairing effects")
print("   → Unpaired nucleons contribute neither bonus nor penalty")
print()
print("2. If W > 0 optimal: Wigner energy exists (np pairing attractive)")
print("   → Odd-odd more stable than expected")
print()
print("3. If W < 0 optimal: Stronger penalty than current")
print("   → Odd-odd even MORE unstable than +11/√A")
print()
print("4. If no improvement: Current model is optimal")
print("   → Odd-odd failures due to OTHER physics:")
print("     - Nuclear deformation")
print("     - Shell structure")
print("     - Isospin effects")
print("     - Individual nuclear structure (not universal pairing)")
print()

print("="*95)
