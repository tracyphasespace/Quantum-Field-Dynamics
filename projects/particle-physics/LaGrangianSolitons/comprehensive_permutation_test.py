#!/usr/bin/env python3
"""
COMPREHENSIVE PERMUTATION TEST
===========================================================================
Systematically test ALL combinations of parameters with new optimal bonus=0.30

OBVIOUS PERMUTATIONS:
1. bonus_strength ∈ {0.30, 0.50, 0.70}
2. κ_vortex ∈ {-0.02, -0.01, 0.0, +0.01, +0.02}
3. κ_e ∈ {-0.0002, 0.0, +0.0001, +0.0002}
4. shield_factor ∈ {0.30, 0.40, 0.52, 0.60}

NON-OBVIOUS PERMUTATIONS:
5. Bonus on NON-magic (anti-magic bonus)
6. Surface term sign flip
7. Asymmetry term sign flip
8. Volume term sign flip
9. Combined vortex + electron effects
10. Shell-weighted vortex with new bonus

Total combinations: ~1000+ configurations
Test top performers only to save time.
===========================================================================
"""

import numpy as np
from itertools import product

# Constants
alpha_fine   = 1.0 / 137.036
beta_vacuum  = 1.0 / 3.058231
M_proton     = 938.272
LAMBDA_TIME_0 = 0.42

ISOMER_NODES = {2, 8, 20, 28, 50, 82, 126}

def get_resonance_bonus(Z, N, E_surface, bonus_strength, anti_magic=False):
    """
    Magic number bonus with optional ANTI-MAGIC mode.

    anti_magic=False: Magic numbers get bonus (stable)
    anti_magic=True: NON-magic numbers get bonus (destabilize magic)
    """
    if anti_magic:
        # Bonus for NON-magic configurations
        bonus = E_surface * bonus_strength
        if Z in ISOMER_NODES: bonus -= E_surface * bonus_strength
        if N in ISOMER_NODES: bonus -= E_surface * bonus_strength
        return bonus
    else:
        # Standard magic bonus
        bonus = 0
        if Z in ISOMER_NODES: bonus += E_surface * bonus_strength
        if N in ISOMER_NODES: bonus += E_surface * bonus_strength
        if Z in ISOMER_NODES and N in ISOMER_NODES:
            bonus *= 1.5
        return bonus

def shell_weighted_shielding(Z):
    """Shell-weighted vortex shielding (1/n² weighting)."""
    # Simplified: K(2e, n=1), L(8e, n=2), M(18e, n=3), etc.
    Z_eff = 0

    if Z >= 1:
        Z_eff += min(Z, 2) * (1/1**2)  # K shell
    if Z > 2:
        Z_eff += min(Z-2, 8) * (1/2**2)  # L shell
    if Z > 10:
        Z_eff += min(Z-10, 18) * (1/3**2)  # M shell
    if Z > 28:
        Z_eff += min(Z-28, 32) * (1/4**2)  # N shell
    if Z > 60:
        Z_eff += (Z-60) * (1/5**2)  # O+ shells

    return Z_eff

def qfd_energy_permutation(A, Z,
                          shield_factor=0.52,
                          bonus_strength=0.30,
                          kappa_vortex=0.0,
                          kappa_e=0.0001,
                          anti_magic=False,
                          sign_surface=1.0,
                          sign_asym=1.0,
                          sign_volume=1.0,
                          shell_weighted=False):
    """
    Generalized QFD energy with ALL permutation options.
    """
    N = A - Z
    q = Z / A if A > 0 else 0

    # Temporal metric
    lambda_time = LAMBDA_TIME_0 + kappa_e * Z

    V_0 = M_proton * (1 - (alpha_fine**2) / beta_vacuum)
    beta_nuclear = M_proton * beta_vacuum / 2
    E_volume  = V_0 * (1 - lambda_time / (12 * np.pi))
    E_surface = beta_nuclear / 15
    a_sym     = (beta_vacuum * M_proton) / 15

    # Volume (with optional sign flip)
    E_bulk = sign_volume * E_volume * A

    # Surface (with optional sign flip)
    E_surf = sign_surface * E_surface * (A**(2/3))

    # Asymmetry (with optional sign flip)
    E_asym = sign_asym * a_sym * A * ((1 - 2*q)**2)

    # Displacement with vortex shielding
    a_disp_bare = (alpha_fine * 197.327 / 1.2)

    if shell_weighted:
        # Shell-weighted vortex shielding
        Z_eff = shell_weighted_shielding(Z)
        vortex_factor = 1 + kappa_vortex * Z_eff
    else:
        # Simple linear vortex shielding
        vortex_factor = 1 + kappa_vortex * Z

    shield_total = shield_factor * vortex_factor
    a_disp = a_disp_bare * shield_total
    E_vac = a_disp * (Z**2) / (A**(1/3))

    # Magic bonus (standard or anti-magic)
    E_iso = -get_resonance_bonus(Z, N, E_surface, bonus_strength, anti_magic)

    return E_bulk + E_surf + E_asym + E_vac + E_iso

def find_stable_Z_permutation(A, **kwargs):
    """Find stable Z with permutation parameters."""
    best_Z = 1
    best_E = qfd_energy_permutation(A, 1, **kwargs)
    for Z in range(1, A):
        E = qfd_energy_permutation(A, Z, **kwargs)
        if E < best_E:
            best_E = E
            best_Z = Z
    return best_Z

def evaluate_config(config, test_nuclides):
    """Evaluate a configuration and return exact match count."""
    exact = sum(1 for name, Z_exp, A in test_nuclides
                if find_stable_Z_permutation(A, **config) == Z_exp)
    return exact

# Load data
with open('qfd_optimized_suite.py', 'r') as f:
    content = f.read()
start = content.find('test_nuclides = [')
end = content.find(']', start) + 1
test_nuclides = eval(content[start:end].replace('test_nuclides = ', ''))

print("="*95)
print("COMPREHENSIVE PERMUTATION TEST")
print("="*95)
print()
print(f"Testing {len(test_nuclides)} nuclides across parameter space")
print()

# Baseline
baseline_config = {
    'shield_factor': 0.52,
    'bonus_strength': 0.30,  # NEW OPTIMAL
    'kappa_vortex': 0.0,
    'kappa_e': 0.0001,
}

baseline_exact = evaluate_config(baseline_config, test_nuclides)
print(f"NEW BASELINE (bonus=0.30): {baseline_exact}/{len(test_nuclides)} ({100*baseline_exact/len(test_nuclides):.1f}%)")
print()

# ============================================================================
# SYSTEMATIC GRID SEARCH
# ============================================================================

print("="*95)
print("GRID SEARCH: Obvious Permutations")
print("="*95)
print()

# Parameter ranges
bonus_values = [0.20, 0.30, 0.40, 0.50]
kappa_vortex_values = [-0.02, -0.01, 0.0, 0.01, 0.02]
kappa_e_values = [-0.0002, 0.0, 0.0001, 0.0002]
shield_values = [0.40, 0.52, 0.60]

best_configs = []

print("Testing parameter grid (this may take a few minutes)...")
print()

total_configs = len(bonus_values) * len(kappa_vortex_values) * len(kappa_e_values) * len(shield_values)
print(f"Total configurations: {total_configs}")
print()

config_count = 0

for bonus, kappa_v, kappa_e, shield in product(bonus_values, kappa_vortex_values,
                                                kappa_e_values, shield_values):
    config_count += 1

    config = {
        'shield_factor': shield,
        'bonus_strength': bonus,
        'kappa_vortex': kappa_v,
        'kappa_e': kappa_e,
    }

    exact = evaluate_config(config, test_nuclides)

    # Track top performers
    if exact > baseline_exact:
        best_configs.append({
            'config': config,
            'exact': exact,
            'pct': 100*exact/len(test_nuclides),
            'improvement': exact - baseline_exact
        })

    # Progress indicator
    if config_count % 20 == 0:
        print(f"  Progress: {config_count}/{total_configs} ({100*config_count/total_configs:.0f}%)")

print()
print(f"Completed {total_configs} configurations")
print()

# Report top performers
if best_configs:
    print("="*95)
    print("TOP PERFORMERS (Better than baseline)")
    print("="*95)
    print()

    # Sort by exact matches
    best_configs.sort(key=lambda x: x['exact'], reverse=True)

    print(f"{'Rank':<6} {'Exact':<15} {'Improvement':<15} {'Configuration'}")
    print("-"*95)

    for i, result in enumerate(best_configs[:20], 1):  # Top 20
        cfg = result['config']
        print(f"{i:<6} {result['exact']}/{len(test_nuclides)} ({result['pct']:.1f}%)"
              f"  +{result['improvement']:<13} "
              f"bonus={cfg['bonus_strength']:.2f}, shield={cfg['shield_factor']:.2f}, "
              f"κ_v={cfg['kappa_vortex']:+.3f}, κ_e={cfg['kappa_e']:+.4f}")

    print()

    # Best overall
    best = best_configs[0]
    print("="*95)
    print("BEST CONFIGURATION")
    print("="*95)
    print()
    print(f"Exact matches: {best['exact']}/{len(test_nuclides)} ({best['pct']:.1f}%)")
    print(f"Improvement: +{best['improvement']} ({best['improvement']/len(test_nuclides)*100:+.1f} pp)")
    print()
    print("Parameters:")
    for key, val in best['config'].items():
        print(f"  {key}: {val}")

else:
    print("No configurations beat baseline")

print()

# ============================================================================
# NON-OBVIOUS PERMUTATIONS
# ============================================================================

print("="*95)
print("NON-OBVIOUS PERMUTATIONS")
print("="*95)
print()

non_obvious = []

# 1. Anti-magic bonus
print("Testing anti-magic bonus...")
config_anti = baseline_config.copy()
config_anti['anti_magic'] = True
exact_anti = evaluate_config(config_anti, test_nuclides)
non_obvious.append(('Anti-magic bonus', exact_anti, config_anti))
print(f"  Anti-magic: {exact_anti}/{len(test_nuclides)} ({100*exact_anti/len(test_nuclides):.1f}%)  "
      f"({exact_anti - baseline_exact:+d})")

# 2. Shell-weighted vortex (with new bonus)
print("Testing shell-weighted vortex...")
for kappa_v in [-0.03, -0.02, -0.01, 0.01, 0.02, 0.03]:
    config_shell = baseline_config.copy()
    config_shell['kappa_vortex'] = kappa_v
    config_shell['shell_weighted'] = True
    exact_shell = evaluate_config(config_shell, test_nuclides)

    if exact_shell > baseline_exact:
        non_obvious.append((f'Shell-weighted κ_v={kappa_v:.3f}', exact_shell, config_shell))
        print(f"  Shell-weighted κ_v={kappa_v:+.3f}: {exact_shell}/{len(test_nuclides)} "
              f"({100*exact_shell/len(test_nuclides):.1f}%)  ({exact_shell - baseline_exact:+d})")

# 3. Negative surface energy
print("Testing surface sign flip...")
config_surf = baseline_config.copy()
config_surf['sign_surface'] = -1.0
exact_surf = evaluate_config(config_surf, test_nuclides)
non_obvious.append(('Negative surface', exact_surf, config_surf))
print(f"  Negative surface: {exact_surf}/{len(test_nuclides)} ({100*exact_surf/len(test_nuclides):.1f}%)  "
      f"({exact_surf - baseline_exact:+d})")

# 4. Negative asymmetry
print("Testing asymmetry sign flip...")
config_asym = baseline_config.copy()
config_asym['sign_asym'] = -1.0
exact_asym = evaluate_config(config_asym, test_nuclides)
non_obvious.append(('Negative asymmetry', exact_asym, config_asym))
print(f"  Negative asymmetry: {exact_asym}/{len(test_nuclides)} ({100*exact_asym/len(test_nuclides):.1f}%)  "
      f"({exact_asym - baseline_exact:+d})")

# 5. Combined vortex + strong electron
print("Testing combined vortex + electron...")
for kappa_v, kappa_e in [(-0.02, 0.0002), (-0.01, 0.0002), (0.01, 0.0002)]:
    config_comb = baseline_config.copy()
    config_comb['kappa_vortex'] = kappa_v
    config_comb['kappa_e'] = kappa_e
    exact_comb = evaluate_config(config_comb, test_nuclides)

    if exact_comb > baseline_exact:
        non_obvious.append((f'Combined κ_v={kappa_v:.3f}, κ_e={kappa_e:.4f}',
                          exact_comb, config_comb))
        print(f"  Combined κ_v={kappa_v:+.3f}, κ_e={kappa_e:+.4f}: "
              f"{exact_comb}/{len(test_nuclides)} ({100*exact_comb/len(test_nuclides):.1f}%)  "
              f"({exact_comb - baseline_exact:+d})")

print()

# Report best non-obvious
if any(exact > baseline_exact for _, exact, _ in non_obvious):
    print("="*95)
    print("BEST NON-OBVIOUS PERMUTATION")
    print("="*95)
    print()

    best_non_obvious = max(non_obvious, key=lambda x: x[1])
    name, exact, config = best_non_obvious

    print(f"{name}:")
    print(f"  Exact: {exact}/{len(test_nuclides)} ({100*exact/len(test_nuclides):.1f}%)")
    print(f"  Improvement: {exact - baseline_exact:+d} ({(exact - baseline_exact)/len(test_nuclides)*100:+.1f} pp)")

print()

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("="*95)
print("FINAL SUMMARY")
print("="*95)
print()

all_results = best_configs + [('Non-obvious: ' + name, exact, config)
                              for name, exact, config in non_obvious]

if all_results:
    # Find absolute best
    best_all = max(all_results, key=lambda x: x[1] if isinstance(x, tuple) else x['exact'])

    if isinstance(best_all, dict):
        best_exact = best_all['exact']
        best_config = best_all['config']
        source = "Grid search"
    else:
        name, best_exact, best_config = best_all
        source = name

    print(f"Baseline (bonus=0.30): {baseline_exact}/{len(test_nuclides)} ({100*baseline_exact/len(test_nuclides):.1f}%)")
    print(f"Best found: {best_exact}/{len(test_nuclides)} ({100*best_exact/len(test_nuclides):.1f}%)")
    print()
    print(f"Total improvement: +{best_exact - baseline_exact} matches "
          f"({(best_exact - baseline_exact)/len(test_nuclides)*100:+.1f} pp)")
    print()
    print(f"Source: {source}")
    print()
    print("Best configuration:")
    for key, val in best_config.items():
        print(f"  {key}: {val}")

    if best_exact > baseline_exact:
        print()
        print("✓✓ IMPROVEMENT FOUND through permutation testing!")

else:
    print("No permutations improved over baseline bonus=0.30")
    print()
    print("This suggests bonus=0.30 with default parameters is locally optimal.")

print()
print("="*95)
