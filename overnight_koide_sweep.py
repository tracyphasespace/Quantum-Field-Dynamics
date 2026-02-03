#!/usr/bin/env python3
"""Koide delta parameter sweep for falsifiability test

Tests whether delta = 3.043233053 rad is uniquely identified or just one of many values.
Sweeps delta from 2.9 to 3.2 rad and computes Q ratio and mass predictions.

Expected runtime: ~30 minutes
"""
import numpy as np
import json
from datetime import datetime

def geometric_mass(k, mu, delta):
    """
    Koide geometric mass formula.

    Args:
        k: Generation index (0=electron, 1=muon, 2=tau)
        mu: Mass scale parameter (MeV)
        delta: Phase angle (radians)

    Returns:
        Mass in MeV
    """
    angle = delta + k * (2 * np.pi / 3)
    term = 1 + np.sqrt(2) * np.cos(angle)
    return mu * term**2


def koide_ratio(m_e, m_mu, m_tau):
    """
    Compute Koide Q ratio.

    Q = (m_e + m_mu + m_tau) / (sqrt(m_e) + sqrt(m_mu) + sqrt(m_tau))^2

    Should equal 2/3 if Koide relation holds.
    """
    numerator = m_e + m_mu + m_tau
    denominator = (np.sqrt(m_e) + np.sqrt(m_mu) + np.sqrt(m_tau))**2
    return numerator / denominator


# Experimental values (PDG 2024)
M_E_EXP = 0.5109989461     # MeV
M_MU_EXP = 105.6583745     # MeV
M_TAU_EXP = 1776.86        # MeV

# Delta sweep parameters
DELTA_MIN = 2.9
DELTA_MAX = 3.2
N_POINTS = 61

print("=" * 80)
print("KOIDE DELTA PARAMETER SWEEP")
print("=" * 80)
print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"\nConfiguration:")
print(f"  Delta range: {DELTA_MIN:.3f} to {DELTA_MAX:.3f} rad")
print(f"  Number of points: {N_POINTS}")
print(f"  Step size: {(DELTA_MAX - DELTA_MIN)/(N_POINTS-1):.5f} rad")
print(f"\nExperimental values:")
print(f"  m_e   = {M_E_EXP:.10f} MeV")
print(f"  m_mu  = {M_MU_EXP:.7f} MeV")
print(f"  m_tau = {M_TAU_EXP:.2f} MeV")
print(f"\nTarget Q ratio: {2/3:.10f}")
print()

# Delta sweep
delta_values = np.linspace(DELTA_MIN, DELTA_MAX, N_POINTS)
results = []

print("Progress:")
for i, delta in enumerate(delta_values):
    # Fit mu from electron mass (fixes electron mass exactly)
    mu = M_E_EXP / geometric_mass(0, 1.0, delta)

    # Compute predictions
    m_e = geometric_mass(0, mu, delta)
    m_mu = geometric_mass(1, mu, delta)
    m_tau = geometric_mass(2, mu, delta)

    # Koide Q ratio
    Q = koide_ratio(m_e, m_mu, m_tau)

    # Compute chi-squared (relative errors)
    chi2 = ((m_e - M_E_EXP)/M_E_EXP)**2 + \
           ((m_mu - M_MU_EXP)/M_MU_EXP)**2 + \
           ((m_tau - M_TAU_EXP)/M_TAU_EXP)**2

    # Store result
    results.append({
        'delta': float(delta),
        'delta_deg': float(delta * 180 / np.pi),
        'mu': float(mu),
        'Q': float(Q),
        'Q_error': float(abs(Q - 2/3)),
        'chi2': float(chi2),
        'm_e': float(m_e),
        'm_mu': float(m_mu),
        'm_tau': float(m_tau),
        'res_e': float(m_e - M_E_EXP),
        'res_mu': float(m_mu - M_MU_EXP),
        'res_tau': float(m_tau - M_TAU_EXP),
        'rel_err_mu': float((m_mu - M_MU_EXP)/M_MU_EXP),
        'rel_err_tau': float((m_tau - M_TAU_EXP)/M_TAU_EXP)
    })

    if (i+1) % 10 == 0:
        print(f"  {i+1:3d}/{N_POINTS} completed ({100*(i+1)/N_POINTS:.0f}%)")

print(f"  {N_POINTS}/{N_POINTS} completed (100%)")
print()

# Find minimum chi2
min_idx = min(range(len(results)), key=lambda i: results[i]['chi2'])
best = results[min_idx]

# Find where Q is closest to 2/3
q_min_idx = min(range(len(results)), key=lambda i: results[i]['Q_error'])
q_best = results[q_min_idx]

# Find delta = 3.043233053 if in range
target_delta = 3.043233053
target_idx = min(range(len(results)), key=lambda i: abs(results[i]['delta'] - target_delta))
at_target = results[target_idx]

print("=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)
print()

print("Best chi-squared fit (minimizes mass errors):")
print(f"  delta     = {best['delta']:.6f} rad ({best['delta_deg']:.3f}°)")
print(f"  mu        = {best['mu']:.6f} MeV")
print(f"  Q         = {best['Q']:.10f}")
print(f"  |Q - 2/3| = {best['Q_error']:.2e}")
print(f"  chi2      = {best['chi2']:.2e}")
print(f"  m_mu  err = {best['rel_err_mu']:+.2e} ({100*best['rel_err_mu']:+.4f}%)")
print(f"  m_tau err = {best['rel_err_tau']:+.2e} ({100*best['rel_err_tau']:+.4f}%)")
print()

print("Best Q ratio fit (Q closest to 2/3):")
print(f"  delta     = {q_best['delta']:.6f} rad ({q_best['delta_deg']:.3f}°)")
print(f"  Q         = {q_best['Q']:.10f}")
print(f"  |Q - 2/3| = {q_best['Q_error']:.2e}")
print(f"  chi2      = {q_best['chi2']:.2e}")
print()

if abs(at_target['delta'] - target_delta) < 0.001:
    print(f"At delta = 3.043233053 rad:")
    print(f"  Q         = {at_target['Q']:.10f}")
    print(f"  |Q - 2/3| = {at_target['Q_error']:.2e}")
    print(f"  chi2      = {at_target['chi2']:.2e}")
    print(f"  Rank      = {min_idx - target_idx:+d} from minimum")
    print()

# Analyze landscape
chi2_values = [r['chi2'] for r in results]
chi2_min = min(chi2_values)
chi2_max = max(chi2_values)
chi2_range = chi2_max - chi2_min

# Count points within 10% of minimum
threshold = chi2_min * 1.1
good_points = sum(1 for chi2 in chi2_values if chi2 <= threshold)

print("Landscape Analysis:")
print(f"  chi2 range   = [{chi2_min:.2e}, {chi2_max:.2e}]")
print(f"  Dynamic range = {chi2_max/chi2_min:.1e}")
print(f"  Points within 10% of min = {good_points}/{N_POINTS} ({100*good_points/N_POINTS:.0f}%)")
print()

if good_points <= 3:
    print("  → SHARP MINIMUM: delta is well-identified ✓")
elif good_points <= 10:
    print("  → MODERATE: delta has some constraint")
else:
    print("  → BROAD/FLAT: delta is poorly constrained ⚠")
print()

# Save results
output = {
    'metadata': {
        'timestamp': datetime.now().isoformat(),
        'sweep_range_rad': [float(DELTA_MIN), float(DELTA_MAX)],
        'n_points': N_POINTS,
        'experimental_masses': {
            'm_e': M_E_EXP,
            'm_mu': M_MU_EXP,
            'm_tau': M_TAU_EXP
        }
    },
    'best_chi2_fit': best,
    'best_Q_fit': q_best,
    'at_delta_3058': at_target if abs(at_target['delta'] - target_delta) < 0.001 else None,
    'landscape': {
        'chi2_min': float(chi2_min),
        'chi2_max': float(chi2_max),
        'dynamic_range': float(chi2_max/chi2_min),
        'points_within_10pct': int(good_points)
    },
    'all_results': results
}

output_file = 'koide_delta_sweep_results.json'
with open(output_file, 'w') as f:
    json.dump(output, f, indent=2)

print(f"Results saved to: {output_file}")
print()

print("=" * 80)
print("ANALYSIS RECOMMENDATIONS")
print("=" * 80)
print()
print("To visualize results:")
print()
print("  import json")
print("  import matplotlib.pyplot as plt")
print()
print("  with open('koide_delta_sweep_results.json') as f:")
print("      data = json.load(f)")
print()
print("  deltas = [r['delta'] for r in data['all_results']]")
print("  chi2s = [r['chi2'] for r in data['all_results']]")
print("  Qs = [r['Q'] for r in data['all_results']]")
print()
print("  fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))")
print("  ax1.semilogy(deltas, chi2s, 'b-', lw=2)")
print("  ax1.axvline(3.043233053, color='r', ls='--', label='delta=3.043233053')")
print("  ax1.set_ylabel('chi^2')")
print("  ax1.legend()")
print("  ax1.grid(True)")
print()
print("  ax2.plot(deltas, Qs, 'g-', lw=2)")
print("  ax2.axhline(2/3, color='k', ls='--', label='Q=2/3')")
print("  ax2.set_xlabel('delta (rad)')")
print("  ax2.set_ylabel('Q ratio')")
print("  ax2.legend()")
print("  ax2.grid(True)")
print()
print("  plt.tight_layout()")
print("  plt.savefig('koide_delta_landscape.png', dpi=150)")
print("  print('Saved: koide_delta_landscape.png')")
print()

print("=" * 80)
print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
