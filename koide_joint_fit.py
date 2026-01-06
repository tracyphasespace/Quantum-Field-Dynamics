#!/usr/bin/env python3
"""
Koide delta and mu joint optimization

Find (delta, mu) that best fit all three lepton masses simultaneously.
This is the CORRECT approach - not fixing mu from electron alone.
"""
import numpy as np
from scipy.optimize import differential_evolution, minimize
import json
from datetime import datetime

def geometric_mass(k, mu, delta):
    """Koide geometric mass formula."""
    angle = delta + k * (2 * np.pi / 3)
    term = 1 + np.sqrt(2) * np.cos(angle)
    return mu * term**2

def koide_ratio(m_e, m_mu, m_tau):
    """Koide Q ratio."""
    numerator = m_e + m_mu + m_tau
    denominator = (np.sqrt(m_e) + np.sqrt(m_mu) + np.sqrt(m_tau))**2
    return numerator / denominator

# Experimental values
M_E_EXP = 0.5109989461
M_MU_EXP = 105.6583745
M_TAU_EXP = 1776.86

def objective(params):
    """
    Objective function: minimize chi-squared error across all three masses.

    params = [mu, delta]
    """
    mu, delta = params

    # Predict masses
    m_e = geometric_mass(0, mu, delta)
    m_mu = geometric_mass(1, mu, delta)
    m_tau = geometric_mass(2, mu, delta)

    # Chi-squared (relative errors)
    chi2 = ((m_e - M_E_EXP)/M_E_EXP)**2 + \
           ((m_mu - M_MU_EXP)/M_MU_EXP)**2 + \
           ((m_tau - M_TAU_EXP)/M_TAU_EXP)**2

    return chi2

print("=" * 80)
print("KOIDE JOINT FIT: Optimize (mu, delta) simultaneously")
print("=" * 80)
print()

# Optimize using differential evolution (global optimizer)
print("Running global optimization (differential evolution)...")
print("This may take 1-2 minutes...")
print()

bounds = [
    (0.1, 10.0),      # mu in MeV (reasonable range)
    (2.5, 3.5)        # delta in radians (around pi)
]

result = differential_evolution(
    objective,
    bounds,
    seed=42,
    maxiter=500,
    popsize=30,
    atol=1e-12,
    tol=1e-12
)

mu_opt, delta_opt = result.x
chi2_opt = result.fun

# Compute predictions
m_e = geometric_mass(0, mu_opt, delta_opt)
m_mu = geometric_mass(1, mu_opt, delta_opt)
m_tau = geometric_mass(2, mu_opt, delta_opt)

Q = koide_ratio(m_e, m_mu, m_tau)

print("=" * 80)
print("OPTIMAL SOLUTION")
print("=" * 80)
print()
print(f"Optimized parameters:")
print(f"  mu    = {mu_opt:.10f} MeV")
print(f"  delta = {delta_opt:.10f} rad ({delta_opt * 180/np.pi:.6f}°)")
print()
print(f"Predicted masses:")
print(f"  m_e   = {m_e:.10f} MeV (exp: {M_E_EXP:.10f})")
print(f"  m_mu  = {m_mu:.10f} MeV (exp: {M_MU_EXP:.7f})")
print(f"  m_tau = {m_tau:.6f} MeV (exp: {M_TAU_EXP:.2f})")
print()
print(f"Residuals:")
print(f"  Δm_e   = {m_e - M_E_EXP:+.2e} MeV ({100*(m_e - M_E_EXP)/M_E_EXP:+.6f}%)")
print(f"  Δm_mu  = {m_mu - M_MU_EXP:+.2e} MeV ({100*(m_mu - M_MU_EXP)/M_MU_EXP:+.6f}%)")
print(f"  Δm_tau = {m_tau - M_TAU_EXP:+.2e} MeV ({100*(m_tau - M_TAU_EXP)/M_TAU_EXP:+.6f}%)")
print()
print(f"Koide Q ratio:")
print(f"  Q         = {Q:.10f}")
print(f"  Target    = {2/3:.10f}")
print(f"  |Q - 2/3| = {abs(Q - 2/3):.2e}")
print()
print(f"Fit quality:")
print(f"  chi2 = {chi2_opt:.2e}")
print()

# Check how close to delta = 3.058
delta_target = 3.058
delta_diff = delta_opt - delta_target
print(f"Comparison to delta = 3.058:")
print(f"  Δdelta = {delta_diff:+.6f} rad ({delta_diff * 180/np.pi:+.4f}°)")
if abs(delta_diff) < 0.01:
    print(f"  → MATCHES claimed value! ✓")
elif abs(delta_diff) < 0.05:
    print(f"  → Close to claimed value")
else:
    print(f"  → DIFFERS significantly from 3.058 ⚠")
print()

# Save results
output = {
    'timestamp': datetime.now().isoformat(),
    'optimization': {
        'method': 'differential_evolution',
        'converged': result.success,
        'iterations': result.nit,
        'function_calls': result.nfev
    },
    'optimal_parameters': {
        'mu': float(mu_opt),
        'delta_rad': float(delta_opt),
        'delta_deg': float(delta_opt * 180/np.pi)
    },
    'predictions': {
        'm_e': float(m_e),
        'm_mu': float(m_mu),
        'm_tau': float(m_tau),
        'Q': float(Q)
    },
    'fit_quality': {
        'chi2': float(chi2_opt),
        'residuals': {
            'm_e': float(m_e - M_E_EXP),
            'm_mu': float(m_mu - M_MU_EXP),
            'm_tau': float(m_tau - M_TAU_EXP)
        },
        'relative_errors': {
            'm_e': float((m_e - M_E_EXP)/M_E_EXP),
            'm_mu': float((m_mu - M_MU_EXP)/M_MU_EXP),
            'm_tau': float((m_tau - M_TAU_EXP)/M_TAU_EXP)
        }
    },
    'comparison_to_3058': {
        'delta_target': 3.058,
        'delta_difference': float(delta_diff),
        'delta_diff_degrees': float(delta_diff * 180/np.pi)
    }
}

with open('koide_joint_fit_results.json', 'w') as f:
    json.dump(output, f, indent=2)

print("Results saved to: koide_joint_fit_results.json")
print()
print("=" * 80)
print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
