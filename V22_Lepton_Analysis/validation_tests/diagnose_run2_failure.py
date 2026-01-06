#!/usr/bin/env python3
"""
Diagnose Run 2 Failure: Why did localized vortex break the fit?
"""

import numpy as np
from lepton_energy_localized_v0 import LeptonEnergyLocalized
from profile_likelihood_boundary_layer import calibrate_lambda

# Physical constants
M_E = 0.511
M_MU = 105.7

# Best fit parameters from Run 2 (FAILED)
beta = 3.0
w = 0.020
eta_target = 0.03
R_c_ref = 0.88
lam = calibrate_lambda(eta_target, beta, R_c_ref)

k_localization = 1.0
p_envelope = 8

# Fitted parameters (from log)
R_c_e, U_e, A_e = 0.5000, 0.1000, 0.7000
R_c_mu, U_mu, A_mu = 0.3000, 0.0500, 1.0000

print("=" * 70)
print("RUN 2 FAILURE DIAGNOSIS")
print("=" * 70)
print()

print("Best fit parameters (all at bounds!):")
print(f"  electron: R_c={R_c_e} (lower), U={U_e} (upper), A={A_e} (lower)")
print(f"  muon:     R_c={R_c_mu} (upper), U={U_mu} (lower), A={A_mu} (upper)")
print()

# Create energy calculator
energy_calc = LeptonEnergyLocalized(
    beta=beta,
    w=w,
    lam=lam,
    k_localization=k_localization,
    p_envelope=p_envelope,
)

# Compute energies
E_e, E_circ_e, E_stab_e, E_grad_e = energy_calc.total_energy(R_c_e, U_e, A_e)
E_mu, E_circ_mu, E_stab_mu, E_grad_mu = energy_calc.total_energy(R_c_mu, U_mu, A_mu)

print("Energy components:")
print()
print(f"{'Lepton':<10} {'E_circ':<12} {'E_stab':<12} {'E_grad':<12} {'E_total':<12}")
print("-" * 70)
print(f"{'electron':<10} {E_circ_e:<12.6f} {E_stab_e:<12.6f} {E_grad_e:<12.6f} {E_e:<12.6f}")
print(f"{'muon':<10} {E_circ_mu:<12.6f} {E_stab_mu:<12.6f} {E_grad_mu:<12.6f} {E_mu:<12.6f}")
print()

# Compute S_opt
energies = np.array([E_e, E_mu])
m_targets = np.array([M_E, M_MU])
sigma_model = 1e-4
sigma_abs = sigma_model * m_targets
weights = 1.0 / sigma_abs**2

numerator = np.sum(m_targets * energies * weights)
denominator = np.sum(energies**2 * weights)
S_opt = numerator / denominator if denominator > 0 else np.nan

print(f"Global scaling:")
print(f"  S_opt = {S_opt:.4f}")
print()

if S_opt < 0:
    print("⚠ NEGATIVE S_opt: Energies have wrong sign!")
    print()
    print("This means: m = S × E requires S < 0 to fit data")
    print("  → E_e and E_mu are both negative or near-zero")
    print()

# Model masses
masses_model = S_opt * energies
print(f"Model masses:")
print(f"  m_e,model = {masses_model[0]:.4f} MeV (target: {M_E})")
print(f"  m_μ,model = {masses_model[1]:.4f} MeV (target: {M_MU})")
print()

# χ²
residuals = (masses_model - m_targets) / sigma_abs
chi2 = np.sum(residuals**2)
print(f"χ² = {chi2:.6e}")
print()

# Diagnosis
print("=" * 70)
print("DIAGNOSIS")
print("=" * 70)
print()

if E_e < 0 or E_mu < 0:
    print("✗ NEGATIVE TOTAL ENERGIES")
    print()
    print("Root cause: Localization killed E_circ, leaving dominant -E_stab")
    print()
    print("Mechanism:")
    print(f"  1. k=1.0 → R_v = R_shell = R_c + w")
    print(f"     electron: R_v = {R_c_e + w:.3f}")
    print(f"     muon:     R_v = {R_c_mu + w:.3f}")
    print(f"  2. Exponential envelope exp[-(r/R_v)^{p_envelope}] is VERY steep")
    print(f"  3. Circulation energy suppressed to near-zero:")
    print(f"     E_circ,e = {E_circ_e:.6f} (vs E_stab,e = {E_stab_e:.6f})")
    print(f"     E_circ,μ = {E_circ_mu:.6f} (vs E_stab,μ = {E_stab_mu:.6f})")
    print(f"  4. E_total = E_circ - E_stab + E_grad < 0 (stabilization dominates)")
    print()
    print("Impact:")
    print("  • Optimizer cannot fit positive masses → pathological χ²")
    print("  • All parameters hit bounds trying to maximize E_circ or minimize E_stab")
    print("  • S_opt < 0 (nonsensical)")
    print()
    print("Conclusion: k=1.0 is TOO AGGRESSIVE for localization")
elif E_e > 0 and E_mu > 0 and E_e < 0.01 and E_mu < 1.0:
    print("✗ ENERGIES TOO SMALL")
    print()
    print("E_circ suppressed so much that E_total << m_e, m_μ")
    print(f"  E_e = {E_e:.6f} vs m_e = {M_E}")
    print(f"  E_μ = {E_mu:.6f} vs m_μ = {M_MU}")
    print()
    print("This requires huge S_opt to fit, but energies are so small that")
    print("numerical precision issues or negative values dominate.")
    print()
    print("Conclusion: k=1.0 TOO AGGRESSIVE")
else:
    print("✗ UNEXPECTED FAILURE MODE")
    print()
    print("Energies seem reasonable but fit still fails.")
    print("Further investigation needed.")

print()
print("=" * 70)
print("RECOMMENDATION")
print("=" * 70)
print()
print("Option A: Try k=1.5 (less aggressive localization)")
print("  • R_v = 1.5 × R_shell (wider envelope)")
print("  • Still passes acceptance criteria from Run 1A:")
print("    F_inner = 61.56%, ΔI/I = 3.62%")
print()
print("Option B: Try k=2.0")
print("  • Even wider: R_v = 2.0 × R_shell")
print("  • Marginal on Run 1A criteria:")
print("    F_inner = 34.48%, ΔI/I = 2.04%")
print()
print("Option C: Revert to baseline and pivot to different approach")
print("  • Localization may not be the right fix")
print("  • Consider vacuum-subtraction or other mechanisms")
print()
