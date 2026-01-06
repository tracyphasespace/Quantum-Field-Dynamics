#!/usr/bin/env python3
"""
Quick Scale Sanity Check (<1 minute)

Single point: β=3.05, w=0.015
30 iterations only
Extract: S_opt, scaled masses, residuals

Tells us immediately if we're in Case 2 (mapping issue) or Case 1A/1B.
"""

from profile_likelihood_boundary_layer import LeptonFitter, calibrate_lambda

# Test point
BETA = 3.05
W = 0.015
ETA_TARGET = 0.03
R_C_REF = 0.88

# Physical constants
M_E = 0.511
M_MU = 105.7
M_TAU = 1776.8

print("=" * 70)
print("QUICK SCALE SANITY CHECK")
print("=" * 70)
print(f"Single point: β={BETA}, w={W}")
print(f"Iterations: 30 (minimal for quick diagnosis)")
print()

# Calibrate λ
lam = calibrate_lambda(ETA_TARGET, BETA, R_C_REF)
print(f"λ = {lam:.6f} (calibrated from η={ETA_TARGET})")
print()

# Run minimal fit
fitter = LeptonFitter(beta=BETA, w=W, lam=lam, sigma_model=1e-4)
result = fitter.fit(max_iter=30, seed=42)

# Extract key metrics
chi2 = result["chi2"]
S_opt = result["S_opt"]
masses_model = result["masses_model"]
masses_target = result["masses_target"]

print("Results:")
print("-" * 70)
print(f"χ² = {chi2:.6f}")
print(f"S_opt = {S_opt:.6e}")
print()

print("Scaled masses vs targets:")
print(f"  Lepton    Model (MeV)   Target (MeV)   Residual      Rel. Error")
print(f"  ------    -----------   ------------   --------      ----------")

leptons = ["e", "μ", "τ"]
for i, (lep, m_model, m_target) in enumerate(zip(leptons, masses_model, masses_target)):
    residual = m_model - m_target
    rel_error = residual / m_target
    print(f"  {lep:6s}    {m_model:11.3f}   {m_target:12.3f}   {residual:+8.3f}      {rel_error:+.6f}")

print()
print("-" * 70)
print()

# Diagnosis
print("DIAGNOSIS:")
print()

if 0.1 < S_opt < 100:
    print(f"✓ S_opt scale reasonable (O(1-10))")
else:
    print(f"⚠ S_opt scale unusual: {S_opt:.2e}")
    print(f"  → Check units/normalization")

print()

if chi2 < 100:
    print(f"✓ χ² ~ O(1-100): Global S profiling working")
    print(f"  → Case 1A or 1B (not Case 2)")
elif chi2 < 10000:
    print(f"~ χ² ~ O(100-10⁴): Reasonable but high")
    print(f"  → Structure partially correct")
elif chi2 > 1e6:
    print(f"✗ χ² ~ 10⁶⁺: Still pathological")
    print(f"  → Case 2: Debug scale/σ/residual definition")

print()

max_rel_error = max(abs((m_model - m_target) / m_target) for m_model, m_target in zip(masses_model, masses_target))
print(f"Max relative error: {max_rel_error:.2%}")

if max_rel_error < 0.01:
    print("  → Excellent fit")
elif max_rel_error < 0.1:
    print("  → Good fit (within 10%)")
else:
    print("  → Poor fit (>10% error)")

print()
print("=" * 70)
print("Quick check complete. If χ² < 100 and errors < 10%:")
print("  → Full 3×3 will complete successfully")
print("If χ² > 10⁶:")
print("  → Debug before waiting for full scan")
print("=" * 70)
