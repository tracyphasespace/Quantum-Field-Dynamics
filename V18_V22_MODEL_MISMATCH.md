# CRITICAL: V18 vs V22 Model Mismatch

**Date**: December 22, 2025
**Status**: ❌ BLOCKER - Fundamental physics inconsistency
**Impact**: V22 cannot use V18 data without model reconciliation

---

## Summary

V18 and V22 implement **completely different physical models**. They cannot be directly combined without resolving the fundamental physics mismatch.

---

## V18 Model (True QFD - Static Universe)

### Distance Law
```python
# From v18/pipeline/stages/stage3_v18.py line 47-52
def qfd_distance_modulus_distance_only(z, k_J_correction):
    c = 299792.458
    k_J = 70.0 + k_J_correction
    D_mpc = z * c / k_J  # LINEAR Hubble law (static universe!)
    return 5 * np.log10(D_mpc) + 25
```

**Physics**:
- **Static Minkowski spacetime** (no expansion, no time dilation)
- **Linear distance-redshift**: D = z × c / k_J
- **k_J ≈ 70 km/s/Mpc** (quantum field drag parameter)

### Dimming Mechanism
```python
# From v18/core/v17_lightcurve_model.py
ln_A_pred(z, k_J_correction, eta_prime, xi):
    k_J_total = 70.0 + k_J_correction
    ln_A_baseline = 2.0 * jnp.log(k_J_total / 70.0)
    ln_A_anomalous = -(eta_prime * z + xi * z/(1+z))
    return ln_A_baseline + ln_A_anomalous
```

**Physics**:
- **η'**: Plasma veil opacity (charged particles around SN)
- **ξ**: Thermal processing (FDR scattering, broadening)
- **ln_A dimming**: Brightness reduction from local physics

### Best-Fit Parameters (4,885 SNe)
- k_J_correction = 19.94 ± 0.06 km/s/Mpc
- η' = -5.998 ± 0.002
- ξ = -5.997 ± 0.003
- **RMS = 2.18 mag** (15.8% better than ΛCDM)

---

## V22 Model (Hybrid - Expanding Universe + Scattering)

### Distance Law
```python
# From V22_Supernova_Analysis/scripts/v22_qfd_fit_lean_constrained.py line 71-88
def luminosity_distance_matter_only(z, H0):
    c_km_s = 299792.458
    # Einstein-de Sitter exact solution (Ω_M = 1, Ω_Λ = 0)
    d_C = (2 * c_km_s / H0) * (1 - 1 / np.sqrt(1 + z))
    d_L = (1 + z) * d_C  # EXPANDING universe!
    return d_L
```

**Physics**:
- **Expanding universe** (NOT static!)
- **Friedmann cosmology** (matter-dominated, no dark energy)
- **Non-linear distance**: D_L = (1+z) × 2c/H0 × [1 - 1/√(1+z)]

### Dimming Mechanism
```python
# From V22 line 90-116
def scattering_optical_depth(z, alpha, beta):
    return alpha * (z ** beta)

def distance_modulus_qfd(z, H0, alpha, beta):
    D_L = luminosity_distance_matter_only(z, H0)
    tau = scattering_optical_depth(z, alpha, beta)
    S = np.exp(-tau)  # Survival fraction

    mu_geometric = 5 * np.log10(D_L) + 25
    mu_scattering = -2.5 * np.log10(S)
    return mu_geometric + mu_scattering
```

**Physics**:
- **α**: Scattering coefficient
- **β**: Power law exponent (0.4 ≤ β ≤ 1.0)
- **Optical depth**: τ = α × z^β
- **Photon scattering**: Generic dimming, not tied to QFD physics

### Parameters
- H0 ∈ [50, 100] km/s/Mpc (Hubble parameter for expanding universe!)
- α ∈ (0, 2) from Lean proof (scattering coefficient)
- β ∈ [0.4, 1.0] (power law exponent)

---

## The Fundamental Contradiction

| Aspect | V18 (True QFD) | V22 (Hybrid) |
|--------|----------------|--------------|
| **Universe** | Static (Minkowski) | Expanding (Friedmann) |
| **Distance Law** | D = z·c/k_J (linear) | D_L = (1+z)·integral[dz'/E(z')] |
| **Time Dilation** | None | Yes (stretch ∝ (1+z)) |
| **Physics Framework** | QFD (quantum field drag) | Standard cosmology + scattering |
| **Dimming Cause** | Plasma veil (η'), thermal (ξ) | Generic scattering (α, z^β) |
| **Parameters** | k_J, η', ξ | H0, α, β |

**KEY PROBLEM**: V22's "QFD model" is actually **standard cosmology with ad-hoc scattering**. It contradicts the QFD framework's core premise (static universe, no expansion).

---

## Why V18 Data Can't Be Used Directly in V22

1. **Different Distance Scales**:
   - V18: D = z × c / k_J (linear)
   - V22: D_L = (1+z) × 2c/H0 × [1 - 1/√(1+z)] (non-linear)

2. **Different Calibration**:
   - V18's mu_obs calibrated to QFD model with k_J=89.94, η'=-5.998, ξ=-5.997
   - V22 needs calibration to Einstein-de Sitter + scattering with H0, α, β

3. **Incompatible Physics**:
   - V18 assumes static universe (consistent with QFD)
   - V22 assumes expanding universe (violates QFD framework)

---

## Root Cause

**V22 was NOT designed to be QFD-consistent.** Despite being called "V22 QFD Fit", it uses standard expanding universe cosmology with an ad-hoc scattering term.

**Possible Origins**:
1. V22 was meant to compare QFD vs expanding cosmology (but doesn't implement true QFD)
2. V22 was a preliminary draft using placeholder cosmology
3. Someone confused "matter-only" with "static universe" (they're not the same!)

---

## Resolution Options

### Option A: Make V22 QFD-Consistent (Recommended)
Replace V22's distance law with V18's linear QFD distance:

```python
def luminosity_distance_qfd_static(z, k_J):
    """QFD distance in static universe"""
    c_km_s = 299792.458
    return z * c_km_s / k_J  # Linear Hubble law

def distance_modulus_qfd(z, k_J, eta_prime, xi):
    """QFD with plasma veil dimming"""
    D_L = luminosity_distance_qfd_static(z, k_J)

    # Brightness dimming from QFD physics
    ln_A_pred = -(eta_prime * z + xi * z/(1+z))

    mu_geometric = 5 * np.log10(D_L) + 25
    mu_dimming = -1.0857 * ln_A_pred
    return mu_geometric + mu_dimming
```

**Lean Constraints** would then apply to:
- k_J ∈ [50, 100] km/s/Mpc (quantum field drag rate)
- η' ∈ [-10, 0] (plasma veil opacity)
- ξ ∈ [-10, 0] (thermal processing)

This would be truly "V22 QFD with Lean constraints" using V18's working physics.

### Option B: Rename V22 to "Standard Cosmology + Scattering"
Acknowledge V22 is NOT QFD. Rename and reframe as:
- "Matter-Only Cosmology with Photon Scattering"
- Comparison benchmark against true QFD (V18)
- Still valuable but different physics

### Option C: Implement Both Models in V22
Create two separate fits:
1. **QFD Model**: Static universe, k_J, η', ξ (using V18 physics)
2. **Cosmology + Scattering**: Expanding universe, H0, α, β (current V22)

Compare both against data and ΛCDM.

---

## Immediate Action Required

**BEFORE** proceeding with V22 analysis:

1. ✅ **Decide on physics model**: True QFD (static) or standard cosmology (expanding)?
2. ✅ **Update V22 code** to match chosen model
3. ✅ **Define Lean constraints** for actual model parameters
4. ✅ **Extract appropriate data** from V18 (or re-process raw data for different model)

**DO NOT** try to fit V22's expanding-universe model to V18's static-universe data. The physics is incompatible.

---

## Bottom Line

**V22 is not actually a QFD model** - it's standard expanding universe cosmology with an ad-hoc scattering term. This contradicts the QFD framework's core premise (static universe, no cosmic expansion).

We must either:
- Fix V22 to use true QFD physics (Option A), or
- Acknowledge it's a different model entirely (Option B)

**Current V18→V22 integration is blocked until this is resolved.**

---

**Status**: Awaiting decision on which physics model V22 should implement.
