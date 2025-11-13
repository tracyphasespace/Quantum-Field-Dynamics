# QFD Supernova Model V15 - Physical Architecture

**Date:** November 13, 2024
**Status:** 2-Parameter Model (k_J FIXED)

## Executive Summary

V15 fits ONLY the anomalous supernova dimming (~0.5 mag at z=0.5) from local, near-source effects. The baseline Hubble Law is already explained by the QVD redshift model and is NOT fitted.

## Physical Components

### 1. Baseline Cosmological Redshift (NOT FITTED)

**k_J = 70.0 km/s/Mpc (FIXED)**

- Source: QVD redshift model (RedShift directory)
- Parameters: α_QVD = 0.85, β = 0.6
- Physical mechanism: Cosmic drag from photon-ψ field interactions
- Formula: z_cosmo = (k_J/c) × D
- Result: H₀ ≈ 70 km/s/Mpc validated separately

**Critical:** This is NOT a free parameter. The baseline Hubble Law is predetermined by QVD cosmology.

### 2. Anomalous Dimming Components (FITTED)

V15 fits TWO global parameters that describe universal strengths of local, near-source effects:

#### η' (eta_prime): Plasma Veil Parameter
- **Physical mechanism:** Scattering/reprocessing in dense SN ejecta (z_plasma stage)
- **Distance dependence:** NONE - This is a local, near-source process
- **What it measures:** Universal strength of plasma veil effect
- **Per-SN variation:** Captured by Stage 1 A_plasma parameter
- **Environmental dependence:** Ejecta density, composition

#### ξ (xi): Flux-Dependent Redshift (FDR) / "Sear" Parameter
- **Physical mechanism:** Photon interactions during intense flux pulse (z_FDR stage)
- **Distance dependence:** NONE - This is a local, near-source process
- **What it measures:** Universal strength of FDR/sear effect
- **Per-SN variation:** Captured by Stage 1 beta, ln_A parameters
- **Environmental dependence:** Photon pulse intensity, field configuration

**Model Expression:**
```
ln_A_pred(z, η', ξ) = -(η' × z + ξ × z/(1+z))
```

The z-dependence in the basis functions does NOT mean these are distance-dependent effects. It allows the MCMC to separate:
- **Trend with z:** From Planck/Wien K-correction (observational effect)
- **Scatter around trend:** From environmental variations in η' and ξ strength

### 3. Planck/Wien Broadening (NOT A FITTED PARAMETER)

**What it is:** Observational K-correction effect, NOT a separate physical mechanism

**Physical basis:**
- Observing redshifted thermal spectrum with fixed filters
- V-band at high-z samples rest-frame B/U-band emission
- SNe are intrinsically less luminous in U-band → apparent dimming

**Effect on observations:**
1. **Brightness (K-correction):** Automatic dimming trend with z
2. **Light curve shape:** Blue light curves are faster than red ones
   - High-z V-band sees redshifted blue emission → faster intrinsic evolution
   - Corrects standard (1+z) time dilation prediction

**How it's handled:**
- NOT a fitted parameter
- Automatically captured in z-dependence of model
- Creates the smooth TREND with redshift in the data

## How Components Separate in MCMC

The MCMC fitter exploits different mathematical dependencies:

### Redshift-Correlated Trend
- Source: Planck/Wien K-correction + fixed k_J baseline
- Mathematical form: Smooth function of z
- Fitted by: Global η', ξ parameters defining trend shape

### Residual Scatter
- Source: Environmental variations in Veil + Sear strength
- Mathematical form: Per-SN deviations from trend
- Captured by: Stage 1 per-SN parameters (A_plasma, beta, ln_A)
- Physical interpretation: Different local conditions at each SN

**Key insight:** The scatter is NOT measurement noise - it's real physics signal of local plasma interactions.

## Distinction from Standard Cosmology

### Standard Model (ΛCDM + SALT2)
- Treats scatter as nuisance parameter
- "Corrects" all SNe to single standard candle using stretch/color
- Attributes smooth trend to Dark Energy acceleration
- Hubble Law slope is fitted parameter

### QFD V15 Model
- Scatter is REAL PHYSICS signal of local effects
- Models environmental variations explicitly
- Attributes trend to known K-correction physics
- Hubble Law slope (k_J) is predetermined by QVD model
- No Dark Energy needed

## Model Assumptions (V15 Preliminary)

1. **Progenitor system:** 2-WD barycentric mass
2. **Compact object:** Small black hole present
3. **Light curve broadening:** Planck/Wien thermal effects (NOT ΛCDM time dilation)
4. **BBH orbital lensing:** Deferred to V16 (applied to outliers only)

## Global Parameters Summary

**3-Parameter Model (OLD - INCORRECT):**
- (k_J, η', ξ) - ALL fitted
- Problem: Attempted to fit ENTIRE Hubble Law

**2-Parameter Model (CURRENT - CORRECT):**
- k_J = 70.0 km/s/Mpc (FIXED from QVD)
- (η', ξ) - ONLY fitted parameters
- Fits ONLY anomalous dimming (~0.5 mag at z=0.5)

## Implementation

**Modified files:**
- `core/v15_model.py`: K_J_BASELINE = 70.0, ln_A_pred(z, eta_prime, xi)
- `stages/stage2_simple.py`: 2-parameter MCMC, feature matrix [N, 2]

**Feature matrix (Stage 2):**
```python
Φ = [z, z/(1+z)]  # Shape: [N, 2]
```

**Back-transformation:**
```python
eta_prime = c[0] / scale[0]
xi = c[1] / scale[1]
```

## Expected Results

**Convergence:**
- Better MCMC mixing (lower dimensional parameter space)
- ~10-30% faster sampling
- More stable parameter estimates

**Physics:**
- η' describes universal plasma veil strength
- ξ describes universal FDR/sear strength
- Per-SN scatter shows environmental variations
- No need for Dark Energy component

## References

- QVD Redshift Model: RedShift directory (α_QVD = 0.85, β = 0.6)
- Previous 3-param results: November 5, 2024 (k_J = 10.770 ± 4.567)
- K-correction physics: Standard observational cosmology

---

**Note:** This document reflects the understanding as of November 13, 2024, following clarification of the distinction between distance-dependent (k_J, Planck/Wien) and distance-independent (Veil, Sear) effects.
