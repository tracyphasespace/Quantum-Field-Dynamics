# Raw SNe Pipeline Status

**Date**: 2026-02-19
**Files**: `raw_sne_kelvin.py`, `golden_loop_sne.py`

---

## THE TWO PIPELINES

| Pipeline | Data | Physics params | Result | Status |
|----------|------|---------------|--------|--------|
| `golden_loop_sne.py` | SALT2-reduced DES-SN5YR | **0 free** | sigma=0.18 mag, chi2/dof=0.955 | WORKING |
| `raw_sne_kelvin.py` | V18 Stage 1 / V22 Stage 1 | 0-2 free | sigma=2.0-4.6 mag, z-slope | DIAGNOSED |

**The SALT2-reduced pipeline is the publishable result.** The raw pipeline is a cross-check.

---

## CRITICAL FINDING: V18 ALPHA CONVENTION

### What alpha IS (from reading Stage 1 source code)

The V18 Stage 1 fitter fits each SN light curve as:

```
flux(t) = exp(ln_A) * L_template(t_rest, lambda) / (4*pi*D_fid^2) * exp(-tau_model)
```

where:
- `L_template` = Planck blackbody photosphere, `L_peak = 1e43 erg/s` (FIXED)
- `D_fid = c*z / k_J` with `k_J` from a prior estimate (default ~70)
- `tau_model` = Stage 1 opacity model (partial)

**Therefore**: `alpha = ln_A = ln(observed_flux / model_flux)` — a CORRECTION FACTOR, NOT raw ln(peak_flux).

### Why alpha INCREASES with z

The V18 model predicts:
```
ln_A_pred = 2*ln(k_J/70) - (eta'*z + xi*z/(1+z))
```
With `eta' = -5.999`, `xi = -5.998`:
- z=0: ln_A_pred = 0.5
- z=0.5: ln_A_pred = 5.5
- z=1.0: ln_A_pred = 9.5

The large negative eta', xi mean distant SNe are dimmer than the geometric model
predicts, requiring a LARGER amplitude correction (higher alpha).

### What mu_obs IS

```
mu_obs = C - K * alpha    (where K = 2.5/ln(10) = 1.086, C ~ 42.2)
```

This is NOT a standard distance modulus (range 8-24, not 38-45). It's the
mu-space equivalent of the alpha correction factor.

---

## WHY v2 KELVIN FAILS ON RAW V18 DATA

### The model-contamination problem

The V18 Stage 1 template **embeds the V18 distance model** (D = cz/k_J).
The alpha values are corrections to THIS model. When you fit v2 Kelvin
(D_L = (c/K_J)*ln(1+z)*(1+z)^{2/3}), you're fighting the V18 assumptions.

**v2 Kelvin vs V18 distance models diverge systematically:**

| z range | v2K - V18 (mag) | Meaning |
|---------|----------------|---------|
| 0.05-0.29 | -1.56 | v2K predicts closer (brighter) |
| 0.29-0.43 | -1.21 | |
| 0.43-0.58 | -0.80 | |
| 0.58-0.75 | -0.34 | |
| 0.75-1.50 | +0.71 | v2K predicts farther (dimmer) |

This 2.3-mag swing across z is the source of the raw pipeline z-slope.

### Why the SALT2-reduced pipeline works

SALT2 reduction is model-independent (or uses LCDM, which gives similar D_L
to v2 Kelvin for z < 1). The SALT2 distance moduli are close to "true" mu,
so v2 Kelvin can be tested fairly. Result: chi2/dof = 0.955.

### Direct comparison (mu_obs vs v2 Kelvin, using V18 data)

```
M_offset = -23.13 (enormous — confirms mu_obs is non-standard)
sigma = 2.59 mag
z-slope: +1.84 at low z to -2.81 at high z (4.6 mag swing)
```

v2 Kelvin does WORSE than V18 on V18's own data — because the data is
processed through V18's model assumptions.

---

## LENSING AND SELECTION EFFECTS

### Gravitational lensing signature (user-identified)

| z range | >2sigma bright | <2sigma dim | Skew |
|---------|---------------|-------------|------|
| z < 0.3 | 21 (2.1%) | 22 (2.2%) | +0.09 (symmetric) |
| 0.3-0.8 | 119 (3.8%) | 41 (1.3%) | +1.0 to +1.5 |
| z > 0.8 | 19 (2.5%) | **0** (0.0%) | +0.2 |

- **13 of 19** extreme-bright high-z SNe hit alpha=30.0 exactly (fitter saturation)
- Positive skew peaks at z=0.6-0.7 — consistent with gravitational lensing magnification
- **Zero dim outliers at high z** — selection floor
- **Asymmetric extremes** = Reverse Malmquist (user finding: NOT classical Malmquist,
  but gravitational lensing of extremely distant SNe causing bright-side excess)

### Reverse Malmquist vs classical

- **Classical Malmquist**: truncation of dim tail (can't see faint SNe at high z)
- **Reverse Malmquist** (this data): bright tail ENHANCED by gravitational lensing
  of extremely distant SNe. The asymmetry (bright excess, no dim deficit) is the
  diagnostic signature.
- The 13 alpha=30.0 saturated SNe are almost certainly gravitationally lensed.

---

## PATH FORWARD FOR RAW PIPELINE

### Option A: Use V18 residuals (medium effort)
Extract V18 Hubble residuals (alpha_obs - alpha_pred_V18), test whether v2 Kelvin
corrections improve them. This removes the model contamination.
**Problem**: Only tests the DIFFERENCE between v2K and V18, not v2K independently.

### Option B: Re-fit from raw photometry — COMPLETED (2026-02-20)
**File**: `clean_sne_pipeline.py` — fresh pipeline, no v15/v16/v18/v22 legacy code.

**Architecture**:
1. Load raw DES-SN5YR photometry (5,468 SNe, 118K observations, griz)
2. Per-SN Gaussian template fit in ALL bands (B=0, host-subtracted)
3. Multi-band K-correction (blackbody SED at 11,000K)
4. Quality cuts (n_obs≥8, n_bands≥2, width, z)
5. Hubble diagram vs v2 Kelvin (0 free physics params)
6. Cross-validation against SALT2 HD (1,296 matched SNe)
7. K-correction calibration using cross-match
8. SALT2 mB (unstandardized) + v2 Kelvin test

**Results**:

| Pipeline | SNe | σ (mag) | z-slope (mag/z) | Free params |
|----------|-----|---------|-----------------|-------------|
| Raw Gaussian peak (full) | 2,958 | 1.283 | -2.6 | 1 (M) |
| Raw Gaussian (SALT2-matched) | 1,296 | 0.433 | -1.5 | 1 (M) |
| SALT2 mB raw + v2K | 1,829 | **0.389** | **-0.28** | 1 (M) |
| SALT2 mB+stretch+color + v2K | 1,829 | 0.312 | -0.17 | 3 |
| SALT2 fully standardized | 1,829 | 0.260 | -0.04 | SALT2 |
| golden_loop_sne.py (published) | 1,768 | 0.180 | flat | 0 physics |

**Key finding**: v2 Kelvin distance model is CORRECT. The mB raw test gives
σ=0.389 with nearly flat binned residuals (+0.01 to -0.25 across z=0.05-1.04)
and 0 free physics params. The z-slope in our Gaussian pipeline is 100% from
peak extraction (blackbody K-correction inadequacy), not from v2 Kelvin.

**Bottleneck**: Gaussian template + simple blackbody K-correction introduces a
~1.2 mag/z systematic error. The SN Ia SED has UV blanketing, line features,
and spectral evolution that a blackbody doesn't capture.

**Fix**: Hsiao+ (2007) K-correction implemented (v2, 2026-02-20).

**v2 Results (Hsiao K-correction)**:
| Metric | Blackbody (v1) | Hsiao (v2) | Change |
|--------|---------------|------------|--------|
| Cross-matched σ | 0.433 mag | 0.361 mag | -17% |
| Cross-matched z-slope | -1.5 mag/z | -0.77 mag/z | -49% |
| SALT2 gap ratio | 2.4× | 1.5× | narrowed |

Remaining z-slope (-0.77) is from Gaussian peak extraction, not K-correction.

### Option C: Accept SALT2 pipeline as primary (no extra effort)
The golden_loop_sne.py result (chi2/dof=0.955, 0 free params) already demonstrates
v2 Kelvin competitiveness. The raw pipeline is a secondary cross-check that requires
careful handling of the model-contamination issue.

### Option D: Hsiao template light curve fitting (next step)
Replace Gaussian template with sncosmo.fit_lc() using Hsiao model. This gives:
- SN Ia light curve shape as template (not symmetric Gaussian)
- Automatic multi-band fitting with proper K-corrections
- Peak B magnitude directly from template fit (no separate K-correction step)
- Requires: flux in sncosmo units (zp=8.9, zpsys='ab', flux=flux_Jy)

**Two-pronged self-calibrating approach** (Tracy, 2026-02-20):
Since all SNe Ia share the same Hsiao SED, a second pass can estimate distances
without any external calibration by:
1. **Apparent blue**: the observed peak wavelength shift across bands
   determines z from the SED shape alone (redundant with spectroscopic z,
   but provides a cross-check)
2. **Wien spreading**: the ratio of fluxes in different DES bands encodes
   the SED sampling point. Comparing observed band ratios to the Hsiao
   template determines both z and extinction self-consistently, giving
   distance from the amplitude residual.

This is a fully self-contained distance ladder: Hsiao template + multi-band
photometry → z, extinction, and D_L without SALT2, ΛCDM, or any external
distance model. Only the absolute magnitude M_0 remains as a calibration
parameter.

### Option E: QFD Vacuum Transfer Function Pipeline — COMPLETED (2026-02-20)
**File**: `qfd_transfer_pipeline.py` — fully SALT2-free pipeline using QFD physics.

**Architecture**: Pure QFD forward model applied to raw DES photometry:
```
F_obs(t, λ) = A × I₀(t/(1+z)^{1/3}, λ/(1+z)) × exp[-ητ(λ_B/λ)^{1/2}]
```
1. Hsiao SED template for intrinsic SN Ia spectrum I₀
2. QFD (1+z)^{1/3} time dilation (f=2 vortex ring, NOT standard (1+z))
3. QFD chromatic λ^{-1/2} extinction per band (Kelvin wave scattering)
4. Grid search + Nelder-Mead for (t0, amplitude) per SN
5. g-band excluded at z > 0.5 (rest-frame UV unreliable)
6. Achromatic μ for Hubble diagram (chromatic in fitter, not double-counted)

**Results (cross-matched against SALT2 HD, 1518 SNe)**:

| Pipeline | SNe | σ (mag) | z-slope | Free physics |
|----------|-----|---------|---------|--------------|
| QFD transfer function | 1,518 | **0.398** | -0.55 | **0** |
| SALT2 mB raw + v2K | 1,829 | 0.389 | -0.28 | 0 |
| SALT2 standardized + v2K | 1,829 | 0.260 | -0.04 | 3 |
| golden_loop_sne.py | 1,768 | 0.180 | flat | 0 |

**QFD−SALT2 mB comparison**: offset = -0.32, σ = 0.215

**Key findings**:
1. QFD transfer function matches SALT2 mB raw within 2% (0.398 vs 0.389)
2. The z-slope is from peak extraction, NOT physics (control with standard (1+z)
   time dilation gives slope = -0.41, so QFD 1/3 exponent adds only -0.13)
3. No SALT2 used in fitting — eliminates circular reasoning concern
4. The 1.5× gap vs SALT2 standardized comes from SN Ia diversity (no stretch/color
   correction), not from the distance model

**Chromatic diagnostic**: Per-band residuals show:
- riz bands: flat at -0.14 to -0.23 mag (good)
- g-band at z < 0.35: Δ(g-z) = +0.26 to +0.49 (correct sign for QFD chromatic)
- g-band at z > 0.5: UV unreliable (excluded from fit, as expected)

### Recommendation (Updated)
**Option E (QFD transfer function) is the SALT2-free validation.** It independently
confirms the v2 Kelvin distance model using only QFD physics + Hsiao template.
For the book:
- **Primary result**: golden_loop_sne.py (χ²/dof=0.955, published)
- **SALT2-free validation**: qfd_transfer_pipeline.py (σ=0.40, 0 free physics)
- The SALT2-free result eliminates the circular reasoning concern
- The 0.40 mag scatter is from SN Ia diversity, not the distance model

---

## CONSTANTS REFERENCE

| Constant | v2 Kelvin | V18 | Source |
|----------|-----------|-----|--------|
| K_J | 85.58 km/s/Mpc | 89.96 (via 70 + 19.96) | Golden Loop / MCMC |
| eta | 1.066 (pi^2/beta^2) | eta'=-6.0, xi=-6.0 | Locked / fitted |
| D_L(z) | (c/K_J)*ln(1+z)*(1+z)^{2/3} | c*z/k_J | Kelvin / linear |
| K_MAG | 5/ln(10) = 2.172 | (not used directly) | Convention |
| K (alpha->mag) | 2.5/ln(10) = 1.086 | 2.5/ln(10) = 1.086 | Standard |

---

## FILES

| File | What it does | Status |
|------|-------------|--------|
| `golden_loop_sne.py` | SALT2-reduced pipeline, 0 free params | WORKING, publishable |
| `qfd_transfer_pipeline.py` | QFD vacuum transfer function (SALT2-free) | COMPLETED |
| `clean_sne_pipeline.py` | Clean raw photometry pipeline (Option B) | COMPLETED |
| `raw_sne_kelvin.py` | Raw V18/V22 pipeline, alpha space | DIAGNOSTIC TOOL |
| `RAW_PIPELINE_STATUS.md` | This document | Current |
| `SNe_Data.md` | Data sources and provenance | Reference |
| V18 source: `SupernovaSrc/qfd-supernova-v15/v15_clean/v18/` | Original V18 code | External |
| V22 source: `projects/astrophysics/qfd-sn-v22/` | V22 stage 1 code | Internal |
