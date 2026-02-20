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

### Option B: Re-fit from raw photometry (high effort)
Replace the V18 Stage 1 template with a v2 Kelvin template:
- D_L = (c/K_J)*ln(1+z)*(1+z)^{2/3} instead of D = cz/k_J
- eta = pi^2/beta^2 extinction instead of eta', xi
This gives a clean test of v2 Kelvin from raw data.
**Source**: DES-SN5YR raw photometry at `SupernovaSrc/qfd-supernova-v15/`.

### Option C: Accept SALT2 pipeline as primary (no extra effort)
The golden_loop_sne.py result (chi2/dof=0.955, 0 free params) already demonstrates
v2 Kelvin competitiveness. The raw pipeline is a secondary cross-check that requires
careful handling of the model-contamination issue.

### Recommendation
**Option C is correct for the book.** The SALT2-reduced result is clean, competitive,
and model-independent. The raw pipeline investigation revealed important systematics
(alpha convention, model contamination, lensing) but these complicate rather than
clarify the physics.

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
| `raw_sne_kelvin.py` | Raw V18/V22 pipeline, alpha space | DIAGNOSTIC TOOL |
| `RAW_PIPELINE_STATUS.md` | This document | Current |
| `SNe_Data.md` | Data sources and provenance | Reference |
| V18 source: `SupernovaSrc/qfd-supernova-v15/v15_clean/v18/` | Original V18 code | External |
| V22 source: `projects/astrophysics/qfd-sn-v22/` | V22 stage 1 code | Internal |
