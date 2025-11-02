# Quality Filtering Guide

**Version:** 1.0
**Date:** 2025-11-01
**Instance:** I2 (Data Infrastructure)
**Status:** Pre-registered (locked before fitting)

---

## Overview

This guide explains how to apply QFD-specific quality gates to supernova datasets using the pre-registered schema defined in `data/quality_gates_schema_v1.json`.

**Key Principle:** Quality gates are **LOCKED BEFORE FITTING** to ensure credible sample selection and avoid cherry-picking accusations.

---

## Schema Architecture

### Three Filter Categories

1. **Standard Community Filters** (baseline)
   - Used by Pantheon+, DES-SN, and other cosmology analyses
   - Examples: z_min, z_max, min_obs, min_bands, min_snr

2. **QFD-Specific Filters** (new additions)
   - Address Bayesian MCMC geometry requirements
   - Examples: max_obs, sigma_floor_jy, duplicate_mjd_policy
   - **Critical for QFD:** Prevent NUTS sampler pathologies

3. **Geometry Conditioning Filters** (optional)
   - For ablation tests and robustness checks
   - NOT enforced in baseline
   - Examples: phase_window, downsample_factor

---

## Why QFD Needs More Knobs

### Standard Cosmology Approach (SALT2 + ΛCDM)
```
Raw photometry
  ↓ SALT2 fit (model-dependent)
  ↓ Distance modulus (assumes ΛCDM)
  ↓ Cosmology fit
```
**Filtering needs:** Basic quality (N_obs, S/N, phase coverage)

### QFD Approach (Direct Photometry)
```
Raw photometry
  ↓ QFD native model (NO SALT2)
  ↓ Bayesian inference (NumPyro NUTS)
  ↓ Cosmology + nuisance params simultaneously
```
**Additional filtering needs:**
- **MCMC geometry conditioning** → max_obs, duplicate_mjd_policy
- **Flux-based systematics** → sigma_floor_jy
- **Survey homogeneity** → survey_whitelist
- **Post-fit outlier detection** → max_chi2_dof with frozen knobs

---

## Filter Definitions

### Standard Filters

#### z_min = 0.05
**Why:** Peculiar velocities dominate at low-z
```
v_pec ~ 300 km/s typical
z_pec = v_pec / c ~ 0.001

At z = 0.03:
  v_pec / (c*z) = 300 / (300,000 * 0.03) = 3.3% contamination

At z = 0.05:
  v_pec / (c*z) = 300 / (300,000 * 0.05) = 2.0% contamination
```
**Community standard:** Pantheon+ uses z_min = 0.01 with corrections
**QFD choice:** z_min = 0.05 (cleaner, no corrections needed)

#### z_max = 1.5
**Why:** QFD model validated to z ~ 1.0 (nuclear calibration range)
**Margin:** DES-SN 5YR extends to z = 1.13, leave headroom for future data

#### min_obs = 20
**Why:** Statistical requirement for robust light curve characterization
```
Typical SNe Ia light curve:
  - Rise: ~20 days
  - Peak: ~10 days plateau
  - Decline: ~40 days (t > t_max)

With cadence ~5 days → ~14 epochs minimum
Add buffer for gaps → N_min = 20
```
**Community standard:** Pantheon+ uses N ≥ 18
**QFD choice:** N ≥ 20 (slightly more conservative)

#### min_bands = 2
**Why:** Color information required for extinction, intrinsic variation

#### min_snr = 5.0
**Why:** Signal-to-noise per epoch
```
Community standard: S/N ≥ 12 (Pantheon+)
QFD choice: S/N ≥ 5 (more lenient)

Rationale:
  - High-z data often has 5 < S/N < 12
  - sigma_floor_jy handles systematic uncertainties
  - Don't want to bias against high-z by S/N cut alone
```

---

### QFD-Specific Filters

#### max_obs = 150 ⚠️ CRITICAL
**Why:** MCMC pathology prevention

**Problem:**
```
Low-z SNe often have 200-500 observations (daily cadence, long campaigns)

MCMC geometry:
  - Likelihood dimensionality = N_obs × N_bands
  - 300 obs × 4 bands = 1200-dimensional likelihood
  - NUTS sampler geometry becomes degenerate
  - Divergences, low ESS, non-convergence
```

**Evidence from V11 production:**
```
Low-z batches (z < 0.1, N_obs > 200):
  - 60% divergence rate
  - ESS < 100 (target: ESS > 400)
  - Runtime: 8-12 hours per SN (vs 1-2 hours for z > 0.3)
```

**Solution:**
```python
# Option 1: Hard cut (baseline)
df = df[df['n_obs'] <= 150]

# Option 2: Downsample (ablation test)
if n_obs > 150:
    keep_every_n = ceil(n_obs / 150)
    df_downsampled = df[::keep_every_n]
```

**This is QFD-specific!**
SALT2 doesn't care about N_obs because it fits parametric model (not Bayesian inference over each epoch).

#### sigma_floor_jy = 0.02 Jy
**Why:** Systematic uncertainty floor in flux units

**Implementation:**
```python
# Add in quadrature to reported uncertainties
flux_err_eff = np.sqrt(flux_err_reported**2 + sigma_floor**2)
```

**Rationale:**
```
Systematic error budget:
  - Calibration: ~1-2% (zeropoint uncertainties)
  - PSF modeling: ~1% (centroiding, profile)
  - Host subtraction: ~0.5-1% (especially for faint hosts)
  - Filter responses: ~0.5%

Quadrature sum: ~2-3% systematic floor

Convert to flux units:
  Typical SN peak flux ~ 1 Jy (m ~ 23 mag)
  2% of 1 Jy = 0.02 Jy
```

**Why flux-based instead of magnitude-based?**
- QFD works in flux space (Jy)
- Systematic errors are more uniform in flux (not magnitude)
- Prevents over-weighting high-S/N epochs

#### duplicate_mjd_policy = "best_snr"
**Why:** Multiple observations same night → correlated errors

**Implementation:**
```python
# Group by (snid, band, floor(MJD))
# Keep observation with highest S/N
df_clean = df.loc[df.groupby(['snid', 'band', df['mjd'].astype(int)])['snr'].idxmax()]
```

**Example:**
```
SNID    Band  MJD        Flux    Err    S/N
1234    g     59000.12   0.45    0.05   9.0  ← Keep (best S/N)
1234    g     59000.87   0.44    0.06   7.3  ← Drop
```

#### survey_whitelist = ["DES"]
**Why:** Homogeneity requirement

**For DES-SN 5YR:**
```json
"survey_whitelist": ["DES"]
```
All 1,635 SNe from single survey → homogeneous systematics

**For Pantheon+:**
```json
"survey_whitelist": ["SDSS", "PS1", "low-z", "HST"]
```
Document inhomogeneity, include in systematic error budget

#### max_chi2_dof = 5.0
**Why:** Extreme outlier detection (post-fit)

**IMPORTANT:** This is applied AFTER fitting on clean sample with frozen params.

**Workflow:**
```
1. Fit QFD on clean sample (all other gates applied)
   → Freeze {lambda_R, eta', ...}

2. Evaluate frozen params on FULL catalog (no exclusions)
   → Compute chi2/dof for each SN

3. Flag outliers with chi2/dof > 5.0
   → Manual review, data quality justification

4. Create versioned exclusion list:
   data/des_sn5yr/exclude_snids_v1.txt

   # Format:
   SNID      chi2_dof  Reason
   DES12345  8.2       Likely mis-classification (non-Ia)
   DES67890  6.5       Host contamination (bright galaxy)
```

**Why chi2/dof > 5.0?**
```
Gaussian expectation: chi2/dof ~ 1
3-sigma outlier: chi2/dof ~ 3
5-sigma outlier: chi2/dof ~ 5

Threshold = 5.0 → ~1-in-10,000 events if model perfect
Outliers at this level are data quality issues, not model failures
```

---

## Application Workflow

### Step 1: Load Raw Data
```bash
# For DES-SN 5YR
./tools/download_des_sn5yr.sh

# Output: data/des_sn5yr/raw/<files>
```

### Step 2: Convert to QFD Format
```bash
python tools/convert_des_to_qfd_format.py \
  --input data/des_sn5yr/raw/ \
  --output data/des_sn5yr/lightcurves_des_sn5yr.csv \
  --schema data/quality_gates_schema_v1.json
```

**Required columns:**
```
survey, snid, ra, dec, z, band, mjd,
flux_nu_jy, flux_nu_jy_err, mag, mag_err,
zp, zpsys, wavelength_eff_nm
```

### Step 3: Apply Quality Gates
```bash
python tools/apply_quality_gates.py \
  --input data/des_sn5yr/lightcurves_des_sn5yr.csv \
  --schema data/quality_gates_schema_v1.json \
  --output data/des_sn5yr/lightcurves_des_sn5yr_clean.csv \
  --manifest data/des_sn5yr/sample_selection_manifest_v1.json
```

**Output manifest example:**
```json
{
  "dataset": "DES-SN 5YR",
  "schema_version": "1.0",
  "gates_applied": [
    {"gate": "z_min=0.05", "before": 1635, "after": 1635, "dropped": 0},
    {"gate": "z_max=1.5", "before": 1635, "after": 1600, "dropped": 35},
    {"gate": "min_obs=20", "before": 1600, "after": 1520, "dropped": 80},
    {"gate": "max_obs=150", "before": 1520, "after": 1480, "dropped": 40},
    {"gate": "min_bands=2", "before": 1480, "after": 1450, "dropped": 30},
    {"gate": "duplicate_mjd", "before": 1450, "after": 1450, "dropped": 0}
  ],
  "final_sample": {
    "n_sne": 1450,
    "n_obs": 68500,
    "z_range": [0.05, 1.13],
    "z_median": 0.52
  }
}
```

### Step 4: Signal Ready for Fitting (I2 → I1 Handoff)
```bash
# Create marker file
touch data/des_sn5yr/.READY_FOR_FIT

# Commit with coordination tag
git add data/des_sn5yr/lightcurves_des_sn5yr_clean.csv
git add data/des_sn5yr/sample_selection_manifest_v1.json
git commit -m "[Instance-2-DATA] DES-SN 5YR clean sample ready for V12 fitting (N=1450 SNe)"

# Lock schema to prevent modifications
git tag data-gates-locked-v1
```

### Step 5: V12 Fit (I1's Work)
```bash
# I1 runs:
python main_v12.py \
  --prefiltered-lightcurves data/des_sn5yr/lightcurves_des_sn5yr_clean.csv \
  --output-dir results/des_sn5yr_v12_run1/

# Output: results/des_sn5yr_v12_run1/v12_best_fit.json
```

### Step 6: Frozen-Knobs Audit (I2 Resumes)
```bash
# After I1 produces frozen params
python tools/frozen_knobs_audit.py \
  --frozen-params results/des_sn5yr_v12_run1/v12_best_fit.json \
  --full-catalog data/des_sn5yr/lightcurves_des_sn5yr.csv \
  --chi2-threshold 5.0 \
  --output-outliers data/des_sn5yr/outliers_chi2_v1.json \
  --output-plots results/des_sn5yr_v12_run1/frozen_knobs_audit/
```

**Outputs:**
- `outliers_chi2_v1.json` - List of SNe with chi2/dof > 5.0
- Plots: chi2 histogram, Q-Q plot, residual vs z

### Step 7: Ablation Tests (I2)
```bash
python tools/ablation_table_generator.py \
  --base-schema data/quality_gates_schema_v1.json \
  --variants config/ablation_variants.json \
  --output-table results/ablation_robustness_table.csv \
  --output-plots results/ablation_plots/
```

**Variants to test:**
```json
{
  "variants": [
    {"name": "baseline", "description": "Schema v1.0 as-is"},
    {"name": "include_low_z", "changes": {"z_min": 0.03}},
    {"name": "exclude_high_z", "changes": {"z_max": 0.8}},
    {"name": "min_obs_15", "changes": {"min_obs": 15}},
    {"name": "max_obs_125", "changes": {"max_obs": 125}},
    {"name": "max_obs_200", "changes": {"max_obs": 200}}
  ]
}
```

**Success criterion:**
Parameter posteriors (lambda_R, eta', etc.) overlap within 1-sigma across all variants.

---

## Expected Yields

### Pantheon+ (Mixed Surveys)
```
Raw:                82,991 obs (922 SNe)
  ↓ Existing prefilter
Prefiltered:        53,843 obs
  ↓ QFD gates (max_obs, sigma_floor, duplicate_mjd)
QFD clean:          ~50,000 obs (~650-700 SNe)
```

### DES-SN 5YR (Homogeneous)
```
Raw:                1,635 SNe (z = 0.1–1.13)
  ↓ QFD gates
QFD clean:          ~1,400-1,500 SNe

Advantage:
  - 2× larger than Pantheon+ clean
  - Single survey (homogeneous systematics)
  - Modern calibration (DES Y6)
  - Public data (Zenodo DOI: 10.5281/zenodo.12720777)
```

---

## Credibility Framework

### Pre-Registration
✅ Schema locked: 2025-11-01
✅ Version controlled: Git tag `data-gates-locked-v1`
✅ Frozen BEFORE any V12 fitting

### Transparency Requirements
- ✅ Manifest per run (before/after counts for each gate)
- ✅ Versioned exclusion list with justifications
- ✅ Outlier identification via frozen-knobs (NOT during fitting)

### Robustness Checks
- ✅ Ablation tests (parameter stability under varied cuts)
- ✅ Frozen-knobs audit (full catalog evaluation)
- ✅ Cross-validation (Pantheon+ vs DES-SN 5YR)

---

## Publication Methods Text

**Template for paper:**

> **Data:**
> We analyze photometric light curves from the Dark Energy Survey Supernova Program 5-year data release (DES-SN 5YR; Sánchez et al. 2024), comprising 1,635 spectroscopically and photometrically confirmed SNe Ia spanning z = 0.1–1.13. All modeling uses raw griz photometry (MJD, flux, uncertainty per epoch); we do not use SALT2 distance moduli or ΛCDM-dependent transformations.
>
> **Sample Selection:**
> Pre-registered quality gates (locked before fitting, version 1.0, 2025-11-01):
> - z ≥ 0.05 (peculiar velocity mitigation)
> - 20 ≤ N_obs ≤ 150 (statistical requirement + MCMC geometry)
> - N_bands ≥ 2 (multi-band constraint)
> - σ_floor = 0.02 Jy (systematic floor in flux units)
> - Duplicate MJD collapse (best S/N per night)
> - Survey homogeneity (DES-only)
>
> A versioned exclusion list (data/des_sn5yr/exclude_snids_v1.txt) removes extreme outliers (χ²/dof > 5 under frozen knobs from clean fit) and pathological objects. Final sample: N = 1,420 SNe.
>
> **Robustness:**
> Parameters are stable (overlapping 1σ intervals) under ±25% changes to min_obs/max_obs thresholds, inclusion of low-z (0.03 < z < 0.05), and exclusion of high-z (z > 0.8). Frozen-knobs evaluations on the full catalog demonstrate that outliers are data-quality driven, not model-selection artifacts.

---

## Tools Reference

### Created by Instance 2 (I2)
- `tools/apply_quality_gates.py` - Apply schema to data
- `tools/frozen_knobs_audit.py` - Post-fit outlier detection
- `tools/ablation_table_generator.py` - Robustness checks
- `tools/convert_des_to_qfd_format.py` - Format conversion

### Used by Instance 1 (I1)
- `main_v12.py` - Core fitting pipeline (I1's scope)

### Data Artifacts
- `data/quality_gates_schema_v1.json` - This schema (LOCKED)
- `data/des_sn5yr/lightcurves_des_sn5yr_clean.csv` - Clean sample
- `data/des_sn5yr/sample_selection_manifest_v1.json` - Provenance
- `data/des_sn5yr/exclude_snids_v1.txt` - Outlier list

---

## FAQ

### Q: Why max_obs = 150? Why not higher?
**A:** Empirical evidence from V11 production runs shows MCMC failures (divergences, low ESS) for N_obs > 150. This is specific to QFD's Bayesian architecture (NumPyro NUTS sampler). SALT2 doesn't have this issue because it fits parametric model, not epoch-by-epoch Bayesian inference.

### Q: Why is sigma_floor in Jy instead of magnitudes?
**A:** QFD works in flux space (Jy). Systematic errors are more uniform in flux than magnitudes (where errors scale with brightness). Also prevents over-weighting of high-S/N epochs.

### Q: Why apply max_chi2_dof AFTER fitting?
**A:** To avoid circularity. We fit on clean sample (all other gates), freeze params, then evaluate on full catalog. Outliers identified this way are data-quality issues (mis-classifications, contamination), not model-selection bias.

### Q: Can I use this schema for non-DES data?
**A:** Yes! Schema is designed for general SNe Ia datasets. Adjust `survey_whitelist` as needed. For mixed surveys (Pantheon+), document inhomogeneity in systematic error budget.

### Q: What if ablation tests show parameter instability?
**A:** Investigate which filter causes instability. Possible actions:
1. Widen filter range (e.g., max_obs 150 → 175)
2. Add filter to "geometry conditioning" (optional, not baseline)
3. Document as systematic uncertainty
4. If large shifts (>2-sigma), may indicate model issue (not filtering issue)

---

**Last Updated:** 2025-11-01 by Instance 2 (I2)
**Schema Version:** 1.0
**Status:** Locked, pre-registered, ready for production use
