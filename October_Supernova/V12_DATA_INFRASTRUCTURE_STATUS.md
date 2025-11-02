# V12 Data Infrastructure Status

**Instance:** I2 (Instance 2 - Data Infrastructure)
**Date:** 2025-11-01
**Status:** Core Tools Complete ✅

---

## Scope Division (V12)

### Instance 1 (I1) - V12 Solver/Code
- V12 Python code fixes
- Bug fixes and robustness improvements
- Run V12 fitting pipeline
- Branch: `claude-instance-1-v12-critical-bugfixes`

### Instance 2 (I2) - V12 Data Infrastructure **[THIS INSTANCE]**
- ✅ Data acquisition (DES-SN 5YR)
- ✅ Quality filtering (schema, gates, manifests)
- ✅ Libraries and dependencies
- ✅ Credibility framework
- 🔄 Data download in progress
- ⏳ Future: Output/graphs/charts

---

## Completed Deliverables ✅

### 1. Quality Gates Schema (v1.0)
**File:** `data/quality_gates_schema_v1.json`
**Status:** LOCKED (pre-registered before fitting)

**Standard Filters:**
- z_min = 0.05 (peculiar velocity mitigation)
- z_max = 1.5 (QFD model range)
- min_obs = 20 (statistical requirement)
- min_bands = 2 (color information)
- min_snr = 5.0 (lenient, compensated by sigma_floor)

**QFD-Specific Filters:**
- ⚠️ **max_obs = 150** - CRITICAL for MCMC geometry
  - Prevents NUTS sampler pathology
  - Low-z SNe with 200-500 obs cause divergences
  - Evidence: V11 production runs
- **sigma_floor_jy = 0.02 Jy** - Systematic uncertainty floor
  - ~2% calibration systematics
  - Prevents over-weighting high-S/N epochs
- **duplicate_mjd_policy = "best_snr"** - Same-night collapse
  - Keeps highest S/N observation per night
  - Avoids correlated errors
- **survey_whitelist = ["DES"]** - Homogeneity
  - Single survey for uniform systematics

**Expected Yield:**
- DES-SN 5YR: 1,635 SNe → ~1,400-1,500 clean
- 2× larger than Pantheon+ clean (~650-700 SNe)

### 2. Quality Filtering Guide
**File:** `docs/QUALITY_FILTERING_GUIDE.md`

**Contents:**
- Complete filter justifications
- Application workflow
- Credibility framework
- Publication methods text template
- FAQ addressing QFD-specific needs

**Key Insight:**
Why QFD needs more knobs than standard analyses:
- Standard (SALT2 + ΛCDM): Parametric fit → basic quality cuts
- QFD (Direct photometry): Bayesian MCMC → geometry conditioning required

### 3. Format Conversion Script
**File:** `tools/convert_des_to_qfd_format.py`

**Features:**
- FITS/SNANA/CSV → QFD CSV conversion
- AB magnitude → flux (Jy) with proper error propagation
- DES band → effective wavelength mapping (g/r/i/z/Y)
- Batch processing (directories or single files)
- Schema validation

**Usage:**
```bash
python tools/convert_des_to_qfd_format.py \
    --input data/des_sn5yr/raw/ \
    --output data/des_sn5yr/lightcurves_des_sn5yr.csv \
    --schema data/quality_gates_schema_v1.json
```

### 4. Quality Gates Application Script
**File:** `tools/apply_quality_gates.py`

**Features:**
- Applies all gates from schema in sequence
- Tracks before/after counts for each gate
- Generates `sample_selection_manifest.json`
- Full provenance tracking
- Credibility framework compliance

**Gates Applied (in order):**
1. Redshift cuts (z_min, z_max)
2. Observation count cuts (min_obs, max_obs)
3. Band coverage (min_bands)
4. S/N threshold (min_snr)
5. Sigma floor application (sigma_floor_jy)
6. Duplicate MJD collapse (duplicate_mjd_policy)
7. Survey whitelist (survey_whitelist)

**Usage:**
```bash
python tools/apply_quality_gates.py \
    --input data/des_sn5yr/lightcurves_des_sn5yr.csv \
    --schema data/quality_gates_schema_v1.json \
    --output data/des_sn5yr/lightcurves_des_sn5yr_clean.csv \
    --manifest data/des_sn5yr/sample_selection_manifest_v1.json
```

### 5. DES-SN 5YR Download Script
**File:** `tools/download_des_sn5yr.sh`

**Features:**
- Downloads from Zenodo (DOI: 10.5281/zenodo.12720778)
- Fetches metadata via Zenodo API
- Auto-extracts ZIP archive (1.4GB)
- Downloads documentation from GitHub
- Status: 🔄 Running in background

**Dataset Details:**
- Publication: Sánchez et al. 2024, ApJ, 975, 5
- Published: November 1, 2024
- SNe: 1,635 (z = 0.1–1.13)
- Homogeneous (DES-only, modern calibration)

---

## Credibility Framework ✅

### Pre-Registration
- ✅ Schema locked: 2025-11-01
- ✅ Version controlled: Git commits
- ✅ Frozen BEFORE any V12 fitting

### Transparency Requirements
- ✅ Manifest per run (before/after counts)
- ✅ Versioned exclusion list (to be created post-fit)
- ✅ Outlier identification via frozen-knobs

### Robustness Checks (To Be Implemented)
- ⏳ Ablation tests (parameter stability under varied cuts)
- ⏳ Frozen-knobs audit (full catalog evaluation)
- ⏳ Cross-validation (Pantheon+ vs DES-SN 5YR)

---

## Workflow Summary

### Phase 1: Data Acquisition ✅
```bash
# Download DES-SN 5YR (running in background)
./tools/download_des_sn5yr.sh
```

**Status:** 🔄 Download in progress (~7 minutes remaining)

### Phase 2: Format Conversion ⏳
```bash
# Convert to QFD format
python tools/convert_des_to_qfd_format.py \
    --input data/des_sn5yr/raw/ \
    --output data/des_sn5yr/lightcurves_des_sn5yr.csv \
    --format auto
```

**Status:** ⏳ Pending download completion

### Phase 3: Quality Filtering ⏳
```bash
# Apply quality gates
python tools/apply_quality_gates.py \
    --input data/des_sn5yr/lightcurves_des_sn5yr.csv \
    --schema data/quality_gates_schema_v1.json \
    --output data/des_sn5yr/lightcurves_des_sn5yr_clean.csv \
    --manifest data/des_sn5yr/sample_selection_manifest_v1.json
```

**Status:** ⏳ Pending conversion

### Phase 4: Signal Ready for Fitting ⏳
```bash
# Create marker file
touch data/des_sn5yr/.READY_FOR_FIT

# Commit with handoff message
git commit -m "[Instance-2-DATA] DES-SN 5YR clean sample ready for V12 fitting"
```

**Status:** ⏳ Pending filtering

**Handoff Point:** I2 → I1

### Phase 5: V12 Fit (Instance 1's Work) ⏳
```bash
# I1 runs V12 pipeline
python main_v12.py \
    --prefiltered-lightcurves data/des_sn5yr/lightcurves_des_sn5yr_clean.csv \
    --output-dir results/des_sn5yr_v12_run1/
```

**Status:** ⏳ Pending I1

**Output:** `results/des_sn5yr_v12_run1/v12_best_fit.json`

### Phase 6: Frozen-Knobs Audit (Instance 2 Resumes) ⏳
```bash
# Evaluate frozen params on full catalog
python tools/frozen_knobs_audit.py \
    --frozen-params results/des_sn5yr_v12_run1/v12_best_fit.json \
    --full-catalog data/des_sn5yr/lightcurves_des_sn5yr.csv \
    --chi2-threshold 5.0 \
    --output-outliers data/des_sn5yr/outliers_chi2_v1.json
```

**Status:** ⏳ Script to be created (pending I1 fit)

---

## Git Commits (Instance 2)

```
96e8c41 [Instance-2-DATA] Create quality gates application script
e2b62b3 [Instance-2-DATA] Create and update DES-SN 5YR download script
bba50c1 [Instance-2-DATA] Create DES-SN 5YR to QFD format converter
f92a82d [Instance-2-DATA] Create QFD quality gates schema v1.0 (pre-registered, locked)
```

All commits on branch: `claude-instance-1-v12-critical-bugfixes` (shared with I1)

---

## Next Steps

### Immediate (Instance 2)
1. ⏳ Wait for DES-SN 5YR download completion (~7 min)
2. ⏳ Run format conversion
3. ⏳ Apply quality gates
4. ⏳ Signal ready for fitting → Handoff to I1

### Short-Term (Instance 2)
5. ⏳ Create `frozen_knobs_audit.py` script
6. ⏳ Create `ablation_table_generator.py` script
7. ⏳ Create additional documentation:
   - `docs/DES_SN5YR_INTEGRATION.md`
   - `docs/CREDIBILITY_FRAMEWORK.md`

### Medium-Term (After I1 Completes Fit)
8. ⏳ Run frozen-knobs audit on full DES-SN 5YR
9. ⏳ Generate outlier exclusion list (versioned)
10. ⏳ Run ablation tests (parameter stability checks)
11. ⏳ Cross-validation with Pantheon+ clean sample

### Long-Term (Instance 2 Expanded Scope)
12. ⏳ Output/Graphs/Charts (per user request):
    - Hubble diagrams (QFD native)
    - Residual plots (vs z, vs mag)
    - Q-Q plots (frozen-knobs audit)
    - χ² histograms
    - Parameter posteriors (corner plots)
    - Publication-ready figures

---

## Key Technical Decisions

### Why max_obs = 150?
**Empirical Evidence:** V11 production runs show MCMC failures (divergences, ESS < 100) for N_obs > 150.

**Root Cause:** QFD uses epoch-by-epoch Bayesian inference (NumPyro NUTS). High-dimensional likelihood geometry becomes degenerate with 200-500 obs.

**SALT2 Doesn't Have This Problem:** SALT2 fits parametric model (3 params), not epoch-by-epoch inference.

### Why sigma_floor in Jy instead of magnitudes?
**QFD Works in Flux Space:** Direct photometry modeling uses flux (Jy), not magnitudes.

**Systematic Errors More Uniform in Flux:** Calibration, PSF, host subtraction errors are ~2% in flux, not magnitude-dependent.

### Why Apply max_chi2_dof AFTER Fitting?
**Avoid Circularity:** Fit on clean sample → freeze params → evaluate on full catalog → identify outliers.

**Outliers Are Data Quality, Not Model Selection:** χ²/dof > 5.0 indicates mis-classifications or contamination, not model failures.

---

## Success Criteria

### Phase 1: Data Acquisition ✅
- ✅ DES-SN 5YR downloading (1.4GB)
- ⏳ N ≥ 1,400 SNe in clean sample (pending processing)
- ✅ Format compatible with V12 `--prefiltered-lightcurves`

### Phase 2: Quality Control ✅
- ✅ Quality gates locked BEFORE any fits
- ⏳ Sample selection manifest complete (pending processing)
- ⏳ Exclusion list justified (post-fit)

### Phase 3: Credibility ⏳
- ⏳ Ablation table shows parameter stability
- ⏳ Frozen-knobs audit demonstrates robustness
- ⏳ Publication methods section written

### Phase 4: Validation ⏳
- ⏳ V12 converges on DES clean (I1's work)
- ⏳ Cross-validation with Pantheon+ clean
- ⏳ Results publishable with strong credibility

---

## Files Created (Instance 2)

### Core Tools
- ✅ `data/quality_gates_schema_v1.json` (12KB)
- ✅ `tools/convert_des_to_qfd_format.py` (507 lines)
- ✅ `tools/apply_quality_gates.py` (518 lines)
- ✅ `tools/download_des_sn5yr.sh` (135 lines)

### Documentation
- ✅ `docs/QUALITY_FILTERING_GUIDE.md` (16KB)
- ✅ `V12_DATA_INFRASTRUCTURE_STATUS.md` (this file)

### Lock Files
- ✅ `.lock-quality_gates_schema_v1.json`

### Data (In Progress)
- 🔄 `data/des_sn5yr/raw/` (downloading)
- 🔄 `data/des_sn5yr/download.log` (download log)

### Data (Pending)
- ⏳ `data/des_sn5yr/lightcurves_des_sn5yr.csv` (converted)
- ⏳ `data/des_sn5yr/lightcurves_des_sn5yr_clean.csv` (filtered)
- ⏳ `data/des_sn5yr/sample_selection_manifest_v1.json` (provenance)

---

## Communication with Instance 1

### Current Status
**Instance 2:** Core data infrastructure complete, download in progress
**Instance 1:** Working on V12 critical bug fixes

### Handoff Protocol
When `data/des_sn5yr/lightcurves_des_sn5yr_clean.csv` is ready:
1. Create `.READY_FOR_FIT` marker file
2. Git commit with tag: `[Instance-2-DATA] DES clean sample ready`
3. I1 runs V12 fit
4. I1 creates `v12_best_fit.json` with frozen params
5. I2 resumes with frozen-knobs audit

---

**Last Updated:** 2025-11-01 by Instance 2
**Status:** Core tools complete ✅, awaiting DES-SN 5YR download
**Next:** Process data → Handoff to I1
