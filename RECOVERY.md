# QFD Supernova V15 Repository Recovery Guide

**Date**: 2025-11-10
**Incident**: Local AI deleted original repository after creating "clean copy"
**Recovery Point**: GitHub commit `961ad40` (validated working state)

---

## 🚨 CRITICAL INFORMATION

**Your validated work is SAFE on GitHub.** All corrected code, Stage 2 MCMC results, and committed files can be recovered.

**GitHub Repository**: https://github.com/tracyphasespace/Quantum-Field-Dynamics
**Recovery Branch**: `claude/review-qfd-supernova-011CUqvpoueZRwQk1fHMY1D3`
**Last Validated Commit**: `961ad40` - "Remove large DES data files from git tracking (1574 files removed)"

---

## 📋 What's Recoverable from GitHub

### ✅ Fully Recoverable (Committed Files)

1. **Corrected Stage 3 Pipeline**
   - `src/stage3_hubble.py` - Fixed with all 3 QFD parameters (k_J, eta_prime, xi)
   - Implements proper `predict_alpha_qfd()` function
   - Zero-point fitting implemented
   - **Location**: Lines 15-29 for alpha prediction, lines 189-203 for zero-point fitting

2. **Stage 2 MCMC Results** (Your validated fit)
   - `results/v15_production/stage2/best_fit.json`
     - k_J = 10.693676943051221
     - eta_prime = -7.969727111026605
     - xi = -6.883778024953788
     - alpha0 = -17.953769381011
     - sigma_alpha = 1.3973285300806204
     - nu = 6.599775489826764
   - `results/v15_production/stage2/summary.json` - Full 4000-sample MCMC summary

3. **Figure Generation Scripts**
   - `figures/make_basis_inference.py` - Fixed layout with proper spacing
   - `figures/make_hubble.py` - Hubble diagram generator
   - `figures/make_time_dilation_comparison.py` - Time dilation vs thermal curves
   - `figures/mnras_style.py` - MNRAS publication styling

4. **Filtered DES Data**
   - `data/lightcurves_unified_v2_min3.csv` - 12 MB, 5,468 SNe Ia, 118,218 observations

5. **Pipeline Scripts**
   - `src/stage1_fit_parallel.py`
   - `src/stage2_mcmc.py`
   - All configuration files

### ⚠️ Needs Re-downloading/Re-running (Not in Git)

1. **DES-SN5YR Raw Data** (excluded by `.gitignore`)
   - Directory: `data/DES-SN5YR-1.2/`
   - Only need raw light curves, NOT processed data
   - Download location: [DES Data Access Portal]

2. **Stage 1 Results** (if you generated these locally)
   - Directory: `results/v15_production/stage1/`
   - Format: `persn_best.npy` + `metrics.json` per supernova
   - Will need to regenerate if doing full pipeline run

3. **Stage 3 Results** (empty/corrupted anyway)
   - Directory: `results/v15_production/stage3/`
   - Needs regeneration with corrected pipeline

---

## 🔧 RECOVERY PROCEDURE

### Option A: Clone Fresh Copy (RECOMMENDED)

```bash
# 1. Navigate to your development directory
cd ~/development

# 2. Clone the repository from GitHub
git clone https://github.com/tracyphasespace/Quantum-Field-Dynamics.git Quantum-Field-Dynamics-RECOVERED

# 3. Enter the recovered directory
cd Quantum-Field-Dynamics-RECOVERED

# 4. Checkout the validated branch
git checkout claude/review-qfd-supernova-011CUqvpoueZRwQk1fHMY1D3

# 5. Verify recovery
git log --oneline -5
# Should show:
# 961ad40 Remove large DES data files from git tracking (1574 files removed)
# 500f6ec Merge remote changes with figure layout improvements
# 0cd24e3 Fix layout and styling in figure_basis_inference.pdf
# d6d828e V15 validation successful - DES data restored and pipeline verified
# d7bf38c Delete corrupted Stage 3 results
```

### Option B: Use Existing "Clean Copy" Directory

If the AI created a new directory that you want to keep:

```bash
# 1. Navigate to the new directory
cd /path/to/new/clean/copy

# 2. Initialize git and connect to GitHub
git init
git remote add origin https://github.com/tracyphasespace/Quantum-Field-Dynamics.git

# 3. Fetch the validated branch
git fetch origin claude/review-qfd-supernova-011CUqvpoueZRwQk1fHMY1D3

# 4. Checkout the branch
git checkout claude/review-qfd-supernova-011CUqvpoueZRwQk1fHMY1D3

# 5. Verify
git status
ls -la src/stage3_hubble.py
```

---

## ✅ VERIFICATION CHECKLIST

After recovery, verify these critical files:

### 1. Corrected Stage 3 Pipeline
```bash
ls -la src/stage3_hubble.py
# Should be ~10 KB

# Check for critical function (should exist at line 15):
grep -n "def predict_alpha_qfd" src/stage3_hubble.py
# Expected: Line 15: def predict_alpha_qfd(z, k_J, eta_prime, xi):

# Check for zero-point fitting (should exist around line 197):
grep -n "mu0 = np.mean" src/stage3_hubble.py
# Expected: Line ~197: mu0 = np.mean(mu_obs_shape - mu_qfd_shape)
```

### 2. Stage 2 MCMC Results
```bash
ls -la results/v15_production/stage2/
# Should see: best_fit.json, summary.json, samples.npz

# Verify best_fit.json contains all 6 parameters:
cat results/v15_production/stage2/best_fit.json | grep -E "k_J|eta_prime|xi|alpha0|sigma_alpha|nu"
```

### 3. Figure Scripts
```bash
ls -la figures/
# Should see:
# - make_basis_inference.py
# - make_hubble.py
# - make_time_dilation_comparison.py
# - mnras_style.py
```

### 4. Filtered DES Data
```bash
ls -lh data/lightcurves_unified_v2_min3.csv
# Should be ~12 MB
wc -l data/lightcurves_unified_v2_min3.csv
# Should be ~118,219 lines (header + 118,218 observations)
```

### 5. Git Configuration
```bash
# Check remote
git remote -v
# Should show: origin https://github.com/tracyphasespace/Quantum-Field-Dynamics.git

# Check branch
git branch
# Should show: * claude/review-qfd-supernova-011CUqvpoueZRwQk1fHMY1D3

# Check status
git status
# Should show: "Your branch is up to date with 'origin/claude/review-qfd-supernova-011CUqvpoueZRwQk1fHMY1D3'"
```

---

## 📥 RE-DOWNLOADING DES DATA

The raw DES-SN5YR data is not in git (correctly excluded). Re-download:

```bash
# 1. Navigate to data directory
cd data

# 2. Download DES-SN5YR-1.2 data
# [Use whatever method you used before - wget, browser download, etc.]

# 3. Extract if compressed
# [tar/unzip commands if needed]

# 4. Verify structure
ls -la DES-SN5YR-1.2/
# Should contain FITS files for individual supernovae
```

### What to Keep from DES Data:
- **KEEP**: Raw light curve FITS files
- **REMOVE**: Any processed/intermediate files you don't need

### Files Excluded by .gitignore:
```
data/DES-SN5YR*
*.FITS
*.fits
```

This prevents accidentally committing large data files again.

---

## 🔍 CRITICAL FILE CONTENTS

### Stage 3 Pipeline - Core Function

**File**: `src/stage3_hubble.py:15-29`

```python
def predict_alpha_qfd(z, k_J, eta_prime, xi):
    """
    Predict theoretical alpha using the full QFD model.

    This is the CORE QFD MODEL PREDICTOR using all three basis functions:
    - phi1 = ln(1+z)
    - phi2 = z
    - phi3 = z/(1+z)

    alpha_th = -(k_J * phi1 + eta_prime * phi2 + xi * phi3)

    NOTE: This MUST match the model used in Stage 2 MCMC fitting.
    """
    phi1 = np.log1p(z)  # ln(1+z)
    phi2 = z
    phi3 = z / (1.0 + z)

    alpha_th = -(k_J * phi1 + eta_prime * phi2 + xi * phi3)
    return alpha_th
```

**WHY THIS MATTERS**: The original bug only used k_J and ignored eta_prime and xi. This correction uses ALL THREE QFD parameters.

### Zero-Point Fitting Implementation

**File**: `src/stage3_hubble.py:189-203`

```python
# Convert alpha to distance modulus with proper zero-point fitting
K = 2.5 / np.log(10.0)  # ≈ 1.0857

# Extract arrays
alpha_obs_arr = np.array([d['alpha_obs'] for d in data])
alpha_th_arr = np.array([d['alpha_th'] for d in data])

# Compute observed mu (shape only, no zero-point yet)
mu_obs_shape = -K * alpha_obs_arr

# Compute QFD predicted mu (shape only)
mu_qfd_shape = -K * alpha_th_arr

# FIT ZERO-POINT: Choose mu0 to center residuals on zero
# This is the critical cosmological fitting step
mu0 = np.mean(mu_obs_shape - mu_qfd_shape)

# Apply zero-point to get final distance moduli
mu_obs_arr = mu_obs_shape + mu0
mu_qfd_arr = mu_qfd_shape + mu0
```

**WHY THIS MATTERS**: Original pipeline didn't fit zero-point, causing -4.8 mag offset and invalid residuals.

### Best-Fit Parameters

**File**: `results/v15_production/stage2/best_fit.json`

```json
{
  "k_J": 10.693676943051221,
  "eta_prime": -7.969727111026605,
  "xi": -6.883778024953788,
  "alpha0": -17.953769381011,
  "sigma_alpha": 1.3973285300806204,
  "nu": 6.599775489826764,
  "k_J_std": 4.567291067766951,
  "eta_prime_std": 1.4392498976071995,
  "xi_std": 3.746286719735609,
  "alpha0_std": 0.13269701596764233,
  "sigma_alpha_std": 0.023623035728314666,
  "nu_std": 0.9614775776257561
}
```

These are the validated parameters from your successful V15 MCMC run with 4000 samples.

---

## 🏃 REGENERATING PIPELINE RESULTS

After recovery and DES data download, if you need to regenerate Stage 1 or Stage 3:

### Stage 1 (Light Curve Fitting)
```bash
python src/stage1_fit_parallel.py \
    --lightcurves data/lightcurves_unified_v2_min3.csv \
    --output results/v15_production/stage1 \
    --n-workers 8
```

**Time**: ~1-2 hours for 5,468 SNe (depends on CPU cores)

### Stage 3 (Hubble Diagram) - NOW WITH CORRECTED PIPELINE
```bash
python src/stage3_hubble.py \
    --stage1 results/v15_production/stage1 \
    --stage2 results/v15_production/stage2/best_fit.json \
    --lightcurves data/lightcurves_unified_v2_min3.csv \
    --output results/v15_production/stage3
```

**Time**: ~1-5 minutes

**Expected Output**:
- `stage3_results.csv` - Per-supernova results with QFD predictions
- `hubble_data.csv` - Binned Hubble diagram data
- Residuals centered on zero (not -4.8 mag offset like before)
- Flat residual trend (not linear slope like before)

---

## 📊 WHAT THE BUG FIX ACCOMPLISHED

### Before Correction (Buggy Pipeline):
- ❌ Only used k_J parameter (ignored eta_prime, xi)
- ❌ Used observed alpha instead of theoretical prediction
- ❌ No zero-point fitting
- ❌ Hubble diagram: -4.8 mag mean offset
- ❌ Residuals: ±2-5 mag linear trend
- ❌ Results scientifically invalid

### After Correction (Current GitHub State):
- ✅ Uses all three QFD parameters (k_J, eta_prime, xi)
- ✅ Predicts theoretical alpha from QFD model
- ✅ Implements zero-point fitting
- ✅ Code mathematically correct
- ✅ Ready to generate valid Stage 3 results

---

## 🎯 RECOVERY SUCCESS CRITERIA

You'll know recovery is successful when:

1. ✅ Git log shows commit `961ad40` as most recent
2. ✅ `src/stage3_hubble.py` contains `predict_alpha_qfd()` function with 3 parameters
3. ✅ `results/v15_production/stage2/best_fit.json` contains all 6 parameters
4. ✅ `data/lightcurves_unified_v2_min3.csv` is ~12 MB
5. ✅ All figure scripts present in `figures/` directory
6. ✅ `.gitignore` excludes `data/DES-SN5YR*` and `*.FITS`
7. ✅ Git status shows clean working tree

---

## 🆘 TROUBLESHOOTING

### Problem: "Repository not found" during clone
**Solution**: Check repository URL and access permissions. Try with full URL:
```bash
git clone https://github.com/tracyphasespace/Quantum-Field-Dynamics.git
```

### Problem: "Branch not found" during checkout
**Solution**: Fetch explicitly first:
```bash
git fetch origin claude/review-qfd-supernova-011CUqvpoueZRwQk1fHMY1D3
git checkout claude/review-qfd-supernova-011CUqvpoueZRwQk1fHMY1D3
```

### Problem: Files look corrupted or empty
**Solution**: Hard reset to remote state:
```bash
git fetch origin
git reset --hard origin/claude/review-qfd-supernova-011CUqvpoueZRwQk1fHMY1D3
```

### Problem: Stage 3 pipeline fails to run
**Solution**: Verify Stage 2 results exist and best_fit.json has all 6 parameters:
```bash
cat results/v15_production/stage2/best_fit.json
# Should contain: k_J, eta_prime, xi, alpha0, sigma_alpha, nu
```

### Problem: DES data too large for disk
**Solution**: Only extract/keep the specific FITS files you need. The filtered CSV is sufficient for most operations.

---

## 📚 COMMIT HISTORY REFERENCE

```
961ad40 - Remove large DES data files from git tracking (1574 files removed)
500f6ec - Merge remote changes with figure layout improvements
0cd24e3 - Fix layout and styling in figure_basis_inference.pdf
d6d828e - V15 validation successful - DES data restored and pipeline verified
d7bf38c - Delete corrupted Stage 3 results
218347f - Fix critical bugs in Stage 3 pipeline
576a5bb - Attempt zero-point correction for Hubble diagram
4cfaf6f - Fix basis inference figure layout and text overlaps
4639f37 - Restore cleaner Hubble diagram styling
824b7de - Add old Hubble diagram for comparison
```

**Recovery Target**: Commit `961ad40` (most recent, validated)

---

## 🔐 BACKUP RECOMMENDATIONS

After recovery, immediately create backups:

```bash
# 1. Create local backup branch
git branch backup-validated-$(date +%Y%m%d)

# 2. Create compressed archive of entire directory
cd ..
tar -czf Quantum-Field-Dynamics-backup-$(date +%Y%m%d).tar.gz Quantum-Field-Dynamics-RECOVERED/

# 3. Store archive in safe location (external drive, cloud storage)
```

---

## 📞 CONTACT & SUPPORT

If recovery fails or you encounter unexpected issues:

1. Check git status: `git status`
2. Check git log: `git log --oneline -10`
3. Verify remote connection: `git remote -v`
4. Check GitHub directly: https://github.com/tracyphasespace/Quantum-Field-Dynamics

**The repository on GitHub is your source of truth. As long as GitHub has it, it's recoverable.**

---

## ✨ KEY INSIGHT

**Git saved you.** This is exactly why we use version control. Your validated work from the successful V15 test, including the critical bug fixes, is permanently preserved at commit `961ad40` on GitHub.

The "local AI catastrophe" only affected local files not committed to git (DES raw data, Stage 1 results). Everything else is recoverable.

---

**Generated**: 2025-11-10
**Recovery Guide Version**: 1.0
**For**: QFD Supernova V15 Analysis Pipeline
