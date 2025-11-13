# Data Requirements

## Quick Start with Test Dataset

The `test_dataset/` directory contains a small subset of data (200 SNe, 4.5 MB) for quick testing and debugging:

```bash
python3 stages/stage2_simple.py \
  --stage1-results test_dataset/stage1_results \
  --lightcurves test_dataset/lightcurves_test.csv \
  --out test_output \
  --nchains 2 \
  --nsamples 2000 \
  --nwarmup 1000 \
  --quality-cut 2000 \
  --use-informed-priors
```

**Note**: The test dataset is designed for debugging Stage 2 MCMC code. Results will differ from production runs due to smaller sample size.

## Full Production Data

For full production runs matching the November 5, 2024 golden reference, you'll need:

### 1. Unified Lightcurves CSV (~12 MB)

**File**: `lightcurves_unified_v2_min3.csv`

**Contents**: Photometric measurements for all supernovae
- Columns: `snid`, `z`, `mjd`, `flux_nu_jy`, `flux_nu_jy_err`, `band`, `wavelength_eff_nm`, etc.
- ~118,000 measurements
- Multiple surveys: DES, Pantheon+, Union3, etc.

**How to obtain**: Contact QFD research team or check internal data repository

### 2. Stage 1 Results Directory (~107 MB)

**Directory**: `stage1_fullscale/`

**Contents**: Per-supernova optimization results from Stage 1
- 4727 supernova directories (one per SN ID)
- Each directory contains:
  - `laplace.json` - Laplace approximation results
  - `metrics.json` - Fit quality metrics (chi2, etc.)
  - `persn_best.npy` - Best-fit parameters
  - `status.txt` - Optimization status

**How to obtain**:
- Run Stage 1: `python3 stages/stage1_optimize.py --lightcurves <csv> --out stage1_results`
- Or request pre-computed results from QFD research team

## Running Full Production Stage 2

Once you have the full data:

```bash
python3 stages/stage2_simple.py \
  --stage1-results /path/to/stage1_fullscale \
  --lightcurves /path/to/lightcurves_unified_v2_min3.csv \
  --out stage2_production \
  --nchains 4 \
  --nsamples 4000 \
  --nwarmup 2000 \
  --quality-cut 2000 \
  --use-informed-priors
```

Expected results (within ±30% of golden reference):
- k_J = 10.770 ± 4.567 km/s/Mpc
- eta' = -7.988 ± 1.439
- xi = -6.908 ± 3.746

## Data Storage Recommendations

**DO NOT** commit full datasets to GitHub:
- Stage 1 results are too large (107 MB)
- Lightcurves CSV is borderline (12 MB) but better to exclude

**Instead**:
- Store full data on local compute resources
- Use test dataset for development and debugging
- Share data links via internal documentation

## Validation

After obtaining full datasets, verify they match expected format:

```bash
# Check lightcurves CSV
head lightcurves_unified_v2_min3.csv
# Should show: snid,dec,band,z,mjd,flux_nu_jy_err,wavelength_eff_nm,survey,ra,source_dataset,flux_nu_jy,snr

# Check Stage 1 structure
ls stage1_fullscale/ | head -10
# Should show SN ID directories like: 1246274, 1246275, etc.

# Check a single SN directory
ls stage1_fullscale/1246274/
# Should show: laplace.json, metrics.json, persn_best.npy, status.txt
```

## Questions?

Contact the QFD research team for data access or see the main README.md for collaboration guidelines.
