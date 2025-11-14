# V15_CLEAN: Data Directory

This directory contains the input data for the QFD supernova pipeline.

## Contents

**`lightcurves_unified_v2_min3.csv`** (12 MB, symlink to `../../data/`)
- 5,468 Type Ia supernovae lightcurves
- Combined Pantheon+ and DES-SN5YR datasets
- Pre-processed and ready for analysis
- Requires ≥3 observations per supernova

## Data Format

### Required Columns

```
snid          - Supernova identifier (string)
mjd           - Modified Julian Date of observation
z             - Redshift
flux_g        - g-band flux (erg/s/cm²/Hz)
flux_r        - r-band flux
flux_i        - i-band flux
flux_z        - z-band flux
fluxerr_g     - g-band flux uncertainty
fluxerr_r     - r-band flux uncertainty
fluxerr_i     - i-band flux uncertainty
fluxerr_z     - z-band flux uncertainty
```

## Data Provenance

### Sources
1. **Pantheon+** - ~1,700 SNe from multiple surveys (CfA, CSP, PS1, SDSS, SNLS, HST)
2. **DES-SN5YR** - ~1,800 SNe from Dark Energy Survey Year 5

### Processing Pipeline
Data preparation tools are in the separate repository:
`../../../October_Supernova/tools/`

Steps:
1. Download DES-SN5YR FITS files from public release
2. Convert DES FITS → CSV format (`convert_des_fits_to_qfd.py`)
3. Parse Pantheon+ ASCII tables (`parse_pantheon_plus.py`)
4. Unify format and filter (`create_unified_dataset.py`)
5. Filter to ≥3 observations per SN (`min3` filter)

## Using This Data

### From v15_clean (Self-Contained)
```bash
cd v15_clean
python stages/stage1_optimize.py \
    --lightcurves data/lightcurves_unified_v2_min3.csv \
    --out results/stage1
```

### From Repository Root (Legacy)
```bash
python v15_clean/stages/stage1_optimize.py \
    --lightcurves data/lightcurves_unified_v2_min3.csv \
    --out results/v15_clean/stage1
```

## Data Statistics

```
Total SNe:        5,468
Redshift range:   0.01 - 1.2
Surveys:          Pantheon+ (1,701), DES (1,813), SDSS (374), etc.
Bands:            g, r, i, z (SDSS/DES photometric system)
Total obs:        ~250,000 individual photometric measurements
```

## Regenerating Data (Optional)

If you need to rebuild from scratch:

1. Access external tools repository:
   ```bash
   cd ../../../October_Supernova/tools
   ```

2. Download raw data sources:
   - DES-SN5YR: https://des.ncsa.illinois.edu/releases/sn
   - Pantheon+: https://pantheonplussh0es.github.io/

3. Run processing pipeline:
   ```bash
   python convert_des_fits_to_qfd.py
   python parse_pantheon_plus.py
   python create_unified_dataset.py
   ```

## Data Quality Notes

- Pre-filtered for chi2 < 1e6 (removes pathological fits)
- Flux units: erg/s/cm²/Hz (consistent across surveys)
- Uncertainties include systematic floors
- Redshifts from spectroscopy (not photo-z)

## Citation

If using this dataset in publications, cite:
- **DES-SN5YR**: Abbott et al. (2024)
- **Pantheon+**: Scolnic et al. (2022), Brout et al. (2022)
