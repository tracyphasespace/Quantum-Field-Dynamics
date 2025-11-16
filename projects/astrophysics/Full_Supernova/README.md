# Full DES-SN5YR Supernova Analysis

**Complete raw photometric light curve data for QFD cosmological analysis**

## Overview

This project uses the complete DES 5-Year Supernova Program (DES-SN5YR) raw photometric data to perform QFD cosmological parameter inference. Unlike the V15 pipeline which uses a filtered subset, this project processes the full dataset of raw light curves directly from the DES public release.

## Dataset Information

**Source**: DES-SN5YR Public Data Release
**Zenodo DOI**: 10.5281/zenodo.12720778
**Data Version**: DES-SN5YR-1.2
**Size**: 1.5 GB (compressed)
**License**: Creative Commons Attribution 4.0 International

### Raw Photometry Data

The dataset contains:
- **31,636 light curves** from DIFFIMG (difference imaging) pipeline
- **19,706 high-quality light curves** from SMP (scene modeling photometry) pipeline
- **Photometric measurements**: MJD, band/filter, flux, flux error, zeropoint, sky level, PSF FWHM
- **Metadata**: Host galaxy properties, redshift, survey information
- **Redshift range**: 0.05 < z < 1.3
- **Bands**: g, r, i, z

### What We Keep vs. Discard

**KEEP (Raw Photometry)**:
- Time series: MJD (Modified Julian Date)
- Photometric measurements: flux, flux_err per band
- Observational metadata: zeropoint, PSF FWHM, sky level
- Object metadata: redshift (z), host properties, coordinates

**DISCARD (Processed Products)**:
- SALT2/SALT3 fitted parameters (x0, x1, c)
- Distance moduli (μ) from ΛCDM fits
- Pre-computed covariance matrices
- Classification probabilities (we only want confirmed Type Ia)

## Project Structure

```
Full_Supernova/
├── data/
│   ├── raw/                          # Extracted DES-SN5YR files
│   ├── processed/                    # Cleaned photometry CSVs
│   └── DES-SN5YR-1.2.zip            # Original download
├── scripts/
│   ├── download_data.py              # Automated download from Zenodo
│   ├── extract_raw_photometry.py     # Extract raw light curves
│   ├── validate_data.py              # Quality checks
│   └── create_unified_csv.py         # Convert to QFD pipeline format
├── src/
│   ├── preprocessing.py              # Core preprocessing functions
│   ├── quality_filters.py            # Data quality filtering
│   └── format_converter.py           # Format conversion utilities
├── docs/
│   ├── DATA_FORMAT.md                # Description of raw data format
│   ├── PREPROCESSING.md              # Preprocessing pipeline details
│   └── QUALITY_CRITERIA.md           # Quality selection criteria
├── notebooks/
│   ├── 01_explore_raw_data.ipynb     # Initial data exploration
│   └── 02_validate_preprocessing.ipynb  # Validation checks
└── README.md                         # This file
```

## Quick Start

### 1. Download Data (Automated)

```bash
cd projects/astrophysics/Full_Supernova
python scripts/download_data.py
```

This downloads and verifies the DES-SN5YR-1.2.zip file from Zenodo.

### 2. Extract Raw Photometry

```bash
python scripts/extract_raw_photometry.py \
    --input data/DES-SN5YR-1.2.zip \
    --output data/processed \
    --pipeline DIFFIMG \
    --quality-cut medium
```

**Pipeline options**:
- `DIFFIMG`: 31,636 light curves (all transient candidates)
- `SMP`: 19,706 high-quality light curves (cleaner photometry)

**Quality cuts**:
- `none`: All light curves
- `medium`: Minimum 5 observations, confirmed Type Ia
- `strict`: Minimum 10 observations, low chi2, confirmed Type Ia

### 3. Create Unified CSV for QFD Pipeline

```bash
python scripts/create_unified_csv.py \
    --input data/processed/lightcurves_raw.parquet \
    --output data/processed/lightcurves_unified_full.csv \
    --format qfd
```

This creates a CSV compatible with the QFD V15 pipeline format:
```
snid,z,mjd,flux_nu_jy,flux_nu_jy_err,band,wavelength_eff_nm,survey,ra,dec,source_dataset,snr
```

### 4. Run QFD Pipeline

Use the same pipeline as V15, but with the full dataset:

```bash
# Stage 1: Per-SN optimization (may take 12-24 hours for full dataset)
python ../qfd-supernova-v15/src/stage1_optimize.py \
    --lightcurves data/processed/lightcurves_unified_full.csv \
    --out results/stage1_full \
    --workers 8

# Stage 2: Global parameter inference
python ../qfd-supernova-v15/src/stage2_mcmc_numpyro.py \
    --stage1-results results/stage1_full \
    --lightcurves data/processed/lightcurves_unified_full.csv \
    --out results/stage2_full \
    --nchains 4 --nsamples 4000 --nwarmup 2000

# Stage 3: Hubble diagram and diagnostics
python ../qfd-supernova-v15/src/stage3_hubble_optimized.py \
    --stage1-results results/stage1_full \
    --stage2-results results/stage2_full \
    --lightcurves data/processed/lightcurves_unified_full.csv \
    --out results/stage3_full
```

## Data Quality and Selection

### Minimum Quality Criteria
- **Confirmed Type Ia**: Spectroscopic or photometric classification
- **Redshift**: 0.05 < z < 1.3 (avoid peculiar velocity regime and high-z uncertainty)
- **Observations**: Minimum 5 photometric measurements (preferably multi-band)
- **Peak coverage**: At least 1 observation within ±10 days of peak
- **Photometric quality**: Flux SNR > 3 for at least 3 measurements

### Optional Stricter Cuts (for comparison)
- **Observations**: Minimum 10 measurements
- **Multi-band**: At least 2 bands with 3+ measurements each
- **Cadence**: No gaps > 20 days around peak
- **Host matching**: Secure host galaxy association
- **Chi2 cut**: Stage 1 fit chi2/dof < 2.0

## Expected Improvements Over V15

### Sample Size
- **V15**: 5,468 SNe (filtered subset)
- **Full**: 19,706 - 31,636 SNe (depending on pipeline and cuts)
- **Improvement**: 3.6× - 5.8× more data

### Statistical Precision
- **k_J uncertainty**: Expected ~30-50% reduction (√N improvement)
- **η' uncertainty**: Expected ~30-50% reduction
- **ξ uncertainty**: Expected ~30-50% reduction

### Redshift Coverage
- **V15**: Limited high-z coverage
- **Full**: Better sampling at z > 0.7
- **Benefit**: Tighter constraints on redshift evolution (η')

### Systematic Checks
- **Survey comparison**: Can split DES subsample vs. full sample
- **Pipeline comparison**: DIFFIMG vs. SMP photometry consistency
- **Holdout validation**: Larger holdout set for robust validation

## Citations

When using this dataset, cite:

1. **Data Release Paper**:
   Sanchez et al. (2024), "The Dark Energy Survey Supernova Program: 5-year Photometry Data Release", arXiv:2406.05046

2. **Cosmology Paper**:
   DES Collaboration (2024), "The Dark Energy Survey Supernova Program: Cosmological Analysis and Systematic Uncertainties", arXiv:2401.02929

3. **Zenodo Dataset**:
   DOI: 10.5281/zenodo.12720778

4. **QFD Framework**:
   McSheery, T. (2025), "Quantum Field Dynamics", GitHub: tracyphasespace/Quantum-Field-Dynamics

## References

- **DES-SN5YR GitHub**: https://github.com/des-science/DES-SN5YR
- **Zenodo Record**: https://zenodo.org/records/12720778
- **QFD V15 Pipeline**: ../qfd-supernova-v15/
- **DES Survey Homepage**: https://www.darkenergysurvey.org/

## Status

- [x] Download data from Zenodo
- [ ] Extract and examine raw photometry format
- [ ] Implement preprocessing pipeline
- [ ] Validate against V15 subset
- [ ] Run full Stage 1-3 pipeline
- [ ] Compare results with V15
- [ ] Generate publication figures

## License

This project is part of the Quantum Field Dynamics research framework. The DES-SN5YR data is licensed under CC-BY-4.0.

---

**Created**: 2025-11-16
**Last Updated**: 2025-11-16
**Maintainer**: QFD Research Team
