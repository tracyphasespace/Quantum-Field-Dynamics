# QFD Supernova V15 - Data

This directory contains the filtered lightcurve dataset used in the V15 analysis.

## Provided Dataset

**File**: `lightcurves_unified_v2_min3.csv` (13 MB)

**Source**: DES-SN5YR (Dark Energy Survey 5-Year Supernova Program)

**Contents**:
- 5,468 Type Ia supernovae from DES
- Pre-filtered: ≥3 observations per band minimum
- 118,218 photometric observations total
- Bands: g, r, i, z (optical)
- Redshift range: 0.05 < z < 1.0

**Columns**:
- `snid`: Supernova ID
- `mjd`: Modified Julian Date
- `flux_nu_jy`: Calibrated flux in Janskys
- `flux_nu_jy_err`: Flux uncertainty
- `z`: Heliocentric redshift
- `band`: Observation band (g, r, i, z)
- `survey`: Source survey (DES)
- `source_dataset`: DES-SN5YR
- `ra`, `dec`: Celestial coordinates
- `wavelength_eff_nm`: Effective wavelength in nanometers
- `snr`: Signal-to-noise ratio

## Building Your Own Dataset from DES-SN5YR

If you want to build the dataset from scratch using DES-SN5YR public data:

### Prerequisites

1. **DES-SN5YR Data** (Public Release)
   - Download from: https://des.ncsa.illinois.edu/releases/sn
   - Files needed:
     - `DES-SN5YR_PHOT.FITS.gz` (~2 GB) - Photometry
     - `DES-SN5YR_SPEC.FITS.gz` (~10 MB) - Spectroscopic redshifts

### Build Instructions

#### Step 1: Convert DES-SN5YR FITS to CSV

```bash
# Use the provided conversion script
python ../../../October_Supernova/tools/convert_des_fits_to_qfd.py \
    --phot DES-SN5YR_PHOT.FITS \
    --spec DES-SN5YR_SPEC.FITS \
    --out des_sn5yr_converted.csv
```

**What this does**:
- Extracts photometry from FITS tables
- Matches spectroscopic redshifts
- Converts DES flux units to standard format
- Applies quality cuts (detections, flags)

#### Step 2: Apply Quality Filters

```bash
# Apply minimum observation filtering
python ../../../October_Supernova/tools/filter_lightcurves.py \
    --input des_sn5yr_converted.csv \
    --min-obs-per-band 3 \
    --out lightcurves_unified_v2_min3.csv
```

**Filtering applied**:
- Minimum 3 observations per band
- Valid redshift (0 < z < 2)
- Flux uncertainty > 0
- Remove flagged observations

### Expected Output

```
Total SNe: 5,468
Total observations: 118,218
Survey: DES-SN5YR
Redshift range: 0.05 - 1.0
Bands: g, r, i, z
```

### Unit Conversions

**DES-SN5YR → Standard (Janskys)**:
- DES native flux units are converted to Janskys (Jy)
- 1 Jy = 10^(-23) erg/s/cm²/Hz
- Zero-point: AB mag 0 = 3631 Jy
- Our provided dataset is already in Janskys (`flux_nu_jy` column)

## Data Format Specification

The unified CSV must have these columns for V15 compatibility:

| Column | Type | Description | Required |
|--------|------|-------------|----------|
| `snid` | str | Unique supernova ID | Yes |
| `mjd` | float | Modified Julian Date | Yes |
| `flux_g` | float | g-band flux (erg/s/cm²/Hz) | Yes |
| `flux_r` | float | r-band flux | Yes |
| `flux_i` | float | i-band flux | Yes |
| `flux_z` | float | z-band flux | Yes |
| `fluxerr_g` | float | g-band flux error | Yes |
| `fluxerr_r` | float | r-band flux error | Yes |
| `fluxerr_i` | float | i-band flux error | Yes |
| `fluxerr_z` | float | z-band flux error | Yes |
| `z` | float | Heliocentric redshift | Yes |
| `survey` | str | Source survey name | Optional |

## Quality Filtering in V15

After loading, V15 Stage 1 applies additional filters:

1. **Per-observation**: σ_flux < flux (SNR > 1)
2. **Per-SN**: Must have ≥5 observations total
3. **Stage 1 output**: χ²/obs < 2000 (keeps 5,124/5,468 SNe = 93.7%)

## Data Acknowledgments

**DES-SN5YR**: Dark Energy Survey Collaboration

This analysis uses data from the Dark Energy Survey (DES) 5-Year Supernova Program.

**Citation**:
- DES Collaboration, Abbott, T. M. C., et al. "The Dark Energy Survey: Data Release 2" ApJS, 2021
- Brout, D., et al. "The Dark Energy Survey Supernova Program: Cosmological Analysis and Systematic Uncertainties" ApJ, 2022

**Data Access**: https://des.ncsa.illinois.edu/releases/sn

Please cite original surveys when using this dataset.

## File Size Note

The filtered CSV (13 MB) is tracked in git. If you generate a larger unfiltered dataset, add it to `.gitignore` to avoid bloating the repo.
