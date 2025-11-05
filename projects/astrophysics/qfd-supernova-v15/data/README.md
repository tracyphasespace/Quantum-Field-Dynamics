# QFD Supernova V15 - Data

This directory contains the filtered lightcurve dataset used in the V15 analysis.

## Provided Dataset

**File**: `lightcurves_unified_v2_min3.csv` (13 MB)

**Contents**:
- 5,468 Type Ia supernovae
- Combined Pantheon+ and DES-SN5YR surveys
- Pre-filtered: ≥3 observations per band minimum
- 118,218 photometric observations total

**Columns**:
- `snid`: Supernova ID
- `mjd`: Modified Julian Date
- `flux_[band]`: Calibrated flux in each band (g, r, i, z)
- `fluxerr_[band]`: Flux uncertainty
- `z`: Heliocentric redshift
- `survey`: Source survey (Pantheon+ or DES-SN5YR)

## Building Your Own Dataset from DES-SN5YR

If you want to build the unified dataset from scratch using DES-SN5YR + Pantheon+:

### Prerequisites

1. **DES-SN5YR Data** (Public Release)
   - Download from: https://des.ncsa.illinois.edu/releases/sn
   - Files needed:
     - `DES-SN5YR_PHOT.FITS.gz` (~2 GB) - Photometry
     - `DES-SN5YR_SPEC.FITS.gz` (~10 MB) - Spectroscopic redshifts

2. **Pantheon+ Data**
   - Download from: https://github.com/PantheonPlusSH0ES/DataRelease
   - Files needed:
     - `Pantheon+SH0ES.dat` - Light curve data
     - Full photometry tables

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

#### Step 2: Parse Pantheon+ Data

```bash
# Parse Pantheon+ format
python ../../../October_Supernova/tools/parse_pantheon_plus.py \
    --input Pantheon+SH0ES.dat \
    --phot-dir pantheon_plus_photometry/ \
    --out pantheon_plus_converted.csv
```

#### Step 3: Create Unified Dataset

```bash
# Combine surveys with quality filtering
python ../../../October_Supernova/tools/create_unified_dataset.py \
    --des des_sn5yr_converted.csv \
    --pantheon pantheon_plus_converted.csv \
    --min-obs 3 \
    --out lightcurves_unified_custom.csv
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
Surveys: Pantheon+ (1,701 SNe), DES-SN5YR (3,767 SNe)
Redshift range: 0.01 - 1.3
```

### Unit Conversions

**DES-SN5YR → Standard**:
- DES flux is in "maggies" (1 maggie = 3631 Jy)
- Conversion: `flux_standard = flux_des × 3631 × 10^(−23)` (erg/s/cm²/Hz)
- Our code handles this automatically

**Pantheon+ → Standard**:
- Pantheon+ uses AB magnitudes
- Conversion: `flux = 10^(−0.4 × mag) × 3631 × 10^(−23)`
- Zero-point: AB mag 0 = 3631 Jy

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

- **DES-SN5YR**: Dark Energy Survey Collaboration (2024)
- **Pantheon+**: Pantheon+ Team, Scolnic et al. (2022)

Please cite original surveys when using this dataset.

## File Size Note

The filtered CSV (13 MB) is tracked in git. If you generate a larger unfiltered dataset, add it to `.gitignore` to avoid bloating the repo.
