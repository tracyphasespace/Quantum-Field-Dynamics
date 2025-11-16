# DES-SN5YR Preprocessing Pipeline

## Overview

This document describes the preprocessing pipeline that converts raw DES-SN5YR SNANA FITS files into a unified CSV format compatible with the QFD supernova analysis pipeline.

## Pipeline Architecture

```
DES-SN5YR-1.2.zip (1.5 GB)
    ↓
[1. Download & Extract]
    ↓
SNANA FITS Files
    ├── HEAD: Metadata (SNID, z, RA/Dec, host properties)
    └── PHOT: Raw photometry (MJD, band, FLUXCAL, errors)
    ↓
[2. Read FITS Tables]
    ↓
[3. Filter Type Ia SNe]
    ├── Spectroscopic confirmation (SNTYPE=1)
    └── Photometric candidates (SNTYPE=0, 4)
    ↓
[4. Apply Quality Cuts]
    ├── Min observations ≥ 5
    ├── Redshift range [0.05, 1.3]
    └── Valid coordinates
    ↓
[5. Extract Photometry]
    ├── Use PTROBS_MIN/MAX pointers
    ├── Convert FLUXCAL → Jy
    ├── Calculate SNR
    └── Add metadata
    ↓
[6. Combine & Export]
    ↓
lightcurves_unified_full.csv (122 MB)
    ↓
[7. Validate]
    ↓
Ready for QFD Pipeline
```

## Processing Steps

### Step 1: Download Data

**Script**: `scripts/download_data.py` (optional, can use wget/browser)

```bash
wget -O data/DES-SN5YR-1.2.zip \
    "https://zenodo.org/records/12720778/files/DES-SN5YR-1.2.zip?download=1"
```

**Output**:
- `data/DES-SN5YR-1.2.zip` (1,534,568,826 bytes)
- MD5: 9019a6ddc569553bc323e9e1b68a55bf

### Step 2: Extract Archive

```bash
unzip DES-SN5YR-1.2.zip -d data/raw/
```

**Output**:
```
data/raw/DES-SN5YR-1.2/
├── 0_DATA/
│   ├── DES-SN5YR_DES/          # Main DES dataset
│   ├── DES-SN5YR_LOWZ/         # Low-z compilation
│   └── DES-SN5YR_Foundation/   # Foundation survey
└── 1_SIMULATIONS/              # Simulated data (not used)
```

### Step 3: Read SNANA FITS Files

**Module**: `src/preprocessing.py`

**Functions**:
- `read_fits_header()`: Read `*_HEAD.FITS.gz` → metadata table
- `read_fits_photometry()`: Read `*_PHOT.FITS.gz` → photometry table

**FITS Structure**:

```python
# HEAD file (one row per SN)
SNID, RA, DEC, REDSHIFT_FINAL, NOBS,
PTROBS_MIN, PTROBS_MAX,  # Pointers to PHOT table
HOSTGAL_LOGMASS, HOSTGAL_LOGSFR, MWEBV, ...

# PHOT file (all SNe concatenated)
MJD, BAND, FLUXCAL, FLUXCALERR, ZEROPT,
PSF_SIG1, SKY_SIG, FIELD, PHOTFLAG, ...
```

**Key insight**: Use `PTROBS_MIN` and `PTROBS_MAX` to extract each SN's light curve from the concatenated PHOT table.

### Step 4: Filter Type Ia Supernovae

**Function**: `filter_type_ia()`

**Selection criteria**:

| SNTYPE | Classification | Include? | Count |
|--------|----------------|----------|-------|
| 0 | Unclassified (likely Ia) | ✓ | 19,138 |
| 1 | Spec-confirmed Ia | ✓ | 353 |
| 4 | Ia-peculiar | ✓ | 1 |
| 23, 29, 32, 33, 39 | Core-collapse SNe | ✗ | 110 |
| 41, 66 | Super-luminous SNe | ✗ | 19 |
| 80, 81, 82 | AGN, TDE, M-star | ✗ | 52 |

**Result**: 19,492 Type Ia candidates (spec + photo)

**Options**:
- `--spec-only`: Only SNTYPE=1 (353 SNe, highest purity)
- Default: SNTYPE ∈ {0, 1, 4} (19,492 SNe, statistical power)

### Step 5: Apply Quality Cuts

**Function**: `filter_quality()`

**Cuts**:

| Parameter | Min | Max | Rationale |
|-----------|-----|-----|-----------|
| `NOBS` | 5 | - | Need multi-epoch data for fitting |
| `REDSHIFT_FINAL` | 0.05 | 1.3 | Avoid peculiar velocity; limit high-z uncertainty |
| `RA`, `DEC` | Valid coords | - | Basic sanity check |

**Optional cuts** (not applied by default):
- `HOSTGAL_NMATCH > 0`: Require host galaxy match
- `NOBS ≥ 10`: Stricter observation requirement
- Peak coverage: ≥1 obs within ±10 days of estimated peak

**Result**: 6,895 SNe pass all cuts

### Step 6: Extract Photometry

**Function**: `extract_sn_photometry()`

**Process**:

1. **Pointer indexing**:
   ```python
   idx_min = PTROBS_MIN - 1  # Convert 1-indexed to 0-indexed
   idx_max = PTROBS_MAX - 1
   sn_phot = phot_table[idx_min:idx_max+1]
   ```

2. **Filter end markers**:
   ```python
   valid = sn_phot['MJD'] > 0  # MJD=-777 marks end of LC
   sn_phot = sn_phot[valid]
   ```

3. **Convert FLUXCAL to Jy**:
   ```python
   mag_AB = ZEROPT - 2.5 * log10(|FLUXCAL|)
   flux_jy = 3631 * 10^(-0.4 * mag_AB)
   # Preserve sign for negative fluxes
   flux_jy = sign(FLUXCAL) * flux_jy
   ```

4. **Calculate SNR**:
   ```python
   snr = |flux_jy| / fluxerr_jy
   ```

5. **Map bands to wavelengths**:
   ```python
   wavelength_nm = {
       'g': 475, 'r': 635, 'i': 780, 'z': 915
   }[band]
   ```

6. **Combine with metadata**:
   - Add SNID, z, RA, Dec from header
   - Add host properties if available
   - Add observational metadata (PSF, sky, field)

### Step 7: Combine and Export

**Function**: `process_des_sn5yr_dataset()`

**Process**:
1. Loop over all selected SNe (6,895)
2. Extract photometry for each SN
3. Concatenate into single DataFrame
4. Write to CSV

**Output**:
- `data/processed/lightcurves_unified_full.csv`
- 649,682 rows × 19 columns
- 121.8 MB

### Step 8: Validation

**Script**: `scripts/validate_data.py`

**Checks**:
1. ✓ Required columns present
2. ✓ No nulls in critical fields
3. ✓ Valid value ranges (MJD, z, RA/Dec, flux errors)
4. ✓ Data completeness (all SNe have ≥5 obs)
5. ✓ Redshift distribution (93% in primary range)
6. ✓ Band coverage (all 4 DES bands present)
7. ✓ SNR distribution (expected low median for raw photometry)

**Result**: All checks pass ✓

## Performance

**Timing** (on typical workstation):

| Step | Time | Memory |
|------|------|--------|
| Download | ~2-5 min | - |
| Extract | ~30 sec | 1.5 GB disk |
| Read FITS | ~10 sec | ~500 MB RAM |
| Filter SNe | <1 sec | - |
| Extract photometry (6,895 SNe) | ~90 sec | ~200 MB RAM |
| Write CSV | ~5 sec | - |
| Validate | ~10 sec | ~200 MB RAM |
| **Total** | **~3-8 min** | **~500 MB RAM** |

**Bottleneck**: Reading gzipped FITS files (~10 sec), extracting photometry (~90 sec)

**Optimization**: Process LOWZ, Foundation, and DES datasets in parallel to save time.

## Reproducibility

### Exact Command

```bash
cd projects/astrophysics/Full_Supernova

# Full dataset (6,895 SNe)
python3 scripts/extract_raw_photometry.py \
    --data-dir data/raw/DES-SN5YR-1.2/0_DATA \
    --output data/processed/lightcurves_unified_full.csv \
    --dataset DES-SN5YR_DES \
    --min-obs 5 \
    --min-z 0.05 \
    --max-z 1.3

# Validate
python3 scripts/validate_data.py \
    data/processed/lightcurves_unified_full.csv
```

### Expected Output

```
Extracted 6895 supernovae
Total measurements: 649,682
File size: 121.8 MB
✓ ALL VALIDATION CHECKS PASSED
```

### Checksums

**Input**:
- `DES-SN5YR-1.2.zip`: MD5 = `9019a6ddc569553bc323e9e1b68a55bf`

**Output** (may vary slightly due to floating point precision):
- `lightcurves_unified_full.csv`: ~122 MB, 649,683 lines (1 header + 649,682 data)

## Customization Options

### Stricter Quality Cuts

```bash
python3 scripts/extract_raw_photometry.py \
    --min-obs 10 \        # Require 10+ observations
    --min-z 0.1 \         # Avoid low-z peculiar velocity
    --max-z 1.0 \         # Avoid high-z uncertainty
    --spec-only           # Only spec-confirmed Type Ia
```

**Result**: ~200-300 SNe (highest quality subsample)

### Test Subset

```bash
python3 scripts/extract_raw_photometry.py \
    --max-sne 100 \       # Only process first 100 SNe
    --output data/processed/lightcurves_test_100sne.csv
```

**Use case**: Quick testing, algorithm development

### Other Datasets

```bash
# LOWZ (low-redshift compilation)
python3 scripts/extract_raw_photometry.py \
    --dataset DES-SN5YR_LOWZ \
    --output data/processed/lightcurves_lowz.csv

# Foundation (Foundation Supernova Survey)
python3 scripts/extract_raw_photometry.py \
    --dataset DES-SN5YR_Foundation \
    --output data/processed/lightcurves_foundation.csv
```

## Error Handling

**Common issues**:

### 1. Missing FITS files

**Error**: `FileNotFoundError: HEAD file not found`

**Fix**: Ensure DES-SN5YR-1.2.zip is extracted to `data/raw/`

### 2. Memory issues

**Error**: `MemoryError` or system slowdown

**Fix**: Use `--max-sne` to process in batches, then concatenate CSVs

### 3. Flux conversion warnings

**Warning**: `RuntimeWarning: divide by zero` or `invalid value in log10`

**Expected**: Some measurements have FLUXCAL=0 or negative. These are handled gracefully (assigned SNR=0).

### 4. FITS read errors

**Error**: `OSError: Not a FITS file`

**Fix**: Re-download DES-SN5YR-1.2.zip (file may be corrupted)

## Next Steps

After successful preprocessing:

1. **Run QFD Stage 1** (per-SN optimization):
   ```bash
   cd ../qfd-supernova-v15
   python src/stage1_optimize.py \
       --lightcurves ../Full_Supernova/data/processed/lightcurves_unified_full.csv \
       --out ../Full_Supernova/results/stage1_full \
       --workers 8
   ```

2. **Run QFD Stage 2** (global MCMC):
   ```bash
   python src/stage2_mcmc_numpyro.py \
       --stage1-results ../Full_Supernova/results/stage1_full \
       --lightcurves ../Full_Supernova/data/processed/lightcurves_unified_full.csv \
       --out ../Full_Supernova/results/stage2_full \
       --nchains 4 --nsamples 4000
   ```

3. **Run QFD Stage 3** (Hubble diagram):
   ```bash
   python src/stage3_hubble_optimized.py \
       --stage1-results ../Full_Supernova/results/stage1_full \
       --stage2-results ../Full_Supernova/results/stage2_full \
       --lightcurves ../Full_Supernova/data/processed/lightcurves_unified_full.csv \
       --out ../Full_Supernova/results/stage3_full
   ```

## References

**Preprocessing Code**:
- `src/preprocessing.py`: Core extraction functions
- `scripts/extract_raw_photometry.py`: User-facing extraction script
- `scripts/validate_data.py`: Data validation

**Data Format**:
- `docs/DATA_FORMAT.md`: Detailed column descriptions
- `README.md`: Project overview and quick start

**DES-SN5YR Documentation**:
- `data/raw/DES-SN5YR-1.2/0_DATA/README.md`: Official data format docs
- SNANA manual: https://github.com/RickKessler/SNANA

---

**Version**: 1.0
**Last Updated**: 2025-11-16
**Author**: QFD Research Team
