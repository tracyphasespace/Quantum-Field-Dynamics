# DES-SN5YR Data Format Documentation

## Overview

This document describes the format of the extracted raw photometry data from the DES-SN5YR dataset.

**Source**: DES 5-Year Supernova Program (DES-SN5YR)
**Original Format**: SNANA FITS files (HEAD + PHOT tables)
**Processed Format**: Unified CSV compatible with QFD pipeline
**Version**: Full dataset extracted 2025-11-16

## File Information

**Location**: `data/processed/lightcurves_unified_full.csv`

**Statistics**:
- **Supernovae**: 6,895 Type Ia (spec-confirmed + photometric candidates)
- **Measurements**: 649,682 photometric observations
- **Redshift Range**: 0.050 - 1.298
- **Bands**: g, r, i, z (DES filters)
- **File Size**: ~122 MB (uncompressed CSV)

## CSV Columns

### Required Columns (for QFD Pipeline)

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `snid` | int | - | Supernova unique identifier |
| `mjd` | float | days | Modified Julian Date of observation |
| `band` | str | - | Filter/band (g, r, i, z) |
| `flux_nu_jy` | float | Jy | Calibrated flux in Janskys (can be negative) |
| `flux_nu_jy_err` | float | Jy | 1-sigma uncertainty on flux |
| `wavelength_eff_nm` | float | nm | Effective wavelength of band |
| `z` | float | - | Redshift (REDSHIFT_FINAL from header) |
| `ra` | float | deg | Right ascension (J2000) |
| `dec` | float | deg | Declination (J2000) |
| `survey` | str | - | Survey name (typically "DES") |

### Additional Metadata Columns

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `snr` | float | - | Signal-to-noise ratio (flux/flux_err) |
| `zeropoint` | float | mag | Photometric zeropoint for this observation |
| `psf_fwhm` | float | arcsec | PSF full-width half-maximum |
| `sky_sig` | float | ADU/pix | Sky noise level |
| `field` | str | - | DES field name (e.g., "X2", "C3") |
| `photflag` | int | - | Photometry flag bit-mask |
| `host_logmass` | float | log(M☉) | Host galaxy stellar mass (if available) |
| `host_logsfr` | float | log(M☉/yr) | Host galaxy star formation rate |
| `mwebv` | float | mag | Milky Way E(B-V) extinction |

## Data Format Details

### Flux Conversion

Raw SNANA FLUXCAL values are converted to Janskys using:

```
mag_AB = ZEROPT - 2.5 * log10(|FLUXCAL|)
flux_nu [Jy] = 3631 * 10^(-0.4 * mag_AB)
```

**Important**: Negative FLUXCAL values (non-detections) are preserved with negative sign in `flux_nu_jy`.

### Band Effective Wavelengths

Standard DES filter effective wavelengths:

| Band | λ_eff (nm) |
|------|------------|
| g | 475 |
| r | 635 |
| i | 780 |
| z | 915 |

### Redshift

Uses `REDSHIFT_FINAL` from DES-SN5YR:
- Spectroscopic redshift if available (highest priority)
- Photometric redshift otherwise
- CMB frame (no peculiar velocity correction applied in this field)

### Quality Filters Applied

The extracted dataset includes SNe that pass the following criteria:

1. **Type**: SNTYPE ∈ {0, 1, 4}
   - 0: Unclassified (mostly photometric Type Ia)
   - 1: Spectroscopically confirmed Type Ia
   - 4: Type Ia-peculiar
2. **Observations**: ≥5 photometric measurements
3. **Redshift**: 0.05 < z < 1.3
4. **Valid coordinates**: RA, Dec within valid ranges

**Not Applied** (user can filter later if needed):
- Host galaxy matching requirement
- Minimum SNR cut
- Peak coverage requirement
- Chi-squared quality cuts

## Data Quality

### Redshift Distribution

| z Range | N_SNe | Fraction |
|---------|-------|----------|
| 0.05-0.1 | 80 | 1.2% |
| 0.1-0.2 | 491 | 7.1% |
| 0.2-0.3 | 969 | 14.1% |
| 0.3-0.5 | 1,879 | 27.3% |
| 0.5-0.7 | 1,712 | 24.8% |
| 0.7-1.0 | 1,311 | 19.0% |
| 1.0-1.3 | 453 | 6.6% |

**Peak coverage**: z ~ 0.3-0.7 (ideal for cosmological analysis)

### Band Coverage

| Band | N_obs | Fraction |
|------|-------|----------|
| g | 167,674 | 25.8% |
| r | 152,792 | 23.5% |
| i | 161,218 | 24.8% |
| z | 167,998 | 25.9% |

**Well-balanced** across all 4 DES bands (~25% each).

### Signal-to-Noise

- **Median SNR**: 1.38 (typical for raw photometry including baseline)
- **High SNR (>5)**: 17.7% of measurements (peak detections)
- **Low SNR (<3)**: 72.3% of measurements (baseline/tails)

This is **expected** for raw light curves. The QFD pipeline Stage 1 will fit models to extract peak brightness, where SNR is much higher.

### Observations per SN

- **Minimum**: 5 measurements (by design)
- **Average**: 94.2 measurements
- **Maximum**: ~500+ measurements (well-sampled SNe)

Good temporal coverage for fitting light curve models.

## Comparison with V15 Dataset

| Metric | V15 | Full DES | Improvement |
|--------|-----|----------|-------------|
| N_SNe | 5,468 | 6,895 | +26% |
| N_measurements | ~118,000 | 649,682 | +451% |
| Avg obs/SN | ~22 | 94 | +327% |
| Redshift range | 0.05-1.0 | 0.05-1.3 | Extended |
| File size | 13 MB | 122 MB | - |

**Key differences**:
- V15 uses multi-survey compilation (DES + Pantheon+ + Union3)
- Full DES is pure DES-SN5YR (single pipeline, consistent systematics)
- Full DES has much deeper temporal coverage per SN

## Usage Examples

### Load Data

```python
import pandas as pd

df = pd.read_csv("data/processed/lightcurves_unified_full.csv")

print(f"Total SNe: {df['snid'].nunique()}")
print(f"Total measurements: {len(df)}")
print(f"Redshift range: {df['z'].min():.3f} - {df['z'].max():.3f}")
```

### Select Single SN

```python
snid = 1642082
sn_data = df[df['snid'] == snid].copy()

# Sort by MJD for light curve
sn_data = sn_data.sort_values('mjd')

# Separate by band
for band in ['g', 'r', 'i', 'z']:
    band_data = sn_data[sn_data['band'].str.strip() == band]
    print(f"{band}: {len(band_data)} measurements")
```

### Filter by Redshift

```python
# Select SNe in specific z range
z_min, z_max = 0.3, 0.7
df_filtered = df[df['z'].between(z_min, z_max)]

n_sne = df_filtered['snid'].nunique()
print(f"SNe in z ∈ [{z_min}, {z_max}]: {n_sne}")
```

### Filter by SNR

```python
# Keep only high-SNR measurements
df_highsnr = df[df['snr'] > 5]

print(f"High-SNR measurements: {len(df_highsnr):,}")
print(f"Fraction of total: {len(df_highsnr)/len(df)*100:.1f}%")
```

## Known Issues and Caveats

### 1. Band String Padding

The `band` column has trailing spaces (FITS string padding):
```python
df['band'].unique()
# ['g                   ', 'r                   ', ...]
```

**Workaround**: Use `str.strip()` when filtering:
```python
band_data = df[df['band'].str.strip() == 'g']
```

### 2. Negative Fluxes

Many measurements have negative flux (non-detections). This is **physically correct** for raw photometry. Do not filter these out before fitting—they constrain baseline and rise/fall times.

### 3. Low Median SNR

Median SNR ~1.4 is expected because:
- Most observations are baseline (pre-peak, post-peak)
- DES is deep, so includes faint measurements
- Peak SNR is much higher (>10 typically)

The QFD Stage 1 fitter uses all data to constrain the full light curve shape.

### 4. Photometric vs. Spectroscopic Type Ia

Only **353 SNe** are spectroscopically confirmed Type Ia (SNTYPE=1). The remaining **6,542 SNe** are photometric candidates (SNTYPE=0). These are high-purity based on DES selection, but may include ~1-5% contamination from other transient types.

**Recommendation**: For initial tests, use `--spec-only` flag to extract only confirmed SNe. For full statistical power, use the combined sample and account for contamination in systematic uncertainties.

### 5. Host Galaxy Properties

Not all SNe have measured host galaxy properties:
- `host_logmass`: Available for ~60% of SNe
- `host_logsfr`: Available for ~40% of SNe

Missing values are represented as NaN. These are not required for basic QFD fitting but can be used for host-property correlations.

## File Naming Convention

| Filename | Description |
|----------|-------------|
| `lightcurves_unified_full.csv` | Full dataset (6,895 SNe, all quality levels) |
| `lightcurves_test_10sne.csv` | Test dataset (10 SNe, for quick validation) |
| `lightcurves_des_spec.csv` | Spec-confirmed only (353 SNe, high purity) |
| `lightcurves_des_lowz.csv` | LOWZ dataset (separate low-z compilation) |

## References

**Data Release Paper**:
- Sanchez et al. (2024), "The Dark Energy Survey Supernova Program: 5-year Photometry Data Release", arXiv:2406.05046

**Cosmology Paper**:
- DES Collaboration (2024), "The Dark Energy Survey Supernova Program: Cosmological Analysis and Systematic Uncertainties", arXiv:2401.02929

**SNANA Format**:
- Kessler et al. (2009), "SNANA: A Public Software Package for Supernova Analysis", PASP, 121, 1028

**Zenodo Dataset**:
- DOI: 10.5281/zenodo.12720778

---

**Last Updated**: 2025-11-16
**Data Version**: DES-SN5YR-1.2 (Full Release)
**Extraction Code**: `src/preprocessing.py` v1.0
