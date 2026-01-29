# V21 Supernova Analysis Package - DEPRECATED

**Archived**: 2026-01-29
**Reason**: Superseded by V22 Supernova Analysis

## Why Deprecated

1. **V22 is self-contained**: Uses `data/raw/des5yr_v21_*.csv` files (already extracted)
2. **Duplicate data**: 63MB lightcurves file not used by any current code
3. **External reference**: Extraction scripts used SupernovaSrc external copy, not this one
4. **Missing components**: Was incomplete (missing Stage2/Stage3 from v18)

## Data Provenance

The V21 Stage2 results were converted to distance modulus format and saved to:
- `data/raw/des5yr_v21_exact.csv`
- `data/raw/des5yr_v21_CORRECTED.csv`
- `data/raw/des5yr_v21_SIGN_CORRECTED.csv`

V22 uses these converted files, not the raw Stage2 output in this package.

## If You Need This Data

The raw lightcurve data (`lightcurves_all_transients.csv`) can be regenerated from
the DES-SN5YR-1.2 dataset if needed for future Stage1 reprocessing.
