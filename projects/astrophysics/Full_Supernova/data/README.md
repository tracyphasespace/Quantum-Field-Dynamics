# Data Directory

## Structure

```
data/
├── raw/                          # Raw SNANA FITS files (gitignored)
│   └── DES-SN5YR-1.2/
│       └── 0_DATA/
│           ├── DES-SN5YR_DES/
│           ├── DES-SN5YR_LOWZ/
│           └── DES-SN5YR_Foundation/
├── processed/                    # Processed CSV files (gitignored)
│   ├── lightcurves_unified_full.csv      # Full dataset (6,895 SNe)
│   ├── lightcurves_test_10sne.csv        # Test dataset (10 SNe)
│   └── lightcurves_test_100sne.csv       # Test dataset (100 SNe)
└── DES-SN5YR-1.2.zip            # Original download (gitignored)
```

## Data Files

**Note**: Large data files are **not** committed to git. Download and extract them locally.

### Download Original Data

```bash
cd data
wget -O DES-SN5YR-1.2.zip \
    "https://zenodo.org/records/12720778/files/DES-SN5YR-1.2.zip?download=1"
```

**Size**: 1.5 GB
**MD5**: 9019a6ddc569553bc323e9e1b68a55bf

### Extract Data

```bash
unzip DES-SN5YR-1.2.zip -d raw/
```

### Process Data

```bash
cd ..
python3 scripts/extract_raw_photometry.py \
    --data-dir data/raw/DES-SN5YR-1.2/0_DATA \
    --output data/processed/lightcurves_unified_full.csv
```

## Processed Data Summary

| File | SNe | Measurements | Size | Description |
|------|-----|--------------|------|-------------|
| `lightcurves_unified_full.csv` | 6,895 | 649,682 | 122 MB | Full DES dataset |
| `lightcurves_test_10sne.csv` | 10 | 973 | 0.2 MB | Quick test (in git) |
| `lightcurves_test_100sne.csv` | 100 | ~9,400 | ~2 MB | Medium test |
| `lightcurves_des_spec.csv` | ~353 | ~33,000 | ~6 MB | Spec-confirmed only |

**Only `lightcurves_test_10sne.csv` is tracked in git for testing purposes.**

## Regenerating Data

If you need to regenerate the processed data:

```bash
# Full dataset
python3 ../scripts/extract_raw_photometry.py \
    --data-dir raw/DES-SN5YR-1.2/0_DATA \
    --output processed/lightcurves_unified_full.csv

# Spec-confirmed only
python3 ../scripts/extract_raw_photometry.py \
    --data-dir raw/DES-SN5YR-1.2/0_DATA \
    --output processed/lightcurves_des_spec.csv \
    --spec-only

# Test datasets
python3 ../scripts/extract_raw_photometry.py \
    --data-dir raw/DES-SN5YR-1.2/0_DATA \
    --output processed/lightcurves_test_100sne.csv \
    --max-sne 100
```

## Data Format

See `../docs/DATA_FORMAT.md` for detailed column descriptions and format specifications.

## Storage Requirements

- **Raw data**: ~1.5 GB (compressed), ~4 GB (extracted)
- **Processed CSV**: ~122 MB (full), ~6 MB (spec-only)
- **Total**: ~5-6 GB recommended free space

## Cleanup

To remove all data files (keeping only code):

```bash
# Remove processed files
rm -f processed/*.csv
rm -f processed/*.parquet

# Remove raw data
rm -rf raw/

# Remove original download
rm -f DES-SN5YR-1.2.zip
```

This will reduce the directory to just code and documentation (~5 MB).
