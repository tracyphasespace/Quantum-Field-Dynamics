# V19 Quick Start Guide

## 1. Download Raw Data

```bash
cd projects/V19/data
wget -O DES-SN5YR-1.2.zip \
    "https://zenodo.org/records/12720778/files/DES-SN5YR-1.2.zip?download=1"
unzip DES-SN5YR-1.2.zip
```

**Alternatively**, if you've already downloaded data for Full_Supernova:
```bash
# Use symbolic link to avoid duplicate download
ln -s ../../astrophysics/Full_Supernova/data/raw raw
```

## 2. Extract Full Multi-Type Dataset

```bash
# All 19,706 SNe (all types)
python3 scripts/extract_full_dataset.py \
    --data-dir data/raw/DES-SN5YR-1.2/0_DATA \
    --output data/lightcurves_full_all_types.csv

# Test with 100 SNe
python3 scripts/extract_full_dataset.py \
    --data-dir data/raw/DES-SN5YR-1.2/0_DATA \
    --output data/lightcurves_test_100_all_types.csv \
    --max-sne 100
```

**Expected output**:
- ~7,087 SNe after quality cuts (min 5 obs, 0.05 < z < 1.3)
- ~650,000 measurements total
- SNTYPE distribution: 92% Type 0 (unclassified), 5% Type 1 (spec Ia), 3% other

## 3. Optional: Exclude Non-SN Transients

```bash
# Exclude AGN, TDE, M-stars (SNTYPE 80, 81, 82)
python3 scripts/extract_full_dataset.py \
    --data-dir data/raw/DES-SN5YR-1.2/0_DATA \
    --output data/lightcurves_supernovae_only.csv \
    --exclude-types 80 81 82 101 180
```

## 4. Validate Data

```bash
# Check SNTYPE distribution
python3 -c "
import pandas as pd
df = pd.read_csv('data/lightcurves_full_all_types.csv')
print(f'Total SNe: {df.snid.nunique()}')
print(f'Total measurements: {len(df)}')
print('\\nSNTYPE distribution:')
print(df.groupby('sntype')['snid'].nunique().sort_values(ascending=False))
"
```

## 5. Run V18 Pipeline on Multi-Type Data

V19 data is compatible with V18 pipeline. The main difference is the additional `sntype` column.

```bash
# Copy V18 pipeline stages (if not already done)
cp -r ../../v18/pipeline .

# Stage 1: Optimize (works for all types)
python3 pipeline/stages/stage1_optimize_v17.py \
    --lightcurves data/lightcurves_full_all_types.csv \
    --out results/stage1_all_types \
    --ncores 8

# Stage 2: MCMC (single-population fit)
python3 pipeline/stages/stage2_mcmc_v18_emcee.py \
    --lightcurves data/lightcurves_full_all_types.csv \
    --stage1-results results/stage1_all_types \
    --out results/stage2_single_pop \
    --max-sne 1000 \
    --nwalkers 32 \
    --nsteps 2000

# Stage 3: Hubble diagram
python3 pipeline/stages/stage3_v18.py \
    --lightcurves data/lightcurves_full_all_types.csv \
    --stage1-results results/stage1_all_types \
    --stage2-results results/stage2_single_pop \
    --out results/stage3_hubble \
    --outlier-sigma-threshold 2.5
```

## 6. Analyze Results by Type

```bash
# Extract Type Ia subset from results
python3 -c "
import pandas as pd
df = pd.read_csv('results/stage3_hubble/hubble_data.csv')
type_ia = df[df['sntype'].isin([0, 1, 4])]
core_collapse = df[df['sntype'].isin([23, 29, 32, 33, 39, 129, 139])]

print(f'Type Ia RMS: {type_ia.residual.std():.3f} mag')
print(f'Core-collapse RMS: {core_collapse.residual.std():.3f} mag')
"
```

## Expected Results

### SNTYPE Distribution (after quality cuts)

| Type | SNTYPE | Expected Count | Percentage |
|------|--------|----------------|------------|
| Unclassified | 0 | ~6,545 | 92.4% |
| Type Ia | 1 | ~349 | 4.9% |
| Type Ia-pec | 4 | ~1 | 0.0% |
| Type II | 29 | ~44 | 0.6% |
| Type Ibc | 39 | ~7 | 0.1% |
| SLSN-I | 41 | ~12 | 0.2% |
| AGN | 80 | ~36 | 0.5% |
| Other | various | ~93 | 1.3% |

### Fit Quality

**Type Ia subset** (SNTYPE 0, 1, 4):
- RMS: ~2.4 mag (similar to V18)
- χ²: Similar to V18

**All types**:
- RMS: ~3-4 mag (higher due to non-standard candles)
- σ_ln_A: ~1.5-2.0 (vs. 1.0 for Ia-only)

### Scientific Insights

1. **Universal QFD?** Compare {k_J, η', ξ} for Ia vs. core-collapse
2. **Type-Dependent Scatter**: Core-collapse should have 2-3× higher RMS
3. **Outliers**: Do all types show 5:1 dark/bright ratio?

## Troubleshooting

### "Too many SNe" error
Use `--max-sne` flag to limit sample size for testing.

### Memory issues
Process in batches or use a machine with 16+ GB RAM.

### SNTYPE column missing
Make sure you're using V19's `extract_full_dataset.py`, not V18/Full_Supernova's preprocessing.

## Next Steps

- **V19 Multi-Population Fit**: Implement type-stratified MCMC (Stage 2B)
- **Type-Specific Analysis**: Create separate Hubble diagrams by type
- **Physics Validation**: Compare QFD predictions across SN classes

---

**Version**: V19.0
**Status**: Data extraction complete, pipeline stages pending
**Last Updated**: 2025-11-16
