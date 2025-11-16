# V19 Pipeline Adaptation Status

## ✅ Completed

### Core Modules (v19_*.py)
- [x] **v19_data.py**: Data loader with SNTYPE support
  - Added `sntype` field to `SupernovaData` class
  - Added `sntype_filter` parameter to `load()` method
  - Extracts SNTYPE from CSV and includes in loaded data

- [x] **v19_lightcurve_model.py**: Light curve fitting (unchanged from V17)
- [x] **v19_qfd_model.py**: QFD cosmology model (unchanged from V17)
- [x] **pipeline_io.py**: I/O utilities (unchanged)

### Stage 1: Per-SN Optimization
- [x] **stage1_optimize_v19.py**: Adapted for multi-type
  - Updated imports from v17 → v19
  - Accepts `sntype` in task dictionary
  - Saves `sntype` in output JSON for each SN
  - Works with all supernova types (type-agnostic fitting)

### Stage 2: MCMC
- [x] **stage2_mcmc_v19_emcee.py**: Fully adapted for V19
  - ✅ Updated imports v17 → v19
  - ✅ Added `--sntype-filter` argument for type-specific MCMC
  - ✅ Saves SNTYPE distribution in summary.json
  - ✅ Prints SNTYPE distribution during data loading
  - ✅ Passes `sntype_filter` to data loader

### Stage 3: Hubble Diagram
- [x] **stage3_v19.py**: Fully adapted for multi-type visualization
  - ✅ Updated imports v17 → v19
  - ✅ Color-coded Hubble diagram points by SNTYPE
  - ✅ Generated type-stratified residual plots
  - ✅ Computes RMS separately for each type
  - ✅ Saves SNTYPE in CSV output
  - ✅ Includes type-stratified statistics in summary.json
  - ✅ Prints type-specific RMS values to console

## 📋 Optional Enhancements

### 1. Type-Stratified Analysis Scripts (Optional)

Create `pipeline/scripts/analyze_by_type.py`:
```python
"""
Analyze V19 results stratified by SNTYPE.
"""
import pandas as pd
import numpy as np

def compute_type_statistics(hubble_csv):
    df = pd.read_csv(hubble_csv)

    type_groups = {
        'Type Ia': [0, 1, 4],
        'Core-collapse': [23, 29, 32, 33, 39, 129, 139],
        'SLSN': [41, 66, 141],
        'Other': [5, 80, 81, 82, 101, 180]
    }

    for name, types in type_groups.items():
        mask = df['sntype'].isin(types)
        subset = df[mask]
        if len(subset) > 0:
            print(f"\n{name} ({len(subset)} SNe):")
            print(f"  RMS: {subset['residual'].std():.3f} mag")
            print(f"  Mean residual: {subset['residual'].mean():.3f} mag")
            print(f"  Median z: {subset['z'].median():.2f}")
```

## 🎯 Quick Test

Once imports are updated, test the pipeline:

```bash
# Stage 1: Works as-is
python3 pipeline/stages/stage1_optimize_v19.py \
    --lightcurves data/lightcurves_test_all_types_10sne.csv \
    --out results/test_stage1 \
    --ncores 2

# Stage 2: After updating imports
python3 pipeline/stages/stage2_mcmc_v19_emcee.py \
    --lightcurves data/lightcurves_test_all_types_10sne.csv \
    --stage1-results results/test_stage1 \
    --out results/test_stage2 \
    --max-sne 10 \
    --nwalkers 8 \
    --nsteps 100

# Stage 3: After updating imports and plots
python3 pipeline/stages/stage3_v19.py \
    --lightcurves data/lightcurves_test_all_types_10sne.csv \
    --stage1-results results/test_stage1 \
    --stage2-results results/test_stage2 \
    --out results/test_stage3
```

## 📊 Expected Multi-Type Results

| Type | N_SNe | Expected RMS | Notes |
|------|-------|--------------|-------|
| Type Ia (0,1,4) | ~6,895 | ~2.4 mag | Standard candles |
| Core-collapse (II, Ib, Ic) | ~110 | ~4-6 mag | NOT standard candles |
| SLSN | ~19 | ~3-5 mag | Intrinsically bright |
| Other (AGN, etc.) | ~85 | ~5-10 mag | Variable, not SNe |

## 🔬 Scientific Use Cases

### Use Case 1: Type Ia Only (V18 Compatibility)
```bash
python3 scripts/extract_full_dataset.py \
    --sntype-filter 0 1 4 \
    --output data/lightcurves_type_ia_only.csv
# Should match V18 results
```

### Use Case 2: Core-Collapse Physics
```bash
python3 scripts/extract_full_dataset.py \
    --sntype-filter 23 29 32 33 39 129 139 \
    --output data/lightcurves_core_collapse.csv
# Test QFD on non-standard candles
```

### Use Case 3: Full Multi-Type Comparison
```bash
# Already extracted: data/lightcurves_full_all_types.csv
# Compare Ia vs. CC vs. SLSN in single analysis
```

## 📝 Documentation

- [x] README.md: Project overview
- [x] QUICK_START.md: Usage guide
- [x] PIPELINE_STATUS.md: This file
- [ ] **TODO**: Add example Jupyter notebooks
- [ ] **TODO**: Add type-stratified analysis scripts

## ✅ Summary

**V19 Pipeline Status: COMPLETE** 🎉

All core functionality is now operational for multi-type supernova analysis:

1. ✅ **Data extraction with SNTYPE** - Full dataset extraction working
2. ✅ **Data loader with type filtering** - v19_data.py handles SNTYPE filtering
3. ✅ **Stage 1: Per-SN optimization** - Saves SNTYPE metadata for each SN
4. ✅ **Stage 2: MCMC global inference** - Type-stratified MCMC with `--sntype-filter`
5. ✅ **Stage 3: Hubble diagram** - Color-coded plots, type-stratified statistics
6. ✅ **All core modules** - v17 → v19 migration complete

**New V19 Features**:
- SNTYPE filtering at every stage (data, MCMC, visualization)
- Color-coded Hubble diagrams by supernova type
- Type-stratified RMS and statistics in summary.json
- Console output showing per-type performance

**Ready for Science**: The full V19 pipeline is ready to test QFD across all supernova types!

---

**Version**: V19.2
**Status**: COMPLETE - All stages adapted for multi-type analysis
**Last Updated**: 2025-11-16
