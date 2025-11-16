# QFD Supernova Analysis V19

**Multi-Type Supernova Analysis Using Full DES-SN5YR Dataset**

## Overview

V19 extends the QFD supernova analysis to include **all supernova types**, not just Type Ia. This allows testing whether QFD predictions apply universally across different transient classes.

**Key Innovation**: Analyzes the complete DES-SN5YR sample (19,706 SNe) including:
- Type Ia supernovae (normal, peculiar, photometric candidates)
- Core-collapse supernovae (Type II, IIb, Ib, Ic, IIn)
- Super-luminous supernovae (SLSN-I, SLSN-II)
- Other transients (AGN, TDE, M-stars)

## Dataset Composition

### DES-SN5YR Full Sample (19,706 Total)

| Type | SNTYPE | Count | Percentage | Description |
|------|--------|-------|------------|-------------|
| **Unclassified** | 0 | 19,138 | 97.1% | Likely Type Ia (photometric) |
| **Type Ia** | 1 | 353 | 1.8% | Spec-confirmed Ia |
| **Type Ia-pec** | 4 | 1 | <0.1% | Peculiar Ia |
| **Type IIb** | 23 | 5 | <0.1% | Core-collapse |
| **Type II** | 29 | 44 | 0.2% | Core-collapse |
| **Type II?** | 129 | 16 | <0.1% | Probable II |
| **Type IIn?** | 122 | 1 | <0.1% | Probable IIn |
| **Type Ib** | 32 | 1 | <0.1% | Core-collapse |
| **Type Ic** | 33 | 4 | <0.1% | Core-collapse |
| **Type Ibc** | 39 | 39 | 0.2% | Core-collapse |
| **Type Ibc?** | 139 | 4 | <0.1% | Probable Ibc |
| **Type I** | 5 | 1 | <0.1% | Unspecified Type I |
| **SLSN-I** | 41 | 15 | <0.1% | Super-luminous Ia-like |
| **SLSN-I?** | 141 | 3 | <0.1% | Probable SLSN-I |
| **SLSN-II** | 66 | 1 | <0.1% | Super-luminous II-like |
| **AGN** | 80 | 49 | 0.2% | Active Galactic Nucleus |
| **AGN?** | 180 | 1 | <0.1% | Probable AGN |
| **TDE** | 81 | 1 | <0.1% | Tidal Disruption Event |
| **M-star** | 82 | 2 | <0.1% | M-dwarf flare |

**Total**: 19,706 objects

## Scientific Motivation

### Why Include All Types?

1. **Universal QFD Test**: Does the QFD distance-redshift relation apply to all transients, or just standard candles?

2. **Systematics Check**: Different SN types have:
   - Different intrinsic brightness ranges
   - Different spectral energy distributions
   - Different light curve shapes
   - If QFD model fits all types → strong evidence for universal physics

3. **Population Studies**:
   - Stratify results by SNTYPE
   - Identify which types follow QFD vs. outliers
   - Core-collapse SNe are NOT standard candles → higher scatter expected

4. **Outlier Classification**:
   - Are BBH outliers specific to Type Ia?
   - Do core-collapse SNe show different lensing signatures?
   - Can SNTYPE help distinguish lensing from intrinsic scatter?

## Key Differences from V18

| Feature | V18 | V19 |
|---------|-----|-----|
| **Sample** | Type Ia only (19,492 SNe) | All types (19,706 SNe) |
| **Assumption** | Standard candles | Heterogeneous population |
| **Scatter** | σ_ln_A ~ 1.0 (intrinsic) | Larger scatter (multi-type) |
| **Analysis** | Single-population fit | Stratified by SNTYPE |
| **Outliers** | BBH lensing candidates | BBH + intrinsic variety |

## V19 Pipeline Stages

### Stage 1: Individual SN Optimization
**Unchanged** - Fits QFD light curve model to each SN independently.

**Parameters per SN**:
- `t0`: Peak time
- `ln_A`: Log-amplitude (brightness)
- `A_plasma`: Plasma density parameter
- `beta`: Thermal parameter

**Addition**: Extract and store `SNTYPE` for each SN.

### Stage 2: Global MCMC Inference
**Modified** - Options for single-population vs. multi-population fitting.

**Option 2A: Single Population (Baseline)**
- Fit all SNe together ignoring type
- Larger σ_ln_A expected (captures intrinsic diversity)
- Tests universal QFD hypothesis

**Option 2B: Type-Stratified**
- Separate {k_J, η', ξ} for each major type
- Example: Fit Type Ia vs. core-collapse separately
- Tests type-dependent QFD parameters

**Global Parameters**:
- `k_J_correction`: J-band correction
- `eta_prime`: QFD plasma veil (η')
- `xi`: QFD refraction (ξ)
- `sigma_ln_A`: Intrinsic scatter (larger for multi-type)

### Stage 3: Hubble Diagram & Type Analysis
**Enhanced** - Stratified residual analysis.

**Outputs**:
- Hubble diagram color-coded by SNTYPE
- Residuals vs. z for each type
- RMS by type
- Outlier detection (accounting for type-dependent scatter)

## Installation

Same as V18:
```bash
cd projects/V19
pip install -r requirements.txt
```

## Data Extraction

### Extract Full DES-SN5YR Dataset (All Types)

```bash
cd projects/V19

# Extract all 19,706 SNe (no Type Ia filter)
python scripts/extract_full_dataset.py \
    --data-dir data/raw/DES-SN5YR-1.2/0_DATA \
    --output data/processed/lightcurves_full_all_types.csv \
    --include-all-types \
    --min-obs 5 \
    --min-z 0.05 \
    --max-z 1.3
```

**Output**: `lightcurves_full_all_types.csv` with additional `sntype` column.

### Test Extraction (10 SNe, All Types)

```bash
python scripts/extract_full_dataset.py \
    --data-dir data/raw/DES-SN5YR-1.2/0_DATA \
    --output data/processed/lightcurves_test_all_types.csv \
    --include-all-types \
    --max-sne 10
```

## Usage

### Stage 1: Optimize All SNe

```bash
python pipeline/stages/stage1_optimize_v19.py \
    --lightcurves data/processed/lightcurves_full_all_types.csv \
    --out results/stage1_full_all_types \
    --ncores 8
```

### Stage 2A: Single-Population MCMC

```bash
python pipeline/stages/stage2_mcmc_v19_single.py \
    --lightcurves data/processed/lightcurves_full_all_types.csv \
    --stage1-results results/stage1_full_all_types \
    --out results/stage2_single_population \
    --max-sne 1000 \
    --nwalkers 32 \
    --nsteps 2000
```

### Stage 2B: Type-Stratified MCMC

```bash
python pipeline/stages/stage2_mcmc_v19_stratified.py \
    --lightcurves data/processed/lightcurves_full_all_types.csv \
    --stage1-results results/stage1_full_all_types \
    --out results/stage2_stratified \
    --type-groups "Ia:0,1,4" "CC:23,29,32,33,39" "Other:41,66,80,81,82" \
    --max-sne 1000
```

### Stage 3: Hubble Diagram (Color-Coded by Type)

```bash
python pipeline/stages/stage3_v19.py \
    --lightcurves data/processed/lightcurves_full_all_types.csv \
    --stage1-results results/stage1_full_all_types \
    --stage2-results results/stage2_single_population \
    --out results/stage3_hubble_by_type \
    --color-by-type \
    --outlier-sigma-threshold 2.5
```

## Expected Results

### Single-Population Fit

**Hypothesis**: If QFD is universal, all SNe should follow same distance-redshift relation.

**Expected**:
- Larger σ_ln_A (~ 1.5-2.0 instead of 1.0)
- Core-collapse SNe contribute more scatter
- Overall RMS residual: ~3-4 mag (vs. 2.4 mag for Ia-only)

### Type-Stratified Analysis

**Predicted Trends**:
- **Type Ia**: Tight relation (RMS ~ 2.4 mag)
- **Core-collapse**: Higher scatter (RMS ~ 4-6 mag) - NOT standard candles
- **SLSN**: Very bright, possible systematic offset
- **AGN**: Variable, may not follow SN relation

**Key Test**: Do {k_J, η', ξ} differ significantly by type?
- If YES → Type-dependent QFD physics
- If NO → Universal QFD parameters

## Output Files

### Data Files
```
data/processed/
├── lightcurves_full_all_types.csv        # Full 19,706 SNe
└── lightcurves_test_all_types.csv        # Test 10 SNe
```

**Additional column**: `sntype` (integer SNTYPE from FITS header)

### Results
```
results/
├── stage1_full_all_types/
│   └── <snid>_fit.json                   # Includes SNTYPE
├── stage2_single_population/
│   ├── samples.npz                       # MCMC chains
│   └── summary.json                      # Fit results
├── stage2_stratified/
│   ├── samples_typeIa.npz                # Type Ia chains
│   ├── samples_CC.npz                    # Core-collapse chains
│   └── summary_by_type.json              # Type-stratified results
└── stage3_hubble_by_type/
    ├── hubble_diagram_by_type.png        # Color-coded plot
    ├── residuals_by_type.png             # Stratified residuals
    └── summary_by_type.json              # RMS, outliers per type
```

## Validation

### Quality Checks

1. **Type Ia Subset Match**: V19 Type Ia results should match V18 within uncertainties
2. **Scatter Increase**: σ_ln_A should increase when including non-Ia types
3. **Core-Collapse Scatter**: CC SNe should have RMS ~ 2-3× larger than Ia
4. **Population Fractions**: Confirm ~97% Ia, ~3% other types

## Scientific Questions V19 Can Answer

1. **Universal QFD?** Do all transient types follow the same QFD distance relation?
2. **Type-Dependent Physics?** Are {k_J, η', ξ} different for core-collapse vs. thermonuclear?
3. **BBH Lensing Universality?** Do all types show 5:1 dark/bright outlier ratio?
4. **Intrinsic vs. Lensing Scatter?** Can SNTYPE stratification separate intrinsic diversity from BBH effects?

## Limitations

- Core-collapse SNe are NOT standard candles → large intrinsic scatter expected
- Small sample sizes for rare types (SLSN, TDE) → poor statistical power
- AGN are variable, not explosions → may not follow SN physics
- Unclassified (SNTYPE=0) are mixed population → mostly Ia but some contamination

## Future Extensions

- **V20**: Type-dependent QFD model with separate physics for CC vs. thermonuclear
- **V21**: Machine learning SNTYPE classification from light curve features
- **V22**: Joint fit of all DES datasets (DES + LOWZ + Foundation)

## Project Structure

```
V19/
├── README.md                              # This file
├── requirements.txt                       # Python dependencies
├── docs/
│   ├── DATA_FORMAT.md                     # Full dataset format
│   ├── SNTYPE_GUIDE.md                    # SNTYPE definitions
│   └── STRATIFIED_ANALYSIS.md             # Multi-type methodology
├── scripts/
│   ├── extract_full_dataset.py            # All-type extraction
│   └── validate_data.py                   # QC checks
├── pipeline/
│   ├── core/
│   │   ├── v19_data.py                    # Multi-type data loader
│   │   ├── v19_lightcurve_model.py        # Light curve model
│   │   └── v19_qfd_model.py               # QFD cosmology
│   └── stages/
│       ├── stage1_optimize_v19.py         # Per-SN optimization
│       ├── stage2_mcmc_v19_single.py      # Single-population MCMC
│       ├── stage2_mcmc_v19_stratified.py  # Type-stratified MCMC
│       └── stage3_v19.py                  # Type-coded Hubble diagram
├── data/
│   ├── raw/                               # DES-SN5YR FITS files
│   └── processed/                         # Extracted CSVs
└── results/                               # Pipeline outputs
```

## Citation

If you use V19 analysis, cite:

```bibtex
@software{qfd_v19_multitype,
  author       = {Tracy Phase Space},
  title        = {QFD Multi-Type Supernova Analysis V19},
  year         = 2025,
  publisher    = {GitHub},
  url          = {https://github.com/tracyphasespace/Quantum-Field-Dynamics}
}
```

Also cite the DES-SN5YR dataset (see V18 README).

## References

- **V18 README**: Type Ia-only analysis
- **Full_Supernova**: Raw data extraction methodology
- **DES-SN5YR**: https://doi.org/10.5281/zenodo.12720778

---

**Version**: 1.0
**Status**: In Development
**Last Updated**: 2025-11-16
**Key Innovation**: First QFD analysis using complete multi-type SN sample
