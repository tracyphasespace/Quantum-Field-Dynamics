# V18: QFD Supernova Pipeline

**Self-contained, consolidated implementation of the QFD supernova analysis pipeline**

## For Researchers: Reproducing Paper Results

**NEW**: Complete reproduction guide available! See **[REPRODUCTION_GUIDE.md](REPRODUCTION_GUIDE.md)** for:
- Environment setup (automated)
- Step-by-step instructions
- Expected results and validation
- Troubleshooting and hardware requirements

Quick setup:
```bash
./setup_environment.sh          # Automated environment setup
python tests/test_regression_nov5.py  # Verify results match paper
```

## Quick Start

### Self-Contained Mode (Recommended)
```bash
# Everything runs from v15_clean directory
cd /home/tracy/development/SupernovaSrc/qfd-supernova-v15/v18

# Run full pipeline (Stage 2 → Stage 3)
./scripts/run_full_pipeline.sh
```

### Legacy Mode (From repository root)
```bash
cd /home/tracy/development/SupernovaSrc/qfd-supernova-v15
./v18/scripts/run_full_pipeline.sh
```

## Directory Structure

```
v18/
├── core/
│   ├── v17_qfd_model.py       # QFD physics model
│   ├── v17_data.py            # Lightcurve data loading
│   ├── v17_lightcurve_model.py # Lightcurve model
│   └── pipeline_io.py         # I/O utilities
├── stages/
│   ├── stage1_optimize_v17.py # Per-SN parameter fits
│   ├── stage2_mcmc_v18_emcee.py # Global MCMC (using emcee)
│   └── stage3_v18.py          # Hubble diagram
├── scripts/
│   └── run_full_pipeline.sh   # Main production script
├── tools/
│   ├── analyze_stage1_simple.py     # Stage 1 results analyzer
│   ├── generate_corner_plot.py      # MCMC visualization
│   ├── make_publication_figures.py  # Publication plots
│   ├── compare_abc_variants.py      # A/B/C testing framework
│   └── README.md                    # Tool documentation
├── data/
│   ├── lightcurves_unified_v2_min3.csv  # 5,468 SNe (symlink)
│   └── README.md              # Data documentation
├── results/                   # Output directory (created on run)
├── tests/                     # Unit tests
└── docs/
    ├── STRUCTURE.md           # Code organization
    ├── STATUS.md              # Current pipeline status
    └── DATA_DEPENDENCIES.md   # Data sources and tools
```

## Key Features

### ✅ Bug Fixes Applied
- **Sign fix** in Stage 2 (lines 414, 424, 439, 448 of stage2_mcmc_numpyro.py)
- **Zero-point calibration** in Stage 3 (lines 258-275 of stage3_hubble_optimized.py)

### ✅ Complete Pipeline
- Stage 1: Per-SN parameter optimization (L-BFGS-B)
- Stage 2: Global cosmology MCMC (NumPyro NUTS)
- Stage 3: Hubble diagram analysis

### ✅ Clean Code
- All imports properly configured
- No code sprawl or duplication
- Self-contained (except data prep tools in external repo)

## Data

**Input**: `../data/lightcurves_unified_v2_min3.csv`
- 5,468 Type Ia SNe
- Pantheon+ and DES-SN5YR combined
- See `DATA_DEPENDENCIES.md` for details

**Output**: Results in `../results/v18/`

## Usage

### Run Full Pipeline
```bash
./v18/scripts/run_pipeline_v17.sh
```

### Run Individual Stages
```bash
# Stage 1 (if needed - usually pre-computed)
python v18/stages/stage1_optimize_v17.py \
    --lightcurves data/lightcurves_unified_v2_min3.csv \
    --out results/v18/stage1_fullscale

# Stage 2
python v18/stages/stage2_mcmc_v18_emcee.py \
    --stage1-results results/v18/stage1_fullscale \
    --out results/v18/stage2_production \
    --lightcurves data/lightcurves_unified_v2_min3.csv \
    --use-ln-a-space \
    --constrain-signs informed

# Stage 3  
python v18/stages/stage3_v18.py \
    --stage1-results results/v18/stage1_fullscale \
    --stage2-results results/v18/stage2_production \
    --out results/v18/stage3_hubble
```

## Documentation

- `STRUCTURE.md` - Code organization and file inventory
- `STATUS.md` - Current pipeline status and debugging notes
- `DATA_DEPENDENCIES.md` - Data sources and external tools

## Code Versions

**IMPORTANT**: Only use code from `v18/`. Other directories (`src/`, `2Compare/`) contain outdated versions.

## Current Status

**Code**: Consolidated and ready ✓
**Data**: Provided dataset ready ✓
**Pipeline**: All bugs fixed and validated ✓
**Reproducibility**: Complete documentation and automated setup ✓

**IMPORTANT REGRESSION FIX** (November 12, 2024): A January 2025 "bugfix" accidentally broke the code by adding incorrect negative signs. The regression has been identified and fixed. See:
- `REGRESSION_ANALYSIS.md` - Technical details of the bug
- `PROBLEM_SOLVED.md` - Quick summary
- `RECOVERY_INSTRUCTIONS.md` - Manual fix steps
- `tests/test_regression_nov5.py` - Regression test to prevent future breakage

**Expected Results** (matching November 5, 2024 golden reference):
- k_J ≈ 10.7 ± 4.6
- η' ≈ -8.0 ± 1.4
- ξ ≈ -7.0 ± 3.8
- Using 4,727 SNe (not 548!)

## Documentation

- **[REPRODUCTION_GUIDE.md](REPRODUCTION_GUIDE.md)** - Complete guide for reproducing paper results
- `STRUCTURE.md` - Code organization and file inventory
- `STATUS.md` - Pipeline status and debugging notes
- `DATA_DEPENDENCIES.md` - Data sources and external tools
- `REGRESSION_ANALYSIS.md` - Details of the January 2025 regression bug
- `PROBLEM_SOLVED.md` - Quick summary of bug fix
- `RECOVERY_INSTRUCTIONS.md` - Manual recovery steps
