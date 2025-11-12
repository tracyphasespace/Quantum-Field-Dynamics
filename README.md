# V15_CLEAN: QFD Supernova Pipeline

**Self-contained, consolidated implementation of the QFD supernova analysis pipeline**

## Quick Start

### Self-Contained Mode (Recommended)
```bash
# Everything runs from v15_clean directory
cd /home/tracy/development/SupernovaSrc/qfd-supernova-v15/v15_clean

# Run full pipeline (Stage 2 → Stage 3)
./scripts/run_full_pipeline.sh
```

### Legacy Mode (From repository root)
```bash
cd /home/tracy/development/SupernovaSrc/qfd-supernova-v15
./v15_clean/scripts/run_full_pipeline.sh
```

## Directory Structure

```
v15_clean/
├── core/
│   ├── v15_model.py           # QFD physics model
│   ├── v15_data.py            # Lightcurve data loading
│   └── pipeline_io.py         # I/O utilities
├── stages/
│   ├── stage1_optimize.py     # Per-SN parameter fits
│   ├── stage2_mcmc_numpyro.py # Global MCMC (WITH SIGN FIX)
│   └── stage3_hubble_optimized.py # Hubble diagram (WITH ZERO-POINT CAL)
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

**Output**: Results in `../results/v15_clean/`

## Usage

### Run Full Pipeline
```bash
./v15_clean/scripts/run_full_pipeline.sh
```

### Run Individual Stages
```bash
# Stage 1 (if needed - usually pre-computed)
python v15_clean/stages/stage1_optimize.py \
    --lightcurves data/lightcurves_unified_v2_min3.csv \
    --out results/v15_clean/stage1_fullscale

# Stage 2
python v15_clean/stages/stage2_mcmc_numpyro.py \
    --stage1-results results/v15_clean/stage1_fullscale \
    --out results/v15_clean/stage2_production \
    --lightcurves data/lightcurves_unified_v2_min3.csv \
    --use-ln-a-space \
    --constrain-signs informed

# Stage 3  
python v15_clean/stages/stage3_hubble_optimized.py \
    --stage1-results results/v15_clean/stage1_fullscale \
    --stage2-results results/v15_clean/stage2_production \
    --out results/v15_clean/stage3_hubble
```

## Documentation

- `STRUCTURE.md` - Code organization and file inventory
- `STATUS.md` - Current pipeline status and debugging notes
- `DATA_DEPENDENCIES.md` - Data sources and external tools

## Code Versions

**IMPORTANT**: Only use code from `v15_clean/`. Other directories (`src/`, `2Compare/`) contain outdated versions.

## Current Status

**Code**: Consolidated and ready ✓
**Data**: Provided dataset ready ✓
**Pipeline**: Sign fix and zero-point calibration applied ✓

**Open Issue**: Production run with 4,727 SNe hits MCMC convergence issues (574 divergences, k_J stuck at lower bound). See `STATUS.md` for details and diagnostic plans.

## Next Steps

See `STATUS.md` for:
- Diagnostic run with unconstrained priors
- Investigation of Stage 1 ln_A systematic bias
- Analysis tools to add to `tools/`
