# V15_CLEAN: Consolidated Codebase Structure

**Purpose**: Self-contained, clean implementation of the QFD Supernova pipeline
**Date**: 2025-11-12

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
│   └── stage3_hubble_optimized.py # Hubble diagram (WITH ZERO-POINT CALIBRATION)
├── scripts/
│   ├── run_full_pipeline.sh   # Main production pipeline
│   └── run_*.sh               # Additional test/diagnostic scripts
├── tools/
│   ├── analyze_stage1_simple.py     # Stage 1 results analyzer
│   ├── generate_corner_plot.py      # MCMC visualization
│   ├── make_publication_figures.py  # Publication plots
│   ├── compare_abc_variants.py      # A/B/C testing framework
│   ├── generate_mock_data.py        # Mock data generator
│   ├── monitor_pipeline.py          # Live pipeline monitor
│   ├── make_per_survey_report.py    # Per-survey analysis
│   └── README.md                    # Tool documentation
├── data/
│   ├── lightcurves_unified_v2_min3.csv  # 5,468 SNe (symlink)
│   └── README.md              # Data documentation
├── results/                   # Output directory (created on run)
├── tests/
│   └── test_*.py              # Unit tests
├── docs/
│   ├── STRUCTURE.md           # This file
│   ├── STATUS.md              # Current pipeline status
│   └── DATA_DEPENDENCIES.md   # Data sources
└── README.md                  # Main documentation
```

## Key Features

**Stage 2 (v36KB, 2025-11-12 01:11):**
- ✅ Sign fix applied (lines 414, 424, 439, 448)
- ✅ Informed priors mode
- ✅ ln_A-space fast model

**Stage 3 (v15KB, 2025-11-12 07:46):**
- ✅ Zero-point calibration (lines 258-275)
- ✅ Parallel processing
- ✅ Proper imports from v15_clean/core/

## Usage

All scripts should be run from the repository root:

```bash
# Stage 1
python v15_clean/stages/stage1_optimize.py ...

# Stage 2
python v15_clean/stages/stage2_mcmc_numpyro.py ...

# Stage 3
python v15_clean/stages/stage3_hubble_optimized.py ...
```

## Code Versions

**ONLY USE CODE FROM v15_clean/** - other directories (src/, 2Compare/) contain outdated versions.
