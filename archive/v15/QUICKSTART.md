# V15_CLEAN: Quick Start Guide

**Your clean, self-contained debugging environment** ✓

## Directory Overview

```
v15_clean/                           ← Work from here!
├── core/          3 files           ← Physics model, data I/O
├── stages/        3 files           ← Pipeline stages (with all bug fixes)
├── scripts/       9 files           ← Run scripts
├── tools/         7 files + README  ← Analysis utilities
├── data/          1 symlink + README ← 5,468 SNe lightcurves
├── results/       (empty)           ← Output created here
├── tests/         4 files           ← Unit tests
└── docs/          3 files + READMEs ← Documentation
```

## Usage: Work Entirely from v15_clean

### Option 1: Self-Contained Mode (Recommended)

```bash
cd /home/tracy/development/SupernovaSrc/qfd-supernova-v15/v15_clean

# Run full pipeline
./scripts/run_full_pipeline.sh

# Run individual stages
python stages/stage1_optimize.py --lightcurves data/lightcurves_unified_v2_min3.csv --out results/stage1

python stages/stage2_mcmc_numpyro.py \
    --stage1-results results/stage1 \
    --lightcurves data/lightcurves_unified_v2_min3.csv \
    --out results/stage2 \
    --use-ln-a-space \
    --constrain-signs informed

python stages/stage3_hubble_optimized.py \
    --stage1-results results/stage1 \
    --stage2-results results/stage2 \
    --lightcurves data/lightcurves_unified_v2_min3.csv \
    --out results/stage3

# Analyze results
python tools/analyze_stage1_simple.py results/stage1
python tools/generate_corner_plot.py --samples results/stage2/samples.json --out corner.png
```

### Option 2: From Repository Root (Legacy)

```bash
cd /home/tracy/development/SupernovaSrc/qfd-supernova-v15

# Still works, but uses absolute paths
./v15_clean/scripts/run_full_pipeline.sh
```

## Key Files

### Pipeline Stages (ALL WITH BUG FIXES)
- `stages/stage1_optimize.py` - Per-SN fits (L-BFGS-B)
- `stages/stage2_mcmc_numpyro.py` - **WITH SIGN FIX** (Nov 12 01:11)
- `stages/stage3_hubble_optimized.py` - **WITH ZERO-POINT CALIBRATION** (Nov 12 07:46)

### Core Modules
- `core/v15_model.py` - QFD physics model
- `core/v15_data.py` - Lightcurve loading
- `core/pipeline_io.py` - I/O utilities

### Analysis Tools
- `tools/analyze_stage1_simple.py` - Stage 1 diagnostics
- `tools/generate_corner_plot.py` - MCMC visualization
- `tools/make_publication_figures.py` - Publication plots
- See `tools/README.md` for full list

## Data

**Location**: `data/lightcurves_unified_v2_min3.csv` (symlink to `../../data/`)
- 5,468 Type Ia SNe
- Pantheon+ + DES-SN5YR combined
- 12 MB file (accessed via symlink, no duplication)

See `data/README.md` for format details.

## Results

All pipeline output goes to `results/` subdirectory:

```
results/
├── stage1/           # Per-SN parameter fits
├── stage2/           # Global MCMC samples
└── stage3/           # Hubble diagram analysis
```

## Clean Debugging Workflow

### 1. Kill Old Background Processes
```bash
pkill -f "python.*stage"  # Kill any old pipeline runs
```

### 2. Start Fresh from v15_clean
```bash
cd v15_clean
./scripts/run_full_pipeline.sh
```

### 3. Monitor Progress
```bash
# Watch logs
tail -f logs/stage2.log
tail -f logs/stage3.log

# Check results
ls -lh results/stage2/
cat results/stage2/best_fit.json
```

## What NOT to Use

**Old directories with code sprawl** (DO NOT USE):
- `../src/` - Outdated Stage 2/3 (Nov 11, no sign fix)
- `../2Compare/` - Ancient versions (Nov 10)
- `../scripts/` - Scattered utilities

**ONLY USE: v15_clean/** ✓

## Bug Fixes Included

### ✅ Stage 2 Sign Fix (Nov 12 01:11)
Lines 414, 424, 439, 448:
```python
c = -jnp.array([k_J, eta_prime, xi]) * scales  # CRITICAL: negative sign!
```

### ✅ Stage 3 Zero-Point Calibration (Nov 12 07:46)
Lines 258-275:
```python
offset = np.mean(mu_obs_raw - mu_qfd_arr)
mu_obs_arr = mu_obs_raw - offset
```

## Testing

```bash
cd v15_clean

# Run unit tests
python -m pytest tests/

# Quick pipeline test (50 SNe)
python stages/stage2_mcmc_numpyro.py \
    --stage1-results ../results/v15_clean/stage1_fullscale \
    --lightcurves data/lightcurves_unified_v2_min3.csv \
    --out results/test_50sne \
    --max-sne 50 \
    --use-ln-a-space \
    --constrain-signs informed
```

## Troubleshooting

### "No such file or directory"
- Make sure you're in the `v15_clean/` directory
- Check that symlink exists: `ls -lh data/`

### "Import error"
- Run from v15_clean directory
- Python will find modules via relative imports

### "Results contamination"
- Use fresh result directories
- Avoid reusing old results from `../results/v15_clean/`

## Documentation

- `README.md` - Main documentation
- `STATUS.md` - Current pipeline status
- `STRUCTURE.md` - Code organization
- `DATA_DEPENDENCIES.md` - Data sources
- `tools/README.md` - Tool documentation
- `data/README.md` - Data format

## Summary

**v15_clean is your clean, self-contained debugging environment.**

Everything you need is here:
- ✓ Code with all bug fixes
- ✓ Data (via symlink)
- ✓ Tools for analysis
- ✓ Documentation
- ✓ No code sprawl

**Just work from this directory and you're good to go!**
