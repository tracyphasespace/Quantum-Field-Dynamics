# V15_CLEAN: Analysis and Utility Tools

Collection of analysis scripts, visualization tools, and utilities for the QFD supernova pipeline.

## Usage

All tools should be run from the repository root:

```bash
# From repository root
cd /home/tracy/development/SupernovaSrc/qfd-supernova-v15

# Run tool
python v15_clean/tools/<tool_name>.py [args]
```

## Available Tools

### Analysis Tools

**`analyze_stage1_simple.py`** - Stage 1 results analyzer
```bash
python v15_clean/tools/analyze_stage1_simple.py results/v15_clean/stage1_fullscale
```
- Shows SN counts, status distribution, chi2 statistics
- Identifies convergence issues
- No external dependencies

**`monitor_pipeline.py`** - Live pipeline monitor
```bash
python v15_clean/tools/monitor_pipeline.py results/v15_clean
```
- Watches for new results and updates
- Displays progress in real-time
- Useful for long-running MCMC

### Visualization Tools

**`generate_corner_plot.py`** - MCMC corner plots
```bash
python v15_clean/tools/generate_corner_plot.py \
    --samples results/v15_clean/stage2_production/samples.json \
    --out results/v15_clean/stage2_production/corner.png
```
- Creates corner plots for Stage 2 MCMC samples
- Shows parameter correlations and distributions

**`make_publication_figures.py`** - Publication-quality figures
```bash
python v15_clean/tools/make_publication_figures.py \
    --stage3-data results/v15_clean/stage3_hubble/hubble_data.csv \
    --out figures/
```
- Generates Hubble diagram (Fig 4)
- Per-survey residual plots (Fig 6)
- Publication-ready style

### Testing and Comparison

**`compare_abc_variants.py`** - A/B/C test Stage 2 variants
```bash
python v15_clean/tools/compare_abc_variants.py \
    --stage1-results results/v15_clean/stage1_fullscale \
    --lightcurves data/lightcurves_unified_v2_min3.csv \
    --nsamples 1000 \
    --nwarmup 500
```
- Compares different MCMC constraint modes
- Tests: unconstrained, sign-constrained, informed priors
- Generates comparison report

**`generate_mock_data.py`** - Create synthetic test data
```bash
python v15_clean/tools/generate_mock_data.py
```
- Generates mock Stage 3 results
- Useful for testing visualization scripts
- Creates data with known parameters

### Survey-Specific Tools

**`make_per_survey_report.py`** - Per-survey analysis
```bash
python v15_clean/tools/make_per_survey_report.py \
    --stage3-data results/v15_clean/stage3_hubble/hubble_data.csv \
    --out reports/survey_breakdown.pdf
```
- Breaks down residuals by survey (Pantheon+, DES, SDSS, etc.)
- Identifies systematic biases
- Useful for data quality checks

## Tool Categories

### Standalone (No imports from v15_clean)
- `analyze_stage1_simple.py`
- `generate_corner_plot.py`
- `generate_mock_data.py`
- `monitor_pipeline.py`
- `make_publication_figures.py`
- `make_per_survey_report.py`

### Integrated (Uses v15_clean modules)
- `compare_abc_variants.py` (runs v15_clean/stages/stage2_mcmc_numpyro.py)

## Adding New Tools

When adding new analysis tools:

1. Place in `v15_clean/tools/`
2. If importing v15_clean modules, add proper sys.path setup:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))
from v15_model import ln_A_pred
```
3. Document in this README
4. Make executable: `chmod +x v15_clean/tools/your_tool.py`

## Common Workflows

### After Stage 1
```bash
# Check Stage 1 quality
python v15_clean/tools/analyze_stage1_simple.py results/v15_clean/stage1_fullscale
```

### After Stage 2
```bash
# Visualize MCMC samples
python v15_clean/tools/generate_corner_plot.py \
    --samples results/v15_clean/stage2_production/samples.json \
    --out corner_plot.png
```

### After Stage 3
```bash
# Generate publication figures
python v15_clean/tools/make_publication_figures.py \
    --stage3-data results/v15_clean/stage3_hubble/hubble_data.csv \
    --out figures/

# Per-survey breakdown
python v15_clean/tools/make_per_survey_report.py \
    --stage3-data results/v15_clean/stage3_hubble/hubble_data.csv \
    --out survey_report.pdf
```

## Dependencies

Most tools only require standard scientific Python:
- numpy
- pandas
- matplotlib
- scipy (for some analysis tools)

Additional dependencies:
- `corner` package (for corner plots)
- `seaborn` (for publication figures)

Install missing packages:
```bash
pip install corner seaborn
```
