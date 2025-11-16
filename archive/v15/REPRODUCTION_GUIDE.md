# Complete Reproduction Guide
**QFD Supernova V15 Analysis**

**Paper Results:** k_J ≈ 10.7 ± 4.6, η' ≈ -8.0 ± 1.4, ξ ≈ -7.0 ± 3.8, RMS ≈ 1.89 mag

**Status:** ✅ **Reproducible** (as of 2025-11-12)

---

## Table of Contents

1. [Quick Start (For Experienced Users)](#quick-start)
2. [Complete Setup (For New Users)](#complete-setup)
3. [Running the Pipeline](#running-the-pipeline)
4. [Expected Results](#expected-results)
5. [Troubleshooting](#troubleshooting)
6. [Hardware Requirements](#hardware-requirements)
7. [Validation](#validation)

---

## Quick Start

If you have Python 3.12+ and CUDA 13+ installed:

```bash
# 1. Clone/download the code
cd /path/to/qfd-supernova-v15/v15_clean

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify data is present
ls data/lightcurves_unified_v2_min3.csv

# 4. Run the full pipeline
./scripts/run_full_pipeline.sh

# 5. Check results
cat ../results/v15_clean/stage2_fullscale/best_fit.json
cat ../results/v15_clean/stage3_hubble/summary.json
```

**Expected runtime:** 4-6 hours (with GPU)

---

## Complete Setup

### Prerequisites

**Required:**
- Python 3.12+ (tested with 3.12.5)
- 16GB+ RAM
- 50GB+ free disk space

**Highly Recommended:**
- NVIDIA GPU with CUDA 13+ (for 50-100x speedup)
- Linux or WSL2 (tested on Ubuntu/WSL)

**Optional:**
- Git (for version control)

### Step 1: System Dependencies

**For Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install python3.12 python3.12-venv python3-pip
```

**For GPU support (NVIDIA):**
```bash
# Check if CUDA is installed
nvidia-smi

# If not, install CUDA 13+ from:
# https://developer.nvidia.com/cuda-downloads
```

### Step 2: Python Environment

We recommend using a virtual environment:

```bash
# Navigate to project directory
cd /path/to/qfd-supernova-v15/v15_clean

# Create virtual environment
python3.12 -m venv .venv

# Activate it
source .venv/bin/activate  # Linux/Mac
# OR
.venv\Scripts\activate     # Windows

# Upgrade pip
pip install --upgrade pip
```

### Step 3: Install Python Packages

```bash
# Install all dependencies
pip install -r requirements.txt

# Verify installation
python -c "import jax; print('JAX version:', jax.__version__); print('Devices:', jax.devices())"
```

**Expected output (with GPU):**
```
JAX version: 0.8.0
Devices: [cuda(id=0)]
```

**Expected output (CPU only):**
```
JAX version: 0.8.0
Devices: [CpuDevice(id=0)]
```

### Step 4: Verify Data

Check that you have the required data files:

```bash
# Lightcurves (should be ~118k rows, 5,468 SNe)
wc -l data/lightcurves_unified_v2_min3.csv
# Expected: 118218 lines

# Verify it's a symlink or real file
ls -lh data/lightcurves_unified_v2_min3.csv

# Check file format
head -5 data/lightcurves_unified_v2_min3.csv
# Should show: snid, mjd, wavelength_nm, flux_jy, flux_err_jy, z
```

If the data file is missing, see `data/README.md` for instructions.

---

## Running the Pipeline

The pipeline has 3 stages that must be run in order:

### Full Pipeline (Automated)

```bash
# Run all 3 stages automatically
./scripts/run_full_pipeline.sh
```

**Runtime:**
- Stage 1: ~1-2 hours (per-SN optimization, 4,727 SNe)
- Stage 2: ~3-4 hours (global MCMC, GPU required for reasonable speed)
- Stage 3: ~3 minutes (Hubble diagram analysis)

**Total:** ~4-6 hours

### Individual Stages (Manual)

If you prefer to run stages individually:

#### Stage 1: Per-Supernova Optimization

```bash
python stages/stage1_optimize.py \
    --lightcurves data/lightcurves_unified_v2_min3.csv \
    --out ../results/v15_clean/stage1_fullscale \
    --workers 7 \
    --quality-cut 2000
```

**Output:**
- `../results/v15_clean/stage1_fullscale/persn_best.npy` (per-SN parameters)
- `../results/v15_clean/stage1_fullscale/L_peaks.npy` (luminosity normalizations)
- ~4,727 SNe with χ² < 2000

#### Stage 2: Global MCMC

```bash
python stages/stage2_mcmc_numpyro.py \
    --stage1-results ../results/v15_clean/stage1_fullscale \
    --lightcurves data/lightcurves_unified_v2_min3.csv \
    --out ../results/v15_clean/stage2_fullscale \
    --nchains 4 \
    --nsamples 2000 \
    --nwarmup 1000 \
    --quality-cut 2000 \
    --constrain-signs informed \
    --use-ln-a-space
```

**Output:**
- `../results/v15_clean/stage2_fullscale/best_fit.json` (global parameters)
- `../results/v15_clean/stage2_fullscale/samples.json` (full posterior)
- `../results/v15_clean/stage2_fullscale/corner_plot.png` (visualization)

**Expected parameters:**
```json
{
  "k_J": 10.7 ± 4.6,
  "eta_prime": -8.0 ± 1.4,
  "xi": -7.0 ± 3.8
}
```

#### Stage 3: Hubble Diagram

```bash
python stages/stage3_hubble_optimized.py \
    --stage1-results ../results/v15_clean/stage1_fullscale \
    --stage2-results ../results/v15_clean/stage2_fullscale \
    --lightcurves data/lightcurves_unified_v2_min3.csv \
    --out ../results/v15_clean/stage3_hubble \
    --quality-cut 2000
```

**Output:**
- `../results/v15_clean/stage3_hubble/summary.json` (statistics)
- `../results/v15_clean/stage3_hubble/hubble_diagram.png` (plot)
- `../results/v15_clean/stage3_hubble/hubble_data.csv` (per-SN distance moduli)

**Expected statistics:**
```json
{
  "qfd_rms": 1.89,  // mag
  "n_sne": 4727
}
```

---

## Expected Results

### Stage 2: Global Parameters

**Target values** (from November 5, 2024 working run and papers):

| Parameter | Expected Value | Acceptable Range |
|-----------|---------------|------------------|
| k_J       | 10.7 ± 4.6    | 9.0 - 12.5       |
| eta_prime | -8.0 ± 1.4    | -9.5 to -6.5     |
| xi        | -7.0 ± 3.8    | -8.5 to -5.5     |

**Convergence diagnostics:**
- R-hat < 1.05 (ideally < 1.01)
- ESS > 1000 (ideally > 10,000)
- Zero divergences
- Acceptance probability ~0.85-0.95

### Stage 3: Hubble Diagram

**Target statistics:**

| Metric | Expected Value |
|--------|---------------|
| QFD RMS | ~1.89 mag |
| Number of SNe | ~4,727 |
| Redshift range | 0.05 - 1.5 |

**Residual properties:**
- Median ≈ 0 (after zero-point calibration)
- Approximately normal distribution
- No strong systematic trends vs. redshift

---

## Validation

### Check Your Results

After running the pipeline, validate your results:

```bash
# Run the validation script
./CHECK_TEST_STATUS.sh

# OR manually check:
cat ../results/v15_clean/stage2_fullscale/best_fit.json
cat ../results/v15_clean/stage3_hubble/summary.json

# Compare to golden reference (November 5, 2024)
cat ../results/abc_comparison_20251105_165123/A_unconstrained/best_fit.json
```

### Regression Test

We provide a regression test that checks your results match the expected values:

```bash
cd tests
pytest test_regression_nov5.py -v
```

This will:
- ✅ Check parameters are within acceptable ranges
- ✅ Verify number of SNe is correct
- ✅ Confirm uncertainties are realistic
- ✅ Validate RMS is reasonable

---

## Troubleshooting

### Issue 1: CUDA Not Found

**Symptoms:**
```
jax.errors.RuntimeError: CUDA backend failed to initialize
```

**Solutions:**
1. Check CUDA is installed: `nvidia-smi`
2. Check CUDA version: `nvcc --version` (need 13+)
3. If CPU-only, install CPU version:
   ```bash
   pip uninstall jax jaxlib
   pip install jax[cpu]
   ```

### Issue 2: Out of Memory (GPU)

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
1. Reduce batch size in Stage 2:
   ```bash
   python stages/stage2_mcmc_numpyro.py ... --batch-size 500
   ```
2. Use fewer chains:
   ```bash
   python stages/stage2_mcmc_numpyro.py ... --nchains 2
   ```
3. Reduce max SNe:
   ```bash
   python stages/stage2_mcmc_numpyro.py ... --max-sne 2000
   ```

### Issue 3: Parameters Out of Range

**Symptoms:**
- k_J < 5 or > 15
- eta_prime > -5 or < -10
- Uncertainties < 0.1 (overfitting!)

**Possible causes:**
1. **Code regression** - Check if the January "bug fix" was applied:
   ```bash
   grep "c = -jnp.array" stages/stage2_mcmc_numpyro.py
   ```
   If found, the bug is present! Run `./fix_regression.sh`

2. **Wrong constraint mode** - Should use `--constrain-signs informed`

3. **Insufficient samples** - Increase `--nsamples 2000` and `--nwarmup 1000`

4. **Bad Stage 1 results** - Check:
   ```bash
   python tools/analyze_stage1_simple.py ../results/v15_clean/stage1_fullscale
   ```

### Issue 4: Only 548 SNe Instead of 4,727

**Symptoms:**
```json
{"n_sne": 548}  // Should be 4727!
```

**Cause:** The January "bug fix" broke quality filtering

**Solution:**
```bash
# Apply the regression fix
./fix_regression.sh

# Re-run Stage 2
python stages/stage2_mcmc_numpyro.py ... (see above)
```

### Issue 5: ImportError

**Symptoms:**
```
ImportError: No module named 'v15_model'
```

**Solutions:**
1. Make sure you're in the `v15_clean` directory
2. Check Python can find the modules:
   ```python
   import sys; print(sys.path)
   ```
3. The code uses `sys.path.insert()` - this should work automatically
4. If persistent, install as package:
   ```bash
   pip install -e .
   ```

### Issue 6: Data File Not Found

**Symptoms:**
```
FileNotFoundError: data/lightcurves_unified_v2_min3.csv
```

**Solutions:**
1. Check data directory:
   ```bash
   ls -lh data/
   ```
2. If symlink is broken, recreate it:
   ```bash
   ln -s ../../data/lightcurves_unified_v2_min3.csv data/
   ```
3. See `data/README.md` for data preparation instructions

---

## Hardware Requirements

### Minimum

- **CPU:** 4+ cores
- **RAM:** 16GB
- **Disk:** 50GB free
- **GPU:** None (but very slow!)

**Runtime:** 24-48 hours (CPU only)

### Recommended

- **CPU:** 8+ cores (Intel/AMD)
- **RAM:** 32GB
- **Disk:** 100GB free (SSD recommended)
- **GPU:** NVIDIA with 8GB+ VRAM (RTX 3070, A4000, or better)
- **CUDA:** 13.0+

**Runtime:** 4-6 hours (with GPU)

### Optimal

- **CPU:** 16+ cores
- **RAM:** 64GB+
- **Disk:** 200GB+ SSD
- **GPU:** NVIDIA with 16GB+ VRAM (RTX 4090, A6000, H100)
- **CUDA:** 13.0+

**Runtime:** 2-3 hours (with high-end GPU)

---

## Files and Directories

### Input Files

- `data/lightcurves_unified_v2_min3.csv` - Lightcurve data (118k rows, 5,468 SNe)

### Code Organization

```
v15_clean/
├── core/                      # Core modules
│   ├── v15_model.py          # Physics model (QFD)
│   ├── v15_data.py           # Data loading
│   └── pipeline_io.py        # Type-safe parameter handling
├── stages/                    # Pipeline stages
│   ├── stage1_optimize.py    # Per-SN optimization
│   ├── stage2_mcmc_numpyro.py # Global MCMC
│   └── stage3_hubble_optimized.py # Hubble diagram
├── tools/                     # Analysis tools
│   ├── analyze_stage1_simple.py
│   ├── compare_abc_variants.py
│   ├── generate_corner_plot.py
│   └── make_publication_figures.py
├── tests/                     # Unit and regression tests
│   ├── test_pipeline.py
│   ├── test_regression_nov5.py
│   └── ...
├── scripts/                   # Shell scripts
│   ├── run_full_pipeline.sh
│   ├── run_stage1_fullscale.sh
│   ├── run_stage2_fullscale.sh
│   └── run_stage3.sh
├── data/                      # Input data (symlink)
├── docs/                      # Documentation (this file!)
└── requirements.txt           # Python dependencies
```

### Output Structure

```
../results/v15_clean/
├── stage1_fullscale/
│   ├── persn_best.npy        # Per-SN parameters
│   ├── L_peaks.npy           # Luminosities
│   └── summary.json          # Statistics
├── stage2_fullscale/
│   ├── best_fit.json         # ⭐ Global parameters
│   ├── samples.json          # Full posterior
│   ├── corner_plot.png       # Visualization
│   └── diagnostics.json      # Convergence checks
└── stage3_hubble/
    ├── summary.json          # ⭐ Hubble diagram statistics
    ├── hubble_data.csv       # Per-SN distance moduli
    ├── hubble_diagram.png    # Main plot
    └── residuals_analysis.png # Diagnostics
```

---

## Advanced Usage

### Custom Configuration

Create a config file `my_config.json`:

```json
{
  "stage1": {
    "quality_cut": 2000,
    "workers": 7
  },
  "stage2": {
    "nchains": 4,
    "nsamples": 2000,
    "nwarmup": 1000,
    "constrain_signs": "informed"
  },
  "stage3": {
    "quality_cut": 2000
  }
}
```

Then run:
```bash
./scripts/run_full_pipeline.sh --config my_config.json
```

### Using Different Datasets

To run on a different dataset (e.g., Pantheon+):

```bash
# Stage 1
python stages/stage1_optimize.py \
    --lightcurves /path/to/pantheon_plus.csv \
    --out ../results/pantheon_plus/stage1 \
    --workers 7

# Stage 2
python stages/stage2_mcmc_numpyro.py \
    --stage1-results ../results/pantheon_plus/stage1 \
    --lightcurves /path/to/pantheon_plus.csv \
    --out ../results/pantheon_plus/stage2 \
    --nchains 4 --nsamples 2000 --nwarmup 1000 \
    --constrain-signs informed --use-ln-a-space

# Stage 3
python stages/stage3_hubble_optimized.py \
    --stage1-results ../results/pantheon_plus/stage1 \
    --stage2-results ../results/pantheon_plus/stage2 \
    --lightcurves /path/to/pantheon_plus.csv \
    --out ../results/pantheon_plus/stage3
```

### Parallel Runs

To run multiple configurations in parallel:

```bash
# Terminal 1
python stages/stage2_mcmc_numpyro.py ... --out results/model_A

# Terminal 2
python stages/stage2_mcmc_numpyro.py ... --out results/model_B --constrain-signs off

# Terminal 3
python stages/stage2_mcmc_numpyro.py ... --out results/model_C --constrain-signs physics
```

---

## Getting Help

### Documentation

1. **This file** - Complete reproduction guide
2. `README.md` - Project overview
3. `PROBLEM_SOLVED.md` - Recent bug fix explanation
4. `REGRESSION_ANALYSIS.md` - Technical details on regression
5. `data/README.md` - Data format and provenance
6. `tests/README.md` - Testing documentation

### Common Questions

**Q: How long does it take?**
A: 4-6 hours with GPU, 24-48 hours with CPU only

**Q: Do I need a GPU?**
A: Highly recommended. CPU-only is 50-100x slower.

**Q: What if my results don't match?**
A: Run `./CHECK_TEST_STATUS.sh` and see Troubleshooting section

**Q: Can I use my own data?**
A: Yes! See "Using Different Datasets" above

**Q: What Python version do I need?**
A: Python 3.12+ (tested with 3.12.5)

**Q: Where are the paper figures?**
A: Run `python tools/make_publication_figures.py` after Stage 3

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{mcsheery2025qfd,
  title={A Physical Origin for the Supernova Progenitor Age Bias:
         A Complete Fit to the Hubble Diagram Without Cosmic Acceleration},
  author={McSheery, Tracy},
  journal={MNRAS},
  year={2025},
  note={In preparation}
}
```

---

## Version History

- **2025-11-12**: Regression fix applied, reproduction guide created
- **2024-11-05**: Working version producing paper results
- **2025-01-12**: ⚠️ Incorrect "bug fix" applied (since reverted)

---

## License

See LICENSE file for details.

---

**Last Updated:** 2025-11-12
**Maintainer:** Tracy McSheery (tracymc@phasespace.com)
**Status:** ✅ Reproducible
