# Reproducibility Guide: V15 QFD Pipeline

Complete instructions to reproduce all results from the V15 publication.

---

## Environment Setup

### Required Software

- **Python:** 3.9 or later
- **CUDA:** 11.0+ (for GPU acceleration)
- **Git:** For repository management

### Installation

```bash
# Clone the repository
git clone https://github.com/tracyphasespace/Quantum-Field-Dynamics.git
cd Quantum-Field-Dynamics/projects/astrophysics/qfd-supernova-v15

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import jax; import numpyro; print('JAX version:', jax.__version__); print('NumPyro version:', numpyro.__version__)"
```

### Environment Variables (Optional)

```bash
# For GPU acceleration
export JAX_PLATFORM_NAME=gpu  # or 'cpu' for CPU-only

# For debugging
export JAX_TRACEBACK_FILTERING=off
export JAX_DEBUG_NANS=True
```

---

## Data Preparation

### Input Data Requirements

The pipeline expects lightcurve data in CSV format with columns:
- `snid`: Supernova identifier
- `survey`: Survey name (e.g., "Pantheon+", "HST")
- `band`: Photometric band (e.g., "F160W", "r")
- `mjd`: Modified Julian Date (observer frame)
- `flux_jy`: Flux in Janskys
- `flux_err_jy`: Flux uncertainty in Janskys
- `z`: Redshift

### Example Data Location

```bash
# Production data (if available)
DATA_DIR=data/lightcurves_unified_v2_min3.csv

# Test/smoke data (subset for quick validation)
TEST_DATA=data/lightcurves_test_subset.csv
```

---

## Pipeline Execution

### Stage 1: Per-SN Parameter Optimization

```bash
# Full production run
python src/stage1_optimize.py \
    --lightcurves $DATA_DIR \
    --out results/stage1_production \
    --batch-size auto \
    --optimizer lbfgs \
    --max-iter 1000 \
    --conv-tol 1e-6

# Smoke test (quick validation, ~5 minutes)
python src/stage1_optimize.py \
    --lightcurves $TEST_DATA \
    --out results/stage1_smoke \
    --n-sne 50 \
    --batch-size 10 \
    --max-iter 100
```

**Expected output:**
- `results/stage1_*/stage1_results.csv` - Per-SN parameters
- `results/stage1_*/summary.json` - Global statistics
- `results/stage1_*/convergence.log` - Optimization logs

**Quality checks:**
```bash
# Analyze Stage 1 results
python src/analyze_stage1_results.py \
    --results results/stage1_production/stage1_results.csv \
    --plot-diagnostics

# Expected: convergence rate > 95%, chi2/obs ~ 1-2
```

### Stage 2: Global Parameter Inference (NumPyro MCMC)

```bash
# Production run (full chains, ~2-6 hours depending on hardware)
python src/stage2_mcmc_numpyro.py \
    --stage1-results results/stage1_production/stage1_results.csv \
    --out results/stage2_production \
    --nchains 4 \
    --nsamples 2000 \
    --nwarmup 1000 \
    --target-accept 0.8

# Smoke test (~5-10 minutes)
bash scripts/run_stage2_numpyro_production.sh --smoke
# Or directly:
python src/stage2_mcmc_numpyro.py \
    --stage1-results results/stage1_smoke/stage1_results.csv \
    --out results/stage2_smoke \
    --nchains 2 \
    --nsamples 100 \
    --nwarmup 50
```

**Expected output:**
- `results/stage2_*/samples.json` - Posterior samples
- `results/stage2_*/summary.csv` - Posterior statistics (mean, std, R̂, ESS)
- `results/stage2_*/convergence_diagnostics.png` - Trace plots

**Quality checks:**
```bash
# Check convergence
python -c "
import pandas as pd
df = pd.read_csv('results/stage2_production/summary.csv')
print('R-hat values:', df['rhat'].values)
print('ESS bulk:', df['ess_bulk'].values)
assert (df['rhat'] < 1.01).all(), 'R-hat > 1.01!'
assert (df['ess_bulk'] > 400).all(), 'ESS < 400!'
print('✓ Convergence checks passed')
"
```

### Stage 3: Residual Analysis

```bash
# Production analysis
python src/stage3_hubble_optimized.py \
    --stage1-results results/stage1_production/stage1_results.csv \
    --stage2-samples results/stage2_production/samples.json \
    --out results/stage3_production \
    --plot-diagnostics

# Smoke test
python src/stage3_hubble_optimized.py \
    --stage1-results results/stage1_smoke/stage1_results.csv \
    --stage2-samples results/stage2_smoke/samples.json \
    --out results/stage3_smoke
```

**Expected output:**
- `results/stage3_*/stage3_results.csv` - Full residual table
- `results/stage3_*/summary.json` - Global RMS, trends
- `results/stage3_*/hubble_diagram.png` - Visualization
- `results/stage3_*/residual_histogram.png` - Residual distribution

**Guard verification:**
```bash
# Should see no RuntimeErrors about zero variance
grep "WIRING BUG" results/stage3_production/*.log
# (Should return no matches)
```

---

## Validation & Testing

### Unit Tests

```bash
# Run all unit tests
pytest tests/ -v

# Run specific test suites
pytest tests/test_stage3_identity.py -v
pytest tests/test_alpha_pred_properties.py -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

**Expected:** All tests pass (19/19 unit + property tests)

### Validation Scripts

```bash
# Run comprehensive alpha-space validation
python test_alpha_space_validation.py

# Expected output:
# ✓ TEST 1: alpha_pred() function works correctly
# ✓ TEST 2: Alpha-space likelihood works correctly
# ✓ TEST 3: alpha_pred is independent of alpha_obs
# ✓ TEST 4: Wiring bug detection works
# ✓ TEST 5: Stage 3 guard works correctly
# 🎉 ALL VALIDATION TESTS PASSED!

# Generate validation plots
python visualize_validation.py
# Output: validation_plots/*.png
```

---

## Report Generation

### Per-Survey Diagnostics

```bash
# Generate comprehensive per-survey reports
python scripts/make_per_survey_report.py \
    results/stage3_production/stage3_results.csv \
    --out-dir results/v15_production/reports

# Expected output files:
#   - summary_overall.csv
#   - summary_by_survey_alpha.csv
#   - summary_by_survey_band_alpha.csv
#   - summary_by_survey_lcdm.csv (if ΛCDM comparison available)
#   - zbin_alpha_by_survey.csv
```

### Publication Figures

```bash
# Generate publication-quality figures
python scripts/make_publication_figures.py \
    --stage3-csv results/stage3_production/stage3_results.csv \
    --report-dir results/v15_production/reports \
    --out-dir results/v15_production/figures

# For hold-out validation (Fig 8):
python scripts/make_publication_figures.py \
    --stage3-csv results/stage3_production/stage3_results.csv \
    --report-dir results/v15_production/reports \
    --train-csv results/v15_production/reports/train_summary.csv \
    --test-csv results/v15_production/reports/test_summary.csv \
    --out-dir results/v15_production/figures

# Expected figures:
#   - fig4_hubble_diagram.png
#   - fig6_per_survey_residuals.png
#   - fig8_holdout_performance.png (if train/test provided)
```

---

## Full Pipeline (End-to-End)

```bash
# Complete pipeline from data to publication figures
bash scripts/run_full_pipeline.sh \
    --data $DATA_DIR \
    --out results/v15_production \
    --nchains 4 \
    --nsamples 2000

# This script runs:
# 1. Stage 1 optimization
# 2. Stage 2 MCMC
# 3. Stage 3 analysis
# 4. Per-survey reports
# 5. Publication figures
# 6. Validation checks

# Estimated runtime:
#   - Stage 1: ~1-4 hours (depends on N_SNe)
#   - Stage 2: ~2-6 hours (depends on chains/samples)
#   - Stage 3: ~10-30 minutes
#   - Reports: ~1-5 minutes
#   Total: ~4-11 hours on GPU

# Progress monitoring
tail -f results/v15_production/pipeline.log
```

---

## Quick Smoke Test (5-10 minutes)

For rapid validation without full dataset:

```bash
# Run smoke test of entire pipeline
bash scripts/run_full_pipeline.sh \
    --data data/lightcurves_test_subset.csv \
    --out results/smoke_test \
    --smoke-test

# Or manually:
# 1. Stage 1 smoke (50 SNe, 100 iter): ~2 min
python src/stage1_optimize.py --lightcurves $TEST_DATA --out results/smoke/s1 --n-sne 50 --max-iter 100

# 2. Stage 2 smoke (2 chains, 100 samples): ~3 min
python src/stage2_mcmc_numpyro.py --stage1-results results/smoke/s1/stage1_results.csv --out results/smoke/s2 --nchains 2 --nsamples 100 --nwarmup 50

# 3. Stage 3 smoke: ~1 min
python src/stage3_hubble_optimized.py --stage1-results results/smoke/s1/stage1_results.csv --stage2-samples results/smoke/s2/samples.json --out results/smoke/s3

# 4. Validation: <1 min
python test_alpha_space_validation.py
pytest tests/test_stage3_identity.py tests/test_alpha_pred_properties.py -v
```

---

## Troubleshooting

### Common Issues

**Issue:** `ModuleNotFoundError: No module named 'numpyro'`
```bash
# Solution: Install dependencies
pip install -r requirements.txt
```

**Issue:** JAX not using GPU
```bash
# Check GPU availability
python -c "import jax; print(jax.devices())"

# Should show: [CudaDevice(id=0), ...]
# If CPU only, check CUDA installation and JAX version
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

**Issue:** Stage 2 MCMC divergences
```bash
# Solution: Increase target acceptance rate
python src/stage2_mcmc_numpyro.py ... --target-accept 0.9

# Or increase warmup samples
python src/stage2_mcmc_numpyro.py ... --nwarmup 2000
```

**Issue:** RuntimeError: "WIRING BUG: alpha_pred ≈ alpha_obs"
```bash
# This is a guard that caught a bug! Check:
# 1. Is alpha_pred accidentally using alpha_obs as input?
# 2. Are Stage 1 and Stage 2 using consistent data?
# 3. Run validation test to isolate issue:
python test_alpha_space_validation.py
```

**Issue:** Tests fail with dtype mismatches
```bash
# Solution: Tests expect float32 (JAX default)
# Check JAX config:
python -c "import jax; print(jax.config.read('jax_enable_x64'))"
# Should be False (default); if True, set to False:
export JAX_ENABLE_X64=0
```

---

## Computational Resources

### Recommended Specifications

**Minimum (smoke tests):**
- CPU: 4+ cores
- RAM: 8 GB
- GPU: Not required (CPU mode works)
- Storage: 5 GB

**Recommended (production):**
- CPU: 16+ cores
- RAM: 32 GB
- GPU: NVIDIA GPU with 8+ GB VRAM (e.g., RTX 3070, A100)
- Storage: 50 GB

### Performance Benchmarks

On NVIDIA RTX 3090 (24 GB VRAM):

| Stage | N_SNe | Time | Memory |
|-------|-------|------|--------|
| Stage 1 | 1000 | ~45 min | 8 GB |
| Stage 2 | 1000 | ~2.5 hr | 12 GB |
| Stage 3 | 1000 | ~15 min | 4 GB |

On CPU only (16-core Xeon):

| Stage | N_SNe | Time | Memory |
|-------|-------|------|--------|
| Stage 1 | 1000 | ~3 hr | 6 GB |
| Stage 2 | 1000 | ~12 hr | 10 GB |
| Stage 3 | 1000 | ~20 min | 4 GB |

---

## Version Information

**Commit Hash:** `[to be filled at publication]`

**Software Versions (tested):**
- Python: 3.11.5
- JAX: 0.4.20
- NumPyro: 0.13.2
- NumPy: 1.24.3
- Pandas: 2.0.3
- Matplotlib: 3.7.2
- SciPy: 1.11.1

**Reproducibility Date:** 2025-11-05

---

## Citation

If you use this pipeline in your research, please cite:

```
@article{v15qfd2025,
  title={A Batched QFD Supernova Pipeline (V15): $\alpha$-space Cosmology Without $\Lambda$CDM Priors},
  author={McSheery, Tracy and collaborators},
  journal={[Journal]},
  year={2025},
  doi={[DOI]}
}
```

---

## Contact

For questions or issues:
- GitHub Issues: https://github.com/tracyphasespace/Quantum-Field-Dynamics/issues
- Email: [contact email]

---

## Appendix: File Manifest

Complete list of files used in pipeline:

### Source Code
```
src/
├── stage1_optimize.py          # Stage 1 optimization
├── stage2_mcmc_numpyro.py      # Stage 2 MCMC
├── stage3_hubble_optimized.py  # Stage 3 analysis
├── v15_model.py                # QFD physics model
├── v15_data.py                 # Data loading
├── v15_config.py               # Configuration
├── v15_sampler.py              # MCMC utilities
├── v15_metrics.py              # Metrics computation
└── analyze_stage1_results.py   # Stage 1 diagnostics
```

### Scripts
```
scripts/
├── make_per_survey_report.py        # Per-survey diagnostics
├── make_publication_figures.py      # Publication figures
├── run_full_pipeline.sh             # End-to-end automation
├── run_stage2_numpyro_production.sh # Stage 2 wrapper
└── check_pipeline_status.sh         # Status monitoring
```

### Tests
```
tests/
├── test_stage3_identity.py          # Core identity tests
└── test_alpha_pred_properties.py    # Property tests

# Validation scripts (root level)
├── test_alpha_space_validation.py   # Comprehensive validation
└── visualize_validation.py          # Validation plots
```

### Documentation
```
docs/
├── REPRODUCIBILITY.md               # This file
├── PUBLICATION_TEMPLATE.md          # Publication scaffold
├── HOTFIX_VALIDATION.md             # Hotfix validation report
├── BUG_ANALYSIS.md                  # Bug analysis
└── README.md                        # Project overview
```

---

**Last Updated:** 2025-11-05
**Status:** Production-ready
