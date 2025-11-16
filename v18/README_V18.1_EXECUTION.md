# V18.1 Controlled Experiment: Execution Guide

## CRITICAL PHYSICS FIXES

This V18.1 branch contains TWO fundamental physics corrections that invalidate all previous V18 and V19 results:

### FIX #1: Enable FDR Iterative Solver
**Bug**: The iterative loop in `qfd_tau_total_jax()` was bypassed (`tau_total = tau_plasma`), meaning **eta_prime and xi were NEVER used in physics calculations**.

**Fix**: Enabled self-consistent iterative solver (20 iterations) in `v18/pipeline/core/v17_lightcurve_model.py:393`:
```python
# BEFORE (THE BUG):
tau_total = tau_plasma

# AFTER (THE FIX):
tau_total, _ = jax.lax.fori_loop(0, OPACITY_MAX_ITER, body_fn, (tau_plasma, 0))
```

### FIX #2: Remove Plasma Veil Double-Counting
**Bug**: Plasma effect applied TWICE - as redshift AND as opacity, causing ~2x error in A_plasma values.

**Fix**: Pure opacity model in `v18/pipeline/core/v17_lightcurve_model.py:535,626`:
```python
# BEFORE (THE BUG):
z_plasma = qfd_plasma_redshift_jax(t_since_explosion, wavelength_obs, A_plasma, beta)
wavelength_rest = wavelength_obs / (1.0 + z_plasma + z_bbh)

# AFTER (THE FIX):
z_plasma = 0.0  # Pure opacity model - NO redshift application
wavelength_rest = wavelength_obs / (1.0 + z_plasma + z_bbh)
```

---

## Dataset: Clean Type Ia Sample

**Purpose**: Controlled experiment with ZERO contamination from other supernova types.

**Specifications**:
- **6,895 Type Ia supernovae** (SNTYPE 0, 1, 4 only)
- **649,682 photometric measurements**
- **Redshift range**: 0.050 - 1.298
- **Quality cuts**: ≥5 observations per SN
- **Excluded types**: Core-collapse (5,23,29,32,33,39), SLSN (41,66,141), AGN/TDE (80,81,82), Other (101,122,129,139,180)

---

## Prerequisites

### 1. Python Environment
```bash
# Recommended: Python 3.9+
python --version

# Install JAX with CPU/GPU support
pip install "jax[cpu]"  # For CPU
# OR
pip install "jax[cuda12]"  # For CUDA 12.x GPU
```

### 2. Install Dependencies
```bash
cd Quantum-Field-Dynamics/v18
pip install -r requirements.txt
```

### 3. Generate Dataset
```bash
cd /home/user/Quantum-Field-Dynamics
python projects/V19/scripts/extract_full_dataset.py \
    --output v18/data/lightcurves_type_ia_clean.csv \
    --include-all-types false \
    --min-observations 5

# Expected output:
# - 6,895 SNe extracted
# - 649,682 measurements
# - File size: ~123 MB
```

**Verify dataset**:
```bash
wc -l v18/data/lightcurves_type_ia_clean.csv
# Should show: 649683 (649682 data rows + 1 header)

head -n 2 v18/data/lightcurves_type_ia_clean.csv
```

---

## Execution: 3-Stage Pipeline

### Stage 1: Per-SN Optimization (~2 hours on CPU, ~30 min on GPU)

**What it does**: Optimizes 8 QFD parameters per supernova using Student-t likelihood (nu=5.0).

**Parameters optimized**:
- `delta_M`: Magnitude offset (QFD correction to absolute magnitude)
- `t0`: Explosion time
- `x0`, `x1`, `c`: SALT3 lightcurve shape parameters
- `A_plasma`, `beta`: Plasma veil opacity parameters (NEW PHYSICS)
- `A_lens`: BBH gravitational lensing amplitude

**Run Stage 1**:
```bash
cd /home/user/Quantum-Field-Dynamics/v18
python run_stage1.py \
    --lightcurves data/lightcurves_type_ia_clean.csv \
    --output results/stage1_optimized_params.csv \
    --num-workers 4

# For parallel processing (recommended):
python run_stage1.py \
    --lightcurves data/lightcurves_type_ia_clean.csv \
    --output results/stage1_optimized_params.csv \
    --num-workers $(nproc)  # Use all CPU cores
```

**Expected outputs**:
- `v18/results/stage1_optimized_params.csv`: Per-SN best-fit parameters (6,895 rows)
- `v18/results/stage1_residuals.csv`: Residuals for each observation
- Progress updates every 100 SNe

**Timing**:
- ~1-2 seconds per SN on modern CPU
- Total: ~2-3 hours on 8-core CPU
- Total: ~30-45 minutes on GPU (RTX 3090 or better)

**Verification**:
```bash
wc -l v18/results/stage1_optimized_params.csv
# Should show: 6896 (6895 SNe + 1 header)

# Check that optimization succeeded
head -n 5 v18/results/stage1_optimized_params.csv
```

---

### Stage 2: MCMC Global Parameter Inference (~30 minutes)

**What it does**: Bayesian inference of 4 global QFD parameters using emcee MCMC sampler.

**Global parameters**:
- `eta_prime`: FDR flux-dependent opacity strength (NOW ACTIVE - was inactive due to bug!)
- `xi`: FDR wavelength scaling exponent (NOW ACTIVE - was inactive due to bug!)
- `H0`: Hubble constant (km/s/Mpc)
- `Omega_m`: Matter density parameter

**Run Stage 2**:
```bash
cd /home/user/Quantum-Field-Dynamics/v18
python pipeline/stages/stage2_mcmc_global.py \
    --stage1-results results/stage1_optimized_params.csv \
    --lightcurves data/lightcurves_type_ia_clean.csv \
    --output results/stage2_mcmc_chains.h5 \
    --nwalkers 32 \
    --nsteps 5000 \
    --burn-in 1000
```

**Expected outputs**:
- `v18/results/stage2_mcmc_chains.h5`: MCMC chains (HDF5 format)
- `v18/results/stage2_corner_plot.png`: Corner plot showing parameter correlations
- `v18/results/stage2_summary.txt`: Parameter posteriors (mean ± std)

**Timing**:
- ~30 minutes on modern CPU (32 walkers, 5000 steps)
- ~10 minutes on GPU

**CRITICAL INTERPRETATION**:
This will be the **first-ever scientifically valid measurement** of eta_prime and xi in QFD theory. Previous values were meaningless artifacts because FDR was disabled!

**Verification**:
```bash
# Check that MCMC completed
python -c "import h5py; f=h5py.File('v18/results/stage2_mcmc_chains.h5', 'r'); print('Chains shape:', f['chain'].shape); f.close()"
# Should show: Chains shape: (32, 5000, 4)

# View summary
cat v18/results/stage2_summary.txt
```

---

### Stage 3: Hubble Diagram Generation (~5 minutes)

**What it does**: Generates distance modulus vs. redshift Hubble diagram comparing QFD to ΛCDM.

**Run Stage 3**:
```bash
cd /home/user/Quantum-Field-Dynamics/v18
python pipeline/stages/stage3_hubble_diagram.py \
    --stage1-results results/stage1_optimized_params.csv \
    --stage2-results results/stage2_mcmc_chains.h5 \
    --lightcurves data/lightcurves_type_ia_clean.csv \
    --output results/stage3_hubble_diagram.png
```

**Expected outputs**:
- `v18/results/stage3_hubble_diagram.png`: Hubble diagram plot
- `v18/results/stage3_residuals_qfd.png`: QFD residuals vs. redshift
- `v18/results/stage3_residuals_lcdm.png`: ΛCDM residuals vs. redshift
- `v18/results/stage3_statistics.txt`: Chi-squared comparison

**Timing**: ~5 minutes

**Interpretation**:
- **If QFD residuals are smaller**: QFD provides better fit than ΛCDM
- **If eta_prime ≠ 0 at >3σ**: Strong evidence for FDR effect
- **Compare A_plasma distributions**: Plasma veil now correctly modeled (expect factor ~2 change from V18 buggy version)

**Verification**:
```bash
ls -lh v18/results/stage3_*.png
cat v18/results/stage3_statistics.txt
```

---

## Quick Start (Full Pipeline)

```bash
# 1. Setup
cd /home/user/Quantum-Field-Dynamics
pip install -r v18/requirements.txt

# 2. Generate dataset (if not exists)
python projects/V19/scripts/extract_full_dataset.py \
    --output v18/data/lightcurves_type_ia_clean.csv \
    --include-all-types false \
    --min-observations 5

# 3. Run pipeline
cd v18
python run_stage1.py --lightcurves data/lightcurves_type_ia_clean.csv --output results/stage1_optimized_params.csv --num-workers $(nproc)
python pipeline/stages/stage2_mcmc_global.py --stage1-results results/stage1_optimized_params.csv --lightcurves data/lightcurves_type_ia_clean.csv --output results/stage2_mcmc_chains.h5
python pipeline/stages/stage3_hubble_diagram.py --stage1-results results/stage1_optimized_params.csv --stage2-results results/stage2_mcmc_chains.h5 --lightcurves data/lightcurves_type_ia_clean.csv --output results/stage3_hubble_diagram.png

# 4. Review results
cat results/stage2_summary.txt
cat results/stage3_statistics.txt
```

**Total time**: ~3 hours on 8-core CPU, ~1 hour on GPU.

---

## Expected Scientific Outcomes

### 1. First Valid QFD Baseline
- **eta_prime posterior**: First measurement with FDR actually enabled
- **xi posterior**: First measurement with FDR actually enabled
- **H0 and Omega_m**: May differ significantly from previous buggy results

### 2. Corrected Plasma Veil Parameters
- **A_plasma values**: Expect ~2x change due to removing double-counting
- **beta values**: Should be more physically consistent

### 3. QFD vs. ΛCDM Comparison
- **Chi-squared ratio**: Meaningful comparison now possible
- **Residual patterns**: Look for systematic trends vs. redshift
- **Physical interpretation**: Can now make defensible claims about QFD viability

### 4. Next Steps After V18.1
- **If results are promising**: Merge fixes into V19 multi-type pipeline
- **If results are problematic**: Debug physics model further
- **Scientific publication**: First valid QFD supernova cosmology paper

---

## Troubleshooting

### Issue: "FileNotFoundError: lightcurves_type_ia_clean.csv"
**Solution**: Regenerate dataset using extraction script (see "Generate Dataset" above)

### Issue: "JAX out of memory"
**Solution**: Reduce batch size in Stage 1 optimization or use CPU mode

### Issue: "MCMC chains not converging"
**Solution**: Increase burn-in period (--burn-in 2000) or adjust priors in stage2_mcmc_global.py

### Issue: Stage 1 very slow
**Solution**: Use `--num-workers $(nproc)` to parallelize across all CPU cores, or use GPU acceleration

---

## File Structure

```
v18/
├── README_V18.1_EXECUTION.md        # This file
├── run_stage1.py                     # Standalone Stage 1 runner (NEW)
├── data/
│   ├── lightcurves_type_ia_clean.csv  # 6,895 Type Ia SNe (generated by user)
│   └── README.md                      # Dataset documentation
├── pipeline/
│   ├── core/
│   │   └── v17_lightcurve_model.py    # CRITICAL PHYSICS FIXES APPLIED
│   └── stages/
│       ├── stage1_optimize_v17.py     # Per-SN optimization module
│       ├── stage2_mcmc_global.py      # MCMC global parameters
│       └── stage3_hubble_diagram.py   # Hubble diagram generation
├── results/                           # Output directory (created during execution)
│   ├── stage1_optimized_params.csv
│   ├── stage2_mcmc_chains.h5
│   ├── stage2_summary.txt
│   ├── stage3_hubble_diagram.png
│   └── stage3_statistics.txt
└── requirements.txt                   # Python dependencies
```

---

## Contact

For questions about this controlled experiment, contact the QFD research team.

**Version**: V18.1 (Physics Bugfix Release)
**Date**: 2025-11-16
**Commit**: Branch `claude/v18.1-fdr-and-plasma-fix-01WB2FNUfRb9JfK78SZ2Fen6`
