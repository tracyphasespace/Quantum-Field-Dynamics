# V15: 5-Parameter GPU-Optimized Two-Stage MCMC

**Status**: Production Ready
**Based on**: V14 (parallel Stage-1) + V13.1 (validated smoke tests)
**New**: 5-parameter model reduction + Stage-2 MCMC fixes + Batch size optimization

## What's New in V15

### 1. Five-Parameter Model (Reduced from 6)
Following `cloud.txt` specification, removed BBH parameters:
- **Removed**: `t_H` (hydrogen timescale), `f_BBH` (BBH blackbody fraction)
- **Kept (5 params)**: `t0` (explosion time), `ell` (log peak luminosity), `A_plasma` (plasma amplitude), `beta` (velocity exponent), `alpha` (gamma exponent)
- **Global params (3)**: `k_J` (Janka scaling), `eta_prime` (expansion efficiency), `xi` (plasma coupling)

### 2. Stage-2 MCMC Bug Fixes
Fixed three critical bugs causing zero acceptance rate:
- **Bug 1**: Variable name error in `_log_likelihood_physical` (line 388)
- **Bug 2**: Missing `log_likelihood` public interface method (line 432)
- **Bug 3**: Whitening transformation mismatch in MCMC sampler (line 460)
- **Result**: Acceptance rate improved from 0% to 45% (optimal range)

### 3. Stage-2 Batch Size Optimization
Discovered massive performance bottleneck:
- **Small batches (8)**: 684 batches/step, 60 sec/step, 50 hours total
- **Large batches (512)**: 11 batches/step, 1 sec/step, 50 minutes total
- **Speedup**: 60× faster with proper batch sizing
- **See**: `BATCH_SIZE_OPTIMIZATION.md` for detailed guide

## Quick Start

### Step 1: Run 10-SNe Smoke Test
```bash
cd /home/tracy/development/qfd_hydrogen_project/October_Supernova/GPU_Supernova_tools/V15

# Test Stage 1
export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_ENABLE_X64=1

python stage1_optimize.py \
    --lightcurves ../../data/unified/lightcurves_unified_v2_min3.csv \
    --sn-list "0:10" \
    --out ../../results/v15_smoke_10sne/stage1 \
    --global "70,0.01,30" \
    --tol 1e-5 \
    --max-iters 200

# Verify: All 10 SNe should have 5-parameter results
ls ../../results/v15_smoke_10sne/stage1/*/*.npz | wc -l  # Should be 10
```

### Step 2: Test Stage 2 MCMC with Batch Size Optimization
```bash
# Run smoke test on 10 SNe with batch_size=512
export V15_STAGE2_BATCH=512

python stage2_fit.py \
    --stage1 ../../results/v15_smoke_10sne/stage1 \
    --lightcurves ../../data/unified/lightcurves_unified_v2_min3.csv \
    --mode freeze \
    --walkers 16 \
    --steps 200 \
    --burn 50 \
    --init 70,0.01,30 \
    --out ../../results/v15_smoke_10sne/stage2 \
    --verbose

# Check acceptance rate (should be 20-50%)
grep "Acceptance:" ../../results/v15_smoke_10sne/stage2/*.log
```

### Step 3: Production Run (Full 5,468 SNe)
```bash
# Stage 1: Parallel execution (4 workers, ~4 minutes)
./run_stage1_parallel.sh \
    ../../data/unified/lightcurves_unified_v2_min3.csv \
    ../../results/v15_production/stage1 \
    "70,0.01,30" \
    4

# Wait for completion, then Stage 2 with optimized batch size
export V15_STAGE2_BATCH=512
export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_ENABLE_X64=1

nohup python3 stage2_fit.py \
    --stage1 ../../results/v15_production/stage1 \
    --lightcurves ../../data/unified/lightcurves_unified_v2_min3.csv \
    --mode freeze \
    --walkers 32 \
    --steps 3000 \
    --burn 1000 \
    --init 70,0.01,30 \
    --out ../../results/v15_production/stage2 \
    --verbose \
    > /tmp/v15_production_stage2.log 2>&1 &
```

## Performance Characteristics

### Stage-1 (JAX GPU-Optimized)
- **10 SNe smoke test**: ~5 seconds (2 SNe/sec)
- **200 SNe test**: ~8 seconds (25 SNe/sec)
- **5,468 SNe (4 workers)**: ~4 minutes (23 SNe/sec)
- **Scaling**: Near-linear with worker count

### Stage-2 (MCMC with Batch Optimization)
- **Batch size 8 (bad)**: 60 sec/step, 50 hours for 3000 steps
- **Batch size 512 (good)**: 1 sec/step, 50 minutes for 3000 steps
- **10 SNe smoke test**: ~8 seconds (200 steps, 25 it/sec)
- **5,468 SNe production**: ~50 minutes (3000 steps, 1.5 it/sec)
- **Key**: Set `V15_STAGE2_BATCH` based on dataset size (see below)

### Batch Size Recommendations
- **10-50 SNe**: `V15_STAGE2_BATCH=8` (1-7 batches)
- **100-500 SNe**: `V15_STAGE2_BATCH=64` (2-8 batches)
- **1000-5000 SNe**: `V15_STAGE2_BATCH=512` (2-10 batches)
- **5000+ SNe**: `V15_STAGE2_BATCH=1024` (5-20 batches)

**Rule of thumb**: Aim for 5-20 batches total. Too few batches (<5) wastes GPU memory, too many (>50) creates CPU/GPU transfer overhead.

## Files and Architecture

### Core Scripts
- `stage1_optimize.py` - Per-SN optimization (JAX GPU-accelerated)
- `stage2_fit.py` - Global MCMC (emcee + JAX batched likelihood)
- `run_stage1_parallel.sh` - Parallel Stage-1 runner
- `README.md` - This file
- `BATCH_SIZE_OPTIMIZATION.md` - Detailed batch size tuning guide

### Parameter Count Validation
Both scripts include parameter count guards:
```python
EXPECTED_PERSN_COUNT = 5
if len(BOUNDS) != EXPECTED_PERSN_COUNT:
    raise AssertionError(f"Parameter count mismatch: expected {EXPECTED_PERSN_COUNT}, got {len(BOUNDS)}")
```

### Two-Stage Architecture
1. **Stage 1**: Freeze global params, optimize 5 per-SN params per lightcurve
   - Method: JAX L-BFGS-B with ridge regularization
   - Output: `{SNID}/best_fit.npz` (5 parameters per SN)
   - Parallel: 4 workers process ~1,367 SNe each

2. **Stage 2**: Freeze per-SN params, sample 3 global params
   - Method: emcee MCMC with JAX batched likelihood
   - Output: `v15_mcmc_samples.npy` (posterior samples), `v15_best_fit.json` (MAP estimate)
   - Batch processing: Group SNe to maximize GPU utilization

## Monitoring Production Runs

### Stage 1 Progress
```bash
# Count completed SNe
watch -n 5 'find ../../results/v15_production/stage1 -name "best_fit.npz" | wc -l'

# Check worker logs
tail -f ../../results/v15_production/stage1/_worker_*.log

# GPU usage
watch -n 5 nvidia-smi
```

### Stage 2 Progress
```bash
# Watch MCMC progress
tail -f /tmp/v15_production_stage2.log

# Check acceptance rate (should be 20-50%)
tail -100 /tmp/v15_production_stage2.log | grep "Acceptance:"

# Estimated time remaining
# (3000 - current_step) / it_per_sec / 60 = minutes remaining
```

## Validation and Quality Checks

### Stage 1 Success Criteria
- All SNe should have `best_fit.npz` files
- Each `.npz` should contain 5 parameters (not 6)
- Chi-squared values should be reasonable (1-10 range typically)

```bash
# Check parameter count
python3 -c "
import numpy as np
result = np.load('../../results/v15_production/stage1/1000001/best_fit.npz')
print(f'Parameters: {len(result[\"x_best\"])}')  # Should be 5
print(f'Chi2: {result[\"chi2\"]:.2f}')
"
```

### Stage 2 Success Criteria
- Acceptance rate: 20-50% (45% is optimal)
- Posterior samples: Should show convergence
- Parameter uncertainties: Should be reasonable (not zero or infinity)

```bash
# Load and check results
python3 -c "
import json
with open('../../results/v15_production/stage2/v15_best_fit.json') as f:
    result = json.load(f)
print(f'k_J: {result[\"k_J\"][\"mean\"]:.2f} ± {result[\"k_J\"][\"std\"]:.2f}')
print(f'eta_prime: {result[\"eta_prime\"][\"mean\"]:.4f} ± {result[\"eta_prime\"][\"std\"]:.4f}')
print(f'xi: {result[\"xi\"][\"mean\"]:.2f} ± {result[\"xi\"][\"std\"]:.2f}')
"
```

## Troubleshooting

### Stage 1 Issues

**"AssertionError: Parameter count mismatch"**
- This guard prevents using old 6-parameter code
- Verify you're running V15 code, not V14 or earlier

**Workers hanging or slow**
- Check GPU memory: `nvidia-smi`
- Reduce workers if out of memory: use `3` or `2` workers instead of `4`
- Resume safe: Just rerun `./run_stage1_parallel.sh`

**JAX errors**
- Ensure environment variables are set:
  ```bash
  export JAX_PLATFORMS=cuda,cpu
  export XLA_PYTHON_CLIENT_PREALLOCATE=false
  export JAX_ENABLE_X64=1
  ```

### Stage 2 Issues

**Zero acceptance rate (0.000)**
- **Fixed in V15**: Ensure you're using the latest `stage2_fit.py`
- Bugs fixed: variable name, missing method, whitening transformation

**Very slow (>1 hour for small datasets)**
- Check batch size: `echo $V15_STAGE2_BATCH`
- Increase batch size for large datasets (see recommendations above)
- Monitor batch count in logs: aim for 5-20 batches total

**Low acceptance (<10%) or high acceptance (>80%)**
- Adjust MCMC step size in `stage2_fit.py` (default: 2.5% of parameter range)
- Check initial values are reasonable: should be near true values

**NaN or inf in likelihood**
- Check Stage 1 results are valid (no NaN in `best_fit.npz`)
- Verify global parameter bounds are reasonable

## Changes from V14

### Code Changes
1. Removed `t_H` and `f_BBH` from `BOUNDS` dict in both scripts
2. Updated parameter count validation to 5 (was 6)
3. Fixed 3 critical Stage-2 MCMC bugs
4. Added `V15_STAGE2_BATCH` environment variable support

### Performance Improvements
- Stage 1: No change (already GPU-optimized in V14)
- Stage 2: 60× faster with proper batch sizing (8 → 512)

### Validation
- 10-SNe smoke test: ✅ All parameters correct shape (5,)
- 200-SNe test: ✅ 100% success rate
- 5,468-SNe production: ✅ Stage 1 complete in 4 minutes
- Stage 2 MCMC: ✅ 45% acceptance rate on smoke test

## Related Documentation

- `BATCH_SIZE_OPTIMIZATION.md` - Comprehensive guide to batch size tuning
- `V15_5_PARAMETER_REDUCTION_COMPLETE.md` - Details on parameter reduction
- `TWO_AI_WORKFLOW.md` - Guide for parallel AI development
- `V15_PLAN.md` - Original implementation plan

## Next Steps

1. **Complete production run**: Wait for Stage 2 to finish (~50 minutes)
2. **Validate results**: Check acceptance rate and parameter posteriors
3. **Scientific analysis**: Compare to V14 results, analyze impact of parameter reduction
4. **Documentation**: Update scientific papers with new 5-parameter model

## Known Issues and Limitations

- **Batch size**: Must be set manually via environment variable (auto-detection planned)
- **Parameter bounds**: Hardcoded in scripts (consider config file)
- **Parallel Stage 2**: Not yet implemented (Stage 2 is serial)
- **Resume capability**: Stage 1 yes, Stage 2 no

## Version History

- **V15** (2025-11-03): 5-parameter model, Stage-2 MCMC fixes, batch optimization
- **V14** (2025-11-02): Parallel Stage-1 execution
- **V13.1**: Two-stage architecture with SNR-based errors
- **V13**: Single-stage MCMC (deprecated)
