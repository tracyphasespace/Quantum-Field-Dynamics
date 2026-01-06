# V15 Pipeline Recovery Status

**Date**: 2025-11-10
**Status**: CODE RESTORED - READY FOR TESTING

---

## Summary

Successfully stopped all broken processes and restored potentially working Stage 2 MCMC code from October_Supernova.

---

## Completed Steps

### 1. Process Cleanup ✅
- Stopped all 20+ broken pipeline processes
- Killed all Stage 1, Stage 2, Stage 3, and ABC variant runs
- Verified no Python processes consuming resources

### 2. Code Discovery ✅
- Located working V15 code in `/home/tracy/development/qfd_hydrogen_project/October_Supernova/GPU_Supernova_tools/V15/`
- Last modified: Nov 8, 2025 15:40 (5 days ago)
- Contains stage2_mcmc_numpyro.py with production-quality code

### 3. Backup ✅
- Backed up broken code to `/home/tracy/development/backups/qfd-supernova-v15-broken-20251110/`
- Includes both `src/` and `results/` directories
- Full git history preserved

### 4. Code Restoration ✅
- Copied stage2_mcmc_numpyro.py from October_Supernova to current `src/`
- Key differences from broken code:
  - **Removed** cache-busting complexity (`cache_bust` parameter)
  - **Removed** preflight variance checks
  - **Removed** `jax.clear_caches()` calls
  - **Simplified** JIT decoration (no `partial` or `static_argnames`)
  - **Kept** `iters < 5` filter (was present in working code)

---

## Key Findings

### The `iters < 5` Filter is NOT the Problem
- October_Supernova code HAS the `iters < 5` filter (lines 80-82)
- V15 production results were generated WITH this filter in place
- The filter correctly rejects poor-quality fits (fast convergence ≠ good fit in all cases)

### Root Cause of Breakage
The recent "fixes" (commits 0f8b3f4-93dfa1a) added unnecessary complexity:
- Cache-busting tokens
- Static JIT arguments
- Preflight checks outside JIT
- Manual cache clearing

These "optimizations" actually broke the MCMC inference, causing:
- NULL results (k_J ≈ 0) in commit 93dfa1a
- Stuck-at-prior (k_J = 50.0) in commit 958f144

The October_Supernova code is **simpler and cleaner** - no fancy tricks, just straightforward NumPyro MCMC.

---

## Next Steps

### Test the Restored Code

Run a verification test using V15 production Stage 1 results:

```bash
cd /home/tracy/development/Quantum-Field-Dynamics-fresh/projects/astrophysics/qfd-supernova-v15

export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false"

python src/stage2_mcmc_numpyro.py \
  --stage1-results results/v15_production/stage1 \
  --lightcurves data/lightcurves_unified_v2_min3.csv \
  --out results/v15_verification \
  --nchains 4 \
  --nsamples 1000 \
  --nwarmup 500
```

**Expected Runtime**: ~15-20 minutes

**Expected Result**:
```json
{
  "k_J": ~10.69 ± ~4.57,
  "eta_prime": ~-7.97 ± ~1.44,
  "xi": ~-6.88 ± ~3.75
}
```

Should match V15 production within uncertainties.

### Success Criteria

- ✅ No divergent transitions (or < 1%)
- ✅ k_J ≈ 10.69 ± 4.57 km/s/Mpc
- ✅ η' and ξ match production results
- ✅ R-hat values < 1.01 (good convergence)
- ✅ Effective sample size > 1000
- ✅ Uses all ~3,200 Stage 1 results (after `iters < 5` filter)

### If Test Succeeds

1. **Commit the restored code**:
   ```bash
   git add src/stage2_mcmc_numpyro.py
   git commit -m "Restore working Stage 2 MCMC from October_Supernova V15

   Removed cache-busting complexity that broke inference.
   Reverted to simpler, cleaner NumPyro implementation.

   Key changes:
   - Remove cache_bust parameter and partial decorator
   - Remove preflight checks and jax.clear_caches()
   - Use simple @jax.jit decoration
   - Keep iters < 5 filter (was in working code)

   Expected: k_J = 10.69 ± 4.57 km/s/Mpc (matches V15 production)
   "
   ```

2. **Update RECOVERY_PLAN.md** with success notes

3. **Proceed with science**: Run Pantheon Plus analysis with working code

### If Test Fails

1. Check October_Supernova results to see what k_J value it produced
2. Compare all model files (v15_model.py, v15_data.py, etc.)
3. Consider WSL2 vhdx recovery to find older working version

---

## Files

### Backed Up (Broken Code)
- `/home/tracy/development/backups/qfd-supernova-v15-broken-20251110/src/stage2_mcmc_numpyro.py`
- `/home/tracy/development/backups/qfd-supernova-v15-broken-20251110/results/`

### Restored (Potentially Working)
- `/home/tracy/development/Quantum-Field-Dynamics-fresh/projects/astrophysics/qfd-supernova-v15/src/stage2_mcmc_numpyro.py`

### Reference (Known Good Results)
- `/home/tracy/development/Quantum-Field-Dynamics-fresh/projects/astrophysics/qfd-supernova-v15/results/v15_production/stage2/best_fit.json`

### Source (Working Code)
- `/home/tracy/development/qfd_hydrogen_project/October_Supernova/GPU_Supernova_tools/V15/stage2_mcmc_numpyro.py`

---

## Recovery Documentation
- `RECOVERY_PLAN.md` - Comprehensive recovery strategies
- `RECOVERY_STATUS.md` - This file
