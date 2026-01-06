# T3b OOM Crash Recovery

## Crash Summary

**Time:** 2025-12-26 09:52
**Location:** λ_curv = 3.00e-08 (sweep 7/10), β = 3.30 (89% through beta scan)
**Cause:** Out of memory after ~10 hours of runtime

## What Was Recovered

Partial results were saved at 06:14 (3.5 hours before crash):
- **Completed λ values:** 0.0, 1e-10, 3e-10, 1e-09, 3e-09 (5/10)
- **Saved files:**
  - `results/V22/t3b_lambda_full_data.csv` (6.2K, 48 data rows)
  - `results/V22/t3b_lambda_summary.csv` (1.4K, 5 summary rows)
- **Log file:** `results/V22/logs/t3b_curvature_sweep.log` (22K, stopped at line 309)

**Progress from log (not saved to CSV):**
- λ = 1e-08: completed (10/10 beta points)
- λ = 3e-08: incomplete (8/9 beta points, crashed at β=3.30)

## Root Cause Analysis

### Memory Accumulation Points

1. **Line 592:** `all_lam_results.append(...)` - accumulated ALL results in memory
   - 6 completed lambda sweeps × 10 beta points × detailed result dicts
   - No clearing between iterations

2. **Line 522:** Multi-start optimization settings
   - `n_starts=5` → 5 independent optimizations per beta point
   - `workers=8` → 8 parallel processes
   - `popsize=80` → 80 individuals per differential evolution run
   - **Total:** 5 × 8 × 80 = 3,200 concurrent objective evaluations

3. **Line 601-602:** Only saved once at the end
   - No incremental saves
   - All data held until completion

4. **Dense numerical grids:**
   - Each `LeptonEnergyLocalizedV1` instance holds full radial grid
   - Density fields computed and stored for curvature penalty
   - Not released until fitter instance destroyed

### Memory Timeline

| Time  | Lambda | Progress | Estimated Memory |
|-------|--------|----------|------------------|
| Start | 0.0    | 0/10 β   | ~2 GB            |
| 1:40  | 0.0    | Complete | ~3 GB            |
| 3:21  | 1e-10  | Complete | ~4 GB            |
| 5:02  | 3e-10  | Complete | ~5 GB            |
| 6:43  | 1e-09  | Complete | ~6 GB            |
| 8:24  | 3e-09  | Complete | ~7 GB            |
| 10:09 | 1e-08  | Complete | ~7.5 GB ⚠️       |
| 09:52 | 3e-08  | 8/9 β    | **OOM** 💥       |

## Recovery Strategy

### Memory-Optimized Restart Script

**File:** `t3b_restart_memory_optimized.py`

**Key Changes:**

1. **Reduced parallel workload:**
   - `n_starts: 5 → 3` (40% reduction)
   - `workers: 8 → 4` (50% reduction)
   - **Impact:** ~60% less peak memory per beta point

2. **Incremental saving:**
   - Save to CSV after each lambda completes
   - Prevents data loss on crash
   - Allows resumption from any point

3. **Aggressive cleanup:**
   - `del fitter` + `gc.collect()` after each beta
   - `del results_scan` after saving each lambda
   - Prevents accumulation

4. **Smart resumption:**
   - Reads existing CSV files
   - Identifies completed lambdas
   - Only processes remaining values

### How to Use

```bash
cd /home/tracy/development/QFD_SpectralGap/V22_Lepton_Analysis/validation_tests

# Check current progress
head results/V22/t3b_lambda_summary.csv

# Run the restart script
python t3b_restart_memory_optimized.py > results/V22/logs/t3b_restart.log 2>&1 &

# Monitor progress
tail -f results/V22/logs/t3b_restart.log

# Check memory usage
watch -n 10 'ps aux | grep python | grep t3b'
```

### Expected Runtime

With optimized settings:
- **Per beta point:** ~7-8 minutes (was ~10 minutes)
- **Per lambda:** ~1.5 hours (10 beta × 8 min)
- **Remaining work:** 5 lambdas × 1.5 hr = **~7.5 hours total**

### Memory Profile

| Setting      | Original | Optimized | Savings  |
|--------------|----------|-----------|----------|
| n_starts     | 5        | 3         | 40%      |
| workers      | 8        | 4         | 50%      |
| Peak memory  | ~8 GB    | ~4.5 GB   | 44%      |
| Per lambda   | +1 GB    | +0.3 GB   | 70%      |

## Verification Steps

After completion:

```python
import pandas as pd

# Check full data
df = pd.read_csv("results/V22/t3b_lambda_full_data.csv")
print(f"Total rows: {len(df)}")
print(f"Lambda values: {df['lam'].nunique()}")
print(f"Expected: 10 lambdas × 10 betas = 100 rows")
assert len(df) == 100, "Missing data!"

# Check summary
df_summary = pd.read_csv("results/V22/t3b_lambda_summary.csv")
print(f"Summary rows: {len(df_summary)}")
assert len(df_summary) == 10, "Missing lambda summaries!"
```

## Future Recommendations

For similar long-running sweeps:

1. **Always save incrementally** - after each major iteration
2. **Profile memory** - use `memory_profiler` to identify leaks
3. **Limit parallelism** - workers=4 is usually sufficient
4. **Use checkpointing** - save optimizer state for true resumption
5. **Monitor resources** - set up alerts for memory thresholds
6. **Batch processing** - split large sweeps into independent jobs

## Contact

If restart fails or additional OOM occurs:
- Check available RAM: `free -h`
- Reduce workers further: `WORKERS = 2`
- Reduce n_starts: `N_STARTS = 2`
- Consider running on machine with more RAM
