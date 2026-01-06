# Overnight Calibration Run - Quick Start

## Launch

```bash
./run_overnight_calibration.sh
```

This will:
- Run 4 parallel workers
- Optimize for 8 hours
- Target: Heavy isotopes (A ≥ 120)
- Goal: Reduce -12% systematic error to < -2%

## Monitor Progress

### Option 1: Watch mode (auto-updating)
```bash
python monitor_batch.py --watch
```

### Option 2: One-time check
```bash
python monitor_batch.py
```

### Option 3: Raw checkpoint
```bash
cat results/batch_optimization_*/checkpoint.json | jq '.n_completed'
```

## Configuration

**Strategy**: Multiple optimization runs with diversity
- Different random seeds (42, 43, 44, ...)
- Different isotope subsets per job
- Varied maxiter (300 → 50, decreasing)
- Varied popsize (15 → 12, decreasing)

**Parameters being optimized** (from `experiments/nuclear_heavy_region.runspec.json`):
- `c_v2_base`: Cohesion baseline [2.312, 2.642] (+5% to +20%)
- `c_v4_base`: Repulsion baseline [4.490, 5.282] (-15% to 0%)
- `c_v2_iso`: Isospin cohesion [0.02163, 0.03244]
- `c_v4_size`: Size repulsion [-0.1276, -0.0638]

**Fixed parameters**: All others (c_sym, kappa_rho, etc.)

## Expected Output

**During run:**
```
[Job 00] Starting optimization...
[Job 01] Starting optimization...
...
[Job 00] ✓ Complete in 43.2 min | Error: -3.21% | Score: 4.53
[Job 01] ✓ Complete in 51.7 min | Error: -2.89% | Score: 4.12
```

**Final summary:**
```
🏆 Top 5 Results:
  Rank 1:
    Score: 3.42
    Mean Error: -2.15%
    Std Error: 2.54%
    Max Error: -5.32%
    Mean Virial: 0.142
```

## Results Location

```
results/batch_optimization_YYYYMMDD_HHMMSS/
├── checkpoint.json               # Progress tracking
├── best_parameters.json          # Best result found
├── optimization.log              # Full log
├── job_00_config.json           # Job configs
├── job_01_config.json
└── ...
```

## Stopping Gracefully

Press `Ctrl+C` - checkpoint will be saved automatically.

To resume later:
```bash
python src/batch_optimize.py \
    --config experiments/nuclear_heavy_region.runspec.json \
    --workers 4 \
    --hours 8 \
    --output-dir results/batch_optimization_PREVIOUS_RUN
```

It will ask if you want to resume from checkpoint.

## Estimations

**Jobs**: ~40-50 total over 8 hours
- Each job: ~30-60 minutes
- 4 parallel workers
- Each job tests 8 heavy isotopes
- Total coverage: ~320-400 isotope evaluations

**Memory**: 4 GB per worker × 4 = 16 GB total

**Success criteria**:
- Mean error: < -3.0% (currently -8.4%)
- Max error: < -6.0% (currently -12%)
- All isotopes converged (virial < 0.18)

## Troubleshooting

### Out of memory
```bash
# Reduce workers
./run_overnight_calibration.sh  # Edit WORKERS=2
```

### Jobs timing out
- Check `optimization.log` for error messages
- May need to reduce `maxiter` in job configs

### No progress after 1 hour
```bash
# Check if processes are running
ps aux | grep batch_optimize

# Check if solvers are running
ps aux | grep qfd_solver
```

### Resuming from crash
The checkpoint system automatically saves progress. If the run crashes, simply re-run the script and it will ask to resume.

## Post-Processing

After completion, apply best parameters:

```bash
# Extract best parameters
python -c "
import json
with open('results/batch_optimization_*/best_parameters.json') as f:
    data = json.load(f)
print(json.dumps(data['parameters'], indent=2))
" > parameters/heavy_region_optimized.params.json
```

Then create a new RunSpec with these parameters for validation.

## Expected Improvements

**Before (Trial 32)**:
- Light (A<60): -0.68% ✓
- Medium (60≤A<120): -5.75% ⚠️
- Heavy (A≥120): -8.42% ✗

**After (Regional calibration)**:
- Light: -0.68% ✓ (unchanged, not optimized)
- Medium: -5.75% ⚠️ (unchanged, separate optimization needed)
- Heavy: **-2.0% to -3.0%** ✓ (target improvement)

**Physics**: Increased cohesion compensates for surface tension underbinding in heavy nuclei. Decreased repulsion allows tighter binding at large A.
