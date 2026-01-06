# Running t3b with 4GB Memory Budget

## Quick Start

```bash
cd /home/tracy/development/QFD_SpectralGap/V22_Lepton_Analysis/validation_tests

# Check current system memory
free -h

# Start the optimized run in background
nohup python t3b_restart_4gb.py > results/V22/logs/t3b_4gb_run.log 2>&1 &

# Save the PID
echo $! > t3b.pid

# Monitor the log
tail -f results/V22/logs/t3b_4gb_run.log
```

## Memory Monitoring

### Real-time Monitor (in separate terminal)

```bash
cd /home/tracy/development/QFD_SpectralGap/V22_Lepton_Analysis/validation_tests
./monitor_memory.sh
```

This shows:
- **Green**: < 3.5 GB (safe)
- **Yellow**: 3.5-3.8 GB (warning)
- **Red**: > 3.8 GB (danger - near limit)

### Manual Check

```bash
# Find PID
pgrep -f t3b_restart_4gb.py

# Check memory (replace PID)
ps aux | grep [P]ID
htop -p PID

# Memory history
while true; do
    ps -p $(pgrep -f t3b_restart_4gb.py) -o rss= | awk '{print $1/1024 " MB"}';
    sleep 30;
done
```

## Configuration Summary

| Setting      | Original | 4GB Optimized | Reduction |
|--------------|----------|---------------|-----------|
| **Beta grid**| 10 pts   | 7 pts         | 30%       |
| **n_starts** | 5        | 2             | 60%       |
| **workers**  | 8        | 2             | 75%       |
| **popsize**  | 80       | 40            | 50%       |
| **Peak RAM** | ~8 GB    | ~3.5 GB       | 56%       |

## Expected Runtime

- **Per beta point:** ~7 minutes
- **Per lambda:** 7 beta × 7 min = **~49 min**
- **Remaining lambdas:** Check CSV for completed ones
- **Total (if 5 remaining):** 5 × 49 min = **~4 hours**

## Progress Tracking

### Check Completed Lambdas

```bash
# View summary
cat results/V22/t3b_lambda_summary.csv

# Count completed
tail -n +2 results/V22/t3b_lambda_summary.csv | wc -l

# List completed lambda values
tail -n +2 results/V22/t3b_lambda_summary.csv | cut -d',' -f1
```

### Live Progress

```bash
# Watch the log file
tail -f results/V22/logs/t3b_4gb_run.log

# Filter for key events
tail -f results/V22/logs/t3b_4gb_run.log | grep -E "λ_curv|RESULTS|Saved|MEM"

# Count beta completions
grep -c "^[0-9]" results/V22/logs/t3b_4gb_run.log
```

## Resumability

The script automatically:
1. **Reads existing CSV files** on startup
2. **Identifies completed lambdas**
3. **Only processes remaining values**
4. **Saves after each lambda** (can resume from any crash)

To manually restart after interruption:
```bash
# Just run again - it will auto-resume
python t3b_restart_4gb.py > results/V22/logs/t3b_4gb_run.log 2>&1 &
```

## Verification

After completion:

```python
import pandas as pd

# Check full data
df = pd.read_csv("results/V22/t3b_lambda_full_data.csv")
print(f"Total rows: {len(df)}")
print(f"Lambda values: {df['lam'].nunique()}")
print(f"Unique betas per lambda: {df.groupby('lam')['beta'].nunique()}")

# Expected: 10 lambdas × 7 betas = 70 rows
assert len(df) == 70, f"Expected 70 rows, got {len(df)}"
assert df['lam'].nunique() == 10, f"Expected 10 lambdas, got {df['lam'].nunique()}"

# Check summary
df_summary = pd.read_csv("results/V22/t3b_lambda_summary.csv")
print(f"\nSummary rows: {len(df_summary)}")
assert len(df_summary) == 10, f"Expected 10 summary rows, got {len(df_summary)}"

print("\n✓ All data present and valid!")
```

## Troubleshooting

### If it crashes with OOM again:

1. **Further reduce workers:**
   ```python
   # In t3b_restart_4gb.py, line 106
   WORKERS = 1  # Change from 2 to 1
   ```

2. **Further reduce n_starts:**
   ```python
   # Line 105
   N_STARTS = 1  # Change from 2 to 1 (single-start only)
   ```

3. **Check for memory leaks:**
   ```bash
   # Monitor for gradual growth
   watch -n 5 'ps aux | grep t3b_restart_4gb'
   ```

### If it's too slow:

Current settings trade speed for memory. To speed up (if you have spare RAM):

```python
WORKERS = 3      # Instead of 2
N_STARTS = 3     # Instead of 2
POPSIZE_MULT = 12  # Instead of 10
```

## Kill/Stop

```bash
# Graceful stop (let current beta finish)
kill $(cat t3b.pid)

# Force stop (immediate)
kill -9 $(cat t3b.pid)

# Cleanup
rm t3b.pid
```

## Performance Comparison

| Configuration        | Runtime | Memory | Accuracy  |
|----------------------|---------|--------|-----------|
| Original (crashed)   | ~10 hr  | 8+ GB  | Best      |
| Memory-optimized     | ~8 hr   | 4.5 GB | Very Good |
| **This (4GB)**       | ~5 hr   | 3.5 GB | Good      |

**Accuracy notes:**
- Fewer multi-starts (2 vs 5) → slightly higher risk of local minima
- Smaller population (40 vs 80) → slightly less exploration
- Coarser beta grid (7 vs 10) → less resolution but captures trend

For production validation, may want to re-run critical lambda values with higher settings.
