# Strategy 4: Fine Scan Overnight Run

## Quick Start

**Wait for current run to finish** (~20 minutes remaining), then:

```bash
cd /home/tracy/development/QFD_SpectralGap/V22_Lepton_Analysis/validation_tests

# Start the fine scan
nohup python t3b_fine_scan_overnight.py > results/V22/logs/t3b_fine_scan.log 2>&1 &
echo $! > t3b_fine.pid

# Monitor
tail -f results/V22/logs/t3b_fine_scan.log
```

## Configuration

```
Lambda:  [0, 5e-10, 1e-09, 2e-09]           (4 values, sweet spot)
Beta:    [1.7, 1.8, 1.9, ..., 2.6, 2.7]    (11 points, step 0.1)
Workers: 4
Starts:  3
Popsize: 48
```

**Total:** 4 λ × 11 β = 44 combinations

## Expected Performance

- **Per beta:** ~6 minutes
- **Per lambda:** ~66 minutes (11 beta × 6 min)
- **Total runtime:** ~4.4 hours
- **Memory peak:** ~6 GB
- **Output files:**
  - `results/V22/t3b_fine_scan_full_data.csv` (44 rows when complete)
  - `results/V22/t3b_fine_scan_summary.csv` (4 rows, one per lambda)

## Monitoring

### Real-time log
```bash
tail -f results/V22/logs/t3b_fine_scan.log | grep -E "λ_curv|β.*e-|RESULTS|MEM"
```

### Progress check
```bash
# Count completed lambdas
wc -l results/V22/t3b_fine_scan_summary.csv

# Count completed betas
wc -l results/V22/t3b_fine_scan_full_data.csv
```

### Memory check
```bash
ps aux | grep t3b_fine | awk '{sum+=$6} END {print sum/1024 " MB"}'
```

## What This Will Tell Us

After completion, analyze with:

```python
import pandas as pd
import numpy as np

df = pd.read_csv("results/V22/t3b_fine_scan_full_data.csv")
summary = pd.read_csv("results/V22/t3b_fine_scan_summary.csv")

# Check if β is identified
for lam in summary['lam'].unique():
    data = df[df['lam'] == lam]
    cv_s = summary[summary['lam'] == lam]['CV_S'].values[0]
    print(f"λ = {lam:.2e}: CV(S) = {cv_s:.1f}%")

    if cv_s < 20:
        print("  → β IDENTIFIED ✓")
    elif cv_s < 40:
        print("  → β MARGINAL (try U scaling)")
    else:
        print("  → β UNIDENTIFIED (need 3-lepton)")
```

## Decision Tree

```
Morning analysis:
  ├─ CV < 20% → SUCCESS! Refine β further, production run
  ├─ CV 20-40% → Marginal, try Strategy 3 (U scaling law)
  └─ CV > 40% → Curvature penalty failed, must do Strategy 2 (3-lepton)
```

## Next Steps Based on Results

### If β Identified (CV < 20%)
- Refine around optimal β with step 0.02
- Production run with higher n_starts for publication

### If Marginal (CV 20-40%)
- Try Strategy 3: U scaling law scan
- Keep curvature penalty at λ = 1e-09

### If Unidentified (CV > 40%)
- Proceed to Strategy 2: 3-lepton fit
- Most robust solution, overconstrained system

## Stop Command

```bash
kill $(cat t3b_fine.pid)
```

## Restart from Crash

The script auto-resumes:
```bash
# Just run again
python t3b_fine_scan_overnight.py > results/V22/logs/t3b_fine_scan.log 2>&1 &
```

---

**Ready to launch once current run completes!**
