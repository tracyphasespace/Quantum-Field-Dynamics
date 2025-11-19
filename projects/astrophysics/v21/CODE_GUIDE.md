# V21 Code Guide - Executables vs Libraries

## Quick Reference: What to Run

**EXECUTABLES (can run directly):**
```bash
python3 plot_canonical_comparison.py           # Generate comparison plots
python3 analyze_bbh_candidates.py              # BBH forensics analysis
python3 stage1_v20_fullscale_runner.py         # Run Stage 1 pipeline
./run_analysis.sh                              # Quick wrapper for plots
```

**LIBRARIES (imported by executables):**
- All other .py files are modules imported by the executables above

---

## Complete File Breakdown

### EXECUTABLE SCRIPTS (4 files)

**1. plot_canonical_comparison.py**
- **Purpose:** Generate ΛCDM vs QFD comparison plots
- **Inputs:** `data/stage2_results_with_redshift.csv`
- **Outputs:** `results/time_dilation_test.png`, `results/canonical_comparison.png`
- **Run:** `python3 plot_canonical_comparison.py`
- **Time:** ~30 seconds

**2. analyze_bbh_candidates.py**
- **Purpose:** Search for periodic BBH lensing signals
- **Inputs:** Stage 2 results + raw lightcurves
- **Outputs:** Periodogram analysis results
- **Run:** See FORENSICS_RESULTS.md for usage
- **Time:** ~minutes (depends on number of candidates)

**3. stage1_v20_fullscale_runner.py** ⚠️
- **Purpose:** Run Stage 1 fitting on all supernovae
- **Inputs:** `data/lightcurves_all_transients.csv`
- **Outputs:** Individual JSON files per SN
- **Run:** `python3 stage1_v20_fullscale_runner.py --lightcurves data/lightcurves_all_transients.csv --output results/stage1_output/ --batch-size 100`
- **Time:** HOURS (8,253 SNe to fit)
- **Note:** Uses parallel processing, memory-intensive

**4. stage2_select_candidates.py**
- **Purpose:** Select BBH candidates from Stage 1 results
- **Inputs:** Stage 1 JSON output directory
- **Outputs:** Summary CSV with candidates
- **Run:** `python3 stage2_select_candidates.py --stage1-dir results/stage1_output/ --output data/`
- **Time:** ~minutes

---

### LIBRARY MODULES (8 files)

**Core Physics & Models:**

**v17_lightcurve_model.py** (37 KB)
- Main lightcurve fitting engine
- Implements QFD + plasma veil model
- Used by: `stage1_v20_fullscale.py`

**v17_qfd_model.py** (3 KB)
- QFD cosmology equations
- Distance modulus calculations
- Used by: `v17_lightcurve_model.py`

**v18_bbh_model.py** (6 KB)
- Binary black hole lensing physics
- Gravitational microlensing calculations
- Used by: BBH analysis scripts

**v17_data.py** (14 KB)
- Data loading and preprocessing
- Lightcurve CSV parsing
- SNID handling (strips whitespace)
- Used by: All pipeline scripts

**Pipeline Support:**

**stage1_v20_fullscale.py** (9 KB)
- Library for Stage 1 fitting logic
- Defines `run_single_sn_optimization()` function
- Used by: `stage1_v20_fullscale_runner.py`
- **NOT EXECUTABLE** - imported by runner

**bbh_initialization.py** (6 KB)
- BBH parameter initialization
- Prior distributions
- Used by: BBH fitting scripts

**bbh_robust_optimizer.py** (10 KB)
- Outlier-resistant optimization
- Robust loss functions
- Used by: Stage 1 fitting

**pipeline_io.py** (5 KB)
- I/O utilities for JSON reading/writing
- Result serialization
- Used by: All pipeline stages

---

## Typical Workflows

### Workflow 1: Just Reproduce Plots (Recommended)
**Use pre-computed results**
```bash
cd v21
python3 plot_canonical_comparison.py
```
Output: `results/time_dilation_test.png`, `results/canonical_comparison.png`

### Workflow 2: Rerun Stage 1 from Raw Data
**Full pipeline recomputation (HOURS)**
```bash
# Stage 1: Fit all supernovae (memory-intensive, parallel)
python3 stage1_v20_fullscale_runner.py \
  --lightcurves data/lightcurves_all_transients.csv \
  --output results/stage1_output/ \
  --batch-size 100 \
  --num-cores 4

# Stage 2: Select candidates
python3 stage2_select_candidates.py \
  --stage1-dir results/stage1_output/ \
  --output data/

# Generate plots
python3 plot_canonical_comparison.py
```

### Workflow 3: BBH Forensics Analysis
```bash
python3 analyze_bbh_candidates.py \
  --stage2-results data/stage2_results_with_redshift.csv \
  --lightcurves data/lightcurves_all_transients.csv \
  --out results/forensics/ \
  --top-n 10
```

---

## Import Dependency Tree

```
Executables:
├── plot_canonical_comparison.py
│   ├── imports: pathlib, numpy, pandas, matplotlib, scipy
│   └── (self-contained, no custom imports)
│
├── stage1_v20_fullscale_runner.py
│   ├── imports: stage1_v20_fullscale
│   └── imports: v17_data
│       └── imports: pandas, numpy
│
├── analyze_bbh_candidates.py
│   ├── imports: v17_data
│   └── imports: scipy (Lomb-Scargle)
│
└── stage2_select_candidates.py
    └── imports: pipeline_io

Libraries (imported by above):
├── stage1_v20_fullscale.py
│   ├── imports: v17_lightcurve_model
│   ├── imports: v17_data
│   └── imports: bbh_robust_optimizer
│
├── v17_lightcurve_model.py
│   ├── imports: v17_qfd_model
│   └── imports: jax, scipy
│
└── v18_bbh_model.py
    └── imports: numpy
```

---

## Memory Requirements

**plot_canonical_comparison.py:** ~100 MB RAM
**analyze_bbh_candidates.py:** ~500 MB RAM
**stage1_v20_fullscale_runner.py:**
- Per core: ~200-500 MB
- Total: batch_size × 5 MB + num_cores × 200 MB
- Example: 100 SNe batch + 4 cores = ~1.3 GB RAM

---

## CORRECTED Usage Examples

**WRONG:**
```bash
# This will NOT work - stage1_v20_fullscale.py is a library!
python3 stage1_v20_fullscale.py  # ✗ NO
```

**CORRECT:**
```bash
# Use the RUNNER script
python3 stage1_v20_fullscale_runner.py --lightcurves data/lightcurves_all_transients.csv ...  # ✓ YES
```

---

## For AI Assistants

When helping users run the pipeline:
1. **Quick analysis:** Direct them to `plot_canonical_comparison.py`
2. **Full rerun:** Direct them to `stage1_v20_fullscale_runner.py` (NOT stage1_v20_fullscale.py)
3. **Forensics:** Direct them to `analyze_bbh_candidates.py`

The pre-computed results in `data/stage2_results_with_redshift.csv` are already available, so users can reproduce all plots immediately without rerunning Stage 1.
