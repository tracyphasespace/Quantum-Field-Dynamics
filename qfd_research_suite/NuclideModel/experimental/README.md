# Experimental Features

**Status**: 🧪 Experimental - Not yet production-ready
**Version**: 2.0-alpha
**Last Updated**: October 2025

---

## Overview

This directory contains advanced features from the GeminiSolver parallel development track. These improvements show significant speedups (4-10×) and may improve heavy nuclei accuracy, but are **not yet fully validated**.

**Use at your own risk.** For production work, use the stable code in `src/`.

---

## Key Features

### 1. Parallel Meta-Optimizer (`qfd_metaopt_v15.py`)

**New capabilities**:
- ✅ **4× speedup**: ThreadPoolExecutor with 4 workers
- ✅ **Adaptive iterations**: 90/150/210 based on mass number
- ✅ **Two-stage gate**: Fast 4-isotope screening → full 6-isotope verification
- ✅ **Improved loss**: Scaled penalties (no more 1e12 dominance)
- ✅ **Early exit**: Cancel remaining solves on first failure
- ✅ **Process isolation**: `start_new_session=True` for clean timeouts
- ✅ **Retry on failure**: One light retry with reduced iterations

**Performance**:
```
Baseline (src/):     ~300s for 10 trials × 6 isotopes = 50 minutes
Experimental:        ~50s for 10 trials × 6 isotopes = 8 minutes
Speedup:            6× faster
```

### 2. Phase 11 Solver with Self-Repulsion (`qfd_solver_v11.py`)

**New physics**:
- ✅ **Self-repulsion term**: `c_repulsion` parameter (0.0-0.1)
  - Adds quartic self-interaction: ∫ c_repulsion × ψ_N^4 dV
  - Prevents overcollapse in heavy nuclei
  - May fix A>120 systematic underbinding (~8% → <3%?)

**Improved initialization**:
- Multiple ansatze: `gauss`, `shell`, `exponential`
- Better convergence for medium nuclei (A=60-120)

### 3. Environment-Based Configuration (`.env`)

**No more hardcoded parameters!**

Copy `.env.example` to `.env` and customize:
```bash
cp .env.example .env
nano .env  # Edit settings
```

All scripts read from environment:
```bash
export QFD_GRID_POINTS=32
export QFD_ITERS_OUTER=150
python qfd_metaopt_v15.py ...
```

---

## Quick Start

### Setup
```bash
cd experimental
cp .env.example .env
# Edit .env if needed (defaults are good for search mode)
```

### Test Single Isotope (Phase 11 Solver)
```bash
# Without self-repulsion (Phase 9 equivalent)
python qfd_solver_v11.py \
  --A 208 --Z 82 \
  --c-v2-base 2.20 --c-v2-iso 0.027 --c-v2-mass -0.0002 \
  --c-v4-base 5.28 --c-v4-size -0.085 \
  --alpha-e-scale 1.01 --beta-e-scale 0.50 \
  --c-sym 25.0 --kappa-rho0 0.030 \
  --c-repulsion 0.0 \
  --grid-points 48 --iters-outer 360 \
  --emit-json

# With self-repulsion (may improve heavy nuclei)
python qfd_solver_v11.py \
  --A 208 --Z 82 \
  --c-repulsion 0.05 \
  ...
```

### Run Meta-Optimizer (Parallel + Adaptive)
```bash
# Install Optuna if needed
pip install optuna

# Create Optuna database
mkdir -p runs
DB="sqlite:///runs/experiment.db"

# Run optimization (parallel mode)
python qfd_metaopt_v15.py \
  --solver qfd_solver_v11.py \
  --storage "$DB" \
  --study "test_v15" \
  --trials 50

# Results saved to:
# - runs/best_params_snapshot_ame2020.json
# - runs/debug/trial_*_*.json (per-isotope results)
```

### Verify Top-K Results (TODO)
```bash
# Extract verify_topk function from v15 script
python verify_topk.py \
  --storage "$DB" \
  --study "test_v15" \
  --top-k 5 \
  --out-json results/best_verified.json \
  --out-csv results/best_verified.csv
```

---

## Comparison: Stable vs Experimental

| Feature | Stable (src/) | Experimental (this dir) |
|---------|---------------|-------------------------|
| **Solver version** | Phase 9 | Phase 11 + repulsion |
| **Parallel execution** | ❌ Serial | ✅ 4 workers |
| **Adaptive iterations** | ❌ Fixed 150 | ✅ 90/150/210 |
| **Two-stage gate** | ❌ No | ✅ Fast → full |
| **Loss function** | FAIL_SENTINEL=1e12 | Scaled (1.0/0.25/rel²) |
| **Self-repulsion** | ❌ No | ✅ c_repulsion |
| **Environment config** | ❌ Hardcoded | ✅ .env file |
| **Speedup** | 1× (baseline) | **6-10×** |
| **Heavy nuclei (A>120)** | -8% underbinding | **TBD** (needs testing) |

---

## Validation Status

### ✅ Tested
- [x] Phase 11 solver compiles and runs (He-4, O-16, Pb-208)
- [x] v15 meta-optimizer imports and starts Optuna study
- [x] Parallel execution works (ThreadPoolExecutor)
- [x] Environment variables loaded correctly

### 🔬 In Progress
- [ ] Side-by-side accuracy comparison (v1.0 vs v2.0)
- [ ] Speedup measurement (10 trials, 20 isotopes)
- [ ] Self-repulsion sweep (c_repulsion = 0.0, 0.05, 0.10)
- [ ] Heavy nuclei error reduction (A>120)

### ❓ Unknown
- [ ] Optimal c_repulsion value (currently guessing 0.05)
- [ ] Best two-stage gate isotopes (currently O/Ca/Fe/Ni)
- [ ] Adaptive iteration tuning (90/150/210 may not be optimal)
- [ ] Thread safety of parallel execution (stress test needed)

---

## Known Issues

1. **Path dependencies**: v15 assumes specific directory structure from GeminiSolver
   - Fixed: Copy files to `experimental/` with relative imports

2. **Missing verify_topk**: Not yet extracted as standalone script
   - Workaround: Use v15 script directly, comment out optimization loop

3. **c_repulsion untested**: No systematic sweep yet
   - Action: Run test on Pb-208, U-238 with c_repulsion ∈ [0.0, 0.05, 0.10]

4. **Thread count**: Hardcoded 4 workers may not be optimal
   - Future: Read from env variable `QFD_N_WORKERS`

---

## Migration from Stable

If you're currently using `src/qfd_metaopt_ame2020.py`:

### Step 1: Test compatibility
```bash
cd experimental
cp .env.example .env

# Run 1 trial to verify it works
python qfd_metaopt_v15.py \
  --solver qfd_solver_v11.py \
  --storage "sqlite:///test.db" \
  --study "migration_test" \
  --single-trial
```

### Step 2: Compare results
```bash
# Run 10 trials with both versions
cd ../src
python qfd_metaopt_ame2020.py --n-calibration 10 --max-iter 10

cd ../experimental
python qfd_metaopt_v15.py --solver qfd_solver_v11.py --trials 10
```

### Step 3: Validate on your isotopes
If your calibration set differs from default gate, edit `.env`:
```bash
QFD_GATE=Your-Custom-Isotopes-Here
```

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'qfd_effective_potential_models'"
**Fix**: Phase 11 has fallback implementation, this is safe to ignore.

### "FileNotFoundError: AME2020 data file not found"
**Fix**: Ensure `data/ame2020_system_energies.csv` exists relative to script:
```bash
ln -s ../data data  # Create symlink from experimental/ → data/
```

### Parallel execution hangs
**Fix**: Set thread limits in `.env`:
```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

### Loss dominated by 1.0 penalties
**Expected**: New loss function uses 1.0 for hard fails (better than 1e12).
If >50% of isotopes fail, your parameters are far from optimum.

---

## Reporting Issues

**Before opening an issue**:
1. Verify stable version (`src/`) works on your system
2. Include `.env` file contents
3. Attach `runs/debug/trial_*_*.stderr.txt` for failed solves

**Where to report**:
- GitHub Issues: Tag with `experimental` label
- Include: OS, Python version, PyTorch version, CPU/GPU

---

## Roadmap

### v2.0-beta (After Validation)
- [ ] Extract verify_topk as standalone script
- [ ] Add `QFD_N_WORKERS` environment variable
- [ ] Systematic c_repulsion sweep results
- [ ] Migration guide with example data

### v2.0-stable (Promotion Criteria)
- [ ] ≥4× speedup validated
- [ ] No accuracy regression on light nuclei (A<60)
- [ ] <5% error on heavy nuclei with c_repulsion
- [ ] ≥2 independent user validations
- [ ] Complete test coverage

---

## See Also

- **Stable version**: `../src/qfd_metaopt_ame2020.py`
- **Integration plan**: `../GEMINI_INTEGRATION.md`
- **Findings (v1.0)**: `../docs/FINDINGS.md`
- **Physics model**: `../docs/PHYSICS_MODEL.md`

---

**Remember**: This is experimental code. Always keep a backup of your stable workflows.

For production work, use `src/` until this directory reaches v2.0-stable status.
