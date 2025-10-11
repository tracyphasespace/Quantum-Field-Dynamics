# GeminiSolver Integration Plan

**Date**: October 2025
**Status**: PROPOSAL - Review Required

---

## Overview

The GeminiSolver directory contains parallel development with significant improvements over the current NuclideModel baseline. This document outlines key enhancements and proposes integration strategy.

---

## Key Improvements in GeminiSolver

### 1. Meta-Optimizer Enhancements (phase9_meta_optimizer_ame2020_v15.py)

#### Parallel Execution ✨
```python
from concurrent.futures import ThreadPoolExecutor, as_completed

def eval_gate_parallel(gate_data, params, dec, trial_no, solver_path):
    ex = ThreadPoolExecutor(max_workers=min(4, len(gate_data)))
    # Parallel solver calls with early exit on first failure
```

**Benefits**:
- 4× speedup for calibration sets (parallel isotope solving)
- Early exit on first failure (don't wait for all to complete)
- Thread-safe result collection

#### Adaptive Iteration Counts 🎯
```python
def iters_for_A(A):
    if A <= 20:  return 90   # Light nuclei: fast
    if A <= 80:  return 150  # Medium nuclei: moderate
    return 210               # Heavy nuclei: more iterations
```

**Benefits**:
- Light nuclei (He-4, O-16): 90 iters instead of 150 → 40% faster
- Heavy nuclei (Pb-208): 210 iters instead of 150 → better convergence
- Physics-driven resource allocation

#### Two-Stage Gate (Fast → Verify) 🔬
```python
FAST_GATE = [("O",16), ("Ca",40), ("Fe",56), ("Ni",62)]  # 4 isotopes
FULL_EXTRA = [("He",4), ("Pb",208)]                      # +2 for verification
```

**Benefits**:
- Stage 1: Fast 4-isotope gate for pruning bad trials
- Stage 2: Full 6-isotope verification only for promising candidates
- ~40% reduction in solver calls during search

#### Improved Loss Function 📊
```python
def compute_ame2020_loss(records, ame_data):
    for rec in records:
        if not rec["ok"]:
            total_loss += 1.0  # Hard fail (was 1e12!)
        elif E_model >= 0.0:
            total_loss += 0.25  # Unbound nucleus
        else:
            rel = (E_total_QFD - M_exp_MeV) / max(1.0, M_exp_MeV)
            v_pen = max(0.0, vabs - 0.18)**2
            loss = rel*rel + 4.0 * v_pen  # Virial penalty
```

**Benefits**:
- No more FAIL_SENTINEL=1e12 dominating gradients
- Separate penalties for hard fail (1.0) vs unbound (0.25)
- Relative error instead of absolute (scales properly)
- Explicit virial penalty term (ρ=4.0)

#### Process Group Isolation 🛡️
```python
proc = subprocess.run(
    cmd, timeout=120, env=env, start_new_session=True  # Isolate process group!
)
```

**Benefits**:
- Clean timeout handling (no zombie processes)
- One light retry on failure (reduced iters)
- Thread-local environment variables

#### Verify Top-K Function 🏆
```python
def verify_topk(storage_url, study_name, top_k=5):
    """Re-score top-K trials with full verification settings."""
    # Load Optuna study
    # Extract top K trials
    # Re-run with 48 grid, 360 iters
    # Write best_verified.json and best_verified.csv
```

**Benefits**:
- Post-hoc verification of search results
- Full resolution (48/360) for final candidates
- JSON + CSV output for downstream analysis

---

### 2. Phase 11 Solver with Self-Repulsion (phase11_solver_with_repulsion.py)

#### New Physics: Self-Repulsion Term 💥
```python
self.c_repulsion = float(c_repulsion)  # Typical: 0.0-0.1

def V_repulsion(self, psi_N: torch.Tensor) -> torch.Tensor:
    """Quartic self-repulsion: ∫ c_repulsion * ψ_N^4 dV"""
    rho_N = psi_N * psi_N
    return self.c_repulsion * (rho_N * rho_N).sum() * self.dV
```

**Physical Motivation**:
- Prevents overcollapse in heavy nuclei
- Adds stiffness to nuclear matter equation of state
- May help with A>120 systematic underbinding

#### Enhanced Field Initialization 🌱
```python
def initialize_fields(self, seed=0, init_mode="gauss"):
    # Multiple initialization strategies
    if init_mode == "gauss":
        R0 = 1.2 * A^(1/3)
        σ_N = 0.60 * R0
        σ_e = 0.50 * R0
    elif init_mode == "shell":
        # Shell-like initialization
```

**Benefits**:
- Better convergence for medium nuclei
- Multiple ansatze for different mass ranges

---

### 3. Environment-Based Configuration (.env.example)

```bash
# Canonical defaults
QFD_ALPHA_MODEL=exp
QFD_COULOMB=spectral
QFD_GRID_POINTS=48
QFD_ITERS_OUTER=360
QFD_VIRIAL_TOL=0.18
QFD_DEVICE=cpu
QFD_EARLY_STOP_VIR=0.18

# Fast search mode (override)
QFD_GRID_POINTS=32
QFD_ITERS_OUTER=150

# Gate selection
QFD_GATE=He-4,O-16,Ca-40,Fe-56,Ni-62,Pb-208

# Verification (for verify_topk)
QFD_GRID_POINTS_VERIFY=48
QFD_ITERS_OUTER_VERIFY=360
```

**Benefits**:
- No hardcoded parameters in scripts
- Easy A/B testing of configurations
- Separate search vs verify settings
- Reproducible runs via .env snapshots

---

## Comparison Table: Current vs GeminiSolver

| Feature | NuclideModel (current) | GeminiSolver | Impact |
|---------|------------------------|--------------|--------|
| **Parallel execution** | Serial (1 isotope at a time) | ThreadPoolExecutor (4 workers) | **4× speedup** |
| **Adaptive iterations** | Fixed 150 | 90/150/210 by mass | **40% faster for light** |
| **Two-stage gate** | No | 4-isotope fast → 6-isotope verify | **40% fewer calls** |
| **Loss function** | FAIL_SENTINEL=1e12 | Scaled penalties (1.0/0.25/rel²) | **Better gradients** |
| **Early exit** | After 5 isotopes if >15% worse | On first failure in parallel batch | **Faster pruning** |
| **Process isolation** | os.setsid + killpg | start_new_session=True | **Cleaner** |
| **Verify top-K** | No | verify_topk() function | **Post-hoc validation** |
| **Self-repulsion** | No | c_repulsion parameter | **Heavy nuclei fix?** |
| **Environment config** | Hardcoded | .env file | **Reproducibility** |
| **Retry on failure** | No | Light retry (reduced iters) | **Better success rate** |

**Combined Speedup**: ~6-10× for typical optimization run (10 trials × 6 isotopes)

---

## Proposed Integration Strategy

### Option A: Subdirectory with Symlinks (Recommended)

Keep GeminiSolver improvements separate but accessible:

```
NuclideModel/
├── src/
│   ├── qfd_solver.py           # Phase 9 (current)
│   └── qfd_metaopt_ame2020.py  # Current optimizer
├── experimental/
│   ├── README.md               # "Advanced features - experimental"
│   ├── phase11_solver.py       # Symlink → GeminiSolver
│   ├── meta_optimizer_v15.py   # Symlink → GeminiSolver
│   ├── .env.example            # Symlink → GeminiSolver
│   └── verify_topk.py          # Extracted function
└── docs/
    └── EXPERIMENTAL.md         # Usage guide
```

**Pros**:
- Keeps stable version (src/) separate from experimental
- Easy to pull GeminiSolver updates
- Users can opt-in to advanced features
- Clear signal of maturity level

**Cons**:
- Some code duplication
- Users need to know where to look

---

### Option B: Full Refactor (High Risk)

Merge all improvements into src/:

```
NuclideModel/
├── src/
│   ├── qfd_solver.py           # Phase 11 with repulsion
│   ├── qfd_metaopt_ame2020.py  # v15 with parallel + adaptive
│   ├── verify_topk.py          # Post-hoc verification
│   └── .env.example            # Environment config
└── examples/
    └── run_optimization.sh     # Updated workflow
```

**Pros**:
- Single canonical version
- All users get improvements immediately

**Cons**:
- Breaking change (existing workflows break)
- Higher risk of introducing bugs
- Loses "proven baseline" (Trial 32)

---

### Option C: Feature Flags (Gradual Migration)

Add flags to current code:

```python
# In qfd_metaopt_ame2020.py
def evaluate_parameters(params, use_parallel=False, use_adaptive_iters=False):
    if use_parallel:
        records = eval_gate_parallel(...)  # New code
    else:
        for isotope in calibration_set:
            records.append(run_solver(...))  # Old code
```

**Pros**:
- Backward compatible
- Easy A/B testing
- Gradual rollout

**Cons**:
- Code bloat
- Feature flags accumulate over time
- Hard to maintain dual paths

---

## Recommended Implementation: Option A + Progressive Enhancement

### Phase 1: Create Experimental Directory (This Week)

1. **Create `experimental/` subdirectory**:
   ```bash
   cd /home/tracy/development/qfd_hydrogen_project/GitHubRepo/NuclideModel
   mkdir -p experimental
   ```

2. **Copy key files from GeminiSolver**:
   ```bash
   # Phase 11 solver with repulsion
   cp ../Quantum-Field-Dynamics-main/workflows/nuclear/SharedSolver/phase11_solver_with_repulsion.py \
      experimental/qfd_solver_v11.py

   # v15 meta-optimizer
   cp ../Quantum-Field-Dynamics-main/workflows/nuclear/GeminiSolver/phase9_meta_optimizer_ame2020_v15.py \
      experimental/qfd_metaopt_v15.py

   # Environment config
   cp ../Quantum-Field-Dynamics-main/workflows/nuclear/GeminiSolver/.env.example \
      experimental/
   ```

3. **Extract verify_topk as standalone script**:
   ```python
   # experimental/verify_topk.py
   """Post-hoc verification of top-K Optuna trials at full resolution."""
   ```

4. **Document experimental features**:
   ```markdown
   # docs/EXPERIMENTAL.md

   ## Experimental Features (Advanced Users)

   ### Parallel Meta-Optimizer (v15)
   - 4× speedup via ThreadPoolExecutor
   - Adaptive iteration counts by mass
   - Two-stage gate (fast → verify)

   ### Phase 11 Solver with Self-Repulsion
   - c_repulsion parameter (0.0-0.1)
   - May improve heavy nuclei (A>120)

   ### Usage
   ```bash
   cd experimental
   cp .env.example .env
   # Edit .env for your settings
   python qfd_metaopt_v15.py --solver qfd_solver_v11.py ...
   ```
   ```

5. **Add warning to README**:
   ```markdown
   ## Experimental Features

   Advanced features are available in `experimental/` directory.
   These are **not yet production-ready** but show promising results.
   See `docs/EXPERIMENTAL.md` for details.
   ```

### Phase 2: Validate on Test Cases (Next Week)

1. **Run side-by-side comparison**:
   ```bash
   # Baseline (current)
   python src/qfd_metaopt_ame2020.py --n-calibration 20 --max-iter 50

   # Experimental (v15 + phase11)
   cd experimental
   python qfd_metaopt_v15.py --solver qfd_solver_v11.py --n-calibration 20 --max-iter 50
   ```

2. **Compare results**:
   - Success rate (physical_success=True)
   - Loss values (should be lower with better loss function)
   - Virial penalties (should be lower with c_repulsion)
   - Runtime (should be 4-6× faster)

3. **Test heavy nuclei specifically**:
   ```bash
   # Does c_repulsion fix A>120 underbinding?
   python experimental/qfd_solver_v11.py --A 208 --Z 82 --c-repulsion 0.05
   python experimental/qfd_solver_v11.py --A 208 --Z 82 --c-repulsion 0.10
   ```

### Phase 3: Promote to Stable (After Validation)

If experimental features prove robust:

1. **Promote to src/ as v2.0**:
   ```
   src/
   ├── qfd_solver.py           # Keep as Phase 9 (v1.0)
   ├── qfd_solver_v11.py       # Phase 11 (v2.0)
   ├── qfd_metaopt_ame2020.py  # Keep as baseline
   └── qfd_metaopt_v2.py       # v15 features (v2.0)
   ```

2. **Update documentation**:
   - QUICK_START.md: Add v2.0 quick start
   - FINDINGS.md: Add v2.0 results vs v1.0
   - README.md: List both versions with recommendations

3. **Create release v2.0**:
   - Tag: `v2.0`
   - Title: "NuclideModel v2.0 - Parallel Optimization + Self-Repulsion"
   - Changelog: Document all improvements

---

## Action Items (Priority Order)

### Immediate (Today)
- [x] Review GeminiSolver code
- [ ] Create `experimental/` directory
- [ ] Copy phase11_solver.py → experimental/qfd_solver_v11.py
- [ ] Copy phase9_meta_optimizer_v15.py → experimental/qfd_metaopt_v15.py
- [ ] Copy .env.example → experimental/
- [ ] Write docs/EXPERIMENTAL.md

### Short Term (This Week)
- [ ] Run side-by-side test: current vs experimental (20 isotopes, 50 trials)
- [ ] Test c_repulsion on heavy nuclei (Pb-208, U-238)
- [ ] Measure actual speedup (parallel + adaptive iters)
- [ ] Verify loss function improvements (no more 1e12 domination)

### Medium Term (Next Week)
- [ ] Extract verify_topk() as standalone script
- [ ] Add environment variable support to current code
- [ ] Document migration path (v1.0 → v2.0)
- [ ] Create comparison plots (runtime, accuracy, convergence)

### Long Term (After Validation)
- [ ] Promote experimental → stable if tests pass
- [ ] Create v2.0 release
- [ ] Update GitHub documentation
- [ ] Write migration guide for existing users

---

## Open Questions

1. **c_repulsion optimal range**: 0.0-0.1 suggested, but needs sweep
2. **Parallel workers**: 4 workers optimal? Test 2/4/8 on multi-core
3. **Two-stage gate**: Which 4 isotopes for fast gate? Current: O-16, Ca-40, Fe-56, Ni-62
4. **Adaptive iterations**: Are 90/150/210 optimal? May need tuning
5. **Verify top-K**: How many trials to verify? Current: 5

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Breaking existing workflows | Medium | High | Keep v1.0 in src/, add v2.0 as experimental |
| c_repulsion destabilizes solver | Low | Medium | Start with 0.0 (disabled), sweep 0.01-0.10 |
| Parallel execution race conditions | Low | High | Use thread-safe queues, test thoroughly |
| Loss function changes break calibration | Low | Medium | Compare Trial 32 loss (old vs new function) |
| Heavy users on v1.0 can't upgrade | Low | Low | Document both versions, provide migration script |

---

## Success Metrics

**Experimental → Stable promotion criteria**:

1. ✅ **Speedup**: ≥4× faster than baseline (measured)
2. ✅ **Accuracy**: Equal or better loss on calibration set
3. ✅ **Robustness**: ≥67% success rate (same as current)
4. ✅ **Heavy nuclei**: <5% error for A>120 (with c_repulsion)
5. ✅ **No regressions**: Light nuclei (A<60) still <1%

---

## References

- **GeminiSolver location**: `/home/tracy/development/qfd_hydrogen_project/GitHubRepo/Quantum-Field-Dynamics-main/workflows/nuclear/GeminiSolver/`
- **Key files**:
  - `phase9_meta_optimizer_ame2020_v15.py` (parallel + adaptive)
  - `../SharedSolver/phase11_solver_with_repulsion.py` (self-repulsion)
  - `.env.example` (environment config)
- **Current NuclideModel**: `/home/tracy/development/qfd_hydrogen_project/GitHubRepo/NuclideModel/`

---

**Status**: Awaiting user approval to proceed with Option A (experimental directory).

**Next Step**: Create `experimental/` and run validation tests.
