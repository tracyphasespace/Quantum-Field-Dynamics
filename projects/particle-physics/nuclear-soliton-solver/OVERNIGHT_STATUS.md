# Overnight Calibration Status

**Start time**: 2025-12-30 05:35:09 PST
**Expected completion**: 2025-12-30 13:35:09 PST (8 hours)
**Workers**: 3 parallel workers
**Total jobs**: 32

---

## All Critical Fixes Applied ✅

### 1. Timeout Retry Slicing (Bug #1)
- **File**: `src/qfd_metaopt_ame2020.py:225`
- **Fix**: `cmd[:-5]` removes all 5 trailing arguments
- **Impact**: Timeout retries will now work correctly

### 2. Field-Dependent Symmetry Energy (Bug #2, #5 - 3 iterations)
- **Files**: `src/qfd_solver.py:247-281, 305-316`
- **Implementation**:
  ```python
  def symmetry_energy(self):
      rho_charge = self.psi_e * self.psi_e
      rho_mass = self.psi_N * self.psi_N
      delta = rho_charge - rho_mass
      E_sym = self.c_sym * (delta * delta).sum() * self.dV
      return E_sym
  ```
- **Integration**: V_sym included in energies() return dict (line 316)
- **Impact**:
  - ∂E_sym/∂ψ_e ≠ 0 and ∂E_sym/∂ψ_N ≠ 0
  - SCF solver minimizes charge-mass spatial separation
  - Gradients backpropagate through loss function

### 3. Deep Copy Job Configs (Bug #3)
- **File**: `src/batch_optimize.py:154`
- **Fix**: `copy.deepcopy(self.base_config)`
- **Impact**: 32 independent job configurations (verified)

### 4. Population Size Lower Bound (Bug #4)
- **File**: `src/batch_optimize.py:175`
- **Fix**: `max(8, 15 - (job_id // 4))`
- **Range**: popsize = 15 (job 0) → 8 (job 28+)
- **Impact**: All jobs have sufficient DE population

---

## Monitoring Commands

### Real-time progress
```bash
python3 monitor_batch.py results/batch_overnight --watch
```

### Log tail
```bash
tail -f overnight_run.log
```

### Check completion status
```bash
ls -lth results/batch_overnight/*.json | grep -v config
```

### Worker processes
```bash
pgrep -f "batch_optimize" -a
```

### Results summary (after jobs complete)
```bash
python3 analyze_batch_results.py results/batch_overnight
```

---

## Expected Timeline

- **05:35 - 05:50**: Job initialization, SCF warmup
- **05:50 - 06:30**: First jobs complete
- **06:30 - 12:30**: Main optimization (bulk of jobs)
- **12:30 - 13:35**: Final jobs, result aggregation
- **13:35**: Automatic shutdown at 8-hour deadline

**First checkpoint**: Expected around 06:00 (after 5 job completions)

---

## What to Check When Jobs Complete

### 1. Symmetry Energy Contribution
```bash
# Extract V_sym values from results
jq '.energies.V_sym' results/batch_overnight/job_00_result.json
```

**Expected**: Non-zero values that change between isotopes
**Bad sign**: All zeros (means c_sym=0 or E_sym not computed)

### 2. Convergence Quality
```bash
# Check virial constraint satisfaction
jq '.convergence.virial' results/batch_overnight/job_*_result.json | head -20
```

**Target**: |virial| < 0.18 for valid SCF solutions
**Bad sign**: Many jobs with |virial| > 0.5 (poor convergence)

### 3. Heavy Isotope Errors
```bash
# Check if we reduced -12% underbinding
python3 -c "
import json, glob
for f in sorted(glob.glob('results/batch_overnight/job_*_result.json')):
    with open(f) as fp:
        result = json.load(fp)
        errors = result.get('isotope_errors', [])
        if errors:
            mean_err = sum(e['rel_error'] for e in errors) / len(errors)
            print(f'{f.split(\"/\")[-1][:8]}: mean error = {mean_err:.1%}')
"
```

**Goal**: Mean error < -3% (improved from -12%)
**Success metric**: At least 5 jobs achieve < -5% error

### 4. Parameter Diversity
```bash
# Check best parameters found
jq '{c_v2: .best_params.c_v2_base, c_v4: .best_params.c_v4_base, c_sym: .best_params.c_sym}' \
   results/batch_overnight/job_*_result.json | head -30
```

**Expected**: Wide range of (c_v2, c_v4, c_sym) values
**Bad sign**: All jobs converged to identical parameters (optimizer stuck)

---

## Success Criteria

### Minimum Success
- [ ] At least 20/32 jobs complete without crashes
- [ ] Mean heavy isotope error < -8% (better than Trial 32's -8.4%)
- [ ] Best job achieves < -5% error on its isotope subset
- [ ] V_sym contributes non-zero gradients (verified in logs)

### Target Success
- [ ] 28+ jobs complete successfully
- [ ] Mean error < -5%
- [ ] Best job achieves < -3% error
- [ ] Multiple distinct parameter regions identified

### Exceptional Success
- [ ] All 32 jobs complete
- [ ] Mean error < -3%
- [ ] Best parameters generalize to full heavy isotope set
- [ ] Clear parameter scaling with A identified

---

## Failure Modes to Watch For

### 1. All Jobs Timeout
**Symptom**: No results after 2 hours
**Possible cause**: SCF not converging, grid too fine
**Action**: Check overnight_run.log for error messages

### 2. Identical Results Despite Different Seeds
**Symptom**: All jobs report same best parameters
**Possible cause**: Deep copy bug not actually fixed
**Action**: Verify job configs have different seeds:
```bash
jq '.metadata.random_seed' results/batch_overnight/job_*_config.json | sort -u | wc -l
# Should be 32
```

### 3. V_sym Always Zero
**Symptom**: All results show V_sym = 0.0
**Possible cause**: c_sym not in parameter search bounds
**Action**: Check RunSpec bounds include c_sym

### 4. Memory Exhaustion
**Symptom**: Workers crash with "Killed" (OOM)
**Possible cause**: Grid too large, insufficient RAM
**Action**: Monitor with `free -h` and reduce workers if needed

---

## Next Steps After Completion

### 1. Aggregate Results
```bash
python3 src/aggregate_results.py results/batch_overnight \
    --output results/overnight_best_params.json
```

### 2. Validate on Full Dataset
```bash
python3 src/qfd_metaopt_ame2020.py \
    --params results/overnight_best_params.json \
    --A-min 120 --A-max 250 \
    --emit-json > results/validation_full.json
```

### 3. Compare to Trial 32 Baseline
```bash
python3 compare_calibrations.py \
    --baseline trial32_heavy_region.json \
    --optimized results/validation_full.json
```

### 4. Update RunSpec with Best Parameters
Edit `experiments/nuclear_heavy_region.runspec.json`:
- Set `c_v2_base`, `c_v4_base`, `c_sym` to optimized values
- Mark as "Phase 10" calibration
- Document improvement over Trial 32

---

## Restart History

1. **05:08:44** - Initial launch (5 bugs present)
2. **05:14:09** - Restart after fixing bugs #1-4 (V_sym still not in energies)
3. **05:35:09** - Final restart with all fixes (current run)

**Total bugs fixed before final launch**: 5 critical bugs across 3 code reviews

---

## Notes

- Jobs use differential evolution (scipy.optimize.differential_evolution)
- Each job optimizes on 8 randomly selected heavy isotopes
- Population sizes vary from 15 (early jobs) to 8 (later jobs)
- All jobs use same RunSpec bounds but different seeds
- Checkpoints saved every 5 completions for crash recovery

**Documentation**: See `BUGS_FIXED_FINAL.md` for complete bug fix history.
