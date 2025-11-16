# V15_CLEAN Pipeline Status
**Date**: 2025-11-12 08:02
**Status**: Code fully consolidated with tools, ready for clean debugging

## Current Situation

### ✅ Code Consolidation COMPLETE
All pipeline code and tools are now in `v15_clean/` with proper imports:
- **Stage 1**: v15_clean/stages/stage1_optimize.py
- **Stage 2**: v15_clean/stages/stage2_mcmc_numpyro.py (36KB, WITH SIGN FIX)
- **Stage 3**: v15_clean/stages/stage3_hubble_optimized.py (15KB, WITH ZERO-POINT CALIBRATION)
- **Model**: v15_clean/core/v15_model.py
- **Data**: v15_clean/core/v15_data.py
- **Tools**: 7 analysis/visualization tools in v15_clean/tools/ (see tools/README.md)

### ⚠️ Results Directories (CONTAMINATED)

**Stage 1** (clean):
- `results/v15_clean/stage1_fullscale/` (4,727 SNe, Nov 11 16:59) ✓

**Stage 2** (multiple versions, CONFUSION):
- `results/v15_clean/stage2_production/` (Nov 11 20:41) ❌ OLD, NO SIGN FIX, k_J=5.01
- `results/v15_clean/stage2_production_corrected/` (Nov 12 05:32) ⚠️ HAS sign fix code but STILL k_J=5.01!
- `results/v15_clean/stage2_signfix_test/` (Nov 12 01:17) ✓ 50 SNe test, k_J=7.73

**Stage 3** (multiple versions):
- `results/v15_clean/stage3_hubble_final/` (Nov 12 05:38) - used stage2_production_corrected
- `results/v15_clean/stage3_hubble_calibrated/` (Nov 12 00:30) - used OLD stage2_production ❌

## The Mystery: Why Did Production Still Fail?

**Test (50 SNe)**: k_J = 7.73 ✓ (54% improvement with sign fix)
**Production (4,727 SNe)**: k_J = 5.01 ❌ (574 divergences, hitting lower bound)

**Most Likely Explanation**: "Battle of Wills"
- With 4,727 SNe, likelihood >>> prior
- Data strongly prefers k_J < 5.0
- Sampler hits truncation bound at k_J=5.0
- 574 divergent transitions = sampler can't explore properly

**Alternative Explanation**: Systematic bias in Stage 1 ln_A values
- If ln_A values have redshift-dependent bias
- Stage 2 tries to compensate by driving k_J low
- Would explain why full dataset behaves differently than 50 SNe subset

## Next Steps

### Option 1: Diagnostic Run (RECOMMENDED)
Run Stage 2 with `--constrain-signs off` to see what data truly wants:
```bash
./v15_clean/scripts/run_diagnostic_unconstrained.sh
```

Expected: k_J either very small (<2) or negative → indicates model/data mismatch

### Option 2: Check Stage 1 for Systematic Bias
Analyze ln_A vs redshift for systematic trends:
```python
# Check if ln_A has redshift-dependent offset
plot(z, ln_A - expected_from_cosmology)
```

### Option 3: Widen Priors
Change truncation from k_J > 5.0 to k_J > 2.0 to allow sampler more room

## File Inventory

### Code (v15_clean/)
```
v15_clean/
├── core/
│   ├── v15_model.py (29KB, Nov 11 15:02) ✓
│   ├── v15_data.py (7.6KB, Nov 11 15:08) ✓
│   └── pipeline_io.py (4.9KB, Nov 11 15:09) ✓
├── stages/
│   ├── stage1_optimize.py (24KB, Nov 11 15:54) ✓
│   ├── stage2_mcmc_numpyro.py (36KB, Nov 12 01:11) ✓ SIGN FIX
│   └── stage3_hubble_optimized.py (15KB, Nov 12 07:46) ✓ ZERO-POINT CAL
└── scripts/
    └── run_full_pipeline.sh (NEW, consolidated script)
```

### Results
```
results/v15_clean/
├── stage1_fullscale/ (4,727 SNe) ✓
├── stage2_production/ (OLD, k_J=5.01) ❌
├── stage2_production_corrected/ (NEW, but k_J=5.01, 574 diverg) ⚠️
├── stage2_signfix_test/ (50 SNe, k_J=7.73) ✓
├── stage3_hubble_final/ (used production_corrected)
└── [14 other test directories]
```

### OLD Code (DO NOT USE)
```
src/ (outdated, Nov 11)
2Compare/ (outdated, Nov 10)
```

## Action Items

1. **STOP all background runs** - they may be using old code/results
2. **Clean results** - move old results to archive
3. **Run diagnostic** - unconstrained priors to understand data
4. **Investigate ln_A bias** - check for systematic errors in Stage 1
5. **Document findings** - update this file with conclusions

## Command Reference

```bash
# Check what's running
ps aux | grep python | grep stage

# Kill old processes
pkill -f "python.*stage"

# Run fresh diagnostic
./v15_clean/scripts/run_full_pipeline.sh
```
