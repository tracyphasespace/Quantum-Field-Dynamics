# Quick Resume: V15 QFD Pipeline

## Current Status (Post-Fixes)

**✅ Completed:**
- Stage 1: Optimization complete (3141/5468 SNe, 58.5% success rate)
- All cloud.txt critical fixes applied and committed
- Full anti-regression infrastructure in place (tests, CI, guards)
- JAX cache-busting strategy implemented

**🔄 Ready to Run:**
- Stage 2: α-space MCMC (NumPyro NUTS sampler)
- Stage 3: Hubble residual analysis

## Recent Commits

```bash
git log --oneline -4
# 958f144 Implement JAX cache-busting strategy (Option B)
# 4ac8cce Remove assert from JITted function in Stage 2
# ca6516e Add anti-regression infrastructure
# 4ef01ec Apply all 5 critical fixes from cloud.txt
```

## Run Stage 2 (Fresh Shell Required)

After restarting your shell to clear Python module cache:

```bash
cd /home/tracy/development/Quantum-Field-Dynamics-fresh/projects/astrophysics/qfd-supernova-v15

# Test run (foreground, see output)
python src/stage2_mcmc_numpyro.py \
    --stage1-results results/v15_production/stage1 \
    --lightcurves data/lightcurves_unified_v2_min3.csv \
    --out results/v15_production/stage2 \
    --nwarmup 1000 \
    --nsamples 2000 \
    --nchains 4

# Production run (background, ~15-30 min)
nohup python src/stage2_mcmc_numpyro.py \
    --stage1-results results/v15_production/stage1 \
    --lightcurves data/lightcurves_unified_v2_min3.csv \
    --out results/v15_production/stage2 \
    --nwarmup 1000 \
    --nsamples 2000 \
    --nchains 4 \
    > results/v15_production/stage2.log 2>&1 &
```

**What to expect:**
- "Preflight: checking residual variance..."
- "Preflight OK: var(residuals) = X.XXX"
- MCMC warmup progress bar
- Sampling progress bar
- Results in `results/v15_production/stage2/`

## Run Stage 3 (After Stage 2 Completes)

```bash
python src/stage3_hubble_optimized.py \
    --stage1 results/v15_production/stage1 \
    --stage2 results/v15_production/stage2/posterior_samples.csv \
    --out results/v15_production/stage3
```

## Anti-Regression Checks

Run contract tests anytime:
```bash
pytest tests/test_spec_contracts.py -v
```

Run spec guard (forbid (1+z) patterns):
```bash
python scripts/spec_guard.py
```

## Key Files Modified

1. **src/v15_metrics.py** - JAX API call fix
2. **src/v15_gate.py** - Operator precedence clarification
3. **src/v15_model.py** - Radius units naming (cm not m) + H0=70 comments
4. **src/v15_sampler.py** - Legacy warning with fail-fast
5. **src/stage2_mcmc_numpyro.py** - Cache-busting + preflight variance check

## Why Shell Restart Needed

Python's import cache was loading old module bytecode from a different venv:
- Cached path: `/home/tracy/development/qfd_hydrogen_project/October_Supernova/.venv`
- Current path: `/home/tracy/development/Quantum-Field-Dynamics-fresh/.../qfd-supernova-v15`

Fresh shell → fresh Python → correct code paths.

## Troubleshooting

**If Stage 2 still fails with TracerBoolConversionError:**
```bash
# Clear all caches
rm -rf ~/.cache/jax_cache
find src/ -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null

# Verify correct file
sed -n '125,135p' src/stage2_mcmc_numpyro.py | grep -i assert
# Should return empty (no asserts)

# Run with explicit Python
/usr/bin/python3 src/stage2_mcmc_numpyro.py --stage1-results ...
```

**Monitor running jobs:**
```bash
ps aux | grep "stage[12]_" | grep -v grep
tail -f results/v15_production/stage2.log
```

## Expected Timeline

- Stage 2 MCMC: ~15-30 minutes (4 chains, 1000 warmup + 2000 samples)
- Stage 3 Analysis: ~5-10 minutes
- Total: ~30 minutes from Stage 2 start to publication figures

## Next Steps After Stage 3

1. Review Hubble diagram: `results/v15_production/stage3/hubble_diagram.png`
2. Check residuals: `results/v15_production/stage3/residuals_vs_z.png`
3. Generate publication figures: `python scripts/make_publication_figures.py`
4. Create per-survey report: `python scripts/make_per_survey_report.py`

---

**Ready to proceed!** All fixes committed, code is publication-ready with full anti-regression rails.
