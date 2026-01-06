# Overnight Test Plan: Localization Configuration Sweep

## Objective
Systematically explore localization parameter space to find configuration that achieves χ² < 10^6 with positive S_opt for e,μ regression.

## Current Status
- **Problem**: Current config (k=1.5, Δv/Rv=0.5, p=6) yields χ² ~ 10^8 despite corrected sign convention
- **Need**: Test alternative localization parameters to find optimal balance

## Test Configurations (6 total)

### Config 1: k=1.0, Δv/Rv=0.5, p=6
**Rationale**: Stronger localization (k=1.0 vs 1.5)
- **Effect**: More aggressive suppression of far-field circulation
- **Hypothesis**: May reduce profile insensitivity by concentrating energy near core
- **Risk**: Could collapse E_circ too much (saw this in earlier tests)

### Config 2: k=2.0, Δv/Rv=0.5, p=6
**Rationale**: Weaker localization (k=2.0 vs 1.5)
- **Effect**: Less suppression of far-field, more "natural" Hill vortex
- **Hypothesis**: May allow better energy balance while maintaining some localization
- **Risk**: May not improve profile sensitivity enough

### Config 3: k=1.5, Δv/Rv=0.25, p=6
**Rationale**: Narrower falloff region (Δv/Rv = 0.25 vs 0.5)
- **Effect**: Sharper cutoff outside R_v, steeper gradient
- **Hypothesis**: May provide better boundary between structured/far-field
- **Risk**: Sharp transition could cause numerical issues

### Config 4: k=1.5, Δv/Rv=0.75, p=6
**Rationale**: Wider falloff region (Δv/Rv = 0.75 vs 0.5)
- **Effect**: Gentler cutoff, smoother transition
- **Hypothesis**: May avoid pathological gradients while still localizing
- **Risk**: Too gentle may not effectively suppress far field

### Config 5: k=1.5, Δv/Rv=0.5, p=4
**Rationale**: Softer envelope power (p=4 vs p=6)
- **Effect**: Slower exponential decay, more gradual falloff
- **Hypothesis**: May reduce sensitivity to exact R_v while maintaining localization
- **Risk**: May not suppress far field strongly enough

### Config 6: k=1.5, Δv/Rv=0.5, p=8
**Rationale**: Sharper envelope power (p=8 vs p=6)
- **Effect**: Faster exponential decay, more abrupt cutoff
- **Hypothesis**: May create cleaner separation between core and far-field
- **Risk**: Very sharp cutoff could cause numerical instabilities

## Technical Specifications

### Computational Resources
- **Workers**: 6 (estimated 4.5 GB RAM usage vs 7 GB available)
- **maxiter**: 50 (reduced from 100 for faster iteration)
- **β points**: 11 (reduced from 21 for faster sweep)
- **Estimated time per config**: 30-40 minutes
- **Total estimated runtime**: 3-4 hours

### Output
Each configuration saves:
- Full β scan results
- Best-fit parameters (β_min, χ²_min, S_opt)
- Per-lepton parameters (R_c, U, A)
- Per-lepton energies (E_circ, E_stab, E_grad, E_total)
- Diagnostics (F_inner, bound hits)
- Outcome classification (PASS/SOFT_PASS/FAIL)

### Files Generated
```
results/V22/overnight_config1_k1.0_dv0.5_p6.json
results/V22/overnight_config2_k2.0_dv0.5_p6.json
results/V22/overnight_config3_k1.5_dv0.25_p6.json
results/V22/overnight_config4_k1.5_dv0.75_p6.json
results/V22/overnight_config5_k1.5_dv0.5_p4.json
results/V22/overnight_config6_k1.5_dv0.5_p8.json
results/V22/overnight_batch_summary.json
```

## Success Criteria

### Primary Goals
1. **Find config with χ² < 10^6** (orders of magnitude reduction from 10^8)
2. **S_opt > 0** (positive mass scale)
3. **Not all parameters at bounds** (≤1 bound hit per lepton)
4. **β interior to scan range** (not at edges)

### Secondary Diagnostics
- **F_inner** > 50%: Indicates good core sensitivity
- **Energy ordering**: E_μ < E_e (before mass scaling)
- **Bound hits**: None is ideal, 1 per lepton acceptable

## Decision Tree (Morning)

### If ALL configs FAIL (χ² > 10^6):
**Action**: Fundamental physics issue - may need:
- Bulk potential term
- Different density profile (non-Gaussian)
- Constraint on total circulation or moment

### If 1-2 configs PASS:
**Action**:
1. Rerun best config with maxiter=100, β points=21 for high-precision fit
2. Test multi-start stability (5+ seeds)
3. Proceed to τ inclusion if stable

### If 3+ configs PASS:
**Action**:
1. Compare χ²_min, S_opt, F_inner across passing configs
2. Choose config with best profile sensitivity (highest F_inner)
3. Run high-precision fit and proceed to τ

### If ANY config gives χ² < 100:
**Action**: Breakthrough!
1. Immediately analyze what's different about that configuration
2. Test neighborhood of that parameter point (fine grid)
3. Proceed to full 3-lepton validation

## Physics Hypotheses Being Tested

### k parameter (localization strength)
- **k=1.0**: Tests if stronger localization resolves spatial orthogonality
- **k=2.0**: Tests if weaker localization allows better energy balance

### Δv/Rv parameter (falloff width)
- **Δv/Rv=0.25**: Tests if sharp boundary improves structure
- **Δv/Rv=0.75**: Tests if smooth transition avoids artifacts

### p parameter (envelope sharpness)
- **p=4**: Tests if gentler rolloff reduces numerical sensitivity
- **p=8**: Tests if sharper cutoff creates cleaner separation

## Fallback Options (if all fail)

1. **Test different β ranges**: Try [2.8, 3.0] or [3.2, 3.4]
2. **Test different w values**: Try w=0.015 or w=0.025
3. **Test fixed S**: Instead of profiling S, fix S=10 and fit only geometry
4. **Add constraint**: Constrain total angular momentum or energy moment
5. **Alternative functional**: E_total = sqrt(E_circ² + E_stab² + E_grad²) (Euclidean norm)

## Launch Command

```bash
nohup python3 overnight_batch_test.py > results/V22/logs/overnight_batch.log 2>&1 &
```

Then monitor progress:
```bash
tail -f results/V22/logs/overnight_batch.log
```

Or check summary in morning:
```bash
tail -100 results/V22/logs/overnight_batch.log
cat results/V22/overnight_batch_summary.json
```
