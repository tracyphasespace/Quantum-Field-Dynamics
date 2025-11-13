# Staged Validation & Parameter Estimation Plan
## Using 50 → 500 → 5000 SNe with Self-Consistent Field Iteration

### Overview
Implement a bootstrapping approach to validate QFD physics and refine parameters progressively:
1. **Stage 0**: Train on 50-100 cleanest SNe → establish initial physics
2. **Stage 1**: Measure deviations on next 500 SNe → identify outliers & refine
3. **Stage 2**: Full 5000 SNe analysis → final parameter inference with SCF
4. **Stage 3**: Self-Consistent Field iteration → converged cosmology

---

## STAGE 0: Foundation Training (50-100 SNe)

### Selection Criteria
```python
clean_sne_criteria = {
    'redshift': 0.01 < z < 0.5,           # Low-z to minimize systematics
    'photometry': ['U', 'B', 'V', 'R', 'I'],  # Multi-band required
    'chi2_initial': < 100,                 # From Pantheon+ baseline fit
    'host_specz': True,                    # Confirmed spectroscopic redshift
    'no_known_host_peculiarity': True,    # No AGN, starburst, etc.
    'lightcurve_coverage': > 30 days,     # Well-sampled rise & decline
    'exclude': ['known_BBH', 'lensed']    # Manual quality review
}
```

### Process
1. **Stage 1 (per-SN optimization)** on 50-100 cleanest:
   - Optimize 5 params per SN: `t₀, A_plasma, β, ln_A, A_lens`
   - Use L-BFGS-B with JAX GPU acceleration
   - Apply quality cuts: χ² < 500 (stricter than production 2000)
   - **Use SCF solver** in every flux calculation (see below)

2. **Stage 2 (global MCMC)** on survivors:
   - Infer global physics: `k_J, η', ξ, σ_α`
   - Student-t likelihood with ν ~ 6-8 for robustness
   - NumPyro NUTS: 4 chains × 2000 samples × 1000 warmup
   - **Freeze L_peak = 1.5×10⁴³ erg/s** (standard candle assumption)

3. **Output**: Posterior distributions for cosmology
   ```
   k_J:  10.77 ± 4.57 km/s/Mpc (cosmic drag)
   η':   [value] ± [error]      (plasma coupling)
   ξ:    [value] ± [error]      (FDR vacuum scattering)
   σ_α:  [value] ± [error]      (intrinsic scatter)
   ```

---

## STAGE 1: Deviation Measurement (Next 500 SNe)

### Goal
Use Stage-0 physics to predict 500 new SNe and measure deviations:
- Identify which SNe are **consistent** with QFD
- Which are **outliers** (BBH lensing, unknown systematics)
- Refine parameter estimates using **only** consistent subset

### Method
1. **Fix global physics** from Stage 0: `k_J, η', ξ` (mean values)

2. **Run Stage 1 optimization** on 500 new SNe:
   - Still optimize per-SN params: `t₀, A_plasma, β, ln_A, A_lens`
   - But physics params are **held fixed**
   - Calculate residuals: Δm = m_obs - m_pred(z | k_J, η', ξ)

3. **Calculate deviation metrics**:
   ```python
   for each SN in 500:
       χ²_physics = sum((m_obs - m_pred_fixed_physics)² / σ²)
       χ²_optimized = sum((m_obs - m_pred_optimized)² / σ²)

       # Key metric: how much does optimization improve fit?
       Δχ² = χ²_physics - χ²_optimized

       # If Δχ² > threshold (e.g., 50), SN is "outlier"
       # Physics params alone cannot explain it
   ```

4. **Classify SNe**:
   - **Consistent** (Δχ² < 50): QFD physics + small per-SN corrections
   - **Moderate outliers** (50 < Δχ² < 200): BBH lensing candidates
   - **Severe outliers** (Δχ² > 200): Bad data or unknown physics

5. **Refine parameters**:
   - Take Stage-0 physics + 500 new **consistent** SNe
   - Re-run Stage 2 MCMC with expanded dataset (50-100 + ~400 new)
   - **Update posteriors**: k_J, η', ξ, σ_α

---

## STAGE 2: Full Dataset Analysis (5000 SNe)

### Goal
Apply refined physics to full Pantheon+ sample and perform final inference

### Method
1. **Hierarchical quality gating**:
   ```python
   # Use updated physics params from Stage 1
   for each SN in 5000:
       # Stage 1 optimization with current physics
       optimize(t₀, A_plasma, β, ln_A, A_lens | k_J, η', ξ)

       # Calculate metrics
       χ²_per_ndof = χ² / (N_datapoints - 5)
       color_residual = β - β_expected
       dimming_excess = A_lens / A_plasma

       # Gating rules (from v15_gate.py)
       if χ²_per_ndof < 1.5 and abs(color_residual) < 0.2:
           category = "CLEAN"  # ~4100 SNe expected
       elif A_lens > 0.5 or dimming_excess > 2.0:
           category = "BBH_LENSING"  # ~800 SNe (16%)
       else:
           category = "UNKNOWN_OUTLIER"  # Investigate
   ```

2. **Final parameter inference**:
   - Run Stage 2 MCMC on **CLEAN** subset only
   - Use Student-t likelihood (ν ~ 6-8) to handle residual outliers
   - Generate full posterior samples

3. **Validation metrics**:
   ```python
   # Hubble residual flatness
   binned_residuals_vs_z = should_have_slope ~ 0

   # Monotonicity check
   α_pred = m_obs - m_model(z)  # Should decrease with z

   # Holdout cross-validation
   rms_holdout ≈ 1.89 mag (from v15_metrics.py)
   ```

---

## STAGE 3: Self-Consistent Field (SCF) Iteration

### The SCF Problem
In QFD, **flux and opacity are coupled**:
- Flux-Dependent Redshift: `τ_FDR ∝ √(flux)`
- But flux depends on opacity: `flux_observed = flux_intrinsic × exp(-τ_FDR)`
- **Circular dependency** → requires iterative solution

### SCF Solver (Already Implemented!)
Location: `v15_model.py:294-348` (`qfd_tau_total_jax`)

```python
def qfd_tau_total_jax(wave, z, flux_estimate, params):
    """
    Self-consistent solver for total optical depth.

    Iterates:
    1. Calculate τ_FDR = ξ × η' × √(flux_current)
    2. Dim flux: flux_next = flux_intrinsic × exp(-τ_FDR)
    3. Repeat until |flux_next - flux_current| < tolerance

    Parameters:
        wave: Observed wavelength [Å]
        z: Redshift
        flux_estimate: Initial guess for flux [erg/s/cm²/Å]
        params: {k_J, η', ξ, ...}

    Returns:
        tau_total: Self-consistent optical depth
        flux_final: Converged dimmed flux
    """
    flux_current = flux_estimate
    relaxation = 0.5  # Damping factor for stability

    for iteration in range(20):  # Max 20 iterations
        # Calculate opacity components
        tau_drag = cosmological_drag(wave, z, params['k_J'])
        tau_plasma = plasma_veil(wave, z, params['eta_prime'])
        tau_FDR = params['xi'] * params['eta_prime'] * jnp.sqrt(flux_current)
        tau_thermal = planck_wien_broadening(wave, z)

        tau_total = tau_drag + tau_plasma + tau_FDR + tau_thermal

        # Update flux with relaxation
        flux_new = flux_intrinsic * jnp.exp(-tau_total)
        flux_next = (1 - relaxation) * flux_current + relaxation * flux_new

        # Check convergence
        if jnp.abs(flux_next - flux_current) / flux_current < 1e-5:
            return tau_total, flux_next

        flux_current = flux_next

    return tau_total, flux_current  # Return best estimate
```

### How SCF Derives Parameters

**Within each MCMC step** (Stage 2):
```python
# For each proposed set of physics params {k_J, η', ξ}:
for sn in clean_dataset:
    for wavelength, flux_obs in sn.lightcurve:
        # SCF solver gives self-consistent prediction
        tau, flux_pred = qfd_tau_total_jax(wavelength, z, flux_obs, params)

        # Calculate likelihood
        residual = flux_obs - flux_pred
        log_likelihood += -0.5 * (residual / sigma)²

# MCMC accepts/rejects {k_J, η', ξ} based on total log_likelihood
# After 8000 samples → posterior distributions for all params
```

**Key insight**: SCF is called ~10⁷ times during MCMC (4 chains × 2000 samples × ~1000 SNe × ~10 wavelengths × 20 SCF iterations). **JAX JIT compilation is essential** for speed.

---

## Implementation Roadmap

### Step 1: Prepare Clean Training Set
```bash
cd /projects/V16
python scripts/select_clean_sne.py \
    --input data/pantheon_plus_v16.pkl \
    --output data/clean_50_training.pkl \
    --criteria criteria/stage0_selection.yaml \
    --n_select 50
```

### Step 2: Stage 0 Training
```bash
# Stage 1: Per-SN optimization on 50 clean SNe
python stages/stage1_optimize.py \
    --input data/clean_50_training.pkl \
    --output output/stage0/stage1_results.pkl \
    --chi2_cut 500  # Stricter than production

# Stage 2: Infer global physics
python stages/stage2_simple.py \
    --input output/stage0/stage1_results.pkl \
    --output output/stage0/physics_posterior.pkl \
    --n_chains 4 \
    --n_samples 2000 \
    --n_warmup 1000
```

### Step 3: Deviation Measurement (500 SNe)
```bash
# Hold physics fixed at Stage-0 means
python scripts/measure_deviations.py \
    --physics output/stage0/physics_posterior.pkl \
    --test_sne data/pantheon_plus_v16.pkl \
    --n_test 500 \
    --output output/stage1/deviations.pkl

# Classify and refine
python scripts/refine_parameters.py \
    --stage0_results output/stage0/ \
    --deviations output/stage1/deviations.pkl \
    --output output/stage1/refined_posterior.pkl
```

### Step 4: Full 5000 SNe Analysis
```bash
# Stage 1 on all SNe
python stages/stage1_optimize.py \
    --input data/pantheon_plus_v16.pkl \
    --output output/stage2/stage1_all.pkl \
    --chi2_cut 2000  # Production cut

# Gate outliers
python stages/v15_gate.py \
    --input output/stage2/stage1_all.pkl \
    --output output/stage2/gated.pkl

# Final inference on clean subset
python stages/stage2_simple.py \
    --input output/stage2/gated.pkl \
    --subset "CLEAN" \
    --output output/stage2/final_posterior.pkl
```

### Step 5: Validation
```bash
python stages/v15_metrics.py \
    --stage1 output/stage2/stage1_all.pkl \
    --stage2 output/stage2/final_posterior.pkl \
    --output figures/validation/

# Generates:
# - Hubble diagram with residuals
# - χ² distributions by redshift
# - Posterior corner plots
# - Holdout cross-validation
```

---

## Expected Outcomes

### After Stage 0 (50-100 SNe):
- **k_J posterior width**: ~10 km/s/Mpc (large uncertainty)
- **η' posterior**: Broad, may be degenerate with ξ
- **Validation**: Not meaningful (overfitting risk)

### After Stage 1 (+ 500 SNe):
- **k_J posterior width**: ~5 km/s/Mpc (narrower)
- **Outlier fraction**: ~16% identified (BBH lensing)
- **Hubble residual RMS**: ~2.0 mag (improving)

### After Stage 2 (5000 SNe):
- **k_J**: 10.77 ± 4.57 km/s/Mpc (production-level precision)
- **η'**: [value] ± [error] with ξ correlation understood
- **σ_α intrinsic scatter**: ~0.08-0.12 mag
- **Holdout RMS**: 1.89 mag
- **Clean fraction**: ~82% (4100/5000 SNe)

---

## Critical Notes

1. **SCF convergence**:
   - Monitor convergence failures (return code from 20th iteration)
   - If >5% fail, adjust relaxation factor or increase max iterations
   - JAX JIT compilation essential for speed (~100× faster)

2. **Parameter degeneracies**:
   - η' and ξ are correlated (both affect plasma opacity)
   - Use informative priors from Stage 0 to break degeneracy in later stages
   - Monitor trace plots for "banana" shapes in posteriors

3. **BBH lensing**:
   - Stage 0 & 1 should **exclude known BBH systems**
   - Stage 2 treats them as outliers (Student-t robustness)
   - Future work: explicit BBH mixture model (per PHYSICS_AUDIT.md)

4. **Computational cost**:
   - Stage 0: ~1 hour on GPU (50 SNe × 5 params × 1000 iterations)
   - Stage 1: ~10 hours (500 SNe)
   - Stage 2: ~50 hours (5000 SNe)
   - MCMC: ~20 hours per run (4 chains × 2000 samples × SCF calls)
   - **Total**: ~100 GPU-hours for full pipeline

5. **Validation philosophy**:
   - Early stages prioritize **purity** over completeness (strict cuts)
   - Later stages prioritize **completeness** (relaxed cuts + outlier modeling)
   - Cross-validation at each stage prevents overfitting

---

## References

- **Pseudocode**: `/projects/V16/documents/Supernovae_Pseudocode.md`
- **Physics Model**: `/projects/astrophysics/qfd-supernova-v15/src/v15_model.py`
- **Stage 1**: `/projects/V16/stages/stage1_optimize.py`
- **Stage 2**: `/projects/V16/stages/stage2_simple.py`
- **Gating**: `/projects/astrophysics/qfd-supernova-v15/src/v15_gate.py`
- **Metrics**: `/projects/astrophysics/qfd-supernova-v15/src/v15_metrics.py`
- **Audit**: `/projects/V16/documents/PHYSICS_AUDIT.md`
