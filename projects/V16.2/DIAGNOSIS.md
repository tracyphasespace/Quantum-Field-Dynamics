# k_J Discrepancy Diagnosis

**Created**: 2025-11-13
**Issue**: New code doesn't reproduce k_J ≈ 10.74 ± 4.6 with the same priors

---

## The Back-Transformation Formula

### Current Implementation (stage2_simple.py:260-272)

```python
def back_transform_to_physics(c_samples, scales):
    """
    Back-transform from standardized space to physics space.

    From November 5 results:
      c = k_phys * scales (forward)
      k_phys = c / scales (back-transform)
    """
    k_J_samples = c_samples[:, 0] / scales[0]
    eta_prime_samples = c_samples[:, 1] / scales[1]
    xi_samples = c_samples[:, 2] / scales[2]

    return k_J_samples, eta_prime_samples, xi_samples
```

### The Model (stage2_simple.py:200)

```python
alpha_pred = ln_A0 + jnp.dot(Phi_std, c)
```

Where `Phi_std` are standardized features:
```python
Phi_std = (Phi - means) / scales
```

---

## Mathematical Derivation

### Forward Model

Starting from the physics model:
```
α = ln_A0 + Φ · k_phys
```

Where:
- Φ = [ln(1+z), z, z/(1+z)] - raw features
- k_phys = [k_J, eta', xi] - physical parameters

### With Standardization

Standardize features:
```
Φ_std = (Φ - μ_Φ) / σ_Φ
```

The model becomes:
```
α = ln_A0 + Φ_std · c
  = ln_A0 + [(Φ - μ_Φ) / σ_Φ] · c
  = ln_A0 + Φ · (c / σ_Φ) - μ_Φ · (c / σ_Φ)
  = (ln_A0 - μ_Φ · c / σ_Φ) + Φ · (c / σ_Φ)
```

Comparing with the original model:
```
k_phys = c / σ_Φ
```

**Therefore: The current back-transformation IS CORRECT!**

```python
k_J = c0 / scales[0]
eta' = c1 / scales[1]
xi = c2 / scales[2]
```

---

## Why Doesn't It Work?

### The Golden Reference Values

From RECOVERY.md (published papers):
- k_J ≈ 10.74 ± 4.6 km/s/Mpc
- η' ≈ -7.97 ± 1.4
- ξ ≈ -6.95 ± 3.8

From standardized coefficients:
- c[0] = 1.857
- c[1] = -2.227
- c[2] = -0.766

### The Critical Question

**What were the `scales` values in the golden run?**

Using the back-transformation formula:
```
k_J = c0 / scales[0]
10.74 = 1.857 / scales[0]
scales[0] = 1.857 / 10.74 ≈ 0.173
```

Similarly:
```
eta' = c1 / scales[1]
-7.97 = -2.227 / scales[1]
scales[1] = -2.227 / -7.97 ≈ 0.279
```

And:
```
xi = c2 / scales[2]
-6.95 = -0.766 / scales[2]
scales[2] = -0.766 / -6.95 ≈ 0.110
```

### Expected Golden Run Standardization Statistics

For the transformation to work, the November 5, 2024 golden run must have had:
- **scales[0] ≈ 0.173** (std of ln(1+z))
- **scales[1] ≈ 0.279** (std of z)
- **scales[2] ≈ 0.110** (std of z/(1+z))

---

## Diagnostic Steps

### Step 1: Check Your Current Standardization Statistics

When you run stage2_simple.py, it prints:
```
Standardizing features...
  Scales: [scales[0], scales[1], scales[2]]
```

**ACTION**: Record these values and compare to the expected values above.

### Step 2: Check If Your Data Matches the Golden Run

The standardization statistics depend on:
1. **Which SNe are included** (quality cuts)
2. **Redshift distribution** of those SNe

If you're using:
- Different quality cuts (chi2 threshold)
- Different SNe subset
- Different Stage 1 results

Then your `scales` will be different, and the back-transformation will produce different k_J values!

### Step 3: Verify the Priors Are Being Used Correctly

The informed priors set:
```python
c0 ~ Normal(1.857, 0.5)
c1 ~ Normal(-2.227, 0.5)
c2 ~ Normal(-0.766, 0.3)
```

Check:
1. Is MCMC sampling near these values? (print c samples median)
2. Or is MCMC moving far away? (suggests data mismatch)

---

## Root Cause Hypotheses

### Hypothesis A: Different Dataset
**Symptom**: scales values differ significantly from expected
**Cause**: Using different SNe or quality cuts than golden run
**Fix**: Need to find/reconstruct the exact dataset used in golden run

### Hypothesis B: Different Stage 1 Results
**Symptom**: alpha_obs values are different
**Cause**: Stage 1 optimization produced different alpha values
**Fix**: Need to verify Stage 1 is producing correct results

### Hypothesis C: Priors Are Wrong
**Symptom**: MCMC posterior for c differs from priors
**Cause**: The hardcoded c values don't actually correspond to the physics values
**Fix**: Need to re-derive priors from scratch

### Hypothesis D: Missing Alpha Standardization
**Symptom**: Results off by a constant factor
**Cause**: Original model may have standardized alpha too?
**Fix**: Check if model should be `(alpha - mean) / std = ln_A0 + Phi_std · c`

---

## What to Check Next

### Immediate Diagnostics

1. **Run stage2_simple.py on test dataset** with `--use-informed-priors`
   ```bash
   cd /home/user/Quantum-Field-Dynamics/projects/V16.2
   python3 stages/stage2_simple.py \
     --stage1-results test_dataset/stage1_results \
     --lightcurves test_dataset/lightcurves_test.csv \
     --out test_output \
     --nchains 2 \
     --nsamples 2000 \
     --nwarmup 1000 \
     --quality-cut 2000 \
     --use-informed-priors
   ```

2. **Record the printed standardization statistics**:
   - What are the `scales` values?
   - What is the redshift range?
   - How many SNe are included?

3. **Record the MCMC results**:
   - What are the c sample medians?
   - What are the k_J, eta', xi results?
   - How far are they from golden reference?

4. **Check MCMC diagnostics**:
   - Are there divergences?
   - What is R-hat?
   - Did it converge?

### Questions to Answer

1. **Do you have the November 5, 2024 Stage 1 results?**
   - If yes, check what the standardization statistics were
   - Extract the exact dataset (SNIDs) used

2. **What results are you currently getting?**
   - What k_J value?
   - What scales values?
   - Is it close or very far?

3. **Are you using the same data?**
   - Same lightcurves CSV?
   - Same quality cuts?
   - Same SNe?

---

## Expected vs. Reality

### If scales match (≈ 0.173, 0.279, 0.110)

**Then the priors should work!**

With c0 ≈ 1.857 and scales[0] ≈ 0.173:
- k_J = 1.857 / 0.173 ≈ 10.74 ✓

### If scales don't match

**Then you have a dataset mismatch!**

Example: If scales[0] = 0.20 (different from 0.173):
- k_J = 1.857 / 0.20 ≈ 9.3 ✗

The priors (c values) are **dataset-specific** despite claims of being "data-independent"!

---

## The Paradox

The PROGRESS.md claims:
> "Informed priors DIRECTLY on standardized coefficients c, making them data-independent"

But this is **FALSE** if:
1. Different datasets have different `scales`
2. The priors are on `c`, not on `k_J`
3. The transformation depends on `scales`

**The priors ARE data-dependent** through the standardization statistics!

This is exactly the bug that PROGRESS.md claimed was fixed!

---

## Recommended Action

**Before proceeding, run the diagnostic command above and report**:

1. What `scales` values you get
2. What k_J result you get
3. How many SNe are in your dataset
4. What redshift range you have

Then we can pinpoint exactly where the discrepancy is coming from.
