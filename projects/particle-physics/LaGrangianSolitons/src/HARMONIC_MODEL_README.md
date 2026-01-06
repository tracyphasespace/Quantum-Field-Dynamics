# Harmonic Model Core Module Documentation

**Module**: `src/harmonic_model.py`
**Purpose**: Core mathematical framework for harmonic family nuclear structure model
**Status**: ✓ PRODUCTION-READY (all tests pass)
**Date**: 2026-01-02

---

## Overview

This module implements the harmonic family model for nuclear structure, providing:
- **Z prediction** for given (A, mode index N, family)
- **Dissonance metric ε** (distance to nearest harmonic)
- **Continuous mode estimation N̂** (inversion of model)
- **Best-family scoring** (which family best describes a nuclide)
- **Parameter validation** and consistency checks

Implements **EXPERIMENT_PLAN.md §1** (Definitions).

---

## Mathematical Model

### 1. Harmonic Family Line

For each family F ∈ {A, B, C}, the predicted proton number for mass number A and mode index N is:

```
Z_pred(A, N) = (c1_0 + N·dc1)·A^(2/3) + (c2_0 + N·dc2)·A + (c3_0 + N·dc3)
```

This can be decomposed as:
```
Z_pred(A, N) = Z_0(A) + N·ΔZ(A)
```

Where:
- **Baseline** (N=0 line):
  `Z_0(A) = c1_0·A^(2/3) + c2_0·A + c3_0`

- **Mode spacing**:
  `ΔZ(A) = dc1·A^(2/3) + dc2·A + dc3`

### 2. Continuous Mode Estimate

Given an observed (A, Z), estimate which mode it's near:

```
N̂(A,Z) = [Z - Z_0(A)] / ΔZ(A)
```

This inverts the model: if N̂ ≈ 2.1, the nuclide is close to the N=2 harmonic.

### 3. Dissonance Metric

**Definition**:
```
ε(A,Z) = |N̂ - round(N̂)| ∈ [0, 0.5]
```

**Interpretation**:
- ε = 0: Exactly on a harmonic mode (resonant)
- ε = 0.5: Maximally between harmonics (dissonant)

**Pre-registered thresholds** (EXPERIMENT_PLAN.md):
- **Harmonic**: ε < 0.05
- **Near-harmonic**: 0.05 ≤ ε < 0.15
- **Dissonant**: ε ≥ 0.15

### 4. The "Clock Step" (dc3)

**dc3** is the A-independent component of mode spacing:

```
ΔZ(A) = dc1·A^(2/3) + dc2·A + dc3
        ↑             ↑      ↑
   surface term  volume  constant step
```

**dc3 universality hypothesis**:
If the harmonic model is a real coordinate system, different families should share nearly identical dc3 values (within ~1-2%).

**Empirical expectation**: dc3 ≈ -0.865 ± 0.013

---

## Core Functions

### `FamilyParams` (dataclass)

**Purpose**: Store parameters for a single harmonic family.

**Attributes**:
```python
name: str          # Family identifier ("A", "B", "C", ...)
c1_0: float        # Baseline A^(2/3) coefficient
c2_0: float        # Baseline A coefficient
c3_0: float        # Baseline constant term
dc1: float         # Per-mode increment for A^(2/3)
dc2: float         # Per-mode increment for A
dc3: float         # Per-mode constant increment (the "clock step")
```

**Methods**:
```python
.to_dict() -> Dict         # Serialize to JSON-compatible dict
.from_dict(data) -> FamilyParams  # Deserialize from dict
```

**Validation**:
- Warns if dc3 > 0 (expected negative for Z-decreasing modes)

**Example**:
```python
params = FamilyParams(
    name="A",
    c1_0=1.5,
    c2_0=0.4,
    c3_0=-5.0,
    dc1=-0.05,
    dc2=-0.01,
    dc3=-0.865,
)
```

---

### `Z_baseline(A, params) -> Z_0`

Calculate baseline Z (N=0 line).

**Formula**:
```
Z_0(A) = c1_0·A^(2/3) + c2_0·A + c3_0
```

**Example**:
```python
Z_0 = Z_baseline(100, params)  # Z at N=0 for A=100
```

---

### `delta_Z(A, params) -> ΔZ`

Calculate mode spacing (Z-increment per mode step).

**Formula**:
```
ΔZ(A) = dc1·A^(2/3) + dc2·A + dc3
```

**Example**:
```python
dZ = delta_Z(100, params)  # Mode spacing for A=100
print(f"Each mode step changes Z by {dZ:.2f}")
```

**Note**: Typically negative (Z decreases with increasing N).

---

### `Z_predicted(A, N, params) -> Z_pred`

Predict Z for given (A, mode index N).

**Formula**:
```
Z_pred(A, N) = Z_0(A) + N·ΔZ(A)
```

**Example**:
```python
# Predict Z for A=100, N=0, 1, 2
for N in [0, 1, 2]:
    Z = Z_predicted(100, N, params)
    print(f"N={N}: Z_pred={Z:.1f}")
```

**Vectorized**:
```python
A = np.array([50, 100, 150])
N = np.array([0, 1, 2])
Z = Z_predicted(A, N, params)  # Element-wise
```

---

### `N_hat(A, Z, params) -> N̂`

Estimate continuous mode for observed (A, Z).

**Formula**:
```
N̂ = [Z - Z_0(A)] / ΔZ(A)
```

**Example**:
```python
A, Z = 238, 92  # U-238
Nhat = N_hat(A, Z, params)
print(f"U-238 is near mode N̂ = {Nhat:.2f}")
print(f"Nearest harmonic: N = {round(Nhat)}")
```

**Inverse property**:
```python
# Should satisfy: N̂ ≈ N for Z on harmonic
N = 2
Z = Z_predicted(A, N, params)
Nhat = N_hat(A, Z, params)
assert abs(Nhat - N) < 1e-6  # Exact inversion
```

---

### `epsilon(A, Z, params) -> ε`

Calculate dissonance (distance to nearest harmonic).

**Formula**:
```
ε = |N̂ - round(N̂)|
```

**Range**: [0, 0.5]

**Example**:
```python
A, Z = 238, 92
eps = epsilon(A, Z, params)
print(f"U-238 dissonance: ε = {eps:.3f}")

if eps < 0.05:
    print("HARMONIC (resonant)")
elif eps < 0.15:
    print("NEAR-HARMONIC")
else:
    print("DISSONANT")
```

**Vectorized**:
```python
# Calculate ε for all nuclides
A_all = df['A'].values
Z_all = df['Z'].values
eps_all = epsilon(A_all, Z_all, params)
```

---

### `residual(A, Z, params) -> residual`

Calculate Z residual (Z_obs - Z_pred at nearest mode).

**Formula**:
```
N_best = round(N̂)
residual = Z - Z_pred(A, N_best)
```

**Example**:
```python
A, Z = 238, 92
resid = residual(A, Z, params)
print(f"U-238 residual: {resid:.2f} protons from harmonic")
```

**Relation to ε**:
```
|residual| ≈ ε · |ΔZ(A)|
```

---

### `score_nuclide(A, Z, params) -> dict`

Score a single nuclide against one family.

**Returns**:
```python
{
    'family': str,       # Family name
    'N_hat': float,      # Continuous mode estimate
    'N_best': int,       # Nearest integer mode
    'epsilon': float,    # Dissonance
    'Z_pred': float,     # Predicted Z at N_best
    'residual': float,   # Z - Z_pred
}
```

**Example**:
```python
score = score_nuclide(238, 92, params)
print(f"U-238 vs family {score['family']}:")
print(f"  N̂ = {score['N_hat']:.2f}")
print(f"  Nearest mode: N = {score['N_best']}")
print(f"  Dissonance: ε = {score['epsilon']:.3f}")
print(f"  Residual: {score['residual']:.2f} protons")
```

---

### `score_best_family(A, Z, families) -> dict`

Score nuclide against all families, return best match.

**Args**:
```python
families: Dict[str, FamilyParams]  # e.g., {'A': params_A, 'B': params_B, 'C': params_C}
```

**Returns**:
```python
{
    'best_family': str,         # Name of best-matching family
    'epsilon_best': float,      # Dissonance for best family
    'N_hat_best': float,        # N̂ for best family
    'N_best': int,              # Nearest mode for best family
    'Z_pred_best': float,       # Z_pred for best family
    'residual_best': float,     # Residual for best family
    'epsilon_A': float,         # Dissonance for family A
    'epsilon_B': float,         # Dissonance for family B
    'epsilon_C': float,         # Dissonance for family C
    # ... (one epsilon_X per family)
}
```

**Example**:
```python
families = {
    'A': FamilyParams(name="A", ...),
    'B': FamilyParams(name="B", ...),
    'C': FamilyParams(name="C", ...),
}

score = score_best_family(238, 92, families)
print(f"U-238 best family: {score['best_family']}")
print(f"  ε = {score['epsilon_best']:.3f}")
print(f"  All families: A={score['epsilon_A']:.3f}, "
      f"B={score['epsilon_B']:.3f}, C={score['epsilon_C']:.3f}")
```

---

### `classify_by_epsilon(eps) -> str`

Classify nuclide by dissonance level.

**Thresholds** (pre-registered in EXPERIMENT_PLAN.md):
- ε < 0.05 → `"harmonic"`
- 0.05 ≤ ε < 0.15 → `"near_harmonic"`
- ε ≥ 0.15 → `"dissonant"`

**Example**:
```python
for eps_test in [0.01, 0.10, 0.30]:
    category = classify_by_epsilon(eps_test)
    print(f"ε={eps_test:.2f} → {category}")

# Output:
# ε=0.01 → harmonic
# ε=0.10 → near_harmonic
# ε=0.30 → dissonant
```

---

### `dc3_comparison(families) -> dict`

Check dc3 universality across families.

**Returns**:
```python
{
    'dc3_values': dict,        # Family name -> dc3 value
    'dc3_mean': float,         # Mean dc3 across families
    'dc3_std': float,          # Standard deviation
    'dc3_range': float,        # Max - min
    'dc3_relative_std': float, # Std / |mean| (universality metric)
}
```

**Example**:
```python
comparison = dc3_comparison(families)
print(f"dc3 values: {comparison['dc3_values']}")
print(f"Mean: {comparison['dc3_mean']:.3f}")
print(f"Std: {comparison['dc3_std']:.3f}")
print(f"Relative std: {comparison['dc3_relative_std']:.1%}")

if comparison['dc3_relative_std'] < 0.02:
    print("✓ dc3 is universal (<2% variation)")
else:
    print("✗ dc3 varies significantly across families")
```

**Interpretation**:
- Relative std < 1-2% suggests dc3 is a universal "clock step"
- Larger variation suggests families are independent parameterizations

---

### `validate_params(params, A_range=(1, 300)) -> dict`

Validate family parameters for physical consistency.

**Checks**:
1. ΔZ(A) ≠ 0 (avoid singularities in N̂ calculation)
2. 0 ≤ Z_0(A) ≤ A (physical proton number range)
3. dc3 < 0 (Z decreases with mode for neutron-rich isotopes)

**Returns**:
```python
{
    'family': str,              # Family name
    'valid': bool,              # True if no warnings
    'warnings': List[str],      # List of warning messages
    'Z_0_range': (min, max),   # Range of Z_0 over A
    'delta_Z_range': (min, max), # Range of ΔZ over A
}
```

**Example**:
```python
validation = validate_params(params, A_range=(10, 300))
if validation['valid']:
    print("✓ Parameters are physically consistent")
else:
    print("⚠ Warnings:")
    for w in validation['warnings']:
        print(f"  - {w}")
```

---

## Usage Examples

### Example 1: Score a single nuclide

```python
from harmonic_model import FamilyParams, score_nuclide

# Define family parameters
params_A = FamilyParams(
    name="A",
    c1_0=1.5,
    c2_0=0.4,
    c3_0=-5.0,
    dc1=-0.05,
    dc2=-0.01,
    dc3=-0.865,
)

# Score U-238
score = score_nuclide(A=238, Z=92, params=params_A)
print(f"U-238 vs Family A:")
print(f"  Continuous mode: N̂ = {score['N_hat']:.2f}")
print(f"  Nearest harmonic: N = {score['N_best']}")
print(f"  Dissonance: ε = {score['epsilon']:.3f}")
print(f"  Classification: {classify_by_epsilon(score['epsilon'])}")
```

---

### Example 2: Find best family for a nuclide

```python
from harmonic_model import FamilyParams, score_best_family

# Define three families
families = {
    'A': FamilyParams(name="A", c1_0=1.5, c2_0=0.4, c3_0=-5.0,
                     dc1=-0.05, dc2=-0.01, dc3=-0.865),
    'B': FamilyParams(name="B", c1_0=1.6, c2_0=0.38, c3_0=-6.0,
                     dc1=-0.048, dc2=-0.012, dc3=-0.860),
    'C': FamilyParams(name="C", c1_0=1.4, c2_0=0.42, c3_0=-4.0,
                     dc1=-0.052, dc2=-0.009, dc3=-0.870),
}

# Score U-238 against all families
score = score_best_family(A=238, Z=92, families=families)
print(f"U-238 best match: Family {score['best_family']}")
print(f"  ε_best = {score['epsilon_best']:.3f}")
print(f"  Comparison: A={score['epsilon_A']:.3f}, "
      f"B={score['epsilon_B']:.3f}, C={score['epsilon_C']:.3f}")
```

---

### Example 3: Score all nuclides in dataset

```python
import pandas as pd
from harmonic_model import score_best_family

# Load NUBASE data
df = pd.read_parquet('data/derived/nuclides_all.parquet')

# Score all nuclides
scores = []
for _, row in df.iterrows():
    score = score_best_family(row['A'], row['Z'], families)
    scores.append(score)

# Convert to DataFrame
df_scores = pd.DataFrame(scores)

# Merge with original data
df_full = pd.concat([df, df_scores], axis=1)

# Summary statistics
print(f"Mean ε: {df_full['epsilon_best'].mean():.3f}")
print(f"Median ε: {df_full['epsilon_best'].median():.3f}")

# Count by classification
for category in ['harmonic', 'near_harmonic', 'dissonant']:
    df_full['category'] = df_full['epsilon_best'].apply(classify_by_epsilon)
    count = (df_full['category'] == category).sum()
    pct = 100 * count / len(df_full)
    print(f"{category}: {count} ({pct:.1f}%)")
```

---

### Example 4: Check dc3 universality

```python
from harmonic_model import dc3_comparison

comparison = dc3_comparison(families)

print("dc3 Universality Check:")
print(f"  Family A: {comparison['dc3_values']['A']:.3f}")
print(f"  Family B: {comparison['dc3_values']['B']:.3f}")
print(f"  Family C: {comparison['dc3_values']['C']:.3f}")
print(f"  Mean: {comparison['dc3_mean']:.3f}")
print(f"  Std: {comparison['dc3_std']:.4f}")
print(f"  Relative std: {comparison['dc3_relative_std']:.2%}")

if comparison['dc3_relative_std'] < 0.02:
    print("✓ dc3 is universal (<2% variation)")
else:
    print("✗ dc3 varies significantly")
```

---

### Example 5: Validate parameters before use

```python
from harmonic_model import validate_params

validation = validate_params(params_A, A_range=(10, 300))

if validation['valid']:
    print("✓ Parameters are physically consistent")
    print(f"  Z_0 range: {validation['Z_0_range']}")
    print(f"  ΔZ range: {validation['delta_Z_range']}")
else:
    print("⚠ Parameter warnings:")
    for w in validation['warnings']:
        print(f"  - {w}")
```

---

## Integration with Experimental Pipeline

This module implements **§1 (Definitions)** of EXPERIMENT_PLAN.md and is used by:

### 1. `fit_families.py` (Parameter Fitting)
- Fits `FamilyParams` to training set (stable nuclides)
- Uses `Z_predicted()` as model function
- Minimizes `residual()` or `epsilon()` over training set
- Outputs: `family_params.json` (serialized via `.to_dict()`)

### 2. `score_harmonics.py` (Scoring All Nuclides)
- Loads fitted parameters from JSON
- Calls `score_best_family()` for each nuclide
- Outputs: `harmonic_scores.parquet` with ε, best_family, etc.

### 3. `null_models.py` (Baseline Comparisons)
- Uses `Z_baseline()` to compute smooth baseline
- Compares harmonic model to polynomial fit

### 4. Experiments (exp1, exp2, exp3, exp4)
- All experiments use ε as primary diagnostic
- `classify_by_epsilon()` for categorical analysis
- `dc3_comparison()` for model validation

---

## Verification and Testing

### Unit Tests

Comprehensive test suite in `test_harmonic_model.py`:

```bash
python -m src.test_harmonic_model
```

**Tests cover**:
- ✓ Basic functions (Z_baseline, delta_Z, Z_predicted, N_hat, epsilon)
- ✓ Scoring (score_nuclide, score_best_family)
- ✓ Classification (classify_by_epsilon)
- ✓ Validation (validate_params, dc3_comparison)
- ✓ Numerical consistency (forward-inverse roundtrip)
- ✓ Edge cases (A=1, A=300, arrays)

**Status**: All 14 tests pass ✓

### Numerical Properties

**Forward-inverse consistency**:
```python
# For any (A, Z):
Nhat = N_hat(A, Z, params)
N_rounded = round(Nhat)
Z_reconstructed = Z_predicted(A, N_rounded, params)
# Then: |Z - Z_reconstructed| ≤ |ΔZ(A)| / 2
```

**Epsilon bounds**:
```python
# Always: 0 ≤ ε ≤ 0.5
eps = epsilon(A, Z, params)
assert 0.0 <= eps <= 0.5
```

---

## Performance

### Timing

All functions are vectorized (accept numpy arrays):

```python
import numpy as np
import time

A = np.arange(1, 301)  # 300 nuclides
Z = np.arange(1, 301)

t0 = time.time()
eps = epsilon(A, Z, params)
t1 = time.time()

print(f"Calculated ε for 300 nuclides in {1000*(t1-t0):.2f} ms")
# Typical: ~1 ms for 300 nuclides
```

### Memory

`FamilyParams` is lightweight (~100 bytes per family):
```python
import sys
print(sys.getsizeof(params))  # ~100 bytes
```

---

## Mathematical Guarantees

### 1. Inversion Property

For Z exactly on a harmonic:
```
N_integer ∈ ℤ
Z = Z_predicted(A, N_integer, params)
⟹ N_hat(A, Z, params) = N_integer  (exact)
⟹ epsilon(A, Z, params) = 0        (exact)
```

### 2. Epsilon Symmetry

```
ε(A, Z) = ε(A, Z')
where Z' is the mirror point across nearest harmonic
```

### 3. Residual-Epsilon Relation

```
|residual(A, Z)| = ε(A, Z) · |ΔZ(A)|
```

---

## Known Limitations

### 1. Singularities

If ΔZ(A) → 0 for some A, `N_hat()` and `epsilon()` return `NaN`.

**Mitigation**: `validate_params()` checks for this.

### 2. Physical Bounds

Model can predict Z < 0 or Z > A for extreme parameters.

**Mitigation**: `validate_params()` warns if Z_0 exceeds physical bounds.

### 3. Integer Rounding

`N_best = round(N_hat)` uses Python's "round half to even" rule.
For N_hat = 0.5 exactly, this may round to 0 or 1 depending on context.

**Impact**: Negligible (affects only exact midpoints, which have ε=0.5 regardless).

---

## Future Enhancements

### Potential Improvements

1. **Uncertainty propagation**:
   - Add parameter covariances to `FamilyParams`
   - Compute σ_ε via error propagation

2. **Constrained fitting**:
   - Add constraints (e.g., Z_0 > 0, ΔZ < 0)
   - Bayesian priors for dc3 universality

3. **Multi-term models**:
   - Extend to A^(1/3) term (shell effects)
   - Pairing term (even-odd effects)

4. **Symbolic mode**:
   - Use SymPy for symbolic derivatives
   - Analytic Jacobians for fitting

---

## Citation

If using this module in publications, cite:

**QFD Harmonic Model:**
[Your citation when published]

**NUBASE2020** (for nuclide data):
F.G. Kondev et al., Chin. Phys. C45, 030001 (2021)

**AME2020** (for mass excess):
W. Huang et al., Chin. Phys. C45, 030002 (2021)

---

## References

- EXPERIMENT_PLAN.md (experimental protocol)
- src/parse_nubase.py (NUBASE data parsing)
- src/parse_ame.py (AME Q-value calculation)
- src/fit_families.py (parameter fitting, to be implemented)
- src/score_harmonics.py (scoring pipeline, to be implemented)

---

**Last Updated**: 2026-01-02
**Module Version**: 1.0
**Status**: ✓ PRODUCTION-READY
**Tests**: All 14 unit tests pass
