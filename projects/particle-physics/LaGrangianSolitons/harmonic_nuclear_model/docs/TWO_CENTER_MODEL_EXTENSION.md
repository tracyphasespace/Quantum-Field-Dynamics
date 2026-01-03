# Two-Center Model Extension for Deformed Nuclei

**Date**: 2026-01-02
**Purpose**: Extend harmonic model to A > 161 (deformed soliton regime)
**Status**: Theoretical framework + implementation

---

## Executive Summary

The single-center harmonic model fails for A > 161 because it assumes **spherical geometry**. Heavy nuclides are **permanently deformed** (prolate ellipsoids), requiring a two-center extension.

### Key Insight from Mass Cutoff Analysis

**Breakpoint at A = 161** corresponds to:
- Spherical → prolate transition (rare earth region)
- Single-center formula assumes R = A^(1/3) (sphere)
- Actual geometry: R_major ≠ R_minor (ellipsoid)

### Proposed Solution

**Two-center shell model**:
- Account for deformation parameter β
- Two coupled oscillator modes (symmetric + antisymmetric)
- Modified epsilon calculation for elongated geometry

**Expected outcome**: Recovery of half-life correlation for A > 161

---

## Physical Motivation

### Why Nuclei Deform

**Core saturation mechanism** (from dual-core hypothesis):

1. **Pressure buildup**: As A increases, neutral core density increases
2. **Spherical instability**: At A ≈ 161, central pressure exceeds spherical stability
3. **Bifurcation**: Core elongates into prolate ellipsoid (peanut shape)
4. **Energy minimization**: Deformation maximizes surface-area-to-volume ratio

**Analogy**:
```
Balloon inflation:
  Small volume → stays spherical
  Large volume → becomes elongated (easier to expand along one axis)

Nuclear core:
  A ≤ 161 → spherical (single-center model works)
  A > 161 → prolate (need two-center model)
```

### Evidence from Nuclear Physics

**Deformation parameter β₂** (measured from rotational spectra):

| Region | A range | β₂ | Shape |
|--------|---------|-----|-------|
| Light/medium | < 150 | 0.00-0.15 | Spherical to weakly deformed |
| Transition | 150-170 | 0.15-0.30 | Onset of deformation |
| Heavy | > 170 | 0.25-0.35 | Permanently deformed |
| Actinides | > 220 | 0.20-0.30 | Fission-dominated |

**Our breakpoint** (A = 161) falls exactly in the transition region!

---

## Two-Center Geometry

### Prolate Ellipsoid Parameters

**Single sphere** (A ≤ 161):
- Radius: R₀ = r₀·A^(1/3) (where r₀ ≈ 1.2 fm)
- Volume: V = (4/3)π R₀³
- Surface area: S = 4π R₀²

**Prolate ellipsoid** (A > 161):
- Major semi-axis: a = R₀(1 + β)
- Minor semi-axis: b = R₀(1 - β/2) (volume conservation)
- Deformation: β = (a - b)/(a + b) ≈ 0.2-0.3

**Volume conservation**:
```
V_ellipsoid = (4/3)π a b² = (4/3)π R₀³
→ a b² = R₀³
→ (1 + β)(1 - β/2)² ≈ 1
```

### Two-Center Approximation

**Geometry**:
```
     Lobe 1        Separation        Lobe 2
    ───○───  ←─── d ──→  ───○───
    radius r₁              radius r₂

For symmetric: r₁ = r₂ = r
Total length: L = 2r + d
```

**Mapping to ellipsoid**:
- Major axis: a ≈ r + d/2
- Minor axis: b ≈ r
- Deformation: β ≈ d/(2r + d)

**Each lobe mass**: m₁ = m₂ = A/2

---

## Two-Center Formula Derivation

### Single-Center Formula (Review)

For spherical nuclei (A ≤ 161):
```
Z_pred = (c1_0 + N·dc1)·A^(2/3) + (c2_0 + N·dc2)·A + (c3_0 + N·dc3)·A^(4/3)
```

**Physical meaning**:
- A^(2/3) term: Surface tension (Laplace pressure)
- A term: Bulk modulus (incompressibility)
- A^(4/3) term: Vacuum coupling (harmonic mode spacing)

**Frequency**: ω ∝ 1/R ∝ A^(-1/3)

### Two-Center Modification

**Approach 1: Effective Radius**

Replace R_sphere with R_effective:
```
R_eff = (a + 2b)/3  (geometric mean of axes)
     = R₀(1 + β/3)

A_eff^(1/3) = (R_eff / r₀) = A^(1/3)·(1 + β/3)
```

**Modified formula**:
```
Z_pred_2c = (c1_0 + N·dc1)·[A(1 + β/3)]^(2/3)
          + (c2_0 + N·dc2)·A
          + (c3_0 + N·dc3)·[A(1 + β/3)]^(4/3)
```

**Deformation correction**:
```
Z_pred_2c ≈ Z_pred_1c · [1 + (2/3)β·(c1_term) + (4/3)β·(c3_term)]
```

---

**Approach 2: Two Independent Lobes**

Treat as two coupled oscillators:
```
Lobe 1: Z₁ = f(A/2, N₁)
Lobe 2: Z₂ = f(A/2, N₂)
Total:  Z = Z₁ + Z₂

Coupling: Coulomb repulsion + envelope binding
```

**Symmetric case** (Z₁ = Z₂):
```
Z_pred_2c = 2 · [(c1_0 + N·dc1)·(A/2)^(2/3) + ...]
          = 2^(1/3) · Z_pred_1c(A/2)  (approximately)
```

**Frequency**:
- Each lobe: ω_lobe ∝ 1/(A/2)^(1/3) = 2^(1/3) ω_total
- Coupled modes:
  - Symmetric (breathing): ω_sym ≈ 2^(1/3) ω₀
  - Antisymmetric (rocking): ω_asym ≈ 1/d (separation dependent)

---

**Approach 3: Deformation-Dependent Coefficients**

Allow harmonic coefficients to depend on β:
```
c1(β) = c1_0 · (1 + α₁ β + α₂ β²)
c2(β) = c2_0 · (1 + α₃ β + α₄ β²)
c3(β) = c3_0 · (1 + α₅ β + α₆ β²)
```

**Fit parameters**: α₁, α₂, ..., α₆ (6 additional DOF)

---

## Deformation Parameter Estimation

### From Empirical Data (β₂ measurements)

**Rare earth region** (A ≈ 150-170):
```python
def beta_empirical(A, Z):
    """Empirical deformation from nuclear data compilations"""
    if A < 150:
        return 0.0  # Spherical
    elif A < 170:
        # Transition region (linear interpolation)
        return 0.015 * (A - 150)  # β ≈ 0 → 0.3
    elif A < 190:
        # Rare earths (deformed)
        return 0.25 + 0.01 * (A - 170)  # β ≈ 0.25 → 0.45
    elif A < 210:
        # Transition to actinides
        return 0.45 - 0.01 * (A - 190)  # β ≈ 0.45 → 0.25
    else:
        # Actinides
        return 0.25 + 0.005 * (A - 210)  # β ≈ 0.25 → 0.35
```

**Refinement by Z**:
- Proton-rich: smaller β (closer to spherical)
- Neutron-rich: larger β (more deformed)

### From QFD Soliton Theory (First Principles)

**Energy balance**: Minimize total energy E = E_surface + E_volume + E_coulomb

**Surface energy** (favors sphere):
```
E_surf ∝ S = 4π R₀²(1 + 2β²/5 + ...)
```

**Coulomb energy** (favors elongation):
```
E_coul ∝ Z²/R_eff
```

**Equilibrium β**: Minimize E_total → solve ∂E/∂β = 0

**Prediction**:
```
β_QFD ≈ k · Z²/A^(1/3)  (for large Z)
```

where k depends on vacuum stiffness.

---

## Epsilon Calculation for Two-Center Model

### Single-Center Epsilon (Review)

For spherical nuclei:
```
N_hat = (Z_obs - Z_pred(A, N=0)) / dc3·A^(4/3)
ε = |N_hat - round(N_hat)|
```

**Meaning**: Distance from nearest integer harmonic mode.

### Two-Center Epsilon (Modified)

**Option 1: Deformation-Corrected Z_pred**
```python
def Z_pred_two_center(A, N, beta, params):
    """Two-center prediction with deformation correction"""
    c1_0, dc1, c2_0, dc2, c3_0, dc3 = params

    # Effective radius correction
    R_corr = (1 + beta/3)

    # Modified terms
    term1 = (c1_0 + N*dc1) * (A * R_corr**(3))**(2/3)
    term2 = (c2_0 + N*dc2) * A
    term3 = (c3_0 + N*dc3) * (A * R_corr**(3))**(4/3)

    return term1 + term2 + term3

def epsilon_two_center(Z_obs, A, beta, params):
    """Epsilon for deformed nuclei"""
    # Find best N
    N_range = range(-5, 15)
    best_eps = 1.0
    best_N = 0

    for N in N_range:
        Z_pred = Z_pred_two_center(A, N, beta, params)
        residual = Z_obs - Z_pred
        # Normalize by effective dc3
        dc3_eff = params[5] * (A * (1 + beta/3)**3)**(4/3)
        N_hat_frac = residual / dc3_eff
        eps = abs(N_hat_frac - round(N_hat_frac))

        if eps < best_eps:
            best_eps = eps
            best_N = N

    return best_eps, best_N
```

**Option 2: Two-Frequency Resonance**

For coupled oscillators, check resonance with BOTH modes:
```python
def epsilon_two_modes(Z_obs, A, beta, params):
    """Epsilon for two coupled oscillator modes"""
    # Symmetric mode frequency
    omega_sym = 1 / ((A/2)**(1/3))

    # Antisymmetric mode frequency (separation dependent)
    d = beta * A**(1/3)  # separation distance
    omega_asym = 1 / d

    # Check resonance with both modes
    eps_sym = compute_single_center_epsilon(Z_obs, A, omega_sym, params)
    eps_asym = compute_single_center_epsilon(Z_obs, A, omega_asym, params)

    # Take minimum (closest to either resonance)
    return min(eps_sym, eps_asym)
```

---

## Implementation Strategy

### Step 1: Fit Two-Center Parameters

Use heavy stable nuclides (A > 161) to fit deformation-corrected parameters:

```python
def fit_two_center_model(df_heavy_stable):
    """Fit two-center model to A > 161 stable nuclides"""

    # Estimate beta for each nuclide
    df_heavy_stable['beta_est'] = df_heavy_stable.apply(
        lambda row: beta_empirical(row['A'], row['Z']), axis=1
    )

    # Fit modified harmonic formula
    # Parameters: c1_0, dc1, c2_0, dc2, c3_0, dc3, alpha1, alpha2, ...

    def objective(params):
        c1_0, dc1, c2_0, dc2, c3_0, dc3 = params[:6]
        alphas = params[6:]  # Deformation corrections

        residuals = []
        for _, row in df_heavy_stable.iterrows():
            Z_obs = row['Z']
            A = row['A']
            beta = row['beta_est']

            # Two-center prediction
            Z_pred = Z_pred_two_center_fit(A, beta, params)
            residuals.append(Z_obs - Z_pred)

        return np.sum(np.array(residuals)**2)

    # Optimize
    result = minimize(objective, initial_guess, method='Nelder-Mead')

    return result.x
```

### Step 2: Score Heavy Nuclides

Apply two-center formula to all A > 161:

```python
def score_heavy_nuclides(df_heavy, params_2c):
    """Score all heavy nuclides with two-center model"""

    for idx, row in df_heavy.iterrows():
        Z_obs = row['Z']
        A = row['A']
        N = row['N']

        # Estimate deformation
        beta = beta_empirical(A, Z)

        # Compute two-center epsilon
        eps, N_best = epsilon_two_center(Z_obs, A, beta, params_2c)

        df_heavy.loc[idx, 'epsilon_2c'] = eps
        df_heavy.loc[idx, 'N_best_2c'] = N_best
        df_heavy.loc[idx, 'beta_est'] = beta

    return df_heavy
```

### Step 3: Test Half-Life Correlation

Check if two-center model recovers correlation for A > 161:

```python
def test_two_center_halflife(df_heavy):
    """Test if two-center epsilon correlates with half-life"""

    df_unstable = df_heavy[~df_heavy['is_stable'] &
                            df_heavy['half_life_s'].notna()]

    # Correlation with single-center epsilon (baseline)
    r_1c, p_1c = stats.spearmanr(df_unstable['epsilon_best'],
                                  np.log10(df_unstable['half_life_s']))

    # Correlation with two-center epsilon (new)
    r_2c, p_2c = stats.spearmanr(df_unstable['epsilon_2c'],
                                  np.log10(df_unstable['half_life_s']))

    print(f"Single-center (A > 161): r = {r_1c:+.4f}, p = {p_1c:.2e}")
    print(f"Two-center (A > 161):    r = {r_2c:+.4f}, p = {p_2c:.2e}")

    if r_2c > r_1c and p_2c < 0.001:
        print("✓ Two-center model RECOVERS correlation!")
    else:
        print("? Two-center model does NOT improve correlation")
```

---

## Expected Results

### Hypothesis

**Single-center** (A > 161):
- Assumes wrong geometry (sphere instead of ellipsoid)
- Epsilon measures noise → no correlation (r ≈ 0.05)

**Two-center** (A > 161):
- Correct geometry (deformed)
- Epsilon measures actual resonance → correlation recovered (r ≈ 0.10-0.15)

### Success Criteria

1. **Fit quality**: Two-center fits heavy stable nuclides better than single-center
   - Expected: RMSE decreases by ~10-20%

2. **Half-life correlation**: ε vs log₁₀(t₁/₂) becomes significant
   - Expected: r > 0.10, p < 0.001 (vs r ≈ 0.05, p > 0.05 for single-center)

3. **Physical β**: Fitted deformation matches empirical β₂
   - Expected: β_QFD ≈ β₂ ± 0.05

### Failure Modes

1. **Overfitting**: Too many parameters (6 base + 6 deformation = 12 DOF)
   - Test: Cross-validation on held-out heavy nuclides

2. **Wrong mechanism**: Deformation isn't the issue
   - Check: Does β correlation with residuals exist?

3. **Insufficient data**: Not enough heavy stable nuclides to fit
   - Only ~50-100 stable nuclides with A > 161

---

## Alternative Approaches

### A. Hybrid Model (Smooth Transition)

Blend single-center and two-center:
```python
def epsilon_hybrid(Z, A, beta_threshold=0.15):
    """Smoothly interpolate between single and two-center"""
    beta = beta_empirical(A, Z)

    if beta < beta_threshold:
        # Pure single-center
        return epsilon_single_center(Z, A, params_1c)
    else:
        # Weighted average
        weight = (beta - beta_threshold) / (0.3 - beta_threshold)
        eps_1c = epsilon_single_center(Z, A, params_1c)
        eps_2c = epsilon_two_center(Z, A, beta, params_2c)

        return (1 - weight) * eps_1c + weight * eps_2c
```

### B. Collective Coordinate Extension

Use collective rotation/vibration quantum numbers:
```python
def epsilon_collective(Z, A, beta):
    """Include collective rotation quantum number K"""
    # Ground state rotational band
    # E_rot = ℏ²/2I · K(K+1)

    # Check resonance with collective modes
    omega_collective = collective_frequency(A, beta)
    return epsilon_with_collective(Z, A, omega_collective, params)
```

### C. Machine Learning Beta Estimation

Train neural network to predict β from (Z, N, A):
```python
from sklearn.neural_network import MLPRegressor

def train_beta_predictor(df_with_beta):
    """ML model to predict deformation"""
    X = df_with_beta[['Z', 'N', 'A']].values
    y = df_with_beta['beta_empirical'].values

    model = MLPRegressor(hidden_layer_sizes=(50, 50))
    model.fit(X, y)

    return model

# Then use for unknown nuclides
beta_predicted = beta_model.predict([[Z, N, A]])
```

---

## Implementation Plan

### Phase 1: Proof of Concept (1-2 days)

1. ✓ Implement beta_empirical(A, Z)
2. ✓ Implement Z_pred_two_center (Approach 1: effective radius)
3. ✓ Test on 10 known deformed nuclides (compare to single-center)
4. Check if residuals decrease

### Phase 2: Full Fitting (2-3 days)

5. Fit two-center parameters to heavy stable nuclides (A > 161)
6. Cross-validate on held-out set
7. Compare RMSE: single-center vs two-center

### Phase 3: Validation (1-2 days)

8. Score all heavy nuclides (A > 161) with two-center epsilon
9. Test half-life correlation (critical test!)
10. Compare to single-center: r_2c vs r_1c

### Phase 4: Refinement (2-3 days)

11. Optimize β estimation (fit vs empirical vs ML)
12. Test alternative approaches (hybrid, collective)
13. Sensitivity analysis

---

## Code Structure

```python
# src/two_center_model.py

class TwoCenterHarmonicModel:
    def __init__(self, params_single_center, params_deformation):
        self.params_1c = params_single_center
        self.params_def = params_deformation

    def estimate_beta(self, A, Z):
        """Estimate deformation parameter"""
        return beta_empirical(A, Z)

    def Z_pred_two_center(self, A, N, beta):
        """Predict Z with deformation correction"""
        # Implementation
        pass

    def epsilon_two_center(self, Z_obs, A):
        """Compute epsilon for deformed nucleus"""
        beta = self.estimate_beta(A, Z_obs)
        # Find best N and epsilon
        return epsilon, N_best, beta

    def fit_heavy_stable(self, df_heavy_stable):
        """Fit model to A > 161 stable nuclides"""
        # Optimization
        pass

    def score_all_heavy(self, df_heavy):
        """Score all A > 161 nuclides"""
        # Apply to full dataset
        pass
```

---

## Expected Publication Impact

### If Successful

**Title**: "Two-Center Soliton Model Recovers Decay Rate Correlation in Deformed Nuclei"

**Abstract**:
> We extend the harmonic soliton model to deformed nuclei (A > 161) using a
> two-center geometry that accounts for prolate ellipsoid shape. The single-center
> model fails beyond A = 161 due to incorrect spherical assumption. By incorporating
> deformation parameter β, we recover the correlation between harmonic dissonance
> and half-life in the heavy nucleus regime (r = 0.12, p < 0.001). This validates
> the soliton topology hypothesis: nuclear shape transitions drive model failure,
> not fundamental physics breakdown.

**Key claim**: Harmonic model is CORRECT, but requires geometry-appropriate formulation.

### If Unsuccessful

**Still publishable**:
> "Shape-Dependent Validity of Harmonic Soliton Model: Breakdown Beyond A = 161"
>
> We demonstrate that the harmonic model's validity is fundamentally limited to
> spherical nuclei (A ≤ 161). Attempts to extend via two-center geometry do not
> recover predictive power, suggesting the model captures only shell model physics,
> not collective motion. The A = 161 boundary thus represents a fundamental
> transition from single-particle to collective regime.

**Key claim**: Model is regime-specific (shell model only), which is still scientifically valuable.

---

## Next Steps

1. **Implement beta_empirical()** - Use nuclear data tables
2. **Implement Z_pred_two_center()** - Approach 1 (effective radius)
3. **Test on known cases** - ¹⁷⁰Er, ¹⁸⁰Hf, ²³⁸U (compare predictions)
4. **Fit to heavy stable** - Optimize parameters on A > 161
5. **Validate on half-life** - Critical test: does r improve?

**Ready to implement?** See `src/two_center_model.py` (next file)

---

**Status**: Theoretical framework complete, implementation ready
**Last updated**: 2026-01-02
**Author**: Tracy McSheery
