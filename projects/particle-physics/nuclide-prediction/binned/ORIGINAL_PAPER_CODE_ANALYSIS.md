# Original Paper Code Analysis

**Location**: `/home/tracy/development/qfd_hydrogen_project/qfd_research_suite/scripts/mixture_backbones.py`

**Date Found**: 2025-12-29

---

## Overview

This is the **actual code used for the paper** "Three Bins Two Parameters". It's significantly different from our implementation in important ways.

---

## Key Differences from Our Implementation

### 1. Includes Constant Term (c₀)

**Original Model**:
```python
Q = c₀ + c₁·A^(2/3) + c₂·A
```

**Our Model**:
```python
Q = c₁·A^(2/3) + c₂·A  # No c₀
```

**Impact**: The constant offset term allows the baselines to shift vertically, not just scale differently.

### 2. Clusters in Residual Space

**Original Approach**:
1. Fit single global backbone first
2. Compute residuals: `ΔZ = Z - Z_single`
3. **Cluster in (A^(1/3), ΔZ) space** using GMM
4. Fit separate backbones (with c₀, c₁, c₂) to each cluster

```python
# Step 1: Single backbone
fit1, Zhat1, resid = fit_backbone(A, Z)

# Step 2: Feature space for clustering
X = np.c_[df_aug["A13"].to_numpy(),     # A^(1/3)
          df_aug["dZ"].to_numpy()]      # Z - Z_single

# Step 3: GMM clustering
gmm = gmm_em(X[train_idx], K, seed=args.seed)

# Step 4: Per-cluster fits
for k in range(Ks):
    mask = (labels==k)
    fit, Zhat_k, _ = fit_backbone(A[mask], Z[mask])
```

**Our Approach**:
- Classify based on deviation from reference backbone
- No clustering algorithm - hard threshold classification

### 3. Proper Train/Test Validation

**Original**:
```python
train_idx, test_idx = split_train_test(N, test_size=0.2, seed=42)

# Fit on train
gmm = gmm_em(X[train_idx], K, seed=args.seed)
labels_train = gmm["labels"]

# Evaluate on test
labels_test = assign_with_gmm(X[test_idx], gmm)
rmse_test = compute_rmse(test data)
```

**Our Approach**:
- Fit on all data
- No train/test split

### 4. Model Selection via BIC/AIC

**Original**:
```python
for K in range(1, args.max_k+1):
    gmm = gmm_em(X[train_idx], K, seed=args.seed)
    bic = -2*prev_ll + p*np.log(N)
    aic = -2*prev_ll + 2*p
```

Automatically selects best K (number of components) based on statistical criteria.

**Our Approach**:
- Fixed K=3 (assumed from paper)
- No model selection

### 5. GMM Implementation

**Original**: Full covariance 2D Gaussian Mixture in (A^(1/3), ΔZ) space
```python
def gmm_em(X, K, seed=0, max_iter=500, tol=1e-6, reg=1e-6):
    # Full covariance matrices
    covs = np.array([np.cov(X.T) + reg*np.eye(d) for _ in range(K)])
```

**Our Approach**: GMM in (A, Z) space with regression residuals

---

## Detailed Algorithm Walkthrough

### Step 1: Load Data
```python
df = load_table(args.input)
A = df["A"].to_numpy(float)
Z = df["Z"].to_numpy(float)
```

### Step 2: Single Backbone (Baseline)
```python
def design_A(A):
    return np.c_[np.ones_like(A),    # constant
                 np.cbrt(A)**2,       # A^(2/3)
                 A]                   # A

fit1, Zhat1, resid = fit_backbone(A, Z)
# fit1.coef = [c₀, c₁, c₂]
```

Results: `Q = c₀ + c₁·A^(2/3) + c₂·A`

### Step 3: Create Feature Space for Clustering
```python
df_aug["A13"] = np.cbrt(A)           # A^(1/3)
df_aug["dZ"] = Z - Zhat1             # Residual from single backbone

X = np.c_[df_aug["A13"], df_aug["dZ"]]  # 2D feature space
```

**Why this feature space?**
- A^(1/3) captures the dominant scaling (since A^(2/3) term dominates)
- ΔZ captures the deviation pattern
- Clusters nuclei with similar deviation behavior

### Step 4: Train/Test Split
```python
train_idx, test_idx = split_train_test(N, test_size=0.2, seed=42)
```

80% train, 20% test for validation

### Step 5: GMM Clustering (on training data)
```python
for K in range(1, max_k+1):  # Try K=1,2,3
    gmm = gmm_em(X[train_idx], K, seed=args.seed)
```

**GMM Components**:
- `weights`: π_k (mixing proportions)
- `means`: μ_k in (A^(1/3), ΔZ) space
- `covs`: Σ_k (full 2×2 covariance matrices)
- `labels`: Hard assignments (MAP)
- `bic`, `aic`: Model selection criteria

### Step 6: Per-Cluster Backbone Fitting
```python
def per_cluster_fits(A, Z, labels):
    for k in range(Ks):
        mask = (labels==k)
        fit, Zhat_k, _ = fit_backbone(A[mask], Z[mask])
        # Returns c₀_k, c₁_k, c₂_k for cluster k
```

Each cluster gets its own `Q_k = c₀_k + c₁_k·A^(2/3) + c₂_k·A`

### Step 7: Test Set Evaluation
```python
# Assign test points to clusters using trained GMM
labels_test = assign_with_gmm(X[test_idx], gmm)

# Predict using cluster-specific models
for k in range(K):
    mask = (labels_test==k)
    Xk = design_A(A[test_idx][mask])
    overall_preds[mask] = Xk @ models[k]  # c₀_k, c₁_k, c₂_k

rmse_test = np.sqrt(np.mean((Z[test_idx] - overall_preds)**2))
```

### Step 8: Model Selection
```python
best_bic_row = df_mix.loc[df_mix["bic"].idxmin()]
best_rmse_row = df_mix.loc[df_mix["rmse_test"].idxmin()]
```

Choose K that minimizes BIC or test RMSE

---

## Why This Works Better

### 1. Constant Term Flexibility
With c₀, each cluster can have a different baseline offset:
```
Cluster 1: Q = -2.0 + 1.2·A^(2/3) + 0.25·A  (charge-rich, higher offset)
Cluster 2: Q =  0.0 + 0.5·A^(2/3) + 0.32·A  (nominal, no offset)
Cluster 3: Q = +1.5 - 0.1·A^(2/3) + 0.40·A  (charge-poor, positive offset + negative c₁)
```

Without c₀, we force all lines through origin in transformed space, limiting flexibility.

### 2. Residual Space Clustering
Clustering in (A^(1/3), ΔZ) is smarter than (A, Z):
- Removes the dominant trend first
- Focuses on deviation patterns
- More stable GMM convergence

### 3. Statistical Rigor
- Train/test split prevents overfitting
- BIC/AIC for principled model selection
- Proper uncertainty quantification

---

## Replication Steps

To replicate the paper results using the original code:

```bash
cd /home/tracy/development/qfd_hydrogen_project/qfd_research_suite

python3 scripts/mixture_backbones.py \
    --input /path/to/NuMass.csv \
    --outdir fit_results/replication \
    --max_k 3 \
    --test_size 0.2 \
    --seed 42 \
    --plots
```

**Expected Outputs**:
- `mixture_summary.csv`: Performance for K=1,2,3
- `coeffs_K3.csv`: Three sets of (c₀, c₁, c₂)
- `summary.json`: Best K by BIC and test RMSE

---

## Why Our Model Had c₁=0 Issue

**Root Cause**: We didn't include the constant term c₀!

Without c₀:
```
Q = c₁·A^(2/3) + c₂·A
```

For charge-poor, the best fit wanted to shift the line down, but without c₀, it could only do that by making c₁ negative.

With c₀ (original):
```
Q = c₀ + c₁·A^(2/3) + c₂·A
```

Charge-poor can shift down via c₀ > 0, keeping c₁ positive (or still negative, but for better reasons).

---

## Action Items

### Immediate
1. **Re-run original code** on NuMass.csv
   - Get exact parameters from paper methodology
   - Compare with our results

2. **Add c₀ term** to our three-track model
   - Update: `Q = c₀ + c₁·A^(2/3) + c₂·A`
   - Re-fit and compare RMSE

3. **Try residual space clustering**
   - Cluster in (A^(1/3), ΔZ) like original
   - See if it improves over threshold classification

### Future
1. **Implement exact paper methodology**
   - Copy original approach verbatim
   - Verify we get RMSE ≈ 1.107 Z

2. **Hybrid approach**
   - Use our physics-based classification
   - But add c₀ term
   - Compare with original GMM clustering

---

## Summary

**Key Insights**:
1. Original uses **three-parameter model** (c₀, c₁, c₂) not two!
2. Clusters in **residual space** (A^(1/3), ΔZ), not (A, Z)
3. Proper **train/test validation** with BIC/AIC model selection
4. Our c₁=0 issue likely due to missing c₀ term

**Next Step**: Re-run original code to replicate paper's 1.107 Z result

---

**Status**: ✅ ORIGINAL CODE FOUND AND ANALYZED
**Location**: `/home/tracy/development/qfd_hydrogen_project/qfd_research_suite/scripts/mixture_backbones.py`
**Action**: Ready to replicate paper methodology
