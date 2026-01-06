# Sequential Screening Analysis: Combinatorial Approach

**Date**: 2025-12-29
**Purpose**: Test if sequential curve screening improves decay predictions

---

## Executive Summary

**Tested approach**: Use charge_poor and charge_rich curves as "high confidence" screens before applying regime method.

**Result**:
- ❌ **Low thresholds** (0-3 Z): Accuracy drops to 55-64% (worse than current 95%)
- ⚠️ **Medium thresholds** (4-7 Z): Accuracy 61-72% (still worse)
- ✅ **High thresholds** (8-10 Z): Accuracy 84-95% (approaches current)

**Conclusion**: Sequential screening **only helps with very high thresholds** (>8 Z), at which point it's nearly equivalent to the current method.

---

## 1. The Overlap Problem

### Zone Analysis:

Curves are NOT cleanly separated:

| Zone | Count | % of Dataset |
|------|-------|--------------|
| Above poor curve | 2,787 | 78.3% |
| Below rich curve | 3,010 | 84.6% |
| **BOTH (above poor AND below rich)** | **2,240** | **63.0%** |
| NEITHER (between curves) | **0** | **0%** |

**Key finding**: 63% of isotopes satisfy BOTH conditions!
- These are in the "middle zone" between the extreme trajectories
- Includes most stable isotopes and ambiguous unstable ones
- Sequential screening has **conflict resolution problem**

---

## 2. Priority Cascade Approach

### Algorithm:

```
For each isotope:
  1. If dist_poor > threshold:
       → Predict β⁺ (high confidence)
  2. Else if dist_rich < -threshold:
       → Predict β⁻ (high confidence)
  3. Else:
       → Use regime method (ambiguous)
```

### Results by Threshold:

| Threshold | Overall Acc | Unstable Correct | Stable Correct | Comments |
|-----------|-------------|------------------|----------------|----------|
| **0 Z** | 63.4% | 2254/3304 (68%) | 0/253 (0%) | ALL stable misclassified! |
| **1 Z** | 63.5% | 2257/3304 (68%) | 1/253 (0.4%) | Still terrible |
| **3 Z** | 55.4% | 1969/3304 (60%) | 1/253 (0.4%) | Worst performance |
| **5 Z** | 67.1% | 2340/3304 (71%) | 48/253 (19%) | Better for stable |
| **6 Z** | 71.7% | 2472/3304 (75%) | 79/253 (31%) | Approaching baseline |
| **8 Z** | **84.0%** | 2910/3304 (88%) | 79/253 (31%) | Good compromise |
| **10 Z** | **94.7%** | 3291/3304 (99.6%) | 79/253 (31%) | Nearly matches current |
| **Current** | **95.1%** | 3304/3304 (100%) | 79/253 (31%) | Regime method |

---

## 3. Why Low Thresholds Fail

### Threshold = 3 Z Example:

**Unstable isotopes**:
- Poor screen catches: 1,717 (52%) → 54.2% accurate ❌
- Rich screen catches: 1,516 (46%) → 63.9% accurate ❌
- Regime method: 71 (2%) → 100% accurate ✓

**Stable isotopes**:
- Poor screen catches: 196 (77%) → 0% accurate (all wrong!) ❌
- Rich screen catches: 48 (19%) → 0% accurate (all wrong!) ❌
- Regime method: 9 (4%) → 11% accurate

**Problem**:
1. Screening is too aggressive (catches 96% of stable isotopes)
2. Most stable isotopes are ABOVE poor curve (neutron-rich from stability)
3. Screening predicts them as β⁺ → all wrong!

---

## 4. Why High Thresholds Work

### Threshold = 10 Z Example:

**Unstable isotopes**:
- Poor screen catches: 13 (0.4%) → 100% accurate ✓
- Rich screen catches: 0 (0%) → N/A
- Regime method: 3,291 (99.6%) → ~100% accurate ✓

**Stable isotopes**:
- Poor screen catches: 0 (0%)
- Rich screen catches: 0 (0%)
- Regime method: 253 (100%) → 31% accurate (current baseline)

**Success**:
1. Only EXTREME outliers get screened
2. Nearly all isotopes use full regime method
3. Screening adds minimal benefit (only 13 isotopes)
4. But doesn't hurt either!

---

## 5. The Stable Isotope Problem

### Why Stable Isotopes Get Caught:

Stable isotopes have mean distances:
- **+0.85 Z** to charge_nominal curve (their assigned regime)
- **+3.1 Z** to charge_poor curve (on average)
- **-2.8 Z** to charge_rich curve (on average)

**With threshold = 3 Z**:
- ~77% of stable isotopes are > 3 Z above poor curve
- ~19% of stable isotopes are < -3 Z below rich curve
- → 96% get screened and misclassified!

**With threshold = 10 Z**:
- ~0% of stable isotopes meet the threshold
- All go to regime method → current 31% recall maintained

---

## 6. Accuracy Breakdown

### Threshold = 8 Z (Recommended Compromise):

#### Unstable Isotopes:

| Stage | Count | Accuracy | Contribution |
|-------|-------|----------|--------------|
| Poor screen (β⁺) | 799 (24%) | ~70% | 559 correct |
| Rich screen (β⁻) | 693 (21%) | ~90% | 624 correct |
| Regime method | 1,812 (55%) | ~95% | 1,727 correct |
| **Total** | **3,304** | **88%** | **2,910 correct** |

#### Stable Isotopes:

| Stage | Count | Accuracy |
|-------|-------|----------|
| Poor screen | 0 | N/A |
| Rich screen | 0 | N/A |
| Regime method | 253 | 31% (79 correct) |

#### Overall:
- **Total correct**: 2,910 + 79 = 2,989 / 3,557
- **Accuracy**: 84.0%
- **vs Current**: 95.1% (-11.1% points)

---

## 7. Why Sequential Doesn't Beat Current

### Current Method Already Does This!

The current regime assignment method:
1. Calculates stress to ALL three curves
2. Assigns to regime with minimum stress
3. Uses regime-specific stability criterion

This **implicitly** implements the screening:
- Isotopes very far from nominal → assigned to poor/rich regime
- Decay prediction uses assigned regime's backbone
- Effectively a "soft" version of sequential screening

### Sequential is a "Hard" Version:

Sequential screening uses **hard cutoffs** (thresholds), while current method uses **soft assignment** (minimum stress).

**Result**: Hard cutoffs perform worse unless threshold is very high (>8 Z), at which point they converge to current method.

---

## 8. Combinatorial Benefit Analysis

### Computational Complexity:

**Sequential (threshold = 8 Z)**:
- Step 1: Calculate dist_poor, dist_rich → O(N)
- Step 2: Screen 24% + 21% = 45% → cheap prediction
- Step 3: Regime method on 55% → expensive

**Current**:
- Calculate stress to all 3 curves → O(3N) = O(N)
- Regime assignment → O(N)
- No savings

**Conclusion**: Sequential offers **no computational benefit** because:
1. Still need to calculate distances to curves (same cost)
2. Regime method is already O(N) (not expensive)
3. Thresholding adds overhead

---

## 9. When Sequential MIGHT Help

### Use Case 1: Real-Time Prediction

If predicting ONE isotope at a time:
```python
if dist_poor > 8:
    return "beta_plus"  # Skip expensive regime calculation
elif dist_rich < -8:
    return "beta_minus"
else:
    return regime_method(isotope)
```

**Benefit**: Saves ~45% of regime calculations (at threshold = 8 Z)

### Use Case 2: Explainability

Sequential provides clearer decision path:
```
"This isotope is 12 Z above the poor curve,
 strongly suggesting beta-plus decay."
```

vs current:
```
"This isotope has minimum stress in charge-poor regime,
 predicting beta-plus decay."
```

**Benefit**: More intuitive for physicists

### Use Case 3: Confidence Scores

Sequential naturally provides confidence levels:
```
If dist_poor > 10 Z: "Very confident β⁺"
If dist_poor > 5 Z:  "Confident β⁺"
If -5 < dist_poor < 5: "Ambiguous, use regime method"
```

**Benefit**: Uncertainty quantification

---

## 10. Recommendations

### ❌ Do NOT use sequential screening for batch prediction

**Reasons**:
1. Lower accuracy than current method (84% vs 95%)
2. No computational benefit
3. More complex code (two thresholds to tune)
4. Threshold selection is non-trivial (need 8-10 Z)

### ✅ DO consider sequential for:

1. **Single-isotope queries** (saves 45% of regime calculations)
2. **Explainability** (clearer physical interpretation)
3. **Confidence scoring** (uncertainty quantification)
4. **Educational visualization** (shows bracket structure)

### Optimal Configuration (if used):

```python
THRESHOLD_POOR = 8.0  # Z units
THRESHOLD_RICH = 8.0  # Z units

def predict_with_screening(A, Z):
    dist_poor = Z - Q_poor(A)
    dist_rich = Z - Q_rich(A)

    if dist_poor > THRESHOLD_POOR:
        return {"mode": "beta_plus", "confidence": "high", "screen": "poor"}
    elif dist_rich < -THRESHOLD_RICH:
        return {"mode": "beta_minus", "confidence": "high", "screen": "rich"}
    else:
        mode = regime_method(A, Z)
        return {"mode": mode, "confidence": "medium", "screen": "regime"}
```

---

## 11. The Paradox Explained

### Why doesn't sequential screening help?

**Expected**: Curves have complementary strengths (poor 100% for β⁺, rich 93.5% for β⁻)

**Reality**: The current regime method ALREADY exploits this!

When an isotope has:
- High dist_poor → low stress in poor regime → assigned to poor → predicts β⁺
- Low dist_rich → low stress in rich regime → assigned to rich → predicts β⁻

**The regime assignment IS the screening**, just done via stress minimization instead of threshold comparison.

### The Overlap Issue:

63% of isotopes satisfy both conditions (above poor AND below rich). For these:
- Sequential: Apply first matching condition (arbitrary priority)
- Current: Use minimum stress across ALL curves (optimal)

**Result**: Current method handles overlaps better.

---

## 12. Threshold Sensitivity

| Threshold | Unstable→Poor | Unstable→Rich | Stable→Poor | Stable→Rich | Overall Acc |
|-----------|---------------|---------------|-------------|-------------|-------------|
| 0 Z | 2534 (77%) | 2757 (83%) | 253 (100%) | 253 (100%) | 63.4% |
| 3 Z | 1717 (52%) | 1516 (46%) | 196 (77%) | 48 (19%) | 55.4% |
| 5 Z | 1298 (39%) | 1107 (33%) | 69 (27%) | 12 (5%) | 67.1% |
| 8 Z | 799 (24%) | 693 (21%) | 0 (0%) | 0 (0%) | 84.0% |
| 10 Z | 13 (0.4%) | 0 (0%) | 0 (0%) | 0 (0%) | 94.7% |

**Optimal range**: 8-10 Z

**Trade-off**:
- Lower threshold → more screening → more errors on stable isotopes
- Higher threshold → less screening → converges to current method

---

## 13. Conclusions

### Key Findings:

1. **Sequential screening requires HIGH thresholds** (>8 Z) to match current performance
2. **At optimal threshold**, sequential is nearly identical to current method
3. **Combinatorial benefit is minimal** for batch predictions
4. **Current regime method already exploits curve complementarity**
5. **The 63% overlap zone** is handled better by soft assignment than hard thresholds

### Bottom Line:

**For production use**: Stick with current regime method (95.1% accuracy)

**For special use cases**: Consider sequential with threshold ≥ 8 Z for:
- Real-time single predictions (computational savings)
- Explainability (clearer physical interpretation)
- Confidence scoring (uncertainty quantification)

### The Insight:

The three curves form a **bracket system**, but the current regime assignment already exploits this through stress minimization. Sequential screening is a "hard" version of what the current method does "softly" - and soft is better for overlapping zones.

---

**Recommendation**: Keep current regime method, but add confidence scores based on distance to curves for interpretability.

---

**Date**: 2025-12-29
**Analysis**: Ground states, 3,557 isotopes
**Current Accuracy**: 95.1%
**Sequential (thresh=8)**: 84.0%
**Conclusion**: Current method is optimal
