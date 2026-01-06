# Overnight Optimization Analysis
**Date:** 2025-12-31
**Runtime:** 2.3 hours (8,203 seconds)
**Evaluations:** 1,090 / 2,700 (converged early)

---

## Executive Summary

❌ **The optimization FAILED and made results significantly worse.**

- Initial parameters: Produce binding energies with -1.44% to +0.51% error, but high virial (non-physical)
- Optimized parameters: **Solver fails completely** - returns only constituent masses (+0.81% to +0.91% error)
- The optimizer converged to parameter space where the QFD solver cannot find ANY solutions

---

## Detailed Comparison

### Initial Parameters (Before Optimization)

| Isotope | E_model (MeV) | Virial | Converged? | Pred vs Exp | Error (%) |
|---------|---------------|--------|------------|-------------|-----------|
| Sn-120  | -2,633.7     | 44.95  | ✗          | 110,075 vs 111,688 | -1.44 |
| Au-197  | -2,612.0     | 166.84 | ✗          | 182,421 vs 183,473 | -0.57 |
| Hg-200  | -1,635.9     | 808.31 | ✗          | 186,215 vs 186,269 | -0.03 |
| Pb-206  | -2,379.8     | 43.99  | ✗          | 191,107 vs 191,864 | -0.39 |
| Pb-207  | -3,150.4     | 743.95 | ✗          | 191,276 vs 192,797 | -0.79 |
| Pb-208  | -3,230.9     | 802.95 | ✗          | 192,135 vs 193,729 | -0.82 |
| U-235   | -1,340.2     | 1563.39| ✗          | 219,386 vs 218,942 | +0.20 |
| U-238   | -675.8       | 2011.83| ✗          | 222,869 vs 221,743 | +0.51 |

**Statistics:**
- Mean error: -0.42%
- RMS error: 0.75%
- Converged solutions: 0/8 (all have virial > 0.18)
- Loss: 2,093,188

### Optimized Parameters (After Overnight Run)

| Isotope | E_model (MeV) | Virial | Converged? | Pred vs Exp | Error (%) |
|---------|---------------|--------|------------|-------------|-----------|
| Sn-120  | **0.0** ❌   | 999.0  | ✗          | 112,709 vs 111,688 | +0.91 |
| Au-197  | **0.0** ❌   | 999.0  | ✗          | 185,033 vs 183,473 | +0.85 |
| Hg-200  | **0.0** ❌   | 999.0  | ✗          | 187,850 vs 186,269 | +0.85 |
| Pb-206  | **0.0** ❌   | 999.0  | ✗          | 193,486 vs 191,864 | +0.85 |
| Pb-207  | **0.0** ❌   | 999.0  | ✗          | 194,426 vs 192,797 | +0.84 |
| Pb-208  | **0.0** ❌   | 999.0  | ✗          | 195,365 vs 193,729 | +0.84 |
| U-235   | **0.0** ❌   | 999.0  | ✗          | 220,726 vs 218,942 | +0.81 |
| U-238   | **0.0** ❌   | 999.0  | ✗          | 223,545 vs 221,743 | +0.81 |

**Statistics:**
- Mean error: +0.85%
- RMS error: 0.85%
- Converged solutions: 0/8 (all solvers FAILED)
- Loss: 1,000 (penalty floor)
- **E_model = 0 means solver returned NO interaction energy**

---

## What Went Wrong?

### Root Cause: Loss Function Design Flaw

The loss function penalizes high virial values heavily:
```python
if virial > 0.18:
    virial_penalty = 4.0 * (virial - 0.18) ** 2
```

The optimizer tried to avoid high virial by:
1. Moving parameters away from regions with virial > 0.18
2. But inadvertently moved INTO regions where the solver can't converge at all
3. Failed solver → E_model = 0, virial = 999 → loss = 1000 (penalty floor)

**The paradox:**
- High virial (bad physics) → loss ~ 2,000,000
- Complete solver failure (no physics) → loss = 1,000

The optimizer "improved" the loss by making the solver fail completely!

---

## Parameter Changes

| Parameter      | Initial  | Optimized | Change  |
|----------------|----------|-----------|---------|
| c_v2_base      | 2.202    | 2.513     | +14.1%  |
| c_v2_iso       | 0.027    | 0.030     | +10.5%  |
| c_v2_mass      | -0.00021 | -0.00006  | -68.8%  |
| c_v4_base      | 5.282    | 5.275     | -0.1%   |
| c_v4_size      | -0.085   | -0.091    | +7.5%   |
| alpha_e_scale  | 1.007    | 0.989     | -1.8%   |
| beta_e_scale   | 0.504    | 0.552     | +9.4%   |
| c_sym          | 25.0     | 25.7      | +2.8%   |
| kappa_rho      | 0.030    | 0.030     | +0.4%   |

Notable: `c_v2_mass` changed by -68.8%, potentially destabilizing the solver.

---

## Recommendations

### 1. Immediate: **REVERT to Initial Parameters**
The initial parameters produce physically meaningful (though not converged) solutions.

### 2. Fix the Loss Function
Current penalty structure is counterproductive. Options:

**Option A: Reject failures entirely**
```python
if E_model == 0 or virial == 999:
    return 1e9  # Much worse than high virial
```

**Option B: Softer virial penalty**
```python
if virial > 0.18:
    virial_penalty = 0.1 * (virial - 0.18)  # Linear, not quadratic
```

**Option C: Multi-stage optimization**
1. Stage 1: Optimize for binding energy accuracy ONLY (ignore virial)
2. Stage 2: Fine-tune for virial convergence with tighter SCF settings

### 3. Increase SCF Iterations
Current: 150 iterations (fast mode)
Try: 360 iterations (accurate mode) to help virial convergence

### 4. Narrow Parameter Bounds
The optimizer explored too broadly. Constrain around initial values:
- c_v2_base: 2.2 ± 10% (not ± 15%)
- c_v2_mass: Keep close to initial value (don't let it approach zero)

### 5. Try Different Optimizer
Differential evolution is global but can be erratic. Consider:
- Nelder-Mead (simplex) for local refinement
- Basin-hopping with tighter bounds
- Multi-start local optimization from initial parameters

---

## The Virial Problem

**Core issue:** No solutions achieve virial < 0.18, even with initial parameters.

Possible causes:
1. **Grid resolution too coarse** (32 points may be insufficient for heavy nuclei)
2. **SCF iterations too few** (150 may not be enough for convergence)
3. **Physics model limitations** (unified density model may need refinement for A > 120)
4. **Parameter space issue** (no physical solution exists in explored region)

**Test:** Run a single isotope (e.g., Pb-208) with:
- grid = 64 (instead of 32)
- iters = 500 (instead of 150)
- Check if virial can get below 0.18

---

## Conclusion

The overnight optimization was a **technical success** (ran smoothly, used resources well) but a **scientific failure** (optimized parameters are worse than initial).

**Next steps:**
1. Revert to initial parameters
2. Investigate why NO solutions achieve virial < 0.18
3. Either fix solver settings OR relax virial criterion
4. Redesign loss function to avoid catastrophic optimization failures

The 99.95% "improvement" in loss was illusory - the optimizer gamed the penalty structure by making the solver fail.
