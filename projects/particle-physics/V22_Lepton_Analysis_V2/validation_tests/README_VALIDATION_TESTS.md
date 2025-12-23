# V22 Hill Vortex Hardening Test Suite

## Overview

This directory contains critical validation tests to move the V22 Hill vortex lepton mass investigation from "fits exist" to "publication ready." These tests address the key limitations identified in the current implementation.

## Quick Start

```bash
cd /home/tracy/development/QFD_SpectralGap/V22_Lepton_Analysis/validation_tests
./run_all_hardening_tests.sh
```

Expected runtime: **20-30 minutes** total

Results will be saved in `./results/` with JSON data and a summary report.

---

## Why These Tests Matter

The current V22 implementation demonstrates that:
> "For β = 3.1, the Hill vortex + parabolic density ansatz admits solutions matching all three lepton mass ratios."

However, this is a **fit with 3 degrees of freedom → 1 target**, not a prediction. Before publication, we must establish:

1. **Numerical robustness** - Results don't change with grid refinement
2. **Solution uniqueness** - Not many arbitrary local minima
3. **Physical robustness** - β = 3.1 isn't fine-tuned to one specific functional form

These tests provide that assurance.

---

## Test 1: Grid Convergence (`test_01_grid_convergence.py`)

### Purpose
Verify that fitted parameters (R, U, amplitude) and energies converge as the integration grid is refined.

### What It Does
Runs optimization at four grid resolutions:
- **Coarse**:   (nr, nθ) = (50, 10)
- **Standard**: (nr, nθ) = (100, 20)  ← current implementation
- **Fine**:     (nr, nθ) = (200, 40)
- **Very Fine**: (nr, nθ) = (400, 80)

Uses Simpson's rule for spherical integration over (r, θ).

### Success Criteria
✓ **Parameter drift < 1%** between Fine and Very Fine
✓ **Energy drift < 0.1%** between Fine and Very Fine
✓ **Monotonic convergence** (errors decrease with refinement)

### What It Tells Us
- If parameters drift significantly → Current grid too coarse, results unreliable
- If converged → Numerical integration is sufficiently accurate for publication
- Convergence rate → Guides choice of production grid resolution

### Expected Outcome
Parameters should be stable to ~0.1% by (200, 40) grid. If they're not, we need adaptive quadrature or higher-order methods.

### Output
- `results/grid_convergence_results.json` - Full data
- Convergence tables showing drift from finest grid
- Pass/fail on success criteria

---

## Test 2: Multi-Start Robustness (`test_02_multistart_robustness.py`)

### Purpose
Determine whether the fitted solution is **unique** or whether multiple local minima exist in parameter space.

### What It Does
Runs 50 independent optimizations from random initial seeds:
- **R** sampled uniformly in [0.2, 0.8]
- **U** sampled uniformly in [0.01, 0.10]
- **amplitude** sampled uniformly in [0.5, 1.0]

Each run uses Nelder-Mead to minimize `(E_total - m_target)²`.

### Success Criteria
✓ **Solution unique**: Coefficient of variation CV < 1% for all parameters
✓ **Residuals consistent**: All converged solutions within 0.01% of target
✓ **High convergence rate**: > 90% of runs converge successfully

### What It Tells Us

**Scenario 1: Single tight cluster**
- All runs converge to (R, U, amplitude) within ~1%
- **Interpretation**: Solution is unique (within numerical tolerance)
- **Action**: Proceed to publication, solution is well-defined

**Scenario 2: Multiple distinct clusters**
- Runs converge to 2-3 different solutions
- **Interpretation**: Multiple local minima exist
- **Action**: Need selection principle:
  - Lowest residual?
  - Stability analysis (second variation)?
  - Physical constraint (e.g., amplitude closest to cavitation)?

**Scenario 3: Wide scatter**
- Large CV (> 5%) across runs
- **Interpretation**: Shallow objective function, weak constraints
- **Action**: Need tighter physical constraints to reduce degeneracy

### Expected Outcome
Likely **Scenario 1** - single cluster near (R=0.44, U=0.024, amp=0.90) for electron. Would validate that the solution is well-determined by the physics, not an arbitrary local minimum.

### Output
- `results/multistart_robustness_results.json` - All 50 solutions
- Cluster analysis (if multiple clusters found)
- Statistical summary (mean, std, CV for each parameter)
- Pass/fail on uniqueness criteria

---

## Test 3: Profile Sensitivity (`test_03_profile_sensitivity.py`)

### Purpose
Test whether **β = 3.1 is robust** to the choice of density profile or whether the parabolic form is essential.

### What It Does
Runs optimization for four different density depression profiles:

1. **Parabolic** (current): `δρ = -a(1 - r²/R²)`
2. **Quartic core**: `δρ = -a(1 - r²/R²)²`  (sharper depression)
3. **Gaussian core**: `δρ = -a exp(-r²/R²)`  (smooth falloff)
4. **Linear**: `δρ = -a(1 - r/R)`  (gentle slope)

For each profile, **β is held fixed at 3.1** (not retuned). Only (R, U, amplitude) are optimized.

### Success Criteria
✓ **β is robust**: All profiles achieve residual < 0.01 with β = 3.1
✓ **Parameters reasonable**: All solutions give physically sensible (R, U, amplitude)
✓ **E_stabilization consistent**: E_stab remains O(0.1-1) across profiles

### What It Tells Us

**Outcome A: All profiles work**
- β = 3.1 produces electron mass across different density shapes
- **Interpretation**: β is a **universal stiffness parameter**, profile shape is secondary
- **Implication**: Strengthens "β unifies physics" narrative
- **Action**: Proceed with confidence in β = 3.1

**Outcome B: Only parabolic works**
- Other profiles fail to converge or give large residuals
- **Interpretation**: Parabolic density gradient is **part of the physics**
- **Implication**: Need to derive why parabolic from first principles (e.g., Euler-Lagrange extremum)
- **Action**: Add derivation to paper, less flexible but more constrained

**Outcome C: Partial success (e.g., parabolic + quartic work, others fail)**
- **Interpretation**: Constraint on functional form but not unique
- **Action**: Investigate common features of working profiles

### Expected Outcome
Likely **Outcome A** or **C**. Since E_stab = ∫ β(δρ)² dV, different profiles give different integrals, but optimization should compensate via (R, U, amplitude) if β is truly fundamental.

If only parabolic works (**Outcome B**), it's still publishable but requires deriving the parabolic form from variational principles.

### Output
- `results/profile_sensitivity_results.json` - Results for all 4 profiles
- Comparison table showing parameters and residuals
- Variation from parabolic baseline
- Interpretation of β robustness

---

## Running Individual Tests

You can run tests individually:

```bash
# Test 1: Grid convergence (~5-10 min)
python3 test_01_grid_convergence.py

# Test 2: Multi-start robustness (~10-15 min)
python3 test_02_multistart_robustness.py

# Test 3: Profile sensitivity (~5 min)
python3 test_03_profile_sensitivity.py
```

---

## Interpreting Results

### For Publication Readiness

**Minimum acceptable outcomes:**
1. **Grid convergence**: Parameters stable to < 1% at (200, 40) grid
2. **Multi-start**: Single dominant cluster with CV < 2%
3. **Profile sensitivity**: At least parabolic + one other profile work

**Ideal outcomes:**
1. **Grid convergence**: Parameters stable to < 0.1% at (200, 40) grid
2. **Multi-start**: Single tight cluster with CV < 1%
3. **Profile sensitivity**: All four profiles work with β = 3.1

**Red flags (require further work):**
- Grid convergence fails (parameters drift > 5%)
- Multiple solution clusters with comparable residuals
- Only parabolic profile works AND no variational derivation available

### What Happens After Tests

**If all tests pass:**
1. Update `PUBLICATION_READY_RESULTS.md` with validation confirmation
2. Remove "⚠️" warnings from limitations section
3. Add "Numerical Validation" subsection to paper
4. Proceed to writing formal publication

**If tests reveal issues:**
1. **Grid convergence fails** → Implement adaptive quadrature or spectral methods
2. **Multiple solutions** → Add stability analysis (second variation of action)
3. **Profile-dependent** → Derive parabolic form from Euler-Lagrange equations

---

## Implementation Details

### Shared Components

All tests use identical core physics:
- `HillVortexStreamFunction` - Velocity field from Hill's solution
- `DensityGradient` - Density perturbation (profile-dependent in Test 3)
- Energy integrals:
  - E_circulation = ∫ (1/2) ρ(r) v² dV
  - E_stabilization = ∫ β (δρ)² dV
  - E_total = E_circ - E_stab

### Optimization Method

All tests use **Nelder-Mead** (derivative-free simplex):
- Robust to noisy objectives (numerical integration)
- No gradient calculations needed
- Tolerances: `xatol=1e-8`, `fatol=1e-8`
- Max iterations: 2000

**Why not other methods?**
- Differential Evolution: Tested in initial work, slower
- BFGS/L-BFGS: Requires smooth gradients (integration noise problematic)
- Basin-hopping: Overkill for Test 2 which does multi-start explicitly

### Output Format

All tests produce:
1. **JSON results file** - Machine-readable data for further analysis
2. **Console output** - Human-readable summary with pass/fail
3. **Consistent schema** - Easy to compare across tests

---

## Next Steps After Hardening Tests

### Immediate (If Tests Pass)

1. **Run tests on muon and tau** - Extend grid convergence to all three leptons
2. **Analytic E_stab validation** - Compare numerical ∫ β(δρ)² dV to closed form
3. **Document U interpretation** - Address U > 1 issue (superluminal?)

### Near-Term

4. **Stability analysis** - Second variation of action δ²E/δψ² > 0
5. **Cavitation quantization** - Implement amplitude → ρ_vac as constraint (removes 1 DOF)
6. **Toroidal components** - Add (ψ_b0, ψ_b1, ψ_b2) for full 4-component structure

### Long-Term

7. **Excited states** - Predict lepton spectrum from mode quantization
8. **Anomalous magnetic moments** - Test g-2 predictions
9. **Cross-scale β unification** - Rigorous unit mapping cosmology ↔ nuclear ↔ particle

---

## Troubleshooting

### Test 1 fails (parameters drift > 1%)

**Possible causes:**
- Simpson's rule insufficient for sharp features
- Angular grid too coarse (try more θ points first, cheaper than r)
- Singularities near r=0 or θ=0,π (check boundary handling)

**Solutions:**
- Try (nr, nθ) = (100, 40) - refine θ before r
- Switch to adaptive quadrature (scipy.integrate.quad)
- Implement spectral methods for smooth functions

### Test 2 finds multiple clusters

**This is not necessarily a failure!** It means:
- Parameter space has structure
- Need physical principle to select correct solution

**Next steps:**
1. Compare residuals - does one cluster have lower error?
2. Compare E_stab - which is closer to analytic prediction?
3. Compare amplitude - which is closer to cavitation (amplitude → ρ_vac)?
4. Perform stability analysis - which has δ²E > 0?

### Test 3: Only parabolic works

**This is publishable** if you can justify it theoretically:
1. Derive parabolic from Euler-Lagrange: δE/δρ = 0
2. Show it minimizes action for Hill vortex geometry
3. Connect to fluid dynamics (Lamb 1932 used parabolic)

**Don't just assert** "we tried parabolic and it worked" - need first-principles derivation.

---

## File Structure

```
validation_tests/
├── README_VALIDATION_TESTS.md          (this file)
├── run_all_hardening_tests.sh          (master runner)
├── test_01_grid_convergence.py         (Test 1)
├── test_02_multistart_robustness.py    (Test 2)
├── test_03_profile_sensitivity.py      (Test 3)
└── results/
    ├── grid_convergence_results.json
    ├── multistart_robustness_results.json
    ├── profile_sensitivity_results.json
    └── hardening_tests_summary.txt
```

---

## References

**Numerical methods:**
- Simpson's rule: scipy.integrate.simps
- Nelder-Mead: scipy.optimize.minimize

**Physics:**
- Hill's spherical vortex: M.J.M. Hill (1894), Phil. Trans. R. Soc. Lond. A
- Lamb's Hydrodynamics (1932), §§159-160
- QFD Electron formal spec: `/projects/Lean4/QFD/Electron/HillVortex.lean`

**Prior work:**
- Initial electron fit: `v22_hill_vortex_with_density_gradient.py`
- Muon fit: `v22_muon_refined_search.py`
- Tau fit: `v22_tau_test.py`
- Current results: `PUBLICATION_READY_RESULTS.md`

---

## Contact / Questions

For questions about:
- **Physics interpretation**: Review `PUBLICATION_READY_RESULTS.md` limitations section
- **Numerical methods**: Check scipy documentation for integration/optimization
- **Test failures**: See Troubleshooting section above
- **Next steps**: See "Next Steps After Hardening Tests" section

These tests are designed to be **defensive** - they identify problems before reviewers do. Passing all tests doesn't guarantee publication acceptance, but failing any of them likely means more work is needed.

**Better to find issues now than in peer review.**
