# QFD Lepton Mass Investigation (V22)

**Status**: Numerical consistency tests completed (December 2025)

---

## Overview

This repository contains numerical evidence that a vacuum stiffness parameter β ≈ 3.043233053, inferred from the fine structure constant through a conjectured relationship, supports Hill vortex solutions matching the charged lepton mass ratios (electron, muon, tau) to better than 10⁻⁷ relative precision.

## Key Result

For **fixed β = 3.043233053** (derived from α = 1/137.036 via conjectured identity):

| Lepton | Target m/m_e | Achieved m/m_e | Residual | Parameters (R, U, amp) |
|--------|--------------|----------------|----------|------------------------|
| Electron | 1.000 | 1.000000000 | 5×10⁻¹¹ | (0.439, 0.024, 0.911) |
| Muon | 206.768 | 206.76828266 | 6×10⁻⁸ | (0.450, 0.315, 0.966) |
| Tau | 3477.228 | 3477.22800 | 2×10⁻⁷ | (0.493, 1.289, 0.959) |

**Same β for all three leptons. No adjustment of stiffness parameter.**

## What This Represents

### Current Status: Consistency Test

- **Three geometric degrees of freedom** (R, U, amplitude) are optimized per lepton
- **One target per lepton**: Mass ratio m_lepton/m_electron
- **Result**: Hill vortex solutions exist and are numerically robust

This demonstrates **existence and robustness**, not yet **unique prediction**.

### Physical Interpretation

- Leptons modeled as Hill spherical vortices (M.J.M. Hill 1894, H. Lamb 1932)
- Mass arises from geometric cancellation:
  - E_total = E_circulation (kinetic) - E_stabilization (potential energy from vacuum stiffness)
  - Small residual (0.2-0.3 MeV scale) yields observed masses
- Mass hierarchy emerges from circulation velocity scaling: U ∝ √m (observed to ~10%)

### β Convergence Across Sectors

| Source | β Value | Uncertainty | Method |
|--------|---------|-------------|--------|
| Fine structure constant (this work) | 3.043233053 | ±0.012 | Conjectured identity with (α, c₁, c₂) |
| Nuclear stability (prior work) | 3.1 | ±0.05 | Direct fit to binding energies |
| Cosmology (prior work) | 3.0-3.2 | — | Dark energy EOS interpretation |

**Interpretation**: Cross-sector overlap supports universal vacuum stiffness hypothesis.

**Caveat**: Identity between α and β is conjectured, not derived. Falsifiable via independent β measurements.

---

## Repository Contents

### Code (Validated)

- `validation_tests/test_all_leptons_beta_from_alpha.py` - Main replication script
- `validation_tests/test_01_grid_convergence.py` - Numerical stability test
- `validation_tests/test_02_multistart_robustness.py` - Solution uniqueness test
- `validation_tests/test_03_profile_sensitivity.py` - Functional form robustness test
- `integration_attempts/` - Development history (various solvers tested)

### Results (Reproducible)

- `validation_tests/results/three_leptons_beta_from_alpha.json` - Main results
- `validation_tests/results/grid_convergence_results.json` - Convergence data
- `validation_tests/results/profile_sensitivity_results.json` - Robustness tests

### Documentation

- `REPLICATION_ASSESSMENT.md` - Independent verification and critical assessment
- `EXECUTIVE_SUMMARY_GOLDEN_LOOP_REVISED.md` - Reviewer-proofed summary ⭐ **Read this first**
- `validation_tests/README_VALIDATION_TESTS.md` - Test suite description
- `INVESTIGATION_INDEX.md` - Complete file navigation

---

## Replication Instructions

### Prerequisites

```bash
# Python 3.8+ with scientific stack
pip install numpy scipy
```

### Quick Start

```bash
cd validation_tests
python3 test_all_leptons_beta_from_alpha.py
```

**Expected runtime**: ~20-30 seconds
**Expected output**: All three leptons converge with residuals < 10⁻⁷

### Validation Tests

```bash
# Grid convergence (5-10 min)
python3 test_01_grid_convergence.py

# Multi-start robustness (10-15 min)
python3 test_02_multistart_robustness.py

# Profile sensitivity (5 min)
python3 test_03_profile_sensitivity.py
```

**All tests passed** as of 2025-12-22:
- ✓ Grid convergence: Parameters stable to 0.8% at production resolution
- ✓ Multi-start: Single solution cluster (CV < 1% for parameters)
- ✓ Profile sensitivity: All 4 functional forms (parabolic, quartic, Gaussian, linear) work with β = 3.1

---

## Known Limitations

### 1. Solution Degeneracy

**Issue**: Three geometric parameters (R, U, amplitude) are optimized to match one observable (mass ratio) per lepton.

**Implication**: A 2-dimensional solution manifold exists for each lepton. Current solutions are not uniquely determined without additional constraints.

**Proposed resolution**:
- Implement cavitation saturation: amplitude → ρ_vac (removes 1 DOF)
- Add charge radius constraint: r_rms = 0.84 fm for electron (removes 1 DOF)
- Apply dynamical stability: δ²E > 0 (selects among remaining solutions)

**Status**: Under investigation

### 2. Lack of Independent Observable Tests

**Current**: Only mass ratios are fit
**Needed**: Predictions for:
- Charge radii (r_e, r_μ, r_τ)
- Anomalous magnetic moments (g-2 values)
- Form factors F(q²) from scattering data

Without independent tests, this remains a **consistency demonstration**, not a **validated prediction**.

### 3. Conjectured β from α Identity

**Current claim**: β derived from fine structure constant via relation involving nuclear binding coefficients (c₁, c₂)

**Status**: Empirical relation, not yet derived from first principles

**Falsifiability**: If improved measurements of (c₁, c₂) or independent β determinations fall outside overlap region, identity is ruled out.

### 4. U > 1 Interpretation

**Observation**: For tau, circulation velocity U = 1.29 in units where c = 1

**Question**: Is this superluminal? Or does U represent:
- Circulation in vortex rest frame (boosted in lab)?
- Dimensionless internal circulation (not real-space velocity)?
- Requires clarification of physical interpretation

---

## Theoretical Background

### Hill's Spherical Vortex

Classical fluid dynamics solution (1894):
- Spherical region of radius R with internal rotational flow
- External irrotational (potential) flow
- Boundary conditions: Continuous velocity and pressure at r = R

**Stream function**:
```
ψ(r,θ) = -(3U/2R²)(R² - r²)r² sin²θ    (r < R, internal)
       = (U/2)(r² - R³/r) sin²θ         (r ≥ R, external)
```

**Lean specification**: `projects/Lean4/QFD/Electron/HillVortex.lean` (136 lines, 0 sorry)

### Density Gradient Ansatz

**Not a hard shell**, but smooth density depression:
```
ρ(r) = ρ_vac - amplitude·(1 - r²/R²)   (r < R)
     = ρ_vac                            (r ≥ R)
```

**Cavitation constraint**: ρ ≥ 0 everywhere → amplitude ≤ ρ_vac

**Claim**: This constraint yields charge quantization (proven in Lean)

### Energy Functional

```
E_total = E_circulation - E_stabilization

E_circulation = ∫ (1/2) ρ(r) v²(r,θ) dV

E_stabilization = ∫ β (δρ)² dV
```

**Key insight**: Small residual from near-perfect cancellation yields observed lepton masses.

**β parameter**: Vacuum stiffness (resistance to density perturbations)

---

## Results Summary

### Numerical Robustness

| Test | Criterion | Result | Pass |
|------|-----------|--------|------|
| Grid convergence | Max parameter drift < 1% at (200,40) | 0.4% | ✓ |
| Multi-start (50 runs) | Solution CV < 1% | 0.8% | ✓ |
| Profile sensitivity | All 4 profiles work with β = 3.1 | 4/4 | ✓ |

### Scaling Law

Circulation velocity U follows approximate √m scaling:

| Particle | √(m/m_e) | U/U_e | Deviation |
|----------|----------|-------|-----------|
| Electron | 1.00 | 1.00 | — |
| Muon | 14.38 | 13.08 | -9% |
| Tau | 58.96 | 53.64 | -9% |

**Interpretation**: Weak dependence on (R, amplitude) causes systematic ~10% deviation from pure √m scaling.

### Cross-Sector β Consistency

All three β determinations overlap within 1σ uncertainties:
- **Particle scale** (this work): 3.043233053 ± 0.012
- **Nuclear scale** (prior): 3.1 ± 0.05
- **Cosmological scale** (prior): 3.0-3.2

**Hypothesis**: Same physical principle (vacuum stiffness) manifests consistently across 26 orders of magnitude in length scale.

---

## What We Can Claim (Defensibly)

### ✓ Appropriate Claims

1. "For β = 3.043233053 inferred from α, Hill vortex solutions exist that reproduce lepton mass ratios to < 10⁻⁷ relative error"

2. "Once β is fixed by α-identity, no additional coupling constants are adjusted between leptons"

3. "Circulation velocity U scales approximately as √m, with ~10% deviations attributable to weak geometric parameter dependence"

4. "β values from α-identity, nuclear fits, and cosmology overlap within uncertainties, suggesting cross-sector consistency"

5. "Solutions are numerically robust: grid-converged, multi-start stable, and profile-insensitive"

### ⚠️ Requires Qualification

1. "No free parameters" → Qualify: "No adjusted coupling constants (3 geometric DOFs per lepton optimized)"

2. "Prediction" → Use only after implementing independent observable tests (charge radius, g-2, etc.)

3. "Universal β" → Qualify: "Hypothesis supported by cross-sector overlap; requires further testing"

### ✗ Avoid

1. "100% accuracy" → Use "Residual < 10⁻⁷" (more precise and honest)

2. "Complete unification" → Premature; use "Cross-sector consistency"

3. "Proof" → Use "Consistency test" or "Supporting evidence"

4. Comparisons to Maxwell/Einstein → Inappropriate at this stage

---

## Next Steps

### Immediate (Critical for Publication)

1. **Implement constraint solver**:
   - Fix amplitude = 0.99 ρ_vac (cavitation saturation)
   - Add r_rms = 0.84 fm for electron (charge radius)
   - Test if unique solution emerges (reduces from 3 DOF to 1 DOF)

2. **Address U > 1 interpretation**:
   - Clarify physical meaning of circulation velocity
   - Ensure dimensionless units properly defined

3. **Uncertainty propagation**:
   - β = 3.043233053 ± 0.012 → How do (R, U, amplitude) vary within ±1σ?
   - Systematic uncertainty from grid resolution, profile choice

### Short-Term (Strengthens Case)

4. **Independent observable predictions**:
   - Charge radii: Compare predicted r_e, r_μ, r_τ to experiment
   - Anomalous magnetic moments: Calculate (g-2) from circulation patterns
   - Form factors: Predict F(q²) and test against scattering data

5. **Derive or validate β from α**:
   - Theoretical derivation from QFD principles, OR
   - Frame as "empirical relation to be tested across sectors"

6. **Cross-validate with Phoenix solver**:
   - Phoenix uses V(ρ) = V2·ρ + V4·ρ² (different formulation, 99.9999% accuracy)
   - Test if V2, V4 can be mapped from (R, U, amplitude) solutions

### Long-Term (Full Theory)

7. **Excited states**: Predict lepton spectrum from vortex mode quantization

8. **Extend to quarks**: Test if β = 3.1 works for quark masses (different topology: Q-balls vs vortices?)

9. **Experimental tests**: Lepton substructure searches at colliders (FCC, muon collider)

---

## How to Contribute

We welcome:
- **Independent replication**: Run the code, verify results
- **Code review**: Check numerical methods, suggest improvements
- **Theoretical analysis**: Derive β from α, analyze stability, etc.
- **Extensions**: Implement constraints, predict observables, test quarks

Please open issues for:
- Replication failures or discrepancies
- Theoretical concerns or alternative interpretations
- Suggestions for additional validation tests

---

## Citation

If you use this work, please cite:

```bibtex
@software{qfd_lepton_v22_2025,
  author = {QFD Collaboration},
  title = {V22 Lepton Mass Investigation: Hill Vortex Consistency Tests},
  year = {2025},
  url = {https://github.com/qfd-project/lepton-masses-v22},
  note = {Numerical evidence for β ≈ 3.043233053 supporting lepton mass ratios}
}
```

**Preprint**: [Planned for arXiv after constraint implementation]

---

## License

[To be determined - suggest MIT or Apache 2.0 for code, CC-BY for documentation]

---

## Contact

For questions or collaboration:
- Open an issue in this repository
- See `REPLICATION_ASSESSMENT.md` for detailed technical analysis
- See `EXECUTIVE_SUMMARY_GOLDEN_LOOP_REVISED.md` for reviewer-proofed summary

---

## Acknowledgments

- M.J.M. Hill (1894): Original spherical vortex solution
- H. Lamb (1932): Hydrodynamics treatise, §§159-160
- Lean theorem prover community: Formal verification framework

---

**Status**: Ready for community feedback and independent verification

**Last Updated**: 2025-12-23

**Version**: 1.0 (Numerically validated, awaiting constraint implementation)
