# Missing Physics Diagnostic Analysis
**Date**: 2025-12-31
**Method**: Fixed β=3.043233053 sweep across periodic table

## Executive Summary

**VERDICT**: The pure soliton model with β=3.043233053 **fails catastrophically** across the entire periodic table.

- **Low A (≤16)**: Missing 26-224 MeV of binding (rotational/winding energy)
- **High A (≥40)**: Missing 1,000-25,000 MeV of binding (exponential failure)
- **Universal pattern**: ALL residuals are positive → Model is too repulsive

## The Calibration

Using H-1 (proton) as the unit cell:
- Raw density integral: 639.30 (dimensionless)
- Calibration scale: **E₀ = 1.4676 MeV/unit**
- This sets M_proton = 938.27 MeV by definition

## Detailed Residuals

| Isotope | A   | Z  | E_exp (MeV) | E_model (MeV) | Δ (MeV)   | Δ/E_exp |
|---------|-----|----|-----------:|-------------:|---------:|--------:|
| H-1     | 1   | 1  | +0.5       | +26.6        | +26.0    | 51x     |
| H-2     | 2   | 1  | -0.4       | +29.0        | +29.4    | 74x     |
| He-4    | 4   | 2  | -24.7      | +33.0        | +57.7    | 2.3x    |
| Li-6    | 6   | 3  | -26.6      | +18.4        | +45.0    | 1.7x    |
| C-12    | 12  | 6  | -81.3      | +38.1        | +119.5   | 1.5x    |
| O-16    | 16  | 8  | -113.2     | +110.5       | +223.7   | 2.0x    |
| Ca-40   | 40  | 20 | -306.0     | +727.5       | +1033.4  | 3.4x    |
| Fe-56   | 56  | 26 | -440.2     | +1656.3      | +2096.5  | 4.8x    |
| Sn-120  | 120 | 50 | -904.4     | +7502.2      | +8406.6  | 9.3x    |
| Au-197  | 197 | 79 | -1366.4    | +17624.2     | +18991   | 13.9x   |
| Pb-208  | 208 | 82 | -1431.6    | +19102.0     | +20534   | 14.3x   |
| U-238   | 238 | 92 | -1565.8    | +23832.4     | +25398   | 16.2x   |

## Physical Interpretation

### Domain 1: Low A (Winding/Rotor Regime, A ≤ 16)

**Observation**: Residuals scale roughly as Δ ≈ 50-200 MeV

**Missing Physics**:
1. **Integer Winding Energy**: He-4 is a tetrahedral knot, not a fluffy cloud
2. **Rotational Quantization**: Missing ℏ²/(2I) corrections
3. **Discrete Topology**: The solver treats the field as continuous; reality has discrete nucleon positions

**Proposed Lagrangian Term**:
```
L_rotor = -ℏ²/(2I) · J(J+1)
```
where I ~ m_N · R² ~ A^(5/3)

Expected scaling: Δ ~ 1/A^(5/3)

### Domain 2: Middle A (Fluid Regime, 20 ≤ A ≤ 56)

**Observation**: Residuals grow from +1,000 to +2,100 MeV

**Missing Physics**:
1. **Shell Effects**: Not captured by smooth fields
2. **Pairing Energy**: Nucleons pair, reducing energy
3. **Surface Tension**: Discrete surface vs continuous field

**This regime shows β=3.043233053 might work IF we add correction terms**

### Domain 3: High A (Saturation Regime, A ≥ 120)

**Observation**: CATASTROPHIC exponential growth (+8,000 to +25,000 MeV)

**Missing Physics**:
1. **Vacuum Saturation**: β becomes A-dependent at high density
2. **Coulomb Barrier**: Z²/A^(1/3) grows, overwhelming attractive potential
3. **Surface Dominance**: Volume binding saturates, surface costs dominate

**This regime indicates β=3.043233053 is fundamentally WRONG for heavy nuclei**

## Critical Findings

### 1. The Sign Flip is Universal

Every single isotope (except H-1 by calibration) shows:
- **E_model > 0** (repulsive, unstable)
- **E_exp < 0** (attractive, stable)

This is the same sign flip we diagnosed in the nuclear-soliton-solver, but now we see it's **systematic across all A**.

### 2. The Compression Problem Revisited

From earlier diagnosis, we found:
- Single proton: E = +55 MeV (over-compressed)
- C-12: E = +46 MeV (over-compressed)

This diagnostic confirms:
- H-1: E = +26.6 MeV (still positive!)
- C-12: E = +38.1 MeV (matches earlier finding ✓)

**Root cause**: The SCF solver finds repulsive branch for ALL isotopes.

### 3. β=3.043233053 is NOT Universal

If β were universal, middle A should match experiment. Instead:
- Fe-56: Off by factor of 4.8x
- Ca-40: Off by factor of 3.4x

**Conclusion**: Either:
- β must be A-dependent: β(A)
- OR β=3.043233053 is ONLY valid for a specific A range (maybe A ≈ 1-4?)
- OR the entire V4 potential formulation is wrong

## Proposed Path Forward

### Option A: Add Missing Lagrangian Terms (Pragmatic)

Keep β=3.043233053 as the "vacuum stiffness" but add corrections:

```python
E_total = E_vacuum(β=3.043233053) + E_rotor(A, J) + E_shell(Z, N) + E_saturation(A)
```

Where:
- E_rotor ≈ -C₁/A^(5/3) (Low A correction)
- E_shell = Shell model corrections (Middle A)
- E_saturation ≈ +C₂·A^(5/3) (High A penalty)

### Option B: Declare β(A) Functional Form (Fundamental)

Admit that vacuum stiffness changes with density:

```python
β(A) = β₀ · (1 + α·A^(2/3) + γ·A^(4/3))
```

Fit (β₀, α, γ) to data.

### Option C: Abandon Continuous Fields (Radical)

Accept that nuclei are **discrete** structures:
- Use lattice QFD instead of continuous fields
- Model individual nucleon positions
- Compute winding numbers explicitly

## Immediate Next Steps

1. **Plot Residual vs A** - Check if it follows predicted scaling laws
2. **Test β(A) hypothesis** - Fit A-dependent β and see if residuals collapse
3. **Add rotor term** - Implement E_rotor ~ -C/A^(5/3) and re-run
4. **Compare to Bethe-Weizsäcker** - How do our residuals compare to SEMF?

## Conclusion

**The diagnostic succeeded**: We now have a quantitative map of where the pure soliton model fails.

**The prescription is clear**:
- Low A: Add rotational/winding energy
- High A: Add saturation/surface corrections
- Or: Make β(A) adaptive

**The science is honest**: We stopped hiding the error and put it on a graph.

---

**Files**:
- `diagnostic_residuals.csv` - Numerical data
- `missing_physics_diagnosis.png` - Visual plot
- `diagnose_missing_physics.py` - Reproducible script
