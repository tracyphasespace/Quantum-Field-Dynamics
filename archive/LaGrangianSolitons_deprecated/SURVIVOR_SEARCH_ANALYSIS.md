# SURVIVOR SEARCH ANALYSIS - Eccentricity Approach

**Date**: 2026-01-01
**Status**: ✗ **FAILED** - Severe regression detected
**Conclusion**: Original asymmetric coupling is physically incorrect

---

## EXECUTIVE SUMMARY

Allowing nuclei to optimize their eccentricity (0 ≤ ecc ≤ 0.25) to minimize energy **worsens** predictions:

- **Baseline (spherical)**: 127/285 exact (44.6%)
- **Survivor (optimized ecc)**: 83/285 exact (29.1%)
- **Regression**: -44 exact matches (-15.5 percentage points)

**Root cause**: `G_disp = 1/(1+ecc)` creates asymmetric coupling that favors higher Z.

---

## METHODOLOGY

### Energy Functional (Original)

```python
def qfd_survivor_energy(A, Z, ecc):
    G_surf = 1 + ecc²           # Surface increases with deformation
    G_disp = 1/(1 + ecc)        # Displacement DECREASES with deformation

    E_surf = E_surface * A^(2/3) * G_surf
    E_vac  = a_disp * Z²/A^(1/3) * G_disp
    # ... other terms ...
```

### Physical Interpretation (Claimed)

- **G_surf**: Ellipsoid has more surface area → costs more energy ✓
- **G_disp**: Deformation reduces mean density → less vacuum displacement ✗

### What Actually Happens

At **ecc = 0.25** (maximum tested):
- Surface cost: +6.25% (small penalty)
- Displacement cost: -20% (large benefit)

**Net effect**: Optimizer favors configurations with **high Z** (more vacuum displacement), then uses eccentricity to reduce the Z² penalty by 20%. But this pushes Z **away** from experimental values.

---

## RESULTS

### Overall Performance

| Model | Exact Matches | Mean \|ΔZ\| | Median \|ΔZ\| |
|-------|---------------|-------------|---------------|
| Baseline (sphere) | 127/285 (44.6%) | 0.853 | 1.0 |
| Survivor (ecc opt) | 83/285 (29.1%) | 1.554 | 1.0 |

### By Mass Region

| Region | Baseline | Survivor | Change |
|--------|----------|----------|--------|
| Light (A<40) | 71.8% | 74.4% | +2.6% ✓ |
| Medium (40≤A<100) | 42.0% | 33.3% | -8.7% ✗ |
| Heavy (100≤A<200) | 39.1% | 14.6% | **-24.5%** ✗✗ |
| Superheavy (A≥200) | 42.9% | 35.7% | -7.2% ✗ |

**Catastrophic failure in heavy region**: 24.5 percentage point drop!

### Effect Breakdown

- **Improved**: 37/285 (13.0%)
- **Worsened**: 115/285 (40.4%)
- **Neutral**: 133/285 (46.7%)

**Worsened cases outnumber improved by 3:1.**

### Key Test Cases

| Nuclide | Z_exp | Baseline | Survivor | ecc_opt | Verdict |
|---------|-------|----------|----------|---------|---------|
| He-4    | 2     | ✓        | ✓        | 0.050   | Neutral |
| O-16    | 8     | ✓        | ✓        | 0.100   | Neutral |
| Ca-40   | 20    | ✓        | ✓        | 0.200   | Neutral |
| Fe-56   | 26    | +2       | +2       | 0.250   | Neutral |
| Ni-58   | 28    | ✓        | ✓        | 0.250   | Neutral |
| **Xe-136** | **54** | **✓** | **+5** | **0.250** | **✗ Worsened** |
| Pb-208  | 82    | ✓        | ✓        | 0.250   | Neutral |
| **U-238** | **92** | **✓** | **+4** | **0.250** | **✗ Worsened** |

**Xe-136**: Perfect → ΔZ=+5
**U-238**: Perfect → ΔZ=+4

Both are previously successful predictions that now fail.

### Deformation Distribution

**95.4% of nuclei max out at ecc ≥ 0.10**, with most at ecc = 0.25 (the upper bound).

**Interpretation**: The optimizer is hitting the constraint, trying to exploit the displacement reduction as much as possible. This is a **pathological behavior**, not a physical preference for deformation.

---

## ROOT CAUSE ANALYSIS

### The Asymmetry Problem

```
ecc = 0.25:
  G_surf = 1.0625  (+6.25%)
  G_disp = 0.8     (-20%)
```

**Ratio**: Displacement reduction is **3.2× stronger** than surface penalty.

This creates a "loophole":
1. Increase Z (wrong direction experimentally)
2. Use eccentricity to reduce E_vac ∝ Z² by 20%
3. Net energy decreases even though Z is wrong

### The Physics Error

**Claimed**: Deformation reduces mean density → less vacuum displacement

**Reality**: Ellipsoidal deformation typically **compresses** charge along one axis, **increasing** peak density and displacement, not decreasing it.

**Correct coupling**:
- Prolate (cigar-shaped): Higher density at poles → **more** displacement
- Oblate (disk-shaped): Higher density in equatorial plane → **more** displacement

Both should have `G_disp > 1`, not `G_disp < 1`.

---

## CORRECTED FORMULATIONS

### Option 1: Symmetric Increase (Conservative)

Both surface and displacement increase with deformation:

```python
G_surf = 1 + ecc²
G_disp = 1 + k·ecc²   # k > 0, typically k ≈ 0.5-1.0
```

**Rationale**: Deformation increases both surface area and charge concentration.

### Option 2: Quadrupole Moment (Nuclear Physics)

Use standard nuclear deformation parameter β₂:

```python
G_surf = 1 + (2/3)·β₂²                  # Liquid drop model
G_disp = 1 + α·β₂²                      # α > 0
```

**Rationale**: Based on empirical nuclear deformation energetics.

### Option 3: No Displacement Coupling

Only surface energy affected by shape:

```python
G_surf = 1 + ecc²
G_disp = 1            # Shape doesn't affect displacement (simpler)
```

**Rationale**: Vacuum displacement depends on total charge Z², not shape.

---

## PHYSICAL INTERPRETATION

### What the Original Model Got Right

- **Concept**: Nuclei might optimize shape to minimize energy ✓
- **Surface term**: Deformation increases surface energy ✓

### What It Got Wrong

- **Direction**: G_disp should increase with deformation, not decrease ✗
- **Magnitude**: 3:1 asymmetry creates exploitable loophole ✗
- **Coupling mechanism**: Deformation doesn't reduce packing density ✗

---

## RECOMMENDATIONS

### Immediate

1. **Do NOT use original formulation** (G_disp = 1/(1+ecc))
   - Results are worse than baseline
   - Physics is backwards

2. **Test symmetric coupling** (Option 1 above)
   ```python
   G_surf = 1 + ecc²
   G_disp = 1 + 0.5·ecc²
   ```

3. **If still fails, drop displacement coupling** (Option 3)
   - Simpler model
   - Only surface energy affected

### Long-Term

4. **Derive G_disp from first principles**
   - How does ellipsoidal charge distribution affect vacuum energy?
   - Use Cl(3,3) geometric framework
   - Connect to β vacuum stiffness

5. **Use nuclear data to constrain coupling**
   - Deformed nuclei (Rare earths, actinides) have known β₂
   - Fit G_surf, G_disp to match experimental deformations
   - Not all 285 nuclides, just ~20 well-measured deformed cases

6. **Separate deformation from eccentricity**
   - Eccentricity (ecc): Geometric shape parameter
   - Deformation (β₂): Nuclear physics observable
   - May not be the same variable!

---

## SURVIVOR PARADIGM: REVISED

**Original claim**: "Survivors are shape-shifters that optimize eccentricity"

**Revised understanding**: "Survivors may have specific shapes, but our coupling formulation was incorrect"

**Path forward**:
1. Keep the survivor mindset (44.6% are special)
2. Investigate **other** topological properties:
   - Spin J (already explored, shows 71% survive for J=0)
   - Pairing structure (even-even vs odd-A)
   - Magic number proximity (70% of survivors have magic Z or N)
3. Deformation is still interesting, but needs correct physics

**Key insight preserved**: Not all nuclei are equal. The 44.6% that survive in our predictions share common topological features (magic numbers, pairing, low J). The question is what else distinguishes them.

---

## FALSIFICATION TEST

**Prediction**: If eccentricity coupling were correct, heavy nuclei (A>100) should show strong preference for ecc > 0.1.

**Observation**: 95% of ALL nuclei (not just heavy) max out at ecc=0.25.

**Conclusion**: This is optimizer pathology (exploiting the asymmetry), not physics.

**Test**: With symmetric coupling (G_surf = G_disp = 1 + ecc²), most nuclei should prefer ecc ≈ 0 (sphere is optimal), with only known deformed regions (Rare earths, A≈150-180) showing ecc > 0.

---

## CONCLUSION

The topological eccentricity approach **fails** due to incorrect coupling formulation:

- ✗ G_disp = 1/(1+ecc) is physically backwards
- ✗ Creates 3:1 asymmetry favoring wrong predictions
- ✗ Worsens accuracy by 15.5 percentage points

**Recommendation**: **REJECT** this approach until coupling is corrected.

**Alternative**: Return to simpler models (spherical + spin + pairing) that achieved 44.6% and understand why those 127 nuclei succeed.

---

**Files Generated**:
- `survivor_search_test.py` - Xe-136 test case
- `survivor_search_285.py` - Full 285 nuclide sweep
- `SURVIVOR_SEARCH_ANALYSIS.md` - This summary

**Date**: 2026-01-01
**Status**: Analysis complete, approach rejected
**Next step**: Test symmetric coupling or investigate other survivor features
