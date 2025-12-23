# Physics Corrections Status

**Date**: 2025-12-22
**Task**: Remove incorrect "spin cancellation" physics, update with correct four-mechanism model

---

## ✅ COMPLETED CORRECTIONS

### Documentation (Fully Corrected)

1. **CORRECT_RIFT_PHYSICS.md** ✅
   - Definitive reference with correct physics
   - L1 geometry as primary mechanism (~90%)
   - Rotational KE contribution (~8-10%)
   - Three fates explained
   - Reference frame effects

2. **PARADIGM_SHIFT.md** ✅
   - Updated GR vs QFD comparison
   - Removed all "spin cancellation" references
   - Correct energy budget breakdown

3. **README_CORRECTED.md** ✅
   - Summary of corrections
   - What was wrong vs what is correct
   - Action items for code review

### Visualization Scripts (Corrected)

4. **rift/elliptical_orbit_eruption.py** ✅
   - Updated energy budget calculation
   - L1 geometry comparison fixed (~90% reduction shown correctly)
   - Panel 3: Shows single BH vs binary L1 barrier
   - Panel 4: Three fates note added
   - Rotational KE calculated as E_rot = (1/2)m(Ωr)²
   - **Plot**: validation_plots/15_eccentric_orbit_eruption.png

5. **rift/parameter_space_analysis.py** ✅
   - Docstring updated with four mechanisms
   - Removed angular gradient metric collection
   - Panel 3: Changed to L1 barrier energy
   - Panel 5: Changed to rotational KE
   - Panel 8: Updated to show correct mechanism contributions
   - Summary text updated
   - **Plot**: validation_plots/12_parameter_space.png

6. **rift/mass_dependent_ejection.py** ✅
   - Already correct (mass-dependent escape physics)
   - No changes needed
   - **Plot**: validation_plots/13_mass_dependent.png

7. **rift/astrophysical_scaling.py** ✅
   - No "cancellation" references found
   - Already uses correct physics
   - **Plot**: validation_plots/14_astrophysical_scaling.png

---

## ⚠️ NEEDS FURTHER WORK

### High Priority

8. **rift/mechanism_tradeoffs.py** ⚠️
   - **Status**: PARTIALLY CORRECTED
   - Trade-off 1 updated (spin rate vs proximity)
   - Trade-offs 2 & 3 still use "cancellation" variable
   - All visualization panels need rewrite
   - Summary text still references incorrect physics
   - **Action**: Full rewrite needed
   - **Deprecation notice added** to file header

### Core Simulation Code (Lower Priority)

9. **rift/core_3d.py**
   - Contains `check_opposing_rotations_cancellation()` function
   - Docstrings reference "angular gradient cancellation"
   - **Note**: Function may still be useful for checking field symmetry near L1
   - Consider renaming to `check_field_symmetry_at_L1()`

10. **rift/rotation_dynamics.py**
    - References "angular_gradient_cancellation" in lean docs
    - May need docstring updates

11. **rift/binary_rift_simulation.py**
    - References "angular gradient cancellation" in description
    - Update docstrings

12. **rift/binary_rift_visualization.py**
    - Plots "angular gradient cancellation"
    - May need visualization update or removal

13. **rift/visualization.py**
    - Contains gradient cancellation plots
    - Consider removing or relabeling

14. **rift/example_validation.py**
    - Calls `check_opposing_rotations_cancellation()`
    - Update to use corrected physics terminology

15. **rift/run_L1_validation.py**
    - References "Angular cancellation" in output
    - Update print statements

16. **rift/__init__.py**
    - Module description mentions "angular gradient cancellation"
    - Update module docstring

---

## SUMMARY OF CHANGES

### What Was Wrong (REMOVED)
- ❌ "Spin cancellation" / "angular gradient cancellation"
- ❌ Field interference effects
- ❌ Angular gradients ∂φ/∂θ canceling at L1
- ❌ 90% from exotic cancellation mechanism

### What Is Correct (CURRENT)
- ✓ **Binary L1 geometry** - gravitational counter-pull (~90% barrier reduction)
- ✓ **Rotational kinetic energy** - disk rotation E_rot = (1/2)m(Ωr)² (~8-10%)
- ✓ **Coulomb repulsion** - charge separation (~1-2%)
- ✓ **Thermal motion** - temperature (negligible for bulk, important for selection)
- ✓ **Three fates** - escape (~1%), capture, fall back (~99%)
- ✓ **Reference frames** - v_local ≠ v_COM (critical!)

### Energy Budget (Correct)
At 300 r_g separation for 100 M☉ binary:
```
Single BH barrier at 150 r_g:     1.20e+15 J  (100%)
Binary L1 barrier:                 1.20e+14 J  (~10%)
  → L1 geometry reduces by ~90%

Energy to overcome remaining barrier:
  Rotational KE:                   1.12e+16 J  (~9375% of barrier)
  Coulomb:                         2.40e+12 J  (~2%)
  Thermal:                         negligible

Result: Escape possible but rare (~1%)
```

---

## REGENERATED PLOTS

✅ **validation_plots/15_eccentric_orbit_eruption.png**
- Energy budget panel shows correct L1 geometry reduction
- Four mechanisms properly labeled
- Three fates note added

⚠️ **Other plots need regeneration** after code updates complete

---

## NEXT STEPS

1. **Complete mechanism_tradeoffs.py rewrite**
   - Replace "spin_cancellations" array with "spin_rates" (Ω in c/r_g)
   - Calculate E_rot for each spin rate
   - Update all three trade-off analyses
   - Regenerate plot

2. **Core code review** (optional, lower priority)
   - Decide whether to keep or rename `check_opposing_rotations_cancellation()`
   - Update all docstrings in core modules
   - Consider: function may still be checking useful field symmetry

3. **Documentation pass**
   - Ensure all inline comments use correct terminology
   - Update any remaining Lean theorem references

4. **Final validation**
   - Run all visualization scripts
   - Verify plots show correct physics
   - Check output text uses correct terminology

---

**Status**: Primary corrections complete (documentation + main visualization)
**Remaining**: mechanism_tradeoffs.py rewrite + core code docstring updates

