# Archive: H1 Validation Iterations (2025-12-29)

**Status**: SUPERSEDED - Historical reference only
**Current validated work**: See parent directory

---

## Contents

This archive contains the iterative development of the H1 spin constraint validation, showing the progression from initial error to final correction.

### Phase 1: No Mass Normalization (Morning)

**File**: `derive_alpha_circ.py`

**Error**: Used dimensionless density ρ = 1.0 without normalization
- L scaled as R⁴ instead of being universal
- Required different U for each lepton (unphysical)

**Documentation**: `SESSION_SUMMARY_2025-12-29.md`

### Phase 2: Static Mass Distribution (Midday)

**File**: `derive_alpha_circ_corrected.py`

**Partial fix**: Added mass normalization ρ_phys = M · f(r/R) / ∫f dV
- Made L universal across leptons ✓
- But used wrong mass distribution (static profile)
- Got L = 0.0112 ℏ instead of 0.5 ℏ ("Factor of 45" error)

**Documentation**:
- `SESSION_SUMMARY_2025-12-29_VALIDATION.md`
- `H1_CORRECTED_ANALYSIS.md` (marked as superseded)

### Phase 3: Energy-Based Density (Evening) ✅

**File**: ../scripts/`derive_alpha_circ_energy_based.py` (current directory)

**Correct physics**: Uses energy-based density ρ_eff = M · v²(r) / ∫v² dV
- Follows QFD Chapter 7 formalism
- Concentrates mass at r ≈ R (flywheel geometry)
- Achieves L = 0.5017 ℏ for all leptons ✓

**Documentation** (current directory):
- `H1_SPIN_CONSTRAINT_VALIDATED.md` (validated physics)
- `SESSION_SUMMARY_2025-12-29_FINAL.md` (complete session)

---

## Key Insight from User Correction

The critical correction came from recognizing that **mass = energy** in QFD.

**Wrong**: Treat mass as static field profile
```python
rho_phys = M * f(r/R) / ∫f dV  # Arbitrary profile
```

**Correct**: Treat mass as energy density
```python
rho_eff = M * v²(r) / ∫v² dV  # Energy-based
```

This changes the particle from a "dense-center sphere" to a "relativistic flywheel" with mass concentrated at the Compton radius R.

---

## Why Archive These Files?

These files document the **scientific process**: error → diagnosis → correction.

### Educational Value

1. **Shows iteration is normal**: Even with established formalism (Chapter 7), implementation errors happen
2. **Documents the correction**: Future readers can see exactly what went wrong and why
3. **Preserves decision trail**: Git history + these files provide complete audit trail

### Do Not Use for Current Work

⚠️ **These files contain physics errors** ⚠️

For current validated calculations, use:
- `../scripts/derive_alpha_circ_energy_based.py`
- `../H1_SPIN_CONSTRAINT_VALIDATED.md`

---

## Timeline

- **Morning**: Phase 1 (no normalization) → discovered R-dependence issue
- **Midday**: Phase 2 (static profile) → achieved universality but L too small
- **Afternoon**: User correction identified energy-based density requirement
- **Evening**: Phase 3 (energy-based) → L = ℏ/2 validated ✓

**Total iteration time**: ~8 hours from initial error to validated solution

---

## Lessons Learned

1. **Trust the source material**: Chapter 7 was correct, shortcuts created artifacts
2. **Physical reasonableness**: Large unexplained factors (45×) indicate conceptual errors
3. **Preserve the framework**: Gyroscopic momentum (L = I·ω) was right; only I calculation was wrong
4. **Mass = Energy**: In field theory, effective mass density follows kinetic energy

---

**Archived**: 2025-12-29
**Superseded by**: Energy-based density implementation
**Status**: Historical reference only
