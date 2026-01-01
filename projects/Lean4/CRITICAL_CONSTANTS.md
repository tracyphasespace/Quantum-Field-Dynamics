# Critical Physical Constants - Validation Required

**Last Validated**: 2025-12-29
**Validation Script**: `scripts/derive_alpha_circ_energy_based.py`
**Status**: ✅ All constants verified against Python numerical validation

---

## ⚠️ CRITICAL WARNING: Common Contamination

**WRONG FORMULA** (Standard Model contamination):
```lean
def alpha_circ : ℝ := 1 / (2 * Real.pi)  -- ❌ WRONG! ≈ 0.159
```

**CORRECT FORMULA** (QFD validated):
```lean
noncomputable def alpha_circ : ℝ := Real.exp 1 / (2 * Real.pi)  -- ✅ e/(2π) ≈ 0.4326
```

**This error propagates through all V₄ calculations!**

---

## 🔒 Authoritative Source

**File**: `QFD/Vacuum/VacuumParameters.lean`

**All other files MUST import from this source:**
```lean
import QFD.Vacuum.VacuumParameters

-- ✅ CORRECT: Reference the authoritative definition
noncomputable def alpha_circ : ℝ := QFD.Vacuum.alpha_circ

-- ❌ WRONG: Do NOT redefine locally
noncomputable def alpha_circ : ℝ := Real.exp 1 / (2 * Real.pi)  -- Duplication!
```

---

## 📊 Validated Constants (2025-12-29)

### Vacuum Stiffness Parameters

| Constant | Value | Validation | Source |
|----------|-------|------------|--------|
| `beta` (compression) | 3.058 | Golden Loop + MCMC | VacuumParameters.lean:39 |
| `xi` (gradient) | 1.0 | MCMC fitted | VacuumParameters.lean:70 |
| `tau` (temporal) | 1.0 | MCMC fitted | VacuumParameters.lean:98 |

### Circulation Parameters (CRITICAL)

| Constant | Value | Formula | Validation | Source |
|----------|-------|---------|------------|--------|
| `alpha_circ` | 0.4326 | e/(2π) | Python: 0.432628 | VacuumParameters.lean:213 |
| `alpha_circ_fitted` | 0.431410 | Muon g-2 fit | Python validation | VacuumParameters.lean:216 |
| `I_circ` | 9.4 | Hill vortex integral | Numerical integration | AnomalousMoment.lean:81 |

### Lepton Flywheel Parameters

| Constant | Value | Validation | Source |
|----------|-------|------------|--------|
| `U_universal` | 0.8759c | L = ℏ/2 constraint | VortexStability.lean |
| `I_eff_ratio` | 2.32 | Energy-based density | VortexStability.lean |

---

## 🔍 V₄ Formula Validation

**Complete Formula**:
```
V₄(R) = -ξ/β + α_circ × I_circ × (R_ref/R)²
```

**Numerical Predictions** (must match Python):

| Lepton | R (fm) | V₄ (predicted) | Validation |
|--------|--------|----------------|------------|
| Electron | 386 | -0.327 | Matches C₂(QED) = -0.328 ✓ |
| Muon | 1.87 | +0.836 | Matches muon g-2 anomaly ✓ |
| Tau | 0.111 | +2.5 (approx) | Prediction (unmeasured) |

**Error Bounds**:
- Electron V₄: |V₄ - C₂(QED)| < 0.002 (0.45% error)
- Muon V₄: Consistent with Fermilab 2021 measurement

---

## 🚨 Validation Protocol

**BEFORE committing any changes to these constants:**

### Step 1: Run Python Validation
```bash
cd ../particle-physics/lepton-mass-spectrum
python derive_alpha_circ_energy_based.py
```

**Expected output** (must match):
```
α_circ (geometric) = e/(2π) = 0.432628
I_eff / I_sphere = 2.32
U (universal) = 0.8759c
L = 0.500 ℏ (error: 0.3%)
V₄(electron) = -0.327
V₄(muon) = +0.836
```

### Step 2: Verify Lean Builds
```bash
lake build QFD.Vacuum.VacuumParameters
lake build QFD.Lepton.AnomalousMoment
```

### Step 3: Check for Hardcoded Duplicates
```bash
# This should ONLY return VacuumParameters.lean
grep -r "Real.exp 1 / (2 \* Real.pi)" QFD/ --include="*.lean"

# This should return NO results (contamination check)
grep -r "def alpha_circ.*1 / (2 \* Real.pi)" QFD/ --include="*.lean"
```

### Step 4: Cross-Reference Check
```bash
# All files using alpha_circ should import VacuumParameters
grep -r "alpha_circ" QFD/ --include="*.lean" -A 2 | grep -B 2 "import.*VacuumParameters"
```

---

## 📝 History of Contamination Events

### 2025-12-29: alpha_circ Correction
- **Contaminated value**: `1 / (2 * Real.pi)` ≈ 0.159
- **Correct value**: `Real.exp 1 / (2 * Real.pi)` ≈ 0.4326
- **Impact**: V₄(muon) was predicted as +0.10 instead of +0.836
- **Root cause**: AI assistant used Standard Model reasoning instead of QFD
- **Resolution**: Python validation confirmed e/(2π), Lean files corrected
- **Affected files**:
  - ✅ VacuumParameters.lean (corrected)
  - ✅ AnomalousMoment.lean (now imports correctly)
  - ✅ All other files verified clean

---

## ✅ Verification Checklist

Before merging ANY changes to vacuum parameters:

- [ ] Python validation script runs successfully
- [ ] All numerical predictions match Python output
- [ ] No hardcoded constant duplications exist
- [ ] All imports reference `QFD.Vacuum.VacuumParameters`
- [ ] `lake build` succeeds for affected modules
- [ ] Git diff shows ONLY intentional changes
- [ ] This document is updated with new validation date

---

## 🔗 Related Documentation

- **Python Validation**: `../particle-physics/lepton-mass-spectrum/derive_alpha_circ_energy_based.py`
- **Numerical Results**: `../particle-physics/lepton-mass-spectrum/H1_SPIN_CONSTRAINT_VALIDATED.md`
- **Protected Files**: `PROTECTED_FILES.md`
- **AI Workflow**: `AI_WORKFLOW.md`

---

**Maintained by**: Tracy (human oversight)
**Enforced by**: AI discipline + build verification
**Next Review**: After any vacuum parameter modification
