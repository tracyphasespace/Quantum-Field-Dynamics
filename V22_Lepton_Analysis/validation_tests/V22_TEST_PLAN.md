# V22 Test Plan — Lepton Energy Functional Validation (e, μ, τ)

**Date**: 2025-12-25
**Scope**: Validate whether the current Hill-vortex + boundary-layer density model can fit lepton masses with a shared β and global S, and diagnose the τ anomaly without over-claiming ontology.

## 0. Background and Current Verified Findings

### 0.1 Quantified τ energy deficit (baseline, non-localized model)
- Required: m_τ/m_μ = 16.81
- Predicted by current circulation scaling: E_τ/E_μ ≈ (U_τ/U_μ)^2 = 9.00
- Deficit: ≈46.5% shortfall for τ relative to μ

### 0.2 Critical model-mechanics issue (orthogonality)
- Circulation energy is effectively profile-insensitive because velocity support and density-deficit support overlap weakly (dominant kinetic contribution arises where ρ≈1).
- Overshoot-shell v0 did not materially change I = E_circ/U² (sub-percent sensitivity), motivating attempts to localize the far-field.

### 0.3 Sign convention correction (must be treated as fixed baseline in V22)
**Correct energy functional (V22)**:
```
E_total = E_circ + E_stab + E_grad
```

All penalty terms ADD (no subtraction). This is now codified and sanity-checked.

### 0.4 Current status of Run 2 (e,μ regression)
Baseline corrected-sign run exhibits:
- χ² ~ 1e8 (pathological)
- β pegged at scan edge (3.0)
- Complete bound saturation (6/6 parameters at bounds)
- Moderate localization diagnostics (F_inner ~ 46–48%)

---

## 1. Success Criteria (Global Gates)

A configuration/run is classified using these gates.

### 1.1 Primary success gates (required)
- **G1**: χ²_min < 1e6 (order-of-magnitude improvement vs ~1e8 baseline)
- **G2**: S_opt > 0
- **G3**: Degeneracy broken: ≤ 1 parameter per lepton at bounds (not 3/3 each)
- **G4**: β not at scan edge (interior to interval)

### 1.2 Secondary diagnostics (informative, not sufficient alone)
- **D1**: F_inner > 50% (suggests profile sensitivity)
- **D2**: Multi-start stability (same basin across seeds)
- **D3**: Smooth χ²(β) minimum (not monotone to an edge)

---

## 2. Test Inventory (T0–T6)

### T0 — Reproducibility + Invariants Check
**Purpose**: Ensure correct functional and conventions before interpreting failures.

**CLI**:
```bash
python3 renormalized_sanity_check.py | tee results/V22/logs/t0_invariants_check.log
```

**Expected output**:
```
✓ SANITY CHECK PASSED
Both electron and muon have E_total > 0 at reasonable parameters.
Corrected sign convention makes all penalty terms add consistently.
```

**Pass conditions**:
- E_total > 0 for representative e, μ parameters
- Prints explicit sign audit confirming corrected convention

**Stop/Go**:
- ✓ PASS → Proceed to T1
- ✗ FAIL → Fix sign convention regression before any optimization

---

### T1 — Bound-Artifact Elimination Sweep
**Purpose**: Confirm whether optimizer constraints manufacture degeneracy.

**Implementation**:
Create `run2_emu_widened_bounds.py` by copying `run2_emu_regression_corrected.py` and modifying bounds:

**Tier A (conservative - recommended first)**:
```python
bounds = [
    (0.20, 1.20),   # R_c_e (was 0.5-1.5)
    (0.005, 0.30),  # U_e (was 0.01-0.10)
    (0.50, 1.20),   # A_e (was 0.70-1.0)
    (0.05, 0.80),   # R_c_mu (was 0.05-0.30)
    (0.02, 0.60),   # U_mu (was 0.05-0.20)
    (0.50, 1.50),   # A_mu (was 0.70-1.0)
]
```

**CLI**:
```bash
python3 run2_emu_widened_bounds.py 2>&1 | tee results/V22/logs/t1_tier_a_widened_bounds.log
```

**Expected outputs**:
- `results/V22/t1_tier_a_results.json`
- Log with χ², S_opt, bound hits

**Stop/Go**:
- If degeneracy breaks (≤1 param per lepton at bounds) AND χ² < 1e6 → **New baseline**, proceed to T5
- If still 6/6 saturated AND χ² ~ 1e8 → Degeneracy is not boxing artifact, proceed to T2

**Tier B (aggressive, only if Tier A fails)**:
```python
bounds = [
    (0.05, 2.00),   # R_c_e
    (0.001, 0.60),  # U_e
    (0.00, 2.00),   # A_e
    (0.02, 1.50),   # R_c_mu
    (0.005, 0.90),  # U_mu
    (0.00, 2.00),   # A_mu
]
```

---

### T2 — Circulation Profile-Sensitivity Audit
**Purpose**: Directly measure whether E_circ "sees" geometry/density.

**Implementation**:
Create `profile_sensitivity_sweep.py`:

```python
#!/usr/bin/env python3
"""
Profile Sensitivity Audit: I(A, R_c) = E_circ / U²

Sweep (A, R_c) at fixed U and measure ΔI/I to quantify
whether circulation energy has geometric sensitivity.
"""
import numpy as np
from lepton_energy_localized_v1 import LeptonEnergyLocalizedV1
from profile_likelihood_boundary_layer import calibrate_lambda
import sys

# Configuration
beta = 3.15
w = 0.020
eta_target = 0.03
R_c_ref = 0.88
lam = calibrate_lambda(eta_target, beta, R_c_ref)

k = 1.5
delta_v = 0.5
p = 6

energy_calc = LeptonEnergyLocalizedV1(
    beta=beta, w=w, lam=lam,
    k_localization=k, delta_v_factor=delta_v, p_envelope=p
)

# Fix U, sweep (A, R_c)
U_fixed = 0.10
A_grid = np.linspace(0.5, 1.0, 11)
R_c_grid = np.linspace(0.3, 0.9, 11)

I_values = []

print("Sweeping (A, R_c) at fixed U =", U_fixed)
print(f"{'A':<8} {'R_c':<8} {'I':<12} {'E_circ':<12}")
print("-" * 50)

for A in A_grid:
    for R_c in R_c_grid:
        R = R_c + w
        _, _, _, E_circ, _ = energy_calc.circulation_energy_with_diagnostics(
            R, U_fixed, A, R_c
        )
        I = E_circ / U_fixed**2 if U_fixed > 0 else 0
        I_values.append(I)
        print(f"{A:<8.3f} {R_c:<8.3f} {I:<12.6f} {E_circ:<12.6f}")
        sys.stdout.flush()

I_values = np.array(I_values)
I_mean = np.mean(I_values)
I_std = np.std(I_values)
I_range = np.max(I_values) - np.min(I_values)
delta_I_over_I = I_range / I_mean if I_mean > 0 else 0

print("-" * 50)
print(f"I_mean:       {I_mean:.6f}")
print(f"I_std:        {I_std:.6f}")
print(f"I_range:      {I_range:.6f}")
print(f"ΔI/I:         {delta_I_over_I:.4%}")
print()

if delta_I_over_I >= 0.05:
    print("✓ PASS: Strong profile sensitivity (ΔI/I ≥ 5%)")
elif delta_I_over_I >= 0.01:
    print("~ MARGINAL: Weak profile sensitivity (1% ≤ ΔI/I < 5%)")
else:
    print("✗ FAIL: Negligible profile sensitivity (ΔI/I < 1%)")
    print("Circulation is geometry-blind; localization tuning unlikely to resolve τ deficit.")
```

**CLI**:
```bash
python3 profile_sensitivity_sweep.py | tee results/V22/logs/t2_profile_sensitivity.log
```

**Pass thresholds**:
- ΔI/I ≥ 5%: Strong sensitivity → proceed to T3
- ΔI/I < 1%: Geometry-blind → proceed to T4 (physics change)

---

### T3 — Run 2 e,μ Regression (correct signs; frozen config)
**Purpose**: Determine whether model is viable for light leptons without τ.

**CLI**:
```bash
python3 run2_emu_regression_corrected.py 2>&1 | tee results/V22/logs/t3_run2_baseline.log
```

**Expected outputs**:
- `results/V22/run2_emu_corrected_results.json`
- Log with acceptance criteria evaluation

**Stop/Go**:
- ✓ Gates G1–G4 met → Proceed to T5 (add τ)
- ✗ χ² ~ 1e8, bounds saturated → Proceed to T4 (physics change)
- ~ χ² improved but β at edge → Expand β range, rerun once; if still edge-pinned → T4

---

### T3b — Localization Configuration Sweep (batch)
**Purpose**: Explore whether any localization settings break degeneracy.

**CLI**:
```bash
python3 overnight_batch_test.py 2>&1 | tee results/V22/logs/t3b_localization_sweep.log
```

**Configurations tested**:
1. k=1.0, Δv/Rv=0.5, p=6 (strong localization)
2. k=2.0, Δv/Rv=0.5, p=6 (weak localization)
3. k=1.5, Δv/Rv=0.25, p=6 (narrow falloff)
4. k=1.5, Δv/Rv=0.75, p=6 (wide falloff)
5. k=1.5, Δv/Rv=0.5, p=4 (soft envelope)
6. k=1.5, Δv/Rv=0.5, p=8 (sharp envelope)

**Expected outputs**:
- `results/V22/overnight_batch_summary.json`
- Individual config JSONs: `results/V22/overnight_config{1-6}_*.json`

**Quick check**:
```bash
# View summary table
tail -100 results/V22/logs/t3b_localization_sweep.log

# Check outcomes
grep -E "PASS|FAIL|SOFT_PASS" results/V22/logs/t3b_localization_sweep.log
```

**Stop/Go**:
- **ALL configs FAIL** (χ² > 1e6) → Localization tuning insufficient, proceed to T4
- **1-2 configs PASS** → Rerun best with maxiter=100, β=21, 5 seeds, then T5
- **Any χ² < 100** → Breakthrough! Fine neighborhood scan, freeze params, proceed to T5

---

### T4 — Pivot: Add Minimal New Physics
**Purpose**: Introduce one defensible modification at a time (only after T1–T3 fail).

#### T4A: Bulk Potential Term (preferred first pivot)

**Implementation**: Modify `lepton_energy_localized_v1.py` to add:

```python
def bulk_energy(self, R_c, A, a2=1.0, a4=0.1):
    """
    Bulk potential penalizing deviation from vacuum density.

    V_bulk = a2*(ρ-1)^2 + a4*(ρ-1)^4
    E_bulk = ∫ V_bulk dV (co-localized)
    """
    # Integration over shell region where δρ ≠ 0
    R_shell = R_c + 3*self.w  # Conservative support estimate

    def integrand(r):
        rho = self.rho_deficit(r, R_c, A) + 1.0  # Total density
        delta_rho = rho - 1.0
        V_bulk = a2 * delta_rho**2 + a4 * delta_rho**4

        # Co-localize with same envelope
        g = self.localization_envelope(r)

        return 4 * np.pi * r**2 * V_bulk * g

    r_grid = np.linspace(0, R_shell, 200)
    integrand_vals = integrand(r_grid)
    E_bulk = np.trapz(integrand_vals, r_grid)

    return E_bulk
```

Then update `total_energy()`:
```python
def total_energy(self, R_c, U, A, a2=1.0, a4=0.1):
    R = R_c + self.w
    _, _, _, E_circ, _ = self.circulation_energy_with_diagnostics(R, U, A, R_c)
    E_stab = self.stabilization_energy(R_c, A)
    E_grad = self.gradient_energy(R_c, A)
    E_bulk = self.bulk_energy(R_c, A, a2, a4)

    E_total = E_circ + E_stab + E_grad + E_bulk
    return E_total, E_circ, E_stab, E_grad, E_bulk
```

**CLI**:
```bash
python3 run2_emu_with_bulk_potential.py 2>&1 | tee results/V22/logs/t4a_bulk_potential.log
```

**Stop/Go**:
- Gates G1–G4 met → Proceed to T5
- Degeneracy improves (bound hits drop) → Widen search, add constraints (T4C)
- Still full saturation → Proceed to T4B

#### T4B: Replace Density Profile Family

**Options** (try one at a time):
1. **Overshoot-allowed**: Allow ρ > 1 in shell region
2. **Non-Gaussian**: Lorentzian, step function, or piecewise profile
3. **Mass-concentrated**: More density near r ≈ R where v² is largest

**CLI**:
```bash
python3 run2_emu_alternative_profile.py 2>&1 | tee results/V22/logs/t4b_alt_profile.log
```

**Stop/Go**:
- Must improve ΔI/I (rerun T2) AND reduce bound saturation
- If neither improves → Try T4C

#### T4C: Add Identifiability Constraint

**Options** (choose one):
1. Fix charge radius (add penalty if <r²> deviates from target)
2. Constrain angular momentum proxy
3. Add magnetic moment term to objective (if calculable)

**Example constraint**:
```python
def charge_radius_penalty(self, R_c, A, target_r_rms=0.8, weight=100.0):
    """Penalize if RMS radius deviates from experimental target."""
    r_rms = self.compute_rms_radius(R_c, A)
    penalty = weight * (r_rms - target_r_rms)**2
    return penalty
```

**CLI**:
```bash
python3 run2_emu_constrained.py 2>&1 | tee results/V22/logs/t4c_constrained.log
```

**Stop/Go**:
- Must reduce "6/6 at bounds" in e,μ fit
- If not, revert and try next pivot

---

### T5 — Reintroduce τ (only after e,μ passes)
**Purpose**: Determine whether τ remains a systematic deficit under validated e,μ model.

**Implementation**: Extend fitter to 3 leptons.

```python
# In fitter class
self.m_targets = np.array([M_E, M_MU, M_TAU])

# Bounds (9 parameters total)
bounds = [
    # Electron
    (R_c_min, R_c_max), (U_min, U_max), (A_min, A_max),
    # Muon
    (R_c_min, R_c_max), (U_min, U_max), (A_min, A_max),
    # Tau
    (R_c_min, R_c_max), (U_min, U_max), (A_min, A_max),
]
```

**CLI**:
```bash
python3 run3_all_leptons.py 2>&1 | tee results/V22/logs/t5_three_lepton_fit.log
```

**Expected outputs**:
- `results/V22/run3_all_leptons_results.json`
- Per-lepton parameters, energies, diagnostics
- Energy ratios: E_τ/E_μ, E_μ/E_e

**Analysis**:
```python
# From results JSON
E_tau = results["energies"]["tau"]["E_total"]
E_mu = results["energies"]["muon"]["E_total"]
E_ratio = E_tau / E_mu

m_tau = 1776.86
m_mu = 105.7
m_ratio = m_tau / m_mu  # = 16.81

deficit_pct = (m_ratio - E_ratio) / m_ratio * 100
print(f"Predicted E_τ/E_μ: {E_ratio:.2f}")
print(f"Required m_τ/m_μ:  {m_ratio:.2f}")
print(f"τ deficit: {deficit_pct:.1f}%")
```

**Stop/Go**:
- **τ deficit persists (~46%)** while e,μ stable → **Publishable finding**:
  - "Systematic τ deficit quantified under validated e,μ calibration"
  - Suggests new physics or missing degree of freedom for τ
- **τ becomes consistent** → Prior anomaly was identifiability/functional issue now resolved

---

### T6 — Reporting/Manuscript Integration
**Purpose**: Keep narrative defensible and modular.

**Required artifacts after each major run**:

1. **One-page summary** (plain text or markdown):
   ```markdown
   # Test TX Summary

   **Configuration**: k=X, Δv/Rv=Y, p=Z
   **Outcome**: PASS/FAIL

   ## Results
   - χ²_min: X.XXe+XX
   - S_opt: X.XXXX
   - β_min: X.XXXX (interior/edge)
   - Bound hits: e=X/3, μ=Y/3, τ=Z/3

   ## Parameters
   [Table of R_c, U, A per lepton]

   ## Energies
   [Table of E_circ, E_stab, E_grad, E_total per lepton]

   ## What Changed
   - Physics: [description]
   - Numerics: [description]
   - Bounds: [description]
   ```

2. **Changelog** (`CHANGELOG_V22.md`):
   ```markdown
   ## 2025-12-25 - T3b Localization Sweep
   - Tested 6 configs: all failed (χ² ~ 1e8)
   - Degeneracy persists across k ∈ [1.0, 2.0]
   - Conclusion: Localization tuning insufficient → proceed to T4A
   ```

3. **Conservative τ narrative** (for manuscript):
   ```
   The circulation-dominated model predicts E_τ/E_μ ≈ 9 from U² scaling,
   but the observed mass ratio requires 16.8, producing a systematic τ
   energy deficit of ≈46%. This quantified discrepancy motivates either
   (a) additional τ-specific physics or (b) a beyond-Hill-vortex framework.
   ```

---

## 3. Quick Reference: Run Order Decision Tree

```
START
  ↓
T0: Invariants check
  ↓ PASS
T1: Widen bounds (Tier A)
  ├─ PASS (degeneracy breaks, χ²<1e6) → T5
  └─ FAIL (still 6/6, χ²~1e8) → T2
       ↓
T2: Profile sensitivity (ΔI/I)
  ├─ Strong (≥5%) → T3
  └─ Weak (<1%) → T4
       ↓
T3/T3b: e,μ regression ± localization sweep
  ├─ Any config PASS → T5
  └─ All FAIL → T4
       ↓
T4: Physics pivots (A→B→C until e,μ passes)
  ├─ A: Bulk potential
  ├─ B: Alternative density profile
  └─ C: Add constraint
       ↓ (any success)
T5: Add τ back
  ├─ τ deficit persists → Publishable finding
  └─ τ consistent → Anomaly resolved
       ↓
T6: Document & report
```

---

## 4. File Map (Where to Find Things)

### Existing scripts (ready to run)
- `renormalized_sanity_check.py` - T0 invariants check
- `run2_emu_regression_corrected.py` - T3 baseline e,μ regression
- `overnight_batch_test.py` - T3b localization sweep
- `lepton_energy_localized_v1.py` - Energy calculator (modify for T4)

### To create for this plan
- `run2_emu_widened_bounds.py` - T1 (copy from run2, change bounds)
- `profile_sensitivity_sweep.py` - T2 (template provided above)
- `run2_emu_with_bulk_potential.py` - T4A (modify energy calc)
- `run2_emu_alternative_profile.py` - T4B (new density parameterization)
- `run2_emu_constrained.py` - T4C (add constraint to objective)
- `run3_all_leptons.py` - T5 (extend to 3 leptons)

### Documentation
- `V22_TEST_PLAN.md` - This file (front door for GitHub)
- `DEVELOPMENT_GUIDELINES.md` - Technical conventions (tqdm, workers, etc.)
- `MORNING_SUMMARY.md` - Context from overnight batch
- `DIAGNOSTIC_SUMMARY.md` - τ deficit quantification
- `CHANGELOG_V22.md` - To create (track what changed between runs)

---

## 5. Next Immediate Action

**Right now** (2025-12-25 morning):

1. **Check overnight batch results**:
   ```bash
   tail -100 results/V22/logs/overnight_batch.log
   grep "BEST CONFIGURATION" results/V22/logs/overnight_batch.log -A 10
   ```

2. **If all configs failed** (expected based on Config 1 preview):
   - Mark T3b as FAIL in your notes
   - Proceed to **T1** (widen bounds, Tier A)

3. **If any config passed**:
   - Rerun best config with maxiter=100, β=21 points
   - Test multi-start stability (5 seeds)
   - If stable → Skip to T5 (add τ)

4. **If χ² < 100 for any config**:
   - Stop everything
   - Fine-grid scan around that (k, Δv/Rv, p)
   - Document as breakthrough and proceed to T5 immediately

---

**END OF V22 TEST PLAN**
