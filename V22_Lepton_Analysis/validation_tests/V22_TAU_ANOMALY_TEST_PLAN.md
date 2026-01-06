# V22 Tau Anomaly Test Plan

**Purpose**: Determine whether the τ energy shortfall (E_τ/E_μ ≈ 9 vs m_τ/m_μ ≈ 17) is:
- (A) A constraint artifact (bounds forcing 9×)
- (B) Profile-insensitivity in E_circ functional
- (C) Genuine new physics (heavy vortex / compression)

**Current status**: τ shows persistent S_τ/S_μ ≈ 1.86 regime change across all closures, with optimizer hitting bounds.

---

## Test 1: Remove Bound Artifacts

### Hypothesis
The 9× scaling is being manufactured by parameter bounds:
- μ: U at lower bound (0.05), R_c at upper bound (0.30)
- τ: U at upper bound (0.15)

These constraints force U_τ/U_μ ≤ 3, hence E_τ/E_μ ≤ 9 if E_circ ∝ U².

### Implementation

Modify `profile_likelihood_boundary_layer.py`:

```python
# Current bounds (saturating)
bounds = {
    "muon": {"R_c": (0.05, 0.30), "U": (0.05, 0.20), "A": (0.7, 1.0)},
    "tau": {"R_c": (0.30, 0.80), "U": (0.02, 0.15), "A": (0.7, 1.0)},
}

# Widened bounds (exploratory)
bounds_wide = {
    "muon": {"R_c": (0.05, 0.50), "U": (0.02, 0.20), "A": (0.7, 1.0)},
    "tau": {"R_c": (0.30, 0.80), "U": (0.02, 0.25), "A": (0.7, 1.0)},
}
```

Key changes:
- μ: U_min 0.05 → 0.02 (allow smaller circulation)
- μ: R_c_max 0.30 → 0.50 (allow larger geometry)
- τ: U_max 0.15 → 0.25 (allow stronger circulation)

### Expected Outcomes

**Outcome 1A (Bound artifact)**:
- With widened bounds, U_τ/U_μ naturally moves toward √16.8 ≈ 4.1
- E_τ/E_μ climbs toward ~17
- S_τ/S_μ approaches 1.0 (universal scaling restored)
- χ² drops to O(1-100)
- **Interpretation**: Most of the "regime change" was a box constraint issue
- **Next step**: Publish with wider bounds, document identifiability constraints

**Outcome 1B (Genuine regime change)**:
- Even with widened bounds, fit still saturates near old values
- S_τ/S_μ remains ≈ 1.86
- χ² remains O(10⁷)
- **Interpretation**: τ truly needs different physics
- **Next step**: Proceed to Test 2

### Success Criteria

**Pass**: U_τ/U_μ > 3.5 AND S_τ/S_μ < 1.5 AND χ² < 1000

**Fail**: Parameters still saturate at new bounds OR S_τ/S_μ > 1.7

### Files to Create

- `test_widened_bounds.py` - Run fit with bounds_wide
- `compare_bound_regimes.py` - Side-by-side comparison (old vs new bounds)

---

## Test 2: Circulation Functional Profile-Sensitivity

### Hypothesis

The circulation energy E_circ is effectively behaving as "constant × U²" across leptons, ignoring profile differences in ρ(r), R_c, A.

**Evidence**: E_circ,τ/E_circ,μ = 9.00 ≈ (U_τ/U_μ)² = 9.00 (too exact to be coincidence)

### Implementation

Add diagnostic to `energy_component_breakdown.py`:

```python
def compute_circulation_prefactor(lepton, params, energies):
    """
    Compute I_ℓ = E_circ,ℓ / U_ℓ²

    If I_τ ≈ I_μ, then E_circ is blind to profile differences.
    """
    E_circ = energies[lepton]["E_circ"]
    U = params[lepton]["U"]

    I = E_circ / (U**2) if U > 0 else 0

    return I

# At best fit, compute for all leptons
I_e = compute_circulation_prefactor("electron", params, energies)
I_mu = compute_circulation_prefactor("muon", params, energies)
I_tau = compute_circulation_prefactor("tau", params, energies)

print(f"Circulation prefactors:")
print(f"  I_e   = {I_e:.6f}")
print(f"  I_μ   = {I_mu:.6f}")
print(f"  I_τ   = {I_tau:.6f}")
print(f"  I_τ/I_μ = {I_tau/I_mu:.4f}")

# Test sensitivity to profile changes
# Hold U fixed, vary A from 0.70 to 0.99
for A_test in [0.70, 0.80, 0.90, 0.99]:
    E_circ_test = energy_calc.circulation_energy(R_c, U, A_test)
    I_test = E_circ_test / U**2
    print(f"  A={A_test:.2f}: I={I_test:.6f}")
```

### Expected Outcomes

**Outcome 2A (Profile-insensitive, bug)**:
- I_τ/I_μ ≈ 1.00 ± 0.05
- I barely changes when varying A or R_c
- **Interpretation**: E_circ implementation is not using ρ(r) correctly
- **Action**: Fix circulation_energy() to integrate ∫ ρ(r)|v(r)|² dV properly
- **Next step**: Re-test after fix

**Outcome 2B (Profile-sensitive)**:
- I_τ/I_μ differs from 1.0 by >20%
- I varies substantially with A, R_c
- **Interpretation**: E_circ is working as intended, but not capturing needed physics
- **Next step**: Proceed to Test 3

### Success Criteria

**Pass**: |I_τ/I_μ - 1.0| > 0.20 AND I changes by >10% when A varies 0.70→0.99

**Fail**: I_τ/I_μ ≈ 1.00 ± 0.05 (functional is profile-blind)

### Files to Create

- `test_circulation_prefactor.py` - Compute I_ℓ and sensitivity
- `inspect_circulation_implementation.py` - Detailed breakdown of E_circ integral

---

## Test 3: Two-Lepton Fit (e, μ Only)

### Hypothesis

If τ is the outlier driving β upward, then fitting only (e, μ) should recover β ≈ 3.058.

### Implementation

Modify `profile_likelihood_boundary_layer.py` to create `LeptonFitterTwoLepton`:

```python
class LeptonFitterTwoLepton(LeptonFitter):
    """Fit only electron + muon (exclude tau)"""

    def __init__(self, beta, w, lam, sigma_model=1e-4):
        super().__init__(beta, w, lam, sigma_model)

        # Override targets for e, μ only
        self.m_targets = np.array([M_E, M_MU])  # No M_TAU
        self.leptons = ["electron", "muon"]  # No tau
```

Then run profile likelihood scan over β:

```python
beta_range = (3.00, 3.15)
n_beta = 15

results = []
for beta in np.linspace(*beta_range, n_beta):
    fitter = LeptonFitterTwoLepton(beta=beta, w=0.020, lam=calibrate_lambda(0.03, beta, 0.88))
    result = fitter.fit(max_iter=200)
    results.append({
        "beta": beta,
        "chi2": result["chi2"],
        "S_opt": result["S_opt"],
    })

# Find minimum
beta_min = min(results, key=lambda x: x["chi2"])["beta"]
```

### Expected Outcomes

**Outcome 3A (τ is outlier)**:
- β_min ≈ 3.058 ± 0.02 for (e, μ) fit
- χ² < 10 (good fit quality)
- S_e/S_μ ≈ 1.00 ± 0.10 (universal scaling)
- **Interpretation**: e, μ validate the Hill vortex model with β from α
- **Narrative**: "Light leptons confirm β=3.058; τ exhibits systematic deviation consistent with hadronic mass scale"
- **Next step**: Publish two-lepton validation, treat τ as future work

**Outcome 3B (universal EM proxy issue)**:
- β_min ≈ 3.15-3.18 even for (e, μ) only
- S_e/S_μ still shows regime split
- **Interpretation**: The EM proxy issue affects all leptons, not just τ
- **Next step**: Deeper review of circulation energy formula or EM coupling assumptions

### Success Criteria

**Pass**: |β_min - 3.058| < 0.03 AND χ² < 20 AND |S_e/S_μ - 1.0| < 0.15

**Fail**: β_min > 3.10 OR S_e/S_μ > 1.15

### Files to Create

- `test_two_lepton_fit.py` - (e, μ) only profile likelihood
- `compare_two_vs_three_lepton.py` - Side-by-side β scans

---

## Test 4: Add Ballast/Compression Physics (If Needed)

**Only run if Tests 1-3 all show τ genuinely needs different physics**

### Option 4A: Density-Weighted Kinetic Energy

Replace current E_circ with true kinetic integral:

```python
def circulation_energy_density_weighted(self, R, U, A):
    """
    E_circ = ∫ ρ(r) |v(r)|² dV

    This makes kinetic energy sensitive to "carried mass."
    τ with smaller deficit but similar U will have higher E_circ.
    """
    # Build density profile
    density = DensityBoundaryLayer(R_c, self.w, A, rho_vac=RHO_VAC)
    rho = density.rho(self.r)

    # Hill vortex velocity (simplified)
    v = self._velocity_field(R, U)

    # Kinetic energy density
    kinetic_density = 0.5 * rho * v**2

    # Integrate over volume
    integrand = kinetic_density * self.r**2
    E_circ = 4 * np.pi * np.trapz(integrand, self.r)

    return E_circ
```

**Expected outcome**: If τ has higher average ρ in the circulation region, this will boost E_τ/E_μ above the pure U² scaling.

### Option 4B: Allow Overshoot Shell (Ballast)

Extend density profile to allow ρ > 1:

```python
def density_with_ballast(self, r, R_c, A_deficit, R_shell, A_ballast):
    """
    ρ(r) = ρ_vac - A_deficit·f_core(r; R_c) + A_ballast·f_shell(r; R_shell)

    A_ballast > 0 creates ρ > ρ_vac overshoot (compression/ballast).
    """
    # Core deficit (as before)
    delta_rho_core = -A_deficit * self._core_profile(r, R_c)

    # Ballast shell (new)
    delta_rho_shell = A_ballast * self._shell_profile(r, R_shell)

    rho = RHO_VAC + delta_rho_core + delta_rho_shell

    return rho
```

Then add bulk potential:

```python
def bulk_energy(self, R_c, A_deficit, R_shell, A_ballast):
    """
    E_bulk = ∫ [λ_b/2 (ρ-ρ_vac)² + κ_b/4 (ρ-ρ_vac)⁴] dV

    Penalizes deviation from vacuum (both deficit and overshoot).
    """
    rho = self.density_with_ballast(self.r, R_c, A_deficit, R_shell, A_ballast)
    delta_rho = rho - RHO_VAC

    potential_density = 0.5 * self.lambda_b * delta_rho**2 + 0.25 * self.kappa_b * delta_rho**4

    integrand = potential_density * self.r**2
    E_bulk = 4 * np.pi * np.trapz(integrand, self.r)

    return E_bulk
```

**Expected outcome**: τ solution will develop A_ballast > 0 (compression shell) to pay for the extra mass, while e, μ stay at A_ballast ≈ 0.

### Success Criteria for Test 4

**Pass**:
- E_τ/E_μ moves toward 16.8
- S_τ/S_μ approaches 1.0
- χ² drops to O(1-100)
- New parameters (R_shell, A_ballast) are well-determined for τ, near-zero for e, μ

**Fail**:
- New parameters poorly determined (wide uncertainties)
- Or: all leptons develop ballast (not τ-selective)
- Or: χ² doesn't improve

---

## Stop/Go Decision Tree

```
START
  ↓
Test 1 (Widen Bounds)
  ↓
  ├─ PASS (χ² < 1000, S_τ/S_μ < 1.5)
  │   → DONE: Bound artifact identified
  │   → Publish with widened bounds
  │
  └─ FAIL (Still saturates)
      ↓
    Test 2 (Profile Sensitivity)
      ↓
      ├─ FAIL (I_τ/I_μ ≈ 1.0)
      │   → FIX E_circ implementation
      │   → Re-run Test 1
      │
      └─ PASS (I_τ/I_μ ≠ 1.0)
          ↓
        Test 3 (Two-Lepton Fit)
          ↓
          ├─ PASS (β_min ≈ 3.058 for e,μ)
          │   → DONE: τ is genuine outlier
          │   → Publish: "e,μ validate β=3.058; τ puzzle"
          │
          └─ FAIL (β_min > 3.10 even for e,μ)
              ↓
            Test 4 (Add Ballast Physics)
              ↓
              Test 4A (Density-weighted E_circ) first
              If fails → Test 4B (Overshoot shell)
```

---

## Publishable Narratives by Outcome

### Narrative A (Test 1 passes): Constraint Artifact

**Title**: "Lepton Mass Spectrum from Quantum Fluid Vortices: Identifiability and Parameter-Box Effects"

**Key claims**:
- Hill vortex model successfully describes all three lepton masses with one global β
- Initial bound choices artificially constrained heavy-lepton (τ) parameter space
- Widened bounds restore universal energy→mass scaling
- Demonstrates importance of identifiability analysis in under-constrained models

**Strength**: Clean resolution, no new physics needed

### Narrative B (Test 3 passes): Tau is Outlier

**Title**: "Light Lepton Masses from Quantum Fluid Vortices and the Tau Anomaly"

**Key claims**:
- Electron and muon masses confirm β = 3.058 ± 0.02 (from α via Golden Loop)
- Circulation-dominated energy scaling validated for m < 200 MeV
- Tau (m_τ ≈ 2 m_proton) exhibits systematic 46% energy deficit
- Deviation consistent with hadronic mass-scale transition where charge-circulation and bulk-compression physics compete
- Tau instability (τ_τ = 2.9×10⁻¹³ s) may reflect this regime tension

**Strength**: Clear two-lepton validation + well-quantified τ puzzle as future work

### Narrative C (Test 4A/B succeeds): Heavy Vortex

**Title**: "Lepton Mass Spectrum from Light to Heavy: Charge-Circulation to Ballast Transition"

**Key claims**:
- Electron, muon: pure circulation-dominated vortices
- Tau: "heavy vortex" carrying ballast (compression/overshoot shell)
- Kinetic energy becomes density-weighted at hadronic scale
- Label-free mechanism: ballast emerges from field equations, not lepton identity
- Predicts fourth-generation lepton (if exists) would show enhanced ballast signature

**Strength**: Unified framework across all three generations with mechanistic explanation

---

## Files to Create

### Immediate (Tests 1-2):
1. `test_widened_bounds.py` - Test 1 implementation
2. `test_circulation_prefactor.py` - Test 2 implementation
3. `compare_bound_regimes.py` - Visualization of old vs new bounds results

### After Test 1/2 results:
4. `test_two_lepton_fit.py` - Test 3 implementation
5. `compare_two_vs_three_lepton.py` - β scan comparison

### Only if needed:
6. `circulation_energy_density_weighted.py` - Test 4A implementation
7. `density_profile_with_overshoot.py` - Test 4B implementation

---

## Current Status

- [x] Diagnostic phase complete (tau_collapse_diagnostics.py, DIAGNOSTIC_SUMMARY.md)
- [x] F_t emergent-time test complete (hypothesis falsified)
- [x] Energy component breakdown complete (E_circ dominance confirmed)
- [ ] Test 1: Widen bounds **← NEXT**
- [ ] Test 2: Circulation prefactor
- [ ] Test 3: Two-lepton fit
- [ ] Test 4: Ballast/compression (if needed)

---

## References

- `tau_collapse_diagnostics.py` - Shows S_τ/S_μ ≈ 1.86 across all closures
- `DIAGNOSTIC_SUMMARY.md` - Complete quantification of 46% energy shortfall
- `test_emergent_time_factor.py` - F_t = ⟨1/ρ⟩ ruled out
- `energy_component_breakdown.py` - E_circ dominance and I_ℓ = E_circ/U² analysis
