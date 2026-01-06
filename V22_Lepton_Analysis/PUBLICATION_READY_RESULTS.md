# V22 Hill Vortex Lepton Mass Investigation: Publication-Grade Results

## Summary of Validated Claims

Using a Hill spherical vortex velocity field with a parabolic density depression and a quadratic stiffness potential, we find that **for a single fixed stiffness parameter β = 3.1**, the model admits optimized solutions matching the electron, muon, and tau mass ratios to high numerical precision. Across these solutions, the radius R and density amplitude remain in a narrow range near the cavitation limit, while the circulation parameter U controls the mass hierarchy and follows approximately U ∝ √m.

## Energy Functional (As Implemented)

```
E_total = E_circulation - E_stabilization

E_circulation = ∫ (1/2) ρ(r) v²(r,θ) dV

E_stabilization = ∫ β (δρ)² dV
```

Where:
- `ρ(r) = ρ_vac - amplitude × (1 - r²/R²)` for r < R (parabolic depression)
- `δρ(r) = ρ(r) - ρ_vac = -amplitude × (1 - r²/R²)` for r < R
- `v(r,θ)` from Hill vortex stream function with velocity scale U
- All energies are **dimensionless mass ratios** (in units of m_e c²)

## Analytic Cross-Check: E_stabilization

For the parabolic density depression δρ(r) = -a(1 - r²/R²) with r < R:

```
E_stab = ∫ β δρ² dV
       = 4πβa² ∫₀ᴿ (1 - r²/R²)² r² dr
       = 4πβa² R³ ∫₀¹ (1 - x²)² x² dx
       = 4πβa² R³ (8/105)
       = (32π/105) β a² R³
```

**Numerical validation**: For β = 3.1, R ≈ 0.45, a ≈ 0.90:
```
E_stab ≈ (32π/105) × 3.1 × (0.90)² × (0.45)³
       ≈ 0.30
```

This matches the observed E_stab ≈ 0.2-0.3 across all three leptons and explains why **E_stabilization is nearly constant** - it depends only on (β, R, amplitude), not on U.

## Fitted Parameters and Energies

All energies reported in **electron-mass units** (multiply by m_e c² = 0.511 MeV to obtain MeV).

### Electron (Reference)
```
R = 0.4392    U = 0.0242    amplitude = 0.8990
E_circulation   = 1.217
E_stabilization = 0.217
E_total         = 1.000
Target: m_e/m_e = 1.000
Residual: |E_total - target| = 6.2×10⁻⁵
```

### Muon (Enhanced Circulation)
```
R = 0.4581    U = 0.3146    amplitude = 0.9433
E_circulation   = 207.0
E_stabilization = 0.26
E_total         = 206.77
Target: m_μ/m_e = 206.768
Residual: |E_total - target| = 2×10⁻³
```

### Tau (Further Enhanced)
```
R = 0.4800    U = 1.2894    amplitude = 0.9599
E_circulation   = 3477.5
E_stabilization = 0.31
E_total         = 3477.2
Target: m_τ/m_e = 3477.228
Residual: |E_total - target| = 2×10⁻⁵
```

## Observed Scaling Relations

### 1. Circulation Energy Dominates
```
E_total ≈ E_circ - E_stab
        ≈ E_circ - O(0.3)
        ≈ E_circ  (for μ, τ)
```

This is **NOT** a "huge cancellation → tiny residual" mechanism. The stabilization term is **small** compared to circulation for heavier leptons. The mass hierarchy is simply E_total ≈ E_circ.

### 2. E_circulation ∝ U² (Expected from Kinetic Energy)

Since v ∝ U in the Hill vortex, we expect E_circ ∝ U².

**Numerical validation**:
```
Particle   U        U²      E_circ    (E_circ)/(E_circ)_e
Electron   0.024    1.0×    1.217     1.0×
Muon       0.315    172×    207.0     170×
Tau        1.289    2877×   3477.5    2858×
```

The ratios U² and E_circ track within ~1%, confirming the quadratic scaling.

### 3. Circulation Parameter Follows U ∝ √m

**Predicted from E_circ ∝ U²**:
If E_total ≈ E_circ ∝ U², then U ∝ √(E_total) ∝ √m.

**Numerical validation**:
```
Particle   m/m_e    √(m/m_e)   U       U/U_e
Electron   1        1.0        0.024   1.0
Muon       207      14.4       0.315   13.1  (within 10%)
Tau        3477     59.0       1.289   53.7  (within 10%)
```

The fitted U values follow the √m scaling to within ~10%.

### 4. Geometric Parameters Remain Constrained

```
Particle   R        amplitude   R/R_e   amplitude/amplitude_e
Electron   0.439    0.899       1.00    1.00
Muon       0.458    0.943       1.04    1.05
Tau        0.480    0.960       1.09    1.07
```

While U varies by factor of ~54×, R varies only by ~9% and amplitude by ~7%. Both remain near the **cavitation limit** (amplitude → ρ_vac).

## What This Demonstrates

### Validated Claims

1. **Single-parameter stiffness**: For fixed β = 3.1, the Hill vortex + parabolic density ansatz admits solutions matching all three lepton mass ratios without adjusting β.

2. **Circulation-dominated hierarchy**: Mass differences arise primarily from different circulation velocities U, while stabilization remains nearly constant.

3. **Consistent scaling**: The fitted parameters obey the expected E_circ ∝ U² and U ∝ √m relations.

4. **Narrow geometric window**: Solutions cluster near cavitation (amplitude → ρ_vac) and similar radii (R ≈ 0.44-0.48).

### Nature of the Fit

The optimization varies **three continuous parameters** (R, U, amplitude) to match **one scalar target** (mass ratio) per lepton. The close numerical agreement is therefore a **fit**, not a prediction.

The scientifically meaningful result is that such solutions exist for a single β across all three leptons.

## Critical Limitations and Required Follow-Up

### 1. Physical Admissibility of U

**Issue**: U ≈ 1.289 for tau exceeds 1 in a velocity-normalized scheme.

**Interpretation options**:
- U is a circulation parameter not bounded by c (dimensionally a length × velocity)
- Missing relativistic corrections for heavier leptons
- U represents a mode number or winding number, not a direct velocity

**Required**: Clarify the physical interpretation and units of U.

### 2. Numerical Convergence Not Demonstrated

**Current**: Integration grid (nr, nθ) = (100, 20) with Simpson's rule

**Required**: Grid convergence study showing parameter stability:
```
Grid            R        U        E_total    Δ from finest
(100, 20)       0.4392   0.0242   1.0001     ?
(200, 40)       ?        ?        ?          ?
(400, 80)       ?        ?        ?          ?
```

### 3. Solution Uniqueness Unknown

**Current**: Single optimization run per lepton

**Required**: Multi-start robustness test (20-50 random initial seeds) showing:
- Whether multiple local minima exist
- Distribution of fitted parameters
- Selection principle if solutions are degenerate (stability, quantization, etc.)

### 4. Functional Form Sensitivity

**Current**: Only parabolic density depression tested

**Required**: Profile sensitivity analysis with alternative forms:
- Quartic core: δρ ∝ (1 - r²/R²)²
- Gaussian core: δρ ∝ exp(-r²/R²)
- Power-law: δρ ∝ (1 - r/R)^n

**Test**: Does β = 3.1 still work without retuning, or is the parabolic form essential?

### 5. From Fit to Prediction

**Current status**: 3 degrees of freedom → 1 constraint (mass)

**Path to prediction**: Reduce DOF via physical constraints:
- Fix amplitude from cavitation/charge quantization
- Fix R from stability (second-variation analysis)
- Then U (or discrete mode n) becomes predictive

### 6. Units and Dimensionalization

**Current**: "Dimensionless" formulation with implicit length/energy scales

**Required for "universal β" claim**:
- Explicit nondimensionalization showing how β_cosmology and β_particle map to the same dimensionless parameter
- Unit conversion protocol from dark energy scale (GeV⁴) to lepton mass scale (MeV)

This is acknowledged as unresolved in the current codebase.

### 7. Multi-Component Structure Not Implemented

**Scripts compute**: Poloidal Hill vortex only (ψ_s)

**Lean specs mention**: 4-component structure (ψ_s, ψ_b0, ψ_b1, ψ_b2)

**Charge quantization (Q*)**: Asserted in prose but not enforced or derived in mass-ratio scripts

**Required**: Either implement full 4-component solver or clearly state this is a reduced (poloidal-only) model.

## Recommended Next Steps for Publication

### Immediate (Required for Publication)

1. **Grid convergence study** (1 week) - Verify numerics are robust
2. **Multi-start robustness** (3 days) - Characterize solution landscape
3. **Rewrite "cancellation" narrative** - Clarify E_total ≈ E_circ for μ, τ
4. **Clarify U interpretation** - Address superluminal issue
5. **Unit transparency** - State all energies in electron-mass units explicitly

### Near-Term (Strengthen Claims)

6. **Profile sensitivity** (1 week) - Test functional form dependence
7. **Analytic stability analysis** (2 weeks) - Second variation of action
8. **Cavitation constraint implementation** (1 week) - Remove amplitude as free parameter

### Long-Term (Full Theory)

9. **4-component implementation** - Match Lean formal specs
10. **Quantization principle** - Derive discrete spectrum from topology/stability
11. **Cross-scale β mapping** - Rigorous unit conversion from cosmology/nuclear

## Publication-Ready Abstract (Corrected)

We investigate whether a single vacuum stiffness parameter β can determine charged lepton mass ratios within a Hill spherical vortex model. Using a parabolic density depression and quadratic stabilization potential, we find that β = 3.1 admits optimized solutions matching the electron, muon, and tau mass ratios to within 10⁻³ to 10⁻⁵ electron masses. The mass hierarchy arises primarily from varying circulation velocity U, which scales approximately as √m, while geometric parameters (radius R and density amplitude) remain within ~10% across all three leptons. Solutions cluster near the cavitation limit, suggesting a physical constraint. While the current implementation fits three continuous parameters to one target per lepton, the existence of solutions with fixed β across three orders of magnitude in mass is a nontrivial consistency check. We identify grid convergence, solution uniqueness, and functional form sensitivity as critical hardening tests, and propose amplitude quantization via cavitation as a path from fit to prediction.

## Files Implementing These Results

**Core solvers**:
- `v22_hill_vortex_with_density_gradient.py` - Electron
- `v22_muon_refined_search.py` - Muon
- `v22_tau_test.py` - Tau

**Results**:
- `results/density_gradient_correction_results.json` - Electron parameters
- `results/muon_refined_search_results.json` - Muon parameters
- `results/tau_test_results.json` - Tau parameters

**Documentation**:
- `COMPLETE_REPLICATION_GUIDE.md` - Step-by-step replication
- This file - Publication-grade summary

## Conclusion

What we have demonstrated is **clean, internally consistent, and publishable** once claims are aligned with the implemented mathematics:

✅ Fixed β = 3.1 works across three leptons
✅ E_circ ∝ U² and U ∝ √m scaling validated
✅ Geometric parameters naturally constrained
✅ Solutions exist with reasonable residuals

⚠️ Numerical convergence not yet verified
⚠️ Solution uniqueness unknown
⚠️ "Universal β" requires explicit unit mapping
⚠️ U > 1 requires physical interpretation
⚠️ Currently a fit (3 DOF → 1 target), not a prediction

The path from "fits exist" to "modes are quantized and therefore predictive" is clear and achievable.
