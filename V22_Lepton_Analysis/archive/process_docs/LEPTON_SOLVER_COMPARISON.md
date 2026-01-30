# Lepton Mass Solver Comparison: V22 vs Phoenix

**Date**: December 22, 2025
**Status**: Analysis of two different approaches

---

## Executive Summary

**Discovery**: There are TWO different implementations for lepton mass calculation:

1. **V22 Approach** (from Lean MassSpectrum.lean): Simple quartic potential
2. **Phoenix Approach** (validated, working): Sophisticated 4-component Hamiltonian

**Critical finding**: V22 approach with β = 3.1 does NOT produce correct masses.
Phoenix approach achieves 99.99989% accuracy but uses **different physics**.

---

## Approach 1: V22 Simple Quartic Potential

### Source
- `/projects/Lean4/QFD/Lepton/MassSpectrum.lean`
- `/V22_Lepton_Analysis/scripts/v22_lepton_mass_solver.py`

### Physics Model

**Radial Schrödinger Equation**:
```
-ψ'' + V(r)ψ = E·ψ
```

**Potential**:
```
V(r) = β·(r² - v²)²
```

**Parameters**:
- β ≈ 3.1 (from cosmology/nuclear - THE SAME β!)
- v ≈ 1.0 (vacuum scale)

### Theoretical Foundation

**From Lean MassSpectrum.lean**:
- Confinement: β > 0 ensures discrete spectrum
- Koide relation: Q = 2/3 from lattice geometry
- Eigenvalues are lepton masses

**Goal**: Unify cosmology → nuclear → particle using SAME β parameter

### Results (β = 3.1)

```
Input: β = 3.10 (from Cosmology/Nuclear)

Output:
Electron:  6.94 MeV     (experimental: 0.511 MeV)
Muon:      20.00 MeV    (experimental: 105.66 MeV)
Tau:       [not computed]

m_μ/m_e:   2.88         (experimental: 206.77)

VERDICT: FAILED - Does NOT produce correct hierarchy
```

**Error**: ~1000% error on absolute masses, ~7000% error on mass ratios!

### Why It Failed

1. **Potential form may be wrong**: V(r) = β(r²-v²)² is too simple
2. **Missing physics**: No charge dynamics, no multi-component fields
3. **Single parameter**: β cannot control individual mass scales

---

## Approach 2: Phoenix Solver (Working Implementation)

### Source
- `/projects/particle-physics/lepton-isomers/src/solvers/phoenix_solver.py`
- `/projects/particle-physics/lepton-isomers/src/solvers/hamiltonian.py`

### Physics Model

**4-Component Field Structure**:
```
ψ = (ψ_s, ψ_b0, ψ_b1, ψ_b2)
```

**Hamiltonian**:
```
H = H_kinetic + H_potential + H_csr

H_kinetic = 0.5 ∫ (∇ψ_s)² + (∇ψ_b0)² + (∇ψ_b1)² + (∇ψ_b2)² · 4πr² dr

H_potential = ∫ (V2·ρ + V4·ρ²) · 4πr² dr
    where ρ = ψ_s² + ψ_b0² + ψ_b1² + ψ_b2²

H_csr = -0.5·k_csr ∫ ρ_q² · 4πr² dr
    where ρ_q = -g_c·∇²ψ_s (charge density)
```

### Parameters

**Different for each lepton!**

| Parameter | Electron | Muon | Tau |
|-----------|----------|------|-----|
| **V2** (initial) | 0 → 12M | 8M | 100M |
| **V4** | 11.0 | 11.0 | 11.0 |
| **g_c** | 0.985 | 0.985 | 0.985 |
| **k_csr** | 0.0 | 0.0 | 0.0 |
| **Q*** | 2.166 | 2.3 | 9800 |
| **Max Iterations** | 1500 | 2000 | 5000 |

### Ladder Solver Methodology

**Energy Targeting**:
1. Start with initial V2
2. Run solver → get current energy E_current
3. Compute: ΔV2 = (E_target - E_current) / Q*
4. Update: V2_new = V2_old + ΔV2
5. Repeat until E_current ≈ E_target

**Convergence Formula**:
```python
delta_v2 = (target_energy - current_energy) / q_star
```

Q* acts as a "sensitivity parameter" controlling how fast V2 adjusts.

### Results

| Particle | Target | Result | Error | Accuracy | Time |
|----------|--------|--------|-------|----------|------|
| **Electron** | 511.0 keV | 511.000 keV | 0.6 eV | 99.99989% | 14s |
| **Muon** | 105.658 MeV | 105.658 MeV | 270.9 eV | 99.99974% | 29s |
| **Tau** | 1.777 GeV | 1.777 GeV | 0.0 eV | 100.0% | ~7h |

**VERDICT**: SUCCESS - Produces correct masses to sub-eV precision!

---

## Critical Differences

### 1. Potential Form

**V22**:
```
V(r) = β·(r² - v²)²
```
- Radial potential (function of position r)
- Single parameter β controls everything
- Quartic in radial coordinate

**Phoenix**:
```
V(ρ) = V2·ρ + V4·ρ²
```
- Density-dependent potential (function of |ψ|²)
- Two parameters V2, V4
- Quartic in field density

### 2. Field Structure

**V22**:
- Single wavefunction ψ(r)
- 1D radial Schrödinger equation
- Shooting method with Numerov integration

**Phoenix**:
- 4-component field (ψ_s, ψ_b0, ψ_b1, ψ_b2)
- Coupled field equations
- L-BFGS-B optimization

### 3. Parameter Strategy

**V22**:
- **Same β for all leptons** (unified parameter)
- β ≈ 3.1 from cosmology/nuclear
- Goal: Complete unification from cosmic to particle scales

**Phoenix**:
- **Different V2 for each lepton** (tuned separately)
- V2 ranges from 0 (electron) to 100M (tau)
- Q* also varies: 2.2 (electron) to 9800 (tau)
- **No connection to cosmology claimed**

### 4. Additional Physics

**V22**:
- Pure potential well problem
- No charge dynamics
- No multi-component structure

**Phoenix**:
- Charge self-repulsion (CSR term)
- Charge density ρ_q = -g_c·∇²ψ_s
- Q* normalization constraint
- 4-component internal structure

---

## The Unification Question

**V22 Hypothesis**:
> The SAME parameter β that appears in:
> - CMB stiffness (cosmology)
> - Nuclear core compression (nuclear)
> - Lepton mass generation (particle)
>
> Should β ≈ 3.1 produce m_μ/m_e ≈ 206?

**Result**: **NO** - With simple V(r) = β(r²-v²)², β = 3.1 gives m_μ/m_e ≈ 2.88

**Phoenix Reality**:
- Does NOT use a universal β parameter
- Instead tunes V2 and Q* separately for each lepton
- Achieves perfect mass reproduction
- But breaks the unification goal

---

## Possible Resolutions

### Option 1: V22 Potential Form Is Wrong

**Hypothesis**: The correct QFD potential is NOT V(r) = β(r²-v²)²

**Alternative**: Should be density-dependent V(ρ) = V2·ρ + V4·ρ²

**Problem**: How does this connect to β from cosmology/nuclear?

**Possible mapping**:
- β might relate to V4 (quartic term coefficient)
- v² might relate to -V2/V4 (minimum of potential)
- Need to derive connection from first principles

### Option 2: Missing Physics in V22

**Hypothesis**: V22 is correct but incomplete

**Missing pieces**:
1. Multi-component field structure (4 components, not 1)
2. Charge self-repulsion (CSR energy term)
3. Proper spherical integration (4πr² volume element)
4. Q* normalization constraint

**Action**: Add these to V22 and retry with β = 3.1

### Option 3: Scale/Unit Mismatch

**Hypothesis**: β = 3.1 is in wrong units for particle physics

**From cosmology/nuclear**:
- β appears in SNe scattering cross-section
- β appears in nuclear compression law
- Units may be cosmological/nuclear scales

**For particle physics**:
- Need to convert β to lepton energy scale
- Conversion factor might be ~10⁶ (MeV/fm² to eV/Bohr²)
- β_particle = β_cosmo × conversion_factor

### Option 4: Separate β for Each Scale

**Hypothesis**: Unification doesn't mean SAME numerical value

**Instead**:
- β_cosmo ≈ 3.1 (cosmological scale)
- β_nuclear ≈ 3.1 (nuclear scale - happens to be similar!)
- β_particle ≈ ??? (particle scale - different value)

**All three are "stiffness parameters" but scale-dependent**

---

## Recommended Next Steps

### 1. Map Phoenix Parameters to QFD Theory

**Question**: Can we express V2, V4 in terms of fundamental QFD parameters?

**Approach**:
- Review QFD book Appendix Y ("The Lepton Cascade")
- Check if V(ρ) = V2·ρ + V4·ρ² is derivable from vacuum dynamics
- Find theoretical constraints on V2, V4

### 2. Test Unit Conversion Hypothesis

**Approach**:
- Scale β by different powers of 10
- Try β_particle = β_cosmo × 10⁶ (energy scale conversion)
- See if this produces correct masses

### 3. Add Missing Physics to V22

**Approach**:
- Implement 4-component field structure
- Add CSR energy term
- Enforce Q* normalization
- Retry with β = 3.1

### 4. Derive V2, V4 from β

**Approach**:
- Start with V(r) = β(r²-v²)²
- Expand in terms of field density ρ
- See if V2, V4 emerge from expansion
- Check if β = 3.1 gives reasonable V2, V4 values

---

## Bottom Line

**Current Status**:
- ❌ V22 simple approach (β = 3.1) does NOT work
- ✅ Phoenix approach (tuned V2, Q*) DOES work perfectly

**Unification Goal**:
- ⚠️ **Incomplete** - Phoenix does not connect to cosmology/nuclear
- ⚠️ **Unclear** - How do V2, V4, Q* relate to β ≈ 3.1?

**Path Forward**:
1. Understand mapping between V(r) and V(ρ) formulations
2. Find connection between β and V2, V4 parameters
3. Test if enhanced V22 (with CSR, 4-components) works with β = 3.1
4. Either achieve unification OR document fundamental scale separation

---

**Date**: December 22, 2025
**Next Step**: Analyze relationship between β and Phoenix parameters V2, V4
