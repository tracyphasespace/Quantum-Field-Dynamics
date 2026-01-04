# Soliton Stability Mechanism: Formal Definition

**Version**: 1.0
**Date**: 2026-01-03
**Formalization**: `QFD/Hydrogen/PhotonSolitonStable.lean`

## Executive Summary

This document defines the formal meaning of "**soliton stability**" and "**non-dispersive propagation**" in the QFD framework as implemented in Lean 4.

**Key Result**: A stable soliton evolves as `Evolve(t, c) = PhaseRotate(θ) ∘ Shift(x, c)` — the shape is preserved; only translation and internal phase rotation occur.

## 1. The Physical Picture

### 1.1 What is a Soliton?

In QFD, particles (electron, proton, photon) are **localized, coherent excitations** of the Ψ-field (multivector-valued field on Cl(3,3)). A soliton is a configuration that:

1. **PhaseClosed**: Topological winding number is quantized (∈ ℤ)
2. **OnShell**: Satisfies the field equations (Euler-Lagrange equations)
3. **FiniteEnergy**: Total energy integral converges

These three conditions guarantee *existence* but not *stability*.

### 1.2 The Stability Problem

A configuration can satisfy the three gates but still be **unstable**:
- **Dispersive decay**: The wavepacket spreads out over time
- **Dissipative collapse**: Energy radiates away as the configuration relaxes

**Soliton Stability** addresses this: the configuration is a **local energy minimum** that resists both dispersion and dissipation.

### 1.3 The Vacuum Superfluid Mechanism

QFD posits that the vacuum medium is a **superfluid** (zero viscosity). This has two consequences:

1. **No dissipation**: Energy cannot be lost to "friction" → total energy is conserved
2. **Dispersion cancellation**: The vacuum has both:
   - **Linear dispersion** (from stiffness β) → tends to spread waves
   - **Nonlinear focusing** (from saturation λ_sat) → tends to concentrate waves
   - At the **soliton balance point**, these exactly cancel

**Result**: A stable soliton propagates **without changing shape** — it only translates and accumulates internal phase.

## 2. Formal Definition (Lean 4)

### 2.1 The Stability Predicate

```lean
structure QFDModelStable (Point : Type u) extends QFDModel Point where
  -- Stability gate: local energy minimum at fixed topological invariants
  Stable : Config Point → Prop

  -- Time evolution operator
  Evolve : ℝ → Config Point → Config Point

  -- Spatial translation (1D for now, generalizable to ℝ³)
  Shift : ℝ → Config Point → Config Point

  -- Internal phase/rotor evolution (group action on internal degrees of freedom)
  PhaseRotate : ℝ → Config Point → Config Point
```

### 2.2 The Non-Dispersive Evolution Axiom

**The Central Statement**:

```lean
evolve_is_shift_phase_of_stable :
  ∀ c,
    PhaseClosed c → OnShell c → FiniteEnergy c → Stable c →
    ∀ t, ∃ x θ, Evolve t c = PhaseRotate θ (Shift x c)
```

**Physics Interpretation**:
- If `c` is a stable soliton (4 gates: PhaseClosed ∧ OnShell ∧ FiniteEnergy ∧ Stable)
- Then at any time `t`, the evolved state `Evolve(t, c)` is equal to:
  - A spatial translation `Shift(x, c)` (the soliton moved to position x)
  - Followed by internal phase rotation `PhaseRotate(θ, ·)` (internal degrees of freedom evolved)

**This is the formal handle for "shape invariance"**.

### 2.3 Conservation Laws

The model enforces three conservation laws during evolution:

```lean
evolve_preserves_charge : ∀ t c, (Evolve t c).charge = c.charge
evolve_preserves_energy : ∀ t c, (Evolve t c).energy = c.energy
evolve_preserves_momentum : ∀ t c, Momentum (Evolve t c) = Momentum c
```

These follow from Noether's theorem (continuous symmetries → conserved quantities).

### 2.4 Symmetry Invariance

The existence gates and stability are preserved under translation and phase rotation:

```lean
gates_invariant_under_shift_phase :
  ∀ c x θ,
    (PhaseClosed c ∧ OnShell c ∧ FiniteEnergy c) →
    (PhaseClosed (PhaseRotate θ (Shift x c)) ∧
     OnShell (PhaseRotate θ (Shift x c)) ∧
     FiniteEnergy (PhaseRotate θ (Shift x c)))

stable_invariant_under_shift_phase :
  ∀ c x θ, Stable c → Stable (PhaseRotate θ (Shift x c))
```

**Physical Meaning**: The vacuum medium is **translation-invariant** and **rotationally symmetric** in internal space. Moving or rotating a stable soliton produces another stable soliton.

## 3. Proven Theorems

### 3.1 Soliton Persistence

**Theorem** (`stableSoliton_persists`):
```lean
theorem stableSoliton_persists
    (s : StableSoliton M) (t : ℝ) :
    ∃ s' : StableSoliton M,
      (s' : Config Point) = M.Evolve t (s : Config Point)
```

**Plain English**: A stable soliton stays a stable soliton under time evolution.

**Physical Interpretation**:
- **No viscosity** → no dissipation → coherence is preserved
- **Soliton balance** → no dispersion → shape is preserved
- The soliton **persists indefinitely** as long as the medium remains superfluid

**Proof Strategy**:
1. Use `evolve_is_shift_phase_of_stable` to express evolved state as `PhaseRotate(θ) ∘ Shift(x, c)`
2. Use `gates_invariant_under_shift_phase` to prove new state satisfies 3 existence gates
3. Use `stable_invariant_under_shift_phase` to prove new state is stable
4. Package result as a `StableSoliton` term

### 3.2 Constructor Theorem

**Theorem** (`stableSoliton_of_config`):
```lean
theorem stableSoliton_of_config
    (c : Config Point)
    (hC : M.PhaseClosed c)
    (hS : M.OnShell c)
    (hF : M.FiniteEnergy c)
    (hStab : M.Stable c) :
    StableSoliton M
```

**Plain English**: If you can exhibit a configuration meeting all 4 gates, you can construct a stable soliton term in Lean.

**Proof Method**: Direct construction using the provided witnesses.

## 4. Photon Kinematics

### 4.1 The PhotonWave Structure

```lean
structure PhotonWave where
  ω : ℝ   -- Angular frequency
  k : ℝ   -- Wavenumber
  λw : ℝ  -- Wavelength
  hλ : λw ≠ 0
  hkλ : k * λw = 2 * Real.pi  -- Exact geometric identity
```

**Key Constraint**: `k · λw = 2π` is **exact**, not approximate. This encodes the absence of dispersion in the vacuum ground state.

### 4.2 Momentum-Wavelength Relation

**Theorem** (`momentum_eq_hbar_twoPi_div_lambda`):
```lean
theorem momentum_eq_hbar_twoPi_div_lambda (γ : PhotonWave) :
    momentum M γ = (M.ℏ * (2 * Real.pi)) / γ.λw
```

**Physics**: This is the de Broglie relation `p = h/λ` proven from geometry.

**Proof**:
1. `p = ℏ k` (definition)
2. `k = 2π/λ` (from `k λ = 2π`)
3. Therefore `p = ℏ (2π/λ)`

### 4.3 Massless Dispersion

**Definition** (`MasslessDispersion`):
```lean
def MasslessDispersion (γ : PhotonWave) : Prop :=
  γ.ω = M.cVac * γ.k
```

**Theorem** (`energy_eq_cVac_mul_momentum`):
```lean
theorem energy_eq_cVac_mul_momentum
    (γ : PhotonWave)
    (hDisp : MasslessDispersion (M := M) γ) :
    energy M γ = M.cVac * momentum M γ
```

**Physics**: This is the hallmark `E = c·p` relation for massless particles.

**Proof**:
```
E = ℏ ω           (definition)
  = ℏ (c k)       (massless dispersion)
  = c (ℏ k)       (commutativity)
  = c p           (definition of p)
```

## 5. Absorption with Momentum Recoil

### 5.1 The HStateP Structure

```lean
structure HStateP (M : QFDModelStable Point) where
  H : QFDModel.Hydrogen (M.toQFDModel)  -- The hydrogen system
  n : ℕ                                 -- Energy level
  P : ℝ                                 -- Center-of-mass momentum (1D)
```

This extends the basic hydrogen state with a momentum tag.

### 5.2 Absorption with Recoil

**Definition** (`AbsorbsP`):
```lean
def AbsorbsP (M : QFDModelStable Point)
    (s : HStateP M) (γ : PhotonWave) (s' : HStateP M) : Prop :=
  s'.H = s.H ∧
  s.n < s'.n ∧
  s'.energy = s.energy + (PhotonWave.energy (M := M) γ) ∧
  s'.momentum = s.momentum + (PhotonWave.momentum (M := M) γ)
```

**Key Addition**: The fourth line `s'.momentum = s.momentum + p_γ` enforces momentum conservation.

### 5.3 Absorption Theorem

**Theorem** (`absorptionP_of_gap`):
```lean
theorem absorptionP_of_gap
    {M : QFDModelStable Point} {H : QFDModel.Hydrogen (M.toQFDModel)}
    {n m : ℕ} (hnm : n < m)
    (P : ℝ) (γ : PhotonWave)
    (hGap : PhotonWave.energy (M := M) γ = M.ELevel m - M.ELevel n) :
    AbsorbsP M ⟨H, n, P⟩ γ ⟨H, m, P + PhotonWave.momentum (M := M) γ⟩
```

**Plain English**: If the photon energy matches the energy gap, then absorption is valid with the hydrogen system recoiling to conserve momentum.

**Proof**: Energy and momentum bookkeeping from definitions.

## 6. The Axiom → Derivation Path

### 6.1 Current State (This File)

**Axioms**:
1. `Stable c` is a **primitive predicate** (provided by the model)
2. `evolve_is_shift_phase_of_stable` is **asserted** (not derived)

**Justification**: This allows us to prove theorems about soliton dynamics without solving PDEs in Lean.

### 6.2 Future Upgrade Path

To derive these from first principles:

**Step 1**: Define stability via energy functional
```lean
def EnergyFunctional : Config Point → ℝ := ...

def Stable (c : Config Point) : Prop :=
  IsLocalMin EnergyFunctional c ∧ TopologicalInvariant c = fixed_value
```

**Step 2**: Prove evolution preserves energy
```lean
theorem evolve_preserves_energy_functional :
  ∀ t c, EnergyFunctional (Evolve t c) = EnergyFunctional c
```
(This follows from Noether's theorem + time-translation symmetry)

**Step 3**: Prove shape invariance from symmetry
```lean
theorem evolve_is_shift_phase_of_stable :
  ∀ c, IsLocalMin EnergyFunctional c →
       ∀ t, ∃ x θ, Evolve t c = PhaseRotate θ (Shift x c)
```
(This follows from: local min + conserved energy + symmetry → orbit is shift+phase)

**Challenge**: This requires formalizing the calculus of variations and Noether's theorem in Lean 4, which is a major undertaking.

**Verdict**: The current approach (axiom + typed interface) is appropriate for the present stage of the formalization.

## 7. Integration with Existing QFD Theory

### 7.1 Connection to Vacuum Parameters

The vacuum parameters (α, β, λ_sat) from the base `QFDModel` control stability:

- **α** (fine-structure coupling): Controls EM interaction strength
- **β** (vacuum stiffness): Controls c_vac = √(β/ρ) and dispersion
- **λ_sat** (saturation scale): Controls nonlinear focusing

The **soliton balance condition** is:
```
Linear dispersion (∝ β) = Nonlinear focusing (∝ 1/λ_sat²)
```

When this balance holds, `Stable c` is true and `evolve_is_shift_phase_of_stable` applies.

### 7.2 Connection to ShapeInvariant (PhotonSoliton.lean)

The earlier `ShapeInvariant` predicate in `PhotonSoliton.lean` is a **precursor** to the full `Stable` predicate:

```lean
-- PhotonSoliton.lean (earlier version)
ShapeInvariant : Config Point → Prop
  -- "Dispersion ↔ Nonlinear focusing balance"
  -- d(Width)/dt = 0

-- PhotonSolitonStable.lean (this file)
Stable : Config Point → Prop
  -- "Local energy minimum + preserved under evolution"
```

**Relationship**: `Stable c → ShapeInvariant c` (stability implies shape invariance).

The converse is not necessarily true (a configuration can have d(Width)/dt = 0 without being a local energy minimum).

### 7.3 Future Work

1. **Define `Stable` from energy functional** (as outlined in Section 6.2)
2. **Prove `Stable c → ShapeInvariant c`** (connect the two predicates)
3. **Extend to 3D**: Replace `Shift : ℝ → Config → Config` with `Shift : ℝ³ → Config → Config`
4. **Include spin**: Add SO(3) rotation to `PhaseRotate` (currently only internal phase)
5. **Multi-soliton interactions**: Prove scattering theorems for soliton collisions

## 8. References

**Lean 4 Files**:
- `QFD/Hydrogen/PhotonSoliton.lean` — Base framework (3 gates, photon kinematics)
- `QFD/Hydrogen/PhotonSolitonStable.lean` — This formalization (4th gate + evolution)

**Mathematical Background**:
- Noether's Theorem: Symmetries → Conservation Laws
- Lyapunov Stability: Energy functionals and local minima
- Soliton Theory: Nonlinear PDEs with dispersion-focusing balance

**Physical Motivation**:
- Vacuum Superfluid Hypothesis: Zero viscosity → no dissipation
- Saturation Mechanism: Nonlinearity prevents dispersion
- Geometric Algebra: Cl(3,3) structure provides internal phase space

## 9. Summary Table

| Concept | Predicate/Operator | Physical Meaning | Status |
|---------|-------------------|------------------|--------|
| **Existence** | `PhaseClosed ∧ OnShell ∧ FiniteEnergy` | Configuration satisfies field equations | ✅ Defined |
| **Stability** | `Stable` | Local energy minimum | ⚠️ Axiom (to be derived) |
| **Evolution** | `Evolve` | Time dynamics operator | ⚠️ Axiom (to be derived) |
| **Shape Invariance** | `evolve_is_shift_phase_of_stable` | Translation + phase rotation only | ⚠️ Axiom (to be derived) |
| **Persistence** | `stableSoliton_persists` | Soliton survives evolution | ✅ Proven |
| **Photon Momentum** | `p = ℏ k` | De Broglie relation | ✅ Proven |
| **Massless Dispersion** | `ω = c k` | Light cone structure | ✅ Defined |
| **Energy-Momentum** | `E = c p` | Massless particle | ✅ Proven |
| **Absorption (recoil)** | `AbsorbsP` | Energy + momentum conservation | ✅ Defined |

**Legend**:
- ✅ Proven: Derived from other axioms/definitions
- ⚠️ Axiom: Asserted without proof (to be derived in future work)

---

**Document Status**: Ready for use
**Next Update**: When `Stable` is derived from energy functional (not yet scheduled)
