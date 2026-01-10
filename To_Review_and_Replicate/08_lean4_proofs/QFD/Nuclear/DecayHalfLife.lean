import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Data.Real.Basic
import Mathlib.Tactic

/-!
# QFD: Nuclear Decay Half-Life from Vacuum Tunneling

**Subject**: Deriving radioactive decay rates from vacuum barrier tunneling
**Reference**: Chapter 14 (Nuclear Physics from Vacuum Geometry)

This module formalizes how radioactive decay half-lives arise from quantum
tunneling through the vacuum's potential barrier. The key insight is that
the decay rate λ depends exponentially on the barrier penetrability, which
is determined by the vacuum stiffness parameter β.

## Physical Model

1. **The Barrier**: The vacuum creates a confining potential V(r) around nucleons
2. **Tunneling**: Particles can quantum-mechanically penetrate the barrier
3. **Half-Life**: T₁/₂ = ln(2)/λ where λ is the tunneling rate

## Key Results

- `tunneling_rate_exponential`: λ ∝ exp(-2κL) where κ is barrier stiffness
- `half_life_from_rate`: T₁/₂ = ln(2)/λ (definition)
- `half_life_barrier_dependence`: Higher barriers → longer half-lives
- `half_life_positive`: T₁/₂ > 0 for any positive decay rate
-/

noncomputable section

namespace QFD.Nuclear.DecayHalfLife

open Real

-- =============================================================================
-- PART 1: BARRIER AND TUNNELING STRUCTURES
-- =============================================================================

/-- A potential barrier characterized by height V₀ and width L. -/
structure PotentialBarrier where
  V₀ : ℝ  -- Barrier height (in energy units)
  L : ℝ   -- Barrier width
  h_V₀_pos : 0 < V₀
  h_L_pos : 0 < L

/-- The decay state of a radioactive nucleus. -/
structure DecayingNucleus where
  E_particle : ℝ       -- Particle energy
  barrier : PotentialBarrier
  h_E_pos : 0 < E_particle
  h_below_barrier : E_particle < barrier.V₀  -- Tunneling requires E < V₀

/-- The effective wave vector inside the barrier: κ = √(2m(V₀-E)/ℏ²) -/
def barrier_wave_vector (nucleus : DecayingNucleus) (m ℏ : ℝ)
    (h_m : 0 < m) (h_ℏ : 0 < ℏ) : ℝ :=
  Real.sqrt (2 * m * (nucleus.barrier.V₀ - nucleus.E_particle) / ℏ^2)

-- =============================================================================
-- PART 2: TUNNELING RATE
-- =============================================================================

/-- The tunneling rate through a barrier.

**Physical Model**: Using WKB approximation, the transmission probability
through a rectangular barrier is T ≈ exp(-2κL). The decay rate is
proportional to this transmission probability times the attempt frequency.

**Parameters**:
- `κ`: Effective wave vector inside the barrier
- `L`: Barrier width
- `ν₀`: Attempt frequency (how often particle "tries" to escape)
-/
def tunneling_rate (κ L ν₀ : ℝ) : ℝ :=
  ν₀ * Real.exp (-2 * κ * L)

/-- The decay rate is positive when attempt frequency and barrier are positive. -/
theorem tunneling_rate_pos (κ L ν₀ : ℝ) (h_ν₀ : 0 < ν₀) :
    0 < tunneling_rate κ L ν₀ := by
  unfold tunneling_rate
  exact mul_pos h_ν₀ (Real.exp_pos _)

/-- Higher barrier wave vector → lower tunneling rate (monotonicity). -/
theorem tunneling_rate_decreasing_in_kappa
    (κ₁ κ₂ L ν₀ : ℝ)
    (h_L : 0 < L)
    (h_ν₀ : 0 < ν₀)
    (h_order : κ₁ < κ₂) :
    tunneling_rate κ₂ L ν₀ < tunneling_rate κ₁ L ν₀ := by
  unfold tunneling_rate
  have h1 : -2 * κ₂ * L < -2 * κ₁ * L := by
    have h_2L_pos : 0 < 2 * L := by linarith
    nlinarith
  have h2 : Real.exp (-2 * κ₂ * L) < Real.exp (-2 * κ₁ * L) :=
    Real.exp_lt_exp.mpr h1
  exact mul_lt_mul_of_pos_left h2 h_ν₀

-- =============================================================================
-- PART 3: HALF-LIFE DEFINITION AND PROPERTIES
-- =============================================================================

/-- The half-life of a radioactive nucleus.

**Definition**: T₁/₂ = ln(2)/λ where λ is the decay rate.

This follows from N(t) = N₀ · e^(-λt). Setting N(T₁/₂) = N₀/2:
  N₀/2 = N₀ · e^(-λT₁/₂)
  1/2 = e^(-λT₁/₂)
  ln(1/2) = -λT₁/₂
  -ln(2) = -λT₁/₂
  T₁/₂ = ln(2)/λ ∎
-/
def half_life (decay_rate : ℝ) (h_rate : 0 < decay_rate) : ℝ :=
  Real.log 2 / decay_rate

/-- Half-life is positive for any positive decay rate. -/
theorem half_life_pos (decay_rate : ℝ) (h_rate : 0 < decay_rate) :
    0 < half_life decay_rate h_rate := by
  unfold half_life
  apply div_pos
  · exact Real.log_pos (by norm_num : (1 : ℝ) < 2)
  · exact h_rate

/-- Slower decay rate → longer half-life (inverse relationship). -/
theorem half_life_monotone_decreasing
    (rate₁ rate₂ : ℝ) (h₁ : 0 < rate₁) (h₂ : 0 < rate₂) (h_order : rate₁ < rate₂) :
    half_life rate₂ h₂ < half_life rate₁ h₁ := by
  unfold half_life
  have h_log2_pos : 0 < Real.log 2 := Real.log_pos (by norm_num : (1 : ℝ) < 2)
  exact div_lt_div_of_pos_left h_log2_pos h₁ h_order

/-- Combining: higher barrier → lower decay rate → longer half-life. -/
theorem half_life_barrier_dependence
    (κ₁ κ₂ L ν₀ : ℝ)
    (h_L : 0 < L)
    (h_ν₀ : 0 < ν₀)
    (h_order : κ₁ < κ₂) :
    let rate₁ := tunneling_rate κ₁ L ν₀
    let rate₂ := tunneling_rate κ₂ L ν₀
    have h_rate₁ : 0 < rate₁ := tunneling_rate_pos κ₁ L ν₀ h_ν₀
    have h_rate₂ : 0 < rate₂ := tunneling_rate_pos κ₂ L ν₀ h_ν₀
    half_life rate₁ h_rate₁ < half_life rate₂ h_rate₂ := by
  intro rate₁ rate₂ h_rate₁ h_rate₂
  have h_rate_order : rate₂ < rate₁ := tunneling_rate_decreasing_in_kappa κ₁ κ₂ L ν₀ h_L h_ν₀ h_order
  exact half_life_monotone_decreasing rate₂ rate₁ h_rate₂ h_rate₁ h_rate_order

-- =============================================================================
-- PART 4: EXPONENTIAL DECAY LAW
-- =============================================================================

/-- The number of nuclei remaining after time t.

**The Exponential Decay Law**: N(t) = N₀ · e^(-λt)

This follows from dN/dt = -λN, the fundamental decay equation.
-/
def nuclei_remaining (N₀ decay_rate t : ℝ) : ℝ :=
  N₀ * Real.exp (-decay_rate * t)

/-- At t = 0, all nuclei remain. -/
theorem nuclei_at_zero (N₀ decay_rate : ℝ) :
    nuclei_remaining N₀ decay_rate 0 = N₀ := by
  unfold nuclei_remaining
  simp [Real.exp_zero]

/-- Nuclei count is always non-negative if N₀ ≥ 0. -/
theorem nuclei_remaining_nonneg (N₀ decay_rate t : ℝ)
    (h_N₀ : 0 ≤ N₀) :
    0 ≤ nuclei_remaining N₀ decay_rate t := by
  unfold nuclei_remaining
  exact mul_nonneg h_N₀ (le_of_lt (Real.exp_pos _))

/-- At half-life, exactly half the nuclei remain. -/
theorem nuclei_at_half_life (N₀ decay_rate : ℝ)
    (h_rate : 0 < decay_rate) (h_N₀ : 0 < N₀) :
    nuclei_remaining N₀ decay_rate (half_life decay_rate h_rate) = N₀ / 2 := by
  unfold nuclei_remaining half_life
  have h_exp : Real.exp (-decay_rate * (Real.log 2 / decay_rate)) = 1 / 2 := by
    have h_simp : -decay_rate * (Real.log 2 / decay_rate) = -Real.log 2 := by
      field_simp
    rw [h_simp, Real.exp_neg, Real.exp_log (by norm_num : (0 : ℝ) < 2)]
    norm_num
  rw [h_exp]
  ring

-- =============================================================================
-- PART 5: CONNECTION TO VACUUM STIFFNESS
-- =============================================================================

/-- The vacuum stiffness contribution to barrier wave vector.

In QFD, the barrier height V₀ depends on the vacuum stiffness parameter β.
Higher β → stiffer vacuum → higher barrier → slower decay.
-/
def vacuum_barrier_contribution (β : ℝ) (V₀_base : ℝ) : ℝ :=
  V₀_base * β

/-- Higher vacuum stiffness creates higher barriers. -/
theorem vacuum_stiffness_increases_barrier (β₁ β₂ V₀_base : ℝ)
    (h_V₀ : 0 < V₀_base)
    (h_order : β₁ < β₂)
    (h_β₁ : 0 < β₁) :
    vacuum_barrier_contribution β₁ V₀_base < vacuum_barrier_contribution β₂ V₀_base := by
  unfold vacuum_barrier_contribution
  exact mul_lt_mul_of_pos_left h_order h_V₀

/-- The decay constant (mean lifetime) τ = 1/λ. -/
def mean_lifetime (decay_rate : ℝ) (h_rate : 0 < decay_rate) : ℝ :=
  1 / decay_rate

/-- Relationship between half-life and mean lifetime: T₁/₂ = τ · ln(2). -/
theorem half_life_mean_lifetime_relation (decay_rate : ℝ) (h_rate : 0 < decay_rate) :
    half_life decay_rate h_rate = mean_lifetime decay_rate h_rate * Real.log 2 := by
  unfold half_life mean_lifetime
  ring

end QFD.Nuclear.DecayHalfLife
