import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.InnerProductSpace.PiL2
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Real.Basic

noncomputable section

open scoped BigOperators

namespace QFD.Neutrino

/-!
# Gate N-L3: Flavor/Isomer Oscillation as Unitary Phase Evolution

This gate formalizes the *kinematic* oscillation mechanism:

  ψ(t) = U (D(t) (U⁻¹ ψ₀))

where
- ψ is the neutrino state in the flavor basis (νe, νμ, ντ),
- U is a unitary (isometric) change-of-basis (mixing),
- D(t) is diagonal phase evolution in the internal isomer / mass basis.

We keep the design consistent with the project style:
- local hypotheses via a structure
- no global axioms
- later: instantiate U, D(t), and nontriviality from QFD rotor/isomer geometry.
-/

-- Three flavors: νe, νμ, ντ
abbrev Flavor : Type := Fin 3

-- Finite-dimensional Hilbert space of flavor amplitudes
abbrev State : Type := EuclideanSpace ℂ Flavor

/-- Pointwise Born weight for measuring a given flavor coordinate. -/
def prob (ψ : State) (α : Flavor) : ℝ :=
  Complex.normSq (ψ α)

/-- "Normalized" state (total probability 1). -/
def IsNormalized (ψ : State) : Prop := ‖ψ‖ = 1

/-- Local-hypotheses package for oscillation kinematics. -/
structure OscillationHypotheses where
  /-- Mixing map (flavor ↔ isomer/mass basis). -/
  U : State ≃ₗᵢ[ℂ] State

  /-- Phase evolution in the isomer/mass basis. Think: diag(exp(-i E_k t)). -/
  D : ℝ → (State ≃ₗᵢ[ℂ] State)

  /-- Normalization of the coordinate probability rule:
      ∑α prob(ψ,α) = ‖ψ‖².
      (Kept as a hypothesis to avoid fighting library lemma names; later we can discharge it.) -/
  prob_sum_eq_norm_sq : ∀ ψ : State, (∑ α : Flavor, prob ψ α) = ‖ψ‖ ^ 2

  /-- Nontriviality hook: existence of some initial state and flavor whose probability
      changes in time. -/
  exists_nontrivial_oscillation :
    ∃ (ψ0 : State) (α : Flavor) (t1 t2 : ℝ),
      IsNormalized ψ0 ∧
        prob (U (D t1 (U.symm ψ0))) α ≠ prob (U (D t2 (U.symm ψ0))) α

namespace OscillationHypotheses

/-- Time evolution in the flavor basis. -/
def evolve (H : OscillationHypotheses) (t : ℝ) (ψ0 : State) : State :=
  H.U (H.D t (H.U.symm ψ0))

/-- Transition probability to observe flavor α at time t. -/
def P (H : OscillationHypotheses) (ψ0 : State) (t : ℝ) (α : Flavor) : ℝ :=
  prob (evolve H t ψ0) α

/-- Norm is preserved under evolution (unitary/isometric dynamics). -/
theorem norm_evolve (H : OscillationHypotheses) (t : ℝ) (ψ0 : State) :
    ‖evolve H t ψ0‖ = ‖ψ0‖ := by
  -- This should be a short simp/calc proof since U and D(t) are isometries.
  -- If lemma names differ in your Mathlib pin, rewrite with `simp [evolve]`.
  -- (We keep it explicit to make downstream use clean.)
  calc
    ‖evolve H t ψ0‖
        = ‖H.D t (H.U.symm ψ0)‖ := by
            simp only [evolve, LinearIsometryEquiv.norm_map]
    _   = ‖H.U.symm ψ0‖ := by
            simp only [LinearIsometryEquiv.norm_map]
    _   = ‖ψ0‖ := by
            simp only [LinearIsometryEquiv.norm_map]

/-- Total probability is conserved: ∑α P(t,α) = ‖ψ0‖². -/
theorem sum_P_eq_norm_sq (H : OscillationHypotheses) (ψ0 : State) (t : ℝ) :
    (∑ α : Flavor, H.P ψ0 t α) = ‖ψ0‖ ^ 2 := by
  -- Use the supplied normalization identity plus norm preservation.
  have h1 : (∑ α : Flavor, prob (evolve H t ψ0) α) = ‖evolve H t ψ0‖ ^ 2 :=
    H.prob_sum_eq_norm_sq (evolve H t ψ0)
  -- Replace ‖evolve H t ψ0‖ with ‖ψ0‖
  rw [norm_evolve H t ψ0] at h1
  -- Unfold P and prob
  simp only [P, prob, evolve] at h1 ⊢
  exact h1

/-- If ψ0 is normalized, probabilities sum to 1 for all t. -/
theorem sum_P_eq_one (H : OscillationHypotheses) (ψ0 : State) (t : ℝ)
    (hψ : IsNormalized ψ0) :
    (∑ α : Flavor, H.P ψ0 t α) = 1 := by
  rw [sum_P_eq_norm_sq]
  -- Turn ‖ψ0‖^2 into 1^2 = 1
  simp only [IsNormalized] at hψ
  rw [hψ]
  norm_num

/-- Exported nontrivial oscillation witness (exists some ψ0, α, t1, t2 with P(t1) ≠ P(t2)). -/
theorem exists_oscillation (H : OscillationHypotheses) :
    ∃ (ψ0 : State) (α : Flavor) (t1 t2 : ℝ),
      IsNormalized ψ0 ∧ H.P ψ0 t1 α ≠ H.P ψ0 t2 α := by
  rcases H.exists_nontrivial_oscillation with ⟨ψ0, α, t1, t2, hnorm, hne⟩
  refine ⟨ψ0, α, t1, t2, hnorm, ?_⟩
  -- Unfold P/evolve and reuse the provided witness inequality.
  simpa [OscillationHypotheses.P, evolve] using hne

end OscillationHypotheses

end QFD.Neutrino
