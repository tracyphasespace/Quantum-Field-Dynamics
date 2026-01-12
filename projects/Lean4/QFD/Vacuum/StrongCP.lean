import Mathlib.Topology.Algebra.InfiniteSum

/-
# Strong CP Problem Solution

**Priority**: 139 (Cluster 5)  
**Goal**: Show that a simple vacuum-relaxation map exponentially damps
`θ_QCD`, driving it to zero.

We model the relaxation step as a contraction
`θ ↦ (1 - κ) · θ` with `0 < κ < 2`.  The condition ensures
`|1 - κ| < 1`, so repeated applications of the map form a geometric
sequence whose limit is zero.  This captures the qualitative statement
that the QCD vacuum relaxes any dipole moment back to zero.
-/

namespace QFD.Vacuum.StrongCP

open Filter Function
open scoped Topology

/-- A discrete relaxation step that damps the CP-violating angle. -/
def relaxationStep (κ θ : ℝ) : ℝ :=
  (1 - κ) * θ

/--
**Theorem: Dynamic Minimization**

If the lattice relaxation removes a fixed positive fraction `κ` of the
dipole each step (with `κ ∈ (0, 2)` so that the map is a contraction),
then the iterated sequence tends to zero.
-/
theorem vacuum_relaxes_to_zero_dipole
    (θ₀ κ : ℝ) (hκ_pos : 0 < κ) (hκ_lt_two : κ < 2) :
    Tendsto (fun n : ℕ => (relaxationStep κ)^[n] θ₀) atTop (𝓝 0) := by
  -- Iterating `relaxationStep` produces the geometric sequence
  -- `((1 - κ) ^ n) * θ₀`.  We rewrite it explicitly.
  have h_iter :
      (fun n : ℕ => (relaxationStep κ)^[n] θ₀)
        = fun n : ℕ => ((1 - κ) ^ n) * θ₀ := by
    funext n
    induction' n with n hn
    · simp [relaxationStep]
    · simp [relaxationStep, hn, pow_succ, mul_comm, mul_left_comm, mul_assoc]
  -- The contraction factor satisfies `|1 - κ| < 1`.
  have h_upper : 1 - κ < 1 := sub_lt_self _ hκ_pos
  have h_diff : κ - 1 < 1 := by
    simpa using sub_lt_sub_right hκ_lt_two 1
  have h_lower : -1 < 1 - κ := by
    have : -(1 : ℝ) < -(κ - 1) :=
      (neg_lt_neg_iff).2 h_diff
    simpa [sub_eq_add_neg, add_comm, add_left_comm] using this
  have h_abs : |1 - κ| < 1 :=
    abs_lt.mpr ⟨h_lower, h_upper⟩
  -- The geometric sequence converges to zero.
  have h_pow :
      Tendsto (fun n : ℕ => (1 - κ) ^ n) atTop (𝓝 0) :=
    tendsto_pow_atTop_nhds_0_of_abs_lt_1 _ h_abs
  -- Multiply by the constant `θ₀`.
  have h_const :
      Tendsto (fun _ : ℕ => θ₀) atTop (𝓝 θ₀) :=
    tendsto_const_nhds
  have h_mul :
      Tendsto (fun n : ℕ => θ₀ * (1 - κ) ^ n) atTop (𝓝 (θ₀ * 0)) :=
    h_const.mul h_pow
  have h_limit :
      Tendsto (fun n : ℕ => θ₀ * (1 - κ) ^ n) atTop (𝓝 0) := by
    simpa using h_mul
  -- Rewrite using `h_iter`.
  have :
      Tendsto (fun n : ℕ => ((1 - κ) ^ n) * θ₀) atTop (𝓝 0) := by
    -- `θ₀ * (1 - κ)^n` equals `((1 - κ)^n) * θ₀`.
    simpa [mul_comm] using h_limit
  simpa [h_iter] using this

end QFD.Vacuum.StrongCP
