import Mathlib

set_option autoImplicit false

namespace QFD

/-!
# AlphaFormFactor

This module formalizes the "α gap" as a geometric form-factor problem.

We keep the physics honest by separating:

- α_meas : the measured fine-structure constant (or any target value)
- α_cont : the continuum / Euclidean-flux prediction
- κ_spin : a positive correction factor encoding spin-bundle (double-cover) effects

We then define the corrected prediction:

- α_corr = α_cont / κ_spin

The key theorems show:

1. The residual between α_cont and α_meas is exactly representable as κ_spin.
2. If κ_spin is computed correctly (from spinor-bundle geometry), the gap closes.

Nothing here asserts what κ_spin numerically is; that is a separate geometric
computation (typically performed numerically, then later replaced by an analytic
derivation).
-/

universe u
variable {Point : Type u}

/-- Relative error: |pred / target - 1|. -/
noncomputable def relErr (target pred : ℝ) : ℝ := abs (pred / target - 1)

/--
Alpha model with an explicit spinor/topology correction factor.

We attach this to `EmergentConstants` because in your book the α formula is
built on top of the (c, ℏ) layer.
-/
structure AlphaFormFactor (Point : Type u) extends QFD.EmergentConstants Point where
  /-- Measured (or reference) fine-structure constant. -/
  α_meas : ℝ
  /-- Continuum (classical-flux) prediction for α. -/
  α_cont : ℝ
  h_cont_ne_zero : α_cont ≠ 0
  /-- Spinor / topology correction factor (geometric form factor). -/
  κ_spin : ℝ

  h_meas_ne_zero : α_meas ≠ 0
  h_k_pos : κ_spin > 0

namespace AlphaFormFactor

variable {A : AlphaFormFactor Point}

/-- Corrected prediction for α after applying the spinor form factor. -/
noncomputable def α_corr (A : AlphaFormFactor Point) : ℝ := A.α_cont / A.κ_spin

/--
Algebraic identity: the corrected relative error depends only on the ratio
r = α_cont / α_meas and the factor κ_spin.
-/
theorem relErr_corr_eq (A : AlphaFormFactor Point) :
    relErr A.α_meas (A.α_corr) =
      abs ((A.α_cont / A.α_meas) / A.κ_spin - 1) := by
  unfold relErr AlphaFormFactor.α_corr
  have h : (A.α_cont / A.κ_spin) / A.α_meas - 1 = (A.α_cont / A.α_meas) / A.κ_spin - 1 := by
    ring
  -- abs preserves equality of the inner term
  rw [h]


/--
If the geometric factor equals the continuum-to-measured ratio, the gap closes
exactly: α_corr = α_meas.

This is the formal statement of "the α gap is a form factor".
-/
theorem gap_closes_if_kappa_matches_ratio
    (hκ : A.κ_spin = A.α_cont / A.α_meas) :
    A.α_corr = A.α_meas := by
  unfold AlphaFormFactor.α_corr
  rw [hκ]
  field_simp [A.h_meas_ne_zero, A.h_cont_ne_zero]

/--
Relative-error bound: if κ_spin approximates the required ratio, the corrected
α has proportionally small relative error.

Let r = α_cont / α_meas. Then
  α_corr / α_meas - 1 = (r - κ_spin) / κ_spin.

So if |r - κ_spin| ≤ ε · |κ_spin|, then relErr ≤ ε.
-/
theorem relErr_corr_le_of_kappa_close
    (ε : ℝ)
    (hε : 0 ≤ ε)
    (h_close : abs (A.α_cont / A.α_meas - A.κ_spin) ≤ ε * abs A.κ_spin) :
    relErr A.α_meas (A.α_corr) ≤ ε := by
  have hκ_ne_zero : A.κ_spin ≠ 0 := ne_of_gt A.h_k_pos
  have hκ_abs_pos : 0 < abs A.κ_spin := abs_pos.mpr hκ_ne_zero

  -- Rewrite relErr into the (r - κ)/κ form
  unfold relErr AlphaFormFactor.α_corr
  have : A.α_cont / A.κ_spin / A.α_meas - 1 =
      (A.α_cont / A.α_meas - A.κ_spin) / A.κ_spin := by
    -- Use 1 = κ/κ and common denominator
    calc
      A.α_cont / A.κ_spin / A.α_meas - 1
          = (A.α_cont / A.α_meas) / A.κ_spin - 1 := by ring
      _ = (A.α_cont / A.α_meas) / A.κ_spin - A.κ_spin / A.κ_spin := by
            simp [hκ_ne_zero]
      _ = (A.α_cont / A.α_meas - A.κ_spin) / A.κ_spin := by ring

  -- Apply absolute value and divide bound
  -- |(r-κ)/κ| = |r-κ| / |κ|
  have habs :
      abs (A.α_cont / A.κ_spin / A.α_meas - 1)
        = abs (A.α_cont / A.α_meas - A.κ_spin) / abs A.κ_spin := by
    simp [this, abs_div]

  -- Now use h_close and divide both sides by |κ| > 0
  have h_div :
      abs (A.α_cont / A.α_meas - A.κ_spin) / abs A.κ_spin
        ≤ (ε * abs A.κ_spin) / abs A.κ_spin := by
    exact (div_le_div_right hκ_abs_pos).mpr h_close

  -- Simplify RHS to ε
  have h_rhs : (ε * abs A.κ_spin) / abs A.κ_spin = ε := by
    field_simp [hκ_ne_zero]

  -- Finish
  simpa [habs, h_rhs]
    using h_div

end AlphaFormFactor

end QFD
