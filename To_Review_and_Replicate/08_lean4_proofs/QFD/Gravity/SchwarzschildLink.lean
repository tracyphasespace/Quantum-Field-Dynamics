import QFD.Gravity.TimeRefraction
import QFD.Gravity.GeodesicForce
import Mathlib.Data.Real.Basic
import Mathlib.Tactic

noncomputable section

namespace QFD.Gravity

/-!
# Gate G-L3: Schwarzschild Link (No Filters, No Series)

We connect the QFD metric ansatz

  g00_QFD(r) = 1 / (1 + κ ρ(r))

to the weak-field Schwarzschild form

  g00_Schw(r) = 1 - 2GM/(r c²)

without Taylor series or Filters, using an **exact algebraic remainder identity**:

  (1 + x)⁻¹ = 1 - x + x² * (1 + x)⁻¹,   provided (1 + x) ≠ 0.

When x = 2GM/(r c²), the first-order term matches Schwarzschild exactly,
and the remainder is explicit and controlled.
-/

/-- Schwarzschild weak-field `g00` in standard coordinates. -/
def schwarzschild_g00 (G M c r : ℝ) : ℝ :=
  1 - (2 * G * M) / (r * c ^ 2)

/-- QFD weak-field coupling choice to match GR first order: κ := 2G / c². -/
def kappa_GR (G c : ℝ) : ℝ := (2 * G) / (c ^ 2)

/-- Build a GravityContext consistent with the GR matching choice. -/
def ctxGR (G c : ℝ) (hc : 0 < c) : GravityContext :=
  { c := c, hc := hc, kappa := kappa_GR G c }

/-- QFD g00 for a point mass using ρ(r) = M/r and κ = 2G/c². -/
def qfd_g00_point (G M c : ℝ) (hc : 0 < c) (r : ℝ) : ℝ :=
  g00 (ctxGR G c hc) (rhoPointMass M) r

/--
Exact identity: `(1 + x)⁻¹ = 1 - x + x² * (1 + x)⁻¹`, assuming `1 + x ≠ 0`.
-/
lemma inv_one_add_decomp (x : ℝ) (hx : 1 + x ≠ 0) :
    (1 + x)⁻¹ = 1 - x + x ^ 2 * (1 + x)⁻¹ := by
  -- Clear denominators safely.
  field_simp [hx]
  ring

/--
Rosetta stone: QFD g00 is exactly an inverse-one-plus-x form where
`x = 2GM/(r c²)` (for ρ = M/r, κ = 2G/c²).
-/
theorem qfd_g00_point_eq_inv
    (G M c : ℝ) (hc : 0 < c) (r : ℝ) (hr : r ≠ 0) :
    qfd_g00_point G M c hc r = (1 + (2 * G * M) / (r * c ^ 2))⁻¹ := by
  -- Expand definitions.
  unfold qfd_g00_point ctxGR kappa_GR g00 n2 rhoPointMass
  simp [hr]
  ring

/--
Weak-field matching statement with an explicit remainder:

Let x = 2GM/(r c²). Then
  g00_QFD(r) = 1 - x + x² * (1 + x)⁻¹
and
  g00_Schw(r) = 1 - x

So the difference is exactly:
  g00_QFD(r) - g00_Schw(r) = x² * (1 + x)⁻¹.
-/
theorem qfd_matches_schwarzschild_first_order
    (G M c : ℝ) (hc : 0 < c) (r : ℝ)
    (hr : r ≠ 0)
    (hx : 1 + (2 * G * M) / (r * c ^ 2) ≠ 0) :
    qfd_g00_point G M c hc r
      = schwarzschild_g00 G M c r
        + ((2 * G * M) / (r * c ^ 2)) ^ 2
          * (1 + (2 * G * M) / (r * c ^ 2))⁻¹ := by
  -- Set x and use the algebraic decomposition.
  set x : ℝ := (2 * G * M) / (r * c ^ 2)
  have hq : qfd_g00_point G M c hc r = (1 + x)⁻¹ := by
    -- use the inv form, then rewrite into x
    have := qfd_g00_point_eq_inv (G := G) (M := M) (c := c) (hc := hc) (r := r) hr
    -- rewrite the concrete fraction as x
    simpa [x] using this

  -- Schwarzschild g00 is 1 - x by definition
  have hs : schwarzschild_g00 G M c r = 1 - x := by
    simp [schwarzschild_g00, x]

  -- Combine:
  -- (1 + x)⁻¹ = (1 - x) + x²*(1 + x)⁻¹
  -- so (1 + x)⁻¹ = schwarzschild + remainder
  rw [hq, hs]
  have hx' : 1 + x ≠ 0 := by simpa [x] using hx
  calc
    (1 + x)⁻¹
        = (1 - x + x ^ 2 * (1 + x)⁻¹) := by
            simpa using (inv_one_add_decomp x hx')
    _ = (1 - x) + x ^ 2 * (1 + x)⁻¹ := by ring

end QFD.Gravity
