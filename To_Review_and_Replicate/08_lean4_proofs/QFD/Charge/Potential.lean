import Mathlib.Analysis.Calculus.Deriv.Basic
import Mathlib.Topology.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Tactic

noncomputable section

namespace QFD.Charge

open scoped Topology
open Filter

/-!
# Gate C-L2: Harmonic Decay (Deriving 1/r from 3D Geometry)

Goal: Prove that if f(r) = k/r, then the spherical Laplacian Δf = 0.
-/

variable (f : ℝ → ℝ)

/-- The Radial Laplacian Operator in 3 Dimensions (Spherical Symmetry) -/
def spherical_laplacian_3d (f : ℝ → ℝ) (r : ℝ) : ℝ :=
  (deriv (deriv f)) r + (2 / r) * (deriv f) r

/-- First derivative of `k/r` for `r ≠ 0`. -/
lemma deriv_one_over_r (k : ℝ) {r : ℝ} (hr : r ≠ 0) :
    deriv (fun x => k / x) r = -k / r ^ 2 := by
  have h_inv : HasDerivAt (fun x : ℝ => x⁻¹) (-1 / r^2) r := by
    simpa using ( (hasDerivAt_id r).inv hr )
  have h : HasDerivAt (fun x : ℝ => k / x) (-k / r^2) r := by
    -- scale the inverse derivative by k
    have h_mul : HasDerivAt (fun x : ℝ => k * x⁻¹) (k * (-1 / r^2)) r := by
      simpa using h_inv.const_mul k
    -- rewrite k/x as k * x⁻¹
    simpa [div_eq_mul_inv, mul_assoc, mul_comm, mul_left_comm] using h_mul
  simpa using h.deriv

/-- Second derivative of `k/r` for `r ≠ 0`. -/
lemma deriv_deriv_one_over_r (k : ℝ) {r : ℝ} (hr : r ≠ 0) :
    deriv (deriv (fun x => k / x)) r = 2 * k / r ^ 3 := by
  -- Locally around r (where x ≠ 0), (k/x)' = -k/x^2
  have h_eventual :
      deriv (fun x => k / x) =ᶠ[nhds r] (fun x => -k / x ^ 2) := by
    have h_open : IsOpen {x : ℝ | x ≠ 0} := isOpen_ne
    have h_mem : r ∈ {x : ℝ | x ≠ 0} := hr
    filter_upwards [h_open.mem_nhds h_mem] with x hx
    exact deriv_one_over_r (k := k) (r := x) hx

  -- Swap inside the outer deriv using local equality
  rw [h_eventual.deriv_eq]

  -- Now compute deriv of (-k / x^2) at r without `deriv_fun_inv''`
  have h_id : HasDerivAt (fun x : ℝ => x) 1 r := by
    simpa using (hasDerivAt_id r)

  -- h_sq: derivative of x*x is r+r at r
  have h_sq : HasDerivAt (fun x : ℝ => x * x) (r + r) r := by
    -- (id * id)' = 1*r + r*1 = r + r
    simpa [mul_assoc, mul_comm, mul_left_comm, add_assoc, add_comm, add_left_comm] using
      (h_id.mul h_id)

  have h_sq_ne : (r * r) ≠ 0 := mul_ne_zero hr hr

  have h_inv_sq : HasDerivAt (fun x : ℝ => (x * x)⁻¹) (-(r + r) / (r * r) ^ 2) r :=
    h_sq.inv h_sq_ne

  have h_main :
      HasDerivAt (fun x : ℝ => -k / x ^ 2) (2 * k / r ^ 3) r := by
    -- rewrite -k / x^2 = (-k) * (x*x)⁻¹
    have h_mul : HasDerivAt (fun x : ℝ => (-k) * (x * x)⁻¹)
        ((-k) * (-(r + r) / (r * r) ^ 2)) r :=
      h_inv_sq.const_mul (-k)
    -- normalize algebra: (-k) * (-(r+r)/(r*r)^2) = 2*k/r^3 and x^2 = x*x
    convert h_mul using 1
    · ext x; rw [pow_two]; ring
    · rw [pow_two, pow_three]; field_simp [hr]; ring

  simpa using h_main.deriv

/-- **Theorem C-L2**: Harmonic Decay 3D. -/
theorem harmonic_decay_3d (k : ℝ) (r : ℝ) (hr : r ≠ 0) :
    let potential := fun x => k / x
    spherical_laplacian_3d potential r = 0 := by
  intro potential
  unfold spherical_laplacian_3d
  rw [deriv_one_over_r (k := k) (r := r) hr]
  rw [deriv_deriv_one_over_r (k := k) (r := r) hr]
  field_simp [hr]
  ring

end QFD.Charge
