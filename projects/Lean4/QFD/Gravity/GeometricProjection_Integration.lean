/-
  Proof: Geometric Projection Integration (Starch)
  Lemma: sphere_projection_factor
  
  Description:
  Formalizes the integration of a 6D sphere's volume projected onto 
  a 4D subspace, defining k_geom.
-/

import Mathlib.Data.Real.Basic
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.FieldSimp

namespace QFD_Proofs.Starch

open Real

/-- Volume of unit n-sphere: V_n = π^(n/2) / Γ(n/2 + 1) -/
noncomputable def vol_sphere_unit (n : ℕ) : ℝ :=
  (Real.pi ^ (n / 2)) / (Real.Gamma (n / 2 + 1))

/--
  The geometric factor k_geom is the ratio of 6D volume to 4D surface flux.
  QFD Hypothesis: The coupling projects from bulk (6D) to surface (4D).
-/
noncomputable def k_geom_integral : ℝ :=
  vol_sphere_unit 6 / vol_sphere_unit 4

/--
  Value Calculation:
  V6 = pi^3 / Γ(4) = pi^3 / 6  (since Γ(4) = 3! = 6)
  V4 = pi^2 / Γ(3) = pi^2 / 2  (since Γ(3) = 2! = 2)
  Ratio = (pi^3/6) / (pi^2/2) = pi / 3 ≈ 1.047

  Note: The empirical k_geom ≈ 4.3813 includes the Topological Tax factor
  (see ProtonBridge_Geometry.lean). This theorem proves the pure geometric ratio.
-/
theorem k_geom_value : abs (k_geom_integral - (Real.pi/3)) < 0.001 := by
  unfold k_geom_integral vol_sphere_unit
  -- vol_sphere_unit 6 = π^3 / Γ(4) = π^3 / 6
  -- vol_sphere_unit 4 = π^2 / Γ(3) = π^2 / 2
  -- ratio = (π^3/6) / (π^2/2) = π/3 exactly

  -- Convert natural division to real: (6:ℕ)/2 becomes (6:ℝ)/2 in the expression
  have h6div2 : ((6 : ℕ) / 2 : ℝ) = 3 := by norm_num
  have h4div2 : ((4 : ℕ) / 2 : ℝ) = 2 := by norm_num

  -- The key insight: ↑(6:ℕ)/2 = 3 and ↑(4:ℕ)/2 = 2 as reals
  simp only [Nat.cast_ofNat]
  norm_num only
  -- After norm_num, goal becomes: |π^3 / Γ(4) / (π^2 / Γ(3)) - π/3| < 0.001

  -- Use Gamma values
  have h_g4 : Real.Gamma 4 = 6 := by
    have := Real.Gamma_nat_eq_factorial 3
    simp only [Nat.factorial, Nat.succ_eq_add_one] at this
    convert this using 1 <;> norm_num
  have h_g3 : Real.Gamma 3 = 2 := by
    have := Real.Gamma_nat_eq_factorial 2
    simp only [Nat.factorial, Nat.succ_eq_add_one] at this
    convert this using 1 <;> norm_num

  rw [h_g4, h_g3]
  -- Now goal is |π^3/6 / (π^2/2) - π/3| < 0.001

  have hpi_pos : (0:ℝ) < Real.pi := Real.pi_pos
  have hpi_ne : Real.pi ≠ 0 := ne_of_gt hpi_pos

  -- Simplify the division: (π^3/6) / (π^2/2) = π^3 * 2 / (6 * π^2) = π/3
  have h_simp : Real.pi^3 / 6 / (Real.pi^2 / 2) = Real.pi / 3 := by
    field_simp
    ring

  rw [h_simp]
  simp only [sub_self, abs_zero]
  norm_num

end QFD_Proofs.Starch