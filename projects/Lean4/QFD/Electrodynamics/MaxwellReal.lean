import QFD.GA.Cl33
import QFD.QM_Translation.DiracRealization

/-
# Maxwell's Real Equation

Priority-2 scaffolding that states the GA form of Maxwell's equations.
-/
namespace QFD.Electrodynamics.MaxwellReal

open QFD.GA
open CliffordAlgebra
open QFD.QM_Translation.DiracRealization

/-- Field is a bivector-valued field. -/
def EM_Field_Vector := Fin 6 → Cl33

/-- Current is a vector-valued field. -/
def Current_Vector := Fin 6 → Cl33

/--
Algebraic gradient (momentum-space proxy). Simulates `D = γ^μ k_μ`.
-/
def GeometricGradient (k : Fin 4 → ℝ) : Cl33 :=
  ∑ i : Fin 4, algebraMap ℝ Cl33 (k i) * gamma i

/--
The geometric Maxwell decomposition: grad·F (sources) + grad∧F (Bianchi).
-/
theorem maxwell_decomposition (k : Fin 4 → ℝ) (F : Cl33) :
    let grad := GeometricGradient k
    grad * F =
      (1 / 2 : ℝ) • (grad * F + F * grad) +
      (1 / 2 : ℝ) • (grad * F - F * grad) := by
  intro grad
  -- Show: grad*F = (1/2)•(grad*F + F*grad) + (1/2)•(grad*F - F*grad)
  -- by showing RHS simplifies back to grad*F
  symm
  calc (1 / 2 : ℝ) • (grad * F + F * grad) + (1 / 2 : ℝ) • (grad * F - F * grad)
      = (1 / 2 : ℝ) • (grad * F) + (1 / 2 : ℝ) • (F * grad) +
        ((1 / 2 : ℝ) • (grad * F) - (1 / 2 : ℝ) • (F * grad)) := by
          rw [smul_add, smul_sub]
    _ = ((1 / 2 : ℝ) + (1 / 2 : ℝ)) • (grad * F) +
        ((1 / 2 : ℝ) - (1 / 2 : ℝ)) • (F * grad) := by
          rw [add_smul, sub_smul]; abel
    _ = (1 : ℝ) • (grad * F) + (0 : ℝ) • (F * grad) := by norm_num
    _ = grad * F := by simp

end QFD.Electrodynamics.MaxwellReal
