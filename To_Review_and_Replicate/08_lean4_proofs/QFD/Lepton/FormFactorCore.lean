import Mathlib.Data.Real.Basic

noncomputable section

namespace QFD

/--
A soliton geometry summarizes the spatial distribution of the field through two
scalar observables: bulk and gradient integrals.
-/
structure SolitonGeometry where
  bulk_integral : ℝ
  grad_integral : ℝ
  h_positive : bulk_integral > 0 ∧ grad_integral > 0

/-- Dimensionless shape factor `F = bulk/grad`. -/
def shape_factor (g : SolitonGeometry) : ℝ :=
  g.bulk_integral / g.grad_integral

end QFD
