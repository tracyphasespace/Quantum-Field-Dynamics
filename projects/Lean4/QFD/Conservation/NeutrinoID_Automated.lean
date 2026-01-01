import QFD.GA.Cl33
import QFD.GA.BasisReduction
import QFD.GA.BasisProducts
import QFD.Conservation.NeutrinoID_Simple

namespace QFD.Conservation.NeutrinoID.Automated

open QFD.GA
open QFD.Conservation.NeutrinoID
open CliffordAlgebra

/-- Disjoint bivectors commute (all four indices distinct). -/
lemma disjoint_bivectors_commute :
    (e 0 * e 1) * (e 3 * e 4) = (e 3 * e 4) * (e 0 * e 1) :=
  QFD.GA.BasisProducts.e01_commutes_e34

theorem no_EM_interaction : Interaction F_EM Nu = 0 := by
  unfold Interaction F_EM Nu
  have h := disjoint_bivectors_commute
  have h_sub :
      (e 0 * e 1) * (e 3 * e 4) - (e 3 * e 4) * (e 0 * e 1) = 0 :=
    sub_eq_zero.mpr h
  simpa [F_EM, Nu, sub_eq_add_neg] using h_sub

section Nontrivial

variable [Nontrivial Cl33]

theorem has_nonzero_spin : Nu * Nu ≠ 0 := by
  simpa using QFD.Conservation.NeutrinoID.has_nonzero_spin

theorem neutrino_is_necessary : Nu ≠ 0 := by
  simpa using QFD.Conservation.NeutrinoID.neutrino_is_necessary

end Nontrivial

end QFD.Conservation.NeutrinoID.Automated
