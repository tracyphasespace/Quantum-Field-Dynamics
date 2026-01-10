/-
  Proof: Fission Asymmetry from Parity
  Theorem: parity_conservation_fission
  
  Description:
  Proves that splitting an Odd-N parent (like U-236*, N=7) into two 
  symmetric integers is mathematically impossible, forcing the observed
  asymmetry in nuclear fission.
-/

import Mathlib.Data.Int.Parity
import Mathlib.Data.Real.Basic
import Mathlib.Tactic.Ring

namespace QFD_Proofs

/-- 
  Fission Rule: The harmonic mode of the parent must equal the sum 
  of the modes of the fragments (Topological Conservation).
-/
def fission_split (Np n1 n2 : ℤ) : Prop :=
  Np = n1 + n2

/--
  Theorem: If the parent is Odd, fragments cannot be equal.
-/
theorem parity_conservation_fission (Np n1 n2 : ℤ) :
  Odd Np ∧ fission_split Np n1 n2 → n1 ≠ n2 := by
  intro h
  rcases h with ⟨h_odd, h_split⟩
  intro h_eq
  -- Assume fragments are equal: n1 = n2
  rw [h_eq] at h_split
  -- Then Np = n2 + n2 = 2 * n2
  have h_even : Even Np := by
    use n2
    rw [h_split]
    ring
  -- Contradiction: Np cannot be both Odd and Even
  exact (Int.odd_iff_not_even.mp h_odd) h_even

end QFD_Proofs