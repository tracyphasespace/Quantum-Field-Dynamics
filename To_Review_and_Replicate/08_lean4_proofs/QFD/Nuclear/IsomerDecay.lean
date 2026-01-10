import QFD.Nuclear.MagicNumbers

/-!
# Isomer Transitions

The discrete shell structure from `MagicNumbers` already encodes the fact that
energy can only be released in quantised steps.  We expose a small helper lemma
showing that moving from shell `n` to `n + 1` always releases a fixed, positive
amount of capacity (which can be mapped to the emitted gamma energy).
-/

namespace QFD.Nuclear.IsomerDecay

open QFD.Nuclear.MagicNumbers

/-- The next shell always adds `2 (n + 2)` available states. -/
theorem gamma_emission_quantized (n : ℕ) :
    shell_capacity (n + 1) = shell_capacity n + 2 * (n + 2) := by
  have h_succ :
      shell_capacity (n + 1) = (n + 2) * (n + 3) := by
    simp [shell_capacity, Nat.succ_eq_add_one, add_comm, add_left_comm, add_assoc]
  have h_prev :
      shell_capacity n + 2 * (n + 2) = (n + 2) * (n + 3) := by
    calc
      shell_capacity n + 2 * (n + 2)
          = (n + 1) * (n + 2) + (n + 2) * 2 := by
            simp [shell_capacity, Nat.mul_comm, Nat.mul_left_comm]
      _ = (n + 2) * (n + 1) + (n + 2) * 2 := by
            simp [Nat.mul_comm]
      _ = (n + 2) * (n + 1 + 2) := by
        have := Nat.mul_add (n + 2) (n + 1) 2
        simpa [Nat.add_comm, Nat.add_left_comm, Nat.add_assoc] using this.symm
      _ = (n + 2) * (n + 3) := by
        simp [Nat.add_comm, Nat.add_left_comm, Nat.add_assoc]
  exact h_succ.trans h_prev.symm

end QFD.Nuclear.IsomerDecay
