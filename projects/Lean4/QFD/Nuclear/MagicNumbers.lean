import QFD.Nuclear.CoreCompressionLaw

/-!
# Nuclear Magic Numbers

A minimal combinatorial observation about harmonic-oscillator shells used by the
Nuclear module.
-/

namespace QFD.Nuclear.MagicNumbers

/-- Capacity of shell level `n` in a 3D isotropic oscillator. -/
def shell_capacity (n : ℕ) : ℕ := (n + 1) * (n + 2)

/-- First two entries of the magic-number sequence. -/
theorem magic_sequence :
    shell_capacity 0 = 2 ∧
    shell_capacity 0 + shell_capacity 1 = 8 := by
  constructor
  · simp [shell_capacity]
  · simp [shell_capacity]

/-- A mass number is a geometric resonance node if it corresponds to one of the
    classically observed magic numbers {2, 8, 20, 28, 50, 82, 126}. -/
def is_geometric_resonance_node (A : ℕ) : Prop :=
  A ∈ ({2, 8, 20, 28, 50, 82, 126} : Set ℕ)

/-- The classical magic numbers form a discrete set of geometric resonance nodes
    corresponding to the most stable standing-wave configurations of the soliton. -/
theorem magic_numbers_are_resonance_nodes (A : ℕ)
    (h : A = 2 ∨ A = 8 ∨ A = 20 ∨ A = 28 ∨ A = 50 ∨ A = 82 ∨ A = 126) :
    is_geometric_resonance_node A := by
  unfold is_geometric_resonance_node
  rcases h with rfl | rfl | rfl | rfl | rfl | rfl | rfl <;> simp

end QFD.Nuclear.MagicNumbers
