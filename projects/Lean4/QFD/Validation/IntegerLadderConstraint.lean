/-!
# Integer Ladder Constraint

This module formalizes the observation that once a prediction function only
hits integers, half-integer residues are impossible.  It mirrors the logic
behind the integer ladder validation script.
-/

import Mathlib.Data.Int.Basic

namespace CodexProofs

/-- Helper: integers cannot equal `k + 1/2`. -/
lemma int_ne_half (n k : ℤ) : (n : ℝ) ≠ (k : ℝ) + (1 / 2 : ℝ) := by
  intro h
  have h' := congrArg (fun t : ℝ => (2 : ℝ) * t) h
  have : ((2 * n : ℤ) : ℝ) - ((2 * k : ℤ) : ℝ) = 1 := by
    simpa [two_mul, add_comm, add_left_comm, add_assoc, sub_eq_add_neg,
      bit0, one_div, mul_comm, mul_left_comm] using h'
  have hℤ : 2 * n - 2 * k = 1 := by exact_mod_cast this
  have hdiv : (2 : ℤ) ∣ 1 := by
    refine ⟨n - k, ?_⟩
    simpa [two_mul, sub_eq_add_neg, mul_add, add_comm, add_left_comm,
      add_assoc] using hℤ
  have : ¬ (2 : ℤ) ∣ 1 := by decide
  exact this hdiv

/--
If every prediction equals an integer, the function never lands on
half-integers.  This captures the “no forbidden zones” logic used in the
integer ladder test.
-/
lemma no_half_integer_predictions
    (Φ : ℝ → ℝ) (hint : ∀ A, ∃ n : ℤ, Φ A = n) :
    ∀ A, ¬ ∃ k : ℤ, Φ A = (k : ℝ) + (1 / 2 : ℝ) := by
  intro A
  rcases hint A with ⟨n, hn⟩
  intro h
  rcases h with ⟨k, hk⟩
  exact (int_ne_half n k) (by simpa [hn] using hk)

end CodexProofs
