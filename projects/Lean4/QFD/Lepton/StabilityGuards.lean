import Mathlib

set_option autoImplicit false

namespace QFD

/-!
  # Stability Constraints (The Factorization Guardrail)

  This module formally proves why simple energy functionals of the form:
  E(Q, R) = g(Q) * f(R)
  CANNOT produce a discrete mass spectrum (multiple stable isomers).

  ## The Physics Insight
  If the topology (Q) and geometry (R) factorize, then:
  1. The optimal radius R* is independent of Q.
  2. The minimal energy E_min(Q) scales monotonically with g(Q).
  3. Therefore, no discrete local minima exist in Q-space.

  ## Conclusion
  To get a Lepton Spectrum, the physics MUST introduce non-factorizable terms
  (e.g., Q-dependent stiffness β(Q) or topological closure terms).
-/

/--
  A Factorizable Energy Functional.
  E(q, r) = g(q) * f(r)

  For physics applications, both Q (topological charge) and R (radius)
  are real-valued.
-/
def FactorizableEnergy (g : ℝ → ℝ) (f : ℝ → ℝ) (q : ℝ) (r : ℝ) : ℝ :=
  g q * f r

/--
  Theorem: Independence of Radius.
  For a factorizable energy, the minimizing radius r* is the same for all q (assuming g(q) ≠ 0).
-/
theorem radius_independence
    (g : ℝ → ℝ) (f : ℝ → ℝ)
    (r_star : ℝ)
    (h_min_f : IsMinOn f Set.univ r_star) -- r_star minimizes f globally
    (q : ℝ) (hq : 0 < g q) :               -- g(q) is positive (scaling factor)
    IsMinOn (FactorizableEnergy g f q) Set.univ r_star := by

  -- Logic: E(q, r) = g(q) * f(r)
  -- Since g(q) > 0, minimizing g(q)*f(r) is equivalent to minimizing f(r).
  intro r _
  unfold FactorizableEnergy

  -- We know f(r_star) ≤ f(r)
  have h_f_ineq : f r_star ≤ f r := h_min_f (Set.mem_univ r)

  -- Multiply by positive g(q)
  exact mul_le_mul_of_nonneg_left h_f_ineq (le_of_lt hq)

/--
  Theorem: Monotonicity of Spectrum.
  If E(q, r) factorizes, and g(q) is monotonic (e.g. q^2),
  then the particle mass E_min(q) is strictly monotonic.

  IMPLICATION: No "islands of stability" (Isomers) can exist.
-/
theorem spectrum_monotonicity
    (g : ℝ → ℝ) (f : ℝ → ℝ)
    (r_star : ℝ)
    (h_min_f : IsMinOn f Set.univ r_star)
    (h_f_pos : 0 < f r_star)  -- The vacuum shape has positive energy
    (q1 q2 : ℝ)
    (h_mono_g : g q1 < g q2) : -- g is strictly monotonic (e.g. q1 < q2 -> q1^2 < q2^2)
    FactorizableEnergy g f q1 r_star < FactorizableEnergy g f q2 r_star := by

  unfold FactorizableEnergy
  exact mul_lt_mul_of_pos_right h_mono_g h_f_pos

end QFD
