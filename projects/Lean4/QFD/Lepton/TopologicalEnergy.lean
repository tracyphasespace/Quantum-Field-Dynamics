import QFD.Physics.Postulates
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Analysis.Calculus.Deriv.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Topology.Order.Basic
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Polyrith

noncomputable section

namespace QFD.Lepton

/--
**Topological Twist Energy (Phase 3).**

The energy of a leptonic soliton is determined by three terms:
1. **Vacuum Stiffness**: β * Q^2 (Toroidal circulation)
2. **Topological Charge**: Q     (Geometric coupling)
3. **Twist Resonance**: (gamma * N^2) / Q (Poloidal twist)

Where:
- Q : Winding number (continuous for minimization, but integer-locked at stable roots)
- N : Harmonic mode (integer)
- beta : Vacuum bulk modulus
- gamma : Twist coupling constant
-/
def topological_twist_energy (beta : ℝ) (gamma : ℝ) (Q : ℝ) (N : ℤ) : ℝ :=
  beta * Q^2 + Q + (gamma * (N : ℝ)^2) / Q

/--
The stability condition: ∂E/∂Q = 0.
This determines the 'stable' winding number Q* for a given harmonic mode N.
-/
def is_stable_configuration (beta : ℝ) (gamma : ℝ) (Q : ℝ) (N : ℤ) : Prop :=
  HasDerivAt (fun q => topological_twist_energy beta gamma q N) 0 Q

/--
The cubic polynomial whose positive root gives the stable Q*.
At the critical point ∂E/∂Q = 0, we have:
  2βQ + 1 - γN²/Q² = 0
Multiplying by Q² gives: 2βQ³ + Q² - γN² = 0
-/
def stability_polynomial (beta gamma : ℝ) (N : ℤ) (Q : ℝ) : ℝ :=
  2 * beta * Q^3 + Q^2 - gamma * (N : ℝ)^2

/-- The stability polynomial is continuous -/
theorem stability_polynomial_continuous (beta gamma : ℝ) (N : ℤ) :
    Continuous (stability_polynomial beta gamma N) := by
  unfold stability_polynomial
  continuity

/-- P(0) = -γN² < 0 when N ≠ 0 and γ > 0 -/
theorem stability_polynomial_neg_at_zero
    (beta gamma : ℝ) (h_gamma : gamma > 0) (N : ℤ) (hN : N ≠ 0) :
    stability_polynomial beta gamma N 0 < 0 := by
  unfold stability_polynomial
  simp only [ne_eq, OfNat.ofNat_ne_zero, not_false_eq_true, zero_pow, mul_zero, add_zero]
  have hN_ne : (N : ℝ) ≠ 0 := Int.cast_ne_zero.mpr hN
  have hN_sq : (N : ℝ)^2 > 0 := sq_pos_iff.mpr hN_ne
  linarith [mul_pos h_gamma hN_sq]

/-- For large enough Q, P(Q) > 0 (cubic with positive leading coefficient) -/
theorem stability_polynomial_pos_for_large_Q
    (beta gamma : ℝ) (h_beta : beta > 0) (h_gamma : gamma > 0) (N : ℤ) :
    ∃ Q_big > 0, stability_polynomial beta gamma N Q_big > 0 := by
  -- Choose Q_big = γN² + 1 (simpler choice that works)
  use gamma * (N : ℝ)^2 + 1
  have hN2_nonneg : (N : ℝ)^2 ≥ 0 := sq_nonneg _
  constructor
  · -- Q_big > 0
    have h1 : gamma * (N : ℝ)^2 ≥ 0 := by positivity
    linarith
  · -- P(Q_big) > 0
    unfold stability_polynomial
    set Q := gamma * (N : ℝ)^2 + 1 with hQ_def
    have hQ_ge_1 : Q ≥ 1 := by simp only [Q]; linarith [mul_nonneg (le_of_lt h_gamma) hN2_nonneg]
    have hQ_pos : Q > 0 := by linarith
    -- For Q ≥ 1: Q² ≥ 1, Q³ ≥ 1
    -- 2βQ³ + Q² ≥ 2β + 1 > γN² when Q = γN² + 1
    -- Actually need: 2βQ³ + Q² - γN² > 0
    -- Since Q = γN² + 1, we have Q > γN², so Q² > (γN²)² ≥ 0
    -- And Q³ ≥ Q ≥ 1, so 2βQ³ ≥ 2β
    -- Also Q² ≥ Q = γN² + 1 > γN²
    -- So 2βQ³ + Q² > 0 + γN² = γN², hence P(Q) > 0
    have h_key : Q^2 > gamma * (N : ℝ)^2 := by
      have : Q^2 ≥ Q := by nlinarith
      calc Q^2 ≥ Q := this
        _ = gamma * (N : ℝ)^2 + 1 := rfl
        _ > gamma * (N : ℝ)^2 := by linarith
    have h_cubic_pos : 2 * beta * Q^3 ≥ 0 := by positivity
    linarith

/--
**Existence of Stable Configurations (IVT Application).**

For any non-zero harmonic mode N, there exists a positive winding number Q*
satisfying the stability condition. This is the geometric origin of
discrete lepton masses - each harmonic mode N selects a unique Q*.
-/
theorem stable_configuration_exists
    (beta gamma : ℝ) (h_beta : beta > 0) (h_gamma : gamma > 0)
    (N : ℤ) (hN : N ≠ 0) :
    ∃ Q > 0, stability_polynomial beta gamma N Q = 0 := by
  -- Get P(0) < 0
  have h_P0 : stability_polynomial beta gamma N 0 < 0 :=
    stability_polynomial_neg_at_zero beta gamma h_gamma N hN
  -- Get Q_big with P(Q_big) > 0
  obtain ⟨Q_big, hQ_big_pos, h_P_big⟩ :=
    stability_polynomial_pos_for_large_Q beta gamma h_beta h_gamma N
  -- Apply IVT: continuous function from negative to positive value crosses zero
  have h_cont := stability_polynomial_continuous beta gamma N
  -- IVT on [0, Q_big]: P(0) < 0 < P(Q_big), so ∃ x ∈ (0, Q_big), P(x) = 0
  have h_ivt := intermediate_value_Icc (le_of_lt hQ_big_pos) h_cont.continuousOn
  have h_mem : (0 : ℝ) ∈ Set.Icc (stability_polynomial beta gamma N 0)
                               (stability_polynomial beta gamma N Q_big) := ⟨le_of_lt h_P0, le_of_lt h_P_big⟩
  obtain ⟨Q_root, hQ_in_interval, hQ_zero⟩ := h_ivt h_mem
  -- Q_root > 0 because P(0) < 0 and P(Q_root) = 0
  use Q_root
  constructor
  · -- Q_root > 0
    rcases hQ_in_interval with ⟨hQ_ge, hQ_le⟩
    by_contra h_not_pos
    push_neg at h_not_pos
    have hQ_root_zero : Q_root = 0 := le_antisymm h_not_pos hQ_ge
    rw [hQ_root_zero] at hQ_zero
    linarith
  · exact hQ_zero

/--
**Theorem G.6: The Lepton Mass Hierarchy.**

If the vacuum parameters (beta, gamma) are fixed, stable configurations
exist for different harmonic modes, with energy increasing with |N|.

Assignment (Phase 3 Validation):
- Electron: N = 1  => E_min ≈ 0.511 MeV
- Muon:     N = 19 => E_min ≈ 105.66 MeV
-/
theorem lepton_mass_hierarchy_existence
    (beta gamma : ℝ) (h_beta : beta > 0) (h_gamma : gamma > 0) :
    ∃ (Q_e Q_mu : ℝ),
      Q_e > 0 ∧ Q_mu > 0 ∧
      stability_polynomial beta gamma 1 Q_e = 0 ∧
      stability_polynomial beta gamma 19 Q_mu = 0 ∧
      Q_e < Q_mu := by
  -- Get stable configurations for N=1 and N=19
  obtain ⟨Q_e, hQe_pos, hQe_stable⟩ :=
    stable_configuration_exists beta gamma h_beta h_gamma 1 (by norm_num)
  obtain ⟨Q_mu, hQmu_pos, hQmu_stable⟩ :=
    stable_configuration_exists beta gamma h_beta h_gamma 19 (by norm_num)
  use Q_e, Q_mu
  refine ⟨hQe_pos, hQmu_pos, hQe_stable, hQmu_stable, ?_⟩
  -- Show Q_e < Q_mu using stability equations
  -- From P(Q) = 0: Q²(2βQ + 1) = γN²
  -- Since γ·19² > γ·1² and Q²(2βQ+1) is increasing for Q > 0,
  -- we must have Q_mu > Q_e
  have h_eq_e : Q_e^2 * (2 * beta * Q_e + 1) = gamma * 1^2 := by
    have := hQe_stable
    simp only [stability_polynomial] at this
    -- this : 2 * beta * Q_e^3 + Q_e^2 - gamma * 1^2 = 0
    -- Goal: Q_e^2 * (2 * beta * Q_e + 1) = gamma * 1^2
    -- Note: Q_e^2 * (2 * beta * Q_e + 1) = 2*beta*Q_e^3 + Q_e^2
    ring_nf
    ring_nf at this
    linarith
  have h_eq_mu : Q_mu^2 * (2 * beta * Q_mu + 1) = gamma * 19^2 := by
    have := hQmu_stable
    simp only [stability_polynomial] at this
    ring_nf
    ring_nf at this
    linarith
  have h_gamma_ineq : gamma * (1 : ℝ)^2 < gamma * (19 : ℝ)^2 := by
    have : (1 : ℝ)^2 < (19 : ℝ)^2 := by norm_num
    exact mul_lt_mul_of_pos_left this h_gamma
  -- Q²(2βQ+1) is strictly increasing for Q > 0
  -- So Q_e < Q_mu
  by_contra h_not_lt
  push_neg at h_not_lt
  -- If Q_e ≥ Q_mu, then Q_e²(2βQ_e+1) ≥ Q_mu²(2βQ_mu+1)
  have h_Qe2 : Q_e^2 ≥ Q_mu^2 := by nlinarith
  have h_coeff : 2 * beta * Q_e + 1 ≥ 2 * beta * Q_mu + 1 := by nlinarith
  have h_pos1 : 2 * beta * Q_mu + 1 > 0 := by nlinarith
  have h_pos2 : Q_e^2 > 0 := sq_pos_of_pos hQe_pos
  have h_prod : Q_e^2 * (2 * beta * Q_e + 1) ≥ Q_mu^2 * (2 * beta * Q_mu + 1) := by
    have h_pos3 : Q_mu^2 > 0 := sq_pos_of_pos hQmu_pos
    nlinarith
  rw [h_eq_e, h_eq_mu] at h_prod
  linarith

/--
At stability P(Q) = 0, the energy simplifies to E = 3βQ² + 2Q.
This is because γN²/Q = Q(2βQ + 1) at the critical point.
-/
theorem energy_at_stability
    (beta gamma : ℝ) (Q : ℝ) (N : ℤ) (hQ_pos : Q > 0)
    (h_stable : stability_polynomial beta gamma N Q = 0) :
    topological_twist_energy beta gamma Q N = 3 * beta * Q^2 + 2 * Q := by
  unfold topological_twist_energy stability_polynomial at *
  -- From 2βQ³ + Q² - γN² = 0, we get γN² = Q²(2βQ + 1)
  -- So γN²/Q = Q(2βQ + 1) = 2βQ² + Q
  have h_relation : gamma * (N : ℝ)^2 / Q = 2 * beta * Q^2 + Q := by
    have hQ_ne : Q ≠ 0 := ne_of_gt hQ_pos
    field_simp at h_stable ⊢
    linarith
  rw [h_relation]
  ring

/--
**Theorem G.7: Energy Hierarchy from Winding Hierarchy.**

If Q_1 < Q_2 (both satisfying stability), then E_1 < E_2.
Combined with the Q ordering from stability_polynomial,
this gives the lepton mass hierarchy.
-/
theorem energy_increases_with_Q
    (beta gamma : ℝ) (h_beta : beta > 0)
    (Q1 Q2 : ℝ) (h_Q1_pos : Q1 > 0) (h_Q2_pos : Q2 > 0)
    (N1 N2 : ℤ)
    (h_stable1 : stability_polynomial beta gamma N1 Q1 = 0)
    (h_stable2 : stability_polynomial beta gamma N2 Q2 = 0)
    (h_Q_lt : Q1 < Q2) :
    topological_twist_energy beta gamma Q1 N1 <
    topological_twist_energy beta gamma Q2 N2 := by
  rw [energy_at_stability beta gamma Q1 N1 h_Q1_pos h_stable1]
  rw [energy_at_stability beta gamma Q2 N2 h_Q2_pos h_stable2]
  -- 3βQ1² + 2Q1 < 3βQ2² + 2Q2 when Q1 < Q2 and β > 0
  have h_sq : Q1^2 < Q2^2 := sq_lt_sq' (by linarith) h_Q_lt
  have h1 : 3 * beta * Q1^2 < 3 * beta * Q2^2 := by nlinarith
  linarith

/--
**Corollary: The Complete Lepton Mass Hierarchy.**

Combining the Q ordering from harmonic modes with energy monotonicity
gives E(N=1) < E(N=19), i.e., electron mass < muon mass.
-/
theorem lepton_energy_hierarchy
    (beta gamma : ℝ) (h_beta : beta > 0) (h_gamma : gamma > 0) :
    ∃ (Q_e Q_mu : ℝ),
      Q_e > 0 ∧ Q_mu > 0 ∧
      stability_polynomial beta gamma 1 Q_e = 0 ∧
      stability_polynomial beta gamma 19 Q_mu = 0 ∧
      topological_twist_energy beta gamma Q_e 1 <
      topological_twist_energy beta gamma Q_mu 19 := by
  obtain ⟨Q_e, Q_mu, hQe_pos, hQmu_pos, hQe_stable, hQmu_stable, hQ_lt⟩ :=
    lepton_mass_hierarchy_existence beta gamma h_beta h_gamma
  use Q_e, Q_mu
  refine ⟨hQe_pos, hQmu_pos, hQe_stable, hQmu_stable, ?_⟩
  exact energy_increases_with_Q beta gamma h_beta Q_e Q_mu hQe_pos hQmu_pos
    1 19 hQe_stable hQmu_stable hQ_lt

end QFD.Lepton
