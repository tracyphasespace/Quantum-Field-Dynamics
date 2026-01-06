import Mathlib
import QFD.Hydrogen.PhotonSolitonStable

set_option autoImplicit false

namespace QFD

universe u
variable {Point : Type u}

/-!
# QBallStructure

A Lean-robust (discrete/lattice) formalization of the QFD Q-ball story.

We model the particle interior as `n` vacuum grid cells indexed by `Fin n`.
This lets us prove the key logical implications without bringing in PDE
existence theory or measure-theory integrals.

Main statements delivered:

* `electron_is_cavitation`:
    if charge = -1 (and κ > 0), then some cell has ρ < ρ_vac.
* `proton_is_compression`:
    if charge = +1 (and κ > 0), then some cell has ρ > ρ_vac.
* `stability_predicts_scale_cubed`:
    if centrifugal = elastic, then scale^3 = (Q*^2)/β.

All proofs are elementary order/algebra over ℝ.
-/

/-- A discrete particle configuration (vacuum grid cells). -/
structure QBallConfig (n : ℕ) where
  /-- Cellwise density ρ_i. -/
  density   : Fin n → ℝ
  /-- Cellwise local time-flow factor τ_i. -/
  timeFlow  : Fin n → ℝ
  /-- Net charge tag (ℤ). -/
  charge    : ℤ
  /-- A macroscopic scale (radius) used in the toy force-balance model. -/
  scale     : ℝ
  /-- Effective spin/rotor amplitude Q*. -/
  Q_star    : ℝ
  /-- We require positive scale so division is safe. -/
  h_scale_pos : scale > 0

/--
A Q-ball model adds "density/charge/time" axioms on top of `QFDModelStable`.

* `κ_charge` is a positive proportionality factor mapping density deviation to
  the numeric charge tag.
* The discrete "integral" is `∑ i, (ρ_i - ρ_vac)`.
-/
structure QBallModel (Point : Type u) extends QFDModelStable Point where
  n : ℕ
  rho_vac : ℝ
  κ_charge : ℝ
  h_κ_pos : κ_charge > 0

  /-- Axiom: time is inverse density (cellwise). -/
  time_is_inverse_density :
    ∀ (c : QBallConfig n) (i : Fin n), c.timeFlow i * c.density i = 1

  /-- Axiom: charge is scaled total density deviation from the vacuum. -/
  charge_is_density_contrast :
    ∀ (c : QBallConfig n),
      (c.charge : ℝ) = κ_charge * (∑ i : Fin n, (c.density i - rho_vac))

namespace QBallModel

variable {M : QBallModel Point}

/-- The discrete total density deviation Σ(ρ_i - ρ_vac). -/
def densityDeviationSum (c : QBallConfig M.n) : ℝ :=
  ∑ i : Fin M.n, (c.density i - M.rho_vac)

/-- Helper: rewrite the charge axiom in terms of `densityDeviationSum`. -/
lemma charge_eq_kappa_mul_dev (c : QBallConfig M.n) :
    (c.charge : ℝ) = M.κ_charge * densityDeviationSum (M := M) c := by
  simpa [densityDeviationSum] using M.charge_is_density_contrast c

/-- Solve the charge-density relation for the deviation sum. -/
lemma densityDeviationSum_eq_charge_div (c : QBallConfig M.n) :
    densityDeviationSum (M := M) c = (c.charge : ℝ) / M.κ_charge := by
  have hk_ne : (M.κ_charge : ℝ) ≠ 0 := ne_of_gt M.h_κ_pos
  -- Start from the axiom and rearrange.
  have hmul : (c.charge : ℝ) = M.κ_charge * densityDeviationSum (M := M) c :=
    charge_eq_kappa_mul_dev (M := M) c
  -- Convert to `dev * κ = charge` to use `eq_div_iff`.
  have hmul' : densityDeviationSum (M := M) c * M.κ_charge = (c.charge : ℝ) := by
    -- commutativity of multiplication in ℝ
    simpa [mul_comm, mul_left_comm, mul_assoc] using hmul.symm
  exact (eq_div_iff hk_ne).2 hmul'

/--
Core lemma: if the total deviation sum is negative, then some cell must be below vacuum density.
-/
lemma exists_density_lt_vac_of_dev_neg
    (c : QBallConfig M.n)
    (hneg : densityDeviationSum (M := M) c < 0) :
    ∃ i : Fin M.n, c.density i < M.rho_vac := by
  classical
  by_contra h
  -- Push negation inward: all cells are ≥ rho_vac.
  have hge : ∀ i : Fin M.n, M.rho_vac ≤ c.density i := by
    -- `h : ¬ ∃ i, density i < rho_vac`
    -- Push_neg converts this to: ∀ i, rho_vac ≤ density i
    push_neg at h
    exact h
  -- Then each term (density i - rho_vac) is ≥ 0, hence the sum is ≥ 0.
  have hsum_nonneg : 0 ≤ densityDeviationSum (M := M) c := by
    unfold densityDeviationSum
    refine Finset.sum_nonneg ?_
    intro i hi
    exact sub_nonneg.mpr (hge i)
  linarith

/--
Core lemma: if the total deviation sum is positive, then some cell must be above vacuum density.
-/
lemma exists_density_gt_vac_of_dev_pos
    (c : QBallConfig M.n)
    (hpos : 0 < densityDeviationSum (M := M) c) :
    ∃ i : Fin M.n, M.rho_vac < c.density i := by
  classical
  by_contra h
  -- Push negation inward: all cells are ≤ rho_vac.
  have hle : ∀ i : Fin M.n, c.density i ≤ M.rho_vac := by
    -- Push_neg converts ¬ ∃ i, rho_vac < density i to: ∀ i, density i ≤ rho_vac
    push_neg at h
    exact h
  -- Then each term (density i - rho_vac) is ≤ 0, hence the sum is ≤ 0.
  have hsum_nonpos : densityDeviationSum (M := M) c ≤ 0 := by
    unfold densityDeviationSum
    refine Finset.sum_nonpos ?_
    intro i hi
    exact sub_nonpos.mpr (hle i)
  linarith

/--
Electron theorem (cavitation): charge = -1 forces some cell to satisfy ρ < ρ_vac.

This is the discrete analogue of: if ∫(ρ-ρ_vac) dV < 0 then ∃x, ρ(x) < ρ_vac.
-/
theorem electron_is_cavitation
    (c : QBallConfig M.n)
    (h_charge : c.charge = -1) :
    ∃ i : Fin M.n, c.density i < M.rho_vac := by
  -- Use the charge-density relation to show the deviation sum is negative.
  have hdev : densityDeviationSum (M := M) c = (c.charge : ℝ) / M.κ_charge :=
    densityDeviationSum_eq_charge_div (M := M) c
  have hk_pos : (0 : ℝ) < M.κ_charge := M.h_κ_pos
  have hdev_neg : densityDeviationSum (M := M) c < 0 := by
    -- substitute charge = -1
    have : (c.charge : ℝ) = (-1 : ℝ) := by simpa [h_charge]
    -- dev = (-1)/κ, κ>0 => dev<0
    simpa [hdev, this] using (div_neg_of_neg_of_pos (show (-1 : ℝ) < 0 by linarith) hk_pos)
  exact exists_density_lt_vac_of_dev_neg (M := M) c hdev_neg

/--
Proton theorem (compression): charge = +1 forces some cell to satisfy ρ > ρ_vac.

We prove it by applying the compression lemma to the positive deviation.
-/
theorem proton_is_compression
    (c : QBallConfig M.n)
    (h_charge : c.charge = 1) :
    ∃ i : Fin M.n, M.rho_vac < c.density i := by
  classical
  -- First show the deviation sum is positive.
  have hdev : densityDeviationSum (M := M) c = (c.charge : ℝ) / M.κ_charge :=
    densityDeviationSum_eq_charge_div (M := M) c
  have hk_pos : (0 : ℝ) < M.κ_charge := M.h_κ_pos
  have hdev_pos : 0 < densityDeviationSum (M := M) c := by
    have : (c.charge : ℝ) = (1 : ℝ) := by simpa [h_charge]
    simpa [hdev, this] using
      (div_pos (show (0 : ℝ) < (1 : ℝ) by linarith)
              (show (0 : ℝ) < M.κ_charge from hk_pos))
  -- Positive total deviation forces at least one cell above ρ_vac.
  exact exists_density_gt_vac_of_dev_pos (M := M) c hdev_pos

/-!
## Mechanical stability (toy force balance)

We keep the same algebraic structure you proposed:

* F_out = (Q*^2) / R
* F_in  = β * R^2

Balance implies R^3 = (Q*^2)/β.
-/

/-- Outward (centrifugal) force proxy. -/
noncomputable def centrifugalForce (c : QBallConfig M.n) : ℝ :=
  (c.Q_star ^ 2) / c.scale

/-- Inward (elastic) force proxy. -/
noncomputable def elasticForce (c : QBallConfig M.n) : ℝ :=
  M.toQFDModel.β * (c.scale ^ 2)

/-- Mechanical stability = force balance. -/
def IsMechanicallyStable (c : QBallConfig M.n) : Prop :=
  centrifugalForce (M := M) c = elasticForce (M := M) c

/--
Balance law in cubed form:

If (Q*^2)/R = β R^2 and R>0 and β≠0, then R^3 = (Q*^2)/β.

This is the algebraic heart of the "size from stiffness" claim. You can choose
later whether to rewrite it as `R = Real.cbrt ((Q*^2)/β)`.
-/
theorem stability_predicts_scale_cubed
    (c : QBallConfig M.n)
    (hβ_ne : M.toQFDModel.β ≠ 0)
    (h_stable : IsMechanicallyStable (M := M) c) :
    c.scale ^ 3 = (c.Q_star ^ 2) / M.toQFDModel.β := by
  have hR_ne : c.scale ≠ 0 := ne_of_gt c.h_scale_pos
  -- Start from stability and clear denominators.
  have h1 : (c.Q_star ^ 2) = M.toQFDModel.β * (c.scale ^ 3) := by
    -- From: (Q^2)/R = β*R^2
    -- Multiply both sides by R:
    have := congrArg (fun t => t * c.scale) h_stable
    -- simplify
    unfold IsMechanicallyStable centrifugalForce elasticForce at this
    -- Left becomes (Q^2)/R * R = Q^2
    -- Right becomes (β*R^2)*R = β*R^3
    field_simp [hR_ne] at this
    -- After field_simp, `this` is the desired identity (maybe up to commutativity).
    -- Normalize multiplicative order.
    simpa [mul_assoc, mul_left_comm, mul_comm, pow_succ] using this
  -- Divide both sides by β.
  -- h1.symm has the exact content we need up to commutativity:
  --   β * (c.scale^3) = c.Q_star^2
  -- We rewrite it into the `eq_div_iff` form.
  exact (eq_div_iff hβ_ne).2 (by
    -- Need: (c.scale^3) * β = c.Q_star^2
    simpa [mul_comm, mul_left_comm, mul_assoc] using h1.symm)

end QBallModel

end QFD
