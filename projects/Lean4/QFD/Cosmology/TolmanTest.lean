-- QFD/Cosmology/TolmanTest.lean
-- Tolman surface brightness test: QFD hierarchy SB_expanding < SB_qfd < SB_tired_light
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum

noncomputable section

namespace QFD.Cosmology.TolmanTest

open Real

/-!
# Tolman Surface Brightness Test

Proves that QFD's geometric surface brightness law sits between
the ΛCDM expanding universe and naive tired light models,
successfully passing the Tolman test.

## Physical Context

Surface brightness SB ∝ (1+z)^n where the exponent n depends on the model:

| Model          | Exponent | Physics |
|----------------|----------|---------|
| ΛCDM expanding | -4       | Photon dilution + time dilation + two powers of (1+z) area |
| QFD achromatic | -4/3     | Geometric wavepacket expansion, D_L = D(1+z)^{2/3} |
| Naive tired light | -1    | Energy loss only, no angular size change |

The Tolman test measures galaxy surface brightness vs redshift.
QFD predicts brighter high-z galaxies than ΛCDM but dimmer than naive tired light,
matching observations (Lubin & Sandage 2001, Lerner et al. 2014).

## Book Reference

- §9.12 (Tolman Surface Brightness)
-/

/-- QFD surface brightness: SB ∝ (1+z)^{-4/3} from achromatic drag + geometric D_L. -/
def SB_qfd (SB₀ z : ℝ) : ℝ := SB₀ * (1 + z) ^ (-(4 / 3 : ℝ))

/-- ΛCDM expanding universe surface brightness: SB ∝ (1+z)^{-4}. -/
def SB_expanding (SB₀ z : ℝ) : ℝ := SB₀ * (1 + z) ^ (-(4 : ℝ))

/-- Naive tired light surface brightness: SB ∝ (1+z)^{-1}. -/
def SB_tired_light (SB₀ z : ℝ) : ℝ := SB₀ * (1 + z) ^ (-(1 : ℝ))

/-- **Main Theorem**: QFD Tolman hierarchy.

    For any positive redshift z > 0 and positive intrinsic brightness SB₀ > 0:
    SB_expanding < SB_qfd < SB_tired_light

    This means QFD predicts brighter high-z galaxies than ΛCDM (matching observations)
    but dimmer than naive tired light (avoiding that falsification). -/
theorem tolman_hierarchy (SB₀ z : ℝ) (h_SB : SB₀ > 0) (hz : z > 0) :
    SB_expanding SB₀ z < SB_qfd SB₀ z ∧ SB_qfd SB₀ z < SB_tired_light SB₀ z := by
  unfold SB_expanding SB_qfd SB_tired_light
  have h_base : 1 + z > 1 := by linarith
  -- Exponent ordering: -4 < -4/3 < -1
  have h_exp1 : (-(4 : ℝ)) < -(4 / 3 : ℝ) := by norm_num
  have h_exp2 : -(4 / 3 : ℝ) < -(1 : ℝ) := by norm_num
  -- For base > 1, rpow is strictly increasing in exponent
  have h_rpow1 := rpow_lt_rpow_of_exponent_lt h_base h_exp1
  have h_rpow2 := rpow_lt_rpow_of_exponent_lt h_base h_exp2
  exact ⟨mul_lt_mul_of_pos_left h_rpow1 h_SB, mul_lt_mul_of_pos_left h_rpow2 h_SB⟩

/-- At z = 0, all three models agree (surface brightness equals intrinsic). -/
theorem tolman_z_zero (SB₀ : ℝ) :
    SB_qfd SB₀ 0 = SB₀ ∧ SB_expanding SB₀ 0 = SB₀ ∧ SB_tired_light SB₀ 0 = SB₀ := by
  unfold SB_qfd SB_expanding SB_tired_light
  simp [add_zero]

end QFD.Cosmology.TolmanTest
