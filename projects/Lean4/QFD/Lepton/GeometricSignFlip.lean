import QFD.Physics.Postulates
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Tactic.Linarith

noncomputable section

open QFD QFD.Physics

namespace QFD.Lepton

/-- 
**Theorem G.5: The Geometric Sign Flip.**

This theorem proves that the sign flip in the g-2 anomaly between the 
electron and muon is a GEOMETRIC NECESSITY.

Specifically:
1. For the electron (R_e = 1), V4 is strictly NEGATIVE.
2. For the muon (R_mu ≈ 0.0048), V4 is strictly POSITIVE.

This occurs because R_mu < R_vac < R_e, where R_vac = 1/√5.
-/
theorem geometric_sign_flip 
    (beta : ℝ) (h_beta_pos : beta > 0) :
    let R_e := (1 : ℝ)
    let R_mu := (0.51099895 / 105.6583755 : ℝ)
    -- Electron V4 is negative
    V4_geometric R_e beta < 0 ∧ 
    -- Muon V4 is positive
    V4_geometric R_mu beta > 0 := by
  
  -- Preliminaries: Constants are positive
  have h_phi_pos : phi_qfd > 0 := by
    unfold phi_qfd
    have h5 : 0 < Real.sqrt 5 := Real.sqrt_pos.mpr (by norm_num)
    linarith
  
  have h_xi_pos : xi_qfd > 0 := by
    unfold xi_qfd
    exact pow_pos h_phi_pos 2

  have h_beta_inv_pos : xi_qfd / beta > 0 := 
    div_pos h_xi_pos h_beta_pos

  -- 1. Prove R_vac < R_e (Electron scale factor is negative)
  have h_rvac : R_vac_qfd = 1 / Real.sqrt 5 := rfl
  have h_rvac_lt_1 : R_vac_qfd < 1 := by
    rw [h_rvac]
    -- 1/√5 < 1 because √5 > 1
    have h5_pos : (0 : ℝ) < Real.sqrt 5 := Real.sqrt_pos.mpr (by norm_num)
    have h5 : 1 < Real.sqrt 5 := by
      rw [Real.lt_sqrt (by norm_num : (0 : ℝ) ≤ 1)]
      norm_num
    calc (1 : ℝ) / Real.sqrt 5 < 1 / 1 := by
          apply div_lt_div_of_pos_left (by norm_num) (by norm_num) h5
       _ = 1 := by norm_num

  -- 2. Prove R_mu < R_vac (Muon scale factor is positive)
  have h_rmu : (0.51099895 / 105.6583755 : ℝ) < R_vac_qfd := by
    rw [h_rvac]
    -- 0.004836 < 1/2.236 ≈ 0.447
    have h_rmu_val : (0.51099895 / 105.6583755 : ℝ) < 0.005 := by norm_num
    -- √5 < 3 because √5 < √9 = 3
    have h_sqrt5_lt_3 : Real.sqrt 5 < 3 := by
      have h1 : Real.sqrt 5 < Real.sqrt 9 := by
        apply Real.sqrt_lt_sqrt (by norm_num : (0 : ℝ) ≤ 5)
        norm_num
      have h2 : Real.sqrt 9 = 3 := by
        have : (9 : ℝ) = 3 ^ 2 := by norm_num
        rw [this, Real.sqrt_sq (by norm_num : (0 : ℝ) ≤ 3)]
      linarith
    have h_rvac_val : (1 : ℝ) / 3 < 1 / Real.sqrt 5 := by
      apply one_div_lt_one_div_of_lt (Real.sqrt_pos.mpr (by norm_num : (5 : ℝ) > 0))
      exact h_sqrt5_lt_3
    have h_third : (0.3 : ℝ) < 1 / 3 := by norm_num
    linarith

  -- Now evaluate V4_geometric for both cases
  constructor
  · -- Case 1: Electron
    unfold V4_geometric
    let S_e := (R_vac_qfd - 1) / (R_vac_qfd + 1)
    have h_se_neg : S_e < 0 := by
      have h_num : R_vac_qfd - 1 < 0 := by linarith
      have h_den : R_vac_qfd + 1 > 0 := by 
        have h_rv_pos : R_vac_qfd > 0 := div_pos (by norm_num) (Real.sqrt_pos.mpr (by norm_num))
        linarith
      exact div_neg_of_neg_of_pos h_num h_den
    -- V4 = S_e * (xi/beta). Since S_e < 0 and xi/beta > 0, V4 < 0.
    exact mul_neg_of_neg_of_pos h_se_neg h_beta_inv_pos

  · -- Case 2: Muon
    unfold V4_geometric
    have h_smu_pos : (R_vac_qfd - 0.51099895 / 105.6583755) /
                     (R_vac_qfd + 0.51099895 / 105.6583755) > 0 := by
      have h_num : R_vac_qfd - 0.51099895 / 105.6583755 > 0 := by linarith
      have h_den : R_vac_qfd + 0.51099895 / 105.6583755 > 0 := by
        have h_rv_pos : R_vac_qfd > 0 := div_pos (by norm_num) (Real.sqrt_pos.mpr (by norm_num))
        have h_rm_pos : (0.51099895 / 105.6583755 : ℝ) > 0 := by norm_num
        linarith
      exact div_pos h_num h_den
    -- V4 = S_mu * (xi/beta). Since S_mu > 0 and xi/beta > 0, V4 > 0.
    exact mul_pos h_smu_pos h_beta_inv_pos

end QFD.Lepton
