import Mathlib
import QFD.Physics.Postulates
import QFD.Hydrogen.PhotonSolitonEmergentConstants
import QFD.Lepton.IsomerCore
import Mathlib.Data.Real.Basic
import Mathlib.Tactic.Linarith

set_option autoImplicit false

namespace QFD

universe u
variable {Point : Type u}

/-!
  # Lepton Isomers: The Geometric Origin of Mass

  This module builds on IsomerCore.lean to provide theorems about lepton stability,
  mass generation, and decay mechanisms.

  ## The Isomer Hypothesis

  1. The Hill Vortex has internal structure defined by RMS charge density Q*.
  2. Only specific Q* values are stable in a β≈3.043 vacuum.
  3. These stable Q* values correspond to the Lepton Generations.

  ## The Three Lepton Isomers (defined in IsomerCore.lean)

  - **Electron** (e⁻): Q* ≈ 2.2 (Ground State, Stable)
  - **Muon** (μ⁻): Q* ≈ 2.3 (First Excitation, Metastable)
  - **Tau** (τ⁻): Q* >> 1 (High Excitation, Unstable)
-/

namespace LeptonModel

/-!
  ## Theorems about Lepton Structure
-/

/--
  Theorem: Electron vs muon winding gap.
  The Q* of an electron is strictly less than that of a muon.
  This is a direct consequence of their disjoint stability intervals.
-/
theorem electron_muon_qstar_gap
    (M : LeptonModel Point)
    (e μ : Config)
    (h_e : IsElectron M e)
    (h_μ : IsMuon M μ) :
    M.Q_star e < M.Q_star μ := by
  -- Electron: Q* ∈ (2.1, 2.25), Muon: Q* ∈ [2.25, 2.4)
  have he_ub : M.Q_star e < 2.25 := h_e.2.2.1
  have hmu_lb : M.Q_star μ ≥ 2.25 := h_μ.2.1
  linarith

/--
  Theorem: Muon vs tau winding gap.
  The Q* of a muon is strictly less than that of a tau.
-/
theorem muon_tau_qstar_gap
    (M : LeptonModel Point)
    (μ τ : Config)
    (h_μ : IsMuon M μ)
    (h_τ : IsTau M τ) :
    M.Q_star μ < M.Q_star τ := by
  -- Muon: Q* < 2.4, Tau: Q* > 1000
  have hmu_ub : M.Q_star μ < 2.4 := h_μ.2.2.1
  have htau_lb : M.Q_star τ > 1000 := h_τ.2.1
  linarith

/-!
  ## Decay Definitions
-/

/--
  Definition: Decay relation.
  A parent decays to a daughter if the parent has higher Q* (more stress).
-/
def DecaysTo (M : LeptonModel Point) (parent daughter : Config) : Prop :=
  M.Q_star parent > M.Q_star daughter

/--
  Definition: Muon decay to electron.
  The muon relaxes to the electron state (lower Q*).
-/
def MuonDecay (M : LeptonModel Point) (μ e : Config) : Prop :=
  IsMuon M μ ∧ IsElectron M e ∧ DecaysTo M μ e

/--
  Theorem: Muon can decay to electron.
  This follows from the Q* ordering: μ has higher Q* than e.
-/
theorem muon_decays_to_electron
    (M : LeptonModel Point)
    (μ e : Config)
    (h_μ : IsMuon M μ)
    (h_e : IsElectron M e) :
    MuonDecay M μ e := by
  refine ⟨h_μ, h_e, ?_⟩
  have h := electron_muon_qstar_gap M e μ h_e h_μ
  exact h

/--
  Theorem: Tau can decay to muon.
  This follows from the Q* ordering: τ has higher Q* than μ.
-/
theorem tau_decays_to_muon
    (M : LeptonModel Point)
    (τ μ : Config)
    (h_τ : IsTau M τ)
    (h_μ : IsMuon M μ) :
    DecaysTo M τ μ := by
  have h := muon_tau_qstar_gap M μ τ h_μ h_τ
  exact h

/--
  Theorem: Tau can decay to electron.
  Transitivity: τ > μ > e in Q* ordering.
-/
theorem tau_decays_to_electron
    (M : LeptonModel Point)
    (τ e : Config)
    (h_τ : IsTau M τ)
    (h_e : IsElectron M e) :
    DecaysTo M τ e := by
  unfold DecaysTo
  have he_ub : M.Q_star e < 2.25 := h_e.2.2.1
  have htau_lb : M.Q_star τ > 1000 := h_τ.2.1
  -- Q_star τ > 1000 > 2.25 > Q_star e
  linarith

end LeptonModel
end QFD
