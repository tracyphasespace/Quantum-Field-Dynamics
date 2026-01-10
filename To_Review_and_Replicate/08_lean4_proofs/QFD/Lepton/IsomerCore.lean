import Mathlib.Data.Real.Basic
import Mathlib.Topology.Order.LocalExtr
import QFD.Hydrogen.PhotonSolitonEmergentConstants

noncomputable section

namespace QFD

universe u
variable {Point : Type u}

/--
Lepton model extends the emergent constants with the winding profile `Q_star`
and the stability potential.
-/
structure LeptonModel (Point : Type u) extends EmergentConstants Point where
  Q_star : Config Point → ℝ
  StabilityPotential : ℝ → ℝ
  h_Qstar_nonneg : ∀ c, Q_star c ≥ 0
  h_V_bounded : ∃ V_min : ℝ, ∀ Q, StabilityPotential Q ≥ V_min

/-- Stable isomer configurations sit at local minima of the stability potential. -/
def IsStableIsomer (M : LeptonModel Point) (c : Config Point) : Prop :=
  IsLocalMin M.StabilityPotential (M.Q_star c)

/--
The electron, muon, and tau predicates (placeholders awaiting refined intervals).
-/
def IsElectron (M : LeptonModel Point) (c : Config Point) : Prop :=
  IsStableIsomer M c ∧
  M.Q_star c > 2.1 ∧ M.Q_star c < 2.25 ∧
  c.charge = -1

def IsMuon (M : LeptonModel Point) (c : Config Point) : Prop :=
  IsStableIsomer M c ∧
  M.Q_star c ≥ 2.25 ∧ M.Q_star c < 2.4 ∧
  c.charge = -1

def IsTau (M : LeptonModel Point) (c : Config Point) : Prop :=
  IsStableIsomer M c ∧
  M.Q_star c > 1000 ∧
  c.charge = -1

end QFD
