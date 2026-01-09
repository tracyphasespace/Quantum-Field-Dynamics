-- Specific imports instead of full Mathlib
import Mathlib.Data.Real.Basic
import Mathlib.Analysis.Real.Pi.Bounds
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring
import QFD.Hydrogen.PhotonSoliton

set_option autoImplicit false

namespace QFD

/-!
  # Photon + Hydrogen soliton: stability + momentum/wavelength addendum

  This file intentionally keeps the physics *axiomatized* but the logic *fully formal*.

  We extend the earlier `QFDModel` scaffold with additional structure needed for a
  "stable non-dispersive soliton" story:

  • a stability predicate `Stable`
  • a time evolution operator `Evolve`
  • a "shape preserved" witness: evolution is (translation ∘ internal phase rotation)

  Separately we define a photon record with (ω, k, lam_w) and prove:

  • p = ℏ k
  • k lam_w = 2π  ⇒  p ∝ 1/lam_w
  • ω = cVac k  ⇒  E = cVac p

  None of this solves PDEs in Lean. The point is to provide a *clean interface* between
  your analytic derivations (paper) and machine-checked bookkeeping / invariants (Lean).
-/

universe u
variable {Point : Type u}

/-- An extension of the base `QFDModel` carrying the extra knobs used for stability
    and non-dispersive propagation. -/
structure QFDModelStable (Point : Type u) extends QFDModel Point where
  /-- Vacuum light speed in the quiescent ground state. -/
  cVac : ℝ

  /-- Momentum functional of a localized configuration.
      (You can later refine this to a multivector/4-vector.) -/
  Momentum : Config → ℝ

  /-- Stability gate: e.g. Lyapunov stability / local minimum of an energy functional
      at fixed topological invariants. -/
  Stable : Config → Prop

  /-- Time evolution of configurations under the Ψ-field dynamics. -/
  Evolve : ℝ → Config → Config

  /-- Spatial translation along the propagation axis (1D for now). -/
  Shift : ℝ → Config → Config

  /-- Internal phase/rotor evolution (abstracted as a group action). -/
  PhaseRotate : ℝ → Config → Config

  /- Invariance axioms (minimal set for proofs below). -/

  evolve_preserves_charge : ∀ t c, (Evolve t c).charge = c.charge
  evolve_preserves_energy : ∀ t c, (Evolve t c).energy = c.energy
  evolve_preserves_momentum : ∀ t c, Momentum (Evolve t c) = Momentum c

  /-- Non-dispersive/orbit form: if `c` is a *stable soliton*, then for every time `t`
      the evolved configuration equals a translate+phase-rotate of the original.

      This is the formal handle for: "shape preserved, just moves and rotates".
  -/
  evolve_is_shift_phase_of_stable :
    ∀ c,
      PhaseClosed c → OnShell c → FiniteEnergy c → Stable c →
      ∀ t, ∃ x θ, Evolve t c = PhaseRotate θ (Shift x c)

  /-- Translating/rotating a valid soliton preserves the three existence gates. -/
  gates_invariant_under_shift_phase :
    ∀ c x θ,
      (PhaseClosed c ∧ OnShell c ∧ FiniteEnergy c) →
      (PhaseClosed (PhaseRotate θ (Shift x c)) ∧
       OnShell (PhaseRotate θ (Shift x c)) ∧
       FiniteEnergy (PhaseRotate θ (Shift x c)))

  /-- Stability is preserved under translation+phase-rotation (symmetry of the medium). -/
  stable_invariant_under_shift_phase :
    ∀ c x θ, Stable c → Stable (PhaseRotate θ (Shift x c))

namespace QFDModelStable

variable {M : QFDModelStable Point}

/-- "Stable soliton" = the three existence gates plus stability plus non-dispersive propagation. -/
def StableSoliton (M : QFDModelStable Point) : Type :=
  { c : Config //
      M.PhaseClosed c ∧ M.OnShell c ∧ M.FiniteEnergy c ∧
      M.Stable c ∧
      (∀ t, ∃ x θ, M.Evolve t c = M.PhaseRotate θ (M.Shift x c)) }

/-- Coerce a StableSoliton to its underlying Config. -/
def StableSoliton.toConfig (s : StableSoliton M) : Config := s.val

/-- Constructor: if you can exhibit a configuration meeting all gates, you can build a
    `StableSoliton` term in Lean. -/
def stableSoliton_of_config
    (c : Config)
    (hC : M.PhaseClosed c)
    (hS : M.OnShell c)
    (hF : M.FiniteEnergy c)
    (hStab : M.Stable c) :
    StableSoliton M := by
  refine ⟨c, ?_⟩
  refine ⟨hC, hS, hF, hStab, ?_⟩
  intro t
  simpa using M.evolve_is_shift_phase_of_stable c hC hS hF hStab t

/-- Persistence theorem: a stable soliton stays a stable soliton under evolution.

    This is the formal statement of "no viscosity → no dissipation → the soliton's
    coherence is preserved; it only translates and accumulates phase".
-/
theorem stableSoliton_persists
    (s : StableSoliton M) (t : ℝ) :
    ∃ s' : StableSoliton M,
      s'.toConfig = M.Evolve t s.toConfig := by
  classical
  rcases s with ⟨c, hC, hS, hF, hStab, hND⟩
  -- Use the nondispersive witness to express the evolved state as shift+phase.
  rcases hND t with ⟨x, θ, hE⟩
  let c' : Config := M.PhaseRotate θ (M.Shift x c)
  -- Gates and stability are invariant under shift+phase.
  have hG' :
      M.PhaseClosed c' ∧ M.OnShell c' ∧ M.FiniteEnergy c' := by
    -- expand c' and apply the model's invariance lemma
    simpa [c'] using M.gates_invariant_under_shift_phase c x θ ⟨hC, hS, hF⟩
  have hStab' : M.Stable c' := by
    simpa [c'] using M.stable_invariant_under_shift_phase c x θ hStab
  -- Non-dispersive orbit from the new state follows from stability + gates.
  have hND' : ∀ τ, ∃ x' θ', M.Evolve τ c' = M.PhaseRotate θ' (M.Shift x' c') := by
    intro τ
    exact M.evolve_is_shift_phase_of_stable c' hG'.1 hG'.2.1 hG'.2.2 hStab' τ
  -- Package everything as a StableSoliton.
  refine ⟨⟨c', ?_⟩, ?_⟩
  · exact ⟨hG'.1, hG'.2.1, hG'.2.2, hStab', hND'⟩
  · -- Identify the stored configuration with the actual evolution at time t.
    -- From hE: Evolve t c = PhaseRotate θ (Shift x c) = c'
    -- and `c` is the underlying config of `s`.
    simp only [StableSoliton.toConfig, c']
    exact hE.symm

/-!
  ## Photon with wavelength/wavenumber/momentum bookkeeping

  We model the photon as a propagating, non-dispersive wavelet of the Ψ-field.

  Here we only formalize:

  • wavelength relation: k · lam_w = 2π
  • momentum: p = ℏ k
  • dispersion (massless): ω = cVac · k
  • therefore: E = ℏ ω = cVac · p
-/

/-- Photon with angular frequency ω, wavenumber k, and wavelength lam_w.
    We postulate the exact geometric identity k·lam_w = 2π (no dispersion in vacuum). -/
structure PhotonWave where
  ω : ℝ
  k : ℝ
  lam_w : ℝ
  h_lam : lam_w ≠ 0
  hk_lam : k * lam_w = 2 * Real.pi

namespace PhotonWave

variable (M : QFDModelStable Point)

/-- Photon energy bookkeeping. -/
def energy (γ : PhotonWave) : ℝ := M.ℏ * γ.ω

/-- Photon momentum bookkeeping. -/
def momentum (γ : PhotonWave) : ℝ := M.ℏ * γ.k

/-- From k·λ = 2π and λ ≠ 0, derive k = 2π/λ. -/
theorem k_eq_twoPi_div_lambda (γ : PhotonWave) :
    γ.k = (2 * Real.pi) / γ.lam_w := by
  have h := congrArg (fun x => x / γ.lam_w) γ.hk_lam
  -- h : (γ.k * γ.lam_w) / γ.lam_w = (2π) / γ.lam_w
  -- simplify LHS using λ ≠ 0
  simpa [mul_div_cancel_right₀ γ.k γ.h_lam, mul_assoc] using h

/-- Momentum is inversely proportional to wavelength: p = ℏ·(2π/λ). -/
theorem momentum_eq_hbar_twoPi_div_lambda (γ : PhotonWave) :
    momentum M γ = (M.ℏ * (2 * Real.pi)) / γ.lam_w := by
  -- momentum = ℏ k, and k = 2π/λ
  simp only [momentum]
  rw [k_eq_twoPi_div_lambda γ]
  ring

/-- Massless dispersion in the quiescent vacuum: ω = cVac · k.

    In QFD, this is the emergent light-cone statement in the vacuum ground state.
-/
def MasslessDispersion (γ : PhotonWave) : Prop :=
  γ.ω = M.cVac * γ.k

/-- If ω = cVac·k, then E = cVac·p (the hallmark of massless propagation). -/
theorem energy_eq_cVac_mul_momentum
    (γ : PhotonWave)
    (hDisp : MasslessDispersion M γ) :
    energy M γ = M.cVac * momentum M γ := by
  -- E = ℏ ω = ℏ (c k) = c (ℏ k) = c p
  simp only [energy, momentum]
  -- Use the dispersion relation: ω = cVac * k
  rw [hDisp]
  -- Now goal is: ℏ * (cVac * k) = cVac * (ℏ * k)
  ring

end PhotonWave

/-!
  ## Optional: Absorption/emission with explicit photon momentum (1D recoil)

  This extends the earlier energy-gap relations by adding a momentum bookkeeping
  constraint. You can later upgrade momentum to a 4-vector and include recoil mass.
-/

/-- Hydrogen state with a center-of-mass momentum tag (1D).
    We reuse the earlier `Hydrogen` definition (from `PhotonSoliton.lean`). -/
structure HStateP (M : QFDModelStable Point) where
  H : QFDModel.Hydrogen (M.toQFDModel)
  n : ℕ
  P : ℝ

namespace HStateP

variable {M : QFDModelStable Point}

def energy (s : HStateP M) : ℝ := M.ELevel s.n

def momentum (s : HStateP M) : ℝ := s.P

end HStateP

/-- Absorption with recoil: same H-pair, level increases, energy/momentum conserved. -/
def AbsorbsP (M : QFDModelStable Point)
    (s : HStateP M) (γ : PhotonWave) (s' : HStateP M) : Prop :=
  s'.H = s.H ∧
  s.n < s'.n ∧
  s'.energy = s.energy + PhotonWave.energy M γ ∧
  s'.momentum = s.momentum + PhotonWave.momentum M γ

/-- Emission with recoil: same H-pair, level decreases, energy/momentum conserved. -/
def EmitsP (M : QFDModelStable Point)
    (s : HStateP M) (s' : HStateP M) (γ : PhotonWave) : Prop :=
  s'.H = s.H ∧
  s'.n < s.n ∧
  s.energy = s'.energy + PhotonWave.energy M γ ∧
  s.momentum = s'.momentum + PhotonWave.momentum M γ

/-- Absorption is valid if photon matches discrete energy gap, with recoil. -/
theorem absorptionP_of_gap
    {M : QFDModelStable Point} {H : QFDModel.Hydrogen (M.toQFDModel)}
    {n m : ℕ} (hnm : n < m)
    (P : ℝ) (γ : PhotonWave)
    (hGap : PhotonWave.energy M γ = M.ELevel m - M.ELevel n) :
    AbsorbsP M ⟨H, n, P⟩ γ ⟨H, m, P + PhotonWave.momentum M γ⟩ := by
  refine ⟨rfl, hnm, ?_, ?_⟩
  · -- energy bookkeeping
    have : M.ELevel m = M.ELevel n + PhotonWave.energy M γ := by
      linarith [hGap]
    simpa [HStateP.energy] using this
  · -- momentum bookkeeping is definitional with the chosen recoil tag
    simp [HStateP.momentum]

end QFDModelStable
end QFD
