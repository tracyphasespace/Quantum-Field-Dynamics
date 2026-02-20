# QFD New Questions & Gaps Beyond Current Tracker

**Created**: 2026-02-19
**Source**: Systematic audit of book v9.9, Lean codebase (263 files), reviewer docs, TODO.md, challenges.md, RED_TEAM_ROADMAP.md, sne_open_items.py, edits43-67
**Purpose**: Items NOT tracked in `open_Questions.md` that need resolution for peer-review readiness

---

## SEVERITY KEY

| Tag | Meaning |
|-----|---------|
| **FATAL** | A hostile reviewer kills the paper with this |
| **SERIOUS** | Undermines a major claim; needs resolution or honest flagging |
| **MODERATE** | Coverage gap; reduces perceived completeness |
| **LOW** | Technical debt; doesn't block publication |

---

## A. FATAL GAPS (Reviewer Kill Shots Against QFD)

### A.1 Deep Inelastic Scattering / Parton Physics
**Source**: challenges.md (lines 79-89)
**Severity**: FATAL
**Difficulty**: EXTREME

QFD has no quantitative treatment of: Bjorken scaling and its violations, parton distribution functions, jet fragmentation cross-sections, R-ratio in e+e- annihilation. This is labeled "QFD's single biggest gap."

**What's needed**: A soliton form factor F(q^2) that reproduces DIS cross-sections at all Q^2. This is a major theoretical program, not a single calculation.

**Defense strategy**: The book should explicitly acknowledge this as an open frontier (it does in Z.4.F.6) and argue that QFD's geometric confinement mechanism predicts the correct qualitative features (asymptotic freedom from vortex core resolution at high Q^2).

### A.2 6C-to-4D Inelastic Breakdown
**Source**: RED_TEAM_ROADMAP.md (lines 34-55)
**Severity**: FATAL
**Difficulty**: VERY HIGH

The spectral gap proof shows Delta_E > 0 exists, but nothing describes what happens when collision energy exceeds Delta_E. The state-mixing matrix elements <psi_4D | H_6C | psi_6D> are completely absent.

**What's needed**: Transition amplitudes between 4D-visible and 6D-internal modes. Without this, the theory only describes cold vacuum and low-energy phenomena.

### A.3 Electroweak W/Z Masses, Higgs, Weinberg Angle
**Source**: challenges.md (lines 109-115), edits49.md (lines 164-176)
**Severity**: FATAL
**Difficulty**: VERY HIGH

QFD has no W/Z boson masses from first principles, no Higgs mechanism equivalent, no electroweak precision observables. The Weinberg angle cos(theta_W) = M_W/M_Z = 0.882 should emerge from symmetry-breaking geometry.

**What's needed**: Full effective potential projected onto the Cl(3,1) subspace.

**Defense strategy**: The book positions W/Z as spectral gap excitations (§15.3.2) — this framing is correct but the computation is explicitly an open problem.

### A.4 Tau Superluminal Circulation (U_tau > c)
**Source**: challenges.md (lines 91-95)
**Severity**: SERIOUS → FATAL if not addressed
**Difficulty**: MEDIUM

The Hill vortex model gives U_tau > c for the tau lepton. Saturation corrections (V6/V8 terms) may resolve this, but it is not yet demonstrated.

**What's needed**: Show that Pade saturation corrections bring U_tau < c while preserving mass predictions. This is directly connected to the Gamma_sat formalism already in the book (edits67).

---

## B. SERIOUS GAPS (Major Claims Undermined)

### B.1 eta = pi^2/beta^2 First-Principles Derivation
**Source**: sne_open_items.py (lines 146-219)
**Severity**: SERIOUS
**Difficulty**: HIGH

The numerical coincidence eta = pi^2/beta^2 (1.24% match to fitted value 1.053) is used in the locked SNe model but the derivation from Kelvin wave coupling is explicitly labeled "INCOMPLETE." The chain requires knowing vortex core radius and vacuum correlation length in terms of beta.

**Blocks**: The "zero free parameters" claim for the SNe pipeline. Without this derivation, eta is a numerological coincidence, not a derived constant.

### B.2 SALT2 Chromatic Erosion Full Simulation
**Source**: sne_open_items.py (lines 229-342)
**Severity**: SERIOUS
**Difficulty**: MEDIUM-HIGH

The chromatic erosion model qualitatively reproduces asymmetry direction, but needs a full SALT2 mock pipeline to confirm quantitative conflation. Currently "QUALITATIVE" only.

**Blocks**: Kill Shot #2 (chromatic SN light curve asymmetry, tracker item 5.6).

### B.3 Why Three Generations?
**Source**: challenges.md (lines 117-120)
**Severity**: SERIOUS
**Difficulty**: HIGH

Both Koide and Hill vortex models accommodate three lepton generations but neither derives the number three from first principles. The Lean proofs show Cl(3,3) has the right structure, but "why 3" remains axiomatic.

**Mitigant**: The book argues three arises from the three stable topological isomers of the vortex (sphere/oblate/peanut). This needs to be made more explicit and, ideally, proved that no 4th stable isomer exists.

### B.4 Fractional Charge Values (2/3, -1/3) from Cl(3,3)
**Source**: Book Z.4.F.6 (explicit open problem)
**Severity**: SERIOUS
**Difficulty**: VERY HIGH

The charge quantization mechanism (topological vortex boundary conditions) is formalized in Lean (`Soliton/Quantization.lean`), but the specific fractional values (2/3, -1/3) are NOT derived from the Cl(3,3) structure.

### B.5 Cavitation Integration Verification
**Source**: TODO.md Gap 3 (lines 128-162)
**Severity**: SERIOUS
**Difficulty**: MEDIUM

The electron mass prediction relies on cavitation void halving the integrated energy density. This is marked "DONE*" with asterisk: "needs integration verify." If the cavitation integral doesn't give exactly 2x, the entire lepton mass derivation needs a different mechanism.

---

## C. MODERATE GAPS (Coverage & Consistency)

### C.1 hbar ~ sqrt(h) Consistency Check
**Source**: TODO.md Gap 2 (lines 111-113)
**Severity**: MODERATE
**Difficulty**: MEDIUM

The -3/8 CMB exponent requires hbar_local = hbar * sqrt(h(psi_s)). Three open questions: (1) Consistency with achromatic redshift? (2) Does photon soliton model produce this scaling independently? (3) Does this create energy quantization issues elsewhere?

### C.2 Neutrino Mass and Oscillations
**Source**: challenges.md (lines 103-107)
**Severity**: MODERATE
**Difficulty**: HIGH

QFD has no mechanism for neutrino mass or flavor oscillations. The Lean formalization (10+ files) treats neutrinos as algebraic remainders but the mass prediction (m_nu ~ 0.005 eV) is labeled "hypothesis, not theorem."

### C.3 Landau Critical Velocity (v_c = c_s/2)
**Source**: edits49.md (lines 194-196)
**Severity**: MODERATE
**Difficulty**: MEDIUM

If v_c = c_s/2 can be derived from vacuum dispersion, then U goes from "constrained parameter" to "derived." The spin prediction L = 0.4996*hbar (0.08% error) becomes a theorem instead of a coincidence.

### C.4 Pade Resummation Lean Proof
**Source**: edits67.md (line 341)
**Severity**: MODERATE
**Difficulty**: VERY HIGH

Proving the divergent V4 + V6 + ... series converges under Pade to Gamma_sat = 0.027 requires implementing rational approximation theory in Lean 4. No Mathlib support exists.

### C.5 Zombie Galaxies — No Lean Backing
**Source**: Book audit (§15.4, Ch.11)
**Severity**: MODERATE
**Difficulty**: LOW (formalization), HIGH (physics)

Over 90% of baryonic matter as non-luminous void structures. This is a falsifiable prediction with zero Lean formalization. `Cosmology/GalacticScaling.lean` addresses scaling laws but not the zombie galaxy mechanism.

### C.6 CMB Peak Positions Derivation
**Source**: Book audit (§10.4, App W.5.6)
**Severity**: MODERATE
**Difficulty**: MEDIUM

Book claims l_1=217, l_2=537, l_3=827 (within 2.1% of Planck). The peak spacing Delta-l = pi * chi_QFD / r_psi derivation is not formalized in Lean.

### C.7 Gravitational Wave Polarization
**Source**: Book audit (§15.3.3)
**Severity**: MODERATE
**Difficulty**: HIGH

QFD predicts GW as transverse elastic ringing of multivector vacuum. No Lean formalization. Gravity modules address geodesics, Schwarzschild, PPN but not GW radiation.

### C.8 N_max = 2*pi*beta^3 Formalization
**Source**: Book audit (Ch.8, line ~3815)
**Severity**: MODERATE
**Difficulty**: LOW

N_max = 2*pi*beta^3 = 177.087 (0.049% match). At ceiling: N/Z=3/2, A/Z=5/2, N/A=3/5. This is a striking result with no Lean proof — should be easy to add as a `norm_num` verification.

---

## D. LOW PRIORITY (Technical Debt)

### D.1 Orphaned Lean Files
**Source**: Lean codebase audit
- `QFD/Gravity/G_Derivation.lean` — 0 theorems, 8 defs, never consumed
- `QFD/Lepton/Physics/Postulates.lean` — stale placeholder, superseded by `Physics/Postulates.lean`

### D.2 Placeholder Definitions (41+ across 23 files)
**Source**: Lean codebase audit
Heaviest concentration in `Physics/Postulates.lean` (30+ placeholders for physics types). Others: `Cosmology/Polarization.lean` (C_ell_TE/EE = 0), `Weak/CPViolation.lean` (J_geometric = 1), `Nuclear/ProtonSpin.lean` (values 0.15/0.35).

### D.3 TODO Comments (10 files)
**Source**: Lean codebase audit
Key TODOs: pi_3(S^3) computation (TopologicalStability), generation-dependent radii (TopologicalCore), running coupling (CoreCompressionLaw), generic imports (PhotonSoliton*.lean).

### D.4 k_geom Downstream Migration
**Source**: open_Questions.md 1.5 (partially tracked)
10+ files still have independent k_geom definitions. `KGeomPipeline.lean` exists as single source of truth but downstream files haven't been updated to import it.

### D.5 Book Edit Application Backlog
**Source**: open_Questions.md 4.2 (partially tracked)
21 unapplied edits: edits64 (5), edits65 (6), edits66 (3), edits67 remaining (7). Ordering: edits64 → edits66 → edits65 → edits67.

### D.6 Neutrino Lean Formalization Roadmap (15 files)
**Source**: appendix_n_formalization_roadmap.md
Complete roadmap exists for 15 Lean files. Estimated 6-10 weeks. 3 tasks marked VERY HARD.

---

## E. LEAN COVERAGE MATRIX

Summary of book claims vs Lean backing:

| Domain | Lean Status |
|--------|-------------|
| Spacetime emergence (Cl(3,3) → Cl(3,1)) | COMPLETE |
| CMB Axis of Evil (11 theorems) | COMPLETE (paper-ready) |
| Golden Loop (alpha ↔ beta) | STRONG (24+ proofs) |
| Nuclear stability valley | STRONG (multiple files) |
| Proton mass bridge | STRONG (3+ files) |
| Neutrino as remainder | STRONG (10+ files) |
| Rift abundance (75/25 H/He) | MODERATE (6 files) |
| Electron g-2 | MODERATE (3 files) |
| SNe Hubble diagram | MODERATE (5+ files) |
| Topological proton stability | MODERATE (3 files) |
| PPN parameters | FORMALIZED (this session) |
| Tolman surface brightness | FORMALIZED (this session) |
| Faddeev-Popov Jacobian | FORMALIZED (this session) |
| Rift boundary distance | FORMALIZED (this session) |
| CP violation | MINIMAL (stub only) |
| Higgs mass from vacuum | NONE |
| W/Z boson masses | NONE |
| Gravitational waves | NONE |
| V6 shear / tau g-2 | NONE |
| BAO scale | NONE |
| Structure formation P(k) | NONE |
| 7Li abundance | NONE |
| Fractional charges from Cl(3,3) | NONE |
| Running alpha_s, DIS | NONE |
| Double-slit from QFD | NONE |

---

## F. CODEBASE HEALTH

| Metric | Value |
|--------|-------|
| Total .lean files | 263 |
| Total theorems + lemmas | 1,355 |
| sorry count | 0 |
| Remaining axioms | 2 |
| Placeholder definitions | 41+ across 23 files |
| TODO/FIXME comments | 10 files |
| Orphaned files | 2 |
| True stubs | 0 |
| Stale imports | 0 |

---

## G. PRIORITY RANKING FOR NEXT SESSIONS

### Immediate (this session, if time permits)
1. Round 3 code review (GoldenLoopLocation.lean, RadialHarmonicODE.lean) — **user must re-submit, lost to context compaction**
2. Apply remaining book edits (edits67 D-J, edits64/65/66)
3. N_max formalization (easy — `norm_num` on 2*pi*beta^3)

### Next session
4. Tau U_tau > c resolution (B.5 cavitation + A.4 tau saturation)
5. eta = pi^2/beta^2 derivation chain (B.1)
6. hbar ~ sqrt(h) consistency (C.1)

### Medium-term
7. CP violation stub → real proof (D.2)
8. Three generations derivation (B.3)
9. Landau critical velocity (C.3)
10. Neutrino Lean roadmap execution (D.6)

### Long-term research programs
11. DIS / parton physics (A.1) — QFD's biggest gap
12. 6C-to-4D inelastic breakdown (A.2)
13. Electroweak sector (A.3)
14. Constructive soliton existence
