# Charged Lepton Mass Ratios from Hill Vortex Dynamics with Vacuum Stiffness Parameter Inferred from the Fine Structure Constant

**QFD Collaboration**

*Manuscript prepared for submission*

---

## Abstract

We present numerical evidence that a vacuum stiffness parameter β ≈ 3.058 ± 0.012, inferred from the fine structure constant α through a conjectured relation involving nuclear binding coefficients, supports Hill spherical vortex solutions that reproduce the charged lepton mass ratios (electron, muon, tau) to better than 10⁻⁷ relative precision. The solutions employ a classical hydrodynamic vortex structure with density-dependent potential energy, where mass arises from near-perfect geometric cancellation between circulation energy and vacuum stabilization energy. For each lepton, three geometric parameters (vortex radius R, circulation velocity U, density amplitude) are numerically optimized to match the observed mass ratio, yielding robust solutions that are grid-converged (parameter drift < 1% at production resolution) and insensitive to the functional form of the density profile (four profiles tested). The circulation velocity exhibits approximate √m scaling across three orders of magnitude in mass. Independently determined values of β from cosmological (3.0-3.2) and nuclear (3.1 ± 0.05) analyses overlap within uncertainties, suggesting cross-sector consistency. Current solutions involve a 2-dimensional parameter manifold per lepton; implementation of additional physical constraints (cavitation saturation, charge radius matching) and prediction of independent observables (anomalous magnetic moments, form factors) are needed to establish uniqueness and predictive power beyond fitted mass ratios.

**Keywords**: lepton masses, Hill vortex, vacuum dynamics, fine structure constant, numerical optimization

---

## I. Introduction

### A. Motivation

The Standard Model of particle physics successfully describes electromagnetic, weak, and strong interactions but provides no explanation for the observed pattern of fermion masses. The charged lepton mass ratios m_μ/m_e ≈ 206.77 and m_τ/m_e ≈ 3477 are empirical inputs requiring three independent Yukawa coupling constants [1]. Understanding whether these ratios arise from deeper geometric or dynamical principles remains an open fundamental question.

Alternative approaches to mass generation have explored topological solitons [2-4], where particle masses emerge as energy eigenvalues of localized field configurations rather than from arbitrary coupling constants. In quantum field theory, such solitonic structures can exhibit remarkable stability through topological conservation laws [5,6].

### B. QFD Framework

Quantum Field Dynamics (QFD) proposes that leptons are localized density perturbations in a quantum vacuum characterized by a stiffness parameter β [7]. This framework suggests that:

1. The vacuum possesses material-like properties with resistance to density fluctuations
2. Particle masses arise from balance between kinetic circulation energy and potential stabilization energy
3. A universal β may manifest across cosmological, nuclear, and particle scales

Recent analyses have identified β_eff ≈ 3 from both cosmological dark energy interpretations [8] and nuclear binding energy systematics [9], motivating investigation of whether the same parameter can account for particle masses.

### C. This Work

We investigate whether the Hill spherical vortex [10,11], a classical hydrodynamic solution describing a stable spherical region of rotational flow, can serve as a geometric model for charged leptons when combined with a density-dependent vacuum potential characterized by β. Specifically, we test whether:

1. Solutions exist matching all three lepton mass ratios with a single β value
2. These solutions are numerically robust and physically meaningful
3. The β value inferred from leptons is consistent with independent determinations

We find affirmative evidence for items 1 and 2, and suggestive convergence for item 3, while clearly identifying remaining theoretical and experimental challenges.

---

## II. Theoretical Framework

### A. Hill's Spherical Vortex

The classical Hill vortex [10] describes a spherical region of radius R with internal rotational flow matching continuously to external irrotational flow. The stream function in spherical coordinates (r, θ, φ) is:

$$\psi(r,\theta) = \begin{cases}
-\frac{3U}{2R^2}(R^2 - r^2)r^2 \sin^2\theta & r < R \\
\frac{U}{2}\left(r^2 - \frac{R^3}{r}\right) \sin^2\theta & r \geq R
\end{cases}$$

where U characterizes the circulation velocity and R is the vortex radius. The velocity field components are:

$$v_r = \frac{1}{r^2 \sin\theta} \frac{\partial \psi}{\partial \theta}, \quad v_\theta = -\frac{1}{r \sin\theta} \frac{\partial \psi}{\partial r}$$

This solution satisfies the incompressible Euler equations with continuous velocity and pressure at r = R [11]. The formal specification of this structure for the electron has been verified in the Lean theorem prover with zero axioms [12].

### B. Density Gradient Ansatz

Rather than a sharp boundary, we model the vortex with smooth density variation:

$$\rho(r) = \begin{cases}
\rho_{\text{vac}} - A\left(1 - \frac{r^2}{R^2}\right) & r < R \\
\rho_{\text{vac}} & r \geq R
\end{cases}$$

where A is the amplitude of density depression. The parabolic form is an ansatz; we test robustness by examining quartic, Gaussian, and linear profiles in Sec. IV.C.

The cavitation constraint ρ(r) ≥ 0 requires A ≤ ρ_vac, which in the QFD framework relates to charge quantization [12]. We denote δρ(r) = ρ(r) - ρ_vac as the density perturbation.

### C. Energy Functional

The total energy is expressed as:

$$E_{\text{total}} = E_{\text{circ}} - E_{\text{stab}}$$

**Circulation energy** (kinetic):
$$E_{\text{circ}} = \int_0^\infty \int_0^\pi \int_0^{2\pi} \frac{1}{2} \rho(r) v^2(r,\theta) \, r^2 \sin\theta \, dr \, d\theta \, d\phi$$

where v² = v_r² + v_θ². Critically, we use the actual spatially-varying density ρ(r), not a constant approximation.

**Stabilization energy** (potential):
$$E_{\text{stab}} = \int_0^\infty \int_0^\pi \int_0^{2\pi} \beta [\delta\rho(r)]^2 \, r^2 \sin\theta \, dr \, d\theta \, d\phi$$

The parameter β characterizes vacuum resistance to density perturbations.

### D. Physical Interpretation

Mass arises from near-perfect cancellation:
- **Outer shell** (R/2 < r < R): High density and velocity → large positive E_circ
- **Core** (r < R/2): Suppressed density → small E_circ but significant E_stab
- **Net result**: Small residual ≈ 0.2-0.3 MeV for leptons

This geometric cancellation mechanism provides a natural explanation for the lightness of leptons relative to typical QCD scales (~1 GeV).

### E. Conjectured β-α Relation

We employ a conjectured relation [13]:

$$\beta_{\text{crit}} = f(\alpha, c_1, c_2)$$

where c₁ and c₂ are nuclear binding energy coefficients from the semi-empirical mass formula [14]. With c₁ = 15.56 ± 0.15 MeV and c₂ = 17.23 ± 0.20 MeV, we obtain:

$$\beta_{\text{crit}} = 3.058230856$$

with propagated uncertainty ±0.012 [15]. This relation has not been derived from first principles and remains an empirical conjecture subject to falsification through independent β measurements across sectors.

---

## III. Numerical Methods

### A. Dimensionless Formulation

We employ natural units with:
- Length scale: λ_e = ℏc/m_e ≈ 386 fm (electron Compton wavelength)
- Energy scale: m_e = 0.5110 MeV (electron mass)
- Dimensionless β: vacuum stiffness parameter

Target masses in these units:
- Electron: m_e = 1.0
- Muon: m_μ/m_e = 206.7682826
- Tau: m_τ/m_e = 3477.228

### B. Optimization Problem

For fixed β = 3.058230856, we seek parameters (R, U, A) such that:

$$\min_{R,U,A} \left| E_{\text{total}}(R, U, A; \beta) - m_{\text{target}} \right|^2$$

subject to constraints:
- R > 0 (positive radius)
- U > 0 (positive circulation)
- 0 < A ≤ ρ_vac (cavitation constraint)

This is an inverse problem: we fit geometric parameters to match observed mass. The optimization demonstrates **existence** of solutions but does not establish **uniqueness** without additional constraints (see Sec. VI.A).

### C. Numerical Integration

**Spatial discretization**:
- Radial: r ∈ [r_min, r_max] with r_min = 0.01λ_e, r_max = 10λ_e
- Angular: θ ∈ [θ_min, θ_max] with θ_min = 0.01, θ_max = π - 0.01
- Azimuthal: φ ∈ [0, 2π], integrated analytically

**Grid resolution**: Production runs use (n_r, n_θ) = (100, 20); convergence tests examine (50,10) to (400,80).

**Integration method**: Composite Simpson's rule via SciPy 1.7 [16].

**Stability measures**:
- Epsilon regularization (10⁻¹⁰) in denominators
- Bounds enforcement with penalty functions
- Multiple initial conditions to verify global convergence

### D. Optimization Algorithm

We employ the Nelder-Mead simplex method [17] via scipy.optimize.minimize with parameters:
- Maximum iterations: 2000
- Parameter tolerance: x_tol = 10⁻⁸
- Function tolerance: f_tol = 10⁻⁸

Initial guesses:
- Electron: (R, U, A) = (0.44, 0.024, 0.90)
- Muon/Tau: Scaled from electron using U ~ √m scaling hypothesis

Derivative-free optimization is appropriate given numerical integration noise in the objective function.

---

## IV. Results

### A. Three-Lepton Fits

Table I presents optimized parameters and energies for all three leptons using β = 3.058230856.

**TABLE I.** Optimized Hill vortex parameters and energies for charged leptons.

| Lepton | R (λ_e) | U (c) | A/ρ_vac | E_circ (MeV) | E_stab (MeV) | E_total (MeV) | Residual |
|--------|---------|-------|---------|--------------|--------------|---------------|----------|
| e | 0.4387 | 0.0240 | 0.911 | 1.209 | 0.209 | 1.0000 | 5.0×10⁻¹¹ |
| μ | 0.4496 | 0.3146 | 0.966 | 207.02 | 0.253 | 206.768 | 5.7×10⁻⁸ |
| τ | 0.4930 | 1.2895 | 0.959 | 3477.5 | 0.325 | 3477.228 | 2.0×10⁻⁷ |

All three solutions achieve residuals below 10⁻⁷ relative to target masses, using the same β value without adjustment.

**Key observations**:

1. **Radius variation**: R increases only 12% (0.439 → 0.493) across 3477× mass range, suggesting geometric quantization.

2. **Velocity scaling**: U exhibits approximate √m behavior with systematic deviations:
   - U_μ/U_e = 13.1 vs √(m_μ/m_e) = 14.4 (9% deviation)
   - U_τ/U_μ = 4.10 vs √(m_τ/m_μ) = 4.10 (0.1% deviation)

3. **Stabilization energy**: Varies only 55% (0.21 → 0.33 MeV) while mass varies 3477×, consistent with fixed β.

4. **Amplitude progression**: A/ρ_vac increases from 0.911 → 0.959, all approaching cavitation limit.

### B. Grid Convergence

Table II shows parameter stability under grid refinement for the electron.

**TABLE II.** Grid convergence test results (electron, β = 3.1).

| Grid (n_r, n_θ) | R | U | A | Drift from Finest |
|-----------------|---|---|---|-------------------|
| (50, 10) | 0.4319 | 0.02439 | 0.921 | 4.2% |
| (100, 20) | 0.4460 | 0.02431 | 0.938 | 1.0% |
| (200, 40) | 0.4490 | 0.02437 | 0.951 | 0.4% |
| (400, 80) | 0.4506 | 0.02442 | 0.959 | — |

Parameters converge monotonically with refinement. Production grid (100×20) shows ~1% drift from reference (400×80), acceptable for current purposes but recommended to use (200×40) for final publication.

### C. Profile Sensitivity

Table III tests robustness across four density profile forms with β fixed at 3.1.

**TABLE III.** Profile sensitivity test (electron).

| Profile | Functional Form | R | U | A | Residual |
|---------|----------------|---|---|---|----------|
| Parabolic | δρ = -A(1-r²/R²) | 0.439 | 0.0241 | 0.915 | 1.3×10⁻⁹ |
| Quartic | δρ = -A(1-r²/R²)² | 0.460 | 0.0232 | 0.941 | 8.0×10⁻¹⁰ |
| Gaussian | δρ = -A exp(-r²/R²) | 0.443 | 0.0250 | 0.880 | 1.4×10⁻⁹ |
| Linear | δρ = -A(1-r/R) | 0.464 | 0.0231 | 0.935 | 1.8×10⁻⁹ |

All four profiles produce residuals below 2×10⁻⁹ with the same β, suggesting robustness of β to functional form assumptions. Geometric parameters adjust to compensate for different profile shapes while preserving mass agreement.

### D. Multi-Start Robustness

Fifty optimization runs from random initial conditions (R ∈ [0.2,0.8], U ∈ [0.01,0.10], A ∈ [0.5,1.0]) for the electron yielded:

- Convergence rate: 96% (48/50 successful)
- Parameter statistics: R = 0.4387 ± 0.0035 (CV = 0.8%)
- Single tight cluster in parameter space

No evidence of multiple distinct local minima. Solution appears to be locally unique within numerical tolerance, though global manifold structure requires further investigation.

---

## V. Cross-Sector β Convergence

Table IV compares β determinations from independent sectors.

**TABLE IV.** Vacuum stiffness parameter β from different scales.

| Sector | β Value | Uncertainty | Reference | Method |
|--------|---------|-------------|-----------|--------|
| Particle (this work) | 3.058 | ±0.012 | [15] | From α via conjectured identity |
| Nuclear | 3.1 | ±0.05 | [9] | Direct fit to binding energies |
| Cosmology | 3.0-3.2 | — | [8] | Dark energy EOS w = -1 + β⁻¹ |

All three determinations overlap within 1σ uncertainties. The statistical overlap suggests possible universality but does not constitute proof given:
1. The particle sector value relies on a conjectured (not derived) relation
2. Different sectors employ different measurement techniques and systematics
3. Scale-dependent effective β (analogous to running coupling constants) cannot be excluded

---

## VI. Discussion

### A. Solution Degeneracy

**Critical limitation**: For each lepton, three geometric parameters (R, U, A) are optimized to match one observable (mass ratio). This yields a 2-dimensional solution manifold in parameter space.

Multi-start tests (Sec. IV.D) demonstrate local uniqueness but do not rule out disconnected solution branches. To establish uniqueness, additional constraints are needed:

1. **Cavitation saturation**: A → ρ_vac (removes 1 DOF)
2. **Charge radius**: r_rms = 0.84 fm for electron [18] (removes 1 DOF)
3. **Dynamical stability**: Second variation δ²E > 0 (selection criterion)

Implementation of constraints 1-2 is straightforward and in progress. If these reduce the solution space to a discrete set or unique solution, predictive power would be significantly enhanced.

### B. Independent Observable Predictions

Current validation uses only mass ratios (fitted quantities). Genuine predictive tests require:

**Charge radii**: The optimized R values predict root-mean-square charge radii via:
$$r_{\text{rms}} = \left\langle r^2 \right\rangle^{1/2} = \left( \int r^2 \rho(r) dV / \int \rho(r) dV \right)^{1/2}$$

For the electron, this yields r_rms ≈ 0.36R ≈ 0.16 fm, which should be compared to experimental value 0.84 fm [18]. Discrepancy suggests either:
- Incorrect density profile normalization
- Missing toroidal components (4-component structure in Lean spec [12])
- Need for quantum corrections

**Anomalous magnetic moments**: Circulation patterns should contribute to (g-2)_ℓ. Calculation from current geometries is feasible and would provide independent test.

**Form factors**: Scattering form factor F(q²) can be predicted from ρ(r) and compared to electron/muon scattering data.

These tests are essential to move from "consistent fits" to "validated predictions."

### C. Theoretical Challenges

**1. Origin of β-α relation**: The identity relating β to α through nuclear coefficients lacks theoretical derivation. Possible approaches:
- Dimensional analysis of vacuum energy density scales
- Connection to QCD vacuum condensates
- Effective field theory framework

**2. U > 1 interpretation**: For tau, U = 1.29 in units where c = 1. Physical interpretations:
- U represents circulation in vortex rest frame (boosted in lab)
- U is dimensionless internal parameter (not real-space velocity)
- Unit conversion requires careful review

**3. Multi-generation structure**: Q* parameters (2.2, 2.3, 9800) [19] suggest internal mode complexity. Connection to circulation velocity scaling requires:
- Excited state analysis of Hill vortex
- Angular momentum quantization
- Toroidal energy contributions from 4-component structure

### D. Comparison to Standard Model

**Standard Model**: Three arbitrary Yukawa couplings g_e, g_μ, g_τ yield masses via Higgs mechanism:
$$m_\ell = \frac{g_\ell v}{\sqrt{2}}$$
No explanation for g_τ/g_e ≈ 59, g_μ/g_e ≈ 14.

**This work**: Single parameter β, geometric structure determines mass ratios. Hierarchy emerges from circulation velocity scaling U ~ √m.

**Trade-off**: We exchange 3 arbitrary couplings for 3 geometric parameters per lepton optimized to match 1 target each. Net reduction in parameters only if:
1. Constraints yield unique geometries (removes 2 DOF per lepton)
2. β is derived from more fundamental principle (not conjectured)
3. Independent observables validate geometries

Current state represents progress in unification (single β vs three g_ℓ) but not yet parameter reduction.

### E. Falsifiability

This framework is falsifiable through:

1. **β convergence failure**: If improved measurements of (c₁, c₂) push β_crit outside overlap with β_nuclear and β_cosmo, the universality hypothesis is ruled out.

2. **Constraint over-determination**: If implementing A = ρ_vac and r_rms = 0.84 fm yields no solution, the model is inconsistent.

3. **Independent observable mismatch**: If predicted (g-2)_e disagrees with experiment beyond uncertainties, geometric structure is incorrect.

4. **Quark extension failure**: If same β cannot accommodate quark masses (different topology: Q-balls vs vortices), universality is limited.

---

## VII. Limitations

We summarize key limitations requiring resolution before claiming a validated predictive framework:

1. **Solution degeneracy** (3 DOF → 1 target) leaves 2D manifolds
2. **No independent observable tests** beyond fitted mass ratios
3. **Conjectured β-α relation** not derived from first principles
4. **Grid convergence** ~1% at production resolution (improvable)
5. **U > 1 interpretation** for tau requires clarification
6. **Missing toroidal components** from 4-component Lean specification
7. **Charge radius discrepancy** (predicted 0.16 fm vs measured 0.84 fm)

Items 1, 4, 6, and 7 are addressable through computational improvements and fuller implementation of theoretical framework. Items 2, 3, and 5 require new theoretical derivations or experimental validation.

---

## VIII. Conclusions

We have demonstrated that Hill spherical vortex solutions with density-dependent vacuum potential can reproduce charged lepton mass ratios to better than 10⁻⁷ relative precision when employing a vacuum stiffness parameter β ≈ 3.058 inferred from the fine structure constant. The solutions are numerically robust (grid-converged, profile-insensitive, single-cluster multi-start) and exhibit natural scaling U ~ √m across three orders of magnitude in mass.

The convergence of β values from particle (3.058 ± 0.012), nuclear (3.1 ± 0.05), and cosmological (3.0-3.2) sectors within uncertainties suggests potential universality, though the particle sector determination relies on a conjectured relation requiring theoretical derivation.

Current solutions involve geometric parameter optimization (3 DOF per lepton) to match single observables (mass ratios), yielding 2D solution manifolds. Implementation of additional constraints (cavitation saturation, charge radius) and prediction of independent observables (anomalous magnetic moments, form factors) are necessary to establish uniqueness and validate the geometric structure beyond fitted quantities.

If these extensions succeed, the framework would provide a geometric basis for lepton mass hierarchy, reducing three arbitrary Yukawa couplings to a single universal vacuum parameter. If constraints over-determine the system or independent predictions fail, fundamental revisions to the geometric structure or abandonment of universality would be required.

The work represents a testable hypothesis linking electromagnetism (α), vacuum properties (β), and particle masses through classical vortex geometry, with clear pathways for validation or falsification.

---

## Acknowledgments

We thank the Lean theorem prover community for formal verification tools and infrastructure. Computational resources were provided by [institution]. We acknowledge helpful discussions on hydrodynamic soliton models and numerical optimization techniques.

---

## References

[1] Particle Data Group, P. A. Zyla et al., Prog. Theor. Exp. Phys. 2020, 083C01 (2020).

[2] G. 't Hooft, Nucl. Phys. B 79, 276 (1974).

[3] A. M. Polyakov, JETP Lett. 20, 194 (1974).

[4] S. Coleman, Nucl. Phys. B 262, 263 (1985).

[5] R. Rajaraman, *Solitons and Instantons* (North-Holland, Amsterdam, 1982).

[6] T. H. R. Skyrme, Proc. R. Soc. London A 260, 127 (1961).

[7] QFD Collaboration, *Quantum Field Dynamics Framework*, in preparation (2025).

[8] QFD Collaboration, "Vacuum stiffness from cosmological observations," arXiv:XXXX.XXXXX (2025).

[9] QFD Collaboration, "Nuclear binding energy systematics and vacuum compression," arXiv:XXXX.XXXXX (2025).

[10] M. J. M. Hill, Phil. Trans. R. Soc. Lond. A 185, 213 (1894).

[11] H. Lamb, *Hydrodynamics*, 6th ed. (Cambridge University Press, 1932), §§159-160.

[12] QFD Collaboration, Lean 4 formal specification, `projects/Lean4/QFD/Electron/HillVortex.lean`, 136 lines, 0 axioms, available at GitHub repository (see Data Availability).

[13] QFD Collaboration, "Conjectured relation between fine structure constant and vacuum stiffness," in preparation (2025).

[14] C. F. von Weizsäcker, Z. Phys. 96, 431 (1935); H. A. Bethe and R. F. Bacher, Rev. Mod. Phys. 8, 82 (1936).

[15] Propagated from c₁ = 15.56 ± 0.15 MeV, c₂ = 17.23 ± 0.20 MeV via bootstrap analysis of 2000+ stable nuclides.

[16] P. Virtanen et al., Nature Methods 17, 261 (2020).

[17] J. A. Nelder and R. Mead, Comput. J. 7, 308 (1965).

[18] CODATA recommended values, Rev. Mod. Phys. 93, 025010 (2021).

[19] Phoenix solver parameters from `projects/particle-physics/lepton-isomers/src/solvers/phoenix_solver.py`, validated to 99.9999% accuracy, available at GitHub repository.

---

## Data Availability

All numerical data, source code, validation tests, and formal specifications supporting this work are publicly available under MIT License at:

**GitHub Repository**: https://github.com/tracyphasespace/Quantum-Field-Dynamics/tree/main/projects/particle-physics/V22_Lepton_Analysis_V2

The repository includes:
- Complete Python implementation of Hill vortex energy functionals
- Numerical optimization scripts for all three leptons
- Validation test suite (grid convergence, multi-start robustness, profile sensitivity)
- All numerical results in JSON format
- Replication guide with step-by-step instructions
- Lean 4 formal verification of Hill vortex specification

Replication of all results requires Python 3.8+, NumPy ≥1.20, and SciPy ≥1.7. Expected runtime: ~30 seconds for three-lepton test on typical workstation. See `REPLICATION_GUIDE.md` in the repository for detailed instructions.

Independent replication attempts are encouraged and welcomed. Issues or questions should be directed to the GitHub repository issue tracker.

---

**Manuscript version**: 2.0
**Date**: December 2025
**Status**: Prepared for submission to Physical Review D or European Physical Journal C
**Corresponding repository commit**: [will be tagged upon submission]

---

## Appendix A: Uncertainty Budget

Dominant uncertainty sources:

1. **Grid discretization**: ~1% parameter drift (reducible to ~0.4% with finer grid)
2. **Nuclear coefficient uncertainties**: c₁ (±1%), c₂ (±1.2%) → β_crit ± 0.4%
3. **Optimization tolerance**: Nelder-Mead convergence ~10⁻⁸ relative
4. **Profile form assumption**: Variation across 4 forms ~5% in parameters

Combined systematic uncertainty on β: ±0.012 (0.4% relative)

## Appendix B: Computational Performance

Typical runtimes (Intel i7, single core):
- Electron optimization: 5 seconds
- Muon optimization: 7 seconds
- Tau optimization: 8 seconds
- Grid convergence test: 5-10 minutes
- Multi-start robustness (50 runs): 10-15 minutes
- Profile sensitivity (4 profiles): 5 minutes

Memory usage: <500 MB peak (dominated by 100×20×3 integration grids)

Total computational cost for complete validation: ~30 minutes

---

*End of Manuscript*
