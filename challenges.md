# QFD: Falsifiable Predictions, Known Limitations, and Experimental Tests

**Version**: 1.0 (February 2026)
**Purpose**: Honest accounting of what QFD predicts, where it fails, and what could kill it.

---

## 1. Falsifiable Predictions (Specific Numbers)

These are quantitative predictions that QFD makes and the Standard Model (SM) does not, or where QFD differs from SM. Each includes a concrete falsification threshold.

### 1.1 Nuclear Coefficients from Alpha Alone

| Quantity | QFD Prediction | Empirical (NuBase 2020) | Error | Falsification |
|----------|---------------|------------------------|-------|---------------|
| c1 (surface) | 0.496351 | 0.496297 | 0.011% | If c1_emp changes by >0.5% in future nuclear data |
| c2 (volume)  | 0.328598 | 0.327040 | 0.48%  | If c2_emp changes by >2% in future nuclear data |

**QFD claim**: c1 = (1-alpha)/2, c2 = 1/beta. Both derived from alpha with zero free parameters.
**SM comparison**: SM has no prediction for these; they are fitted parameters in the liquid-drop model.

### 1.2 Proton-Electron Mass Ratio

| Quantity | QFD Prediction | Experiment | Error |
|----------|---------------|------------|-------|
| m_p/m_e | 1836.124 (k_geom=4.4028) | 1836.153 | 0.0016% |
| m_p | 938.251 MeV | 938.272 MeV | 0.0023% |

**QFD claim**: m_p = k_geom * beta * (m_e/alpha), where k_geom is a vacuum-renormalized eigenvalue: k_geom = k_Hill × (π/α)^(1/5), combining bare vortex geometry with vacuum electromagnetic enhancement.
**Falsification**: If a first-principles computation of k_geom gives a value outside [4.38, 4.41], the derivation chain breaks.

### 1.3 Hubble Refraction Parameter K_J

| Quantity | QFD Prediction | Observation | Status |
|----------|---------------|-------------|--------|
| K_J | 85.76 km/s/Mpc | H0 = 67-74 km/s/Mpc (tension) | K_J != H0 |

**QFD claim**: Cosmological redshift is vacuum refraction. The dimensionless scattering rate κ̃ = xi_QFD * beta^(3/2) ≈ 85.6 is derived from alpha alone. The identification κ̃ → K_J [km/s/Mpc] is a numerical coincidence whose dimensional bridge is not yet derived. The physical prediction is the SHAPE of μ(z) (zero free physics parameters, χ²/dof = 1.005 against DES-SN5YR).
**Falsification**: If the shape of μ(z) deviates significantly from the QFD prediction at higher-z surveys (LSST, Roman), the scattering model is ruled out.
**Current status**: SN Ia light-curve broadening is attributed to chromatic dispersion (σ ∝ E²), not kinematic wave-crest conservation. The asymmetric broadening predicted by QFD is a testable signature.

### 1.4 CMB Acoustic Peaks (QFD-Native)

| Quantity | QFD Prediction | Planck Observation | Error |
|----------|---------------|--------------------|-------|
| Peak 1 position | l = 217 | l = 220 | 1.4% |
| Peak 2 position | l = 537 | l = 540 | 0.6% |
| Peak 3 position | l = 827 | l = 810 | 2.1% |
| 2nd/1st height ratio | 0.460 | 0.45 | 2.2% |
| SW plateau fraction | 18.7% | ~18% | ~4% |
| 1st peak spacing | 320 | 320 | 0.0% |

**QFD claim**: Peaks arise from crystallographic soliton lattice (structure factor + form factor), not baryon acoustic oscillations. Only 3 free parameters: r_psi, sw_amp, sigma_skin.
**Falsification**: If QFD cannot reproduce peaks 4-7 (currently only 3 detected), the crystallographic model is incomplete.

### 1.5 Lepton g-2 Anomalous Magnetic Moments

| Particle | QFD Prediction | Experiment | Error |
|----------|---------------|------------|-------|
| Electron a_e | 0.001159652 | 0.001159652 | 0.0013% |
| Muon a_mu | 0.001165917 | 0.001165921 | 0.0063% (within 1.3 sigma) |

**QFD claim**: g-2 = alpha/(2*pi) + V4*(alpha/pi)^2, where V4 is a scale-dependent vortex integral (not the QED C2 coefficient).
**Falsification**: If the muon g-2 experimental value shifts by >3 sigma from the QFD prediction, the V4 vortex model fails.

### 1.6 Vacuum Sound Speed

| Quantity | QFD Prediction | Implication |
|----------|---------------|-------------|
| c_s = sqrt(beta) * c | 1.745c | Vacuum is hyper-stiff solid |

**QFD claim**: The vacuum has a sound speed exceeding c (superluminal phase velocity, not signal velocity). This appears in the CMB solver as the effective sound speed.
**Falsification**: If any experiment detects superluminal signal propagation or rules out superluminal phase velocities in vacuum, this needs revision.

---

## 2. Known Limitations (Honest Assessment)

### 2.1 Deep Inelastic Scattering (DIS) -- MAJOR GAP

**Problem**: The Standard Model's greatest triumph is DIS/parton physics. QFD has no quantitative treatment of:
- Bjorken scaling and its violations
- Parton distribution functions
- Jet fragmentation cross-sections
- R-ratio in e+e- annihilation

**Status**: Exploratory. See `projects/particle-physics/soliton-fragmentation/` for initial ideas on soliton form factors and fission. This is QFD's single biggest gap.

**What would close it**: A soliton form factor F(q^2) that reproduces DIS cross-sections at all Q^2.

### 2.2 Tau Superluminal Circulation

**Problem**: The Hill vortex model gives U_tau > c for the tau lepton, which is unphysical.
**Status**: Under investigation. Saturation corrections (V6/V8 terms from Appendix V) may resolve this. See `projects/particle-physics/lepton-isomer-ladder/` for the extended model.
**What would close it**: Show that saturation corrections bring U_tau < c while preserving mass predictions.

### 2.3 SN Ia Time Dilation

**Problem**: Type Ia supernovae show light-curve stretch proportional to (1+z), consistent with expansion. QFD's refraction model must produce equivalent time dilation or be falsified.
**Status**: Partially addressed (photon transport simulations), but the (1+z) stretch remains a serious challenge.
**What would close it**: Derive (1+z) time dilation from vacuum refraction properties.

### 2.4 Neutrino Mass and Oscillations

**Problem**: QFD has no mechanism for neutrino mass or flavor oscillations.
**Status**: Not attempted. Would require extending the soliton model to neutral, weakly-interacting configurations.
**What would close it**: A topological model of neutrinos as very light soliton states with generation mixing.

### 2.5 Electroweak Unification

**Problem**: The SM successfully unifies electromagnetic and weak forces (W, Z bosons, Higgs mechanism). QFD has:
- No W/Z boson masses from first principles
- No Higgs mechanism equivalent
- No electroweak precision observables (rho parameter, sin^2(theta_W))
**Status**: Not attempted.

### 2.6 Three Generations -- Why Three?

**Problem**: Both Koide and Hill vortex models accommodate three lepton generations, but neither derives the number three from first principles.
**Status**: The Lean proofs show Cl(3,3) has the right structure, but "why 3" remains axiomatic.

### 2.7 GIGO Risk (3 DOF -> 3 Targets)

**Problem**: The Hill vortex lepton model fits 3 masses with 3 parameters (beta, c1, c2). This is fitting, not prediction.
**Mitigation**: The parameters are claimed to derive from alpha (zero free parameters in the full chain), and the same beta appears in nuclear physics. But independent confirmation is needed.

---

## 3. Experimental Tests (Ordered by Feasibility)

### Tier 1: Testable Now (Archival Data)

| Test | Data Source | QFD Prediction | Status |
|------|------------|----------------|--------|
| Nuclear Z(A) for superheavy elements | AME2020 | c1*A^(2/3) + c2*A | Validated for A < 300 |
| CMB peaks 4-7 | Planck 2018 | Crystallographic harmonics | Only 3 peaks reproduced |
| SN Ia distance-redshift | Pantheon+ | Refraction: d_L = (c/K_J)*ln(1+z)*(1+z) | Partially tested |
| Nuclear drip lines | FRIB data | Soliton stability boundary | Predicted |

### Tier 2: Testable with Dedicated Analysis

| Test | Requirement | QFD Prediction |
|------|------------|----------------|
| Electron charge radius | Precision ep scattering data | r_e from Hill vortex geometry |
| Proton form factor shape | JLab data | F(q^2) from soliton profile |
| CMB polarization parity | Planck polarization maps | Sign-flip in E-mode under axis inversion |

### Tier 3: Requires New Experiments

| Test | Experiment | QFD Prediction |
|------|-----------|----------------|
| Vacuum birefringence | PVLAS-type | Non-zero (vacuum is anisotropic medium) |
| Photon mass bound | Coulomb law precision | m_gamma = 0 exactly (topological) |
| Beta decay asymmetry | Precision neutron decay | Possible deviations from SM CKM |

### Tier 4: Decisive but Difficult

| Test | What It Would Prove |
|------|-------------------|
| Derive beta from lattice QCD | Whether beta = 3.043 emerges from SM foundations |
| DIS from soliton form factors | Whether quarks are soliton substructure (or not) |
| Gravitational wave speed = c | Already confirmed by GW170817 (QFD consistent) |

---

## 4. What Would Kill QFD

The following observations would **definitively falsify** the framework:

1. **c2 != 1/beta**: If improved nuclear mass data gives c2 differing from 1/beta by >1%, the Golden Loop -> nuclear connection breaks.

2. **SN Ia time dilation is exactly (1+z)**: If time dilation is proven to be purely kinematic expansion with zero deviation, the refraction model is dead.

3. **Proton mass ratio not geometric**: If a first-principles k_geom computation gives a value outside [4.3, 4.5], the Proton Bridge equation fails.

4. **DIS partons are pointlike**: If DIS measurements at ever-higher Q^2 continue to show pointlike partons with no soliton substructure, QFD cannot explain the strong interaction.

5. **CMB peaks require baryonic physics**: If the odd-even peak height asymmetry is shown to require baryon loading (as in LCDM), the crystallographic model fails.

6. **Vacuum is not a medium**: If precision tests of Lorentz invariance (e.g., AUGER, IceCube) show no dispersive vacuum effects at any energy, the "vacuum as medium" premise is weakened.

---

## 5. What QFD Gets Right That SM Doesn't Attempt

For balance, areas where QFD provides explanations that SM treats as inputs:

1. **Why alpha = 1/137**: Golden Loop derives it from vacuum topology (1 equation, 1 solution)
2. **Why m_p/m_e ~ 1836**: Proton Bridge derives the ratio from alpha + geometry
3. **Why nuclear c1, c2 have those values**: Derived from alpha, not fitted
4. **CMB peak positions from 3 parameters**: vs LCDM's 6+ parameters
5. **g-2 without Feynman diagrams**: V4 vortex integral replaces perturbative QED C2

---

## 6. Open Questions (No QFD Answer Yet)

- Why does the fine structure constant have the value it does? (QFD relates it to beta, but what determines beta's value fundamentally?)
- What is dark matter? (QFD has no candidate)
- What is the baryon asymmetry mechanism?
- How does gravity quantize? (QFD describes classical geometry, not quantum gravity)
- What happens at the Planck scale?

---

## Summary

QFD makes specific, falsifiable numerical predictions in nuclear physics, lepton physics, and cosmology. Its greatest strengths are the zero-free-parameter nuclear predictions and the Golden Loop alpha-beta connection. Its greatest weaknesses are the absence of DIS/parton physics and the SN Ia time dilation challenge. An honest assessment: QFD is a promising geometric framework with rigorous mathematical foundations (1,145 Lean theorems) but significant empirical gaps that must be closed before it can claim to supersede the Standard Model.
