# The Physics of the Quantum Field Dynamics (QFD) Supernova Model

**Version:** V21 Canonical
**Last Updated:** 2025-01-18

---

## 1. Core Hypothesis

The Quantum Field Dynamics (QFD) model provides an alternative physical explanation for the observed dimming of Type Ia supernovae (SNe Ia) without invoking cosmic acceleration or dark energy. The central hypothesis is that a photon's energy loss during its journey through the intergalactic medium is not solely dependent on the expansion of spacetime (cosmological redshift, z). Instead, there is an additional energy loss mechanism, termed **Flux-Dependent Redshift (FDR)**, where the amount of dimming is proportional to the photon flux itself.

This model posits that supernovae are not "standard candles" but are better described as **"standardizable explosions"** whose apparent brightness is modulated by both traditional plasma interactions and this novel flux-dependent effect.

---

## 2. The Fundamental Observable: Distance Modulus (μ)

**CRITICAL CONVENTION:** In astronomy, we observe **distance modulus** (μ), not flux directly. Understanding the sign convention is essential.

### 2.1. Standard Astronomical Definitions

**Flux (F):** Energy received per unit area per unit time
- Higher flux → **Brighter** object
- Lower flux → **Fainter** object

**Magnitude (m):** Logarithmic brightness scale
- Lower magnitude → **Brighter** object
- Higher magnitude → **Fainter** object
- **Formula:** m = -2.5 log₁₀(F) + C

**Distance Modulus (μ):** Difference between apparent and absolute magnitude
- **Formula:** μ = m - M = 5 log₁₀(D_L) - 5
- Where D_L is luminosity distance in parsecs

### 2.2. The QFD Observable

The QFD model predicts observed distance modulus:

```
μ_obs = -2.5 log₁₀(F_obs) + C
```

Where the observed flux is:

```
F_obs = L_peak · exp(α) · exp(-τ_total) / (4π D²)
```

**Key Parameters:**
- **L_peak:** Fixed, universal intrinsic peak luminosity (e.g., 1.5×10⁴³ erg/s)
- **α (alpha):** The dimming parameter (negative for faint, positive for bright)
- **τ_total:** Total optical depth from all scattering/absorption
- **D:** Distance to supernova

### 2.3. Connecting α to μ

The relationship between the internal dimming parameter α and the observable μ:

```
μ = -2.5/ln(10) · α + μ_static(z)
```

Where μ_static(z) is the distance modulus in a static (non-expanding) universe.

**Sign Convention:**
- α < 0 → Dimming → μ increases (fainter)
- α > 0 → Brightening → μ decreases (brighter)

**For the Paper:** We work directly in μ space:
- **Observable:** μ_obs
- **Theory:** μ_model
- **Comparison:** Δμ = μ_obs - μ_model
  - Δμ > 0: Supernova is **fainter** than model (dimming)
  - Δμ < 0: Supernova is **brighter** than model (lensing/flashlight)

---

## 3. The Physical Model of Opacity (τ)

The total opacity τ_total is the sum of two distinct physical effects: conventional plasma interaction and the novel FDR interaction.

```
τ_total = τ_plasma + τ_FDR
```

### 3.1. Plasma Opacity (τ_plasma)

This term represents conventional scattering and absorption by dust and plasma along the line of sight, similar to models like SALT2. It is modeled as a power law dependent on wavelength (λ).

**Equation:**
```
τ_plasma = A_plasma · (λ_B / λ)^β
```

**Parameters:**
- **A_plasma:** Per-SN nuisance parameter representing plasma/dust amount
- **β (beta):** Per-SN nuisance parameter for spectral slope of extinction
- **λ_B:** Reference B-band wavelength (440 nm)

**Physical Interpretation:**
- Represents Thomson scattering off free electrons
- Wavelength-dependent: blue light scattered more than red
- Time-dependent: effect decays as plasma expands (~30 days)

### 3.2. Flux-Dependent Redshift (FDR) Opacity (τ_FDR)

This is the core of the QFD model's new physics. It posits that the interaction cross-section for energy loss is dependent on the local photon flux density.

**Equation:**
```
τ_FDR = ξ · η' · √(F_dimmed / F_REF)
```

**Parameters:**
- **ξ (xi):** Global parameter - fundamental interaction strength/coupling constant of FDR
- **η' (eta_prime):** Global parameter - effective density of interacting plasma medium
- **F_dimmed:** Actual flux after accounting for all dimming (makes equation implicit)
- **F_REF:** Reference flux constant for dimensionless term

**Physical Mechanism:**
Photons traveling through quantum foam experience self-interaction. The scattering cross-section scales with local photon density, creating a nonlinear dimming effect that mimics cosmological expansion in ΛCDM.

---

## 4. The Cosmological Basis and the Necessity of Degeneracy

The model relates the global physical parameters (k_J, η', ξ) to the observed dimming α through a set of basis functions dependent on redshift z. The inherent nature of these functions and the underlying astrophysics make **parameter degeneracy a required physical feature**, not a numerical bug.

### 4.1. The Basis Functions (Φ)

The predicted dimming α_pred is a linear combination of three redshift-dependent basis functions:

**Basis Vector:**
```
Φ(z) = [ln(1 + z), z, z/(1 + z)]
```

**Physical Interpretation:**
- **ln(1 + z):** Captures standard low-redshift expansion behavior
- **z:** Represents linear component of Hubble flow
- **z/(1 + z):** Models saturation or non-linear effects at high redshift

**Prediction Equation:**
```
α_pred(z) = -(k_J · ln(1 + z) + η' · z + ξ · z/(1 + z))
```

(Note: The code uses orthogonalized form -Q·c for fitting, which is mathematically equivalent after back-transformation)

### 4.2. Physical Degeneracy and Collinearity

The three basis functions are mathematically **highly collinear** (i.e., they are not independent of each other over the observed redshift range). This results in a high condition number (κ ≈ 2×10⁵), making direct fit unstable.

**Crucially, this mathematical collinearity reflects a true physical degeneracy in the observed data:**

**Entangled Signals:**
A single supernova light curve is a composite signal. Its apparent brightness is affected by:
1. Its distance
2. The mass of its progenitor system
3. The density of its local environment
4. Potential gravitational lensing from companions (e.g., Binary Black Holes)

**Astrophysical Example:**
A supernova with a massive (5-100 M☉) binary companion at closer distance can produce a similar dimming signal to a standard supernova at farther distance. The effects of:
- Gravitational mass (k_J)
- Plasma environment (η')
- FDR interaction strength (ξ)

are **physically entangled**.

**Model Requirement:**
A rigid, non-degenerate model would fail to capture this complex interplay and would incorrectly attribute these physical effects to noise, leading to biased results. The QFD model's degenerate basis is **intentionally flexible** enough to describe this physical reality.

### 4.3. The Role of QR Decomposition

To solve the numerical instability of fitting a collinear model, the pipeline uses **QR Decomposition**.

**Process:**
1. The basis matrix Φ is decomposed into Q (orthogonal matrix) and R (upper-triangular matrix)
2. **Fitting:** MCMC samples coefficients c in the stable, orthogonal Q basis
3. **Interpretation:** Physically meaningful parameters (k_J, η', ξ) are recovered by back-transforming: `params = R · c`

This technique separates the numerical fitting challenge from the necessary physical model, allowing the fit to remain stable while preserving the required physical degeneracy.

---

## 5. Handling Astrophysical Realities: Outliers

Real supernova datasets contain significant outliers (~1/6 of the DES-SN5YR sample). These are not treated as statistical noise but as **physically significant events**, such as:

- Strong gravitational lensing
- Direct interaction with massive binary companion (BBH)
- Extreme progenitor environments

To model this, the global likelihood function uses **Student's t-distribution** instead of pure Gaussian.

**Likelihood:**
```
α_obs ~ StudentT(ν, α_pred, σ_alpha)
```

**The degrees-of-freedom parameter ν (nu)** controls the "heaviness" of the tails:
- Small ν (e.g., ν ≈ 6.5 from DES data) allows accommodation of extreme outliers
- Prevents outliers from disproportionately influencing the fit
- More robust and physically motivated statistical choice

---

## 6. Summary of Key Parameters

| Parameter | Symbol | Type | Physical Interpretation |
|-----------|--------|------|------------------------|
| **k_J** | k_J | Global | Gravitational constant proxy; relates to mass distribution and lensing effects |
| **Eta Prime** | η' | Global | Effective density of intergalactic plasma medium involved in FDR |
| **Xi** | ξ | Global | Fundamental coupling constant/interaction strength of FDR effect |
| **Dimming** | α | Per-SN | Logarithmic dimming factor relative to L_peak (internal variable) |
| **Distance Modulus** | μ | Per-SN | Observable magnitude difference (STANDARD OUTPUT) |
| **Peak Luminosity** | L_peak | Constant | Fixed, universal intrinsic peak luminosity of Type Ia SN |
| **Plasma Amount** | A_plasma | Per-SN | Nuisance parameter for conventional dust/plasma extinction |
| **Plasma Slope** | β | Per-SN | Nuisance parameter for spectral slope of extinction |
| **Intrinsic Scatter** | σ_alpha | Global | Intrinsic scatter in SN brightness after model applied |
| **DoF** | ν | Global | Degrees of freedom for Student-t likelihood; quantifies "outlierness" |

---

## 7. QFD vs ΛCDM: Key Differences

### 7.1. ΛCDM (Standard Cosmology)

**Assumptions:**
- Expanding spacetime with cosmological constant Λ
- Photon wavelengths stretched by (1+z) due to metric expansion
- Time dilation: observed timescales stretched by (1+z)
- Distance modulus: μ_ΛCDM = 5 log₁₀(D_L) - 5

**Predictions:**
- Stretch parameter: s = 1 + z (linear increase with redshift)
- Distance modulus follows Friedmann equations with Ω_m = 0.3, Ω_Λ = 0.7

### 7.2. QFD (Static Space + Photon Damping)

**Assumptions:**
- Static (non-expanding) spacetime
- Photon energy loss via FDR and plasma scattering
- NO cosmological time dilation
- Distance modulus: μ_QFD = μ_static + f(z) where f(z) accounts for FDR

**Predictions:**
- Stretch parameter: s ≈ 1.0 (flat, no time dilation)
- Distance modulus includes FDR correction term

---

## 8. Critical Sign Conventions for Paper

When writing the manuscript, always use:

**Observable:** Distance modulus μ_obs (magnitudes)
**Theory:** Distance modulus μ_model (magnitudes)
**Residual:** Δμ = μ_obs - μ_model

**Interpretation:**
- Δμ > 0: Data is **fainter** than model (needs more dimming)
- Δμ < 0: Data is **brighter** than model (lensed/flashlight)

**DO NOT USE** α (dimming parameter) in figures or main text. It is an internal solver variable. Convert to μ for all public-facing outputs.

---

## 9. The Degeneracy is a Feature, Not a Bug

**Traditional View:** Parameter degeneracy is bad. Models should have unique, well-constrained parameters.

**QFD Physical Reality:** The astrophysical environment is inherently degenerate:
- Mass of companion affects local gravitational field
- Plasma density affects scattering optical depth
- FDR strength affects energy loss rate

These effects are **physically coupled** in nature. A model that artificially separates them would be **unphysical**.

**Solution:** Embrace the degeneracy. Use stable numerical techniques (QR decomposition, MCMC) to handle it, but preserve the physical model structure.

---

## 10. Reference

This document defines the **canonical physics** for the QFD V21 analysis pipeline. All code, plots, and publications must conform to these definitions.

**Key Principle:** DO NOT modify the physics to make the code work. Modify the code to match the physics.

---

**Contact:** For questions about QFD physics interpretation, see ANALYSIS_SUMMARY.md and LCDM_VS_QFD_TESTS.md.
