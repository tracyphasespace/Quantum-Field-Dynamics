# QFD Physics: Emergent Time Dilation from Component Confluence

**Author:** Tracy McSheery
**Date:** January 2025
**Purpose:** Core physics documentation for Quantum Field Dynamics supernova analysis

---

## Executive Summary

In QFD cosmology, **time dilation is not fundamental** - it is an **emergent appearance** created by the confluence of five distinct physical mechanisms operating at different scales with different wavelength dependencies. This architecture explains the ~15-20% of supernova data that ΛCDM models discard as "outliers" or "bad data," treating these observations as **real physical signals** from gravitational lensing, variable black hole masses, and field interactions.

**Key Insight**: What appears as FRW (1+z) time dilation is actually:
- Planck/Wien thermal broadening (wavelength-dependent)
- Plasma veil scattering after thermal effects (wavelength-dependent)
- Field Damping Redshift operating on dimmed flux (flux-dependent, hence wavelength-dependent)
- BBH gravitational effects (time-varying, mass-variable)
- Cosmological drag (wavelength-independent amplitude loss)

---

## The Five Core Physical Mechanisms

### 1. Black Holes with Variable Mass (1-1000× White Dwarf)

**Implementation**: `projects/astrophysics/qfd-supernova-v15/src/v15_model.py:400-443`

```python
def compute_bbh_gravitational_redshift(
    t_rest, A_lens, P_orb, t_rise, radius_peak, radius_fall_tau
):
    # Δz ≈ G * (M_total - M_WD) / (R_photosphere * c²)
    # Scaled by |A_lens| to modulate effective BBH mass
```

**Physics**:
- Binary Black Holes (BBH) with masses **1-1000× white dwarf mass**
- Creates gravitational redshift much larger than singular WD
- Modulated by orbital parameters and lensing amplitude
- Capped at MAX_GRAVITATIONAL_Z = 0.5 for numerical stability

**QFD Prediction**: All Type Ia supernovae are caused by BBH systems, not singular white dwarfs.

---

### 2. Local Gravitational Lensing (Time-Varying by Week)

**Implementation**: `v15_model.py:352-394`

```python
def compute_bbh_magnification(mjd, t0_mjd, A_lens, P_orb, phi_0):
    # μ(MJD) = 1 + A_lens * cos(2π * (MJD - t₀) / P_orb + φ₀)
```

**Physics**:
- **Time-dependent**: Changes by observation night/week
- BBH orbital motion causes magnification/demagnification
- Affects a fraction of SNe (~16% show strong effects)
- **Data changes by week**: Not noise, but physical orbital variation

**Observable**: Night-to-night flux variations in multi-epoch photometry

**Key Point**: What ΛCDM treats as "scatter" or "bad nights" is actually **real BBH orbital lensing**.

---

### 3. FDR (Field Damping Redshift) - Not in Standard Models

**Implementation**: `v15_model.py:294-349`

```python
def qfd_tau_total_jax(...):
    # τ_FDR = ξ * η' * √(flux_dimmed / flux_ref)
    # Self-consistent iterative solver
```

**Physics**:
- **Flux-dependent opacity**: Brighter sources scatter more
- **Self-consistent problem**:
  - Dimmed flux depends on τ
  - But τ_FDR depends on dimmed flux
- Solved iteratively (20 iterations, relaxation = 0.5)
- **Wavelength-dependent** indirectly (via flux)

**Critical Detail**: FDR operates on **already-dimmed flux** after Planck/Wien broadening and veil scattering have modified the spectral shape.

---

### 4. Plasma Veil - Not in Standard Models

**Implementation**: `v15_model.py:267-291`

```python
def qfd_plasma_redshift_jax(t_days, wavelength_nm, A_plasma, beta, tau_decay=30.0):
    temporal_factor = 1.0 - jnp.exp(-t_days / tau_decay)
    wavelength_factor = (LAMBDA_B / wavelength_nm) ** beta  # Wavelength dependence!
    return A_plasma * temporal_factor * wavelength_factor
```

**Physics**:
- **Wavelength-dependent**: z_veil ∝ (λ_B/λ)^β
- **Time-dependent**: Grows as plasma expands, saturates at ~30 days
- **Blue scatters more than red**: β > 0 means shorter wavelengths affected more
- Reference wavelength: λ_B = 440 nm (B-band)

**Observable**: Spectral evolution and color changes over time

---

### 5. E144 Photon-Photon Interaction

**Implementation**: `projects/astrophysics/redshift-analysis/RedShift/qfd_cmb/`

**Documented**: `RedShift/docs/PHYSICS_DISTINCTION.md:5-6, 95-98`
> "Both models use QFD (Quantum Field Dynamics) physics scaled from **SLAC E144 experiments**"
>
> "Both scale from SLAC E144 measurements but in different limits:
> - **RedShift**: Low-density, long-path-length regime
> - **Supernova**: High-density, short-path-length regime"

**Physics**:

**kernels.py** - Sin²(θ) scattering:
```python
def sin2_mueller_coeffs(mu):
    # Differential cross-section ∝ sin²(θ)
    w_T = 1.0 - mu**2  # intensity weight
    w_E = (1.0 - mu**2)  # polarization
```

**ppsi_models.py** - Oscillatory power spectra:
```python
def oscillatory_psik(k, A=1.0, ns=0.96, rpsi=147.0, Aosc=0.55, sigma_osc=0.025):
    # P_ψ(k) = A * k^(ns-1) * [1 + Aosc * cos(k * rpsi) * exp(-(k * sigma_osc)^2)]^2
```

**Two Regimes**:
1. **Cosmological (RedShift)**: Wavelength-independent, direct photon-ψ field interaction
2. **Supernova (Local)**: Wavelength-dependent, plasma-mediated scattering

---

### 6. QFD Redshift ≠ ΛCDM Redshift

**Implementation**: `v15_model.py:72-111`

| Aspect | ΛCDM | QFD |
|--------|------|-----|
| **Time dilation** | t_rest = t_obs/(1+z) | t_rest = t_obs (absolute time) |
| **Luminosity distance** | FRW formula | D_QFD(z, k_J, η', ξ) |
| **Redshift origin** | Space expansion | Photon-field interactions |
| **Components** | Single z | Multiple: z_cosmo + z_plasma + z_FDR + z_BBH |
| **Cosmology** | (1+z) factors everywhere | NO (1+z) factors - pure field theory |

**From V15_Architecture.md:94-98**:
```python
# V15 (pure QFD) - NO (1+z) factors!
t_rest = t_obs  # Absolute time, no FRW dilation
D_L = D_QFD(z_obs, k_J, eta_prime, xi)  # QFD-native distance
```

---

## The Confluence: Emergent Apparent Time Dilation

### The Core Architectural Principle

**Time dilation appears** from the combination/confluence of:

1. **Planck/Wien thermal broadening** (wavelength-dependent)
2. **Plasma veil scattering** AFTER thermal effects (wavelength-dependent)
3. **FDR on dimmed flux** AFTER veil (flux-dependent, hence wavelength-dependent)
4. **BBH gravitational effects** (time-varying, observation-dependent)
5. **Cosmological drag** (wavelength-independent amplitude loss)

### Wavelength Dependency Table

| Component | Wavelength Dependence | Amplitude Loss | Time Dependence |
|-----------|----------------------|----------------|-----------------|
| **Cosmological Drag (k_J)** | **Independent** | Distance-dependent | Redshift z |
| **Plasma Veil (η', β)** | **DEPENDENT**: (λ_B/λ)^β | Blue >> Red | t_days (temporal evolution) |
| **FDR (ξ)** | **Indirect** (via flux) | Flux-dependent | Via flux evolution |
| **Planck Blackbody** | **DEPENDENT**: B_λ(T) | Wien peak shifts | Temperature T(t) |
| **BBH Lensing (A_lens)** | Independent | Magnification varies | MJD (orbital period) |
| **BBH Gravitational (z_grav)** | Independent | Mass-dependent | Photosphere radius R(t) |

### The Sequential Cascade

**Order matters** - these effects operate in sequence:

```
1. Intrinsic Emission
   ↓ (Planck blackbody at T(t), wavelength-dependent)

2. Thermal Broadening
   ↓ (Wien's law shifts spectral peak)

3. Plasma Veil Scattering
   ↓ (wavelength-dependent: β exponent, blue scatters more)

4. FDR Operates on Dimmed Flux
   ↓ (flux-dependent: τ_FDR ∝ √flux, iterative solution)

5. BBH Lensing Magnification
   ↓ (time-varying: orbital phase, changes by week)

6. BBH Gravitational Redshift
   ↓ (mass-dependent: 1-1000× WD, photosphere depth)

7. Cosmological Drag
   ↓ (wavelength-independent: k_J * distance)

8. Observed Flux
```

**Implementation**: `v15_model.py:450-550` (full flux model combining all components)

### Mathematical Expression of Confluence

**From v15_model.py:63-107**:

```python
def alpha_pred(z, k_J, eta_prime, xi):
    """
    Combines three QFD attenuation channels.
    Signs chosen so increasing z produces dimming (negative alpha).
    """
    phi1 = jnp.log1p(z)      # Cosmological drag: ln(1+z)
    phi2 = z                  # Plasma veil: linear in z
    phi3 = z / (1.0 + z)      # FDR: saturating at high z

    return -(k_J * phi1 + eta_prime * phi2 + xi * phi3)
```

**This creates the APPEARANCE of (1+z) time dilation** without actually invoking FRW metric or expansion.

---

## Explaining the "Outlier" Data

### The Problem with ΛCDM

**From V15_Architecture.md:57-59**:
> "**Evidence**:
> - **84.2% of fits have chi2/ndof > 2.0**
> - Median chi2/ndof = 7.1 (residuals ~2.7σ off)
> - This is NOT noise or contamination - it's **missing time-varying physics**"

ΛCDM treats this as:
- "Bad data"
- Systematic errors
- Contamination
- **Solution**: Discard or heavily downweight

### The QFD Solution

**Treat it as real physics**:

1. **Time-varying BBH lensing** explains night-to-night variations
2. **Variable BBH masses** explain the scatter in peak brightness
3. **Wavelength-dependent veil** explains color evolution outliers
4. **Flux-dependent FDR** explains brightness-dependent scatter

**Statistical Treatment**: `stage2_mcmc_numpyro.py:491-496`

```python
# Student-t degrees of freedom for heavy-tail robustness
nu = numpyro.sample('nu', dist.Exponential(0.1)) + 2.0

# Student-t likelihood (NOT Gaussian)
numpyro.sample('ln_A_obs',
               dist.StudentT(df=nu, loc=alpha_pred, scale=sigma_total))
```

**From README.md:488**:
> "**Two-Component Mixture**:
>   - Core: Normal(α_pred, σ_α) for clean SNe
>   - Tail: Normal(α_pred + b_occ, κσ_α) for BBH/occluded SNe
>   - Fit (π, b_occ, κ) to **isolate ~16% tail** without biasing core"

### The ~16-20% Fraction

**Observable Signatures**:
- Q-Q plots show **heavy tails** (Student-t distribution)
- Residual histograms show **non-Gaussian wings**
- Time-series show **systematic variations** (not random noise)
- Wavelength-dependent **color anomalies**

**Physical Interpretation**:
- ~16% of SNe have **strong BBH lensing effects**
- These show time-varying magnification by >0.1 mag
- Orbital periods: days to weeks
- **Data changes by observation week** - real signal!

---

## Thermal Broadening: Planck/Wien Effects

### Temperature Evolution

**Implementation**: `v15_model.py:130-135`

```python
def _temperature(t_rest, temp_peak, temp_floor, temp_tau):
    """Exponential cooling from peak to floor."""
    decay = jnp.exp(-t_rest / temp_tau)
    return temp_floor + (temp_peak - temp_floor) * decay
```

**Typical Values**:
- T_peak ≈ 10,000 - 15,000 K (early times)
- T_floor ≈ 3,000 - 5,000 K (late times)
- τ_temp ≈ 30-50 days

### Wien's Law: Spectral Peak Shift

**As temperature cools**:
```
λ_peak = b / T

T decreases → λ_peak increases (shifts to red)
```

**This creates a REDSHIFT** in spectral features that **mimics time dilation** but is actually thermal cooling.

### Planck Function: Wavelength-Dependent Emission

**Implementation**: `v15_model.py:198-204`

```python
wavelength_cm = wavelength_nm * 1e-7
expo = H_PLANCK * C_CM_S / (wavelength_cm * K_BOLTZ * T_eff)
planck = (2.0 * H_PLANCK * C_CM_S**2) / (
    wavelength_cm**5 * jnp.expm1(jnp.clip(expo, a_max=100))
)
```

**B_λ(T) = (2hc²/λ⁵) × 1/(e^(hc/λkT) - 1)**

**Wavelength dependence**:
- Blue (short λ): Strong T-dependence, rapid evolution
- Red (long λ): Weaker T-dependence, slower evolution

**This creates wavelength-dependent light curve shapes** that **appear** like differential time dilation but are actually thermal physics.

---

## The Critical Complexity: Sequential Processing

### Why Order Matters

**FDR operates on DIMMED flux**:

```python
# v15_model.py:335-337
flux_current = flux_lambda_geometric * jnp.exp(-tau_total)
flux_normalized = jnp.maximum(flux_current / FLUX_REFERENCE, 1e-40)
tau_fdr = xi * eta_prime * jnp.sqrt(flux_normalized)
```

**The flux has ALREADY been modified by**:
1. Planck function B_λ(T) - wavelength-dependent
2. Temperature evolution T(t) - time-dependent
3. Plasma veil scattering - wavelength-dependent via β
4. Photospheric geometry - radius evolution

**Therefore**: FDR is **indirectly wavelength-dependent** even though the formula looks wavelength-independent, because it operates on flux that has wavelength-dependent history.

### Self-Consistent Iterations

**The iterative loop** (`v15_model.py:333-344`):

```python
def body_fn(i, val):
    tau_total, _ = val
    flux_current = flux_lambda_geometric * jnp.exp(-tau_total)  # Current dimmed flux
    flux_normalized = jnp.maximum(flux_current / FLUX_REFERENCE, 1e-40)
    tau_fdr = xi * eta_prime * jnp.sqrt(flux_normalized)  # FDR depends on flux
    tau_new = tau_plasma + tau_fdr
    # Relaxation for stability
    tau_total = 0.5 * tau_new + 0.5 * tau_total
    return tau_total, i

tau_total, _ = jax.lax.fori_loop(0, 20, body_fn, (tau_plasma, 0))
```

**Why iterative?**
- Flux depends on τ_total
- But τ_FDR depends on flux
- **Circular dependency** requires fixed-point iteration
- Converges in ~10-15 iterations with relaxation

---

## Observational Predictions

### Wavelength-Dependent Predictions

| Observable | ΛCDM | QFD |
|------------|------|-----|
| **Color evolution** | Constant (after K-correction) | Time-varying (plasma veil β) |
| **Spectral shape** | Fixed template × (1+z) | Planck B_λ(T) evolution |
| **Light curve stretch** | Parametric (SALT2) | Physical T(t), R(t) |
| **Night-to-night scatter** | Random noise | Systematic (BBH orbital) |
| **Brightness-scatter correlation** | Malmquist bias | FDR (flux-dependent) |

### Time-Series Predictions

**QFD predicts periodic or quasi-periodic variations** in some SNe:

- **Period**: P_orb (days to weeks)
- **Amplitude**: A_lens (up to 0.5 mag)
- **Phase-coherent**: Not random scatter

**Observable**: Multi-epoch photometry should show:
```
μ(MJD) = 1 + A_lens * cos(2π * (MJD - t₀) / P_orb + φ₀)
```

### Statistical Predictions

**QFD expects**:
- **Heavy-tailed residuals** (Student-t with ν ≈ 6-8)
- **~16% outlier fraction** (BBH-dominated)
- **Wavelength-dependent scatter** (more in blue bands)
- **Redshift-dependent evolution** (more scatter at higher z)

**ΛCDM expects**:
- Gaussian residuals
- Outliers are "bad data" (<5% acceptable)
- Wavelength-independent scatter (after K-correction)
- Constant scatter with z

---

## Summary: The Confluence Creates Appearance

### What ΛCDM Attributes to (1+z) Time Dilation

**In QFD, these are distinct physical effects**:

1. **Light curves stretched** → Temperature cooling + plasma expansion (not time dilation)
2. **Spectral features redshifted** → Wien peak shift + plasma veil (not Doppler)
3. **Dimming vs distance** → Photon-field drag + FDR + BBH (not expansion)
4. **Color evolution** → Planck B_λ(T) + veil β-dependence (not K-correction)
5. **Scatter and "outliers"** → BBH lensing variations (not systematics)

### The Power of the Confluence

**By treating these as separate mechanisms**, QFD can:

1. **Explain outlier data** (16-20% that ΛCDM discards)
2. **Predict night-to-night variations** (BBH orbital periods)
3. **Model wavelength-dependent effects** (veil β, Planck B_λ)
4. **Handle flux-dependent scatter** (FDR self-consistency)
5. **Avoid (1+z) assumptions** (no FRW metric required)

### The Falsifiable Difference

**Key Test**: Do "outlier" SNe show **periodic time-series variations**?

- **ΛCDM**: No, they're just bad data (random scatter)
- **QFD**: Yes, with periods P_orb ~ days to weeks (BBH orbits)

**Multi-wavelength Test**: Is scatter **wavelength-dependent**?

- **ΛCDM**: No, after K-correction (universal expansion)
- **QFD**: Yes, β-dependent (plasma veil scattering)

---

## Implementation Summary

### Code Locations

| Component | File | Lines | Function |
|-----------|------|-------|----------|
| **Alpha prediction (3 channels)** | `v15_model.py` | 72-111 | `alpha_pred(z, k_J, eta_prime, xi)` |
| **Planck blackbody** | `v15_model.py` | 198-204 | Inside `spectral_luminosity()` |
| **Temperature evolution** | `v15_model.py` | 130-135 | `_temperature()` |
| **Plasma veil** | `v15_model.py` | 267-291 | `qfd_plasma_redshift_jax()` |
| **FDR iterative solver** | `v15_model.py` | 294-349 | `qfd_tau_total_jax()` |
| **BBH magnification** | `v15_model.py` | 352-394 | `compute_bbh_magnification()` |
| **BBH gravitational z** | `v15_model.py` | 400-443 | `compute_bbh_gravitational_redshift()` |
| **Student-t likelihood** | `stage2_mcmc_numpyro.py` | 491-496 | `model_standardized()` |
| **Photon-photon kernels** | `qfd_cmb/kernels.py` | 11-34 | `sin2_mueller_coeffs()` |
| **E144 power spectra** | `qfd_cmb/ppsi_models.py` | 10-47 | `oscillatory_psik()` |

### Golden Reference Parameters

**From PROGRESS.md:135-141** (November 5, 2024):
```
k_J = 10.770 ± 4.567 km/s/Mpc    (Cosmological drag)
η'  = -7.988 ± 1.439              (Plasma veil coupling)
ξ   = -6.908 ± 3.746              (FDR parameter)
σ_α = 1.398 ± 0.024               (Intrinsic scatter)
ν   = 6.522 ± 0.961               (Student-t degrees of freedom)
```

**The ν ≈ 6.5** confirms **heavy tails** - much heavier than Gaussian (ν → ∞).

---

## Theoretical Foundation

### From QFD Lagrangian

**From QFD_Whitepaper.md:88-89**:
```
L_QFD = -1/4 F_μν F^μν + 1/2 (∂_μ ψ)(∂^μ ψ) - 1/2 m_ψ² ψ² + g ψ F_μν F^μν
```

**The coupling term** `g ψ F_μν F^μν` gives:
- **Cosmological**: Direct photon-ψ interaction (low density, long path)
- **Supernova**: Plasma-mediated photon-electron-ψ (high density, short path)

### E144 Experimental Foundation

**SLAC E144 measured photon-photon scattering** in:
- High-intensity laser fields (gigawatts)
- Ultra-short pulses (zeptoseconds)

**QFD scales this to astrophysics**:
- **Low intensity × long timescales** = cosmological (gigayears)
- **High density × short paths** = supernova plasma (parsecs)

**Relation**: `gigawatts × zeptoseconds = zeptowatts × gigayears`

### No Big Bang Required

**From QFD_Whitepaper.md:66-67**:
> "- No Big Bang signature; CMB arises from ψ equilibrium
> - BAO peak from ψ harmonic length, not early plasma"

**Time is absolute** in QFD - no expansion, no (1+z) dilation. The **appearance** emerges from field interactions.

---

## Conclusions

### The Central Thesis

**Time dilation is not fundamental physics in QFD** - it is an **emergent appearance** from the confluence of:

1. **Thermal physics** (Planck, Wien) - wavelength-dependent
2. **Plasma interactions** (veil β-scattering) - wavelength-dependent
3. **Field damping** (FDR on dimmed flux) - flux-dependent
4. **Gravitational effects** (BBH masses, lensing) - time-varying
5. **Cosmological drag** (photon-field energy transfer) - wavelength-independent

### The Key Advantage

**This complexity allows QFD to explain**:
- The 84.2% of data with chi2/ndof > 2 (ΛCDM struggles)
- The ~16% heavy-tailed "outliers" (ΛCDM discards)
- Night-to-night systematic variations (ΛCDM calls noise)
- Wavelength-dependent scatter (ΛCDM averages out)

**By treating "bad data" as real physics**, QFD achieves better fits and makes testable predictions.

### Falsifiable Predictions

1. **Time-series periodicity** in multi-epoch photometry (BBH orbits)
2. **Wavelength-dependent scatter** increasing with β in blue bands
3. **Flux-dependent residuals** (brighter sources show specific patterns)
4. **Heavy-tailed residuals** (Student-t with ν ≈ 6, not Gaussian)
5. **No (1+z) factors** - can fit data without FRW assumptions

---

## References

### Code Documentation
- `projects/astrophysics/qfd-supernova-v15/src/v15_model.py` - Core physics implementation
- `projects/astrophysics/qfd-supernova-v15/docs/V15_Architecture.md` - Architecture design
- `projects/V16/documents/Supernovae_Pseudocode.md` - Algorithm specifications
- `projects/V16/documents/PROGRESS.md` - Development status and validation
- `projects/astrophysics/redshift-analysis/RedShift/docs/PHYSICS_DISTINCTION.md` - E144 scaling

### Theory Papers
- `Books and Documents/QFD_Whitepaper.md` - Core QFD theory
- `Books and Documents/QFD_Technical_Review_GitHub.md` - Technical details

### Data Analysis
- Golden reference: November 5, 2024 results
- Dataset: DES-SN5YR + Pantheon+ combined (5,468 SNe)
- Quality cut: chi2 < 2000 retains 4,727 SNe (86.5%)
- Outlier fraction: ~16% with Student-t modeling

---

**END OF DOCUMENT**

This document describes the physics as implemented in the QFD codebase as of January 2025. For implementation details, see the referenced source files. For theoretical foundations, see the QFD Book series (versions 6.2-7.0).
