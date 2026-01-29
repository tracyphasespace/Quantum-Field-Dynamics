# ΛCDM vs QFD: Distinguishing Tests
**Date:** January 16, 2025
**Analysis:** How to observationally distinguish expanding vs static universe

---

## 🎯 **The Central Question**

**ΛCDM Framework:**
- Universe expanding at H₀ = 70 km/s/Mpc
- Dark energy (Ω_Λ ≈ 0.7) drives acceleration
- Time dilation: (1+z) factor for all processes
- SNe appear dimmer due to: luminosity distance, redshift, time stretching

**QFD Framework:**
- Universe STATIC (Minkowski spacetime)
- Photon energy loss via quantum field drag (k_J = 70 km/s/Mpc)
- NO time dilation - intrinsic rates at all z
- SNe appear dimmer due to: photon aging, FDR opacity

**Both achieve similar fit quality (σ ~ 0.9 mag vs 0.8 mag)** - but completely different physics!

---

## 📊 **Current Results: Fit Quality vs Redshift**

### Key Finding: QFD Fits HIGH-z SNe BETTER!

**Normalized fit quality (neg_logL per observation):**

| Redshift Range | QFD Fit Quality | Relative Performance |
|---------------|----------------|---------------------|
| Low-z (0.0-0.3) | 4.285 ± 2.432 | Baseline (worst) |
| Mid-z (0.6-0.8) | 2.195 ± 0.906 | **2.0× better** |
| High-z (1.0-1.5) | 2.337 ± 1.163 | **1.8× better** |

**Statistical Significance:**
- Correlation: r = -0.40
- P-value: p < 10⁻²¹⁰
- Sample: 5367 SNe (standard population only)

### Interpretation:

**If time dilation were MISSING from QFD model:**
- We'd expect POSITIVE correlation (worse fits at high-z)
- Light curve template would be too narrow at z~1
- Should see systematic residuals increasing with z

**What we actually observe:**
- **Strong NEGATIVE correlation** (r = -0.40)
- High-z SNe fit **twice as well** per observation
- Model performance **improves** where time dilation should be strongest

**Possible Explanations:**

1. **Selection Bias**
   - High-z: Only cleanest, most standardizable SNe detected
   - Low-z: Includes heterogeneous nearby samples with environmental contamination
   - Homogeneity dominates over time dilation systematics

2. **QFD Physics Compensates**
   - FDR frequency-dependent opacity (eta_prime = -6.0) mimics time dilation effects
   - Photon aging changes observed spectral shape in z-dependent way
   - Accidental degeneracy between expansion effects and photon drag

3. **Published Time Dilation Results Need Re-examination**
   - Previous analyses assumed ΛCDM framework
   - May have attributed intrinsic variations to cosmological (1+z) stretch
   - QFD corrections might explain width variations without expansion

---

## 🧪 **Definitive Observational Tests**

### **Test #1: Light Curve Width/Stretch Parameter** ⭐ CRITICAL

**Method:** Add time-axis stretch parameter `s` to lightcurve fits

**ΛCDM Prediction:**
```python
# Rest-frame time with expansion time dilation
t_rest = (t_obs - t0) / (1 + z)
flux = template(t_rest / s)

# Expected: s = constant (intrinsic stretch only)
# Observed width ∝ (1+z) comes from coordinate transformation
```

**QFD Prediction:**
```python
# Observer-frame time (no cosmological dilation)
t_obs_frame = t_obs - t0
flux = template(t_obs_frame / s)

# Expected: s = constant at all z
# If time dilation exists, we'd need s ∝ (1+z) to compensate
```

**Test Analysis:**
```python
# Fit all 5468 SNe with free stretch parameter
# Then check correlation:
correlation = pearsonr(stretch_values, 1 + z_values)

if correlation.r > 0.7 and correlation.p < 0.001:
    # Time dilation observed
    # → ΛCDM supported
    # → QFD must explain why it still fits well via other mechanisms
else:
    # No time dilation
    # → QFD supported
    # → Published width measurements need reinterpretation
```

**Literature Status:**
- Goldhaber et al. (2001): Measured width ∝ (1+z) in SN light curves
- Blondin et al. (2008): Confirmed time dilation in multi-band photometry
- **Considered definitive proof of cosmological expansion**

**Critical Question:**
Can QFD framework explain these observations without invoking expansion?

---

### **Test #2: Tolman Surface Brightness**

**ΛCDM Prediction:**
Extended sources (galaxies) dim as **(1+z)⁴**:
- (1+z)² from photon energy redshift
- (1+z)² from angular area expansion

**QFD Prediction:**
Only photon energy loss (no geometric expansion):
- Different functional form
- Depends on k_J and FDR parameters

**Observational Test:**
Measure surface brightness μ (mag/arcsec²) vs redshift for standard galaxies

**Status:**
- Some measurements exist (Lerner 2018, Lubin & Sandage 2001)
- Results controversial, claimed to favor static universe
- Need reanalysis with modern data

---

### **Test #3: CMB Temperature Evolution**

**ΛCDM Prediction:**
CMB is cosmological blackbody cooling with expansion:
- T_CMB(z) = T₀ × (1+z)
- At z=2.5: T_CMB ≈ 9.6 K (vs 2.73 K today)

**QFD Prediction:**
CMB NOT cosmological - different origin required:
- Temperature should NOT scale as (1+z)
- Need alternative explanation (e.g., thermalized starlight, local processes)

**Observational Test:**
Measure CMB temperature at high-z via:
- CO molecular line excitation temperatures
- Sunyaev-Zel'dovich effect
- Direct spectroscopy

**Status:**
- Measurements confirm T ∝ (1+z) up to z~3 (Fixsen 2009, Noterdaeme+ 2011)
- **Strong evidence for ΛCDM**
- QFD needs compelling alternative explanation

---

### **Test #4: Angular Diameter Distance**

**ΛCDM Prediction:**
Non-Euclidean relation due to expansion:
- Angular size θ vs z has turnover (objects appear LARGER at very high-z)
- Maximum at z ~ 1.6 for standard cosmology

**QFD Prediction:**
Euclidean geometry (static universe):
- θ ∝ 1/z monotonically
- No turnover

**Observational Test:**
Measure angular sizes of standard rulers (e.g., BAO, galaxy clusters) vs z

---

### **Test #5: Number Counts**

**ΛCDM Prediction:**
Volume element dV/dz deviates from Euclidean:
- N(m) ≠ 10^(0.6m) at high-z

**QFD Prediction:**
Euclidean space:
- N(m) ∝ 10^(0.6m) at all magnitudes (with photon aging corrections)

---

## 🔬 **Implementation: Adding Stretch to Stage 1**

### Current Stage 1 Parameters (V17):
```python
per_sn_params = (t0, ln_A, A_plasma, beta, L_peak)
# t0: explosion time
# ln_A: log amplitude (brightness)
# A_plasma, beta: frozen at (0.1, 1.5)
# L_peak: fixed at 1e43 erg/s
```

### Modified Stage 1 with Stretch (V18):
```python
per_sn_params = (t0, ln_A, stretch, A_plasma, beta, L_peak)
# NEW: stretch parameter (s)
#   s = 1.0 → intrinsic template width
#   s > 1.0 → broader light curve
#   s < 1.0 → narrower light curve
```

### Model Modification:
```python
def qfd_lightcurve_model_with_stretch(obs_data, global_params, persn_params, z):
    """
    obs_data: [mjd, wavelength]
    persn_params: (t0, ln_A, stretch, A_plasma, beta, L_peak)
    """
    t_obs, wavelength = obs_data
    t0, ln_A, stretch, A_plasma, beta, L_peak = persn_params

    # Apply stretch to time axis
    t_scaled = (t_obs - t0) / stretch  # ← KEY CHANGE

    # Rest of model unchanged
    phase = t_scaled / 20.0  # Normalize to typical SN timescale
    template_value = template_function(phase, wavelength)

    # Apply QFD corrections (FDR, plasma veil, distance)
    flux = template_value * jnp.exp(ln_A) * opacity_corrections(...)

    return flux
```

### Optimization Bounds:
```python
stretch_bounds = (0.5, 2.0)  # Allow 2× variation in width
# s=0.5: Fast/narrow light curve (rare)
# s=1.0: Standard template width
# s=2.0: Slow/broad light curve (rare)
```

### Analysis After Fitting:
```python
# Load all results with stretch
stretches = [r['best_fit_params']['stretch'] for r in results]
redshifts = [r['z'] for r in results]

# Test correlation
corr, pval = pearsonr(stretches, 1 + np.array(redshifts))

print(f"Stretch vs (1+z) correlation: r={corr:.3f}, p={pval:.2e}")

if corr > 0.7:
    print("Time dilation detected!")
    print("  → Stretch increases with redshift as (1+z)")
    print("  → Supports ΛCDM expansion")
else:
    print("No time dilation!")
    print("  → Stretch independent of redshift")
    print("  → Supports QFD static universe")
```

---

## 📈 **Expected Outcomes**

### Scenario A: Time Dilation EXISTS (ΛCDM Correct)

**Result:** stretch ∝ (1+z) with r > 0.9

**Implications for QFD:**
- QFD model fits well despite missing explicit (1+z) factor
- Must investigate how FDR/plasma effects mimic expansion
- Possible degeneracy: photon aging ↔ time dilation
- QFD would need to incorporate time dilation or explain mimicry

**Next Steps:**
- Compare QFD+stretch vs ΛCDM fits
- Understand physical degeneracies
- Test if FDR alone can explain width variations

---

### Scenario B: Time Dilation ABSENT (QFD Correct)

**Result:** stretch = constant, no correlation with z

**Implications for ΛCDM:**
- Published time dilation measurements need reinterpretation
- Width variations may be intrinsic, selection effects, or QFD physics
- Revolutionary challenge to expansion paradigm

**Next Steps:**
- Re-analyze published datasets with QFD framework
- Explain CMB temperature evolution without expansion
- Develop full QFD cosmology (structure formation, BBN, etc.)

---

### Scenario C: Complex Behavior (Both Models Incomplete)

**Result:** Partial correlation, or redshift-dependent scatter

**Implications:**
- Neither pure ΛCDM nor pure QFD fully explains observations
- Hybrid model may be needed
- Population evolution, selection biases, or new physics

**Next Steps:**
- Investigate population-dependent effects
- Search for additional parameters (luminosity evolution, environment)
- Consider whether both expansion AND photon drag occur

---

## 🎓 **Scientific Context**

### Why This Matters:

**If ΛCDM is correct:**
- Dark energy is real (~68% of universe)
- Universe had a beginning (Big Bang)
- Finite age (~13.8 Gyr)
- Accelerating expansion continues

**If QFD is correct:**
- No dark energy needed
- Universe potentially eternal (no Big Bang required)
- Static spacetime
- New quantum field theory needed for photon drag

**Current Status:**
- Both frameworks fit SN brightness-distance relation
- Both have σ ~ 0.8-0.9 mag residual scatter
- Time dilation test is CRITICAL distinguisher
- CMB temperature test is CRITICAL confirmation

---

## 📚 **References**

### Time Dilation Measurements:
- Goldhaber et al. (2001), ApJ, 558, 359: "Timescale Stretch Parameterization of Type Ia Supernova B-Band Light Curves"
- Blondin et al. (2008), ApJ, 682, 724: "Time Dilation in Type Ia Supernova Spectra at High Redshift"

### CMB Temperature:
- Fixsen et al. (2009), ApJ, 707, 916: "The Temperature of the Cosmic Microwave Background"
- Noterdaeme et al. (2011), A&A, 526, L7: "The evolution of the cosmic microwave background temperature"

### Surface Brightness:
- Lubin & Sandage (2001), AJ, 122, 1084: "The Tolman Surface Brightness Test for the Reality of the Expansion"
- Lerner (2018), MNRAS, 477, 3185: "Observations contradict galaxy size and surface brightness predictions"

### Supernova Cosmology:
- Riess et al. (1998), AJ, 116, 1009: "Observational Evidence from Supernovae for an Accelerating Universe"
- Perlmutter et al. (1999), ApJ, 517, 565: "Measurements of Omega and Lambda from 42 High-Redshift Supernovae"

---

## ✅ **Action Items**

### Immediate (Week 1):
- [x] Document ΛCDM vs QFD distinguishing tests
- [ ] Add stretch parameter to V18 lightcurve model
- [ ] Update Stage 1 optimizer for stretch fitting
- [ ] Test on sample of 100 SNe

### Short-term (Week 2):
- [ ] Run full DES dataset with stretch (5468 SNe)
- [ ] Analyze stretch vs (1+z) correlation
- [ ] Create diagnostic plots (stretch distribution, stretch vs z)
- [ ] Compare fit quality with/without stretch

### Medium-term (Month 1):
- [ ] Re-run outliers with free stretch
- [ ] Test BBH lensing + stretch combined model
- [ ] Implement plasma veil + stretch
- [ ] Publication-quality figures

### Long-term (Months 2-3):
- [ ] Compare to published time dilation measurements
- [ ] Analyze by photometric band (time dilation should be achromatic)
- [ ] Test alternative stretch parameterizations
- [ ] Write manuscript: "Testing Time Dilation in SNe Ia with QFD Framework"

---

**Status:** Test #1 (stretch parameter) implementation in progress
