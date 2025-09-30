# Getting Started with QFD Supernova Analysis

This guide will help you get up and running with the QFD Supernova Analysis Package.

## 🚀 Quick Start (5 minutes)

### 1. Installation
```bash
git clone [your-repository-url]
cd QFD_Supernova_Analysis_Package
pip install -r requirements.txt
```

### 2. Test Installation
```bash
python tests/test_basic_functionality.py
```

### 3. Run Basic Example
```bash
python examples/basic_cosmology_fit.py
```

## 📚 Understanding QFD Theory

### The Three Stages of QFD
QFD (Quantum Field Dynamics) modifies standard cosmology through three physical mechanisms:

1. **Stage 1: Plasma Veil** 🌫️
   - **Effect**: Time and wavelength-dependent redshift near supernovae
   - **Formula**: `z_plasma(t,λ) = A_plasma * (1-e^(-t/τ)) * (λ_B/λ)^β`
   - **Observables**: Individual light curve structure

2. **Stage 2: Field Depletion Region (FDR)** 📉
   - **Effect**: Distance-dependent dimming from quantum field depletion
   - **Formula**: `z_FDR ∝ 1/D + 1/D³`
   - **Observables**: Systematic magnitude offsets

3. **Stage 3: Cosmological Drag** 🌌
   - **Effect**: Modified expansion history from drag interactions
   - **Formula**: Modified Friedmann equation with drag terms
   - **Observables**: Overall Hubble diagram shape

## 🔬 Analysis Philosophy

### Two-Script Synergistic Approach

**The Big Picture**: We use a novel two-script approach that tests QFD physics across different observational domains:

```
Script 1: COSMIC CANVAS          Script 2: INDIVIDUAL PORTRAITS
│                                │
├─ Union2.1 distance data        ├─ Raw light curve data
├─ Tests Stages 2+3 (FDR+drag)   ├─ Tests Stage 1 (plasma veil)
├─ Outputs: η', ξ, H₀            ├─ Inputs: FIXED η', ξ, H₀
└─ Result: Background cosmology  └─ Result: Per-SN plasma params
```

**Why This Works**:
- **Cosmic Canvas** determines the universe's background properties
- **Individual Portraits** test time-flux physics with fixed background
- **Cross-correlation** provides the smoking gun evidence

## 🎯 Scientific Workflow

### Step 1: Model Comparison (μ-z Domain)
Compare QFD and ΛCDM on distance-redshift data:

```bash
python src/compare_qfd_lcdm_mu_z.py \
    --data data/union2.1_data_with_errors.txt \
    --out results/model_comparison.json
```

**Expected Result**: ΛCDM preferred (ΔAIC ~+300)
**Interpretation**: QFD geometry indistinguishable from ΛCDM

### Step 2: Background Cosmology (QFD Parameters)
Determine QFD cosmological parameters:

```bash
python src/QFD_Cosmology_Fitter_v5.6.py \
    --walkers 32 --steps 3000 \
    --outdir results/cosmology
```

**Output**: Best-fit η' and ξ values for Stages 2+3
**Use**: Fixed parameters for light curve analysis

### Step 3: Light Curve Physics (Plasma Veil)
Test QFD Stage 1 mechanism:

```bash
python src/qfd_plasma_veil_fitter.py \
    --data data/sample_lightcurves/lightcurves_osc.csv \
    --snid SN2011fe \
    --cosmology results/cosmology/best_fit_params.json \
    --outdir results/plasma_analysis
```

**Output**: Plasma parameters (A_plasma, τ_decay, β) per supernova
**Test**: Do QFD predictions match time-wavelength structure?

### Step 4: Smoking Gun Analysis 🔫
Cross-correlate plasma strength with Hubble residuals:

- **Prediction**: Stronger plasma → larger distance residuals
- **Evidence**: Significant correlation unique to QFD
- **Framework**: Ready for multi-SN deployment

## 📊 Interpreting Results

### Model Comparison Metrics

**AIC/BIC Differences**:
- `|Δ| < 2`: Models equivalent
- `2 < |Δ| < 10`: Weak/moderate evidence
- `|Δ| > 10`: Strong evidence

**Typical Results**:
- μ-z domain: ΛCDM strongly preferred
- Light curve domain: Mixed results, depends on baseline

### QFD Parameter Ranges

**Cosmological Parameters** (from Union2.1):
- η' ~ 10⁻³ to 10⁻⁴ (FDR strength)
- ξ ~ 0.1 to 10 (drag efficiency)

**Plasma Parameters** (from light curves):
- A_plasma: 0.001 to 0.1 (redshift amplitude)
- τ_decay: 10 to 100 days (buildup timescale)
- β: 0.1 to 4.0 (wavelength dependence)

## 🛠️ Advanced Usage

### Custom Data Analysis

**Add New Survey**:
1. Format data as CSV with columns: `snid, mjd, band, mag, mag_err`
2. Update wavelength mapping in `qfd_ingest_lightcurves.py`
3. Run standard analysis pipeline

**Modify QFD Model**:
1. Edit physics functions in plasma veil fitter
2. Update parameter bounds for new physics
3. Rerun analysis with modified model

### Performance Optimization

**For Large Datasets**:
- Use `--walkers 64 --steps 5000` for better MCMC sampling
- Enable parallel processing where available
- Consider data quality cuts for noisy light curves

**For Quick Testing**:
- Use `--walkers 8 --steps 100` for rapid prototyping
- Limit to well-sampled supernovae (>1000 points)
- Use smaller redshift ranges for focused analysis

## 🤔 Troubleshooting

### Common Issues

**"MCMC did not converge"**:
- Increase `--steps` (try 3000-5000)
- Check data quality (remove outliers)
- Verify parameter bounds are reasonable

**"Plasma fit failed"**:
- Check light curve has enough points per band
- Verify wavelength information is available
- Try different initial parameter guesses

**"Import errors"**:
- Ensure all requirements are installed: `pip install -r requirements.txt`
- Check Python version (3.8+ required)
- Verify package paths are correct

### Getting Help

1. **Check Examples**: Review `examples/` directory for working code
2. **Run Tests**: Use `python tests/test_basic_functionality.py`
3. **Documentation**: See `docs/` for detailed methodology
4. **Issues**: Report problems on GitHub repository

## 🎯 Next Steps

### For New Users
1. Run all examples to understand workflow
2. Read methodology documentation
3. Try analysis on your own supernova data
4. Explore parameter sensitivity

### For Researchers
1. Scale up to larger supernova samples
2. Implement smoking gun correlation analysis
3. Compare with other modified gravity theories
4. Develop enhanced light curve templates

### For Developers
1. Add new surveys and data sources
2. Implement advanced statistical methods
3. Optimize numerical performance
4. Contribute new QFD model variants

---

**Ready to explore QFD theory with real data?** Start with the examples and work your way up to the complete analysis workflow!