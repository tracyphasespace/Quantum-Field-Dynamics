# QFD Quick Start Guide (Genesis Constants v3.2)

## 🚀 Get Started in 30 Seconds

### 1. Validate Genesis Constants
```bash
python test_genesis_constants.py
```
Expected: Virial ~0.047, Physical Success ✅

### 2. Run Basic Deuterium Simulation
```bash
python run_target_deuterium.py
```
Generates: JSON + Markdown + CSV summaries + .pt state file

### 3. View Results
```bash
# Quick JSON view
python -m json.tool runs_D128/D128_a4_g6_lin.json

# Human-readable summary
cat runs_D128/D128_a4_g6_lin_summary.md
```

## 🎯 Genesis Constants (Locked In)

- **α = 4.0** (electrostatic coupling)
- **γₑ = 6.0** (electron quartic coupling)
- **Virial = 0.0472** (excellent stability)
- **Regime**: "Gentle Equilibrium"

## 📊 Common Use Cases

### Test Different Isotopes
```bash
python run_target_deuterium.py --mass 1.0 --outfile hydrogen.json
python run_target_deuterium.py --mass 2.0 --outfile deuterium.json
python run_target_deuterium.py --mass 3.0 --outfile tritium.json
```

### Test Mass Scaling Strategies
```bash
python run_target_deuterium.py --dilate-exp 1.0 --outfile linear.json
python run_target_deuterium.py --dilate-exp 2.0 --outfile quadratic.json
```

### High-Resolution Studies
```bash
python run_target_deuterium.py --grid 160 --iters 1500 --outfile hires.json
```

## 📁 Key Files

### Production Ready ✅
- `Deuterium.py` - Main solver
- `run_target_deuterium.py` - Convenience wrapper
- `test_genesis_constants.py` - Validation

### Analysis Tools
- `visualize.py` - 3D field plots from .pt files
- `calibrate_from_state.py` - Scale calibration
- `polish_from_state.py` - Solution refinement

### Legacy (Development)
- `AutopilotHydrogen.py` - Parameter exploration
- `AllNightLong.py` - Extended sweeps
- `smoketest_*.py` - Various validation scenarios

## 🔍 Success Criteria

### Physical Success (Primary)
- **Virial residual < 0.1** ✅
- **Penalties < 1e-5** ✅

### Convergence (Secondary)
- Energy stabilization (when achievable in flat landscape)

## 🆘 Troubleshooting

### "FAIL" Status but Good Physics
- **Normal in Genesis Constants regime** (flat energy landscape)
- **Check virial and penalties** instead of convergence flag
- **Physical success is what matters**

### High Virial Residual
- Try longer runs: `--iters 1500`
- Check parameters are near Genesis Constants
- Verify grid resolution: `--grid 128` or higher

### NaN/Inf Errors
- Reduce learning rates in Deuterium.py
- Check spectral cutoff settings
- Verify mass/charge parameters are reasonable

---

**Ready to explore? Start with `python test_genesis_constants.py`**