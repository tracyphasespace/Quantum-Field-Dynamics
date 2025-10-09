# Quick Start Guide

Get running with NuclideModel in 5 minutes.

---

## 1. Install Dependencies

```bash
pip install torch numpy scipy pandas
```

Or use the requirements file:
```bash
pip install -r requirements.txt
```

---

## 2. Run Your First Calculation

Compute He-4 (helium-4, a doubly-magic nucleus):

```bash
cd examples
./run_he4.sh
```

**Expected output**:
```
E_model: -23.96 MeV
virial_abs: 0.024
physical_success: true
```

This means:
- Binding energy ≈ 24 MeV (correct!)
- Virial converged (< 0.03)
- Physical solution found

---

## 3. Try a Heavy Nucleus

Compute Pb-208 (lead-208, heaviest stable doubly-magic):

```bash
./run_pb208.sh
```

**Expected output**:
```
E_model: -17898 MeV
virial_abs: 0.098
physical_success: true
```

**Note**: Pb-208 shows -8.4% error (systematic heavy isotope underbinding).

---

## 4. Run Your Own Nucleus

```bash
python ../src/qfd_solver.py \
  --A 16 --Z 8 \
  --c-v2-base 2.201711 \
  --c-v2-iso 0.027035 \
  --c-v2-mass -0.000205 \
  --c-v4-base 5.282364 \
  --c-v4-size -0.085018 \
  --alpha-e-scale 1.007419 \
  --beta-e-scale 0.504312 \
  --c-sym 25.0 \
  --kappa-rho 0.029816 \
  --grid-points 48 \
  --iters-outer 360 \
  --emit-json \
  --out-json my_result.json
```

Change `--A` (mass number) and `--Z` (atomic number) for different nuclei.

---

## 5. Understanding Output

The JSON output contains:

```json
{
  "status": "ok",                    // Solver succeeded
  "A": 16,                          // Mass number
  "Z": 8,                           // Atomic number
  "E_model": -58.64,                // Interaction energy (MeV)
  "virial": 0.113,                  // Virial (should be < 0.18)
  "virial_abs": 0.113,
  "physical_success": true,         // Converged + virial OK
  "alpha_eff": 2.25,                // Effective cohesion
  "beta_eff": 5.07,                 // Effective repulsion
  "T_N": 123.45,                    // Nuclear kinetic energy
  "V4_N": -234.56,                  // Nuclear cohesion
  "V_coul_cross": 78.90,            // Coulomb energy
  "V_sym": 0.0,                     // Symmetry energy (N=Z)
  ...
}
```

**Key fields**:
- `E_model`: Nuclear interaction energy (negative = bound)
- `virial_abs`: Convergence quality (< 0.18 = good)
- `physical_success`: Overall success flag

---

## 6. Compare to Experiment

```python
import json

# Load result
with open('my_result.json') as f:
    data = json.load(f)

# Extract interaction energy
E_model = data['E_model']
A, Z = data['A'], data['Z']
N = A - Z

# Add rest masses
M_proton = 938.272088
M_neutron = 939.565420
M_electron = 0.51099895
M_constituents = Z*M_proton + N*M_neutron + Z*M_electron

# Total QFD prediction
E_total_QFD = M_constituents + E_model

# Compare to AME2020 experimental value
# (Look up E_exp from data/ame2020_system_energies.csv)
E_exp = ...  # From database

rel_error = (E_total_QFD - E_exp) / E_exp
print(f"Error: {rel_error*100:.2f}%")
```

---

## 7. Recalibrate Parameters

Run the meta-optimizer to find your own parameters:

```bash
cd ../src
python qfd_metaopt_ame2020.py \
  --n-calibration 30 \
  --max-iter 100 \
  --out-json my_params.json
```

This will:
1. Select 30 physics-driven calibration isotopes
2. Optimize 9 parameters using differential evolution
3. Save results to `my_params.json`

**Warning**: Takes ~10-60 minutes depending on CPU.

---

## 8. Common Issues

### Issue: "RuntimeError: CUDA out of memory"

**Solution**: Use CPU instead
```bash
python qfd_solver.py ... --device cpu
```

### Issue: "physical_success: false"

**Causes**:
- Virial didn't converge (|virial| > 0.30)
- Increase `--iters-outer` (try 500-700)
- Adjust `--grid-points` (try 32, 48, 64)

### Issue: Large errors (> 10%)

**Causes**:
- Heavy isotopes (A > 120) have systematic underbinding
- Very light/asymmetric isotopes (Li, Be) may not converge
- Extreme proton-rich/neutron-rich nuclei off valley of stability

---

## 9. Next Steps

- Read `docs/PHYSICS_MODEL.md` for field theory background
- Check `docs/FINDINGS.md` for performance analysis
- See `docs/CALIBRATION_GUIDE.md` for detailed parameter tuning
- Explore `examples/` for more test cases

---

## 10. Getting Help

- **Issues**: https://github.com/yourusername/NuclideModel/issues
- **Discussions**: https://github.com/yourusername/NuclideModel/discussions
- **Documentation**: `docs/` directory

---

## Performance Expectations

| Mass Range | Typical Error | Example Nuclei |
|------------|---------------|----------------|
| A < 60     | < 1%          | He-4, C-12, O-16, Ca-40 |
| 60 ≤ A < 120 | 2-5%        | Fe-56, Ni-62, Sn-100 |
| A ≥ 120    | 7-9%          | Pb-208, Au-197, U-238 |

**Magic numbers** (Z or N = 2, 8, 20, 28, 50, 82) typically show best performance.

---

Happy computing! 🚀
