# Checklist for New AI: Restore V18 MCMC to V21

**Goal**: Add V18's Stage2 MCMC and Stage3 to V21 to achieve RMS=2.18 mag results

---

## Prerequisites Checklist

- [ ] V18 directory accessible at: `/home/tracy/development/SupernovaSrc/qfd-supernova-v15/v15_clean/v18/`
- [ ] V21 directory accessible at: `/home/tracy/development/SupernovaSrc/qfd-supernova-v15/v15_clean/projects/astrophysics/V21 Supernova Analysis package/`
- [ ] V21 Stage1 results exist in: `V21/data/stage2_results_with_redshift.csv` or `V21/results/stage1_output/`
- [ ] Python 3.8+ installed
- [ ] Read `/home/tracy/development/QFD_SpectralGap/V22_Supernova_Analysis/V18_COMPONENTS_MISSING_FROM_V21.md`

---

## Step 1: Copy Required Files ⬜

### Copy Stage2 MCMC
```bash
cp /home/tracy/development/SupernovaSrc/qfd-supernova-v15/v15_clean/v18/pipeline/stages/stage2_mcmc_v18_emcee.py \
   "/home/tracy/development/SupernovaSrc/qfd-supernova-v15/v15_clean/projects/astrophysics/V21 Supernova Analysis package/"
```

- [ ] File copied successfully
- [ ] File is readable
- [ ] File imports check OK

### Copy Stage3 Hubble Diagram
```bash
cp /home/tracy/development/SupernovaSrc/qfd-supernova-v15/v15_clean/v18/pipeline/stages/stage3_v18.py \
   "/home/tracy/development/SupernovaSrc/qfd-supernova-v15/v15_clean/projects/astrophysics/V21 Supernova Analysis package/"
```

- [ ] File copied successfully
- [ ] File is readable
- [ ] File imports check OK

### Optional: Copy pipeline_io if needed
```bash
cp /home/tracy/development/SupernovaSrc/qfd-supernova-v15/v15_clean/v18/core/pipeline_io.py \
   "/home/tracy/development/SupernovaSrc/qfd-supernova-v15/v15_clean/projects/astrophysics/V21 Supernova Analysis package/"
```

- [ ] Check if Stage2/Stage3 need pipeline_io
- [ ] Copy if needed

---

## Step 2: Install Dependencies ⬜

### Check current packages
```bash
cd "/home/tracy/development/SupernovaSrc/qfd-supernova-v15/v15_clean/projects/astrophysics/V21 Supernova Analysis package/"
cat requirements.txt
```

- [ ] requirements.txt reviewed

### Install emcee (CRITICAL)
```bash
pip install emcee
```

- [ ] emcee installed successfully
- [ ] Version: `python -c "import emcee; print(emcee.__version__)"`
- [ ] Test import: `python -c "import emcee; print('OK')"`

### Install corner (optional but useful)
```bash
pip install corner
```

- [ ] corner installed (for MCMC diagnostics)

---

## Step 3: Verify V21 Stage1 Results ⬜

### Check Stage1 output format
```bash
cd "/home/tracy/development/SupernovaSrc/qfd-supernova-v15/v15_clean/projects/astrophysics/V21 Supernova Analysis package/"
head -20 data/stage2_results_with_redshift.csv
```

**Expected columns**: snid, n_obs, chi2_dof, stretch, residual, ln_A, t0, A_plasma, beta, z

- [ ] CSV file exists
- [ ] Has ln_A column
- [ ] Has redshift (z) column
- [ ] ~8,253 rows

### Count SNe
```bash
wc -l data/stage2_results_with_redshift.csv
```

- [ ] Count: _______ SNe (should be ~8,253)

---

## Step 4: Run Stage2 MCMC ⬜

### Prepare output directory
```bash
mkdir -p results/stage2_mcmc_v18
```

- [ ] Directory created

### Run Stage2 (This will take time!)
```bash
cd "/home/tracy/development/SupernovaSrc/qfd-supernova-v15/v15_clean/projects/astrophysics/V21 Supernova Analysis package/"

python3 stage2_mcmc_v18_emcee.py \
  --stage1-results data/stage2_results_with_redshift.csv \
  --lightcurves data/lightcurves_all_transients.csv \
  --out results/stage2_mcmc_v18 \
  --use-ln-a-space \
  --constrain-signs informed
```

**Note**: This may take 30 minutes to several hours depending on data size.

- [ ] Script started without errors
- [ ] MCMC is running (check progress output)
- [ ] Script completed successfully
- [ ] Output files created in `results/stage2_mcmc_v18/`

### Verify Stage2 outputs
```bash
ls -la results/stage2_mcmc_v18/
cat results/stage2_mcmc_v18/summary.json
```

**Expected files**:
- `summary.json` - Best-fit parameters
- `samples.npz` or `stage2_samples.json` - MCMC chain samples

- [ ] summary.json exists
- [ ] Contains k_J_correction, eta_prime, xi
- [ ] Values reasonable: k_J_correction ≈ 20, eta_prime ≈ -6, xi ≈ -6

---

## Step 5: Run Stage3 Hubble Diagram ⬜

### Check Stage1 JSON format (if Stage3 needs it)
```bash
ls results/stage1_output/*.json | head -5
cat results/stage1_output/1246274.json
```

**If Stage1 JSON results don't exist**, may need to:
- [ ] Option A: Regenerate Stage1 with V18's code
- [ ] Option B: Adapt Stage3 to use CSV format

### Run Stage3
```bash
python3 stage3_v18.py \
  --stage1-results data/stage2_results_with_redshift.csv \
  --stage2-results results/stage2_mcmc_v18 \
  --lightcurves data/lightcurves_all_transients.csv \
  --out results/stage3_hubble \
  --quality-cut 2000
```

- [ ] Script started
- [ ] Script completed
- [ ] Output files created

### Verify Stage3 outputs
```bash
ls -la results/stage3_hubble/
cat results/stage3_hubble/summary.json
head -20 results/stage3_hubble/hubble_data.csv
```

**Expected files**:
- `summary.json` - Statistics including RMS
- `hubble_data.csv` - Distance moduli for all SNe

- [ ] summary.json exists
- [ ] RMS value: _______ mag (should be ~2.18)
- [ ] N_SNe: _______ (should be ~4,885)
- [ ] hubble_data.csv has proper format

---

## Step 6: Validate Results ⬜

### Compare to V18 benchmark
```bash
# V18 reference
cat /home/tracy/development/SupernovaSrc/qfd-supernova-v15/v15_clean/v18/results/stage2_fullscale_5468sne/summary.json

# V21 new results
cat results/stage2_mcmc_v18/summary.json
```

**V18 Benchmark**:
- k_J_correction: 19.96 ± 0.06
- eta_prime: -5.999 ± 0.002
- xi: -5.998 ± 0.003

**V21 Results**:
- k_J_correction: _______ ± _______
- eta_prime: _______ ± _______
- xi: _______ ± _______

- [ ] Parameters match V18 within ~3σ
- [ ] RMS ≈ 2.18 mag
- [ ] ~4,885 SNe used

### Check for red flags
- [ ] ❌ RMS > 5 mag? (PROBLEM)
- [ ] ❌ Parameters at bounds? (PROBLEM)
- [ ] ❌ < 1000 SNe? (PROBLEM)
- [ ] ❌ Non-convergence? (PROBLEM)

If any red flags, see Troubleshooting section in V18_COMPONENTS_MISSING_FROM_V21.md

---

## Step 7: Document Results ⬜

### Create validation report
```bash
cd "/home/tracy/development/SupernovaSrc/qfd-supernova-v15/v15_clean/projects/astrophysics/V21 Supernova Analysis package/"

# Create summary
cat > V21_V18_INTEGRATION_RESULTS.md << 'EOF'
# V21 + V18 MCMC Integration Results

**Date**: $(date)
**Status**: [SUCCESS / NEEDS WORK]

## Stage2 MCMC Results
- k_J_correction: _______
- eta_prime: _______
- xi: _______

## Stage3 Hubble Results
- N_SNe: _______
- RMS: _______ mag
- Comparison to V18: [MATCH / DIFFER]

## Files Created
- results/stage2_mcmc_v18/summary.json
- results/stage3_hubble/summary.json
- results/stage3_hubble/hubble_data.csv

## Next Steps
[If successful]: Ready for V22 Lean constraint validation
[If issues]: See troubleshooting notes below

EOF
```

- [ ] Report created
- [ ] Results documented

---

## Step 8: Prepare for V22 ⬜

### If results match V18:
- [ ] Copy hubble_data.csv to V22 data directory
- [ ] Update V22 to use these validated distance moduli
- [ ] Run V22 Lean constraint validation

### File to use in V22:
```bash
cp results/stage3_hubble/hubble_data.csv \
   /home/tracy/development/QFD_SpectralGap/data/raw/des5yr_v21_with_v18_mcmc.csv
```

- [ ] Data copied for V22 use

---

## Success Criteria Summary

✅ **COMPLETE** if all true:
1. Stage2 runs without errors
2. k_J_correction ≈ 20 ± 1
3. eta_prime ≈ -6 ± 1
4. xi ≈ -6 ± 1
5. RMS ≈ 2.18 ± 0.3 mag
6. N_SNe ≈ 4,885 ± 500

⚠️ **NEEDS WORK** if any:
1. Import errors
2. MCMC doesn't converge
3. Parameters at bounds
4. RMS >> 2.5 mag
5. Very few SNe (<1000)

---

## Common Issues & Solutions

### Issue: "No module named emcee"
**Solution**: `pip install emcee`

### Issue: "Cannot find stage1 results"
**Solution**: Check path, may need to point to CSV instead of JSON directory

### Issue: "Parameters hit bounds"
**Solution**: Check data format, signs in ln_A conversion, quality cuts

### Issue: "Poor fit (RMS >> 2.5)"
**Solution**: Verify using raw data (not SALT), check Stage1 quality

---

**Detailed guide**: See `V18_COMPONENTS_MISSING_FROM_V21.md`

**Status**: ⬜ Not Started | ⏳ In Progress | ✅ Complete
