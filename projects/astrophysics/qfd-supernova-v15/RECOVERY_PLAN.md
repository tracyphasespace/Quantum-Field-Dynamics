# V15 Pipeline Recovery Plan

## Current Situation (2025-11-10)

### What Broke
1. **V15 Production Code**: Cannot reproduce k_J = 10.69 ± 4.57 km/s/Mpc
   - Current code (commit 93dfa1a): k_J collapses to ~0 (NULL result)
   - Reverted code (commit 958f144): k_J stuck at 50.0 ± 0.00007 (prior bound)

2. **Deleted Files**: Backup .zip files from `/home/tracy/development/backups/`

3. **Root Cause**: Today's "fixes" (commits 0f8b3f4-93dfa1a) broke Stage 2 MCMC

### What Still Works
1. **V15 Production Results** (untouched):
   - `/home/tracy/development/Quantum-Field-Dynamics-fresh/projects/astrophysics/qfd-supernova-v15/results/v15_production/stage2/best_fit.json`
   - k_J = 10.69 ± 4.57, η' = -7.97 ± 1.44, ξ = -6.88 ± 3.75

2. **Original Data Files**:
   - `data/lightcurves_unified_v2_min3.csv` (5,468 SNe)
   - All V15 production Stage 1 results (3,238 successful fits)

3. **October_Supernova V15 Code** (POTENTIALLY WORKING):
   - `/home/tracy/development/qfd_hydrogen_project/October_Supernova/GPU_Supernova_tools/V15/`
   - Contains stage2_mcmc_numpyro.py (modified Nov 8 15:40)
   - Has production results in `results/` subdirectories

---

## Recovery Steps

### Option 1: Restore from October_Supernova (RECOMMENDED)

October_Supernova V15 directory contains working code from Nov 8. This is likely the code that produced the V15 production results.

**Steps:**

1. **Compare October_Supernova V15 with Current Code**
   ```bash
   cd /home/tracy/development/Quantum-Field-Dynamics-fresh/projects/astrophysics/qfd-supernova-v15
   diff src/stage2_mcmc_numpyro.py /home/tracy/development/qfd_hydrogen_project/October_Supernova/GPU_Supernova_tools/V15/stage2_mcmc_numpyro.py
   ```

2. **Backup Current Broken Code**
   ```bash
   mkdir -p /home/tracy/development/backups/qfd-supernova-v15-broken-20251110
   cp -r src/ results/ /home/tracy/development/backups/qfd-supernova-v15-broken-20251110/
   ```

3. **Copy Working Files from October_Supernova**
   ```bash
   # Copy all source files
   cp /home/tracy/development/qfd_hydrogen_project/October_Supernova/GPU_Supernova_tools/V15/stage2_mcmc_numpyro.py src/
   cp /home/tracy/development/qfd_hydrogen_project/October_Supernova/GPU_Supernova_tools/V15/v15_*.py src/
   ```

4. **Test with V15 Validation Run**
   ```bash
   export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false"
   python src/stage2_mcmc_numpyro.py \
     --stage1-results results/v15_production/stage1 \
     --lightcurves data/lightcurves_unified_v2_min3.csv \
     --out results/v15_verification \
     --nchains 4 --nsamples 1000 --nwarmup 500
   ```

5. **Expected Result**: k_J ≈ 10.69 ± 4.57 (should match V15 production)

---

### Option 2: WSL2 File Recovery (For Deleted Backups)

**WARNING**: This requires shutting down WSL2 and working with Windows PowerShell.

**Steps:**

1. **Shutdown WSL (from Windows PowerShell)**
   ```powershell
   wsl --shutdown
   ```

2. **Locate WSL2 Virtual Disk**
   ```powershell
   $vhdxPath = Get-ChildItem -Path "$env:LOCALAPPDATA\Packages" -Recurse -Filter "ext4.vhdx" | Select-Object -First 1
   Write-Host "Found vhdx: $($vhdxPath.FullName)"
   ```

3. **Copy vhdx to Backup Location**
   ```powershell
   Copy-Item $vhdxPath.FullName "C:\recovery_backup\ext4_backup.vhdx"
   ```

4. **Mount and Scan for Deleted Files**
   - Use Windows-based ext4 recovery tools:
     - DiskInternals Linux Reader
     - R-Studio for Windows
     - TestDisk (Windows version)

5. **Target Files to Recover**:
   - `/home/tracy/development/backups/*.zip`
   - Any other deleted V15 backup files

---

### Option 3: Git History Search (LAST RESORT)

Search for orphaned commits or reflog entries that might contain working code:

```bash
cd /home/tracy/development/Quantum-Field-Dynamics-fresh/projects/astrophysics/qfd-supernova-v15

# Search all commits (including unreachable)
git fsck --lost-found

# Search reflog for deleted branches
git reflog show --all

# Search commit messages for "production" or "v15"
git log --all --grep="production\|v15" --oneline
```

---

## Critical Files Comparison

### stage2_mcmc_numpyro.py Key Differences

**Current Broken Code** (commit 93dfa1a):
- Has `iters < 5` filter (line 81-83) - rejects fast-converging SNe
- Uses alpha-space likelihood with cache-busting
- Added preflight variance check

**Expected Working Code** (from October_Supernova):
- Unknown differences - needs comparison
- Likely does NOT have `iters < 5` filter
- May use different prior bounds

---

## Validation Checklist

After recovery, verify:
- [ ] Stage 2 MCMC produces k_J ≈ 10.69 ± 4.57
- [ ] No divergent transitions
- [ ] Uses all ~3,200 Stage 1 results (not filtered down to 1,094)
- [ ] R-hat values < 1.01 (good convergence)
- [ ] Effective sample size > 1000

---

## Notes

- **All processes stopped**: No pipeline processes currently running
- **Current branch**: claude/critical-bugfixes-011CUpmVGWvwHfZMWhqw37VM
- **Last working commit**: Unknown (not in current repo history)
- **October_Supernova last modified**: Nov 8 15:40 (5 days ago)

---

## Contact

If recovery fails, escalate to:
1. Check October_Supernova/.claude for previous conversation logs
2. Check git reflog for dropped commits
3. Perform WSL2 vhdx recovery as last resort
