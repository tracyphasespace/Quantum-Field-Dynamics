# Deployment Checklist

## Pre-Deployment Verification

### 1. Code Quality
- [x] All Python files have docstrings
- [x] No debug print statements in production code
- [x] No hardcoded file paths (all relative)
- [x] Error handling implemented
- [x] Type hints where appropriate

### 2. Documentation
- [x] README.md complete and professional
- [x] QUICKSTART.md tested and accurate
- [x] THEORY.md scientifically rigorous
- [x] METHODOLOGY.md technically complete
- [x] RESULTS.md data-driven and objective
- [x] No sensationalistic language
- [x] All claims cited or caveated

### 3. Reproducibility
- [x] requirements.txt includes all dependencies
- [x] Example results provided for comparison
- [x] Installation verified
- [x] Scripts executable (chmod +x)
- [ ] Tested on clean system (recommended)

### 4. Legal & Attribution
- [x] LICENSE file (MIT)
- [x] CITATION.cff for academic use
- [x] Proper citations in documentation
- [x] Acknowledgments included

### 5. Version Control
- [x] .gitignore excludes generated files
- [x] No large binary files (except example plots)
- [x] Repository structure clean

## Deployment Steps

### Step 1: Final Review
```bash
cd lepton-mass-spectrum

# Check all files present
ls -la
ls -la src/
ls -la scripts/
ls -la docs/
ls -la data/
ls -la results/

# Verify no __pycache__ or generated files
find . -name "__pycache__" -o -name "*.pyc"
# Should be empty
```

### Step 2: Test Installation
```bash
# Create clean virtual environment
python -m venv test_env
source test_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python scripts/verify_installation.py
# Should show all checks passed

# Run quick test (reduce n_steps for speed)
# Edit run_mcmc.py temporarily: n_steps=100
python scripts/run_mcmc.py

# Check outputs
ls -la results/
# Should have: mcmc_chain.h5, results.json, corner_plot.png

# Clean up
deactivate
rm -rf test_env
```

### Step 3: Copy to GitHub Location
```bash
# Navigate to QFD repo
cd ~/development/QFD_SpectralGap

# Ensure GitHub repo is up to date
cd path/to/Quantum-Field-Dynamics
git pull

# Copy lepton-mass-spectrum
cp -r ~/development/QFD_SpectralGap/projects/particle-physics/lepton-mass-spectrum \
      ./projects/particle-physics/

# Verify copy
ls -la projects/particle-physics/lepton-mass-spectrum/
```

### Step 4: Git Operations
```bash
cd projects/particle-physics/lepton-mass-spectrum

# Check status
git status

# Stage all files
git add .

# Review what will be committed
git status
git diff --cached

# Commit
git commit -m "Add lepton mass spectrum MCMC analysis

- Complete QFD model for charged lepton masses
- Bayesian parameter estimation (β, ξ, τ)
- Compton-scale Hill vortex implementation
- Full documentation (theory, methodology, results)
- Reproducible MCMC pipeline
- Example results for verification

Addresses: #[issue-number] (if applicable)"

# Push to branch
git push origin main
# Or create feature branch:
# git checkout -b feature/lepton-mass-spectrum
# git push -u origin feature/lepton-mass-spectrum
```

### Step 5: GitHub UI
1. Navigate to: `https://github.com/tracyphasespace/Quantum-Field-Dynamics`
2. Go to `projects/particle-physics/lepton-mass-spectrum`
3. Verify all files visible
4. Check README.md renders correctly
5. Test all internal links in documentation
6. Verify corner plot displays (if included)

### Step 6: Create Release (Optional)
```bash
# Tag version
git tag -a v1.0.0 -m "Initial release: Lepton mass spectrum QFD model"
git push origin v1.0.0
```

On GitHub:
1. Go to Releases → Draft new release
2. Choose tag: v1.0.0
3. Title: "Lepton Mass Spectrum QFD Model v1.0.0"
4. Description:
```
Initial release of the charged lepton mass spectrum model.

**Features**:
- Bayesian MCMC parameter estimation for (β, ξ, τ)
- Compton-scale Hill vortex implementation
- Complete documentation (1500+ lines)
- Reproducible results with example outputs

**Quick Start**:
See QUICKSTART.md for 5-minute installation and 20-minute first run.

**Citation**:
See CITATION.cff for academic citation format.
```

## Post-Deployment

### Announce
- [ ] Update main project README to link to new analysis
- [ ] Add to project index / table of contents
- [ ] Notify collaborators

### Monitor
- [ ] Watch for issues opened
- [ ] Respond to questions within 48 hours
- [ ] Track usage (stars, forks, clones)

### Maintain
- [ ] Address bugs promptly
- [ ] Consider feature requests
- [ ] Update dependencies periodically
- [ ] Add unit tests over time

## Rollback Plan

If critical issues found after deployment:

```bash
# Revert commit
git revert <commit-hash>
git push

# Or delete tag
git tag -d v1.0.0
git push origin :refs/tags/v1.0.0

# Fix issues, then re-deploy
```

## Success Criteria

Repository is successfully deployed when:
- [x] All files visible on GitHub
- [ ] README renders correctly
- [ ] Installation instructions work on clean system
- [ ] MCMC runs and produces expected results
- [ ] No broken links in documentation
- [ ] Example results match fresh runs (within tolerance)
- [ ] Issue tracker enabled and monitored

## Notes

**Target URL**:
`https://github.com/tracyphasespace/Quantum-Field-Dynamics/tree/main/projects/particle-physics/lepton-mass-spectrum`

**Estimated deployment time**: 30-60 minutes (including verification)

**Recommended test platforms**:
- Ubuntu 22.04 LTS
- macOS (latest)
- Windows 11 with WSL2

**Python versions to test**:
- 3.8 (minimum)
- 3.10 (recommended)
- 3.11 (latest stable)
