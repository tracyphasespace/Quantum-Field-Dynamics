# Changelog

All notable changes to the V22 Lepton Analysis project will be documented in this file.

## [2.0.0] - 2025-12-23

### Changed - Documentation Rewrite
- **README.md**: Complete rewrite with scientific tone, prominent limitations
- **REPLICATION_GUIDE.md**: Removed promotional language, added caveats throughout
- Replaced "100% accuracy" with actual residuals (5×10⁻¹¹, 6×10⁻⁸, 2×10⁻⁷)
- Stated β from α as "conjectured" (not "derived from first principles")
- Removed comparisons to Einstein/Maxwell
- Deleted all emojis from technical documentation

### Added
- `REPLICATION_ASSESSMENT.md` - Independent verification and peer review assessment
- `RHETORIC_GUIDE.md` - Scientific language guidelines with examples
- `GITHUB_READINESS_CHECKLIST.md` - Document review tool
- `VERSION.md` - Version tracking
- `CHANGELOG.md` - This file
- `LICENSE` - MIT License
- `requirements.txt` - Python dependencies
- `.gitignore` - Standard Python/IDE exclusions
- Prominent limitations sections in all major documents

### Emphasized
- Solution degeneracy (3 geometric DOF → 1 target per lepton)
- Need for independent observable tests (charge radius, g-2, form factors)
- Grid convergence limitations (~1% drift at production resolution)
- U > 1 interpretation issue for tau

### Kept Unchanged
- All Python code (validated, working)
- All numerical results (verified, reproducible)
- Validation test data
- Mathematical formulations
- Hill vortex theory

## [1.0.0] - 2025-12-22

### Initial Release
- Working numerical solvers for electron, muon, tau masses
- Hill vortex formulation with density gradient
- Validation test suite (grid convergence, multi-start, profile sensitivity)
- Documentation with technical content (but promotional tone)

### Known Issues (Addressed in 2.0)
- Overclaimed "100% accuracy" and "complete unification"
- Missing prominent limitations section
- β from α presented as derived (actually conjectured)
- Solution degeneracy not emphasized
- Celebratory language inappropriate for scientific publication

---

**Versioning**: We use [Semantic Versioning](https://semver.org/)
- MAJOR: Incompatible changes (theory/code revisions)
- MINOR: Backwards-compatible additions (new features, tests)
- PATCH: Backwards-compatible fixes (bugs, typos)
