# LaGrangianSolitons Archive Manifest

**Archive Date**: 2026-01-10
**Archived By**: Claude Opus 4.5 (automated archival)
**Original Location**: `projects/particle-physics/LaGrangianSolitons/`

## Purpose

This archive contains **deprecated development files** from the LaGrangianSolitons nuclear physics investigation. These files represent intermediate exploration, testing, and analysis work that has been superseded by the final validated code in the main repository.

## Why Archived

The LaGrangianSolitons project went through multiple iterations:
1. Initial exploration of nuclear binding energies
2. Development of the 3-family harmonic resonance model
3. Discovery of the Tacoma Narrows interpretation (resonance = instability)
4. Final consolidation into validated scripts

The archived files include:
- Session summaries and progress notes
- Intermediate solver attempts (many approaches tried)
- Test scripts for various hypotheses
- Debugging and analysis scripts
- Early documentation drafts

## Current Active Code

The **validated, production code** remains in:
- `projects/particle-physics/LaGrangianSolitons/src/` - Final validated scripts
- `projects/particle-physics/LaGrangianSolitons/reports/` - Final documentation
- `qfd/` - Core constants and derivations (single source of truth)
- `QFD-Universe` repo - Public-facing validation suite

## Archive Contents

### Summary
| Type | Count |
|------|-------|
| Python scripts | 151 |
| Markdown docs | 86 |
| Images | 10 |
| Shell scripts | 6 |
| Other files | 11 |
| **Total** | **264** |

### Categories

#### Session Summaries (Markdown)
Development session notes documenting the exploration process:
- `SESSION_SUMMARY_*.md` - Daily progress notes
- `*_BREAKTHROUGH_*.md` - Discovery documentation
- `*_RESULTS.md` - Intermediate results

#### Solver Attempts (Python)
Various approaches tried during development:
- `*_solver.py` - Different solver implementations
- `qfd_*.py` - QFD-specific calculation scripts
- `test_*.py` - Hypothesis testing scripts
- `analyze_*.py` - Analysis and debugging scripts
- `calibrate_*.py` - Parameter tuning scripts

#### harmonic_nuclear_model/
A standalone package attempt (superseded by QFD-Universe):
- Complete folder structure with its own src/, docs/, scripts/
- Includes CITATION.cff, LICENSE, README.md
- Validation scripts and data

#### Visualization & Data
- `*.png` - Generated plots and figures
- `*.csv` - Intermediate data exports
- `*.txt` - Results and logs

## Recovery

To recover any file from this archive:
```bash
# Copy back to original location
cp archive/LaGrangianSolitons_deprecated/<filename> \
   projects/particle-physics/LaGrangianSolitons/

# Or access via git history
git log --all -- projects/particle-physics/LaGrangianSolitons/<filename>
git show <commit>:projects/particle-physics/LaGrangianSolitons/<filename>
```

## Key Findings Preserved

The important discoveries from this exploration are preserved in:

1. **Tacoma Narrows Interpretation**: Resonance causes instability (not stability)
   - See: `reports/` and `QFD-Universe/analysis/nuclear/`

2. **3-Family Harmonic Model**: Nuclei classified into A, B, C families
   - See: `src/nucleus_classifier.py`

3. **Golden Loop Derivation**: β = 3.043233 from α = 1/137.036
   - See: `qfd/shared_constants.py`

4. **Fission Mode Analysis**: Parent stable → fragments unstable
   - See: `src/experiments/validate_fission_pythagorean.py`

## Notes

- These files are kept for historical reference and potential future investigation
- The git history also preserves all changes for complete traceability
- Contact Tracy McSheery for questions about specific archived content
