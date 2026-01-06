# Zenodo Upload Instructions

**Package**: Harmonic Nuclear Model v1.0.0
**File**: `harmonic_nuclear_model_v1.0.0.zip`
**Size**: 6.4 MB
**Created**: 2026-01-03

---

## File Checksums (for verification)

**SHA256**: `87bc972cbcd314b75a32d39c6d410e6375a9f63b5313896363c205568261077f`

**MD5**: `c60c60075516652749e69407fa43cd78`

---

## Zenodo Upload Steps

### 1. Create New Upload

Go to: https://zenodo.org/deposit/new

### 2. Upload Files

**Main archive**:
- `harmonic_nuclear_model_v1.0.0.zip` (6.4 MB)

**Verification files** (optional):
- `harmonic_nuclear_model_v1.0.0.zip.sha256`
- `harmonic_nuclear_model_v1.0.0.zip.md5`

### 3. Fill Metadata

**Required Fields**:

```
Title: Harmonic Nuclear Model: Soliton Topology and Nuclear Stability

Upload type: Software

Publication date: 2026-01-03

Creators:
  - Name: McSheery, Tracy
    Affiliation: Independent Researcher
    ORCID: (your ORCID if available)

Description:
Complete implementation and validation of the Harmonic Family Model for
nuclear structure, based on QFD (Quantum Field Dynamics) soliton theory.
Includes single-center model for spherical nuclei (A ≤ 161) and two-center
extension for deformed nuclei (A > 161).

Key Results:
- Validates dual-core soliton hypothesis across full NUBASE2020 (3,558 nuclides)
- Demonstrates Tacoma Narrows mechanism (resonance → instability) is universal
- Two-center model recovers half-life correlation for heavy nuclei (r = 0.34, p < 10⁻³¹)
- Predicts shape transition at A = 161 from soliton core saturation

Package Contents:
- Complete Python implementation (~3,000 lines)
- Comprehensive documentation (~145 KB)
- Publication-quality figures (10 PNG, 300 DPI)
- Full reproducibility scripts
- Validation on complete NUBASE2020 dataset

License: MIT

Version: 1.0.0

Language: Python

Keywords (select from list or add):
- nuclear physics
- soliton theory
- QFD
- quantum field dynamics
- nuclear structure
- harmonic resonance
- decay rates
- nuclear stability
- NUBASE2020
- machine learning

Related identifiers:
  - GitHub Repository: https://github.com/tracyphasespace/Quantum-Field-Dynamics/tree/main/projects/particle-physics/harmonic_nuclear_model
  - Relation: isSupplementTo

References:
  - Kondev, F.G., Wang, M., Huang, W.J., Naimi, S., Audi, G. (2021).
    "The NUBASE2020 evaluation of nuclear physics properties".
    Chinese Physics C, 45(3), 030001.
    DOI: 10.1088/1674-1137/abddae

Funding: None

Access right: Open Access

License: MIT License (selected from dropdown)
```

### 4. Select Communities (Optional)

Search and add to relevant communities:
- Nuclear Physics
- Computational Physics
- Open Science
- Python

### 5. Publish

- Review all metadata
- Click "Publish"
- Note the assigned DOI (format: 10.5281/zenodo.XXXXXXX)

---

## Post-Publication Tasks

### 1. Update CITATION.cff

In the GitHub repository, update `CITATION.cff` with the Zenodo DOI:

```yaml
doi: 10.5281/zenodo.XXXXXXX
```

### 2. Update README.md

Add DOI badge to README:

```markdown
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
```

### 3. Add to arXiv (if submitting paper)

In the paper's acknowledgments/data availability section:

```latex
\section*{Data Availability}
All code and data used in this study are available at
\url{https://doi.org/10.5281/zenodo.XXXXXXX}
(McSheery, 2026). The code is also maintained at
\url{https://github.com/tracyphasespace/Quantum-Field-Dynamics}.
```

### 4. Social Media Announcement Template

```
🚀 Just released the Harmonic Nuclear Model v1.0.0!

Complete implementation validating dual-core soliton hypothesis across
3,558 nuclides (NUBASE2020). Two-center extension recovers half-life
correlation for heavy nuclei with r=0.34, p<10⁻³¹.

📦 Archive: https://doi.org/10.5281/zenodo.XXXXXXX
💻 Code: https://github.com/tracyphasespace/Quantum-Field-Dynamics

#NuclearPhysics #SolitonTheory #OpenScience
```

---

## Additional Repository Platforms

### Figshare

1. Upload: https://figshare.com/account/projects
2. Use same metadata as Zenodo
3. Cross-link DOIs

### OSF (Open Science Framework)

1. Create project: https://osf.io/
2. Upload files and link to Zenodo DOI
3. Add project description and tags

### GitHub Release

1. Navigate to repository: https://github.com/tracyphasespace/Quantum-Field-Dynamics
2. Click "Releases" → "Create a new release"
3. Tag: `v1.0.0-harmonic-nuclear-model`
4. Title: "Harmonic Nuclear Model v1.0.0"
5. Description: Same as Zenodo
6. Attach `harmonic_nuclear_model_v1.0.0.zip`
7. Link to Zenodo DOI in release notes

### Software Heritage

Zenodo automatically archives to Software Heritage for long-term preservation.
Verify at: https://archive.softwareheritage.org/

---

## Citation Templates

### Software Citation (APA)

```
McSheery, T. (2026). Harmonic Nuclear Model: Soliton Topology and Nuclear
Stability (Version 1.0.0) [Computer software]. Zenodo.
https://doi.org/10.5281/zenodo.XXXXXXX
```

### Software Citation (Chicago)

```
McSheery, Tracy. 2026. "Harmonic Nuclear Model: Soliton Topology and Nuclear
Stability." Version 1.0.0. Zenodo. https://doi.org/10.5281/zenodo.XXXXXXX.
```

### Software Citation (BibTeX)

```bibtex
@software{mcsheery2026harmonic,
  author       = {McSheery, Tracy},
  title        = {{Harmonic Nuclear Model: Soliton Topology and
                   Nuclear Stability}},
  month        = jan,
  year         = 2026,
  publisher    = {Zenodo},
  version      = {v1.0.0},
  doi          = {10.5281/zenodo.XXXXXXX},
  url          = {https://doi.org/10.5281/zenodo.XXXXXXX}
}
```

---

## Version History

**v1.0.0** (2026-01-03):
- Initial public release
- Single-center model for spherical nuclei (A ≤ 161)
- Two-center extension for deformed nuclei (A > 161)
- Complete validation on NUBASE2020 (3,558 nuclides)
- Publication-quality figures and documentation

**Future versions**: Will be uploaded as new Zenodo versions under the same concept DOI.

---

## Contact

For questions about the Zenodo deposit:
- GitHub Issues: https://github.com/tracyphasespace/Quantum-Field-Dynamics/issues
- Email: (add your email)

---

## Archival Metadata

**Package Name**: harmonic_nuclear_model
**Version**: 1.0.0
**Release Date**: 2026-01-03
**File Format**: ZIP archive
**Compression**: Standard ZIP deflate
**Total Files**: 50
**Uncompressed Size**: 7.5 MB
**Compressed Size**: 6.4 MB
**Compression Ratio**: 85%

**Contents**:
- Documentation: 12 files (~145 KB)
- Source Code: 10 Python files (~3,000 lines)
- Scripts: 5 shell scripts
- Figures: 10 PNG images (300 DPI)
- Configuration: requirements.txt, CITATION.cff, LICENSE

**Excluded** (regenerated from scripts):
- data/derived/*.parquet (NUBASE2020 parsed data)
- reports/*/*.json (analysis results)

**Data Availability**:
- NUBASE2020 source: https://www-nds.iaea.org/amdc/
- Regeneration: Run `scripts/run_all.sh` after downloading NUBASE2020

---

**Archive Prepared**: 2026-01-03
**Ready for Upload**: ✓
**Verification**: SHA256 and MD5 checksums provided
**Long-term Preservation**: Suitable for Zenodo, Figshare, OSF
