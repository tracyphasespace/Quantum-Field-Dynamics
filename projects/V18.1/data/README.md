# V18.1 Dataset: Clean Type Ia Sample

## Dataset Generation

This directory contains the clean Type Ia dataset for the V18.1 controlled experiment.

### File: `lightcurves_type_ia_clean.csv` (123.1 MB)

**Specifications:**
- **6,895 Type Ia supernovae** (SNTYPE 0, 1, 4 only)
- **649,682 photometric measurements**
- Redshift range: 0.050 - 1.298
- Quality cuts: ≥5 observations per SN

**SNTYPE Distribution:**
- SNTYPE 0: 6,545 SNe (94.9%) - Photometric Type Ia
- SNTYPE 1: 349 SNe (5.1%) - Spectroscopic Type Ia  
- SNTYPE 4: 1 SNe (0.0%) - Peculiar Type Ia

**Excluded Types** (no cross-contamination):
- Core-collapse: 5, 23, 29, 32, 33, 39
- SLSN: 41, 66, 141
- AGN/TDE: 80, 81, 82
- Other: 101, 122, 129, 139, 180

## Regeneration Command

```bash
python3 projects/V19/scripts/extract_full_dataset.py \
  --data-dir <DES-SN5YR-1.2/0_DATA> \
  --output v18/data/lightcurves_type_ia_clean.csv \
  --dataset DES-SN5YR_DES \
  --include-all-types \
  --exclude-types 5 23 29 32 33 39 41 66 80 81 101 122 129 139 141 180 \
  --min-obs 5 --min-z 0.05 --max-z 1.3
```

**Purpose:** Controlled baseline for testing V18.1 physics corrections with zero contamination from prior workflows.
