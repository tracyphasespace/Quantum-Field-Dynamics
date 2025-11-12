# V15_CLEAN: Data Dependencies

## Current Data Status

### ✅ Provided Dataset (Ready to Use)
**File**: `../data/lightcurves_unified_v2_min3.csv` (13 MB)
- 5,468 SNe from Pantheon+ and DES-SN5YR
- Pre-processed and ready for Stage 1
- Located at repository root: `/data/`

### ⚠️ Data Preparation Tools (NOT in this repo)

The data preparation pipeline is in a SEPARATE repository:
```
../../../October_Supernova/tools/
├── convert_des_fits_to_qfd.py
├── parse_pantheon_plus.py
└── create_unified_dataset.py
```

**These are NOT needed** unless you want to rebuild the dataset from scratch.

## Data Workflow

### If Using Provided Dataset (RECOMMENDED)
```bash
# Data is already here, just run the pipeline:
cd /home/tracy/development/SupernovaSrc/qfd-supernova-v15
./v15_clean/scripts/run_full_pipeline.sh
```

The pipeline will use: `data/lightcurves_unified_v2_min3.csv`

### If Rebuilding from Raw Data Sources

**Prerequisites**:
1. DES-SN5YR public release (2 GB FITS files)
2. Pantheon+ data release
3. Data preparation tools from `October_Supernova` repo

**Steps** (see `../data/README.md` for details):
1. Download DES-SN5YR FITS files
2. Download Pantheon+ data
3. Run conversion scripts (requires external repo)
4. Generate unified CSV
5. Place in `data/` directory

## Missing Tools (Not Critical)

### Data Preparation (in external repo)
- `convert_des_fits_to_qfd.py` - Convert DES FITS to CSV
- `parse_pantheon_plus.py` - Parse Pantheon+ format
- `create_unified_dataset.py` - Combine surveys

### Analysis and Utility Scripts (NOW IN v15_clean/tools/)

✅ **Consolidated into v15_clean/tools/**:
- `analyze_stage1_simple.py` - Analyze Stage 1 results
- `generate_corner_plot.py` - Create parameter corner plots
- `generate_mock_data.py` - Create synthetic test data
- `monitor_pipeline.py` - Monitor running pipelines
- `compare_abc_variants.py` - Compare model variants
- `make_publication_figures.py` - Generate paper figures
- `make_per_survey_report.py` - Per-survey analysis

See `v15_clean/tools/README.md` for usage documentation.

## v15_clean Target Structure

```
v15_clean/
├── core/          # Model and data loading
├── stages/        # Pipeline stages 1-3
├── scripts/       # Run scripts
├── tools/         # Analysis and utilities (TO ADD)
├── tests/         # Unit tests
└── docs/          # Documentation
```

## Data Format

**Input**: `lightcurves_unified_v2_min3.csv`

Required columns:
- `snid`, `mjd`, `z`
- `flux_g`, `flux_r`, `flux_i`, `flux_z` (erg/s/cm²/Hz)
- `fluxerr_g`, `fluxerr_r`, `fluxerr_i`, `fluxerr_z`

**Output**: Stage 1 creates per-SN JSON files with fitted parameters
```
results/v15_clean/stage1_fullscale/
├── SN001.json
├── SN002.json
└── ...
```

## Next Steps

1. **For running pipeline**: You have everything you need!
2. **For adding utilities**: Copy analysis scripts from root to `v15_clean/tools/`
3. **For data preparation**: Need access to `October_Supernova` repo
