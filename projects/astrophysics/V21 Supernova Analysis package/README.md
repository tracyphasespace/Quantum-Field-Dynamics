# V21 Supernova Analysis Package - README

## 1. Overview
This package provides a complete, self-contained pipeline to reproduce the analysis and key findings related to the QFD (Quantum-Field Dynamics) and ΛCDM (Lambda-Cold Dark Matter) cosmological models. The analysis is based on the Dark Energy Survey 5-Year (DES-SN5YR) supernova data.

The primary result of this analysis is the falsification of cosmological time dilation as predicted by the standard ΛCDM model.

## 2. Key Result: Time Dilation Test

The core finding of this package is demonstrated in the **Time Dilation Test** (`results/time_dilation_test.png`). This plot compares the observed "stretch" parameter of Type Ia supernovae against redshift.

-   **ΛCDM Prediction**: The standard ΛCDM model predicts that the duration of supernova light curves should be stretched by a factor of `(1+z)`, where `z` is the redshift. Therefore, the stretch parameter `s` should follow `s = 1+z`.
-   **QFD Prediction**: The QFD model, which is based on a static universe with a different physical mechanism for redshift, predicts no such time dilation. The stretch parameter should be constant, `s ≈ 1.0`, across all redshifts.
-   **Observed Data**: The data clearly shows a flat stretch parameter across all redshifts, consistent with the QFD prediction and inconsistent with ΛCDM.

**Conclusion**: The absence of (1+z) time dilation in the supernova data challenges a foundational assumption of standard supernova cosmology and the evidence for dark energy.

## 3. How to Reproduce the Results

### Prerequisites
Before running any scripts, install the required Python packages:
```bash
pip install -r requirements.txt
```

### Option 1: Reproduce Plots from Pre-computed Results (Quickest)
This option uses the pre-computed Stage 1 and Stage 2 results included in the `data/` directory to generate the final plots and run the forensics analysis.

1.  **Generate the canonical plots**:
    ```bash
    python3 plot_canonical_comparison.py
    ```
    This will generate `canonical_comparison.png` and `time_dilation_test.png` in the `results/` directory.

2.  **Run the BBH forensics analysis**:
    ```bash
    python3 analyze_bbh_candidates.py \
      --stage1-dir results/stage1_output/ \
      --stage2-results data/stage2_results_with_redshift.csv \
      --lightcurves data/lightcurves_all_transients.csv \
      --out results/forensics_output \
      --top-n 10
    ```
    This will analyze the top 10 candidates and save the results in `results/forensics_output/`.

### Option 2: Run the Full Pipeline from Scratch (Hours)
This option regenerates all results from the raw lightcurve data included in this package.

1.  **Run Stage 1: Full-scale fitting**:
    This will fit all 8,277 supernovae. It is a long-running process.
    ```bash
    python3 stage1_v20_fullscale_runner.py \
      --lightcurves data/lightcurves_all_transients.csv \
      --out results/stage1_output/ \
      --save-incremental
    ```

2.  **Run Stage 2: Candidate Selection**:
    ```bash
    python3 stage2_select_candidates.py \
      --stage1-dir results/stage1_output/ \
      --lightcurves data/lightcurves_all_transients.csv \
      --out data/
    ```

3.  **Generate plots and run forensics**:
    Follow the instructions in **Option 1**.

### Option 3: Obtain Data from Original Source
The raw lightcurve data used in this analysis is from the Dark Energy Survey 5-Year (DES-SN5YR) data release. The data is publicly available and can be obtained from:

-   **Zenodo**: `https://doi.org/10.5281/zenodo.12720778`
-   **GitHub**: `https://github.com/des-science/DES-SN5YR`

If you download the data from the original source, ensure it is in the format expected by the `LightcurveLoader` in `v17_data.py`. The included `data/lightcurves_all_transients.csv` can be used as a reference for the expected format.

## 4. Code Guide
For a detailed breakdown of the executable scripts vs. library modules in this package, please refer to the `CODE_GUIDE.md` file.

## 5. Physics Model
An explanation of the QFD model physics, including the corrected conventions for distance modulus and residuals, can be found in `QFD_PHYSICS.md`.

## 6. Citation
If you use this analysis, code, or data in your research, please cite this package. You can use the following format:

```
V21 Supernova Analysis Package
https://github.com/tracyphasespace/Quantum-Field-Dynamics/tree/main/projects/astrophysics/V21%20Supernova%20Analysis%20package
```