# A Researcher's Guide to the QFD Supernova Pipeline (V18)

**A Physically Motivated, Non-Cosmological Model for the Supernova Hubble Diagram**

## 1. Executive Summary: What is This and Why Should I Care?

This repository contains the complete computational pipeline for testing an alternative physical model for the observed properties of Type Ia supernovae. The standard cosmological model (ΛCDM) explains the anomalous dimming of distant supernovae by postulating that the expansion of the universe is accelerating, driven by a mysterious "Dark Energy."

This project tests a fundamentally different hypothesis, rooted in a theory called Quantum Field Dynamics (QFD):

> **The anomalous dimming of distant supernovae is not a cosmological effect. It is a local, physical phenomenon caused by the intense environment of the supernova explosion itself. The light is physically processed before it ever leaves the host galaxy.**

Our model requires no cosmic acceleration and no Dark Energy. Instead, it explains the observed data using a combination of known and physically-grounded principles: plasma physics, non-linear optics, and thermal spectroscopy.

The purpose of this pipeline is to take this more complex physical model and rigorously fit it to the raw, unfiltered light curves of the **full 4,831 "clean" supernovae from the Dark Energy Survey 5-Year (DES-SN5YR) dataset**. Unlike standard analyses that often discard a significant fraction of data as "outliers," our approach uses a robust statistical framework (a Student's t-likelihood) to explain the entire population, including its variance.

This document will explain the physical model, how it differs from ΛCDM, and how our computational pipeline is designed to provide a direct, falsifiable test of this alternative paradigm. We invite skeptical review, as the goal is to determine which model—the one with magical cosmic components or the one with complex local physics—best describes the universe we observe.

## 2. The QFD Supernova Model: A Multi-Component Physical Framework

Our model proposes that the journey of a photon from a supernova explosion to our telescope is governed by three distinct physical processes, which together produce the full redshift-distance relationship.

### Component 1: The Baseline Redshift (The "Cosmic Drag")

*   **What it is:** A weak, cumulative energy exchange that occurs as photons travel over vast intergalactic distances. This is a "tired light" mechanism, but one rigorously derived from the QFD framework, which avoids the image-blurring problems of older models.
*   **What it explains:** The baseline Hubble-Lemaître Law (`z ≈ H₀D`). This is the dominant source of redshift for all cosmic objects.
*   **In the Code (`v17_qfd_model.py`):** This is implemented in the `calculate_z_drag(k_J, distance_mpc)` function. The parameter `k_J` sets the strength of this universal drag. For our fits, we anchor this to the observed value of the Hubble constant (`k_J ≈ 70 km/s/Mpc`), as this part of the model is not controversial. We then fit for a small correction (`k_J_correction`) to account for any residual systematic effects.

### Component 2: The Local Anomalous Redshift (The "Veil" and the "Sear")

This is the core of our "Dark Energy" replacement. It posits that the environment immediately surrounding the supernova is an extreme-physics laboratory that physically alters the light.

*   **A. The Plasma Veil:**
    *   **Physics:** The supernova's own ejecta forms a hot, dense, expanding cloud of plasma. Photons must "fight their way out" of this veil, undergoing Compton scattering and atomic interactions. This is known physics.
    *   **Effect:** This process is wavelength-dependent, preferentially scattering higher-energy (blue) photons. It causes the supernova to appear intrinsically dimmer and redder.
    *   **In the Code:** This is modeled by the per-supernova parameters `A_plasma` (strength of the veil) and `β` (the wavelength slope `~λ⁻β`).

*   **B. The Flux-Dependent Redshift (FDR) / "Sear":**
    *   **Physics:** The photon pulse at its peak is so intense that it alters the properties of the vacuum itself, turning it into a non-linear optical medium. This is analogous to effects seen in high-intensity laser experiments (e.g., SLAC E144), where light-by-light scattering becomes significant.
    *   **Effect:** This intense self-interaction "cools" the photon pulse, further dimming it and shifting its energy to the red.
    *   **In the Code:** This is modeled by the global parameter `η'` (eta_prime), which sets the strength of this non-linear effect.

These two effects are combined in the `calculate_z_local(...)` function. Because they are local to each supernova, their strength varies depending on the specific progenitor environment.

### Component 3: The Thermal Broadening Effect (The Planck/Wien Correction)

*   **Physics:** This is not a new physical mechanism, but a well-known observational effect in spectroscopy, often included in the "K-correction." A supernova is a cooling thermal source. At high redshift, a fixed observer filter (e.g., V-band) is actually measuring light that was emitted in the supernova's rest-frame U-band or even UV. Since the supernova is hotter but less luminous in the deep UV (the Wien tail of its spectrum), it will naturally appear dimmer. This effect also changes the *shape* of the observed light curve, making it appear broader.
*   **What it explains:** A significant portion of the anomalous dimming trend that is usually attributed to cosmic acceleration.
*   **In the Code:** This is modeled as a magnitude correction `delta_mu_thermal` in the `predict_apparent_magnitude(...)` function, driven by the global parameter `ξ` (xi).

### The Full QFD Model vs. ΛCDM

| Observable                 | Standard ΛCDM Interpretation                                     | QFD Physical Interpretation                                                                                                    |
| -------------------------- | ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| **Baseline Redshift**      | Geometric stretching of spacetime due to cosmic expansion.       | **Cosmic Drag:** A weak, physical energy exchange over cosmological distances.                                                   |
| **Anomalous Dimming**      | Geometric effect of **Accelerated Expansion**, driven by Dark Energy. | **Local Physics:** A combination of Plasma Veil scattering, FDR self-interaction, and the Planck/Wien thermal observation effect. |
| **Light Curve Broadening** | **Time Dilation:** The "stretching of time itself" by a factor of `(1+z)`. | **Thermo-Dynamic Broadening:** An intrinsic effect caused by the same local physics that causes the dimming. They are causally linked. |
| **Scatter / "Outliers"**     | **Noise:** Peculiar supernovae or systematic errors to be removed.    | **Signal:** A direct measurement of the physical diversity of supernova environments (e.g., some have denser plasma veils).     |

## 3. The Computational Pipeline: A Rigorous and Robust Approach

Our goal is to fit this complex, multi-component physical model to the raw DES 5-year data. This requires a sophisticated and robust computational pipeline.

### Stage 1: Per-Supernova Fitting (`stage1_optimize_v17.py`)

*   **Goal:** For each of the ~5,000 supernovae, find the best-fit values for its unique local parameters: `t₀` (explosion time), `ln_A` (distance/amplitude), `A_plasma`, and `β`.
*   **Methodology:**
    1.  **Likelihood:** We use a **Student's t-likelihood**. Unlike a standard Gaussian (chi-squared) fit, this method is robust to outliers. It can fit a light curve well even if a few data points are contaminated, preventing the entire supernova from being discarded.
    2.  **Optimizer:** We use a professional-grade, gradient-based optimizer (L-BFGS-B).
    3.  **Gradients:** To ensure speed and accuracy, we provide the optimizer with the *exact* analytical gradient of our entire physical model, calculated automatically using **JAX**.
*   **Output:** For each supernova, a file containing its best-fit local parameters and a measure of the fit quality.

### Stage 2: Global MCMC Fitting (`stage2_mcmc_v17.py`)

*   **Goal:** Using the results from Stage 1, find the best-fit values for the **global physical parameters** of the QFD model (`k_J_correction`, `η'`, `ξ`) that describe the entire population.
*   **Methodology:**
    1.  **Model:** We use the results from Stage 1 (`ln_A_obs` and `z_obs`) as the input data. Our model `ln_A_pred(z, ...)` predicts the expected amplitude for a given redshift based on the global QFD parameters.
    2.  **Sampler:** We use the No-U-Turn Sampler (NUTS) provided by **NumPyro**. This is a state-of-the-art MCMC algorithm that is far more efficient and robust than traditional methods.
    3.  **Likelihood:** Again, we use a global Student's t-likelihood to naturally account for the population of "outlier" supernovae (which we hypothesize are those with extreme local environments, like BBH progenitors). The model learns the "fatness" of the tails directly from the data.
*   **Output:** The posterior probability distributions for the global QFD parameters, which tell us their best-fit values and uncertainties.

### Why This is a Fair Test

*   **No Cherry-Picking:** We fit the full "clean" dataset of 4,831 SNe, not a pre-selected "cosmology-grade" subset.
*   **Apples-to-Apples:** We explicitly compare our final Hubble diagram to the standard ΛCDM model fit on the *exact same data* with the *exact same likelihood function*, providing a direct and fair comparison of model performance.
*   **Falsifiability:** If our model cannot produce a good fit (e.g., has high residuals, fails to converge, or requires unphysical parameters), it is falsified.

## 4. How to Reproduce Our Results

The entire pipeline is provided in this repository. A detailed guide is available in **[REPRODUCTION_GUIDE.md](REPRODUCTION_GUIDE.md)**. The basic steps are:

1.  **Set up the environment:** Run `./setup_environment.sh` to install all required libraries.
2.  **Download the data:** The data loader will automatically fetch the required DES-SN5YR light curve files.
3.  **Run the pipeline:** Execute the shell scripts in order to run Stage 1 and Stage 2.
    *   `./v18/scripts/run_stage1_parallel.py`
    *   `./v18/stages/stage2_mcmc_v18_emcee.py`
4.  **Analyze the output:** The scripts will generate plots (Hubble diagram, corner plots) and data files containing the final fitted parameters and their uncertainties.

We believe that transparency and reproducibility are the bedrock of good science. We invite you to run the code, scrutinize the model, and challenge the results. The data is public, the code is open, and the question is profound: Is the universe accelerating, or have we simply been misinterpreting the light from its most distant explosions? This pipeline is our tool for answering that question.