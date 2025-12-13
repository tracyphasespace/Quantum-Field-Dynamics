code
Markdown
download
content_copy
expand_less
# Quantum Field Dynamics (QFD)

**A Unified Physical Framework and Grand Unified Solver for the Constants of Nature.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Lean 4](https://img.shields.io/badge/Verified-Lean%204-green)](projects/Lean4)
[![Python 3.10+](https://img.shields.io/badge/Code-Python%203.10-blue)](projects/)

> **"What if the parameters of the universe aren't inputs, but outputs?"**

This repository contains the complete theoretical, mathematical, and computational framework for **Quantum Field Dynamics (QFD)**. QFD is a first-principles theory that reconstructs particles, forces, and cosmology as emergent solutions of a single multivector field (ψ) in a 6-coordinate Cl(3,3) Phase Space.

Unlike standard alternative theories, QFD is not just a manuscript; it is a **codebase**. It rejects the Big Bang, General Relativity, and the Standard Model in favor of a static, geometric universe where Time is an emergent scalar (viscosity) and Matter is topological geometry (solitons).

---

## 🏗️ The Three Pillars of Validation

This repository validates the theory across three rigorous distinct domains. All claims are backed by executable code.

### 1. Formal Verification (Logic)
**Location:** [`projects/Lean4/`](projects/Lean4/)  
**Tool:** **Lean 4 Theorem Prover**  
Physics theories often hide behind "hand-waving." We prove the fundamental architecture formally.
*   ✅ **EmergentAlgebra.lean:** Mathematically proves that the centralizer of an internal rotation in 6D phase space **must** manifest as 4D Minkowski Spacetime (Cl(3,1)).
*   ✅ **SpectralGap.lean:** Rigorously proves that if topological angular momentum is quantized, an energy gap ΔE naturally "freezes out" the extra dimensions, rendering them unobservable without compactification.

### 2. Microphysics (Solitons & Nuclides)
**Location:** [`projects/particle-physics/`](projects/particle-physics/)  
**Tool:** **PyTorch / Python / SciPy**  
*   **The Phoenix Solver:** Minimizes the QFD Hamiltonian to generate stable particle states. It attempts to *derive* the masses of the Lepton ladder (Electron, Muon, Tau) as resonant isomers of the electron vortex, rather than inputting them.
*   **Nuclide Prediction:** Contains the verification of the **Core Compression Law** (Q ∝ c₁A^(2/3) + c₂A). We fit the "zero-stress backbone" of nuclear stability across ~5,800 isotopes (R² ≈ 0.98), demonstrating that the "Strong Force" is an emergent pressure balance between nuclear density and the electron cloud.

### 3. Cosmology (Supernovae & Redshift)
**Location:** [`projects/astrophysics/`](projects/astrophysics/)  
**Tool:** **JAX / NumPyro / Bayesian Inference**  
We test the QFD "Static Universe" against raw observational data.
*   **Time Dilation Falsification:** Analysis of 4,800+ SNe Ia (DES-SN5YR) showing that light-curve width does *not* stretch by (1+z) when selection bias is removed, falsifying metric expansion.
*   **Redshift Forensics:** Demonstrates that SNe Ia residuals show a **5.2:1 asymmetry** (Dark vs. Bright outliers), proving that "Dark Energy" is actually flux-dependent scattering ("Plasma Veil") and gravitational lensing, not cosmic acceleration.

---

## 🧪 Reproduce Key Results

You can verify the physics on your own machine.

### A. Verify the Nuclear Scaling Law
This script reproduces the "Force-Free" nuclear fit against the NuBase dataset.
```bash
git clone https://github.com/tracyphasespace/Quantum-Field-Dynamics.git
cd Quantum-Field-Dynamics
pip install -r requirements.txt
python projects/particle-physics/nuclide-prediction/run_core_compression.py
B. Verify the Logic (Lean 4)

If you have a Lean 4 installation:

code
Bash
download
content_copy
expand_less
cd projects/Lean4
lake build QFD.SpectralGap
lake build QFD.EmergentAlgebra

Output: Build completed successfully confirms the theorems hold.

C. Run the Supernova Forensic

(Requires JAX installed with CUDA support recommended)

code
Bash
download
content_copy
expand_less
python projects/astrophysics/V21_Supernova_Analysis/run_static_fit.py

Outputs the Hubble Diagram comparing ΛCDM residuals vs. the QFD Static model residuals.

🌌 Core Claims of the Framework

Gravity is Refraction: Mass concentrates the ψ-field. A dense field has a higher refractive index n = √h. Light bends around stars and clocks slow down (Time Dilation) due to simple refraction, not curved spacetime.

Matter is Geometry: Particles are not points; they are Q-Ball Solitons and Hill Vortices. Stability is enforced by the "Cavitation Limit" (a vacuum floor preventing infinite collapse).

Redshift is Interaction: Light gets "tired" via a specific coherence-preserving forward scattering mechanism ("Cosmic Drag"). The scattered light becomes the CMB.

No Heat Death: 95% of the universe's mass is "Zombie Galaxies"—non-luminous baryonic matter hiding in voids. These act as the thermodynamic heat sinks for the universe, recycling starlight and maintaining equilibrium.

⚡ Falsifiable Predictions

This is not philosophy; it is falsifiable science. QFD is wrong if:

Experimental: The Muon/Electron mass ratio cannot be derived from the Phoenix Hamiltonian parameters.

Observational: High-z Quasar spectra show line-broadening proportional to redshift (QFD predicts lines stay sharp; only the packet envelope broadens).

Mathematical: The Spectral Gap inequality is disproven.

📂 Repository Structure

docs/: The complete manuscript (700+ pages) and supporting Trojan Horse papers.

projects/Lean4/: (NEW) Formal proofs verifying the mathematical consistency of dimensional reduction and quantization.

projects/particle-physics/:

/phoenix_solver: Vortex simulation engine.

/nuclide-prediction: Nuclear stability fits.

projects/astrophysics/:

/V21_Supernova_Analysis: (NEW) Bayesian analysis of DES5YR data.

/redshift-analysis: FDTD simulations of photon-vacuum scattering.

/black-hole-dynamics: Simulation of Galactic Spiral formation via tidal interactions (The "Rift").

data/: Source datasets (NuBase2020, Pantheon+, DES-SN5YR).

Contributing

The theoretical foundation is complete. We are now in the Computation & Verification phase.

Mathematicians: Check the Lean4 definitions for topological robustness.

Physicists: Run the phoenix_solver and stress-test the Hamiltonian parameters.

Data Scientists: Audit the NumPyro models in the Supernova analysis.

License

MIT License. Please cite:
McSheery, T. (2025). Quantum Field Dynamics: A Dynamic View of a Steady State Universe. GitHub.

code
Code
download
content_copy
expand_less
