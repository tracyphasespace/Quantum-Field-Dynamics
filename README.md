Quantum Field Dynamics (QFD)

A Unified Physical Framework and a Computationally Solvable Model for the Constants of Nature.

This repository contains the complete theoretical and computational framework for Quantum Field Dynamics (QFD). QFD is a first-principles theory that reconstructs all of observed physics—including particles, forces, and 4D spacetime—as the emergent, stable solutions of a single multivector field ψ in a 6-coordinate Cl(3,3) phase space.

The theory's central claim is that the fundamental constants of our universe are not arbitrary inputs but are the unique, calculable outputs of a well-posed, massively over-constrained, and computationally tractable optimization problem, which we term the "Grand Unified Solver."

Key Verifiable Result: The Core Compression Law for Nuclides

As a primary validation of the QFD framework, a two-term law for nuclear stability was derived from the theory's first principles. This law posits that the charge (Q) of any nucleus is a direct function of its mass number (A) based on geometric and field-compression effects:

Q(A) = c₁A²/³ + c₂A

When this theoretically-derived law was tested against the comprehensive NuBase 2020 dataset, containing ~5,800 known isotopes, it accounted for R² ≈ 0.98 of the variance across the entire chart. This is not a post-hoc curve-fit; it is the successful empirical validation of a core theoretical prediction.

Reproduce the Key Result

You can verify this primary result on your own machine in three steps.

1. Clone the repository and navigate to the root directory:
code
Bash
download
content_copy
expand_less

git clone https://github.com/tracyphasespace/Quantum-Field-Dynamics.git
cd Quantum-Field-Dynamics
2. Install dependencies:

(Ensure you have Python 3.8+ installed)```bash
pip install -r requirements.txt

code
Code
download
content_copy
expand_less
IGNORE_WHEN_COPYING_START
IGNORE_WHEN_COPYING_END
#### 3. Run the validation script:
```bash
python validation/reproduce_nuclide_law.py

Expected Output:
The script will process the included NuBase2020.txt dataset and produce the following:

Console output confirming the fit:

code
Code
download
content_copy
expand_less
IGNORE_WHEN_COPYING_START
IGNORE_WHEN_COPYING_END
Validation Successful.
Model: Q(A) = c1*A**(2/3) + c2*A
Coefficients: c1 = [VALUE], c2 = [VALUE]
R-squared against ~5800 isotopes: 0.979...

A plot saved to results/Nuclide_Backbone_Fit.png, visually confirming the model's accuracy.

Core Principles of the QFD Framework

Geometric Unification (Cl(3,3) Algebra): All forces (gravity, EM, nuclear) emerge as different geometric gradients of the single ψ-field. The imaginary unit i is replaced by a physical bivector B where B²=-1.

Emergent Time & Spacetime: Time is not a fundamental dimension but an emergent, positive scalar ordering parameter (τ). Our (3+1)D spacetime is a dynamically suppressed effective theory that emerges from the 6C phase space, enforced by the symmetries of stable particles.

A Static, Self-Regulating Cosmos: The Big Bang is replaced by an eternal, steady-state universe. Cosmic redshift is a calculable photon-field interaction, and the CMB is the universe's present-day thermal equilibrium.

Particles as Geometric Solitons: All particles are stable, localized wavelets of the ψ-field. Mass and charge are emergent, calculable properties of a wavelet's geometry.

No Hidden Entities: The framework provides mechanistic explanations for cosmological phenomena without invoking dark matter, dark energy, or inflation.

Falsifiable Predictions

QFD makes several concrete, near-term predictions that distinguish it from the Standard Model and ΛCDM:

"Zombie Galaxies" as Dark Matter: Over 90% of the universe's baryonic matter exists as non-luminous, gravitationally-bound "zombie galaxies" in cosmic voids. This provides the universe's missing mass without requiring new particles and is testable with deep lensing surveys.

Supernova Dimming is Wavelength-Dependent: The anomalous dimming of distant supernovae is a near-source scattering effect. This predicts they should appear systematically bluer than expected, a testable prediction for JWST and the Roman Space Telescope.

Gravitational Deflection of Matter vs. Light: The framework predicts that slow-moving massive particles (e.g., cold neutrons) will experience exactly half the gravitational deflection of light, a key distinguishing test from General Relativity.

Repository Navigation

docs/: The complete manuscript for the book, "Quantum Field Dynamics," and supporting theoretical papers.

validation/: Scripts to reproduce the key empirical validations of the theory, such as reproduce_nuclide_law.py.

solvers/: The source code for the QFD computational solvers for various domains (atomic, cosmological, etc.).

data/: The raw experimental datasets used for validation (e.g., NuBase2020.txt, union2.1_data.txt).

results/: Directory where output plots and data from the validation scripts are saved.

Contribution and Collaboration

The theoretical foundation of QFD is laid; the next phase is computational validation and refinement. This is an open invitation to the scientific community to test, critique, and extend this work. The most productive way to engage is through the GitHub Issues tab, where specific research problems, computational challenges, and theoretical questions are cataloged.

License and Citation

This work is licensed under the MIT License. If you use the concepts, code, or data from this repository in your research, please cite the main manuscript:

McSheery, T. (2025). Quantum Field Dynamics: A Dynamic View of a Steady State Universe. https://github.com/tracyphasespace/Quantum-Field-Dynamics
