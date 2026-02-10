# Golden Loop Supernova Pipeline

Zero-free-parameter fit to 1,768 DES-SN5YR Type Ia supernovae using
the QFD (Quantum Field Dynamics) geometric vacuum model.

**Author**: Tracy McSheery
**Data**: DES-SN5YR (DES Collaboration, 2024)

## Key Result

| Model | RMS [mag] | chi2/dof | Physics params |
|-------|-----------|----------|----------------|
| QFD (locked) | 0.184 | 1.005 | 0 |
| LCDM (best fit) | 0.181 | 0.955 | 2 |

QFD matches LCDM to 1.8% in RMS with zero free physics parameters.

## Derivation Chain

```
alpha = 1/137.036               (fine structure constant, measured)
    | Golden Loop: 1/alpha = 2*pi^2*(e^beta/beta) + 1
beta = 3.04323                   (vacuum stiffness)
    | Hill vortex eigenvalue
k = 7*pi/5                      (soliton boundary condition)
    | Gravitational coupling
xi_QFD = k^2 * 5/6 = 49*pi^2/30
    | Volume stiffness
K_J = xi_QFD * beta^(3/2) = 85.58 km/s/Mpc
    | Scattering opacity
eta = pi^2/beta^2 = 1.066       (0.34% match to fitted value)
```

## Quick Start

```bash
pip install -r requirements.txt
python golden_loop_sne.py
```

Output is printed to stdout. Pre-generated output is in `results_output.txt`.

## Physics

The photon is modeled as a toroidal soliton with poloidal and toroidal
circulation. Traversing the vacuum (psi-field), its circulation decays:

    dE/dD = -(K_J/c) * E

giving exponential energy loss and the distance-redshift relation:

    D(z) = (c/K_J) * ln(1+z)

The four-photon vertex (sigma proportional to E^2) removes photons from
the beam. Because sigma decreases as the photon redshifts, the opacity
saturates at high z:

    tau(z) = eta * [1 - 1/(1+z)^2]

This saturation creates the Hubble diagram curvature that LCDM attributes
to dark energy.

## Chromatic Signature

Multi-band photometry (DES g/r/i/z) shows wavelength-dependent dimming
with correlation r = -0.986 against the predicted lambda^(-2) dependence,
distinguishing QFD vacuum scattering from dust (lambda^(-1)) and
Rayleigh scattering (lambda^(-4)).

## Data

Requires DES-SN5YR Hubble diagram data (included in the parent repository
under `qfd-supernova-v15/data/DES-SN5YR-1.2/`). Multi-band photometry
from Pantheon+SH0ES is used for the chromatic test.

## License

MIT (code). Data citations per DES-SN5YR and Pantheon+ publications.
