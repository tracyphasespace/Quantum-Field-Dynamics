#!/usr/bin/env python3
"""
Nucleus Classification using 3-Family Harmonic Resonance Model

This module classifies nuclei into three families (A, B, C) based on
geometric quantization of the nuclear binding energy.

Model: Z = c1·A^(2/3) + c2·A + c3·ω(N,A)

Where:
  - c1: Volume coefficient
  - c2: Surface coefficient
  - c3: Resonance frequency coefficient (≈ -0.865 MeV, universal)
  - N: Harmonic mode quantum number (discrete)

Families:
  - Family A: Volume-dominated (c2/c1 = 0.26), N ∈ {-3, +3}
  - Family B: Surface-dominated (c2/c1 = 0.12), N ∈ {-3, +3}
  - Family C: Neutron-rich high modes (c2/c1 = 0.20), N ∈ {4, +10}
"""

# 3-Family Model Parameters
# Format: [c1_0, c2_0, c3_0, dc1, dc2, dc3]
# where c_i = c_i0 + N * dc_i

PARAMS_A = [0.9618, 0.2475, -2.4107, -0.0295, 0.0064, -0.8653]
PARAMS_B = [1.473890, 0.172746, 0.502666, -0.025915, 0.004164, -0.865483]
PARAMS_C = [1.169611, 0.232621, -4.467213, -0.043412, 0.004986, -0.512975]


def classify_nucleus(A, Z):
    """
    Classify a nucleus using the 3-family harmonic resonance model.

    Parameters
    ----------
    A : int
        Mass number (number of nucleons)
    Z : int
        Atomic number (number of protons)

    Returns
    -------
    N : int or None
        Harmonic mode quantum number if classified, None otherwise
    family : str or None
        Family label ('A', 'B', or 'C') if classified, None otherwise

    Examples
    --------
    >>> classify_nucleus(4, 2)  # He-4
    (-1, 'A')

    >>> classify_nucleus(238, 92)  # U-238
    (2, 'A')

    >>> classify_nucleus(14, 6)  # C-14
    (1, 'A')
    """
    # Try each family in order
    for params, N_min, N_max, family in [
        (PARAMS_A, -3, 3, 'A'),
        (PARAMS_B, -3, 3, 'B'),
        (PARAMS_C, 4, 10, 'C')
    ]:
        c1_0, c2_0, c3_0, dc1, dc2, dc3 = params

        # Search through all harmonic modes for this family
        for N in range(N_min, N_max + 1):
            # Calculate coefficients for this mode
            c1 = c1_0 + N * dc1
            c2 = c2_0 + N * dc2
            c3 = c3_0 + N * dc3

            # Predict Z from A using the geometric formula
            Z_pred = c1 * (A**(2.0/3.0)) + c2 * A + c3

            # Check if prediction matches (within rounding)
            if int(round(Z_pred)) == Z:
                return N, family

    # Nucleus not classified by any family
    return None, None


def get_family_info(family):
    """
    Get information about a nuclear family.

    Parameters
    ----------
    family : str
        Family label ('A', 'B', or 'C')

    Returns
    -------
    info : dict
        Dictionary with keys: 'name', 'description', 'N_range', 'params'
    """
    families = {
        'A': {
            'name': 'Volume-dominated',
            'description': 'Most stable nuclei, c2/c1 = 0.26',
            'N_range': (-3, 3),
            'params': PARAMS_A
        },
        'B': {
            'name': 'Surface-dominated',
            'description': 'Neutron-deficient, c2/c1 = 0.12',
            'N_range': (-3, 3),
            'params': PARAMS_B
        },
        'C': {
            'name': 'Neutron-rich high modes',
            'description': 'High harmonic modes, c2/c1 = 0.20',
            'N_range': (4, 10),
            'params': PARAMS_C
        }
    }
    return families.get(family)


if __name__ == '__main__':
    # Test classification on some well-known nuclei
    test_cases = [
        (1, 1, 'H-1'),
        (4, 2, 'He-4'),
        (12, 6, 'C-12'),
        (14, 6, 'C-14'),
        (16, 8, 'O-16'),
        (56, 26, 'Fe-56'),
        (208, 82, 'Pb-208'),
        (238, 92, 'U-238'),
    ]

    print("Testing nucleus classification:")
    print(f"{'Nucleus':<10} {'A':<5} {'Z':<5} {'N mode':<10} {'Family':<10}")
    print("-" * 50)

    for A, Z, name in test_cases:
        N, family = classify_nucleus(A, Z)
        N_str = str(N) if N is not None else 'None'
        family_str = family if family is not None else 'None'
        print(f"{name:<10} {A:<5} {Z:<5} {N_str:<10} {family_str:<10}")
