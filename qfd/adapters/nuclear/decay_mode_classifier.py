"""
Multi-Mode Nuclear Decay Classifier

Predicts decay modes (beta, alpha, fission, exotic) based on three-regime curve positions.

Based on analysis of NuBase 2020 data showing:
- charge_nominal curve is the stability valley (95.36% beta accuracy)
- Decay modes occupy specific zones relative to curves
- Alpha decay: heavy nuclei, +1 Z above nominal
- Fission: superheavy nuclei, ON nominal curve
- Exotic modes: extremely proton-rich, +2 to +6 Z above nominal

Author: QFD Nuclear Physics Team
Date: 2025-12-29
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, Dict, List
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from qfd.adapters.nuclear.charge_prediction_three_regime import get_em_three_regime_params


class DecayModeClassifier:
    """
    Multi-mode nuclear decay classifier using three-regime curves.

    Predicts:
    - stable
    - beta_minus (n → p)
    - beta_plus_ec (p → n, includes β⁺ and electron capture)
    - alpha (He-4 emission)
    - fission (spontaneous fission)
    - proton_emission
    - other_exotic
    """

    def __init__(self, regime_params=None, config=None):
        """
        Initialize classifier.

        Parameters
        ----------
        regime_params : list of dict, optional
            Three-regime parameters. If None, uses EM-fitted values.
        config : dict, optional
            Configuration parameters for thresholds.
        """
        if regime_params is None:
            regime_params = get_em_three_regime_params()

        self.regime_params = regime_params

        # Extract curve parameters
        self.c1_poor = regime_params[0]['c1']
        self.c2_poor = regime_params[0]['c2']
        self.c1_nominal = regime_params[1]['c1']
        self.c2_nominal = regime_params[1]['c2']
        self.c1_rich = regime_params[2]['c1']
        self.c2_rich = regime_params[2]['c2']

        # Default configuration (empirically determined from NuBase 2020)
        self.config = {
            # Mass thresholds
            'alpha_mass_threshold': 200,      # A > 200 → consider alpha
            'superheavy_mass_threshold': 220, # A > 220 → consider fission
            'light_mass_threshold': 50,       # A < 50 → exotic modes possible

            # Distance thresholds (Z units) - tuned from data
            'stable_tolerance': 0.8,          # |dist_nominal| < 0.8 → stable
            'beta_minus_threshold': -0.8,     # dist_nominal < -0.8 → beta_minus
            'beta_plus_threshold': 0.8,       # dist_nominal > +0.8 → beta_plus
            'extreme_proton_rich': 5.0,       # dist_nominal > 5 → exotic

            # Alpha decay zone (heavy nuclei only, A > 200)
            # Alpha has wide dist_nominal range (-3.7 to +9.9), mean +1.0, std 3.4
            # Cannot separate from beta_plus by distance alone
            'alpha_prob_min_dist': -2.0,      # Alpha possible if > -2.0
            'alpha_prob_max_dist': 8.0,       # Alpha possible if < +8.0

            # Fission: A > 220 overrides everything
            # Fission has mean -0.2, std 2.2, but overlaps with all other modes
        }

        if config:
            self.config.update(config)

    def calculate_distances(self, A: np.ndarray, Z: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate signed distances to all three curves.

        Parameters
        ----------
        A : np.ndarray
            Mass numbers
        Z : np.ndarray
            Proton numbers

        Returns
        -------
        dict
            Dictionary with keys 'poor', 'nominal', 'rich' containing distance arrays
        """
        A_23 = np.power(A, 2.0/3.0)

        Q_poor = self.c1_poor * A_23 + self.c2_poor * A
        Q_nominal = self.c1_nominal * A_23 + self.c2_nominal * A
        Q_rich = self.c1_rich * A_23 + self.c2_rich * A

        return {
            'poor': Z - Q_poor,
            'nominal': Z - Q_nominal,
            'rich': Z - Q_rich
        }

    def predict_single(self, A: int, Z: int) -> Dict[str, Union[str, float, dict]]:
        """
        Predict decay mode for a single isotope with detailed output.

        Parameters
        ----------
        A : int
            Mass number
        Z : int
            Proton number

        Returns
        -------
        dict
            {
                'A': mass number,
                'Z': proton number,
                'decay_mode': predicted mode,
                'confidence': 'high', 'medium', or 'low',
                'reason': explanation string,
                'distances': {poor, nominal, rich},
                'zone': zone classification
            }
        """
        A_arr = np.array([A])
        Z_arr = np.array([Z])

        distances = self.calculate_distances(A_arr, Z_arr)
        dist_poor = distances['poor'][0]
        dist_nominal = distances['nominal'][0]
        dist_rich = distances['rich'][0]

        # Decision cascade
        result = self._classify_isotope(A, Z, dist_poor, dist_nominal, dist_rich)

        result['A'] = A
        result['Z'] = Z
        result['distances'] = {
            'poor': float(dist_poor),
            'nominal': float(dist_nominal),
            'rich': float(dist_rich)
        }

        return result

    def _classify_isotope(self, A: int, Z: int, dist_poor: float,
                         dist_nominal: float, dist_rich: float) -> dict:
        """
        Core classification logic using empirically-tuned thresholds.

        Decision cascade:
        1. Superheavy (A > 220) → primarily fission
        2. Heavy (A > 200) → alpha decay competes with beta/fission
        3. Standard mass → beta decay based on dist_nominal
        4. Light (A < 50) + extreme proton → exotic modes
        5. Stable detection requires tight bracket
        """

        # STEP 1: Superheavy nuclei (A > 220) - fission dominates
        if A >= self.config['superheavy_mass_threshold']:
            # Fission is primary decay mode for superheavy
            # But beta decay still possible if far from nominal
            if abs(dist_nominal) > 4.0:
                # Very far from stability → beta decay
                if dist_nominal > 0:
                    return {
                        'decay_mode': 'beta_plus_ec',
                        'confidence': 'medium',
                        'reason': f'Superheavy but very proton-rich (dist={dist_nominal:+.2f} Z)',
                        'zone': 'superheavy_beta_plus'
                    }
                else:
                    return {
                        'decay_mode': 'beta_minus',
                        'confidence': 'medium',
                        'reason': f'Superheavy but very neutron-rich (dist={dist_nominal:+.2f} Z)',
                        'zone': 'superheavy_beta_minus'
                    }
            else:
                # Near stability → fission or stable
                if abs(dist_nominal) < self.config['stable_tolerance']:
                    # Very close → could be stable
                    if dist_poor > 0 and dist_rich < 0:
                        return {
                            'decay_mode': 'stable',
                            'confidence': 'medium',
                            'reason': f'Superheavy on stability valley (dist={dist_nominal:+.2f} Z)',
                            'zone': 'superheavy_stable'
                        }
                # Default for superheavy near stability
                return {
                    'decay_mode': 'fission',
                    'confidence': 'high',
                    'reason': f'Superheavy nucleus (A={A})',
                    'zone': 'superheavy_fission'
                }

        # STEP 2: Check for stable isotopes (must be tightly bracketed)
        if abs(dist_nominal) < self.config['stable_tolerance']:
            # Must be bracketed by poor and rich curves
            if dist_poor > 0 and dist_rich < 0:
                return {
                    'decay_mode': 'stable',
                    'confidence': 'high',
                    'reason': f'On stability valley (dist={dist_nominal:+.2f} Z), bracketed',
                    'zone': 'stable_valley'
                }

        # STEP 3: Extreme proton-rich (exotic modes)
        if dist_nominal > self.config['extreme_proton_rich']:
            if A < self.config['light_mass_threshold']:
                return {
                    'decay_mode': 'proton_emission',
                    'confidence': 'high',
                    'reason': f'Light nucleus, extreme proton excess (dist={dist_nominal:+.2f} Z)',
                    'zone': 'proton_drip_line'
                }
            else:
                return {
                    'decay_mode': 'other_exotic',
                    'confidence': 'medium',
                    'reason': f'Extreme proton excess (dist={dist_nominal:+.2f} Z)',
                    'zone': 'extreme_proton_rich'
                }

        # STEP 4: Heavy nuclei (A > 200) - alpha decay region
        # Alpha competes with beta_plus in heavy nuclei
        # Empirical data: alpha has mean dist_nominal = +1.0, std = 3.4
        # Beta_plus has mean dist_nominal = +3.2, std = 2.4
        # Cannot reliably separate, so predict both as possible

        if A >= self.config['alpha_mass_threshold']:
            # Heavy nuclei - check if in alpha zone
            if (self.config['alpha_prob_min_dist'] < dist_nominal < self.config['alpha_prob_max_dist']):
                # Alpha is primary for heavy nuclei with moderate proton excess
                if -1.0 < dist_nominal < 3.0:
                    return {
                        'decay_mode': 'alpha',
                        'confidence': 'high',
                        'reason': f'Heavy nucleus (A={A}) in alpha zone (dist={dist_nominal:+.2f} Z)',
                        'zone': 'alpha_decay'
                    }
                elif dist_nominal >= 3.0:
                    # High proton excess → beta_plus competes
                    return {
                        'decay_mode': 'beta_plus_ec',
                        'confidence': 'medium',
                        'reason': f'Heavy nucleus, high proton excess (dist={dist_nominal:+.2f} Z)',
                        'zone': 'heavy_beta_plus'
                    }
                else:
                    # dist_nominal < -1.0 → neutron-rich heavy
                    return {
                        'decay_mode': 'beta_minus',
                        'confidence': 'high',
                        'reason': f'Heavy nucleus, neutron-rich (dist={dist_nominal:+.2f} Z)',
                        'zone': 'heavy_beta_minus'
                    }

        # STEP 5: Standard beta decay prediction (light to medium mass)
        if dist_nominal < self.config['beta_minus_threshold']:
            # Neutron-rich → beta-minus
            return {
                'decay_mode': 'beta_minus',
                'confidence': 'high',
                'reason': f'Neutron-rich, {-dist_nominal:.2f} Z below stability',
                'zone': 'beta_minus'
            }

        elif dist_nominal > self.config['beta_plus_threshold']:
            # Proton-rich → beta-plus or EC
            return {
                'decay_mode': 'beta_plus_ec',
                'confidence': 'high',
                'reason': f'Proton-rich, {dist_nominal:.2f} Z above stability',
                'zone': 'beta_plus'
            }

        else:
            # In ambiguous zone (-0.8 < dist_nominal < +0.8)
            # Weak signal - predict based on slight preference
            if dist_nominal > 0:
                return {
                    'decay_mode': 'beta_plus_ec',
                    'confidence': 'low',
                    'reason': f'Weak proton excess (dist={dist_nominal:+.2f} Z)',
                    'zone': 'weak_beta_plus'
                }
            else:
                return {
                    'decay_mode': 'beta_minus',
                    'confidence': 'low',
                    'reason': f'Weak neutron excess (dist={dist_nominal:+.2f} Z)',
                    'zone': 'weak_beta_minus'
                }

    def predict(self, A: Union[int, np.ndarray, pd.Series],
                Z: Union[int, np.ndarray, pd.Series],
                detailed: bool = False) -> Union[pd.DataFrame, np.ndarray]:
        """
        Predict decay modes for one or more isotopes.

        Parameters
        ----------
        A : int, array-like
            Mass number(s)
        Z : int, array-like
            Proton number(s)
        detailed : bool, default False
            If True, return DataFrame with full details.
            If False, return array of decay mode strings.

        Returns
        -------
        pd.DataFrame or np.ndarray
            Predictions with optional details
        """
        # Convert to arrays
        if isinstance(A, (int, float)):
            A = np.array([A])
            Z = np.array([Z])
        else:
            A = np.asarray(A)
            Z = np.asarray(Z)

        # Calculate distances
        distances = self.calculate_distances(A, Z)

        # Classify each isotope
        results = []
        for i in range(len(A)):
            result = self._classify_isotope(
                A[i], Z[i],
                distances['poor'][i],
                distances['nominal'][i],
                distances['rich'][i]
            )

            if detailed:
                result.update({
                    'A': int(A[i]),
                    'Z': int(Z[i]),
                    'dist_poor': float(distances['poor'][i]),
                    'dist_nominal': float(distances['nominal'][i]),
                    'dist_rich': float(distances['rich'][i])
                })

            results.append(result)

        if detailed:
            return pd.DataFrame(results)
        else:
            return np.array([r['decay_mode'] for r in results])

    def get_confidence_score(self, A: int, Z: int) -> float:
        """
        Get numerical confidence score (0-1) for prediction.

        Based on distance from zone boundaries.

        Parameters
        ----------
        A : int
            Mass number
        Z : int
            Proton number

        Returns
        -------
        float
            Confidence score between 0 and 1
        """
        result = self.predict_single(A, Z)

        # Map confidence levels to scores
        confidence_map = {
            'high': 0.9,
            'medium': 0.7,
            'low': 0.5
        }

        base_score = confidence_map.get(result['confidence'], 0.5)

        # Adjust based on distance from stability valley
        dist_nominal = abs(result['distances']['nominal'])

        # Isotopes very far from nominal have higher confidence
        if dist_nominal > 5.0:
            adjustment = 0.1
        elif dist_nominal > 3.0:
            adjustment = 0.05
        elif dist_nominal < 0.5:
            adjustment = -0.1  # Lower confidence near boundaries
        else:
            adjustment = 0.0

        return min(1.0, max(0.0, base_score + adjustment))


def predict_decay_modes(A, Z, detailed=False, config=None):
    """
    Convenience function for quick predictions.

    Parameters
    ----------
    A : int or array-like
        Mass number(s)
    Z : int or array-like
        Proton number(s)
    detailed : bool, default False
        Return detailed DataFrame
    config : dict, optional
        Custom configuration

    Returns
    -------
    pd.DataFrame or np.ndarray
        Decay mode predictions

    Examples
    --------
    >>> # Single isotope
    >>> predict_decay_modes(14, 6)
    array(['beta_plus_ec'], dtype='<U16')

    >>> # Multiple isotopes
    >>> predict_decay_modes([14, 14, 60], [6, 8, 27])
    array(['beta_plus_ec', 'stable', 'stable'], dtype='<U16')

    >>> # Detailed output
    >>> predict_decay_modes(238, 92, detailed=True)
         A   Z   decay_mode confidence  ...
    0  238  92        alpha       high  ...
    """
    classifier = DecayModeClassifier(config=config)
    return classifier.predict(A, Z, detailed=detailed)


if __name__ == "__main__":
    # Test the classifier
    print("="*80)
    print("MULTI-MODE DECAY CLASSIFIER - TEST")
    print("="*80)

    # Test cases covering different zones
    test_cases = [
        # (A, Z, expected_mode, description)
        (4, 2, 'stable', 'He-4 (doubly magic)'),
        (14, 6, 'beta_plus_ec', 'C-14 (proton-rich)'),
        (14, 8, 'stable', 'O-14 would be N-14 (typo), testing'),
        (3, 1, 'beta_minus', 'H-3 (tritium)'),
        (238, 92, 'alpha', 'U-238 (heavy, alpha)'),
        (252, 98, 'fission', 'Cf-252 (fission)'),
        (60, 27, 'stable', 'Co-60 region'),
        (8, 5, 'proton_emission', 'B-8 (very proton-rich)'),
        (208, 82, 'stable', 'Pb-208 (doubly magic)'),
        (294, 118, 'fission', 'Og-294 (superheavy)'),
    ]

    classifier = DecayModeClassifier()

    print("\nTest Cases:\n")
    for A, Z, expected, description in test_cases:
        result = classifier.predict_single(A, Z)
        match = "✓" if result['decay_mode'] == expected else "✗"

        print(f"{match} A={A:3d} Z={Z:3d} | {description:30s}")
        print(f"  Predicted: {result['decay_mode']:20s} (confidence: {result['confidence']})")
        print(f"  Expected:  {expected:20s}")
        print(f"  Reason: {result['reason']}")
        print(f"  Distances: poor={result['distances']['poor']:+6.2f}, "
              f"nominal={result['distances']['nominal']:+6.2f}, "
              f"rich={result['distances']['rich']:+6.2f}")
        print()

    print("="*80)
    print("BATCH PREDICTION TEST")
    print("="*80)

    # Test batch prediction
    A_batch = np.array([4, 14, 3, 238, 252, 60, 208])
    Z_batch = np.array([2, 6, 1, 92, 98, 27, 82])

    results_simple = classifier.predict(A_batch, Z_batch, detailed=False)
    print("\nSimple predictions:")
    for i, (a, z, mode) in enumerate(zip(A_batch, Z_batch, results_simple)):
        print(f"  {i+1}. A={a:3d} Z={z:3d} → {mode}")

    print("\nDetailed predictions:")
    results_detailed = classifier.predict(A_batch, Z_batch, detailed=True)
    print(results_detailed[['A', 'Z', 'decay_mode', 'confidence', 'zone']].to_string())

    print("\n" + "="*80)
    print("Testing complete!")
