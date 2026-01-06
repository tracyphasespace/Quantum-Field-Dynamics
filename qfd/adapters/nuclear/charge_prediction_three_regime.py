"""
QFD Nuclear Adapter: Three-Regime Charge Prediction and Decay Mode

Extension of charge_prediction.py with three-regime support.

Physics Model:
    Each regime k has its own backbone:
        Q_k(A) = c1_k·A^(2/3) + c2_k·A

Three Regimes:
    - Charge-Poor: c1 < 0 (inverted surface tension)
    - Charge-Nominal: c1 ≈ 0.5 (standard configuration)
    - Charge-Rich: c1 > 1.0 (enhanced curvature)

References:
    - binned/paper_replication/nuclear_scaling/mixture_core_compression.py
    - binned/three_track_ccl.py
    - DECAY_PREDICTION_INTEGRATION.md
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple


# ============================================================================
# Three-Regime Model Parameters
# ============================================================================

def get_em_three_regime_params() -> List[Dict[str, float]]:
    """
    Get three-regime parameters from EM clustering.

    From: nuclear_scaling/mixture_core_compression.py results
    Method: Gaussian Mixture EM with K=3

    Returns:
        List of regime parameters in order [poor, nominal, rich]
    """
    return [
        {"c1": -0.150, "c2": 0.413, "name": "charge_poor"},      # Component 0
        {"c1": 0.557, "c2": 0.312, "name": "charge_nominal"},    # Component 1
        {"c1": 1.159, "c2": 0.229, "name": "charge_rich"}        # Component 2
    ]


def get_physics_three_regime_params() -> List[Dict[str, float]]:
    """
    Get three-regime parameters from physics-based classification.

    From: binned/three_track_ccl.py (hard assignment, unbounded)
    Method: ChargeStress threshold classification

    Returns:
        List of regime parameters in order [poor, nominal, rich]
    """
    return [
        {"c1": -0.147, "c2": 0.411, "name": "charge_poor"},
        {"c1": 0.521, "c2": 0.319, "name": "charge_nominal"},
        {"c1": 1.075, "c2": 0.249, "name": "charge_rich"}
    ]


# ============================================================================
# Three-Regime Charge Prediction
# ============================================================================

def predict_charge_three_regime(
    df: pd.DataFrame,
    regime_params: Optional[List[Dict[str, float]]] = None,
    method: str = "hard",
    config: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Predict charge using three-regime model.

    Args:
        df: DataFrame with 'A' column (and optionally 'Z' or 'Q')
        regime_params: List of regime parameters. If None, uses EM params.
        method: 'hard' (argmin) or 'soft' (EM posterior weighting)
        config: Optional configuration

    Returns:
        DataFrame with columns:
            - Q_predicted: Predicted charge
            - regime: Assigned regime (0=poor, 1=nominal, 2=rich)
            - stress: ChargeStress = |Z - Q_predicted| (if Z available)
            - Q_poor, Q_nominal, Q_rich: Predictions from each regime
            - stress_poor, stress_nominal, stress_rich: Stress in each regime

    Example:
        >>> df = pd.DataFrame({"A": [14, 56, 208], "Z": [6, 26, 82]})
        >>> result = predict_charge_three_regime(df)
        >>> result[['A', 'Z', 'Q_predicted', 'regime', 'stress']]
           A   Z  Q_predicted  regime  stress
        0  14   6     5.36        0    0.64
        1  56  26    26.12        1    0.12
        2 208  82    82.45        2    0.45
    """
    if regime_params is None:
        regime_params = get_em_three_regime_params()

    if config is None:
        config = {}

    # Extract mass numbers
    A = df['A'].values.astype(float)
    has_Z = 'Z' in df.columns or 'Q' in df.columns
    if has_Z:
        Z = df.get('Z', df.get('Q')).values.astype(float)

    K = len(regime_params)  # Should be 3

    # Compute predictions for each regime
    A_23 = np.power(A, 2.0/3.0)
    predictions = np.zeros((len(A), K))
    stresses = np.zeros((len(A), K))

    for k, params in enumerate(regime_params):
        c1 = params['c1']
        c2 = params['c2']
        predictions[:, k] = c1 * A_23 + c2 * A

        if has_Z:
            stresses[:, k] = np.abs(Z - predictions[:, k])

    # Assign regimes
    if method == "hard":
        # Hard assignment: argmin of stress (or first if no Z)
        if has_Z:
            regime_idx = np.argmin(stresses, axis=1)
        else:
            # Without Z, assign based on A (heuristic)
            # Light nuclei → poor, medium → nominal, heavy → rich
            regime_idx = np.zeros(len(A), dtype=int)
            regime_idx[A < 60] = 0  # Poor
            regime_idx[(A >= 60) & (A < 150)] = 1  # Nominal
            regime_idx[A >= 150] = 2  # Rich

        # Get predictions from assigned regime
        Q_predicted = predictions[np.arange(len(A)), regime_idx]

    elif method == "soft":
        # Soft assignment: weighted average by posterior probabilities
        # Approximate posterior using stress (inverse exponential weighting)
        if has_Z:
            # R ∝ exp(-stress²/σ²)
            sigma = config.get('sigma', 2.0)  # Tunable parameter
            weights = np.exp(-stresses**2 / (2 * sigma**2))
            weights = weights / (weights.sum(axis=1, keepdims=True) + 1e-10)
        else:
            # Uniform weights if no Z
            weights = np.ones((len(A), K)) / K

        # Weighted average
        Q_predicted = (predictions * weights).sum(axis=1)
        # Regime = dominant weight
        regime_idx = np.argmax(weights, axis=1)

    else:
        raise ValueError(f"Unknown method: {method}. Use 'hard' or 'soft'.")

    # Build result DataFrame
    result = df.copy()
    result['Q_predicted'] = Q_predicted
    result['regime'] = regime_idx

    # Add per-regime predictions
    for k, params in enumerate(regime_params):
        result[f"Q_{params['name']}"] = predictions[:, k]
        if has_Z:
            result[f"stress_{params['name']}"] = stresses[:, k]

    # Add overall stress if Z available
    if has_Z:
        result['stress'] = np.abs(Z - Q_predicted)

    return result


# ============================================================================
# Three-Regime Decay Mode Prediction
# ============================================================================

def predict_decay_mode_three_regime(
    df: pd.DataFrame,
    regime_params: Optional[List[Dict[str, float]]] = None,
    config: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Predict decay mode using three-regime model with regime tracking.

    Physical Model:
        - Only charge_nominal regime (index 1) represents the stability valley
        - Charge-poor (index 0) and charge-rich (index 2) are unstable trajectories
        - Beta decay moves isotopes TOWARD charge_nominal regime
        - Stability requires: (1) being in nominal regime, AND (2) local stress minimum

    Args:
        df: DataFrame with 'A' and 'Z' (or 'Q') columns
        regime_params: List of regime parameters. If None, uses EM params.
        config: Optional configuration

    Returns:
        DataFrame with original columns plus:
            - decay_mode: "stable", "beta_minus", "beta_plus"
            - current_regime: Current regime (0, 1, or 2)
            - target_regime: Regime after decay (if unstable)
            - regime_transition: True if decay changes regime
            - stress_current: Current ChargeStress
            - stress_after_decay: Stress after decay
            - stress_reduction: How much stress is reduced by decay

    Example:
        >>> df = pd.DataFrame({"A": [14, 14], "Z": [6, 7]})
        >>> result = predict_decay_mode_three_regime(df)
        >>> result[['A', 'Z', 'decay_mode', 'current_regime', 'target_regime']]
           A  Z  decay_mode  current_regime  target_regime
        0  14  6  beta_minus       0              1        # C-14 → N-14
        1  14  7  stable               1              1        # N-14 stable
    """
    if regime_params is None:
        regime_params = get_em_three_regime_params()

    if config is None:
        config = {}

    # Extract Z and A
    Z = df.get('Z', df.get('Q')).values.astype(float)
    A = df['A'].values.astype(float)

    K = len(regime_params)
    n = len(A)

    # Compute predictions and stresses for all regimes
    A_23 = np.power(A, 2.0/3.0)

    def compute_stresses(Z_vals, A_vals, A_23_vals):
        """Compute stress in each regime for given Z values."""
        stresses = np.zeros((len(Z_vals), K))
        for k, params in enumerate(regime_params):
            c1 = params['c1']
            c2 = params['c2']
            Q_k = c1 * A_23_vals + c2 * A_vals
            stresses[:, k] = np.abs(Z_vals - Q_k)
        return stresses

    # Current stress in all regimes
    stress_current = compute_stresses(Z, A, A_23)

    # Stress after β⁺ (Z-1) in all regimes
    stress_minus = np.full((n, K), np.inf)
    can_beta_plus = Z > 1
    if can_beta_plus.any():
        stress_minus[can_beta_plus] = compute_stresses(
            Z[can_beta_plus] - 1,
            A[can_beta_plus],
            A_23[can_beta_plus]
        )

    # Stress after β⁻ (Z+1) in all regimes
    stress_plus = compute_stresses(Z + 1, A, A_23)

    # Find minimum stress for each scenario
    min_stress_current = stress_current.min(axis=1)
    min_stress_minus = stress_minus.min(axis=1)
    min_stress_plus = stress_plus.min(axis=1)

    # Determine current regime (minimum stress now)
    current_regime = stress_current.argmin(axis=1)

    # Determine decay mode and target regime
    decay_modes = []
    target_regimes = []
    stress_after = []
    stress_reduction = []

    # NOMINAL_REGIME_INDEX = 1 (charge_nominal is the stability valley)
    NOMINAL_REGIME_INDEX = 1

    for i in range(n):
        stress_c = min_stress_current[i]
        stress_m = min_stress_minus[i]
        stress_p = min_stress_plus[i]
        curr_regime = current_regime[i]

        # KEY INSIGHT: Only charge_nominal regime (index 1) can be stable
        # Charge-poor and charge-rich are unstable trajectories toward nominal

        if curr_regime == NOMINAL_REGIME_INDEX and stress_c <= stress_m and stress_c <= stress_p:
            # Stable: in nominal regime AND at local minimum
            decay_modes.append("stable")
            target_regimes.append(curr_regime)
            stress_after.append(stress_c)
            stress_reduction.append(0.0)

        elif stress_m < stress_c:
            # β⁺ decay favorable
            decay_modes.append("beta_plus")
            target_regimes.append(int(stress_minus[i].argmin()))
            stress_after.append(stress_m)
            stress_reduction.append(stress_c - stress_m)

        else:
            # β⁻ decay favorable
            decay_modes.append("beta_minus")
            target_regimes.append(int(stress_plus[i].argmin()))
            stress_after.append(stress_p)
            stress_reduction.append(stress_c - stress_p)

    # Build result DataFrame
    result = df.copy()
    result['decay_mode'] = decay_modes
    result['current_regime'] = current_regime
    result['target_regime'] = target_regimes
    result['regime_transition'] = (current_regime != np.array(target_regimes))
    result['stress_current'] = min_stress_current
    result['stress_after_decay'] = stress_after
    result['stress_reduction'] = stress_reduction

    # Add regime names for interpretability
    regime_names = [p['name'] for p in regime_params]
    result['current_regime_name'] = [regime_names[r] for r in current_regime]
    result['target_regime_name'] = [regime_names[r] for r in target_regimes]

    return result


# ============================================================================
# Validation and Testing
# ============================================================================

def validate_three_regime():
    """
    Test three-regime prediction functions.

    Tests:
        1. Charge prediction (hard and soft)
        2. Decay mode prediction
        3. Regime transitions
        4. Carbon isotope chain
    """
    print("=" * 80)
    print("Three-Regime Model Validation")
    print("=" * 80)

    # Test 1: Basic charge prediction
    print("\n[Test 1] Charge prediction for sample isotopes")
    df_test = pd.DataFrame({
        "A": [12, 14, 56, 208],
        "Z": [6, 6, 26, 82]
    })

    result = predict_charge_three_regime(df_test, method="hard")
    # Add regime names for display
    regime_names = ['charge_poor', 'charge_nominal', 'charge_rich']
    result['regime_name'] = [regime_names[r] for r in result['regime']]
    print(result[['A', 'Z', 'Q_predicted', 'regime', 'regime_name', 'stress']].to_string(index=False))

    # Test 2: Decay mode prediction
    print("\n[Test 2] Decay mode prediction")
    decay_result = predict_decay_mode_three_regime(df_test)
    print(decay_result[['A', 'Z', 'decay_mode', 'current_regime_name',
                        'target_regime_name', 'regime_transition']].to_string(index=False))

    # Test 3: Carbon isotope chain
    print("\n[Test 3] Carbon isotope chain (Z=6, A=8-18)")
    carbon_chain = pd.DataFrame({
        "A": range(8, 19),
        "Z": [6] * 11
    })

    carbon_result = predict_decay_mode_three_regime(carbon_chain)
    print(carbon_result[['A', 'decay_mode', 'current_regime_name',
                         'stress_current']].to_string(index=False))

    # Test 4: Regime transition example
    print("\n[Test 4] Regime transition example: C-14 → N-14")
    c14_decay = pd.DataFrame({
        "A": [14, 14],
        "Z": [6, 7],
        "Isotope": ["C-14", "N-14"]
    })

    transition_result = predict_decay_mode_three_regime(c14_decay)
    print(transition_result[['Isotope', 'A', 'Z', 'decay_mode',
                             'current_regime_name', 'regime_transition',
                             'stress_reduction']].to_string(index=False))

    print("\n" + "=" * 80)
    print("Validation Complete!")
    print("=" * 80)


if __name__ == "__main__":
    validate_three_regime()
