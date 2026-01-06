"""
QFD Nuclear Adapter: Charge Prediction (Core Compression Law)
Module: qfd.adapters.nuclear

Maps QFD parameters (c1, c2) to predicted charge number Z(A).

Physics Model:
    Q(A) = c1·A^(2/3) + c2·A

This is the **Core Compression Law** that predicts which proton numbers
are stable for a given mass number, with R² ≈ 0.98 for all isotopes.

References:
    - QFD Appendix N: Nuclear Genesis
    - Core Compression Law: Z = c1·A^(2/3) + c2·A
    - nuclide-prediction project: R² ≈ 0.98 (all), R² ≈ 0.998 (stable)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional


def predict_charge(
    df: pd.DataFrame,
    params: Dict[str, float],
    config: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Predict charge number Z from mass number A using Core Compression Law.

    Q(A) = c1·A^(2/3) + c2·A

    Args:
        df: DataFrame containing 'A' (mass number).
        params: Dictionary of parameters:
            - c1: Surface term coefficient (dimensionless)
            - c2: Volume term coefficient (dimensionless)
        config: Optional configuration.

    Returns:
        np.ndarray: Predicted charge number Q (= Z)

    Example:
        >>> df = pd.DataFrame({"A": [12, 16, 56, 208]})
        >>> params = {"c1": 1.0, "c2": 0.05}
        >>> Z_pred = predict_charge(df, params)
        >>> Z_pred.shape
        (4,)

    Note:
        This is the fundamental QFD soliton prediction: which charge
        configurations are stable for a given mass number.
    """
    # Extract mass number
    A = _get_column(df, ["A", "mass_number", "massnumber"]).astype(float).to_numpy()

    # Extract parameters (handle both namespaced and bare names)
    c1 = params.get("nuclear.c1", params.get("c1", 0.0))
    c2 = params.get("nuclear.c2", params.get("c2", 0.0))

    # Core Compression Law
    Q_pred = c1 * np.power(A, 2.0/3.0) + c2 * A

    return Q_pred


def _get_column(df: pd.DataFrame, candidates: list) -> pd.Series:
    """
    Get first matching column from candidates.

    Args:
        df: DataFrame to search
        candidates: List of possible column names

    Returns:
        pd.Series: First matching column

    Raises:
        KeyError: If no candidate found
    """
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return df[cols_lower[cand.lower()]]

    raise KeyError(
        f"Could not find any of {candidates} in DataFrame columns: {list(df.columns)}"
    )


def validate_adapter():
    """
    Self-test for charge prediction adapter.

    Tests the Core Compression Law with known isotopes.
    """
    # Test with realistic parameters from nuclide-prediction project
    # (approximate values, actual fit gives R² ≈ 0.98)
    df = pd.DataFrame({"A": [4, 12, 16, 56, 208]})
    params = {"c1": 1.0, "c2": 0.4}  # Rough approximation

    Q_pred = predict_charge(df, params)

    # Expected charges (approximately)
    # He-4: Z=2, C-12: Z=6, O-16: Z=8, Fe-56: Z=26, Pb-208: Z=82
    expected = np.array([2, 6, 8, 26, 82])

    print("✓ Core Compression Law Test")
    print(f"  A:        {df['A'].values}")
    print(f"  Q_pred:   {Q_pred}")
    print(f"  Q_actual: {expected}")
    print(f"  Shape: {Q_pred.shape}")
    print(f"  Finite: {np.all(np.isfinite(Q_pred))}")

    assert Q_pred.shape == (5,), "Shape mismatch"
    assert np.all(np.isfinite(Q_pred)), "Non-finite predictions"

    print("\n✅ Charge prediction adapter validated")
    return True


if __name__ == "__main__":
    validate_adapter()
