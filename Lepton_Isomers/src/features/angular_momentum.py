# src/features/angular_momentum.py
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

@dataclass
class LFeatures:
    L_mag: float                  # ||L||
    LdotQ_over_I: Optional[float] # (L·Q)/I if Q available; else None

# Replicated extract_feature for now
def extract_feature(bundle: Dict[str, Any],
                    key: str,
                    summary_key: str,
                    exit_code: int) -> Tuple[float, str]:
    """
    Extract scalar from summary first, then extras. Raise ValueError(message, exit_code) if missing/non-numeric.
    """
    source = "missing"
    val = bundle.get("summary", {}).get(summary_key)
    if val is not None:
        source = "summary"
    else:
        val = bundle.get("features", {}).get(key)
        if val is not None:
            source = "features"

    if val is None:
        raise ValueError(f"Feature '{key}' not found in summary('{summary_key}') or extras.", exit_code)

    try:
        return float(val), source
    except Exception:
        raise ValueError(f"Feature '{key}' is not numeric (got: {val!r}).", exit_code)

def compute_L_features(bundle: Dict[str, Any]) -> LFeatures:
    """
    Expects bundle['features'] to contain I and Q (vector or magnitude),
    and bundle['state'] or equivalent to provide angular momentum (vector L).
    Returns L_mag and optional LdotQ_over_I.
    """
    # TODO: L_vec is not directly available from summary.json or bundle_manifest.json.
    # It is likely contained within the .npy files (psi_field, psi_b_field).
    # For now, L_mag will be 0.0 and LdotQ_over_I will be None.
    # This function assumes L_vec will be provided in the bundle['state'] or similar in the future.

    I, _ = extract_feature(bundle, "I", "I_final", 8)
    Q, _ = extract_feature(bundle, "Q", "Q_proxy_final", 8) # Q is scalar here

    L_mag = 0.0 # Placeholder
    LdotQ_over_I = None # Placeholder

    # If L_vec becomes available, the following logic would apply:
    # L_vec = bundle.get("state", {}).get("L_vec") # Assuming L_vec is a list/tuple of floats
    # if L_vec:
    #     L_mag = (sum(x**2 for x in L_vec))**0.5
    #     if I != 0:
    #         # Assuming Q_vec is also available if L_vec is
    #         Q_vec = bundle.get("state", {}).get("Q_vec")
    #         if Q_vec and len(L_vec) == len(Q_vec):
    #             dot_product = sum(l * q for l, q in zip(L_vec, Q_vec))
    #             LdotQ_over_I = dot_product / I

    return LFeatures(L_mag=L_mag, LdotQ_over_I=LdotQ_over_I)

