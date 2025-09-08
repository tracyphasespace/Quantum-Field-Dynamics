# src/features/csr_handle.py
from dataclasses import dataclass
from typing import Dict, Any, Tuple

@dataclass
class CSRFeatures:
    Hcsr: float       # scalar CSR handle
    hcsr_over_I: float

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

def compute_csr_features(bundle: Dict[str, Any]) -> CSRFeatures:
    """
    Expects bundle['features'] to contain I and necessary CSR state.
    Define Hcsr consistently with prior docs; return hcsr_over_I = Hcsr / I.
    """
    # Assuming exit_code 8 for feature extraction errors
    I, _ = extract_feature(bundle, "I", "I_final", 8)
    Hcsr, _ = extract_feature(bundle, "Hcsr", "Hcsr_final", 8)

    if I == 0:
        # Handle division by zero, perhaps raise an error or return a default
        # For now, let's raise an error as it indicates an issue with the bundle data
        raise ValueError("Cannot compute hcsr_over_I: I is zero.", 8)

    hcsr_over_I = Hcsr / I
    return CSRFeatures(Hcsr=Hcsr, hcsr_over_I=hcsr_over_I)

