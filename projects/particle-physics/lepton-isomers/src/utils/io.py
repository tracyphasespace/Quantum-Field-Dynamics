# src/utils/io.py
import json
from pathlib import Path

import numpy as np


def save_results(results, output_path):
    """
    Save results to disk.
    If results is a dictionary, it separates numpy arrays and saves them
    as .npy files, while the rest is saved as a .json file.
    The base name for the json file is taken from output_path.
    """
    output_path = Path(output_path)
    output_dir = output_path.parent
    output_stem = output_path.stem

    if isinstance(results, dict):
        json_results = {}
        for key, value in results.items():
            # Check if value is a numpy array
            if isinstance(value, np.ndarray):
                # Save numpy array as a separate .npy file in the same directory
                # and add a reference to the json
                npy_filename = f"{output_stem}_{key}.npy"
                np.save(output_dir / npy_filename, value)
                json_results[key] = {"npy_file": npy_filename}
            else:
                json_results[key] = value

        # Save the rest of the dictionary as a .json file
        # Add a default handler to convert non-serializable objects to strings
        def default_serializer(o):
            try:
                return str(o)
            except (TypeError, ValueError):
                return f"Non-serializable type: {type(o)}"

        with open(output_path.with_suffix(".json"), "w") as f:
            json.dump(json_results, f, indent=4, default=default_serializer)

    elif isinstance(results, np.ndarray):
        np.save(output_path.with_suffix(".npy"), results)
    else:
        raise ValueError(f"Unsupported results type: {type(results)}")


# --- SURGICAL FIX: Robust particle constants loader ---

from typing import Dict, Any

def _coerce_numbers(d):
    """Coerce string numbers to proper int/float types"""
    INT_KEYS = {
        "num_radial_points", "nr", "max_iters", "save_every", "log_every",
        "warm_start_every", "rungs", "seed"
    }
    FLOAT_KEYS = {
        "r_max", "dt_max", "V2", "V4", "g_c", "k_csr", "mass_eV", "q_star",
        "target_energy_eV", "tolerance", "beta_cap", "step_scale", "q_star_rms_internal"
    }
    if isinstance(d, dict):
        out = {}
        for k, v in d.items():
            if isinstance(v, str):
                if k in INT_KEYS:
                    try: v = int(float(v))
                    except: pass
                elif k in FLOAT_KEYS:
                    try: v = float(v)
                    except: pass
            out[k] = _coerce_numbers(v)
        return out
    elif isinstance(d, list):
        return [_coerce_numbers(x) for x in d]
    return d

def load_particle_constants(species: str, search_dirs=None) -> Dict[str, Any]:
    """
    Robustly load particle constants JSON for 'electron' | 'muon' | 'tau'.
    Looks in common locations so we don't rely on fragile Python modules.
    """
    from pathlib import Path
    import json

    if search_dirs is None:
        # Most common layouts
        search_dirs = [
            Path.cwd() / "src" / "constants",
            Path.cwd() / "constants",
            Path.cwd(),  # fall back to cwd
        ]

    candidate_names = [
        f"{species}.json",
        f"particle_{species}.json",
    ]

    for d in search_dirs:
        for name in candidate_names:
            p = d / name
            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    constants = json.load(f)
                    # Apply type coercion to prevent string->int errors
                    constants = _coerce_numbers(constants)
                    return constants

    raise FileNotFoundError(
        f"Could not find constants for {species}. "
        f"Searched {search_dirs} for {candidate_names}"
    )
