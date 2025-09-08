# src/utils/io.py
import json
import numpy as np
from pathlib import Path

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
