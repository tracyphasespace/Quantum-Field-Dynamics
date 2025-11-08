#!/usr/bin/env python3
"""
Generate lock files for reproducibility:
- environment.lock: Python package versions
- data.lock: Dataset hashes and metadata

Usage:
    python make_locks.py --env      # Generate environment.lock
    python make_locks.py --data     # Generate data.lock
    python make_locks.py --all      # Generate both
"""

import sys
import json
import hashlib
import argparse
from pathlib import Path
from datetime import datetime

def get_package_versions():
    """Get versions of key packages."""
    versions = {}

    try:
        import numpy
        versions['numpy'] = numpy.__version__
    except ImportError:
        versions['numpy'] = 'not installed'

    try:
        import scipy
        versions['scipy'] = scipy.__version__
    except ImportError:
        versions['scipy'] = 'not installed'

    try:
        import matplotlib
        versions['matplotlib'] = matplotlib.__version__
    except ImportError:
        versions['matplotlib'] = 'not installed'

    try:
        import pandas
        versions['pandas'] = pandas.__version__
    except ImportError:
        versions['pandas'] = 'not installed'

    try:
        import jax
        versions['jax'] = jax.__version__
    except ImportError:
        versions['jax'] = 'not installed'

    try:
        import numpyro
        versions['numpyro'] = numpyro.__version__
    except ImportError:
        versions['numpyro'] = 'not installed'

    return versions

def compute_file_hash(filepath):
    """Compute SHA256 hash of file."""
    sha256 = hashlib.sha256()
    try:
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()
    except FileNotFoundError:
        return None

def generate_environment_lock():
    """Generate environment.lock with package versions."""
    print("Generating environment.lock...")

    versions = get_package_versions()

    # Get Python version
    import sys
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    # Try to get git SHA
    git_sha = None
    try:
        import subprocess
        git_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                         stderr=subprocess.DEVNULL).decode().strip()
    except:
        pass

    lock_data = {
        'generated': datetime.now().isoformat(),
        'python_version': python_version,
        'git_sha': git_sha,
        'packages': versions,
    }

    with open('environment.lock', 'w') as f:
        json.dump(lock_data, f, indent=2)

    print("✓ environment.lock created")
    print(f"  Python: {python_version}")
    for pkg, ver in versions.items():
        print(f"  {pkg}: {ver}")

def generate_data_lock():
    """Generate data.lock with dataset hashes."""
    print("Generating data.lock...")

    # Find data files
    data_dir = Path("..") / "data"
    results_dir = Path("..") / "results" / "v15_production"

    data_files = {}

    # Look for lightcurves
    lightcurve_file = data_dir / "lightcurves_unified_v2_min3.csv"
    if lightcurve_file.exists():
        print(f"  Hashing {lightcurve_file.name}...")
        data_files['lightcurves'] = {
            'path': str(lightcurve_file),
            'size': lightcurve_file.stat().st_size,
            'sha256': compute_file_hash(lightcurve_file),
        }

    # Look for Stage 3 results
    stage3_file = results_dir / "stage3" / "stage3_results.csv"
    if not stage3_file.exists():
        stage3_file = results_dir / "stage3" / "hubble_data.csv"

    if stage3_file.exists():
        print(f"  Hashing {stage3_file.name}...")
        data_files['stage3_results'] = {
            'path': str(stage3_file),
            'size': stage3_file.stat().st_size,
            'sha256': compute_file_hash(stage3_file),
        }

    # Look for Stage 2 best fit
    stage2_file = results_dir / "stage2" / "best_fit.json"
    if stage2_file.exists():
        with open(stage2_file) as f:
            best_fit = json.load(f)

        data_files['stage2_best_fit'] = {
            'path': str(stage2_file),
            'parameters': best_fit,
        }

    # Count SNe if available
    n_sne = None
    z_range = None
    if stage3_file.exists():
        import pandas as pd
        df = pd.read_csv(stage3_file)
        n_sne = len(df)
        z_range = [float(df['z'].min()), float(df['z'].max())]

    lock_data = {
        'generated': datetime.now().isoformat(),
        'data_files': data_files,
        'dataset_metadata': {
            'n_sne': n_sne,
            'z_range': z_range,
        }
    }

    with open('data.lock', 'w') as f:
        json.dump(lock_data, f, indent=2)

    print("✓ data.lock created")
    if n_sne:
        print(f"  SNe count: {n_sne}")
    if z_range:
        print(f"  z range: [{z_range[0]:.3f}, {z_range[1]:.3f}]")
    print(f"  Data files hashed: {len(data_files)}")

def main():
    parser = argparse.ArgumentParser(
        description='Generate lock files for reproducibility')
    parser.add_argument('--env', action='store_true',
                       help='Generate environment.lock')
    parser.add_argument('--data', action='store_true',
                       help='Generate data.lock')
    parser.add_argument('--all', action='store_true',
                       help='Generate all lock files')

    args = parser.parse_args()

    # Default to --all if no flags
    if not (args.env or args.data or args.all):
        args.all = True

    if args.env or args.all:
        generate_environment_lock()
        print()

    if args.data or args.all:
        generate_data_lock()
        print()

    print("Lock file generation complete!")

if __name__ == '__main__':
    main()
