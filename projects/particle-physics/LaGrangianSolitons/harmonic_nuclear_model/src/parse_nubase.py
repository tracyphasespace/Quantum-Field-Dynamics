#!/usr/bin/env python3
"""
Parse NUBASE2020 raw text file into structured Parquet dataset.

Extracts:
  - A, Z, N (nuclide identity)
  - is_stable (bool)
  - half_life_s (float, NaN for stable)
  - dominant_mode (categorical)
  - branching ratios
  - mass excess, excitation energy

Implements EXPERIMENT_PLAN.md §2 data pipeline.
"""

import re
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# Half-life unit conversions to seconds
HALFLIFE_UNITS = {
    'ys': 1e-24,  # yoctosecond
    'zs': 1e-21,  # zeptosecond
    'as': 1e-18,  # attosecond
    'fs': 1e-15,  # femtosecond
    'ps': 1e-12,  # picosecond
    'ns': 1e-9,   # nanosecond
    'us': 1e-6,   # microsecond
    'ms': 1e-3,   # millisecond
    's':  1.0,    # second
    'm':  60.0,   # minute
    'h':  3600.0, # hour
    'd':  86400.0,           # day
    'y':  31557600.0,        # year (365.25 days)
    'ky': 31557600.0 * 1e3,  # kiloyear
    'My': 31557600.0 * 1e6,  # megayear
    'Gy': 31557600.0 * 1e9,  # gigayear
    'Ty': 31557600.0 * 1e12, # terayear
    'Py': 31557600.0 * 1e15, # petayear
    'Ey': 31557600.0 * 1e18, # exayear
    'Zy': 31557600.0 * 1e21, # zettayear
    'Yy': 31557600.0 * 1e24, # yottayear
}


def parse_halflife(halflife_str, unit_str):
    """
    Convert half-life string + unit to seconds.

    Args:
        halflife_str: Half-life value (e.g., "12.32", "stbl", "p-unst", ">10", "~5")
        unit_str: Unit (e.g., "y", "ms", "s")

    Returns:
        float: Half-life in seconds (np.nan for stable/unknown, np.inf for p-unst)
    """
    halflife_str = halflife_str.strip()
    unit_str = unit_str.strip()

    # Handle special cases
    if halflife_str == 'stbl':
        return np.nan  # Stable
    if halflife_str == 'p-unst':
        return np.inf  # Particle unstable (treat as very short-lived)
    if not halflife_str or halflife_str == '':
        return np.nan  # Unknown

    # Handle systematics marker (#)
    halflife_str = halflife_str.replace('#', '')

    # Handle inequality symbols (>, <, ~)
    # Strip them but keep the value
    halflife_str = halflife_str.replace('>', '').replace('<', '').replace('~', '').strip()

    try:
        value = float(halflife_str)
    except ValueError:
        logger.debug(f"Could not parse half-life: '{halflife_str}' {unit_str}")
        return np.nan

    # Convert to seconds
    if unit_str in HALFLIFE_UNITS:
        return value * HALFLIFE_UNITS[unit_str]
    else:
        logger.warning(f"Unknown half-life unit: '{unit_str}'")
        return np.nan


def parse_decay_modes(decay_str):
    """
    Parse decay mode string into structured data.

    Args:
        decay_str: Decay modes string (e.g., "B-=100", "B-=99.9;B-n=0.1")

    Returns:
        dict: {
            'modes': list of mode names,
            'branchings': list of branching ratios (floats),
            'dominant_mode': mode with highest branching
        }
    """
    decay_str = decay_str.strip()

    if not decay_str or decay_str == '':
        return {
            'modes': [],
            'branchings': [],
            'dominant_mode': 'unknown'
        }

    # Split by semicolon
    parts = decay_str.split(';')

    modes = []
    branchings = []

    for part in parts:
        part = part.strip()

        # Skip isotopic abundance (IS=...)
        if part.startswith('IS='):
            continue

        # Parse mode=value or mode ?
        if '=' in part:
            mode, value = part.split('=', 1)
            mode = mode.strip()

            # Extract numeric value (ignore uncertainty in parentheses)
            value = value.split()[0]  # Take first part before space/uncertainty
            try:
                branching = float(value)
            except ValueError:
                branching = 0.0
        else:
            # Mode without branching (e.g., "p ?")
            mode = part.replace('?', '').strip()
            branching = 100.0 if len(parts) == 1 else 0.0  # Assume 100% if only mode

        modes.append(mode)
        branchings.append(branching)

    # Determine dominant mode
    if modes and branchings:
        dominant_idx = np.argmax(branchings)
        dominant_mode = modes[dominant_idx]
    else:
        dominant_mode = 'unknown'

    return {
        'modes': modes,
        'branchings': branchings,
        'dominant_mode': dominant_mode
    }


def normalize_decay_mode(mode_str):
    """
    Normalize decay mode name to canonical form.

    Maps:
      B- → beta_minus
      B+ → beta_plus
      EC → EC
      A, a, α → alpha
      n → neutron
      2n, 3n → neutron
      p → proton
      IT → IT
      SF → fission
      ... etc.
    """
    mode_str = mode_str.strip()

    # Handle special markers (~100, etc.)
    # Strip these for classification
    mode_clean = mode_str.replace('~100', '').replace('~', '').strip()

    # Map to canonical names
    if 'B-' in mode_clean or mode_clean in ['2B-', '3B-']:
        return 'beta_minus'
    elif 'B+' in mode_clean or mode_clean in ['2B+', '3B+']:
        return 'beta_plus'
    elif 'EC' in mode_clean:
        return 'EC'
    elif mode_clean in ['A', 'a', 'α', '2a', '3a', '4a'] or 'A+' in mode_clean:
        return 'alpha'
    elif mode_clean in ['n', '2n', '3n', '4n'] or mode_clean.startswith('n+'):
        return 'neutron'
    elif mode_clean in ['p', '2p', '3p', '4p'] or mode_clean.startswith('p+'):
        return 'proton'
    elif mode_clean == 'IT':
        return 'IT'
    elif mode_clean in ['SF', 'fission'] or mode_clean.startswith('SF'):
        return 'fission'
    elif mode_clean == '':
        return 'unknown'
    else:
        # Keep other modes as-is for diagnostics
        return mode_clean


def parse_nubase_line(line):
    """
    Parse a single line from NUBASE2020 raw file.

    NUBASE format (fixed-width):
      Columns 1-3:   A (mass number)
      Columns 5-8:   Z + isomer flag
      Columns 12-16: Element symbol
      Columns 19-31: Mass excess (keV)
      Columns 43-54: Excitation energy (keV)
      Columns 70-78: Half-life
      Columns 79-80: Half-life unit
      Columns 120-209: Decay modes

    Returns:
        dict or None: Parsed data or None if line is invalid
    """
    # Skip comment lines
    if line.startswith('#') or len(line) < 120:
        return None

    try:
        # Extract fields (1-indexed columns, convert to 0-indexed)
        A_str = line[0:3].strip()
        Z_isomer_str = line[4:8].strip()
        element = line[11:16].strip()
        mass_excess_str = line[18:31].strip()
        exc_energy_str = line[42:54].strip()
        halflife_str = line[69:78].strip()
        unit_str = line[78:80].strip()

        # Decay modes field (if line is long enough)
        if len(line) >= 120:
            decay_str = line[119:209].strip()
        else:
            decay_str = ''

        # Parse A
        A = int(A_str)

        # Parse Z and isomer flag
        # Z_isomer_str is like "0010" (Z=1, isomer=0) or "0021" (Z=2, isomer=1)
        if len(Z_isomer_str) == 4:
            Z = int(Z_isomer_str[:3])
            isomer_flag = int(Z_isomer_str[3])
        elif len(Z_isomer_str) == 3:
            Z = int(Z_isomer_str)
            isomer_flag = 0
        else:
            logger.warning(f"Invalid Z format: '{Z_isomer_str}'")
            return None

        # Only process ground states (isomer_flag == 0)
        if isomer_flag != 0:
            return None

        N = A - Z

        # Parse mass excess
        mass_excess_str = mass_excess_str.replace('#', '')  # Remove systematics marker
        try:
            mass_excess = float(mass_excess_str)
        except ValueError:
            mass_excess = np.nan

        # Parse excitation energy
        exc_energy_str = exc_energy_str.replace('#', '')
        try:
            exc_energy = float(exc_energy_str)
        except ValueError:
            exc_energy = np.nan

        # Parse half-life
        half_life_s = parse_halflife(halflife_str, unit_str)

        # Determine stability
        is_stable = pd.isna(half_life_s)
        is_particle_unstable = np.isinf(half_life_s)

        # Parse decay modes
        decay_info = parse_decay_modes(decay_str)
        dominant_mode_raw = decay_info['dominant_mode']
        dominant_mode = normalize_decay_mode(dominant_mode_raw)

        return {
            'A': A,
            'Z': Z,
            'N': N,
            'element': element,
            'mass_excess_keV': mass_excess,
            'excitation_keV': exc_energy,
            'half_life_s': half_life_s,
            'is_stable': is_stable,
            'is_particle_unstable': is_particle_unstable,
            'dominant_mode': dominant_mode,
            'dominant_mode_raw': dominant_mode_raw,
            'decay_modes': ';'.join(decay_info['modes']),
            'branching_ratios': ';'.join(map(str, decay_info['branchings'])),
        }

    except Exception as e:
        logger.warning(f"Error parsing line: {e}")
        logger.warning(f"Line: {line[:80]}")
        return None


def parse_nubase(input_file, output_file=None):
    """
    Parse NUBASE2020 raw text file.

    Args:
        input_file: Path to nubase2020_raw.txt
        output_file: Path to output parquet file (optional)

    Returns:
        pd.DataFrame: Parsed nuclide data
    """
    input_path = Path(input_file)

    if not input_path.exists():
        raise FileNotFoundError(f"NUBASE file not found: {input_file}")

    logger.info(f"Parsing NUBASE file: {input_file}")

    records = []

    with open(input_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            parsed = parse_nubase_line(line)
            if parsed is not None:
                records.append(parsed)

    df = pd.DataFrame(records)

    logger.info(f"Parsed {len(df)} ground-state nuclides")

    # Summary statistics
    n_stable = df['is_stable'].sum()
    n_unstable = (~df['is_stable']).sum()
    n_particle_unstable = df['is_particle_unstable'].sum()

    logger.info(f"  Stable: {n_stable}")
    logger.info(f"  Radioactive: {n_unstable - n_particle_unstable}")
    logger.info(f"  Particle unstable: {n_particle_unstable}")

    # Decay mode counts
    mode_counts = df['dominant_mode'].value_counts()
    logger.info("Decay mode distribution:")
    for mode, count in mode_counts.head(10).items():
        logger.info(f"  {mode}: {count}")

    # Save to parquet if output path provided
    if output_file is not None:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        logger.info(f"Saved to: {output_file}")

    return df


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Parse NUBASE2020 raw text file into structured Parquet dataset'
    )
    parser.add_argument(
        '--input',
        required=True,
        help='Path to nubase2020_raw.txt'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Path to output parquet file (e.g., data/derived/nuclides_all.parquet)'
    )

    args = parser.parse_args()

    df = parse_nubase(args.input, args.output)

    print("\n" + "="*80)
    print("NUBASE PARSING COMPLETE")
    print("="*80)
    print(f"Total nuclides: {len(df)}")
    print(f"Stable: {df['is_stable'].sum()}")
    print(f"Output: {args.output}")
    print("="*80)


if __name__ == '__main__':
    main()
