#!/usr/bin/env python3
"""
Extract raw photometry from DES-SN5YR dataset.

This script processes the SNANA FITS files and creates a unified CSV
compatible with the QFD pipeline.

Usage:
    python extract_raw_photometry.py --help
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from preprocessing import process_des_sn5yr_dataset
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Extract raw DES-SN5YR photometry",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process full DES dataset (19,706 SNe)
  python extract_raw_photometry.py \\
      --data-dir data/raw/DES-SN5YR-1.2/0_DATA \\
      --output data/processed/lightcurves_des_full.csv

  # Test with 100 SNe
  python extract_raw_photometry.py \\
      --data-dir data/raw/DES-SN5YR-1.2/0_DATA \\
      --output data/processed/lightcurves_des_test.csv \\
      --max-sne 100

  # Only spectroscopically confirmed Type Ia (353 SNe)
  python extract_raw_photometry.py \\
      --data-dir data/raw/DES-SN5YR-1.2/0_DATA \\
      --output data/processed/lightcurves_des_spec.csv \\
      --spec-only

  # Process LOWZ dataset
  python extract_raw_photometry.py \\
      --data-dir data/raw/DES-SN5YR-1.2/0_DATA \\
      --output data/processed/lightcurves_lowz.csv \\
      --dataset DES-SN5YR_LOWZ
        """
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw/DES-SN5YR-1.2/0_DATA"),
        help="Path to 0_DATA directory (default: data/raw/DES-SN5YR-1.2/0_DATA)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/lightcurves_unified_full.csv"),
        help="Output CSV path (default: data/processed/lightcurves_unified_full.csv)"
    )
    parser.add_argument(
        "--dataset",
        default="DES-SN5YR_DES",
        choices=["DES-SN5YR_DES", "DES-SN5YR_LOWZ", "DES-SN5YR_Foundation"],
        help="Dataset to process (default: DES-SN5YR_DES)"
    )
    parser.add_argument(
        "--spec-only",
        action="store_true",
        help="Only include spectroscopically confirmed Type Ia (~353 SNe for DES)"
    )
    parser.add_argument(
        "--min-obs",
        type=int,
        default=5,
        help="Minimum number of observations per SN (default: 5)"
    )
    parser.add_argument(
        "--min-z",
        type=float,
        default=0.05,
        help="Minimum redshift (default: 0.05)"
    )
    parser.add_argument(
        "--max-z",
        type=float,
        default=1.3,
        help="Maximum redshift (default: 1.3)"
    )
    parser.add_argument(
        "--max-sne",
        type=int,
        default=None,
        help="Maximum SNe to process (for testing, default: all)"
    )
    parser.add_argument(
        "--quality",
        choices=["none", "medium", "strict"],
        default="medium",
        help="Quality level: none (all SNe), medium (5+ obs), strict (10+ obs)"
    )

    args = parser.parse_args()

    # Adjust parameters based on quality level
    if args.quality == "strict":
        min_obs = 10
    elif args.quality == "medium":
        min_obs = args.min_obs
    else:  # none
        min_obs = 1

    # Create output directory if needed
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Check if data directory exists
    if not args.data_dir.exists():
        print(f"Error: Data directory not found: {args.data_dir}")
        print("\nPlease ensure you have extracted DES-SN5YR-1.2.zip to data/raw/")
        sys.exit(1)

    print("=" * 70)
    print("DES-SN5YR Raw Photometry Extraction")
    print("=" * 70)
    print(f"Dataset: {args.dataset}")
    print(f"Quality level: {args.quality}")
    print(f"Type Ia selection: {'Spec-confirmed only' if args.spec_only else 'Spec + Photometric'}")
    print(f"Redshift range: {args.min_z} - {args.max_z}")
    print(f"Min observations: {min_obs}")
    if args.max_sne:
        print(f"Max SNe: {args.max_sne} (TEST MODE)")
    print("=" * 70)
    print()

    # Process dataset
    try:
        df = process_des_sn5yr_dataset(
            data_dir=args.data_dir,
            output_path=args.output,
            dataset_name=args.dataset,
            require_spec_confirmed=args.spec_only,
            include_photometric=not args.spec_only,
            min_observations=min_obs,
            min_redshift=args.min_z,
            max_redshift=args.max_z,
            max_sne=args.max_sne,
            verbose=True
        )

        print("\n" + "=" * 70)
        print("SUCCESS!")
        print("=" * 70)
        print(f"Extracted {df['snid'].nunique()} supernovae")
        print(f"Total measurements: {len(df):,}")
        print(f"Output: {args.output}")
        print(f"File size: {args.output.stat().st_size / 1e6:.1f} MB")
        print("\nNext steps:")
        print("  1. Validate data: python scripts/validate_data.py")
        print("  2. Run QFD pipeline: see README.md for details")
        print("=" * 70)

    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
