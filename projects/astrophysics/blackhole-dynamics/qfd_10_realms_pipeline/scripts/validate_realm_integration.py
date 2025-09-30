#!/usr/bin/env python3
"""
Validation script for realm integration with coupling constants framework.

This script validates that individual realms can be properly integrated
with the coupling constants analysis system.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from coupling_constants.integration.workflow_scripts import (
    validate_realm_integration, get_default_realm_sequence
)


def main():
    """Main entry point for realm integration validation."""
    parser = argparse.ArgumentParser(
        description="Validate realm integration with coupling constants framework"
    )
    
    parser.add_argument('--config', default='qfd_params/defaults.yaml',
                       help='Path to QFD configuration file')
    parser.add_argument('--realm', help='Specific realm to validate (validates all if not specified)')
    parser.add_argument('--plugins', nargs='*',
                       choices=['photon_mass', 'vacuum_stability', 'cosmological_constant'],
                       help='Enable constraint plugins for validation')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set up logging
    level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(level=level)
    
    print("QFD Realm Integration Validation")
    print("=" * 40)
    
    # Determine which realms to validate
    if args.realm:
        realms_to_validate = [args.realm]
    else:
        realm_sequence = get_default_realm_sequence()
        realms_to_validate = [realm_name for realm_name, _ in realm_sequence]
    
    # Validate each realm
    results = {}
    for realm_name in realms_to_validate:
        print(f"\nValidating {realm_name}...")
        success = validate_realm_integration(
            config_path=args.config,
            realm_name=realm_name,
            enable_plugins=args.plugins
        )
        results[realm_name] = success
    
    # Print summary
    print(f"\n{'='*40}")
    print("VALIDATION SUMMARY")
    print(f"{'='*40}")
    
    successful = [name for name, success in results.items() if success]
    failed = [name for name, success in results.items() if not success]
    
    print(f"Total realms validated: {len(results)}")
    print(f"Successful integrations: {len(successful)}")
    print(f"Failed integrations: {len(failed)}")
    
    if successful:
        print(f"\n✓ Successfully integrated realms:")
        for realm_name in successful:
            print(f"  - {realm_name}")
    
    if failed:
        print(f"\n✗ Failed integrations:")
        for realm_name in failed:
            print(f"  - {realm_name}")
        print(f"\nRun with --verbose for detailed error information.")
    
    # Exit with appropriate code
    sys.exit(0 if len(failed) == 0 else 1)


if __name__ == "__main__":
    main()