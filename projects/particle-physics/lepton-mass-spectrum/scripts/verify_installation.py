#!/usr/bin/env python3
"""
Verify that the installation is correct and all dependencies are available.
"""

import sys
import importlib

def check_import(module_name, package_name=None):
    """Check if a module can be imported."""
    if package_name is None:
        package_name = module_name

    try:
        mod = importlib.import_module(module_name)
        version = getattr(mod, '__version__', 'unknown')
        print(f"✓ {package_name:20s} version {version}")
        return True
    except ImportError as e:
        print(f"✗ {package_name:20s} MISSING - {e}")
        return False

def check_local_modules():
    """Check local src modules."""
    sys.path.insert(0, '../src')

    try:
        import functionals
        import solvers
        print(f"✓ {'Local modules':20s} imported successfully")
        return True
    except ImportError as e:
        print(f"✗ {'Local modules':20s} FAILED - {e}")
        return False

def main():
    print("="*60)
    print("Installation Verification")
    print("="*60)
    print()

    print("Required Dependencies:")
    print("-"*60)

    all_ok = True
    all_ok &= check_import('numpy')
    all_ok &= check_import('scipy')
    all_ok &= check_import('matplotlib')
    all_ok &= check_import('emcee')
    all_ok &= check_import('corner')
    all_ok &= check_import('h5py')

    print()
    print("Local Modules:")
    print("-"*60)
    all_ok &= check_local_modules()

    print()
    print("="*60)
    if all_ok:
        print("✓ All checks passed - ready to run!")
    else:
        print("✗ Some checks failed - install missing dependencies")
        print("\nRun: pip install -r requirements.txt")
    print("="*60)

    return 0 if all_ok else 1

if __name__ == '__main__':
    sys.exit(main())
