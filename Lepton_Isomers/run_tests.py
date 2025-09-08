#!/usr/bin/env python3
"""
Test Runner for QFD Phoenix
===========================

Convenient script to run all tests from the project root.

Usage:
    python run_tests.py [--integration] [--isomer] [--all]
"""

import sys
import subprocess
from pathlib import Path
import argparse

def run_integration_tests():
    """Run main integration test suite."""
    print("Running integration tests...")
    result = subprocess.run([
        sys.executable, 
        "tests/integration/test_package.py"
    ], cwd=Path(__file__).parent)
    return result.returncode == 0

def run_isomer_tests():
    """Run isomer framework tests."""
    print("Running isomer framework tests...")
    result = subprocess.run([
        sys.executable, 
        "tests/integration/test_isomer_framework.py", 
        "--quick", "--device", "cpu"
    ], cwd=Path(__file__).parent)
    return result.returncode == 0

def main():
    parser = argparse.ArgumentParser(description="QFD Phoenix Test Runner")
    parser.add_argument("--integration", action="store_true", 
                       help="Run integration tests only")
    parser.add_argument("--isomer", action="store_true",
                       help="Run isomer framework tests only") 
    parser.add_argument("--all", action="store_true",
                       help="Run all tests")
    
    args = parser.parse_args()
    
    # Default to integration tests if no specific test specified
    if not any([args.integration, args.isomer, args.all]):
        args.integration = True
    
    success = True
    
    if args.integration or args.all:
        success &= run_integration_tests()
        print()
    
    if args.isomer or args.all:
        success &= run_isomer_tests()
        print()
    
    if success:
        print("SUCCESS: All tests passed!")
        return 0
    else:
        print("FAILED: Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())