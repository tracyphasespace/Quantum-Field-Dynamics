#!/usr/bin/env python3
"""
Entry point script for QFD Coupling Constants Analysis CLI.

This script provides a convenient way to run the coupling constants analysis
from the command line without needing to install the package.
"""

import sys
import os

# Add the current directory to Python path so we can import coupling_constants
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from coupling_constants.cli.main import main

if __name__ == '__main__':
    sys.exit(main())