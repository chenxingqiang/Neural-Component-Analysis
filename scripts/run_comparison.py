#!/usr/bin/env python3
"""
Wrapper script to run secom_comparison_detection.py properly by adjusting the Python path.
This helps avoid import errors when running scripts directly.
"""

import os
import sys

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Import and run the main function from the comparison script
from scripts.secom_comparison_detection import main

if __name__ == "__main__":
    # Pass command line arguments to the main function
    main() 