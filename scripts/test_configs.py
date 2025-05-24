#!/usr/bin/env python3
"""
Script to run configuration system tests.
"""

import pytest
import sys
import os
from pathlib import Path

def main():
    """Run configuration tests."""
    # Add project root to Python path
    project_root = Path(__file__).parent.parent
    sys.path.append(str(project_root))
    
    # Run tests
    test_path = project_root / "tests" / "test_configs.py"
    return pytest.main([str(test_path), "-v"])

if __name__ == "__main__":
    sys.exit(main()) 