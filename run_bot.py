#!/usr/bin/env python3
"""
AETHER Trading Bot - Launcher Script
Wrapper for the main CLI entry point.
"""

import sys
import os

# Add the current directory to sys.path to ensure src is importable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.cli import main

if __name__ == "__main__":
    main()
