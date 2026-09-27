#!/usr/bin/env python
"""
Convenience launcher for Hearthstone AI Assistant.
Can be executed from any working directory.
"""

import sys
from pathlib import Path

# Add project root to sys.path so 'src' is always importable
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.live.main import main

if __name__ == "__main__":
    main()
