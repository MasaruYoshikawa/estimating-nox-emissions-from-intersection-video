"""Pytest root configuration.

Adds the `src/` layout to sys.path so tests can import the package
without an editable install.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
