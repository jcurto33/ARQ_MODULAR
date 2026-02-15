"""
tests/conftest.py
──────────────────
Fixtures compartidas para los tests.
"""

import sys
from pathlib import Path
import pytest

# Asegurar que el proyecto está en el path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
