"""
scripts/iris_ingest.py
──────────────────────
Wrapper CLI para descargar el dataset Iris a data/files/raw/.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.modules.iris_classifier.data_processing.loader import download_iris


if __name__ == "__main__":
    print("═" * 50)
    print("  Ingesta — Iris Dataset")
    print("═" * 50)

    path = download_iris()
    print(f"\n✅ Dataset guardado en: {path}")
