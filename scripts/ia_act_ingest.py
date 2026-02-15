"""
scripts/ingest.py
──────────────────
Wrapper CLI para ingestar el HTML del IA Act en Qdrant.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.modules.ia_act_chatbot.data_processing.ingest import run_ingestion

# Ruta al HTML del IA Act (ajustar si se mueve a data/files/raw/)
_HTML_PATH = Path(__file__).resolve().parent.parent / "data" / "files" / "raw" / "ia_act.html"


if __name__ == "__main__":
    print("═" * 50)
    print("  Ingesta — IA Act Chatbot")
    print("═" * 50)

    if not _HTML_PATH.exists():
        print(f"❌ No se encontró el archivo: {_HTML_PATH}")
        print("   Copia el HTML del IA Act a data/files/raw/ia_act.html")
        sys.exit(1)

    n_chunks = run_ingestion(_HTML_PATH)
    print(f"\n✅ Ingesta completada: {n_chunks} chunks almacenados en Qdrant")
