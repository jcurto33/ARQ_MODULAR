"""
shared.database.vector_store
─────────────────────────────
Gestión de la conexión a Qdrant (BBDD vectorial).
Centraliza la inicialización del cliente para todo el proyecto.
"""

from pathlib import Path
from qdrant_client import QdrantClient

from shared.config_loader import load_config

_config = load_config()
_VECTOR_DB_DIR = Path(_config["paths"]["vector_db_dir"])


def get_qdrant_client() -> QdrantClient:
    """Devuelve un cliente Qdrant persistente en disco."""
    _VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(_VECTOR_DB_DIR))
