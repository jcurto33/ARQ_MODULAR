"""
shared.config_loader
─────────────────────
Carga centralizada de la configuración global del proyecto.
"""

from pathlib import Path
import yaml

_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_PATH = _ROOT / "config" / "global_config.yaml"


def load_config() -> dict:
    """Lee y devuelve el diccionario de configuración global."""
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
