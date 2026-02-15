"""
shared.utils.llm_client
────────────────────────
Abstracción para modelos de lenguaje compatibles con la API de OpenAI.
Permite cambiar el proveedor (OpenAI, Azure, local) modificando solo .env.
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


def get_llm_client() -> OpenAI:
    """Devuelve un cliente OpenAI configurado desde variables de entorno."""
    return OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )


def get_model_name() -> str:
    """Devuelve el nombre del modelo configurado."""
    return os.getenv("OPENAI_MODEL", "gpt-4o-mini")
