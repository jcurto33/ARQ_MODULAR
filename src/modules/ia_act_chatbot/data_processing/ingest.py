"""
src.modules.ia_act_chatbot.data_processing.ingest
──────────────────────────────────────────────────
Procesa el HTML del IA Act, lo divide en chunks y los almacena en Qdrant.
"""

import logging
from pathlib import Path

from bs4 import BeautifulSoup
from qdrant_client.models import Distance, VectorParams, PointStruct

from shared.config_loader import load_config
from shared.database.vector_store import get_qdrant_client
from shared.utils.llm_client import get_llm_client

logger = logging.getLogger(__name__)
_config = load_config()
_chatbot_cfg = _config["ia_act_chatbot"]


# ── 1. Extraer texto del HTML ─────────────────────────────

def _extract_text(html_path: str | Path) -> str:
    """Lee el HTML y extrae el texto plano limpio."""
    with open(html_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")
    return soup.get_text(separator="\n", strip=True)


# ── 2. Dividir en chunks ─────────────────────────────────

def _split_into_chunks(text: str) -> list[str]:
    """Divide el texto en fragmentos con solapamiento."""
    chunk_size = _chatbot_cfg["chunk_size"]
    overlap = _chatbot_cfg["chunk_overlap"]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


# ── 3. Obtener embeddings ─────────────────────────────────

def _get_embeddings(texts: list[str]) -> list[list[float]]:
    """Genera embeddings usando la API de OpenAI."""
    client = get_llm_client()
    response = client.embeddings.create(
        input=texts,
        model="text-embedding-3-small",
    )
    return [item.embedding for item in response.data]


# ── 4. Pipeline completo ─────────────────────────────────

def run_ingestion(html_path: str | Path) -> int:
    """Ejecuta el pipeline de ingesta completo.

    Returns:
        Número de chunks insertados.
    """
    logger.info("Extrayendo texto de %s", html_path)
    text = _extract_text(html_path)

    logger.info("Dividiendo en chunks (size=%d, overlap=%d)",
                _chatbot_cfg["chunk_size"], _chatbot_cfg["chunk_overlap"])
    chunks = _split_into_chunks(text)
    logger.info("Chunks generados: %d", len(chunks))

    logger.info("Generando embeddings…")
    # Procesar en lotes de 100 para evitar límites de API
    all_embeddings: list[list[float]] = []
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        all_embeddings.extend(_get_embeddings(batch))

    # ── Insertar en Qdrant ────────────────────────────────
    client = get_qdrant_client()
    collection = _chatbot_cfg["collection_name"]
    vector_size = len(all_embeddings[0])

    # Recrear colección (idempotente)
    client.recreate_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )

    points = [
        PointStruct(id=i, vector=emb, payload={"text": chunk})
        for i, (chunk, emb) in enumerate(zip(chunks, all_embeddings))
    ]

    # Insertar en lotes
    for i in range(0, len(points), batch_size):
        client.upsert(collection_name=collection, points=points[i : i + batch_size])

    logger.info("Ingesta completada: %d chunks en colección '%s'", len(chunks), collection)
    return len(chunks)
