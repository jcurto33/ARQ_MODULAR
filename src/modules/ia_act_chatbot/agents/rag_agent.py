"""
src.modules.ia_act_chatbot.agents.rag_agent
────────────────────────────────────────────
Agente RAG con memoria de sesión temporal.
Busca contexto en Qdrant y genera respuestas con un LLM.
"""

import logging

from shared.config_loader import load_config
from shared.database.vector_store import get_qdrant_client
from shared.utils.llm_client import get_llm_client, get_model_name

logger = logging.getLogger(__name__)
_config = load_config()
_chatbot_cfg = _config["ia_act_chatbot"]


class RAGAgent:
    """Agente conversacional con RAG sobre el IA Act.

    Mantiene un historial de mensajes en memoria (desaparece al reiniciar).
    """

    def __init__(self):
        self.history: list[dict] = [
            {"role": "system", "content": _chatbot_cfg["system_prompt"]},
        ]
        self._qdrant = get_qdrant_client()
        self._llm = get_llm_client()
        self._model = get_model_name()
        self._collection = _chatbot_cfg["collection_name"]
        self._top_k = _chatbot_cfg["top_k"]

    # ── Búsqueda de contexto ──────────────────────────────

    def _retrieve_context(self, query: str) -> str:
        """Busca los chunks más relevantes en Qdrant."""
        query_embedding = self._llm.embeddings.create(
            input=[query], model="text-embedding-3-small"
        ).data[0].embedding

        results = self._qdrant.query_points(
            collection_name=self._collection,
            query=query_embedding,
            limit=self._top_k,
        ).points

        fragments = [r.payload["text"] for r in results if r.payload]
        return "\n---\n".join(fragments)

    # ── Respuesta ─────────────────────────────────────────

    def answer(self, user_message: str) -> str:
        """Genera una respuesta usando RAG + historial de conversación."""
        context = self._retrieve_context(user_message)

        augmented_message = (
            f"Contexto del Reglamento:\n{context}\n\n"
            f"Pregunta del usuario:\n{user_message}"
        )

        self.history.append({"role": "user", "content": augmented_message})

        response = self._llm.chat.completions.create(
            model=self._model,
            messages=self.history,
            temperature=0.3,
        )

        assistant_reply = response.choices[0].message.content or ""
        self.history.append({"role": "assistant", "content": assistant_reply})

        return assistant_reply
