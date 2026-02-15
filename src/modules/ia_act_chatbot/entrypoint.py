"""
src.modules.ia_act_chatbot.entrypoint
──────────────────────────────────────
INTERFAZ PÚBLICA del módulo.
Función expuesta: chat().
"""

from src.modules.ia_act_chatbot.agents.rag_agent import RAGAgent


# ── Caché de agentes por sesión (memoria temporal) ────────
_sessions: dict[str, RAGAgent] = {}


def chat(session_id: str, user_message: str) -> str:
    """Envía un mensaje al chatbot y devuelve la respuesta.

    Cada session_id mantiene su propio historial en memoria.
    Al reiniciar el servidor, las sesiones se pierden.

    Args:
        session_id: Identificador de la sesión activa.
        user_message: Mensaje del usuario.

    Returns:
        Respuesta del chatbot (str).
    """
    if session_id not in _sessions:
        _sessions[session_id] = RAGAgent()

    return _sessions[session_id].answer(user_message)
