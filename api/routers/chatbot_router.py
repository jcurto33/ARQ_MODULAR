"""
api.routers.chatbot_router
──────────────────────────
Endpoints del módulo IA Act Chatbot.
"""

from fastapi import APIRouter
from pydantic import BaseModel

from src.modules.ia_act_chatbot.entrypoint import chat

router = APIRouter()


class ChatRequest(BaseModel):
    """Mensaje del usuario con ID de sesión."""
    session_id: str
    message: str


@router.post("/chat")
def chat_endpoint(req: ChatRequest):
    """Envía un mensaje al chatbot del IA Act."""
    reply = chat(session_id=req.session_id, user_message=req.message)
    return {"reply": reply}
