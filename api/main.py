"""
api.main
─────────
Punto de entrada de la aplicación FastAPI.
Configura middlewares, CORS y registra routers.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import iris_router, chatbot_router

app = FastAPI(
    title="IA Demo API",
    description="API de ejemplo con un módulo ML (Iris) y un chatbot RAG (IA Act).",
    version="0.1.0",
)

# ── CORS ──────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────
app.include_router(iris_router.router, prefix="/iris", tags=["Iris Classifier"])
app.include_router(chatbot_router.router, prefix="/chatbot", tags=["IA Act Chatbot"])


# ── Health check ──────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}
