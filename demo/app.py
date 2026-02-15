"""
demo.app
─────────
Gestor de navegación de la aplicación Streamlit.
"""

import sys
from pathlib import Path

# Asegurar que la raíz del repositorio está en el path para importar src/shared
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import streamlit as st

st.set_page_config(
    page_title="IA Demo",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 IA Demo")
st.markdown(
    """
    Repositorio de ejemplo con dos módulos:

    - **🌸 Iris Classifier** — Clasificación ML clásica con tuning de hiperparámetros.
    - **📜 IA Act Chatbot** — Chatbot RAG experto en el Reglamento de IA europeo.

    Usa el menú lateral para navegar entre las páginas.
    """
)
