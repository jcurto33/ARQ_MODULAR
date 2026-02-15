"""
demo.app
─────────
Gestor de navegación de la aplicación Streamlit.
"""

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
