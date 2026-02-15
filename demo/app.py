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

# ── CSS global para toda la app ──────────────────────────
st.markdown("""
<style>
    /* Tipografía general más grande y legible */
    .main .block-container { max-width: 1200px; padding-top: 2rem; }
    html, body, [class*="css"] { font-size: 16px; }
    h1 { font-size: 2.8rem !important; margin-bottom: 0.5rem !important; }
    h2 { font-size: 2rem !important; }
    h3 { font-size: 1.5rem !important; }
    p, li, .stMarkdown, .stText { font-size: 1.1rem !important; line-height: 1.7; }

    /* Sidebar más legible */
    .css-1d391kg, [data-testid="stSidebar"] { min-width: 280px; }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] li,
    [data-testid="stSidebar"] .stMarkdown { font-size: 1.05rem !important; }

    /* Tabs más grandes */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.15rem !important;
        padding: 12px 24px;
        border-radius: 8px 8px 0 0;
    }

    /* Inputs más accesibles */
    .stNumberInput label, .stSlider label, .stSelectbox label,
    .stTextInput label, .stTextArea label {
        font-size: 1.1rem !important;
        font-weight: 500;
    }

    /* Botones con mejor presencia */
    .stButton > button {
        font-size: 1.1rem !important;
        border-radius: 10px;
        padding: 0.6rem 1.8rem;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .stButton > button:hover { transform: translateY(-1px); }
</style>
""", unsafe_allow_html=True)

st.title("🤖 IA Demo")
st.markdown(
    """
    <p style="font-size: 1.3rem; color: #a0aec0; margin-bottom: 2rem;">
    Repositorio de ejemplo con dos módulos de Inteligencia Artificial.
    Usa el menú lateral para navegar entre las páginas.
    </p>
    """,
    unsafe_allow_html=True,
)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #1a3a2a 0%, #1e4d3a 100%);
                    border: 1px solid #38a169; border-radius: 16px; padding: 2rem;">
            <h3 style="margin-top: 0; color: #c6f6d5;">🌸 Iris Classifier</h3>
            <p style="color: #c6f6d5;">
                Clasificación ML clásica con tuning de hiperparámetros.
                Entrena modelos, explora los datos y haz predicciones interactivas.
            </p>
            <ul style="color: #c6f6d5;">
                <li>GridSearch sobre RF, GB y SVM</li>
                <li>Gráficas EDA del dataset</li>
                <li>Predicción con contexto visual</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #1a2a3a 0%, #1e3a5a 100%);
                    border: 1px solid #3182ce; border-radius: 16px; padding: 2rem;">
            <h3 style="margin-top: 0; color: #bee3f8;">📜 IA Act Chatbot</h3>
            <p style="color: #bee3f8;">
                Chatbot RAG experto en el Reglamento de IA europeo.
                Respuestas fundamentadas en el texto oficial.
            </p>
            <ul style="color: #bee3f8;">
                <li>Retrieval-Augmented Generation</li>
                <li>Cita artículos del reglamento</li>
                <li>Memoria de conversación</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
