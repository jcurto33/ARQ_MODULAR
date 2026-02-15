"""
demo.pages.2_IA_Act_Chatbot
─────────────────────────────
Página de Streamlit para el chatbot del IA Act.
Interfaz de chat con memoria de sesión.
"""

import uuid
import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.header("📜 IA Act Chatbot")
st.caption("Pregunta lo que quieras sobre el Reglamento de Inteligencia Artificial (UE 2024/1689)")

# ── Sesión ────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Botón de nueva conversación ───────────────────────────
if st.sidebar.button("🗑️ Nueva conversación"):
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.rerun()

# ── Historial de mensajes ────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Entrada del usuario ──────────────────────────────────
if prompt := st.chat_input("Escribe tu pregunta sobre el IA Act…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Pensando…"):
            try:
                resp = requests.post(
                    f"{API_URL}/chatbot/chat",
                    json={"session_id": st.session_state.session_id, "message": prompt},
                    timeout=60,
                )
                reply = resp.json()["reply"]
            except requests.exceptions.ConnectionError:
                reply = "⚠️ No se pudo conectar con la API. ¿Está corriendo en el puerto 8000?"

        st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})
