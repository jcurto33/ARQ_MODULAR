"""
demo.pages.1_Iris_Classifier
──────────────────────────────
Página de Streamlit para el módulo Iris Classifier.
Permite entrenar el modelo y hacer predicciones interactivas.
"""

import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.header("🌸 Iris Classifier")

tab_train, tab_predict = st.tabs(["Entrenar modelo", "Predecir especie"])

# ── Pestaña de entrenamiento ──────────────────────────────
with tab_train:
    st.markdown("Lanza el entrenamiento con búsqueda de hiperparámetros sobre 3 modelos candidatos.")
    if st.button("🚀 Entrenar modelo"):
        with st.spinner("Entrenando…"):
            try:
                resp = requests.post(f"{API_URL}/iris/train", timeout=120)
                data = resp.json()
                st.success("Entrenamiento completado")

                metrics = data.get("metrics", {})
                st.subheader(f"Mejor modelo: `{metrics.get('selected_model')}`")

                st.json(metrics.get("candidates", {}))
            except requests.exceptions.ConnectionError:
                st.error("No se pudo conectar con la API. ¿Está corriendo en el puerto 8000?")

# ── Pestaña de predicción ─────────────────────────────────
with tab_predict:
    st.markdown("Introduce las 4 medidas de una flor Iris:")

    col1, col2 = st.columns(2)
    with col1:
        sepal_length = st.number_input("Sepal length (cm)", 0.0, 10.0, 5.1, 0.1)
        sepal_width = st.number_input("Sepal width (cm)", 0.0, 10.0, 3.5, 0.1)
    with col2:
        petal_length = st.number_input("Petal length (cm)", 0.0, 10.0, 1.4, 0.1)
        petal_width = st.number_input("Petal width (cm)", 0.0, 10.0, 0.2, 0.1)

    if st.button("🔮 Predecir"):
        payload = {
            "sepal_length": sepal_length,
            "sepal_width": sepal_width,
            "petal_length": petal_length,
            "petal_width": petal_width,
        }
        try:
            resp = requests.post(f"{API_URL}/iris/predict", json=payload, timeout=30)
            result = resp.json()

            st.success(f"Predicción: **{result['prediction']}**")
            st.bar_chart(result["probabilities"])
        except requests.exceptions.ConnectionError:
            st.error("No se pudo conectar con la API.")
