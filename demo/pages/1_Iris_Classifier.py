"""
demo.pages.1_Iris_Classifier
──────────────────────────────
Página de Streamlit para el módulo Iris Classifier.
Permite entrenar el modelo y hacer predicciones interactivas.
"""

import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import streamlit as st
from src.modules.iris_classifier.entrypoint import train, predict

st.header("🌸 Iris Classifier")

tab_train, tab_predict = st.tabs(["Entrenar modelo", "Predecir especie"])

# ── Pestaña de entrenamiento ──────────────────────────────
with tab_train:
    st.markdown("Lanza el entrenamiento con búsqueda de hiperparámetros sobre 3 modelos candidatos.")
    if st.button("🚀 Entrenar modelo"):
        with st.spinner("Entrenando…"):
            try:
                result = train()
                st.success("Entrenamiento completado")

                metrics = result.get("metrics", result)
                st.subheader(f"Mejor modelo: `{metrics.get('selected_model')}`")
                st.json(metrics.get("candidates", {}))
            except Exception as e:
                st.error(f"Error durante el entrenamiento: {e}")

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
        try:
            features = [sepal_length, sepal_width, petal_length, petal_width]
            result = predict(features)

            st.success(f"Predicción: **{result['prediction']}**")
            st.bar_chart(result["probabilities"])
        except Exception as e:
            st.error(f"Error en la predicción: {e}")
