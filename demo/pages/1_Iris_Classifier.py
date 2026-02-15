"""
demo.pages.1_Iris_Classifier
──────────────────────────────
Página de Streamlit para el módulo Iris Classifier.
Interfaz mejorada: entrenamiento, predicción interactiva,
visualización de EDA y métricas de evaluación.
"""

import sys
import json
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import streamlit as st
import pandas as pd

from shared.config_loader import load_config
from src.modules.iris_classifier.entrypoint import train, predict

# ── Configuración ────────────────────────────────────────
_config = load_config()
_EDA_DIR = Path(_config["paths"]["data_dir"]) / "files" / "eda" / "iris"
_METRICS_DIR = Path(_config["paths"]["models_dir"]) / "metrics" / "iris_classifier"
_RAW_DIR = Path(_config["paths"]["raw_dir"])

SPECIES_EMOJI = {"setosa": "🌸", "versicolor": "🌺", "virginica": "🌻"}
SPECIES_COLORS = {"setosa": "#2ecc71", "versicolor": "#3498db", "virginica": "#e74c3c"}
FEATURE_LABELS = {
    "sepal_length": "Sepal Length (cm)",
    "sepal_width": "Sepal Width (cm)",
    "petal_length": "Petal Length (cm)",
    "petal_width": "Petal Width (cm)",
}

# ── CSS personalizado ────────────────────────────────────
st.markdown("""
<style>
    /* Fuentes más grandes y legibles */
    .main .block-container { max-width: 1100px; padding-top: 2rem; }
    h1 { font-size: 2.8rem !important; color: #f0f0f0 !important; }
    h2 { font-size: 2rem !important; color: #e8e8e8 !important; }
    h3 { font-size: 1.5rem !important; color: #e0e0e0 !important; }
    p, li, .stMarkdown { font-size: 1.1rem !important; color: #d4d4d4 !important; }
    .stTabs [data-baseweb="tab"] { font-size: 1.15rem !important; padding: 12px 24px; }

    /* Texto claro sobre fondo oscuro */
    .stMarkdown h4, .stMarkdown h5 { color: #e8e8e8 !important; }
    .stMarkdown strong, .stMarkdown b { color: #f0f0f0 !important; }
    label, .stSlider label span, .stNumberInput label span { color: #d4d4d4 !important; }

    /* Tarjetas de métricas */
    .metric-card {
        background: linear-gradient(135deg, #1a1d23 0%, #2d3748 100%);
        border: 1px solid #4a5568;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        margin-bottom: 0.8rem;
    }
    .metric-card h4 {
        color: #a0aec0;
        font-size: 0.95rem !important;
        margin-bottom: 0.3rem;
        font-weight: 500;
    }
    .metric-card .value {
        font-size: 2rem !important;
        font-weight: 700;
    }

    /* Resultado de predicción */
    .prediction-result {
        background: linear-gradient(135deg, #1a3a2a 0%, #1a4731 100%);
        border: 2px solid #38a169;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin: 1.5rem 0;
    }
    .prediction-result .species-name {
        font-size: 2.5rem !important;
        font-weight: 800;
        margin: 0.5rem 0;
    }
    .prediction-result .species-emoji {
        font-size: 3.5rem;
    }

    /* Barras de probabilidad */
    .prob-bar-container { margin: 0.4rem 0; }
    .prob-bar-label {
        font-size: 1rem !important;
        font-weight: 600;
        margin-bottom: 0.2rem;
    }
    .prob-bar-bg {
        background: #2d3748;
        border-radius: 8px;
        height: 28px;
        overflow: hidden;
        position: relative;
    }
    .prob-bar-fill {
        height: 100%;
        border-radius: 8px;
        display: flex;
        align-items: center;
        padding-left: 10px;
        font-weight: 700;
        font-size: 0.9rem;
        color: white;
        transition: width 0.5s ease;
    }

    /* Sliders más grandes y con más separación */
    .stSlider { margin-bottom: 2rem !important; padding-top: 0.5rem; }
    .stSlider > div > div { padding-top: 0.5rem; }
    .stSlider label { font-size: 1.1rem !important; font-weight: 500; color: #d4d4d4 !important; margin-bottom: 0.5rem !important; }
    .stSlider [data-baseweb="slider"] { margin-top: 0.8rem !important; }
    .stSlider [data-testid="stTickBarMin"],
    .stSlider [data-testid="stTickBarMax"] { color: #a0aec0 !important; font-size: 0.85rem !important; }

    /* Botones */
    .stButton > button {
        font-size: 1.15rem !important;
        padding: 0.7rem 2rem;
        border-radius: 10px;
        font-weight: 600;
    }

    /* Imágenes EDA */
    .eda-section img { border-radius: 12px; border: 1px solid #333; }

    /* Info boxes */
    .info-box {
        background: #1a2332;
        border-left: 4px solid #3498db;
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.2rem;
        margin: 1rem 0;
        font-size: 1.05rem !important;
        color: #d4d4d4 !important;
    }
    .info-box b, .info-box strong { color: #e8e8e8 !important; }

    /* Separador visual */
    .section-divider {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #4a5568, transparent);
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ── Encabezado ───────────────────────────────────────────
st.markdown("# 🌸 Iris Classifier")
st.markdown(
    '<p style="font-size: 1.25rem; color: #a0aec0;">'
    "Clasificación de especies de Iris con Machine Learning — "
    "entrena modelos, explora los datos y haz predicciones interactivas."
    "</p>",
    unsafe_allow_html=True,
)
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)


# ── Funciones auxiliares ─────────────────────────────────
def _load_metrics() -> dict | None:
    """Carga métricas de entrenamiento si existen."""
    path = _METRICS_DIR / "training_metrics.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def _load_dataset() -> pd.DataFrame | None:
    """Carga el dataset Iris si existe."""
    path = _RAW_DIR / "iris.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


def _eda_image_exists(name: str) -> bool:
    return (_EDA_DIR / name).exists()


def _show_eda_image(name: str, caption: str = ""):
    path = _EDA_DIR / name
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(
            f"📊 Imagen no disponible. Ejecuta `python scripts/iris_eda.py` "
            f"para generar las gráficas de EDA."
        )


def _render_prob_bars(probabilities: dict):
    """Renderiza barras de probabilidad estilizadas."""
    for species, prob in sorted(probabilities.items(), key=lambda x: -x[1]):
        pct = prob * 100
        color = SPECIES_COLORS.get(species, "#718096")
        emoji = SPECIES_EMOJI.get(species, "🌿")
        st.markdown(f"""
        <div class="prob-bar-container">
            <div class="prob-bar-label">{emoji} {species.capitalize()}</div>
            <div class="prob-bar-bg">
                <div class="prob-bar-fill" style="width: {max(pct, 5):.0f}%; background: {color};">
                    {pct:.1f}%
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


def _render_metric_card(label: str, value: str, color: str = "#63b3ed"):
    st.markdown(f"""
    <div class="metric-card">
        <h4>{label}</h4>
        <div class="value" style="color: {color};">{value}</div>
    </div>
    """, unsafe_allow_html=True)


# ── Pestañas principales ─────────────────────────────────
tab_predict, tab_train, tab_eda = st.tabs([
    "🔮  Predecir especie",
    "🚀  Entrenar modelo",
    "📊  Explorar datos (EDA)",
])


# ══════════════════════════════════════════════════════════
# TAB: PREDICCIÓN
# ══════════════════════════════════════════════════════════
with tab_predict:
    st.markdown("### 📋 Introduce las medidas de una flor Iris")
    st.markdown(
        '<div class="info-box">'
        "Ajusta los <b>sliders</b> con las 4 medidas de la flor. "
        "Los rangos corresponden a los valores reales del dataset. "
        "Consulta la pestaña <b>Explorar datos</b> para ver las distribuciones por especie."
        "</div>",
        unsafe_allow_html=True,
    )

    # Cargar dataset para obtener rangos reales
    df = _load_dataset()
    if df is not None:
        ranges = {col: (float(df[col].min()), float(df[col].max())) for col in FEATURE_LABELS}
    else:
        ranges = {
            "sepal_length": (4.3, 7.9),
            "sepal_width": (2.0, 4.4),
            "petal_length": (1.0, 6.9),
            "petal_width": (0.1, 2.5),
        }

    st.markdown("")  # Spacing
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("**🌿 Sépalo**")
        sepal_length = st.slider(
            "Longitud del sépalo (cm)",
            min_value=ranges["sepal_length"][0],
            max_value=ranges["sepal_length"][1],
            value=5.1, step=0.1,
        )
        sepal_width = st.slider(
            "Anchura del sépalo (cm)",
            min_value=ranges["sepal_width"][0],
            max_value=ranges["sepal_width"][1],
            value=3.5, step=0.1,
        )

    with col2:
        st.markdown("**🌷 Pétalo**")
        petal_length = st.slider(
            "Longitud del pétalo (cm)",
            min_value=ranges["petal_length"][0],
            max_value=ranges["petal_length"][1],
            value=1.4, step=0.1,
        )
        petal_width = st.slider(
            "Anchura del pétalo (cm)",
            min_value=ranges["petal_width"][0],
            max_value=ranges["petal_width"][1],
            value=0.2, step=0.1,
        )

    st.markdown("")
    predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])
    with predict_col2:
        predict_clicked = st.button("🔮 Predecir especie", use_container_width=True, type="primary")

    if predict_clicked:
        try:
            features = [sepal_length, sepal_width, petal_length, petal_width]
            result = predict(features)

            species = result["prediction"]
            emoji = SPECIES_EMOJI.get(species, "🌿")
            color = SPECIES_COLORS.get(species, "#63b3ed")

            # Resultado principal
            st.markdown(f"""
            <div class="prediction-result">
                <div class="species-emoji">{emoji}</div>
                <div class="species-name" style="color: {color};">
                    {species.capitalize()}
                </div>
                <p style="font-size: 1.1rem; color: #a0aec0; margin: 0;">
                    Especie predicha por el modelo
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Probabilidades
            st.markdown("#### Probabilidades por especie")
            _render_prob_bars(result["probabilities"])

            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

            # Contexto visual: mostrar dónde cae la predicción
            st.markdown("#### 📊 Contexto: ¿dónde caen tus valores?")
            st.markdown(
                '<div class="info-box">'
                "Estas gráficas muestran cómo se distribuyen los valores de cada feature "
                "en el dataset original. Las líneas rojas indican los valores que has introducido."
                "</div>",
                unsafe_allow_html=True,
            )

            if df is not None:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                import seaborn as sns

                sns.set_theme(style="whitegrid")
                plt.rcParams.update({
                    "figure.facecolor": "#0e1117",
                    "axes.facecolor": "#1a1d23",
                    "axes.edgecolor": "#444",
                    "axes.labelcolor": "#fafafa",
                    "text.color": "#fafafa",
                    "xtick.color": "#ccc",
                    "ytick.color": "#ccc",
                    "grid.color": "#333",
                    "legend.facecolor": "#1a1d23",
                    "legend.edgecolor": "#444",
                })

                feature_values = {
                    "sepal_length": sepal_length,
                    "sepal_width": sepal_width,
                    "petal_length": petal_length,
                    "petal_width": petal_width,
                }

                fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
                palette = SPECIES_COLORS

                for ax, col in zip(axes, FEATURE_LABELS.keys()):
                    for sp, clr in palette.items():
                        subset = df[df["species"] == sp][col]
                        ax.hist(subset, bins=12, alpha=0.4, color=clr, label=sp.capitalize(),
                                edgecolor="none")
                    ax.axvline(feature_values[col], color="#ff4757", linewidth=2.5,
                               linestyle="--", label="Tu valor")
                    ax.set_xlabel(FEATURE_LABELS[col], fontsize=11)
                    ax.set_ylabel("")
                    ax.legend(fontsize=9)

                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("Ejecuta `python scripts/iris_ingest.py` para cargar el dataset y ver el contexto visual.")

            # Métricas del modelo
            metrics = _load_metrics()
            if metrics:
                st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
                st.markdown("#### 🏆 Rendimiento del modelo actual")

                report = metrics.get("classification_report", {})
                selected = metrics.get("selected_model", "—")

                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    _render_metric_card("Modelo", selected.replace("_", " ").title(), "#63b3ed")
                with m2:
                    acc = report.get("accuracy", 0)
                    _render_metric_card("Accuracy", f"{acc:.1%}", "#48bb78")
                with m3:
                    macro_f1 = report.get("macro avg", {}).get("f1-score", 0)
                    _render_metric_card("F1 (macro)", f"{macro_f1:.1%}", "#ed8936")
                with m4:
                    candidates = metrics.get("candidates", {})
                    cv = candidates.get(selected, {}).get("cv_score", 0)
                    _render_metric_card("CV Score", f"{cv:.1%}", "#9f7aea")

        except FileNotFoundError:
            st.warning(
                "⚠️ No hay modelo entrenado. Ve a la pestaña **Entrenar modelo** primero.",
                icon="⚠️",
            )
        except Exception as e:
            st.error(f"Error en la predicción: {e}")


# ══════════════════════════════════════════════════════════
# TAB: ENTRENAMIENTO
# ══════════════════════════════════════════════════════════
with tab_train:
    st.markdown("### ⚙️ Entrenamiento con búsqueda de hiperparámetros")

    st.markdown(
        '<div class="info-box">'
        "El proceso entrena <b>3 modelos candidatos</b> (Random Forest, Gradient Boosting, SVM) "
        "con <b>GridSearch + validación cruzada (5 folds)</b>. Se selecciona automáticamente "
        "el mejor modelo según la puntuación CV."
        "</div>",
        unsafe_allow_html=True,
    )

    # Mostrar métricas existentes si hay
    existing_metrics = _load_metrics()
    if existing_metrics:
        st.markdown("#### 📋 Último entrenamiento registrado")

        selected = existing_metrics.get("selected_model", "—")
        candidates = existing_metrics.get("candidates", {})
        report = existing_metrics.get("classification_report", {})

        # Tarjetas resumen
        c1, c2, c3 = st.columns(3)
        with c1:
            _render_metric_card("Mejor modelo", selected.replace("_", " ").title(), "#48bb78")
        with c2:
            acc = report.get("accuracy", 0)
            _render_metric_card("Test Accuracy", f"{acc:.1%}", "#63b3ed")
        with c3:
            macro_f1 = report.get("macro avg", {}).get("f1-score", 0)
            _render_metric_card("F1 Macro", f"{macro_f1:.1%}", "#ed8936")

        # Tabla comparativa de candidatos
        st.markdown("#### 🏅 Comparativa de modelos candidatos")
        candidate_rows = []
        for name, data in candidates.items():
            candidate_rows.append({
                "Modelo": name.replace("_", " ").title(),
                "Mejores Params": str(data.get("best_params", {})),
                "CV Score": f"{data.get('cv_score', 0):.2%}",
                "Test Accuracy": f"{data.get('test_accuracy', 0):.2%}",
                "✅": "🏆" if name == selected else "",
            })
        st.dataframe(
            pd.DataFrame(candidate_rows),
            use_container_width=True,
            hide_index=True,
        )

        # Classification report por clase
        st.markdown("#### 📊 Reporte de clasificación por especie")
        report_rows = []
        for species in ["setosa", "versicolor", "virginica"]:
            sp_data = report.get(species, {})
            report_rows.append({
                "Especie": f"{SPECIES_EMOJI.get(species, '')} {species.capitalize()}",
                "Precision": f"{sp_data.get('precision', 0):.2%}",
                "Recall": f"{sp_data.get('recall', 0):.2%}",
                "F1-Score": f"{sp_data.get('f1-score', 0):.2%}",
                "Support": int(sp_data.get("support", 0)),
            })
        st.dataframe(
            pd.DataFrame(report_rows),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Botón de entrenamiento
    train_col1, train_col2, train_col3 = st.columns([1, 2, 1])
    with train_col2:
        train_clicked = st.button(
            "🚀 Lanzar nuevo entrenamiento",
            use_container_width=True,
            type="primary",
        )

    if train_clicked:
        with st.spinner("🔄 Entrenando modelos… esto puede tardar unos segundos."):
            try:
                result = train()
                st.success("✅ ¡Entrenamiento completado con éxito!")
                st.balloons()

                metrics = result.get("metrics", result)
                selected = metrics.get("selected_model", "")
                st.markdown(
                    f"**Mejor modelo seleccionado:** `{selected.replace('_', ' ').title()}`"
                )
                st.json(metrics.get("candidates", {}))
                st.rerun()  # Refrescar para mostrar las nuevas métricas
            except Exception as e:
                st.error(f"❌ Error durante el entrenamiento: {e}")


# ══════════════════════════════════════════════════════════
# TAB: EDA
# ══════════════════════════════════════════════════════════
with tab_eda:
    st.markdown("### 📊 Análisis Exploratorio del Dataset Iris")

    eda_available = _EDA_DIR.exists() and any(_EDA_DIR.glob("*.png"))

    if not eda_available:
        st.warning(
            "⚠️ No se encontraron gráficas de EDA. Ejecuta el siguiente comando para generarlas:",
            icon="⚠️",
        )
        st.code("python scripts/iris_eda.py", language="bash")
        st.markdown(
            '<div class="info-box">'
            "El script genera automáticamente distribuciones, box plots, violin plots, "
            "correlaciones y un resumen estadístico completo del dataset."
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        # Balance de clases y resumen
        col_balance, col_summary = st.columns(2)
        with col_balance:
            st.markdown("#### ⚖️ Balance de clases")
            _show_eda_image("class_balance.png")
        with col_summary:
            st.markdown("#### 📋 Resumen estadístico")
            _show_eda_image("feature_summary.png")

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        # Distribuciones
        st.markdown("#### 📈 Distribuciones de features por especie")
        st.markdown(
            '<div class="info-box">'
            "Estas gráficas muestran la distribución de valores de cada medición "
            "separada por especie. Útil para entender qué rangos son típicos de cada clase."
            "</div>",
            unsafe_allow_html=True,
        )
        _show_eda_image("distributions.png", "Histogramas + KDE por especie")

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        # Box plots y Violin plots lado a lado
        col_box, col_violin = st.columns(2)
        with col_box:
            st.markdown("#### 📦 Box Plots")
            _show_eda_image("boxplots.png", "Mediana, cuartiles y outliers")
        with col_violin:
            st.markdown("#### 🎻 Violin Plots")
            _show_eda_image("violin_plots.png", "Distribución completa + cuartiles")

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        # Correlaciones y Pairplot
        st.markdown("#### 🔗 Correlaciones entre features")
        col_corr, col_pair = st.columns([1, 2])
        with col_corr:
            _show_eda_image("correlation_heatmap.png", "Matriz de correlación")
        with col_pair:
            st.markdown("#### 🔍 Scatter Matrix (Pairplot)")
            _show_eda_image("pairplot.png", "Relaciones entre pares de features")
