"""
src.modules.iris_classifier.inference.predictor
────────────────────────────────────────────────
Carga del modelo y predicción.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from shared.config_loader import load_config

_config = load_config()
_ARTIFACTS_DIR = Path(_config["paths"]["models_dir"]) / "artifacts" / "iris_classifier"

TARGET_NAMES = ["setosa", "versicolor", "virginica"]

_model = None  # Cache en memoria


def _load_model():
    global _model
    if _model is None:
        model_path = _ARTIFACTS_DIR / "model.pkl"
        if not model_path.exists():
            raise FileNotFoundError(
                f"No se encontró el modelo en {model_path}. Ejecuta primero el entrenamiento."
            )
        _model = joblib.load(model_path)
    return _model


def run_prediction(features: list[float]) -> dict:
    """Predice la especie de Iris dadas 4 features.

    Returns:
        {"prediction": str, "probabilities": {species: float, ...}}
    """
    model = _load_model()
    feature_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    X = pd.DataFrame([features], columns=feature_names)

    pred_idx = model.predict(X)[0]
    proba = model.predict_proba(X)[0]

    return {
        "prediction": TARGET_NAMES[pred_idx],
        "probabilities": {name: round(float(p), 4) for name, p in zip(TARGET_NAMES, proba)},
    }
