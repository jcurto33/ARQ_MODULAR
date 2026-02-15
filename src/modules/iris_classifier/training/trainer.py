"""
src.modules.iris_classifier.training.trainer
─────────────────────────────────────────────
Entrenamiento con búsqueda de hiperparámetros.
Prueba varios modelos y guarda el mejor según configuración.
"""

import json
import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score

from shared.config_loader import load_config
from src.modules.iris_classifier.data_processing.loader import load_splits

logger = logging.getLogger(__name__)
_config = load_config()
_ARTIFACTS_DIR = Path(_config["paths"]["models_dir"]) / "artifacts" / "iris_classifier"
_METRICS_DIR = Path(_config["paths"]["models_dir"]) / "metrics" / "iris_classifier"

# ── Candidatos para la búsqueda de modelos ────────────────
CANDIDATES = {
    "random_forest": {
        "estimator": RandomForestClassifier(random_state=42),
        "params": {
            "n_estimators": [50, 100],
            "max_depth": [3, 5, None],
        },
    },
    "gradient_boosting": {
        "estimator": GradientBoostingClassifier(random_state=42),
        "params": {
            "n_estimators": [50, 100],
            "max_depth": [3, 5],
            "learning_rate": [0.05, 0.1],
        },
    },
    "svm": {
        "estimator": SVC(probability=True, random_state=42),
        "params": {
            "C": [0.1, 1, 10],
            "kernel": ["rbf", "linear"],
        },
    },
}

TARGET_NAMES = ["setosa", "versicolor", "virginica"]


def run_training() -> dict:
    """Ejecuta grid search sobre todos los candidatos, guarda el mejor modelo."""
    X_train, X_test, y_train, y_test = load_splits()

    best_score = 0.0
    best_name = ""
    best_model = None
    results = {}

    # ── Probar cada candidato ─────────────────────────────
    for name, spec in CANDIDATES.items():
        logger.info("Entrenando candidato: %s", name)
        gs = GridSearchCV(
            spec["estimator"],
            spec["params"],
            cv=5,
            scoring="accuracy",
            n_jobs=-1,
        )
        gs.fit(X_train, y_train)

        test_acc = accuracy_score(y_test, gs.best_estimator_.predict(X_test))
        results[name] = {
            "best_params": gs.best_params_,
            "cv_score": round(gs.best_score_, 4),
            "test_accuracy": round(test_acc, 4),
        }
        logger.info("  %s → CV=%.4f  Test=%.4f", name, gs.best_score_, test_acc)

        if gs.best_score_ > best_score:
            best_score = gs.best_score_
            best_name = name
            best_model = gs.best_estimator_

    # ── Guardar artefacto y métricas ──────────────────────
    _ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    _METRICS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = _ARTIFACTS_DIR / "model.pkl"
    joblib.dump(best_model, model_path)

    y_pred = best_model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=TARGET_NAMES, output_dict=True)

    metrics = {
        "selected_model": best_name,
        "candidates": results,
        "classification_report": report,
    }
    metrics_path = _METRICS_DIR / "training_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False, default=str)

    logger.info("Mejor modelo: %s (guardado en %s)", best_name, model_path)
    return metrics
