"""
src.modules.iris_classifier.entrypoint
───────────────────────────────────────
INTERFAZ PÚBLICA del módulo.
Funciones expuestas: train(), predict().
"""

from src.modules.iris_classifier.training.trainer import run_training
from src.modules.iris_classifier.inference.predictor import run_prediction


def train() -> dict:
    """Entrena el modelo configurado y devuelve las métricas."""
    return run_training()


def predict(features: list[float]) -> dict:
    """Predice la clase de Iris a partir de 4 features.

    Args:
        features: [sepal_length, sepal_width, petal_length, petal_width]

    Returns:
        {"prediction": str, "probabilities": dict}
    """
    return run_prediction(features)
