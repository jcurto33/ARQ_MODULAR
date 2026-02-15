"""
api.routers.iris_router
───────────────────────
Endpoints del módulo Iris Classifier.
"""

from fastapi import APIRouter
from pydantic import BaseModel

from src.modules.iris_classifier.entrypoint import train, predict

router = APIRouter()


class PredictRequest(BaseModel):
    """4 features de una flor Iris."""
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@router.post("/train")
def train_model():
    """Entrena el modelo con búsqueda de hiperparámetros."""
    metrics = train()
    return {"message": "Entrenamiento completado", "metrics": metrics}


@router.post("/predict")
def predict_species(req: PredictRequest):
    """Predice la especie de Iris."""
    features = [req.sepal_length, req.sepal_width, req.petal_length, req.petal_width]
    return predict(features)
