"""
tests/unit/test_iris_predictor.py
──────────────────────────────────
Tests unitarios para la lógica de predicción de Iris.
"""

import pytest
import numpy as np


class TestRunPrediction:
    """Tests para la función de predicción (requiere modelo entrenado)."""

    def test_prediction_returns_expected_keys(self):
        """La respuesta debe contener 'prediction' y 'probabilities'."""
        from src.modules.iris_classifier.entrypoint import predict

        result = predict([5.1, 3.5, 1.4, 0.2])
        assert "prediction" in result
        assert "probabilities" in result

    def test_prediction_species_is_valid(self):
        """La especie predicha debe ser una de las tres clases de Iris."""
        from src.modules.iris_classifier.entrypoint import predict

        result = predict([5.1, 3.5, 1.4, 0.2])
        assert result["prediction"] in ["setosa", "versicolor", "virginica"]

    def test_probabilities_sum_to_one(self):
        """Las probabilidades deben sumar ~1.0."""
        from src.modules.iris_classifier.entrypoint import predict

        result = predict([6.7, 3.0, 5.2, 2.3])
        total = sum(result["probabilities"].values())
        assert abs(total - 1.0) < 0.01
