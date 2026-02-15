"""
tests/integration/test_api.py
──────────────────────────────
Tests de integración para la API FastAPI.
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_ok(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestIrisEndpoints:
    def test_predict_returns_200(self):
        """El endpoint de predicción debe devolver 200 (requiere modelo entrenado)."""
        response = client.post(
            "/iris/predict",
            json={
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "probabilities" in data

    def test_predict_validates_input(self):
        """Debe devolver 422 si falta un campo."""
        response = client.post(
            "/iris/predict",
            json={"sepal_length": 5.1},
        )
        assert response.status_code == 422


class TestChatbotEndpoints:
    def test_chat_validates_input(self):
        """Debe devolver 422 si falta un campo."""
        response = client.post(
            "/chatbot/chat",
            json={"message": "hola"},  # falta session_id
        )
        assert response.status_code == 422
