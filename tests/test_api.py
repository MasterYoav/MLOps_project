"""
test_api.py

Goal:
- Automated tests for the FastAPI service
- Ensure /health works (service is ready + model can load)
- Ensure /predict returns the correct response shape

Why this matters for MLOps:
- Tests catch regressions early (before Docker/CI/Deploy).
- CI pipelines run these tests on every push/PR.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

# Import the FastAPI app from your service module
from src.app.main import app

# TestClient runs the app in-process (no need to start uvicorn).
client = TestClient(app)


def test_health_endpoint_returns_ok():
    """
    /health should:
    - return HTTP 200
    - return JSON with status=ok
    - also implicitly verify the model can be loaded
    """
    resp = client.get("/health")
    assert resp.status_code == 200

    data = resp.json()
    assert data["status"] == "ok"
    assert "model_path" in data


def test_predict_endpoint_returns_prediction_and_probabilities():
    """
    /predict should:
    - accept a JSON body with 4 floats in "features"
    - return HTTP 200
    - return JSON with:
        - an integer prediction
        - a list of 3 probabilities (for 3 Iris classes)
    """
    payload = {"features": [5.1, 3.5, 1.4, 0.2]}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200

    data = resp.json()

    # Basic schema checks
    assert isinstance(data["prediction"], int)
    assert isinstance(data["probabilities"], list)
    assert len(data["probabilities"]) == 3

    # Probabilities should sum to ~1 (floating point tolerance)
    prob_sum = sum(data["probabilities"])
    assert abs(prob_sum - 1.0) < 1e-6
