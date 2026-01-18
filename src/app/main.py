"""
main.py

Goal:
- Expose a simple prediction service over HTTP (FastAPI)
- Load the trained model artifact (models/model.joblib)
- Provide:
  1) GET  /health   -> quick readiness check (and model load check)
  2) POST /predict  -> return predicted class + probabilities

Why this matters for MLOps:
- Serving is a separate concern from training.
- The service should be stateless and reproducible: it loads a versioned artifact.
"""

from __future__ import annotations

import os
from typing import List, Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# We load the model from an environment variable to make it configurable in Docker/Cloud.
# If MODEL_PATH isn't set, we default to the local path used by the training script.
MODEL_PATH = os.getenv("MODEL_PATH", "models/model.joblib")

app = FastAPI(title="MLOps Model Service", version="0.1.0")

# -----------------------------
# Request/Response Schemas
# -----------------------------

class PredictRequest(BaseModel):
    """
    The request body expected by /predict.

    Iris has exactly 4 numeric features:
    [sepal_length, sepal_width, petal_length, petal_width]
    """
    features: List[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="Exactly 4 floats for the Iris dataset features",
        examples=[[5.1, 3.5, 1.4, 0.2]],
    )


class PredictResponse(BaseModel):
    """
    What we return to the client.
    - prediction: the predicted class index (0/1/2)
    - probabilities: per-class probabilities (length 3)
    """
    prediction: int
    probabilities: List[float]


# -----------------------------
# Model loading (cached in memory)
# -----------------------------

_model: Optional[object] = None

def get_model():
    """
    Lazy-load the model and cache it in memory.
    This avoids re-loading the model from disk on every request, which is slow.

    Important:
    - In production, the service process stays alive; caching is fine.
    - If you redeploy a new model, you redeploy a new container/service version.
    """
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model file not found at '{MODEL_PATH}'. "
                "Make sure you ran training or set MODEL_PATH correctly."
            )
        _model = joblib.load(MODEL_PATH)
    return _model


# -----------------------------
# Routes
# -----------------------------

@app.get("/health")
def health():
    """
    Health/readiness endpoint.

    We attempt to load the model here.
    If the model can't load, this endpoint should fail, signaling the service is not ready.
    """
    try:
        get_model()
    except Exception as e:
        # For real production, you'd log the exception too.
        raise HTTPException(status_code=500, detail=str(e))
    return {"status": "ok", "model_path": MODEL_PATH}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    Prediction endpoint.

    Steps:
    1) Load model (cached after first request)
    2) Validate/reshape input
    3) Predict probabilities and pick the most likely class
    """
    try:
        model = get_model()

        # Convert list -> numpy array and reshape to (1, 4)
        x = np.array(req.features, dtype=float).reshape(1, -1)

        # Predict class probabilities (LogisticRegression supports predict_proba)
        proba = model.predict_proba(x)[0].tolist()

        # Choose the class with maximum probability
        pred = int(np.argmax(proba))

        return PredictResponse(prediction=pred, probabilities=proba)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
