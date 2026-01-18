# MLOps Model Service (Iris) — FastAPI + Docker + CI + GHCR

![CI](https://github.com/MasterYoav/MLOps_project/actions/workflows/ci.yml/badge.svg)
![Release](https://github.com/MasterYoav/MLOps_project/actions/workflows/release.yml/badge.svg)

A small, production-shaped MLOps project that trains a scikit-learn model, serves it via a FastAPI API, tests it with pytest, packages it with Docker, and publishes container images to GitHub Container Registry (GHCR) using GitHub Actions.

## What this project demonstrates
- **Reproducible training artifact**: `src/ml/train.py` trains a Logistic Regression model on Iris and saves `models/model.joblib`.
- **Model serving**: `src/app/main.py` loads the model artifact and exposes:
  - `GET /health` — readiness check (also validates the model can be loaded)
  - `POST /predict` — returns prediction + class probabilities
- **Automated testing**: `pytest` + FastAPI `TestClient` validate the API contract.
- **Containerization**: Docker image runs the service consistently across environments.
- **CI/CD foundations**:
  - **CI** workflow runs tests and validates Docker builds on every push/PR.
  - **Release** workflow builds & pushes images to **GHCR**.

---

## API

### Health
```bash
curl -s http://127.0.0.1:8000/health
```

Expected output:
```json
{"status":"ok","model_path":"/app/models/model.joblib"}
```

### Predict
```bash
curl -s -X POST http://127.0.0.1:8000/predict   -H "Content-Type: application/json"   -d '{"features":[5.1,3.5,1.4,0.2]}'
```

Expected output (example):
```json
{"prediction":0,"probabilities":[0.9783996842813543,0.02160026063077707,5.508786878462229e-08]}
```

Request format:
```json
{ "features": [float, float, float, float] }
```

Response format:
```json
{ "prediction": 0, "probabilities": [0.97, 0.02, 0.00] }
```

---

## Local setup (Python)

Create venv and install dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Train the model artifact
```bash
python -m src.ml.train
```

Expected output (example):
```text
Saved model to: models/model.joblib
Accuracy: 0.9667
```

### Run the service locally
```bash
uvicorn src.app.main:app --reload --port 8000
```

### Run tests
```bash
python -m pytest -q
```

Expected output (example):
```text
..                                                                       [100%]
2 passed in 0.58s
```

---

## Docker

### Build
```bash
docker build -t mlops-model-service:0.1.0 .
```

### Run
```bash
docker run --rm -p 8000:8000 mlops-model-service:0.1.0
```

Verify:
```bash
curl -s http://127.0.0.1:8000/health
```

---

## Pull & run from GHCR

After the release workflow publishes the image:

```bash
docker pull ghcr.io/masteryoav/mlops_project:latest
docker run --rm -p 8000:8000 ghcr.io/masteryoav/mlops_project:latest
```

> If the package is private, you must authenticate to GHCR:
> ```bash
> docker login ghcr.io -u MasterYoav
> ```

---

## Project structure
```
.
├── src/
│   ├── app/
│   │   └── main.py            # FastAPI service (health + predict)
│   └── ml/
│       ├── train.py           # Training script (produces models/model.joblib)
│       └── inspect_model.py   # Optional: inspect saved model details
├── tests/
│   └── test_api.py            # API tests (health + predict contract)
├── models/
│   └── model.joblib           # Generated artifact (created by training)
├── Dockerfile
├── requirements.txt
└── .github/workflows/
    ├── ci.yml                 # tests + docker build
    └── release.yml            # push image to GHCR
```

---

## Notes
- The model artifact is produced by running the training script (and is also generated during CI so builds are reproducible).
- The Dockerfile supports cloud platforms by respecting the `PORT` environment variable (fallback to `8000`).

---

## Next improvements (roadmap)
- Add experiment tracking and a registry (MLflow).
- Add data validation (Pandera / Great Expectations).
- Add monitoring and drift detection (Prometheus/Grafana + Evidently).
- Deploy to a managed service (Cloud Run / ECS).
