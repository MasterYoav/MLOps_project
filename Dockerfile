# Dockerfile
#
# Goal:
# - Build a container image that runs the FastAPI service
# - Install Python dependencies
# - Copy source code + model artifact into the image
#
# Why this matters for MLOps:
# - Docker gives reproducibility: "works on my machine" becomes "works anywhere"
# - This is the standard unit for deployment (Cloud Run / ECS / Kubernetes)

FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Create and set working directory inside the container
WORKDIR /app

# Copy dependency list first (better build caching)
COPY requirements.txt /app/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy application code into the image
COPY src /app/src

# Copy model artifact into the image
# (In larger projects, you'd often download the model from a registry instead.)
COPY models /app/models

# Configure where the service should look for the model artifact
ENV MODEL_PATH=/app/models/model.joblib

# Expose the port Uvicorn will listen on
EXPOSE 8000

# Use Cloud Run's PORT if provided, fallback to 8000
CMD ["sh", "-c", "uvicorn src.app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
