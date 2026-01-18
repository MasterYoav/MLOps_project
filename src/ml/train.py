"""
train.py

Goal:
- Train a simple machine learning model (Logistic Regression) on the Iris dataset
- Evaluate it on a test split
- Save the trained model as a binary artifact (Joblib) to: models/model.joblib

Why this matters for MLOps:
- In real systems, training code is separate from serving code.
- Training produces an artifact (the model file) that serving loads later.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# A small typed container to return useful results from the training function.
# This makes the training code easier to test and integrate into pipelines.
@dataclass(frozen=True)
class TrainResult:
    accuracy: float      # Model accuracy on the test split
    model_path: str      # Where the model file was saved


def train_and_save(model_path: str = "models/model.joblib") -> TrainResult:
    """
    Train a model and save it to disk.

    Args:
        model_path: Path to save the trained model artifact (joblib file)

    Returns:
        TrainResult containing evaluation metric(s) and saved artifact path.
    """

    # Ensure the output directory exists before trying to write the model file.
    # For example, if model_path="models/model.joblib", we must create "models/".
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Load the Iris dataset (classic toy dataset).
    # X: features (150 samples x 4 features)
    # y: labels (150 samples), values are 0,1,2 corresponding to 3 species
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Split into train and test sets.
    # test_size=0.2 => 80% training, 20% testing
    # random_state fixed => reproducible split
    # stratify=y => preserve class proportions in train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Create the model.
    # Logistic Regression is a simple baseline that also supports predict_proba,
    # which is useful for APIs returning probabilities.
    # max_iter increases iterations so the optimizer converges reliably.
    model = LogisticRegression(max_iter=200)

    # Fit the model on the training data.
    model.fit(X_train, y_train)

    # Evaluate on the test data.
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    # Persist (serialize) the trained model to disk.
    # The output is a binary file; it's not human-readable.
    joblib.dump(model, model_path)

    # Return info useful for logs/pipelines.
    return TrainResult(accuracy=float(acc), model_path=model_path)


if __name__ == "__main__":
    # This section runs only when you execute:
    #   python -m src.ml.train
    #
    # It trains the model and prints where it was saved and its accuracy.
    result = train_and_save()
    print(f"Saved model to: {result.model_path}")
    print(f"Accuracy: {result.accuracy:.4f}")
