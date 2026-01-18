"""
inspect_model.py

Goal:
- Load the saved model artifact (models/model.joblib)
- Print model details to understand what's inside the binary file

This is NOT required for production serving,
but it is great for learning and debugging.
"""

from __future__ import annotations

import joblib


def main(model_path: str = "models/model.joblib") -> None:
    # Load the serialized model from disk (binary artifact).
    model = joblib.load(model_path)

    # Print the model type and parameters.
    print("Model type:", type(model))
    print("Model params:", model.get_params())

    # LogisticRegression stores learned weights in coef_ and intercept_.
    # coef_ shape: (n_classes, n_features)
    print("coef_ shape:", model.coef_.shape)
    print("intercept_ shape:", model.intercept_.shape)

    # Print the class labels the model was trained on.
    print("classes_:", model.classes_)


if __name__ == "__main__":
    main()
