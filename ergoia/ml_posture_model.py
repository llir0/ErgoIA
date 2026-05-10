"""Carga e inferencia de modelo supervisado de postura."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from config import POSTURE_LABELS
from features import FEATURE_COLUMNS, extract_posture_features
from posture_rules import PostureResult


DEFAULT_MODEL_PATH = Path("models/posture_random_forest.joblib")


class MLPostureClassifier:
    def __init__(self, model_path: str | Path = DEFAULT_MODEL_PATH):
        self.model_path = Path(model_path)
        self.model = joblib.load(self.model_path)

    def predict(self, landmarks: Any) -> PostureResult:
        features = extract_posture_features(landmarks)
        if features is None:
            return PostureResult("no_person", POSTURE_LABELS["no_person"], {})
        row = pd.DataFrame([[features[column] for column in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)
        try:
            key = str(self.model.predict(row)[0])
        except ValueError as exc:
            raise RuntimeError(
                "El modelo entrenado usa features antiguas. Recolecta datos y entrena de nuevo con train_model.py."
            ) from exc
        confidence = 0.0
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(row)[0]
            confidence = float(max(probabilities))
        metrics = {**features, "model_confidence": confidence}
        return PostureResult(key, POSTURE_LABELS.get(key, key), metrics)

    def analyze(self, landmarks: Any) -> PostureResult:
        return self.predict(landmarks)


def load_classifier_if_available(model_path: str | Path = DEFAULT_MODEL_PATH) -> MLPostureClassifier | None:
    path = Path(model_path)
    if not path.exists():
        return None
    return MLPostureClassifier(path)
