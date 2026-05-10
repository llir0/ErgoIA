"""Reglas geometricas para clasificar postura con landmarks de MediaPipe."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

from features import LOWER_BODY_LANDMARKS, extract_posture_features, get_landmark_list, landmarks_visible


@dataclass
class PostureResult:
    key: str
    label: str
    metrics: Dict[str, float]


class PostureAnalyzer:
    """Clasificador por reglas para modo demo sin datasets ni modelo entrenado.

    Las reglas usan coordenadas normalizadas de MediaPipe Pose. En imagenes de
    webcam, y crece hacia abajo, por eso una cabeza adelantada suele verse como
    nariz muy separada horizontalmente del centro de hombros y torso inclinado.
    """

    def __init__(
        self,
        shoulder_tilt_threshold: float = 14.0,
        side_lean_threshold: float = 0.16,
        forward_head_threshold: float = 0.24,
        head_drop_threshold: float = 0.075,
        hunched_back_threshold: float = 0.16,
        visibility_threshold: float = 0.35,
    ):
        self.shoulder_tilt_threshold = shoulder_tilt_threshold
        self.side_lean_threshold = side_lean_threshold
        self.forward_head_threshold = forward_head_threshold
        self.head_drop_threshold = head_drop_threshold
        self.hunched_back_threshold = hunched_back_threshold
        self.visibility_threshold = visibility_threshold

    def analyze(self, landmarks: Any) -> PostureResult:
        if landmarks is None:
            return PostureResult("no_person", "No se detecta persona", {})

        lm = get_landmark_list(landmarks)
        metrics = extract_posture_features(lm, self.visibility_threshold)
        if metrics is None:
            return PostureResult("no_person", "No se detecta persona", {})

        shoulder_tilt = metrics["shoulder_tilt_degrees"]
        ear_forward_ratio = metrics["ear_forward_ratio"]
        head_drop_ratio = metrics["head_drop_ratio"]
        side_lean_ratio = metrics["side_lean_ratio"]
        hunch_ratio = metrics["hunch_ratio"]
        hips_visible = bool(metrics["hips_visible"])

        if shoulder_tilt > self.shoulder_tilt_threshold:
            return PostureResult("shoulder_misalignment", "Hombros desalineados", metrics)
        if hips_visible and side_lean_ratio > self.side_lean_threshold:
            return PostureResult("side_lean", "Inclinado hacia un lado", metrics)
        if ear_forward_ratio > self.forward_head_threshold or head_drop_ratio > self.head_drop_threshold:
            return PostureResult("forward_head", "Cabeza adelantada", metrics)
        if hunch_ratio > self.hunched_back_threshold:
            return PostureResult("hunched_back", "Espalda encorvada", metrics)
        return PostureResult("correct", "Postura correcta", metrics)

    def _visible(self, landmarks: Any, indexes: Tuple[int, ...] | list[int]) -> bool:
        return landmarks_visible(landmarks, indexes, self.visibility_threshold)
