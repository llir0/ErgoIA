"""Extraccion de features de postura para reglas y aprendizaje automatico."""

from __future__ import annotations

from typing import Any

from utils import angle_degrees, euclidean_distance, midpoint, slope_degrees


UPPER_BODY_LANDMARKS = {
    "nose": 0,
    "left_eye": 2,
    "right_eye": 5,
    "left_ear": 7,
    "right_ear": 8,
    "mouth_left": 9,
    "mouth_right": 10,
    "left_shoulder": 11,
    "right_shoulder": 12,
}

LOWER_BODY_LANDMARKS = {
    "left_hip": 23,
    "right_hip": 24,
}

FEATURE_COLUMNS = [
    "shoulder_tilt_degrees",
    "head_forward_ratio",
    "head_drop_ratio",
    "ear_forward_ratio",
    "side_lean_ratio",
    "hunch_ratio",
    "nose_to_shoulder_y_ratio",
    "ear_to_shoulder_y_ratio",
    "neck_torso_angle",
    "shoulder_width",
    "torso_height",
    "hips_visible",
]


def get_landmark_list(landmarks: Any):
    if landmarks is None:
        return None
    return landmarks.landmark if hasattr(landmarks, "landmark") else landmarks


def landmarks_visible(landmarks: Any, indexes, threshold: float = 0.35) -> bool:
    lm = get_landmark_list(landmarks)
    if lm is None:
        return False
    return all(getattr(lm[index], "visibility", 1.0) >= threshold for index in indexes)


def extract_posture_features(landmarks: Any, visibility_threshold: float = 0.35) -> dict[str, float] | None:
    lm = get_landmark_list(landmarks)
    if lm is None or not landmarks_visible(lm, UPPER_BODY_LANDMARKS.values(), visibility_threshold):
        return None

    required = {**UPPER_BODY_LANDMARKS, **LOWER_BODY_LANDMARKS}
    points = {name: (lm[index].x, lm[index].y) for name, index in required.items()}

    left_shoulder = points["left_shoulder"]
    right_shoulder = points["right_shoulder"]
    shoulder_mid = midpoint(left_shoulder, right_shoulder)
    ear_mid = midpoint(points["left_ear"], points["right_ear"])
    mouth_mid = midpoint(points["mouth_left"], points["mouth_right"])
    nose = points["nose"]

    hips_visible = landmarks_visible(lm, LOWER_BODY_LANDMARKS.values(), visibility_threshold)
    hip_mid = midpoint(points["left_hip"], points["right_hip"]) if hips_visible else None

    shoulder_width = max(euclidean_distance(left_shoulder, right_shoulder), 1e-6)
    torso_height = max(euclidean_distance(shoulder_mid, hip_mid), 1e-6) if hip_mid else shoulder_width * 1.8
    posture_reference = torso_height if hips_visible else shoulder_width

    shoulder_tilt = abs(slope_degrees(left_shoulder, right_shoulder))
    head_forward_ratio = abs(nose[0] - shoulder_mid[0]) / posture_reference
    ear_forward_ratio = abs(ear_mid[0] - shoulder_mid[0]) / posture_reference
    side_lean_ratio = abs(shoulder_mid[0] - hip_mid[0]) / torso_height if hip_mid else 0.0
    nose_to_shoulder_y_ratio = (shoulder_mid[1] - nose[1]) / torso_height
    ear_to_shoulder_y_ratio = (shoulder_mid[1] - ear_mid[1]) / torso_height
    # En vista frontal de laptop, una postura encorvada suele bajar la cabeza
    # hacia los hombros. Por eso el score crece cuando esta distancia vertical
    # baja de un rango esperado, en vez de esperar que la nariz quede debajo de
    # los hombros.
    expected_head_height = 0.46 if not hips_visible else 0.38
    head_drop_ratio = max(0.0, expected_head_height - nose_to_shoulder_y_ratio)
    front_hunch_ratio = head_drop_ratio
    neck_torso_angle = angle_degrees(ear_mid, shoulder_mid, hip_mid) if hip_mid else 180.0
    lateral_hunch_ratio = max(0.0, (165.0 - neck_torso_angle) / 45.0) if hips_visible else 0.0
    hunch_ratio = max(front_hunch_ratio, lateral_hunch_ratio)

    features = {
        "shoulder_tilt_degrees": shoulder_tilt,
        "head_forward_ratio": head_forward_ratio,
        "head_drop_ratio": head_drop_ratio,
        "ear_forward_ratio": ear_forward_ratio,
        "side_lean_ratio": side_lean_ratio,
        "hunch_ratio": hunch_ratio,
        "nose_to_shoulder_y_ratio": nose_to_shoulder_y_ratio,
        "ear_to_shoulder_y_ratio": ear_to_shoulder_y_ratio,
        "neck_torso_angle": neck_torso_angle,
        "shoulder_width": shoulder_width,
        "torso_height": torso_height,
        "hips_visible": 1.0 if hips_visible else 0.0,
    }
    return {column: float(features[column]) for column in FEATURE_COLUMNS}
