"""Entrena Random Forest con MultiPosture y lo aplica a ErgoIA."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from download_datasets import MULTIPOSTURE_PATH, download_multiposture
from features import FEATURE_COLUMNS, extract_posture_features
from ml_posture_model import DEFAULT_MODEL_PATH


LABEL_MAP = {
    "TUP": "correct",
    "TLF": "hunched_back",
    "TLR": "side_lean",
    "TLL": "side_lean",
}

LANDMARK_NAMES = [
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
]


@dataclass
class Landmark:
    x: float
    y: float
    z: float
    visibility: float = 1.0


def row_to_landmarks(row: pd.Series) -> list[Landmark]:
    return [
        Landmark(
            x=float(row[f"{name}_x"]),
            y=float(row[f"{name}_y"]),
            z=float(row[f"{name}_z"]),
        )
        for name in LANDMARK_NAMES
    ]


def build_feature_dataset(input_path: Path, output_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(input_path)
    rows: list[dict[str, float | str]] = []
    for _, row in raw.iterrows():
        label = LABEL_MAP.get(str(row["upperbody_label"]))
        if label is None:
            continue
        features = extract_posture_features(row_to_landmarks(row), visibility_threshold=0.0)
        if features is None:
            continue
        rows.append({"label": label, **features})

    dataset = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)
    return dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrena ErgoIA con MultiPosture")
    parser.add_argument("--dataset", default=str(MULTIPOSTURE_PATH))
    parser.add_argument("--features-output", default="datasets/multiposture/ergoia_features.csv")
    parser.add_argument("--output", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--trees", type=int, default=400)
    parser.add_argument("--test-size", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.dataset)
    if not input_path.exists():
        input_path = download_multiposture()

    feature_path = Path(args.features_output)
    dataset = build_feature_dataset(input_path, feature_path)
    counts = dataset["label"].value_counts()
    print(f"Features guardadas en: {feature_path}")
    print("\nDistribucion de clases:")
    print(counts)

    x = dataset[FEATURE_COLUMNS]
    y = dataset["label"]
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=args.test_size,
        random_state=42,
        stratify=y,
    )

    model = Pipeline(
        steps=[
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=args.trees,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            )
        ]
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)

    labels = sorted(y.unique())
    print(f"\nModelo guardado en: {output_path}")
    print("\nMatriz de confusion:")
    print(confusion_matrix(y_test, predictions, labels=labels))
    print("\nReporte:")
    print(classification_report(y_test, predictions, labels=labels, zero_division=0))
    print("\nLa aplicacion ya puede usarlo con:")
    print("python run.py --classifier auto --source 1")


if __name__ == "__main__":
    main()
