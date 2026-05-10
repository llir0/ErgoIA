"""Descarga datasets publicos compatibles con ErgoIA."""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve


MULTIPOSTURE_URL = "https://zenodo.org/records/14230872/files/data.csv?download=1"
MULTIPOSTURE_PATH = Path("datasets/multiposture/data.csv")


def download_multiposture() -> Path:
    MULTIPOSTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if MULTIPOSTURE_PATH.exists() and MULTIPOSTURE_PATH.stat().st_size > 0:
        print(f"MultiPosture ya existe: {MULTIPOSTURE_PATH}")
        return MULTIPOSTURE_PATH
    print("Descargando MultiPosture desde Zenodo...")
    urlretrieve(MULTIPOSTURE_URL, MULTIPOSTURE_PATH)
    print(f"Guardado en: {MULTIPOSTURE_PATH}")
    return MULTIPOSTURE_PATH


def main() -> None:
    download_multiposture()
    print("\nNota:")
    print("- MultiPosture es el dataset publico principal usado para entrenar postura sentada.")
    print("- Roboflow Sitting Posture requiere export/API key.")
    print("- UCI HAR usa sensores inerciales, no webcam, por eso no se mezcla con este modelo.")
    print("- COCO-Pose no trae etiquetas ergonomicas de postura sentada.")


if __name__ == "__main__":
    main()
