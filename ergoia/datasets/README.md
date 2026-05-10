# Datasets de ErgoIA

## MultiPosture

Dataset publico usado para entrenar el modelo base de postura sentada:

- Fuente: Zenodo, DOI `10.5281/zenodo.14230872`
- Licencia: Creative Commons Attribution 4.0 International
- Archivo local: `datasets/multiposture/data.csv`
- Features convertidas: `datasets/multiposture/ergoia_features.csv`

Mapeo de etiquetas usado:

- `TUP` -> `correct`
- `TLF` -> `hunched_back`
- `TLR` / `TLL` -> `side_lean`
- `TLB` se excluye porque ErgoIA no tiene una alerta equivalente para tronco inclinado hacia atras.

## Datasets no mezclados automaticamente

- Roboflow Sitting Posture: puede requerir cuenta, export o API key.
- UCI HAR: usa sensores inerciales, no webcam ni landmarks de pose.
- COCO-Pose: tiene keypoints humanos, pero no etiquetas ergonomicas de postura sentada.
- SitPose: esta orientado a sensor de profundidad; no se integra directamente con webcam RGB sin adaptar el pipeline.
