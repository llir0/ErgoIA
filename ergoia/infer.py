"""Inferencia en tiempo real de ErgoIA con webcam, OpenCV y MediaPipe Pose."""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

from config import ALERT_MESSAGES, CONFIG
from ml_posture_model import DEFAULT_MODEL_PATH, load_classifier_if_available
from posture_rules import PostureAnalyzer
from storage import CsvStorage
from utils import format_duration


POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
POSE_MODEL_PATH = Path("models/pose_landmarker_lite.task")
OBJECT_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/object_detector/"
    "efficientdet_lite0/int8/1/efficientdet_lite0.tflite"
)
OBJECT_MODEL_PATH = Path("models/efficientdet_lite0.tflite")
DRINK_OBJECT_LABELS = {"bottle", "cup"}
BUTTONS: dict[str, tuple[int, int, int, int]] = {}
PENDING_ACTION: str | None = None
WINDOW_NAME = "ErgoIA - Analisis de postura"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ErgoIA - analisis ergonomico con webcam")
    parser.add_argument("--source", default="0", help="Indice de webcam o ruta de video. Ej: 0")
    parser.add_argument("--bad-posture-seconds", type=int, default=CONFIG.bad_posture_seconds)
    parser.add_argument("--break-minutes", type=int, default=CONFIG.sitting_break_minutes)
    parser.add_argument("--hydration-minutes", type=int, default=CONFIG.hydration_minutes)
    parser.add_argument("--history-path", default=CONFIG.history_path)
    parser.add_argument("--hydration-log-path", default=CONFIG.hydration_log_path)
    parser.add_argument(
        "--classifier",
        choices=["auto", "rules", "ml"],
        default="auto",
        help="auto usa modelo entrenado si existe; rules usa reglas; ml exige modelo.",
    )
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    return parser.parse_args()


def open_source(source: str) -> cv2.VideoCapture:
    if source.isdigit():
        return cv2.VideoCapture(int(source), cv2.CAP_DSHOW)
    return cv2.VideoCapture(source)


def configure_capture(cap: cv2.VideoCapture) -> None:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG.camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG.camera_height)


def scan_available_cameras(current_source: str | None = None, max_index: int = 10) -> list[str]:
    cameras: list[str] = []
    if current_source and current_source.isdigit():
        cameras.append(current_source)

    for index in range(max_index):
        source = str(index)
        if source in cameras:
            continue
        cap = open_source(source)
        configure_capture(cap)
        ok = cap.isOpened()
        if ok:
            ok, _ = cap.read()
        cap.release()
        if ok:
            cameras.append(source)

    return sorted(cameras, key=int)


def switch_camera(current_cap: cv2.VideoCapture, next_source: int | str) -> tuple[cv2.VideoCapture, str, str]:
    next_source_text = str(next_source)
    next_cap = open_source(next_source_text)
    configure_capture(next_cap)

    if not next_cap.isOpened():
        next_cap.release()
        return current_cap, "", f"No se pudo abrir camara {next_source_text}"

    ok, _ = next_cap.read()
    if not ok:
        next_cap.release()
        return current_cap, "", f"La camara {next_source_text} no entrego imagen"

    current_cap.release()
    return next_cap, next_source_text, f"Camara activa: {next_source_text}"


def ensure_pose_model() -> Path:
    POSE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not POSE_MODEL_PATH.exists() or POSE_MODEL_PATH.stat().st_size == 0:
        print("Descargando modelo de MediaPipe Pose. Esto ocurre solo la primera vez...")
        urlretrieve(POSE_MODEL_URL, POSE_MODEL_PATH)
    return POSE_MODEL_PATH


def ensure_object_model() -> Path:
    OBJECT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not OBJECT_MODEL_PATH.exists() or OBJECT_MODEL_PATH.stat().st_size == 0:
        print("Descargando detector de objetos para vaso/botella. Esto ocurre solo la primera vez...")
        urlretrieve(OBJECT_MODEL_URL, OBJECT_MODEL_PATH)
    return OBJECT_MODEL_PATH


def draw_landmarks(frame, landmarks, connections) -> None:
    if not landmarks:
        return
    height, width = frame.shape[:2]
    for connection in connections:
        start = landmarks[connection.start]
        end = landmarks[connection.end]
        if getattr(start, "visibility", 1.0) < 0.4 or getattr(end, "visibility", 1.0) < 0.4:
            continue
        p1 = (int(start.x * width), int(start.y * height))
        p2 = (int(end.x * width), int(end.y * height))
        cv2.line(frame, p1, p2, (90, 210, 255), 2)

    for landmark in landmarks:
        if getattr(landmark, "visibility", 1.0) < 0.4:
            continue
        center = (int(landmark.x * width), int(landmark.y * height))
        cv2.circle(frame, center, 3, (80, 240, 120), -1)


def draw_object_detections(frame, detections) -> None:
    if not detections:
        return
    for detection in detections:
        if not detection.categories:
            continue
        category = detection.categories[0]
        label = category.category_name or ""
        if label not in DRINK_OBJECT_LABELS:
            continue
        bbox = detection.bounding_box
        x1, y1 = int(bbox.origin_x), int(bbox.origin_y)
        x2, y2 = int(bbox.origin_x + bbox.width), int(bbox.origin_y + bbox.height)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (91, 180, 255), 2)
        cv2.putText(
            frame,
            f"{label} {category.score:.2f}",
            (x1, max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (91, 180, 255),
            1,
            cv2.LINE_AA,
        )


def add_event(events: list[str], message: str) -> None:
    events.append(f"{datetime.now():%H:%M:%S}  {message}")
    del events[:-8]


def format_clock(value: datetime | None) -> str:
    return value.strftime("%H:%M:%S") if value else "--:--:--"


def posture_recommendation(posture_key: str) -> str:
    recommendations = {
        "forward_head": "Lleva la barbilla ligeramente hacia atras y alinea la cabeza con los hombros.",
        "hunched_back": "Endereza la espalda, abre el pecho y acerca la silla al escritorio.",
        "side_lean": "Centra el torso sobre la silla y reparte el peso en ambos lados.",
        "shoulder_misalignment": "Relaja los hombros y revisa que ambos esten a una altura similar.",
        "no_person": "Ajusta la camara para que se vean cabeza y hombros.",
    }
    return recommendations.get(posture_key, "Postura estable. Mantén cabeza, hombros y espalda alineados.")


def posture_report_lines(posture_key: str, posture_label: str, metrics: dict[str, float]) -> list[str]:
    if posture_key == "no_person":
        return ["No se detecta usuario", "Ajusta camara e iluminacion"]
    confidence = metrics.get("model_confidence")
    confidence_text = f" | Conf {confidence:.0%}" if confidence is not None and confidence > 0 else ""
    lines = [f"{posture_label}{confidence_text}"]
    shoulder = metrics.get("shoulder_tilt_degrees")
    head = max(
        metrics.get("head_forward_ratio", 0.0),
        metrics.get("ear_forward_ratio", 0.0),
        metrics.get("head_drop_ratio", 0.0),
    )
    hunch = metrics.get("hunch_ratio")
    side = metrics.get("side_lean_ratio")
    neck_angle = metrics.get("neck_torso_angle")
    detail = []
    if shoulder is not None:
        detail.append(f"Hombros {shoulder:.1f} grados")
    detail.append(f"Cabeza {head:.2f}")
    if hunch is not None:
        detail.append(f"Encorvamiento {hunch:.2f}")
    if neck_angle is not None and neck_angle < 179:
        detail.append(f"Cuello-torso {neck_angle:.0f} grados")
    if side is not None and side > 0:
        detail.append(f"Lateral {side:.2f}")
    lines.append(" | ".join(detail[:4]))
    if posture_key != "correct":
        lines.append(posture_recommendation(posture_key))
    return lines


def draw_text_block(frame, text: str, origin: tuple[int, int], scale: float, color: tuple[int, int, int], width: int) -> int:
    x, y = origin
    words = text.split()
    line = ""
    line_height = int(24 * scale) + 10
    for word in words:
        candidate = f"{line} {word}".strip()
        (tw, _), _ = cv2.getTextSize(candidate, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        if tw > width and line:
            cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
            y += line_height
            line = word
        else:
            line = candidate
    if line:
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
        y += line_height
    return y


def draw_card(frame, x: int, y: int, w: int, h: int, title: str, value: str, accent: tuple[int, int, int]) -> None:
    cv2.rectangle(frame, (x, y), (x + w, y + h), (29, 34, 44), -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (46, 54, 69), 1)
    cv2.rectangle(frame, (x, y), (x + 5, y + h), accent, -1)
    cv2.putText(frame, title, (x + 16, y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 170, 185), 1, cv2.LINE_AA)
    cv2.putText(frame, value, (x + 16, y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (245, 247, 250), 1, cv2.LINE_AA)


def draw_button(
    frame,
    key: str,
    label: str,
    x: int,
    y: int,
    w: int,
    h: int,
    accent: tuple[int, int, int] = (70, 155, 255),
) -> None:
    BUTTONS[key] = (x, y, w, h)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (31, 37, 48), -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (62, 72, 90), 1)
    cv2.rectangle(frame, (x, y), (x + 4, y + h), accent, -1)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    tx = x + max(8, (w - tw) // 2)
    ty = y + (h + th) // 2 - 2
    cv2.putText(frame, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (226, 232, 240), 1, cv2.LINE_AA)


def on_mouse(event, x, y, flags, param) -> None:
    global PENDING_ACTION
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    for action, (bx, by, bw, bh) in BUTTONS.items():
        if bx <= x <= bx + bw and by <= y <= by + bh:
            PENDING_ACTION = action
            return


def compose_dashboard(
    frame,
    posture_label: str,
    session_seconds: float,
    hydration_seconds: float,
    alerts: list[str],
    camera_source: str,
    status_message: str,
    events: list[str],
    last_bad_posture: datetime | None,
    last_hydration: datetime | None,
    last_break: datetime | None,
    drinking_candidate: bool,
    drink_object_detected: bool,
    posture_key: str,
    report_lines: list[str],
    show_landmarks: bool,
    available_cameras: list[str],
    camera_menu_open: bool,
):
    output_w, output_h = 1280, 720
    side_w = 350
    video_w = output_w - side_w
    canvas = np.full((output_h, output_w, 3), (14, 17, 23), dtype=np.uint8)

    scale = min(video_w / frame.shape[1], output_h / frame.shape[0])
    resized_w = int(frame.shape[1] * scale)
    resized_h = int(frame.shape[0] * scale)
    resized = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    x0 = (video_w - resized_w) // 2
    y0 = (output_h - resized_h) // 2
    canvas[y0 : y0 + resized_h, x0 : x0 + resized_w] = resized

    posture_ok = posture_key == "correct"
    pill_color = (64, 180, 105) if posture_ok else (45, 126, 245)
    report_h = 92 if len(report_lines) <= 2 else 122
    cv2.rectangle(canvas, (18, 18), (650, 18 + report_h), (18, 24, 33), -1)
    cv2.rectangle(canvas, (18, 18), (650, 18 + report_h), pill_color, 2)
    y_report = 50
    for index, line in enumerate(report_lines[:3]):
        scale_text = 0.76 if index == 0 else 0.45
        color = (245, 247, 250) if index == 0 else (205, 214, 226)
        y_report = draw_text_block(canvas, line, (34, y_report), scale_text, color, 590)
    if status_message:
        cv2.rectangle(canvas, (18, 675), (video_w - 18, 708), (18, 24, 33), -1)
        cv2.putText(canvas, status_message, (32, 698), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (120, 205, 255), 1, cv2.LINE_AA)

    sx = video_w
    cv2.rectangle(canvas, (sx, 0), (output_w, output_h), (18, 22, 30), -1)
    cv2.putText(canvas, "ErgoIA", (sx + 22, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (245, 247, 250), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"Camara {camera_source}", (sx + 245, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (160, 170, 185), 1, cv2.LINE_AA)

    draw_card(canvas, sx + 18, 72, 150, 74, "Sentado", format_duration(session_seconds), (70, 155, 255))
    draw_card(canvas, sx + 182, 72, 150, 74, "Sin agua", format_duration(hydration_seconds), (255, 176, 70))
    draw_card(canvas, sx + 18, 162, 150, 74, "Mala postura", format_clock(last_bad_posture), (245, 94, 94))
    draw_card(canvas, sx + 182, 162, 150, 74, "Ult. pausa", format_clock(last_break), (91, 209, 150))
    draw_card(canvas, sx + 18, 252, 314, 74, "Ultima hidratacion", format_clock(last_hydration), (91, 180, 255))

    cv2.putText(canvas, "Estado", (sx + 22, 365), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (245, 247, 250), 1, cv2.LINE_AA)
    if drinking_candidate and drink_object_detected:
        status = "Gesto + vaso/botella detectados"
    elif drinking_candidate:
        status = "Gesto de tomar agua detectado"
    elif drink_object_detected:
        status = "Vaso/botella en escena"
    else:
        status = "Monitoreando postura"
    status_color = (91, 180, 255) if drinking_candidate else (160, 170, 185)
    draw_text_block(canvas, status, (sx + 22, 394), 0.45, status_color, 304)
    y = 432
    bad_posture_keys = {"forward_head", "hunched_back", "side_lean", "shoulder_misalignment"}
    if posture_key in bad_posture_keys:
        y = draw_text_block(canvas, posture_recommendation(posture_key), (sx + 22, y), 0.43, (255, 176, 70), 304)
    for alert in [a for a in alerts if a != ALERT_MESSAGES["bad_posture"]][:2]:
        y = draw_text_block(canvas, alert, (sx + 22, y), 0.43, (85, 170, 255), 304)

    cv2.putText(canvas, "Registro", (sx + 22, 498), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (245, 247, 250), 1, cv2.LINE_AA)
    y = 528
    for event in events[-6:]:
        y = draw_text_block(canvas, event, (sx + 22, y), 0.39, (185, 194, 208), 304)

    BUTTONS.clear()
    cv2.rectangle(canvas, (sx + 18, 632), (output_w - 18, 710), (29, 34, 44), -1)
    points_state = "puntos on" if show_landmarks else "puntos off"
    draw_button(canvas, "quit", "Salir", sx + 30, 644, 68, 26, (245, 94, 94))
    draw_button(canvas, "hydrate", "Agua", sx + 106, 644, 68, 26, (91, 180, 255))
    draw_button(canvas, "break", "Pausa", sx + 182, 644, 72, 26, (91, 209, 150))
    draw_button(canvas, "toggle_points", points_state, sx + 30, 678, 104, 26, (255, 176, 70))
    draw_button(canvas, "toggle_camera_menu", "Camara", sx + 142, 678, 148, 26, (160, 130, 255))

    if camera_menu_open:
        menu_x = sx + 142
        menu_w = 148
        item_h = 30
        visible_cameras = available_cameras or [camera_source]
        menu_h = 12 + item_h * len(visible_cameras)
        menu_y = 672 - menu_h
        cv2.rectangle(canvas, (menu_x, menu_y), (menu_x + menu_w, menu_y + menu_h), (18, 22, 30), -1)
        cv2.rectangle(canvas, (menu_x, menu_y), (menu_x + menu_w, menu_y + menu_h), (74, 86, 106), 1)
        for index, source in enumerate(visible_cameras):
            y_item = menu_y + 6 + index * item_h
            label = f"Camara {source}"
            accent = (91, 209, 150) if source == camera_source else (160, 130, 255)
            draw_button(canvas, f"camera_{source}", label, menu_x + 6, y_item, menu_w - 12, 24, accent)
    return canvas


def detect_drinking_gesture(landmarks) -> bool:
    if not landmarks or len(landmarks) <= 20:
        return False
    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]
    shoulder_width = abs(left_shoulder.x - right_shoulder.x)
    shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
    mouth_x = (landmarks[9].x + landmarks[10].x) / 2
    mouth_y = (landmarks[9].y + landmarks[10].y) / 2

    # Cuando la persona esta de lado el ancho visible de hombros se reduce.
    # Usar solo shoulder_width exagera distancias, asi que se limita con una
    # referencia minima en coordenadas normalizadas.
    reference = max(shoulder_width, abs(shoulder_y - mouth_y), 0.16)
    mouth_targets = [(mouth_x, mouth_y), (landmarks[0].x, landmarks[0].y)]
    hand_indexes = (15, 16, 19, 20, 21, 22)  # munecas, indices y pulgares

    for hand_index in hand_indexes:
        hand_point = landmarks[hand_index]
        if getattr(hand_point, "visibility", 1.0) < 0.20:
            continue
        if hand_point.y > shoulder_y + 0.25:
            continue
        for target_x, target_y in mouth_targets:
            dx = (hand_point.x - target_x) / reference
            dy = (hand_point.y - target_y) / reference
            if (dx * dx + dy * dy) ** 0.5 < 0.72:
                return True
    return False


def has_drink_object(detections) -> bool:
    if not detections:
        return False
    for detection in detections:
        for category in detection.categories:
            if (category.category_name or "") in DRINK_OBJECT_LABELS and category.score >= 0.35:
                return True
    return False


def maybe_log_alert(storage: CsvStorage, state: dict, alert_type: str, message: str, duration: float = 0) -> None:
    now = datetime.now()
    last_key = f"last_{alert_type}"
    last_time = state.get(last_key)
    if last_time is None or (now - last_time).total_seconds() > 60:
        storage.log_alert(alert_type, message, duration)
        state[last_key] = now


def analyze_posture(analyzer, fallback_analyzer: PostureAnalyzer, landmarks, events: list[str]):
    try:
        return analyzer.analyze(landmarks)
    except RuntimeError as exc:
        add_event(events, "Modelo incompatible; usando reglas")
        print(exc)
        return fallback_analyzer.analyze(landmarks)


def get_user_action(key: int) -> str | None:
    global PENDING_ACTION
    if PENDING_ACTION:
        action = PENDING_ACTION
        PENDING_ACTION = None
        return action
    if key == ord("q"):
        return "quit"
    if key == ord("h"):
        return "hydrate"
    if key == ord("b"):
        return "break"
    if key == ord("t"):
        return "toggle_points"
    if key == ord("c"):
        return "toggle_camera_menu"
    if key == ord("n"):
        return "next_camera"
    if key == ord("p"):
        return "prev_camera"
    if ord("0") <= key <= ord("9"):
        return f"camera_{chr(key)}"
    if key == ord("r"):
        return "reset"
    return None


class HybridPostureAnalyzer:
    """Combina Random Forest para tronco con reglas para cabeza/hombros."""

    def __init__(self, ml_analyzer, rules_analyzer: PostureAnalyzer):
        self.ml_analyzer = ml_analyzer
        self.rules_analyzer = rules_analyzer

    def analyze(self, landmarks):
        rules_result = self.rules_analyzer.analyze(landmarks)
        if rules_result.key in {"no_person", "forward_head", "shoulder_misalignment"}:
            return rules_result
        ml_result = self.ml_analyzer.analyze(landmarks)
        # El dataset externo no tiene clases de cabeza/hombros, asi que se
        # mezclan las metricas de reglas para conservar el reporte completo.
        ml_result.metrics = {**rules_result.metrics, **ml_result.metrics}
        return ml_result


def main() -> None:
    args = parse_args()
    storage = CsvStorage(args.history_path, args.hydration_log_path)
    rules_analyzer = PostureAnalyzer()
    ml_analyzer = None
    if args.classifier in {"auto", "ml"}:
        ml_analyzer = load_classifier_if_available(args.model_path)
        if args.classifier == "ml" and ml_analyzer is None:
            raise FileNotFoundError(
                f"No existe el modelo {args.model_path}. Entrena primero con: python train_model.py"
            )
    analyzer = HybridPostureAnalyzer(ml_analyzer, rules_analyzer) if ml_analyzer else rules_analyzer
    classifier_name = "Random Forest + reglas" if ml_analyzer else "Reglas"

    current_source = str(args.source)
    cap = open_source(current_source)
    configure_capture(cap)

    if not cap.isOpened():
        raise RuntimeError(
            "No se pudo abrir la webcam. Prueba con --source 1, cierra apps que usen la camara "
            "o revisa permisos de privacidad de Windows."
        )

    pose_model_path = ensure_pose_model()
    object_model_path = ensure_object_model()
    pose_options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(pose_model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=CONFIG.min_detection_confidence,
        min_pose_presence_confidence=CONFIG.min_detection_confidence,
        min_tracking_confidence=CONFIG.min_tracking_confidence,
    )
    object_options = vision.ObjectDetectorOptions(
        base_options=BaseOptions(model_asset_path=str(object_model_path)),
        running_mode=vision.RunningMode.VIDEO,
        max_results=5,
        score_threshold=0.35,
    )

    session_start = datetime.now()
    last_hydration = storage.last_hydration_datetime() or datetime.now()
    last_break = session_start
    last_bad_posture = None
    bad_posture_start = None
    drink_gesture_start = None
    last_auto_hydration = None
    drink_logged_this_gesture = False
    last_posture_key = None
    displayed_posture_key = "no_person"
    displayed_posture_label = "No se detecta persona"
    displayed_posture_metrics: dict[str, float] = {}
    last_report_update = 0.0
    show_landmarks = True
    camera_menu_open = False
    available_cameras = scan_available_cameras(current_source)
    alert_state: dict = {}
    events: list[str] = []
    add_event(events, f"Sesion iniciada. Camara {current_source}")
    add_event(events, f"Clasificador: {classifier_name}")
    status_message = f"Camara activa: {current_source}"
    status_until = time.monotonic() + 3
    frame_counter = 0
    last_object_detections = []

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    with vision.PoseLandmarker.create_from_options(pose_options) as pose, vision.ObjectDetector.create_from_options(
        object_options
    ) as object_detector:
        while True:
            ok, frame = cap.read()
            if not ok:
                status_message = f"No se pudo leer camara {current_source}. Presiona n o 0-9."
                status_until = time.monotonic() + 5
                frame = np.zeros((CONFIG.camera_height, CONFIG.camera_width, 3), dtype=np.uint8)
                posture = analyze_posture(analyzer, rules_analyzer, None, events)
                now = datetime.now()
                session_seconds = (now - session_start).total_seconds()
                hydration_seconds = (now - last_hydration).total_seconds()
                dashboard = compose_dashboard(
                    frame,
                    displayed_posture_label,
                    session_seconds,
                    hydration_seconds,
                    [ALERT_MESSAGES["no_person"]],
                    current_source,
                    status_message,
                    events,
                    last_bad_posture,
                    last_hydration,
                    last_break,
                    False,
                    False,
                    displayed_posture_key,
                    posture_report_lines(displayed_posture_key, displayed_posture_label, displayed_posture_metrics),
                    show_landmarks,
                    available_cameras,
                    camera_menu_open,
                )
                cv2.imshow(WINDOW_NAME, dashboard)
                key = cv2.waitKey(250) & 0xFF
                action = get_user_action(key)
                if action == "quit":
                    break
                if action == "toggle_camera_menu":
                    camera_menu_open = not camera_menu_open
                    if camera_menu_open:
                        available_cameras = scan_available_cameras(current_source)
                        status_message = f"{len(available_cameras)} camara(s) detectada(s)"
                        status_until = time.monotonic() + 3
                if action in {"next_camera", "prev_camera"} or (action and action.startswith("camera_")):
                    if action == "next_camera":
                        next_index = int(current_source) + 1 if current_source.isdigit() else 0
                    elif action == "prev_camera":
                        next_index = max(0, int(current_source) - 1) if current_source.isdigit() else 0
                    else:
                        next_index = action.split("_", 1)[1]
                    cap, new_source, status_message = switch_camera(cap, next_index)
                    if new_source:
                        current_source = new_source
                        available_cameras = scan_available_cameras(current_source)
                        add_event(events, f"Camara cambiada a {current_source}")
                    camera_menu_open = False
                    status_until = time.monotonic() + 3
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(time.monotonic() * 1000)
            results = pose.detect_for_video(mp_image, timestamp_ms)
            frame_counter += 1
            if frame_counter % 5 == 0:
                object_results = object_detector.detect_for_video(mp_image, timestamp_ms)
                last_object_detections = object_results.detections
            drink_object_detected = has_drink_object(last_object_detections)

            pose_landmarks = results.pose_landmarks[0] if results.pose_landmarks else None
            if pose_landmarks and show_landmarks:
                draw_landmarks(frame, pose_landmarks, vision.PoseLandmarksConnections.POSE_LANDMARKS)
            if show_landmarks:
                draw_object_detections(frame, last_object_detections)

            posture = analyze_posture(analyzer, rules_analyzer, pose_landmarks, events)
            now = datetime.now()
            now_monotonic = time.monotonic()
            if now_monotonic - last_report_update >= 1.0:
                displayed_posture_key = posture.key
                displayed_posture_label = posture.label
                displayed_posture_metrics = posture.metrics
                last_report_update = now_monotonic
            session_seconds = (now - session_start).total_seconds()
            hydration_seconds = (now - last_hydration).total_seconds()
            alerts: list[str] = []
            drinking_candidate = detect_drinking_gesture(pose_landmarks)

            if posture.key != last_posture_key:
                if posture.key == "correct":
                    add_event(events, "Postura corregida")
                elif posture.key != "no_person":
                    add_event(events, f"Postura: {posture.label}")
                last_posture_key = posture.key

            if posture.key in {"forward_head", "hunched_back", "side_lean", "shoulder_misalignment"}:
                if bad_posture_start is None:
                    bad_posture_start = now
                last_bad_posture = now
                bad_seconds = (now - bad_posture_start).total_seconds()
                if bad_seconds >= args.bad_posture_seconds:
                    alerts.append(ALERT_MESSAGES["bad_posture"])
                    maybe_log_alert(storage, alert_state, posture.key, posture.label, bad_seconds)
            else:
                bad_posture_start = None

            if posture.key == "no_person":
                alerts.append(ALERT_MESSAGES["no_person"])
                maybe_log_alert(storage, alert_state, "no_person", ALERT_MESSAGES["no_person"], 0)

            if session_seconds >= args.break_minutes * 60:
                alerts.append(ALERT_MESSAGES["active_break"])
                maybe_log_alert(storage, alert_state, "active_break", ALERT_MESSAGES["active_break"], session_seconds)

            if hydration_seconds >= args.hydration_minutes * 60:
                alerts.append(ALERT_MESSAGES["hydration"])
                maybe_log_alert(storage, alert_state, "hydration", ALERT_MESSAGES["hydration"], hydration_seconds)

            if drinking_candidate:
                if drink_gesture_start is None:
                    drink_gesture_start = now
                gesture_seconds = (now - drink_gesture_start).total_seconds()
                can_log_auto = last_auto_hydration is None or (now - last_auto_hydration).total_seconds() > 8
                strong_drink_signal = drink_object_detected and gesture_seconds >= 0.25
                gesture_only_signal = gesture_seconds >= 0.55
                if (strong_drink_signal or gesture_only_signal) and can_log_auto and not drink_logged_this_gesture:
                    last_hydration = now
                    last_auto_hydration = now
                    hydration_seconds = 0
                    drink_logged_this_gesture = True
                    storage.log_hydration()
                    status_message = f"Hidratacion detectada: {now:%H:%M:%S}"
                    status_until = time.monotonic() + 4
                    add_event(events, "Hidratacion detectada por gesto")
            else:
                drink_gesture_start = None
                drink_logged_this_gesture = False

            visible_status = status_message if time.monotonic() < status_until else ""
            dashboard = compose_dashboard(
                frame,
                displayed_posture_label,
                session_seconds,
                hydration_seconds,
                alerts,
                current_source,
                visible_status,
                events,
                last_bad_posture,
                last_hydration,
                last_break,
                drinking_candidate,
                drink_object_detected,
                displayed_posture_key,
                posture_report_lines(displayed_posture_key, displayed_posture_label, displayed_posture_metrics),
                show_landmarks,
                available_cameras,
                camera_menu_open,
            )
            cv2.imshow(WINDOW_NAME, dashboard)

            key = cv2.waitKey(1) & 0xFF
            action = get_user_action(key)
            if action == "quit":
                break
            if action == "toggle_camera_menu":
                camera_menu_open = not camera_menu_open
                if camera_menu_open:
                    available_cameras = scan_available_cameras(current_source)
                    status_message = f"{len(available_cameras)} camara(s) detectada(s)"
                    status_until = time.monotonic() + 3
            if action and action.startswith("camera_"):
                cap, new_source, status_message = switch_camera(cap, action.split("_", 1)[1])
                if new_source:
                    current_source = new_source
                    available_cameras = scan_available_cameras(current_source)
                    bad_posture_start = None
                    add_event(events, f"Camara cambiada a {current_source}")
                camera_menu_open = False
                status_until = time.monotonic() + 3
            if action in {"next_camera", "prev_camera"}:
                if current_source.isdigit():
                    delta = 1 if action == "next_camera" else -1
                    next_index = max(0, int(current_source) + delta)
                else:
                    next_index = 0
                cap, new_source, status_message = switch_camera(cap, next_index)
                if new_source:
                    current_source = new_source
                    available_cameras = scan_available_cameras(current_source)
                    bad_posture_start = None
                    add_event(events, f"Camara cambiada a {current_source}")
                camera_menu_open = False
                status_until = time.monotonic() + 3
            if action == "hydrate":
                last_hydration = datetime.now()
                last_auto_hydration = last_hydration
                hydration_seconds = 0
                storage.log_hydration()
                alert_state["last_hydration_keypress"] = last_hydration
                status_message = f"Hidratacion registrada: {last_hydration:%H:%M:%S}"
                status_until = time.monotonic() + 3
                add_event(events, "Hidratacion manual")
                print(f"Hidratacion registrada: {last_hydration:%Y-%m-%d %H:%M:%S}")
            if action == "toggle_points":
                show_landmarks = not show_landmarks
                status_message = "Puntos ocultos" if not show_landmarks else "Puntos visibles"
                status_until = time.monotonic() + 3
                add_event(events, status_message)
            if action == "break":
                last_break = datetime.now()
                session_start = last_break
                bad_posture_start = None
                status_message = f"Pausa registrada: {last_break:%H:%M:%S}"
                status_until = time.monotonic() + 3
                add_event(events, "Pausa activa registrada")
            if action == "reset":
                session_start = datetime.now()
                last_break = session_start
                bad_posture_start = None
                alert_state.clear()
                status_message = "Sesion reiniciada"
                status_until = time.monotonic() + 3
                add_event(events, "Sesion reiniciada")
                print("Sesion reiniciada.")

            time.sleep(0.001)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
