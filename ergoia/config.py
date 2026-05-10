"""Configuracion central de ErgoIA.

Los valores se pueden cambiar desde aqui o temporalmente desde la linea de
comandos al ejecutar infer.py.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class AppConfig:
    bad_posture_seconds: int = 30
    sitting_break_minutes: int = 45
    hydration_minutes: int = 120
    posture_recovery_seconds: int = 5
    camera_width: int = 1280
    camera_height: int = 720
    min_detection_confidence: float = 0.55
    min_tracking_confidence: float = 0.55
    history_path: str = "data/historial_alertas.csv"
    hydration_log_path: str = "data/hidratacion.csv"


CONFIG = AppConfig()


POSTURE_LABELS = {
    "correct": "Postura correcta",
    "forward_head": "Cabeza adelantada",
    "hunched_back": "Espalda encorvada",
    "side_lean": "Inclinado hacia un lado",
    "shoulder_misalignment": "Hombros desalineados",
    "no_person": "No se detecta persona",
}


ALERT_MESSAGES = {
    "bad_posture": "Corrige tu postura: espalda recta, hombros relajados y cabeza alineada.",
    "active_break": "Pausa activa: levantate, camina o estirate unos minutos.",
    "hydration": "Recordatorio de hidratacion: registra agua cuando tomes.",
    "no_person": "No se detecta usuario frente a la camara.",
}
