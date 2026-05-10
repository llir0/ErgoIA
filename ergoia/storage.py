"""Persistencia simple en CSV para alertas e hidratacion."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd


ALERT_COLUMNS = ["fecha", "hora", "tipo_alerta", "mensaje", "duracion_segundos"]
HYDRATION_COLUMNS = ["fecha", "hora", "evento"]


class CsvStorage:
    def __init__(self, history_path: str, hydration_log_path: str):
        self.history_path = Path(history_path)
        self.hydration_log_path = Path(hydration_log_path)
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        self.hydration_log_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_csv(self.history_path, ALERT_COLUMNS)
        self._ensure_csv(self.hydration_log_path, HYDRATION_COLUMNS)

    @staticmethod
    def _ensure_csv(path: Path, columns: list[str]) -> None:
        if path.exists() and path.stat().st_size > 0:
            return
        with path.open("w", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow(columns)

    def log_alert(self, alert_type: str, message: str, duration_seconds: float = 0) -> None:
        now = datetime.now()
        with self.history_path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    now.strftime("%Y-%m-%d"),
                    now.strftime("%H:%M:%S"),
                    alert_type,
                    message,
                    int(duration_seconds),
                ]
            )

    def log_hydration(self) -> None:
        now = datetime.now()
        with self.hydration_log_path.open("a", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow(
                [now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"), "agua_registrada"]
            )

    def last_hydration_datetime(self) -> Optional[datetime]:
        try:
            df = pd.read_csv(self.hydration_log_path)
        except (FileNotFoundError, pd.errors.EmptyDataError):
            return None
        if df.empty:
            return None
        last = df.iloc[-1]
        return datetime.strptime(f"{last['fecha']} {last['hora']}", "%Y-%m-%d %H:%M:%S")

    def read_alerts(self) -> pd.DataFrame:
        try:
            return pd.read_csv(self.history_path)
        except (FileNotFoundError, pd.errors.EmptyDataError):
            return pd.DataFrame(columns=ALERT_COLUMNS)

    def read_hydration(self) -> pd.DataFrame:
        try:
            return pd.read_csv(self.hydration_log_path)
        except (FileNotFoundError, pd.errors.EmptyDataError):
            return pd.DataFrame(columns=HYDRATION_COLUMNS)
