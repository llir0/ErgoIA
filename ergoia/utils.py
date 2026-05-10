"""Utilidades matematicas y de formato para ErgoIA."""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Iterable, Tuple

Point = Tuple[float, float]


def euclidean_distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def midpoint(a: Point, b: Point) -> Point:
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def slope_degrees(a: Point, b: Point) -> float:
    """Devuelve la inclinacion de la linea a-b respecto a la horizontal.

    El resultado queda normalizado entre 0 y 90 grados. Sin esta normalizacion,
    una linea casi horizontal dibujada de derecha a izquierda puede aparecer
    como 178 grados, aunque visualmente este alineada.
    """
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    if abs(dx) < 1e-6:
        return 90.0
    raw = abs(math.degrees(math.atan2(dy, dx)))
    return min(raw, abs(180.0 - raw))


def angle_degrees(a: Point, b: Point, c: Point) -> float:
    """Angulo ABC en grados."""
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    dot = ba[0] * bc[0] + ba[1] * bc[1]
    mag_ba = math.hypot(*ba)
    mag_bc = math.hypot(*bc)
    if mag_ba == 0 or mag_bc == 0:
        return 0.0
    value = max(-1.0, min(1.0, dot / (mag_ba * mag_bc)))
    return math.degrees(math.acos(value))


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return str(timedelta(seconds=seconds))


def minutes_since(timestamp: datetime) -> float:
    return (datetime.now() - timestamp).total_seconds() / 60.0


def most_frequent(items: Iterable[str], default: str = "") -> str:
    counts: dict[str, int] = {}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
    if not counts:
        return default
    return max(counts, key=counts.get)
