from __future__ import annotations

import json
import math
import pickle
from pathlib import Path
from typing import Any, Iterable


ANNOTATION_EXTENSIONS = {".json", ".jsonl", ".csv", ".txt", ".pkl", ".pickle", ".npy", ".npz"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
LIDAR_EXTENSIONS = {".bin", ".pcd", ".ply", ".las", ".laz"}


def warn(message: str) -> None:
    print(f"[WARN] {message}")


def info(message: str) -> None:
    print(f"[INFO] {message}")


def ensure_dir(path: Path | str) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def euclidean(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def heading_to_vector(heading: float) -> tuple[float, float]:
    return math.cos(heading), math.sin(heading)


def relative_angle(origin: tuple[float, float], heading: float, point: tuple[float, float]) -> float:
    dx = point[0] - origin[0]
    dy = point[1] - origin[1]
    return normalize_angle(math.atan2(dy, dx) - heading)


def normalize_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


def load_structured_file(path: Path) -> Any | None:
    suffix = path.suffix.lower()
    try:
        if suffix == ".json":
            return json.loads(path.read_text(encoding="utf-8"))
        if suffix == ".jsonl":
            return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if suffix in {".pkl", ".pickle"}:
            with path.open("rb") as f:
                return pickle.load(f)
        if suffix == ".npy":
            import numpy as np

            return np.load(path, allow_pickle=True)
        if suffix == ".npz":
            import numpy as np

            data = np.load(path, allow_pickle=True)
            return {key: data[key] for key in data.files}
    except Exception as exc:
        warn(f"Could not load {path}: {exc}")
    return None


def summarize_keys(value: Any, depth: int = 0, max_depth: int = 2) -> list[str]:
    if depth > max_depth:
        return []
    prefix = "  " * depth
    lines: list[str] = []
    if isinstance(value, dict):
        for key, child in list(value.items())[:30]:
            lines.append(f"{prefix}- {key}: {type(child).__name__}")
            lines.extend(summarize_keys(child, depth + 1, max_depth))
    elif isinstance(value, list) and value:
        lines.append(f"{prefix}- list[{len(value)}] item: {type(value[0]).__name__}")
        lines.extend(summarize_keys(value[0], depth + 1, max_depth))
    return lines


def first_existing(mapping: dict[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def object_type_flags(object_type: str) -> tuple[int, int]:
    normalized = object_type.lower().replace(" ", "_")
    is_occluder = int(any(token in normalized for token in ["parked","vehicle","car","bus","truck","building","wall","motorcycle",]))
    is_vru = int(any(token in normalized for token in ["scooter","e_scooter","electric_scooter","personal_mobility","bicycle","bike","cyclist","pedestrian","person","walker","wheelchair","stroller",]))
    return is_occluder, is_vru


def polygon_area(points: list[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    for idx, (x1, y1) in enumerate(points):
        x2, y2 = points[(idx + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def polygon_centroid(points: list[tuple[float, float]]) -> tuple[float, float]:
    if not points:
        return (0.0, 0.0)
    signed_area = 0.0
    cx = 0.0
    cy = 0.0
    for idx, (x1, y1) in enumerate(points):
        x2, y2 = points[(idx + 1) % len(points)]
        cross = x1 * y2 - x2 * y1
        signed_area += cross
        cx += (x1 + x2) * cross
        cy += (y1 + y2) * cross
    signed_area *= 0.5
    if abs(signed_area) < 1e-9:
        return (sum(x for x, _ in points) / len(points), sum(y for _, y in points) / len(points))
    return (cx / (6.0 * signed_area), cy / (6.0 * signed_area))


def point_in_polygon(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
    if len(polygon) < 3:
        return False
    x, y = point
    inside = False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        intersects = (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / max(yj - yi, 1e-12) + xi
        if intersects:
            inside = not inside
        j = i
    return inside
