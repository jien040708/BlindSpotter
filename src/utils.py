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
    normalized = object_type.lower()
    is_occluder = int(any(token in normalized for token in ["parked", "vehicle", "car", "bus", "truck", "building", "wall"]))
    is_vru = int(any(token in normalized for token in ["scooter", "bicycle", "bike", "pedestrian", "walker"]))
    return is_occluder, is_vru
