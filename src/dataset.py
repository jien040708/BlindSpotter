from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .utils import ANNOTATION_EXTENSIONS, first_existing, load_structured_file, safe_float, warn


@dataclass
class ObjectState:
    object_id: str
    object_type: str = "unknown"
    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    heading: float = 0.0
    visible: bool = True
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class Frame:
    frame_id: str
    timestamp: float = 0.0
    ego: ObjectState | None = None
    objects: list[ObjectState] = field(default_factory=list)
    map_info: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class Scene:
    scene_id: str
    frames: list[Frame] = field(default_factory=list)
    source_path: str = ""
    raw: Any = None


class FlexibleSceneDataset:
    """Loads a small sample dataset into a conservative common scene format."""

    def __init__(self, root: str | Path, max_files: int | None = None):
        self.root = Path(root)
        self.max_files = max_files
        self.annotation_files = self.discover_annotation_files()

    def discover_annotation_files(self) -> list[Path]:
        if not self.root.exists():
            warn(f"Dataset root does not exist: {self.root}")
            return []
        files = [p for p in self.root.rglob("*") if p.is_file() and p.suffix.lower() in ANNOTATION_EXTENSIONS]
        files = sorted(files)
        if self.max_files:
            files = files[: self.max_files]
        return files

    def load_scenes(self) -> list[Scene]:
        scenes: list[Scene] = []
        for path in self.annotation_files:
            scene = self.load_scene(path)
            if scene and scene.frames:
                scenes.append(scene)
        return scenes

    def load_scene(self, path: Path) -> Scene | None:
        if path.suffix.lower() == ".csv":
            return self._load_csv_scene(path)
        data = load_structured_file(path)
        if data is None:
            return None
        return normalize_scene(data, scene_id=path.stem, source_path=str(path))

    def _load_csv_scene(self, path: Path) -> Scene | None:
        try:
            with path.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        except Exception as exc:
            warn(f"Could not load CSV {path}: {exc}")
            return None
        return normalize_scene(rows, scene_id=path.stem, source_path=str(path))


def normalize_scene(data: Any, scene_id: str, source_path: str = "") -> Scene:
    if isinstance(data, dict):
        scene_id = str(first_existing(data, ["scene_id", "scene", "sequence_id", "log_id", "id"], scene_id))
        frame_items = first_existing(data, ["frames", "samples", "timestamps", "annotations", "data"], None)
        if frame_items is None and any(key in data for key in ["objects", "ego_pose", "ego", "frame_id"]):
            frame_items = [data]
        map_info = first_existing(data, ["map_info", "map", "road_geometry"], {}) or {}
    elif isinstance(data, list):
        frame_items = data
        map_info = {}
    else:
        warn(f"Unsupported scene data type in {source_path}: {type(data).__name__}")
        return Scene(scene_id=scene_id, source_path=source_path, raw=data)

    frames: list[Frame] = []
    if not isinstance(frame_items, list):
        warn(f"No frame list found in {source_path}; expected a list-like frames field.")
        return Scene(scene_id=scene_id, source_path=source_path, raw=data)

    for idx, item in enumerate(frame_items):
        if not isinstance(item, dict):
            continue
        frame = normalize_frame(item, default_frame_id=str(idx), inherited_map=map_info)
        frames.append(frame)

    frames.sort(key=lambda frame: (frame.timestamp, frame.frame_id))
    return Scene(scene_id=scene_id, frames=frames, source_path=source_path, raw=data)


def normalize_frame(item: dict[str, Any], default_frame_id: str, inherited_map: dict[str, Any]) -> Frame:
    frame_id = str(first_existing(item, ["frame_id", "sample_id", "timestamp_id", "id"], default_frame_id))
    timestamp = safe_float(first_existing(item, ["timestamp", "time", "t", "frame_time"], len(default_frame_id)))
    ego_raw = first_existing(item, ["ego_pose", "ego", "ego_vehicle", "ego_state"], None)
    ego = normalize_object(ego_raw, fallback_id="ego", fallback_type="ego_vehicle") if isinstance(ego_raw, dict) else None
    if ego is None:
        warn(f"Frame {frame_id}: missing ego pose/state; using origin fallback.")
        ego = ObjectState(object_id="ego", object_type="ego_vehicle")

    objects_raw = first_existing(item, ["objects", "instances", "agents", "tracks", "annotations"], []) or []
    objects: list[ObjectState] = []
    if isinstance(objects_raw, dict):
        objects_raw = list(objects_raw.values())
    if isinstance(objects_raw, list):
        for obj_idx, raw in enumerate(objects_raw):
            if isinstance(raw, dict):
                objects.append(normalize_object(raw, fallback_id=f"obj_{obj_idx}", fallback_type="unknown"))
    else:
        warn(f"Frame {frame_id}: object container is not list/dict, skipping objects.")

    map_info = first_existing(item, ["map_info", "map", "road_geometry"], inherited_map) or {}
    return Frame(frame_id=frame_id, timestamp=timestamp, ego=ego, objects=objects, map_info=map_info, raw=item)


def normalize_object(raw: dict[str, Any], fallback_id: str, fallback_type: str) -> ObjectState:
    object_id = str(first_existing(raw, ["object_id", "track_id", "instance_id", "id", "token"], fallback_id))
    object_type = str(first_existing(raw, ["object_type", "type", "category", "class", "label"], fallback_type))
    position = first_existing(raw, ["position", "translation", "center", "location"], {}) or {}
    velocity = first_existing(raw, ["velocity", "vel"], {}) or {}
    if isinstance(position, (list, tuple)):
        x = safe_float(position[0] if len(position) > 0 else 0.0)
        y = safe_float(position[1] if len(position) > 1 else 0.0)
    else:
        x = safe_float(first_existing(raw, ["x", "pos_x"], first_existing(position, ["x", "0"], 0.0)))
        y = safe_float(first_existing(raw, ["y", "pos_y"], first_existing(position, ["y", "1"], 0.0)))
    if isinstance(velocity, (list, tuple)):
        vx = safe_float(velocity[0] if len(velocity) > 0 else 0.0)
        vy = safe_float(velocity[1] if len(velocity) > 1 else 0.0)
    else:
        vx = safe_float(first_existing(raw, ["vx", "vel_x"], first_existing(velocity, ["x", "vx", "0"], 0.0)))
        vy = safe_float(first_existing(raw, ["vy", "vel_y"], first_existing(velocity, ["y", "vy", "1"], 0.0)))
    heading = safe_float(first_existing(raw, ["heading", "yaw", "theta", "rotation"], 0.0))
    visible = bool(first_existing(raw, ["visible", "visibility", "is_visible"], True))
    return ObjectState(object_id=object_id, object_type=object_type, x=x, y=y, vx=vx, vy=vy, heading=heading, visible=visible, raw=raw)
