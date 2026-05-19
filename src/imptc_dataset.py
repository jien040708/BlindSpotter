from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .dataset import Frame, ObjectState, Scene
from .utils import warn


VRU_TYPES = {
    "pedestrian",
    "person",
    "cyclist",
    "bicycle",
    "bike",
    "scooter",
    "e_scooter",
    "wheelchair",
    "stroller",
    "personal_mobility",
    "vru",
}

VEHICLE_TYPES = {
    "car",
    "truck",
    "bus",
    "vehicle",
    "motorcycle",
}


def is_imptc_root(root: str | Path) -> bool:
    """
    Expected structure:
      root/
        train/{track_id}/track.json
        eval/{track_id}/track.json
        test/{track_id}/track.json
    """
    root = Path(root)
    if not root.exists():
        return False
    return any((root / split).is_dir() for split in ["train", "eval", "test"])


def load_imptc_scenes(
    root: str | Path,
    splits: list[str] | None = None,
    max_tracks_per_split: int | None = None,
    max_frames_per_split: int | None = None,
    frame_stride: int = 5,
) -> list[Scene]:
    """
    IMPTC split 하나를 Scene 하나로 변환한다.

    반환:
      train scene
      eval scene
      test scene
    """
    root = Path(root)
    splits = splits or ["train", "eval", "test"]

    scenes: list[Scene] = []

    for split in splits:
        split_dir = root / split
        if not split_dir.exists():
            warn(f"IMPTC split not found: {split_dir}")
            continue

        scene = load_imptc_split(
            split_dir=split_dir,
            split_name=split,
            max_tracks=max_tracks_per_split,
            max_frames=max_frames_per_split,
            frame_stride=frame_stride,
        )

        if scene.frames:
            scenes.append(scene)

    return scenes


def load_imptc_split(
    split_dir: Path,
    split_name: str,
    max_tracks: int | None = None,
    max_frames: int | None = None,
    frame_stride: int = 5,
) -> Scene:
    tracks = load_track_group_flat(split_dir, max_tracks=max_tracks)

    if not tracks:
        warn(f"No IMPTC tracks found in {split_dir}")
        return Scene(scene_id=split_name, source_path=str(split_dir))

    reference_track = choose_reference_vehicle(tracks)

    timestamps = sorted(
        {ts for track in tracks for ts in track["states"].keys()},
        key=int,
    )

    if frame_stride > 1:
        timestamps = timestamps[::frame_stride]

    if max_frames:
        timestamps = timestamps[:max_frames]

    if not timestamps:
        return Scene(scene_id=split_name, source_path=str(split_dir))

    first_ts = int(timestamps[0])
    frames: list[Frame] = []

    for frame_idx, ts in enumerate(timestamps):
        ego = state_to_object(reference_track, ts)

        if ego is None:
            ego = nearest_state_to_object(reference_track, ts)

        if ego is None:
            continue

        ego.object_id = "ego"
        ego.object_type = "ego_vehicle"

        objects: list[ObjectState] = []

        for track in tracks:
            if track is reference_track:
                continue

            obj = state_to_object(track, ts)
            if obj is not None:
                objects.append(obj)

        timestamp_seconds = (int(ts) - first_ts) / 1_000_000.0

        frames.append(
            Frame(
                frame_id=str(frame_idx),
                timestamp=timestamp_seconds,
                ego=ego,
                objects=objects,
                map_info={
                    "split": split_name,
                    "split_dir": str(split_dir),
                },
                raw={
                    "timestamp_us": ts,
                    "split": split_name,
                },
            )
        )

    return Scene(
        scene_id=split_name,
        frames=frames,
        source_path=str(split_dir),
        raw={
            "split": split_name,
            "reference_track": reference_track["id"],
        },
    )


def load_track_group_flat(split_dir: Path, max_tracks: int | None = None) -> list[dict[str, Any]]:
    """
    split_dir/{track_id}/track.json 구조를 읽는다.
    """
    track_files = sorted(split_dir.glob("*/track.json"))

    if max_tracks:
        track_files = track_files[:max_tracks]

    tracks: list[dict[str, Any]] = []

    for track_file in track_files:
        try:
            data = json.loads(track_file.read_text(encoding="utf-8"))
        except Exception as exc:
            warn(f"Could not load IMPTC track {track_file}: {exc}")
            continue

        overview = data.get("overview", {})
        raw_states = data.get("track_data", {})

        if not isinstance(raw_states, dict):
            continue

        track_id = track_file.parent.name
        class_name = str(overview.get("class_name", "unknown"))
        object_type = normalize_imptc_type(class_name)

        states = normalize_track_states(raw_states)

        if not states:
            continue

        tracks.append(
            {
                "id": track_id,
                "object_type": object_type,
                "raw_class_name": class_name,
                "states": states,
                "path": str(track_file),
                "duration": float(overview.get("duration", 0.0) or 0.0),
            }
        )

    return tracks


def normalize_track_states(raw_states: dict[str, Any]) -> dict[str, dict[str, float]]:
    ordered = []

    for _, state in sorted(raw_states.items(), key=lambda item: int(item[0])):
        if not isinstance(state, dict):
            continue
        if "ts" not in state or "coordinates" not in state:
            continue

        coords = state.get("coordinates") or [0.0, 0.0, 0.0]

        ordered.append(
            {
                "ts": str(state["ts"]),
                "x": float(coords[0]),
                "y": float(coords[1]),
                "z": float(coords[2]) if len(coords) > 2 else 0.0,
                "speed": float(state.get("velocity", 0.0) or 0.0),
                "visible": int(state.get("status", 1) or 1) > 0,
                "ground_type": float(state.get("ground_type", -1) or -1),
            }
        )

    states: dict[str, dict[str, float]] = {}

    for idx, state in enumerate(ordered):
        prev_state = ordered[max(0, idx - 1)]
        next_state = ordered[min(len(ordered) - 1, idx + 1)]

        dt = max((int(next_state["ts"]) - int(prev_state["ts"])) / 1_000_000.0, 1e-6)

        vx = (next_state["x"] - prev_state["x"]) / dt
        vy = (next_state["y"] - prev_state["y"]) / dt

        heading = math.atan2(vy, vx) if abs(vx) + abs(vy) > 1e-6 else 0.0

        prev_vx = 0.0
        prev_vy = 0.0

        if idx > 0:
            prev_prev_state = ordered[max(0, idx - 2)]
            prev_dt = max((int(state["ts"]) - int(prev_prev_state["ts"])) / 1_000_000.0, 1e-6)
            prev_vx = (state["x"] - prev_prev_state["x"]) / prev_dt
            prev_vy = (state["y"] - prev_prev_state["y"]) / prev_dt

        ax = (vx - prev_vx) / dt
        ay = (vy - prev_vy) / dt

        state.update(
            {
                "vx": vx,
                "vy": vy,
                "ax": ax,
                "ay": ay,
                "heading": heading,
            }
        )

        states[state["ts"]] = state

    return states


def normalize_imptc_type(class_name: str) -> str:
    value = class_name.lower().strip().replace(" ", "_")

    if value in {"person", "pedestrian"}:
        return "pedestrian"

    if value in {"bicycle", "bike", "cyclist"}:
        return "cyclist"

    if value in {"scooter", "electric_scooter", "e-scooter", "personal_mobility"}:
        return "e_scooter"

    if value in {"car", "truck", "bus", "motorcycle", "vehicle"}:
        return value

    if value in {"wheelchair", "stroller"}:
        return value

    return value or "unknown"


def choose_reference_vehicle(tracks: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Ego vehicle 역할을 할 reference track 선택.
    우선 vehicle 계열 중 가장 긴 track을 선택한다.
    """
    vehicles = [
        track for track in tracks
        if track["object_type"] in {"car", "truck", "bus", "vehicle", "motorcycle"}
    ]

    if not vehicles:
        warn("No vehicle track available; using the longest track as reference.")
        return max(tracks, key=lambda track: len(track["states"]))

    return max(vehicles, key=lambda track: len(track["states"]))


def state_to_object(track: dict[str, Any], timestamp: str) -> ObjectState | None:
    state = track["states"].get(timestamp)

    if state is None:
        return None

    obj = ObjectState(
        object_id=str(track["id"]),
        object_type=str(track["object_type"]),
        x=float(state["x"]),
        y=float(state["y"]),
        vx=float(state["vx"]),
        vy=float(state["vy"]),
        heading=float(state["heading"]),
        visible=bool(state["visible"]),
        raw={
            "source_path": track["path"],
            "timestamp_us": timestamp,
            "ground_type": state.get("ground_type"),
            "raw_class_name": track.get("raw_class_name"),
            "speed": state.get("speed", 0.0),
            "ax": state.get("ax", 0.0),
            "ay": state.get("ay", 0.0),
        },
    )

    return obj


def nearest_state_to_object(track: dict[str, Any], timestamp: str) -> ObjectState | None:
    if not track["states"]:
        return None

    nearest_ts = min(track["states"], key=lambda ts: abs(int(ts) - int(timestamp)))
    return state_to_object(track, nearest_ts)