from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .dataset import Frame, ObjectState, Scene
from .utils import warn


def is_imptc_root(root: str | Path) -> bool:
    root = Path(root)
    return any((child / "vehicles").is_dir() and (child / "vrus").is_dir() for child in root.iterdir() if child.is_dir())


def load_imptc_scenes(
    root: str | Path,
    max_sequences: int | None = None,
    max_frames_per_sequence: int | None = None,
    frame_stride: int = 5,
) -> list[Scene]:
    root = Path(root)
    sequence_dirs = sorted(
        child for child in root.iterdir() if child.is_dir() and (child / "vehicles").is_dir() and (child / "vrus").is_dir()
    )
    if max_sequences:
        sequence_dirs = sequence_dirs[:max_sequences]
    scenes: list[Scene] = []
    for sequence_dir in sequence_dirs:
        scene = load_imptc_sequence(sequence_dir, max_frames=max_frames_per_sequence, frame_stride=frame_stride)
        if scene.frames:
            scenes.append(scene)
    return scenes


def load_imptc_sequence(sequence_dir: Path, max_frames: int | None = None, frame_stride: int = 5) -> Scene:
    tracks = []
    tracks.extend(load_track_group(sequence_dir / "vehicles", default_prefix="veh"))
    tracks.extend(load_track_group(sequence_dir / "vrus", default_prefix="vru"))
    if not tracks:
        warn(f"No IMPTC tracks found in {sequence_dir}")
        return Scene(scene_id=sequence_dir.name, source_path=str(sequence_dir))

    reference_track = choose_reference_vehicle(tracks)
    timestamps = sorted({ts for track in tracks for ts in track["states"].keys()}, key=int)
    if frame_stride > 1:
        timestamps = timestamps[::frame_stride]
    if max_frames:
        timestamps = timestamps[:max_frames]
    if not timestamps:
        return Scene(scene_id=sequence_dir.name, source_path=str(sequence_dir))

    first_ts = int(timestamps[0])
    frames: list[Frame] = []
    for frame_idx, ts in enumerate(timestamps):
        ego = state_to_object(reference_track, ts)
        if ego is None:
            # Keep a stable reference so graph coordinates remain meaningful even
            # when the chosen vehicle is not present in every timestamp.
            ego = nearest_state_to_object(reference_track, ts)
        if ego:
            ego.object_id = "reference_vehicle"
            ego.object_type = "ego_vehicle"
        objects = []
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
                map_info={"sequence_dir": str(sequence_dir)},
                raw={"timestamp_us": ts},
            )
        )
    return Scene(scene_id=sequence_dir.name, frames=frames, source_path=str(sequence_dir), raw={"reference_track": reference_track["id"]})


def load_track_group(group_dir: Path, default_prefix: str) -> list[dict[str, Any]]:
    tracks = []
    if not group_dir.exists():
        return tracks
    for track_file in sorted(group_dir.glob("*/track.json")):
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
        class_name = str(overview.get("class_name", default_prefix))
        states = normalize_track_states(raw_states)
        tracks.append(
            {
                "id": track_id,
                "object_type": normalize_imptc_type(class_name, default_prefix),
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
        if not isinstance(state, dict) or "ts" not in state or "coordinates" not in state:
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
        state.update({"vx": vx, "vy": vy, "ax": ax, "ay": ay, "heading": heading})
        states[state["ts"]] = state
    return states


def normalize_imptc_type(class_name: str, default_prefix: str) -> str:
    value = class_name.lower().strip()
    if value == "person":
        return "pedestrian"
    if value == "bicycle":
        return "cyclist"
    if value == "scooter":
        return "e_scooter"
    if value in {"car", "truck", "bus", "motorcycle"}:
        return value
    return value or default_prefix


def choose_reference_vehicle(tracks: list[dict[str, Any]]) -> dict[str, Any]:
    vehicles = [track for track in tracks if track["object_type"] in {"car", "truck", "bus"}]
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
        raw={"source_path": track["path"], "timestamp_us": timestamp, "ground_type": state.get("ground_type")},
    )
    obj.raw.update(
        {
            "raw_class_name": track.get("raw_class_name"),
            "speed": state.get("speed", 0.0),
            "ax": state.get("ax", 0.0),
            "ay": state.get("ay", 0.0),
        }
    )
    return obj


def nearest_state_to_object(track: dict[str, Any], timestamp: str) -> ObjectState | None:
    if not track["states"]:
        return None
    nearest_ts = min(track["states"], key=lambda ts: abs(int(ts) - int(timestamp)))
    return state_to_object(track, nearest_ts)
