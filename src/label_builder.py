from __future__ import annotations

from .dataset import Frame, ObjectState, Scene
from .utils import euclidean


SCOOTER_TOKENS = {"scooter", "e-scooter", "electric_scooter", "micromobility"}
OCCLUDER_TOKENS = {"parked", "vehicle", "car", "bus", "truck", "building", "wall", "occluder"}


def is_scooter(object_type: str) -> bool:
    normalized = object_type.lower().replace(" ", "_")
    return any(token in normalized for token in SCOOTER_TOKENS)


def is_occluder(object_type: str) -> bool:
    normalized = object_type.lower().replace(" ", "_")
    return any(token in normalized for token in OCCLUDER_TOKENS)


def build_risk_label(scene: Scene, time_window: float = 3.0, distance_threshold: float = 10.0) -> int:
    """
    Returns 1 if an e-scooter appears after being absent/invisible near an occluder
    and comes within the ego-vehicle risk distance during the time window.
    """
    if not scene.frames:
        return 0

    first_seen: dict[str, float] = {}
    previous_visible: set[str] = set()

    for frame in scene.frames:
        ego_xy = (frame.ego.x, frame.ego.y) if frame.ego else (0.0, 0.0)
        occluders = [obj for obj in frame.objects if is_occluder(obj.object_type)]

        for obj in frame.objects:
            if not is_scooter(obj.object_type) or not obj.visible:
                continue
            obj_xy = (obj.x, obj.y)
            appeared_now = obj.object_id not in previous_visible
            first_seen.setdefault(obj.object_id, frame.timestamp)
            near_ego = euclidean(ego_xy, obj_xy) <= distance_threshold
            near_occluder = any(euclidean((occ.x, occ.y), obj_xy) <= distance_threshold for occ in occluders)

            if appeared_now and near_ego and near_occluder:
                return 1

            if frame.timestamp - first_seen[obj.object_id] <= time_window and near_ego and near_occluder:
                return 1

        previous_visible = {obj.object_id for obj in frame.objects if obj.visible}
    return 0


def build_frame_risk_label(frame: Frame, distance_threshold: float = 10.0) -> int:
    ego_xy = (frame.ego.x, frame.ego.y) if frame.ego else (0.0, 0.0)
    occluders = [obj for obj in frame.objects if is_occluder(obj.object_type)]
    for obj in frame.objects:
        if not is_scooter(obj.object_type) or not obj.visible:
            continue
        obj_xy = (obj.x, obj.y)
        near_ego = euclidean(ego_xy, obj_xy) <= distance_threshold
        near_occluder = any(euclidean((occ.x, occ.y), obj_xy) <= distance_threshold for occ in occluders)
        if near_ego and near_occluder:
            return 1
    return 0


def build_blind_zone_label(
    scene: Scene,
    frame_index: int,
    blind_zone: ObjectState,
    time_window: float = 3.0,
    distance_threshold: float = 6.0,
) -> int:
    start_time = scene.frames[frame_index].timestamp
    for future in scene.frames[frame_index + 1 :]:
        if future.timestamp - start_time > time_window:
            break
        for obj in future.objects:
            if is_scooter(obj.object_type) and obj.visible:
                if euclidean((blind_zone.x, blind_zone.y), (obj.x, obj.y)) <= distance_threshold:
                    return 1
    return 0
