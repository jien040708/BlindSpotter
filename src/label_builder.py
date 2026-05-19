from __future__ import annotations

from .dataset import Frame, ObjectState, Scene
from .utils import euclidean, point_in_polygon


SCOOTER_TOKENS = {
    "scooter",
    "e_scooter",
    "e-scooter",
    "electric_scooter",
    "micromobility",
    "personal_mobility",
}

VRU_TOKENS = {
    "scooter",
    "e_scooter",
    "e-scooter",
    "electric_scooter",
    "micromobility",
    "personal_mobility",
    "pedestrian",
    "person",
    "walker",
    "cyclist",
    "bicycle",
    "bike",
    "wheelchair",
    "stroller",
}

OCCLUDER_TOKENS = {
    "parked",
    "vehicle",
    "car",
    "bus",
    "truck",
    "building",
    "wall",
    "occluder",
}


def is_scooter(object_type: str) -> bool:
    normalized = object_type.lower().replace(" ", "_")
    return any(token in normalized for token in SCOOTER_TOKENS)


def is_vru(object_type: str) -> bool:
    normalized = object_type.lower().replace(" ", "_")
    return any(token in normalized for token in VRU_TOKENS)


def is_occluder(object_type: str) -> bool:
    normalized = object_type.lower().replace(" ", "_")
    return any(token in normalized for token in OCCLUDER_TOKENS)


def is_positive_target(object_type: str, label_target: str = "scooter") -> bool:
    """
    label_target:
      - "scooter": e-scooter / personal mobility only
      - "vru": scooter + pedestrian + cyclist + wheelchair + stroller ...
    """
    if label_target == "scooter":
        return is_scooter(object_type)
    if label_target == "vru":
        return is_vru(object_type)
    raise ValueError(f"Unknown label_target: {label_target}")


def build_risk_label(
    scene: Scene,
    time_window: float = 3.0,
    distance_threshold: float = 10.0,
    label_target: str = "scooter",
) -> int:

    if not scene.frames:
        return 0

    first_seen: dict[str, float] = {}
    previous_visible: set[str] = set()

    for frame in scene.frames:
        ego_xy = (frame.ego.x, frame.ego.y) if frame.ego else (0.0, 0.0)
        occluders = [obj for obj in frame.objects if is_occluder(obj.object_type)]

        for obj in frame.objects:
            if not is_positive_target(obj.object_type, label_target=label_target):
                continue
            if not obj.visible:
                continue

            obj_xy = (obj.x, obj.y)
            appeared_now = obj.object_id not in previous_visible
            first_seen.setdefault(obj.object_id, frame.timestamp)

            near_ego = euclidean(ego_xy, obj_xy) <= distance_threshold
            near_occluder = any(
                euclidean((occ.x, occ.y), obj_xy) <= distance_threshold
                for occ in occluders
            )

            if appeared_now and near_ego and near_occluder:
                return 1

            if frame.timestamp - first_seen[obj.object_id] <= time_window and near_ego and near_occluder:
                return 1

        previous_visible = {
            obj.object_id
            for obj in frame.objects
            if obj.visible
        }

    return 0


def build_frame_risk_label(
    frame: Frame,
    distance_threshold: float = 10.0,
    label_target: str = "scooter",
) -> int:

    ego_xy = (frame.ego.x, frame.ego.y) if frame.ego else (0.0, 0.0)
    occluders = [obj for obj in frame.objects if is_occluder(obj.object_type)]

    for obj in frame.objects:
        if not is_positive_target(obj.object_type, label_target=label_target):
            continue
        if not obj.visible:
            continue

        obj_xy = (obj.x, obj.y)

        near_ego = euclidean(ego_xy, obj_xy) <= distance_threshold
        near_occluder = any(
            euclidean((occ.x, occ.y), obj_xy) <= distance_threshold
            for occ in occluders
        )

        if near_ego and near_occluder:
            return 1

    return 0


def build_blind_zone_label(
    scene: Scene,
    frame_index: int,
    blind_zone: ObjectState,
    time_window: float = 3.0,
    distance_threshold: float = 6.0,
    label_target: str = "scooter",
) -> int:
    """
    현재 frame의 blind-zone node에 대해,
    앞으로 time_window 초 안에 target object가 해당 blind-zone 영역 안에 나타나면 1.

    label_target="scooter": scooter만 positive
    label_target="vru": pedestrian/cyclist/scooter 등 VRU 전체 positive
    """
    start_time = scene.frames[frame_index].timestamp
    polygon = blind_zone.raw.get("polygon") if isinstance(blind_zone.raw, dict) else None

    for future in scene.frames[frame_index + 1:]:
        if future.timestamp - start_time > time_window:
            break

        for obj in future.objects:
            if not is_positive_target(obj.object_type, label_target=label_target):
                continue
            if not obj.visible:
                continue

            obj_xy = (obj.x, obj.y)

            if polygon and point_in_polygon(obj_xy, polygon):
                return 1

            if euclidean((blind_zone.x, blind_zone.y), obj_xy) <= distance_threshold:
                return 1

    return 0