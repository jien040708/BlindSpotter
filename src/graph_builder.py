from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .dataset import Frame, ObjectState, Scene
from .label_builder import build_blind_zone_label, build_frame_risk_label, build_risk_label
from .utils import euclidean, normalize_angle, object_type_flags, polygon_area, polygon_centroid, relative_angle


OBJECT_TYPE_TO_ID = {
    "unknown": 0,
    "ego_vehicle": 1,
    "e_scooter": 2,
    "scooter": 2,
    "pedestrian": 3,
    "cyclist": 12,
    "vehicle": 4,
    "car": 4,
    "parked_vehicle": 5,
    "bus": 6,
    "truck": 7,
    "crosswalk": 8,
    "sidewalk": 9,
    "lane": 10,
    "occlusion_zone": 11,
}

NODE_FEATURE_NAMES = [
    "x",
    "y",
    "vx",
    "vy",
    "heading",
    "object_type_id",
    "distance_to_reference",
    "relative_angle_to_reference",
    "visibility",
    "is_occluder",
    "is_vulnerable_road_user",
    "speed",
    "acceleration",
    "blind_zone_area",
]

EDGE_FEATURE_NAMES = [
    "distance",
    "relative_velocity_x",
    "relative_velocity_y",
    "relative_heading",
    "time_to_collision",
    "visibility_blocked",
]


def type_id(object_type: str) -> int:
    normalized = object_type.lower().replace(" ", "_")
    return OBJECT_TYPE_TO_ID.get(normalized, OBJECT_TYPE_TO_ID["unknown"])


def build_scene_graph(scene: Scene, neighbor_radius: float = 30.0) -> dict[str, Any]:
    frame_graphs = [
        build_frame_graph(frame, neighbor_radius=neighbor_radius, scene=scene, frame_index=idx)
        for idx, frame in enumerate(scene.frames)
    ]
    temporal_edges = build_temporal_edges(scene)
    return {
        "scene_id": scene.scene_id,
        "source_path": scene.source_path,
        "reference_track": scene.raw.get("reference_track") if isinstance(scene.raw, dict) else None,
        "node_feature_names": NODE_FEATURE_NAMES,
        "edge_feature_names": EDGE_FEATURE_NAMES,
        "y": build_risk_label(scene),
        "frames": frame_graphs,
        "temporal_edges": temporal_edges,
    }


def build_frame_graph(
    frame: Frame,
    neighbor_radius: float = 30.0,
    scene: Scene | None = None,
    frame_index: int | None = None,
) -> dict[str, Any]:
    nodes = [frame.ego] if frame.ego else []
    nodes.extend(frame.objects)
    blind_nodes = build_blind_zone_nodes(frame)
    blind_start_idx = len(nodes)
    nodes.extend(blind_nodes)
    node_ids = [node.object_id for node in nodes]
    x = [node_features(node, frame.ego) for node in nodes]

    edge_index: list[list[int]] = [[], []]
    edge_attr: list[list[float]] = []
    edge_type: list[str] = []
    for src_idx, src in enumerate(nodes):
        for dst_idx, dst in enumerate(nodes):
            if src_idx == dst_idx:
                continue
            distance = euclidean((src.x, src.y), (dst.x, dst.y))
            if distance > neighbor_radius:
                continue
            edge_index[0].append(src_idx)
            edge_index[1].append(dst_idx)
            edge_attr.append(edge_features(src, dst, distance))
            edge_type.append(infer_edge_type(src, dst, distance))

    return {
        "frame_id": frame.frame_id,
        "timestamp": frame.timestamp,
        "node_ids": node_ids,
        "node_types": [node.object_type for node in nodes],
        "x": x,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "edge_type": edge_type,
        "y": build_frame_risk_label(frame),
        "blind_node_indices": list(range(blind_start_idx, blind_start_idx + len(blind_nodes))),
        "blind_y": build_blind_labels(scene, frame_index, blind_nodes),
    }


def node_features(node: ObjectState, ego: ObjectState | None) -> list[float]:
    ego = ego or ObjectState(object_id="ego", object_type="ego_vehicle")
    distance = euclidean((ego.x, ego.y), (node.x, node.y))
    angle = relative_angle((ego.x, ego.y), ego.heading, (node.x, node.y))
    is_occluder, is_vru = object_type_flags(node.object_type)
    return [
        node.x,
        node.y,
        node.vx,
        node.vy,
        node.heading,
        float(type_id(node.object_type)),
        distance,
        angle,
        float(node.visible),
        float(is_occluder),
        float(is_vru),
        float(node.raw.get("speed", (node.vx * node.vx + node.vy * node.vy) ** 0.5) if isinstance(node.raw, dict) else 0.0),
        float(
            (
                (node.raw.get("ax", 0.0) ** 2 + node.raw.get("ay", 0.0) ** 2) ** 0.5
                if isinstance(node.raw, dict)
                else 0.0
            )
        ),
        float(node.raw.get("area", 0.0) if isinstance(node.raw, dict) else 0.0),
    ]


def edge_features(src: ObjectState, dst: ObjectState, distance: float) -> list[float]:
    rel_vx = dst.vx - src.vx
    rel_vy = dst.vy - src.vy
    rel_heading = normalize_angle(dst.heading - src.heading)
    closing_speed = -(((dst.x - src.x) * rel_vx + (dst.y - src.y) * rel_vy) / max(distance, 1e-6))
    ttc = distance / max(closing_speed, 1e-6) if closing_speed > 0 else 999.0
    visibility_blocked = float((not dst.visible) or (object_type_flags(src.object_type)[0] and distance < 10.0))
    return [distance, rel_vx, rel_vy, rel_heading, min(ttc, 999.0), visibility_blocked]


def infer_edge_type(src: ObjectState, dst: ObjectState, distance: float) -> str:
    if src.object_type == "occlusion_zone" or dst.object_type == "occlusion_zone":
        return "blind_zone_relation"
    src_occ = object_type_flags(src.object_type)[0]
    dst_vru = object_type_flags(dst.object_type)[1]
    if src_occ and dst_vru and distance < 10.0:
        return "occludes"
    if distance < 5.0:
        return "potential_conflict"
    return "spatial_near"


def build_blind_zone_nodes(
    frame: Frame,
    max_zones: int = 8,
    occluder_radius: float = 2.5,
    depth_m: float = 25.0,
    max_occluder_distance: float = 35.0,
) -> list[ObjectState]:
    if frame.ego is None:
        return []
    ego_xy = (frame.ego.x, frame.ego.y)
    candidates = []
    for obj in frame.objects:
        is_occluder = obj.object_type in {"car", "truck", "bus", "vehicle", "parked_vehicle"}
        if not is_occluder:
            continue
        distance = euclidean(ego_xy, (obj.x, obj.y))
        if distance > max_occluder_distance or distance <= occluder_radius:
            continue
        polygon = compute_blind_zone_polygon(ego_xy, (obj.x, obj.y), occluder_radius=occluder_radius, depth_m=depth_m)
        if not polygon:
            continue
        cx, cy = polygon_centroid(polygon)
        area = polygon_area(polygon)
        angle_center = math.atan2(obj.y - frame.ego.y, obj.x - frame.ego.x)
        candidates.append(
            (
                distance,
                ObjectState(
                    object_id=f"blind_{obj.object_id}",
                    object_type="occlusion_zone",
                    x=cx,
                    y=cy,
                    vx=0.0,
                    vy=0.0,
                    heading=angle_center,
                    visible=False,
                    raw={
                        "occluder_id": obj.object_id,
                        "occluder_type": obj.object_type,
                        "occluder_distance": distance,
                        "polygon": polygon,
                        "area": area,
                        "occluder_radius": occluder_radius,
                        "depth_m": depth_m,
                    },
                ),
            )
        )
    candidates.sort(key=lambda item: item[0])
    return [node for _, node in candidates[:max_zones]]


def compute_blind_zone_polygon(
    ego_xy: tuple[float, float],
    occluder_xy: tuple[float, float],
    occluder_radius: float = 2.5,
    depth_m: float = 25.0,
) -> list[tuple[float, float]] | None:
    ex, ey = ego_xy
    ox, oy = occluder_xy
    dx = ox - ex
    dy = oy - ey
    distance = math.hypot(dx, dy)
    if distance <= occluder_radius:
        return None
    angle_center = math.atan2(dy, dx)
    half_angle = math.asin(min(occluder_radius / distance, 1.0))
    angle_left = angle_center + half_angle
    angle_right = angle_center - half_angle
    tangent_left = (ox - occluder_radius * math.sin(angle_center), oy + occluder_radius * math.cos(angle_center))
    tangent_right = (ox + occluder_radius * math.sin(angle_center), oy - occluder_radius * math.cos(angle_center))
    far_left = (ox + depth_m * math.cos(angle_left), oy + depth_m * math.sin(angle_left))
    far_center = (ox + depth_m * math.cos(angle_center), oy + depth_m * math.sin(angle_center))
    far_right = (ox + depth_m * math.cos(angle_right), oy + depth_m * math.sin(angle_right))
    return [tangent_left, far_left, far_center, far_right, tangent_right]


def build_blind_labels(scene: Scene | None, frame_index: int | None, blind_nodes: list[ObjectState]) -> list[int]:
    if scene is None or frame_index is None:
        return [0 for _ in blind_nodes]
    return [build_blind_zone_label(scene, frame_index, node) for node in blind_nodes]


def build_temporal_edges(scene: Scene) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    previous: dict[str, tuple[int, int]] = {}
    for frame_idx, frame in enumerate(scene.frames):
        nodes = ([frame.ego] if frame.ego else []) + frame.objects
        for node_idx, node in enumerate(nodes):
            if node.object_id in previous:
                prev_frame, prev_node = previous[node.object_id]
                edges.append(
                    {
                        "source_frame": prev_frame,
                        "source_node": prev_node,
                        "target_frame": frame_idx,
                        "target_node": node_idx,
                        "edge_type": "temporal_next",
                    }
                )
            previous[node.object_id] = (frame_idx, node_idx)
    return edges


def save_graph_json(graph: dict[str, Any], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.write_text(json.dumps(graph, indent=2), encoding="utf-8")
