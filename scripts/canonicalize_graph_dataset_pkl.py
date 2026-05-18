#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import pickle
from copy import deepcopy
from pathlib import Path
from typing import Any


CANONICAL_NODE_FEATURES = [
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

CANONICAL_EDGE_FEATURES = [
    "distance",
    "relative_velocity_x",
    "relative_velocity_y",
    "relative_heading",
    "time_to_collision",
    "visibility_blocked",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert generated graph_dataset.pkl to the canonical IMPTC graph feature contract.")
    parser.add_argument("--input", default="graph_dataset.pkl")
    parser.add_argument("--output", default="outputs/models/graph_dataset_canonical.pkl")
    args = parser.parse_args()

    with Path(args.input).open("rb") as file:
        payload = pickle.load(file)

    converted = deepcopy(payload)
    converted["node_feature_names"] = CANONICAL_NODE_FEATURES
    converted["edge_feature_names"] = CANONICAL_EDGE_FEATURES
    for split, records in converted["dataset"].items():
        converted["dataset"][split] = [convert_record(record) for record in records]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as file:
        pickle.dump(converted, file)
    print(f"[OK] wrote {output}")
    for split, records in converted["dataset"].items():
        positives = sum(int(record["label"]) for record in records)
        print(f"{split}: samples={len(records)} positives={positives}")


def convert_record(record: dict[str, Any]) -> dict[str, Any]:
    old_x = record["x"]
    old_edge_attr = record["edge_attr"]
    edge_index = record["edge_index"]
    bz_idx = int(record["bz_node_idx"])
    new_record = deepcopy(record)
    new_record["x"] = [convert_node_features(features, idx == bz_idx) for idx, features in enumerate(old_x)]
    new_record["edge_attr"] = [
        convert_edge_features(old_edge_attr[idx], old_x[src], old_x[dst])
        for idx, (src, dst) in enumerate(zip(edge_index[0], edge_index[1]))
    ]
    return new_record


def convert_node_features(features: list[float], is_blind_zone: bool) -> list[float]:
    x, y, vx, vy, heading, type_id, speed, is_occluder = features
    distance = math.hypot(x, y)
    relative_angle = math.atan2(y, x)
    visibility = 0.0 if is_blind_zone else 1.0
    is_vru = 1.0 if int(type_id) in {2, 3, 4} else 0.0
    blind_zone_area = 0.0
    return [
        float(x),
        float(y),
        float(vx),
        float(vy),
        float(heading),
        float(type_id),
        float(distance),
        float(relative_angle),
        visibility,
        float(is_occluder),
        is_vru,
        float(speed),
        0.0,
        blind_zone_area,
    ]


def convert_edge_features(edge_attr: list[float], src_features: list[float], dst_features: list[float]) -> list[float]:
    distance, rel_vx, rel_vy, rel_heading, visibility_blocked = edge_attr
    dx = float(dst_features[0] - src_features[0])
    dy = float(dst_features[1] - src_features[1])
    closing_speed = -((dx * rel_vx + dy * rel_vy) / max(float(distance), 1e-6))
    ttc = float(distance) / max(closing_speed, 1e-6) if closing_speed > 0 else 999.0
    return [
        float(distance),
        float(rel_vx),
        float(rel_vy),
        float(rel_heading),
        min(ttc, 999.0),
        float(visibility_blocked),
    ]


if __name__ == "__main__":
    main()
