from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any


RELATION_TO_ID = {
    "spatial_near": 0,
    "potential_conflict": 1,
    "occludes": 2,
    "blind_zone_relation": 3,
    "temporal_next": 4,
    "ego_to_blind_zone": 5,
    "blind_zone_to_ego": 6,
    "unknown": 99,
}


@dataclass
class AdapterSummary:
    name: str
    num_samples: int
    num_targets: int
    num_positive_targets: int
    node_types: dict[str, int]
    relation_types: dict[str, int]


def frame_to_mrgcn_input(frame: dict[str, Any]) -> dict[str, Any]:
    """Convert a canonical frame graph to relation-aware MR-GCN input."""
    edge_types = frame.get("edge_type", [])
    relation_ids = [RELATION_TO_ID.get(edge_type, RELATION_TO_ID["unknown"]) for edge_type in edge_types]
    return {
        "x": frame.get("x", []),
        "edge_index": frame.get("edge_index", [[], []]),
        "relation_ids": relation_ids,
        "target_indices": frame.get("blind_node_indices", []),
        "y": frame.get("blind_y", []),
        "node_ids": frame.get("node_ids", []),
        "node_types": frame.get("node_types", []),
    }


def frame_to_eigcn_input(frame: dict[str, Any]) -> dict[str, Any]:
    """Convert a canonical frame graph to expert-informed edge-feature input."""
    return {
        "x": frame.get("x", []),
        "edge_index": frame.get("edge_index", [[], []]),
        "edge_attr": frame.get("edge_attr", []),
        "edge_type": frame.get("edge_type", []),
        "target_indices": frame.get("blind_node_indices", []),
        "y": frame.get("blind_y", []),
        "node_ids": frame.get("node_ids", []),
        "node_types": frame.get("node_types", []),
    }


def frames_to_stgcn_input(frames: list[dict[str, Any]], current_index: int, history: int = 5) -> dict[str, Any]:
    """Create a temporal window ending at current_index for STGCN-style models."""
    start = max(0, current_index - history + 1)
    window = frames[start : current_index + 1]
    current = frames[current_index]
    return {
        "frames": [
            {
                "x": frame.get("x", []),
                "edge_index": frame.get("edge_index", [[], []]),
                "edge_attr": frame.get("edge_attr", []),
                "edge_type": frame.get("edge_type", []),
                "node_ids": frame.get("node_ids", []),
                "node_types": frame.get("node_types", []),
            }
            for frame in window
        ],
        "target_indices": current.get("blind_node_indices", []),
        "target_node_ids": [current.get("node_ids", [])[idx] for idx in current.get("blind_node_indices", [])],
        "y": current.get("blind_y", []),
        "frame_id": current.get("frame_id"),
        "timestamp": current.get("timestamp"),
    }


def summarize_adapter_inputs(graphs: list[dict[str, Any]], history: int = 5) -> dict[str, AdapterSummary]:
    mrgcn_targets = 0
    eigcn_targets = 0
    stgcn_targets = 0
    positives = 0
    node_types = Counter()
    relation_types = Counter()
    frame_count = 0
    temporal_count = 0

    for graph in graphs:
        frames = graph.get("frames", [])
        for idx, frame in enumerate(frames):
            targets = frame.get("blind_y", [])
            target_count = len(targets)
            if target_count == 0:
                continue
            frame_count += 1
            temporal_count += 1 if idx >= 0 else 0
            mrgcn_targets += target_count
            eigcn_targets += target_count
            stgcn_targets += target_count
            positives += sum(int(v) for v in targets)
            node_types.update(frame.get("node_types", []))
            relation_types.update(frame.get("edge_type", []))

    return {
        "mrgcn": AdapterSummary(
            name="mrgcn",
            num_samples=frame_count,
            num_targets=mrgcn_targets,
            num_positive_targets=positives,
            node_types=dict(sorted(node_types.items())),
            relation_types=dict(sorted(relation_types.items())),
        ),
        "eigcn": AdapterSummary(
            name="eigcn",
            num_samples=frame_count,
            num_targets=eigcn_targets,
            num_positive_targets=positives,
            node_types=dict(sorted(node_types.items())),
            relation_types=dict(sorted(relation_types.items())),
        ),
        "stgcn": AdapterSummary(
            name="stgcn",
            num_samples=temporal_count,
            num_targets=stgcn_targets,
            num_positive_targets=positives,
            node_types=dict(sorted(node_types.items())),
            relation_types=dict(sorted(relation_types.items())),
        ),
    }
