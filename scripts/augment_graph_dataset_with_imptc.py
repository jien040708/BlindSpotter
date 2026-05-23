#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
from copy import deepcopy
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.gnn_dataset import EDGE_ALIASES, NODE_ALIASES, feature_indices
from src.gnn_dataset import load_graph_jsons, select_feature_columns


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge canonical IMPTC graph JSON samples into graph_dataset.pkl.")
    parser.add_argument("--base", default="graph_dataset.pkl", help="Base generated graph dataset pickle")
    parser.add_argument("--graphs", default="outputs/graphs_validation_all", help="Canonical IMPTC graph JSON directory")
    parser.add_argument("--output", default="outputs/models/graph_dataset_imptc_augmented.pkl")
    parser.add_argument("--train-scenes", nargs="*", default=None, help="Scene IDs to append to train")
    parser.add_argument("--val-scenes", nargs="*", default=None, help="Scene IDs to append to val")
    parser.add_argument("--test-scenes", nargs="*", default=None, help="Scene IDs to append to test")
    args = parser.parse_args()

    with Path(args.base).open("rb") as file:
        payload = pickle.load(file)
    augmented = deepcopy(payload)
    graphs = load_graph_jsons(args.graphs)
    scene_ids = [str(graph.get("scene_id")) for graph in graphs]

    train_scenes = set(args.train_scenes or scene_ids[:2])
    val_scenes = set(args.val_scenes or scene_ids[2:3])
    test_scenes = set(args.test_scenes or scene_ids[3:])
    split_by_scene = {scene_id: "train" for scene_id in train_scenes}
    split_by_scene.update({scene_id: "val" for scene_id in val_scenes})
    split_by_scene.update({scene_id: "test" for scene_id in test_scenes})

    stats: dict[str, dict[str, int]] = {
        split: {"samples": 0, "positives": 0}
        for split in ("train", "val", "test", "skipped")
    }
    for graph in graphs:
        scene_id = str(graph.get("scene_id"))
        split = split_by_scene.get(scene_id, "skipped")
        records = graph_to_records(graph, payload["node_feature_names"], payload["edge_feature_names"], split)
        stats[split]["samples"] += len(records)
        stats[split]["positives"] += sum(int(record["label"]) for record in records)
        if split != "skipped":
            augmented["dataset"][split].extend(records)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as file:
        pickle.dump(augmented, file)

    print(f"[OK] wrote {output}")
    print(f"scenes: train={sorted(train_scenes)}, val={sorted(val_scenes)}, test={sorted(test_scenes)}")
    for split in ("train", "val", "test", "skipped"):
        before = len(payload["dataset"].get(split, [])) if split in payload["dataset"] else 0
        after = len(augmented["dataset"].get(split, [])) if split in augmented["dataset"] else 0
        print(
            f"{split}: added={stats[split]['samples']} positives={stats[split]['positives']} "
            f"before={before} after={after}"
        )


def graph_to_records(
    graph: dict[str, Any],
    target_node_features: list[str],
    target_edge_features: list[str],
    split: str,
) -> list[dict[str, Any]]:
    node_indices = feature_indices(graph.get("node_feature_names", []), target_node_features, NODE_ALIASES)
    edge_indices = feature_indices(graph.get("edge_feature_names", []), target_edge_features, EDGE_ALIASES)
    records = []
    scene_id = str(graph.get("scene_id", "unknown"))
    for frame in graph.get("frames", []):
        blind_indices = frame.get("blind_node_indices", [])
        blind_y = frame.get("blind_y", [])
        if not blind_indices:
            continue
        x = select_feature_columns_tensor(frame.get("x", []), node_indices)
        edge_attr = select_feature_columns_tensor(frame.get("edge_attr", []), edge_indices)
        node_ids = [str(node_id) for node_id in frame.get("node_ids", [])]
        for blind_pos, blind_idx in enumerate(blind_indices):
            label = int(blind_y[blind_pos]) if blind_pos < len(blind_y) else 0
            records.append(
                {
                    "scene_id": f"imptc_{scene_id}_{frame.get('frame_id', '')}_{blind_idx}",
                    "split": split,
                    "pm_class": "imptc_blind_zone",
                    "label": label,
                    "x": x,
                    "edge_index": frame.get("edge_index", [[], []]),
                    "edge_attr": edge_attr,
                    "edge_type": frame.get("edge_type", []),
                    "node_types": frame.get("node_types", []),
                    "bz_node_idx": int(blind_idx),
                    "occ_node_idx": infer_occluder_index(node_ids, int(blind_idx)),
                    "meta": {
                        "source": "imptc_canonical_graph",
                        "original_scene_id": scene_id,
                        "frame_id": str(frame.get("frame_id", "")),
                        "timestamp": float(frame.get("timestamp", 0.0)),
                    },
                }
            )
    return records


def select_feature_columns_tensor(values: list[list[float]], indices: list[int | None]) -> list[list[float]]:
    import torch

    tensor = torch.tensor(values, dtype=torch.float32)
    return select_feature_columns(tensor, indices).tolist()


def infer_occluder_index(node_ids: list[str], blind_idx: int) -> int:
    if blind_idx < 0 or blind_idx >= len(node_ids):
        return -1
    blind_id = node_ids[blind_idx]
    if not blind_id.startswith("blind_"):
        return -1
    occluder_id = blind_id.removeprefix("blind_")
    try:
        return node_ids.index(occluder_id)
    except ValueError:
        return -1


if __name__ == "__main__":
    main()
