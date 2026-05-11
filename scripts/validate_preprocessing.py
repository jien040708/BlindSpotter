#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate generated graph JSON files and print preprocessing statistics.")
    parser.add_argument("--graphs", default="outputs/graphs", help="Directory containing graph JSON files")
    parser.add_argument("--write-summary", action="store_true", help="Write validation_summary.json into the graph directory")
    args = parser.parse_args()

    graph_dir = Path(args.graphs)
    files = sorted(
        path
        for path in graph_dir.glob("*.json")
        if path.name not in {"preprocess_summary.json", "validation_summary.json"}
    )
    if not files:
        print(f"[WARN] No graph JSON files found in {graph_dir}")
        return

    summaries = [validate_graph_file(path) for path in files]
    total_frames = sum(item["frames"] for item in summaries)
    total_nodes = sum(item["nodes"] for item in summaries)
    total_edges = sum(item["edges"] for item in summaries)
    total_blind = sum(item["blind_nodes"] for item in summaries)
    total_positive_blind = sum(item["positive_blind_labels"] for item in summaries)
    errors = [error for item in summaries for error in item["errors"]]
    node_types = Counter()
    for item in summaries:
        node_types.update(item["node_types"])

    print("Validation summary")
    print(f"- graph_files: {len(files)}")
    print(f"- frames: {total_frames}")
    print(f"- nodes: {total_nodes}")
    print(f"- edges: {total_edges}")
    print(f"- blind_nodes: {total_blind}")
    print(f"- positive_blind_labels: {total_positive_blind}")
    print(f"- node_types: {dict(sorted(node_types.items()))}")
    print(f"- errors: {len(errors)}")

    if errors:
        for error in errors[:30]:
            print(f"[ERROR] {error}")
        raise SystemExit(1)

    if args.write_summary:
        output_path = graph_dir / "validation_summary.json"
        output_path.write_text(
            json.dumps(
                {
                    "graph_files": len(files),
                    "frames": total_frames,
                    "nodes": total_nodes,
                    "edges": total_edges,
                    "blind_nodes": total_blind,
                    "positive_blind_labels": total_positive_blind,
                    "node_types": dict(sorted(node_types.items())),
                    "files": summaries,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[OK] Wrote {output_path}")

    print("[OK] Graph validation passed")


def validate_graph_file(path: Path) -> dict[str, Any]:
    graph = json.loads(path.read_text(encoding="utf-8"))
    node_feature_dim = len(graph.get("node_feature_names", []))
    edge_feature_dim = len(graph.get("edge_feature_names", []))
    errors: list[str] = []
    nodes = 0
    edges = 0
    blind_nodes = 0
    positive_blind = 0
    node_types = Counter()

    if not graph.get("frames"):
        errors.append(f"{path}: no frames")

    for frame_idx, frame in enumerate(graph.get("frames", [])):
        frame_prefix = f"{path.name}:frame[{frame_idx}]"
        x = frame.get("x", [])
        edge_index = frame.get("edge_index", [[], []])
        edge_attr = frame.get("edge_attr", [])
        blind_indices = frame.get("blind_node_indices", [])
        blind_y = frame.get("blind_y", [])
        frame_node_types = frame.get("node_types", [])
        n = len(x)
        e = len(edge_attr)
        nodes += n
        edges += e
        blind_nodes += len(blind_indices)
        positive_blind += sum(int(v) for v in blind_y)
        node_types.update(frame_node_types)

        if node_feature_dim and any(len(row) != node_feature_dim for row in x):
            errors.append(f"{frame_prefix}: node feature dimension mismatch")
        if edge_feature_dim and any(len(row) != edge_feature_dim for row in edge_attr):
            errors.append(f"{frame_prefix}: edge feature dimension mismatch")
        if len(edge_index) != 2:
            errors.append(f"{frame_prefix}: edge_index must have two rows")
        elif len(edge_index[0]) != len(edge_index[1]) or len(edge_index[0]) != e:
            errors.append(f"{frame_prefix}: edge_index and edge_attr length mismatch")
        elif any(idx < 0 or idx >= n for row in edge_index for idx in row):
            errors.append(f"{frame_prefix}: edge index out of node range")
        if len(blind_indices) != len(blind_y):
            errors.append(f"{frame_prefix}: blind_node_indices and blind_y length mismatch")
        if any(idx < 0 or idx >= n for idx in blind_indices):
            errors.append(f"{frame_prefix}: blind node index out of range")
        if any(frame_node_types[idx] != "occlusion_zone" for idx in blind_indices if idx < len(frame_node_types)):
            errors.append(f"{frame_prefix}: blind node index does not point to occlusion_zone")

    return {
        "file": str(path),
        "scene_id": graph.get("scene_id"),
        "frames": len(graph.get("frames", [])),
        "nodes": nodes,
        "edges": edges,
        "blind_nodes": blind_nodes,
        "positive_blind_labels": positive_blind,
        "node_types": dict(node_types),
        "errors": errors,
    }


if __name__ == "__main__":
    main()
