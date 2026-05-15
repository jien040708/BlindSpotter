#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.model_adapters import (
    frame_to_eigcn_input,
    frame_to_mrgcn_input,
    frames_to_stgcn_input,
    summarize_adapter_inputs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect canonical graph adapters for MR-GCN, EIGCN, and STGCN.")
    parser.add_argument("--graphs", default="outputs/graphs", help="Directory containing canonical graph JSON files")
    parser.add_argument("--history", type=int, default=5, help="Temporal history window for STGCN adapter")
    parser.add_argument("--output", default=None, help="Optional JSON summary output path")
    args = parser.parse_args()

    graphs = load_graph_jsons(args.graphs)
    if not graphs:
        raise SystemExit(f"No graph JSON files found in {args.graphs}")

    summaries = summarize_adapter_inputs(graphs, history=args.history)
    payload = {name: asdict(summary) for name, summary in summaries.items()}
    payload["example_shapes"] = build_example_shapes(graphs, args.history)

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[OK] Wrote {output_path}")


def build_example_shapes(graphs: list[dict], history: int) -> dict:
    for graph in graphs:
        frames = graph.get("frames", [])
        for idx, frame in enumerate(frames):
            if not frame.get("blind_node_indices"):
                continue
            mrgcn = frame_to_mrgcn_input(frame)
            eigcn = frame_to_eigcn_input(frame)
            stgcn = frames_to_stgcn_input(frames, idx, history=history)
            return {
                "scene_id": graph.get("scene_id"),
                "frame_id": frame.get("frame_id"),
                "mrgcn": {
                    "num_nodes": len(mrgcn["x"]),
                    "num_edges": len(mrgcn["relation_ids"]),
                    "num_targets": len(mrgcn["target_indices"]),
                },
                "eigcn": {
                    "num_nodes": len(eigcn["x"]),
                    "num_edges": len(eigcn["edge_attr"]),
                    "edge_feature_dim": len(eigcn["edge_attr"][0]) if eigcn["edge_attr"] else 0,
                    "num_targets": len(eigcn["target_indices"]),
                },
                "stgcn": {
                    "history_frames": len(stgcn["frames"]),
                    "num_targets": len(stgcn["target_indices"]),
                },
            }
    return {}


def load_graph_jsons(graph_dir: str | Path) -> list[dict]:
    graph_dir = Path(graph_dir)
    paths = sorted(
        path
        for path in graph_dir.glob("*.json")
        if path.name not in {"preprocess_summary.json", "validation_summary.json"}
    )
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


if __name__ == "__main__":
    main()
