#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import FlexibleSceneDataset
from src.graph_builder import build_scene_graph, save_graph_json
from src.imptc_dataset import is_imptc_root, load_imptc_scenes
from src.utils import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess sample scenes into graph JSON or pickle files.")
    parser.add_argument("--root", default="data/sample", help="Dataset root directory")
    parser.add_argument("--output", default="outputs/graphs", help="Output directory")
    parser.add_argument("--max-files", type=int, default=10, help="Maximum annotation-like files to parse")
    parser.add_argument("--max-sequences", type=int, default=None, help="Maximum IMPTC sequences to parse")
    parser.add_argument("--max-frames", type=int, default=500, help="Maximum frames per IMPTC sequence")
    parser.add_argument("--frame-stride", type=int, default=5, help="Use every Nth IMPTC frame")
    parser.add_argument("--neighbor-radius", type=float, default=30.0, help="Spatial graph neighbor radius in meters")
    parser.add_argument(
        "--label-target",
        choices=["scooter", "vru"],
        default="scooter",
        help="Positive blind-zone target: scooter-only or all VRU classes",
    )
    parser.add_argument("--format", choices=["json", "pkl"], default="json", help="Output graph format")
    args = parser.parse_args()

    output_dir = ensure_dir(args.output)
    if is_imptc_root(args.root):
        scenes = load_imptc_scenes(
            args.root,
            max_sequences=args.max_sequences,
            max_frames_per_sequence=args.max_frames,
            frame_stride=args.frame_stride,
        )
    else:
        dataset = FlexibleSceneDataset(args.root, max_files=args.max_files)
        scenes = dataset.load_scenes()
    if not scenes:
        print("[WARN] No parseable scenes found. Add sample annotation files under data/sample and rerun.")
        return

    summaries = []
    for scene in scenes:
        graph = build_scene_graph(scene, neighbor_radius=args.neighbor_radius, label_target=args.label_target)
        output_path = output_dir / f"{scene.scene_id}.{args.format}"
        if args.format == "json":
            save_graph_json(graph, output_path)
        else:
            with output_path.open("wb") as f:
                pickle.dump(graph, f)
        summary = summarize_graph(graph)
        summaries.append({"output_path": str(output_path), **summary})
        print(
            f"[OK] Saved {output_path} "
            f"(frames={summary['frames']}, blind_nodes={summary['blind_nodes']}, "
            f"positive_blind_labels={summary['positive_blind_labels']})"
        )

    summary_path = output_dir / "preprocess_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"[OK] Wrote {summary_path}")


def summarize_graph(graph: dict) -> dict:
    frames = graph.get("frames", [])
    blind_nodes = sum(len(frame.get("blind_node_indices", [])) for frame in frames)
    positive_blind = sum(sum(int(v) for v in frame.get("blind_y", [])) for frame in frames)
    nodes = sum(len(frame.get("node_ids", [])) for frame in frames)
    edges = sum(len(frame.get("edge_attr", [])) for frame in frames)
    node_types = {}
    for frame in frames:
        for node_type in frame.get("node_types", []):
            node_types[node_type] = node_types.get(node_type, 0) + 1
    return {
        "scene_id": graph.get("scene_id"),
        "frames": len(frames),
        "nodes": nodes,
        "edges": edges,
        "blind_nodes": blind_nodes,
        "positive_blind_labels": positive_blind,
        "scene_label": graph.get("y"),
        "label_target": graph.get("label_target", "scooter"),
        "node_types": dict(sorted(node_types.items())),
    }


if __name__ == "__main__":
    main()
