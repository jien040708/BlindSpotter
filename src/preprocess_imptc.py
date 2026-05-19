from __future__ import annotations

import argparse
import json
from pathlib import Path

from blindspot_risk.graph_builder import build_scene_graph, save_graph_json
from blindspot_risk.imptc_dataset import load_imptc_scenes
from blindspot_risk.utils import ensure_dir, info


def summarize_graph(graph: dict) -> dict:
    num_frames = len(graph.get("frames", []))
    num_blind_targets = 0
    num_positive_targets = 0
    num_nodes = 0
    num_edges = 0

    for frame in graph.get("frames", []):
        blind_y = frame.get("blind_y", [])
        num_blind_targets += len(blind_y)
        num_positive_targets += sum(int(v) for v in blind_y)
        num_nodes += len(frame.get("node_ids", []))
        edge_index = frame.get("edge_index", [[], []])
        num_edges += len(edge_index[0]) if edge_index and len(edge_index) > 0 else 0

    return {
        "scene_id": graph.get("scene_id"),
        "label_target": graph.get("label_target"),
        "num_frames": num_frames,
        "num_nodes_total": num_nodes,
        "num_edges_total": num_edges,
        "num_blind_targets": num_blind_targets,
        "num_positive_targets": num_positive_targets,
        "positive_rate": num_positive_targets / max(num_blind_targets, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="IMPTC dataset root")
    parser.add_argument("--out", type=str, required=True, help="Output graph json directory")
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "eval", "test"])
    parser.add_argument("--max-tracks-per-split", type=int, default=None)
    parser.add_argument("--max-frames-per-split", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--neighbor-radius", type=float, default=30.0)
    parser.add_argument(
        "--label-target",
        type=str,
        choices=["scooter", "vru"],
        default="scooter",
        help="scooter: scooter-only positive, vru: all VRU positive",
    )

    args = parser.parse_args()

    out_dir = ensure_dir(args.out)

    scenes = load_imptc_scenes(
        root=args.root,
        splits=args.splits,
        max_tracks_per_split=args.max_tracks_per_split,
        max_frames_per_split=args.max_frames_per_split,
        frame_stride=args.frame_stride,
    )

    summaries = []

    for scene in scenes:
        graph = build_scene_graph(
            scene,
            neighbor_radius=args.neighbor_radius,
            label_target=args.label_target,
        )

        output_path = out_dir / f"{scene.scene_id}.json"
        save_graph_json(graph, output_path)

        summary = summarize_graph(graph)
        summaries.append(summary)
        info(f"Saved {output_path}")
        info(json.dumps(summary, indent=2))

    total_targets = sum(item["num_blind_targets"] for item in summaries)
    total_positive = sum(item["num_positive_targets"] for item in summaries)

    global_summary = {
        "root": str(Path(args.root)),
        "out": str(out_dir),
        "splits": args.splits,
        "frame_stride": args.frame_stride,
        "neighbor_radius": args.neighbor_radius,
        "label_target": args.label_target,
        "graphs": summaries,
        "total_blind_targets": total_targets,
        "total_positive_targets": total_positive,
        "total_positive_rate": total_positive / max(total_targets, 1),
    }

    summary_path = out_dir / "preprocess_summary.json"
    summary_path.write_text(json.dumps(global_summary, indent=2), encoding="utf-8")

    info(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()