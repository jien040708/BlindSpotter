#!/usr/bin/env python3
from __future__ import annotations

import argparse
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

    for scene in scenes:
        graph = build_scene_graph(scene, neighbor_radius=args.neighbor_radius)
        output_path = output_dir / f"{scene.scene_id}.{args.format}"
        if args.format == "json":
            save_graph_json(graph, output_path)
        else:
            with output_path.open("wb") as f:
                pickle.dump(graph, f)
        print(f"[OK] Saved {output_path} (frames={len(graph['frames'])}, scene_label={graph['y']})")


if __name__ == "__main__":
    main()
