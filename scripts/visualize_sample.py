#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import FlexibleSceneDataset
from src.graph_builder import build_blind_zone_nodes
from src.imptc_dataset import is_imptc_root, load_imptc_scenes
from src.label_builder import build_frame_risk_label, is_occluder, is_scooter
from src.utils import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Save 2D top-view visualizations of sample scenes.")
    parser.add_argument("--root", default="data/sample", help="Dataset root directory")
    parser.add_argument("--output", default="outputs/figures", help="Figure output directory")
    parser.add_argument("--max-files", type=int, default=3, help="Maximum scene files to visualize")
    parser.add_argument("--max-frames", type=int, default=20, help="Maximum frames per scene to save")
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib is required for visualization. Install it and rerun.")
        return

    output_dir = ensure_dir(args.output)
    if is_imptc_root(args.root):
        scenes = load_imptc_scenes(args.root, max_sequences=args.max_files, max_frames_per_sequence=args.max_frames)
    else:
        scenes = FlexibleSceneDataset(args.root, max_files=args.max_files).load_scenes()
    if not scenes:
        print("[WARN] No parseable scenes found. Add sample annotation files under data/sample and rerun.")
        return

    for scene in scenes:
        for frame in scene.frames[: args.max_frames]:
            save_frame_plot(scene.scene_id, frame, output_dir, plt)


def save_frame_plot(scene_id: str, frame, output_dir: Path, plt) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    ego = frame.ego
    if ego:
        ax.scatter([ego.x], [ego.y], s=140, marker="s", c="#2563eb", label="ego")
        ax.arrow(ego.x, ego.y, 2.0, 0.0, head_width=0.5, color="#2563eb")

    for obj in frame.objects:
        color = "#6b7280"
        marker = "o"
        label = obj.object_type
        if is_scooter(obj.object_type):
            color = "#dc2626"
            marker = "^"
            label = "e-scooter"
        elif is_occluder(obj.object_type):
            color = "#111827"
            marker = "s"
            label = "occluder"
            ax.add_patch(plt.Circle((obj.x, obj.y), 5.0, color="#111827", alpha=0.08))
        ax.scatter([obj.x], [obj.y], s=70, marker=marker, c=color, label=label)
        ax.text(obj.x + 0.4, obj.y + 0.4, obj.object_id, fontsize=8)

    for blind in build_blind_zone_nodes(frame):
        polygon = blind.raw.get("polygon", [])
        if polygon:
            xs = [p[0] for p in polygon] + [polygon[0][0]]
            ys = [p[1] for p in polygon] + [polygon[0][1]]
            ax.fill(xs, ys, color="#f97316", alpha=0.16, label="blind-zone")
            ax.plot(xs, ys, color="#f97316", linestyle="--", linewidth=1.0)
        ax.scatter([blind.x], [blind.y], s=45, marker="x", c="#f97316")

    risk = build_frame_risk_label(frame)
    ax.set_title(f"{scene_id} frame={frame.frame_id} risk={risk}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.25)
    ax.axis("equal")
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    if unique:
        ax.legend(unique.values(), unique.keys(), loc="upper right")
    fig.tight_layout()
    output_path = output_dir / f"{scene_id}_frame_{frame.frame_id}.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"[OK] Saved {output_path}")


if __name__ == "__main__":
    main()
