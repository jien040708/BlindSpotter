#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create repeated scene-level train/val splits for canonical graph JSONs.")
    parser.add_argument("--summary", default="outputs/graphs_imptc_set01/preprocess_summary.json")
    parser.add_argument("--output-dir", default="outputs/splits/repeated")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--prefix", default="imptc_scene", help="Output filename prefix")
    args = parser.parse_args()

    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    positive_scenes = [row["scene_id"] for row in summary if int(row["positive_blind_labels"]) > 0]
    negative_scenes = [row["scene_id"] for row in summary if int(row["positive_blind_labels"]) == 0]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for repeat in range(args.repeats):
        rng = random.Random(args.seed + repeat)
        pos = positive_scenes[:]
        neg = negative_scenes[:]
        rng.shuffle(pos)
        rng.shuffle(neg)
        train_pos_count = max(1, int(round(len(pos) * args.train_ratio)))
        train_neg_count = max(1, int(round(len(neg) * args.train_ratio)))
        split = {
            "train": sorted(pos[:train_pos_count] + neg[:train_neg_count]),
            "val": sorted(pos[train_pos_count:] + neg[train_neg_count:]),
            "test": [],
            "metadata": {
                "seed": args.seed + repeat,
                "train_ratio": args.train_ratio,
                "positive_scene_count": len(pos),
                "negative_scene_count": len(neg),
                "split_unit": "scene",
            },
        }
        train_pct = int(round(args.train_ratio * 100))
        val_pct = 100 - train_pct
        path = output_dir / f"{args.prefix}_{train_pct}_{val_pct}_repeat_{repeat + 1}.json"
        path.write_text(json.dumps(split, indent=2), encoding="utf-8")
        print(
            f"[OK] {path}: train_scenes={len(split['train'])}, val_scenes={len(split['val'])}, "
            f"train_pos_scenes={train_pos_count}, val_pos_scenes={len(pos) - train_pos_count}"
        )


if __name__ == "__main__":
    main()
