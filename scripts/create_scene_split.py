#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def ensure_minimum_nonempty_splits(split: dict[str, list[str]]) -> None:
    """Keep tiny sample runs trainable while preserving scene-level separation."""
    if not any(split.values()):
        return

    def move_one(source_keys: list[str], target_key: str) -> bool:
        for source_key in source_keys:
            if len(split[source_key]) > 1:
                split[target_key].append(split[source_key].pop())
                return True
        return False

    if not split["train"]:
        move_one(["val", "test"], "train")
    if not split["val"]:
        move_one(["test", "train"], "val")
    if not split["test"]:
        move_one(["val", "train"], "test")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a stratified scene-level train/val/test split.")
    parser.add_argument("--summary", required=True, help="preprocess_summary.json from graph preprocessing")
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise SystemExit("train/val/test ratios must sum to 1.0")

    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    positive_scenes = [row["scene_id"] for row in summary if int(row["positive_blind_labels"]) > 0]
    negative_scenes = [row["scene_id"] for row in summary if int(row["positive_blind_labels"]) == 0]

    rng = random.Random(args.seed)
    rng.shuffle(positive_scenes)
    rng.shuffle(negative_scenes)
    split = {"train": [], "val": [], "test": []}
    for scenes in (positive_scenes, negative_scenes):
        train_count = max(1, round(len(scenes) * args.train_ratio)) if len(scenes) >= 3 else max(0, len(scenes) - 2)
        val_count = max(1, round(len(scenes) * args.val_ratio)) if len(scenes) - train_count >= 2 else max(0, len(scenes) - train_count - 1)
        split["train"].extend(scenes[:train_count])
        split["val"].extend(scenes[train_count : train_count + val_count])
        split["test"].extend(scenes[train_count + val_count :])

    ensure_minimum_nonempty_splits(split)

    split = {key: sorted(value) for key, value in split.items()}
    split["metadata"] = {
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "positive_scene_count": len(positive_scenes),
        "negative_scene_count": len(negative_scenes),
        "split_unit": "scene",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(split, indent=2), encoding="utf-8")
    print(
        f"[OK] {output}: train={len(split['train'])}, val={len(split['val'])}, test={len(split['test'])}, "
        f"positive_scenes={len(positive_scenes)}, negative_scenes={len(negative_scenes)}"
    )


if __name__ == "__main__":
    main()
