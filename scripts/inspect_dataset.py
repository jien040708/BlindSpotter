#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import FlexibleSceneDataset
from src.imptc_dataset import is_imptc_root
from src.utils import ANNOTATION_EXTENSIONS, IMAGE_EXTENSIONS, LIDAR_EXTENSIONS, load_structured_file, summarize_keys


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a sample dataset folder without assuming a fixed schema.")
    parser.add_argument("--root", default="data/sample", help="Dataset root directory")
    parser.add_argument("--max-files", type=int, default=5, help="Maximum structured files to attempt loading")
    args = parser.parse_args()

    root = Path(args.root)
    print(f"Dataset root: {root.resolve()}")
    if not root.exists():
        print("[WARN] Dataset root does not exist yet. Put sample data under this folder and rerun.")
        return

    dirs = [p for p in root.rglob("*") if p.is_dir()]
    files = [p for p in root.rglob("*") if p.is_file()]
    suffix_counts = Counter(p.suffix.lower() or "<no_ext>" for p in files)
    annotation_files = [p for p in files if p.suffix.lower() in ANNOTATION_EXTENSIONS]
    image_files = [p for p in files if p.suffix.lower() in IMAGE_EXTENSIONS]
    lidar_files = [p for p in files if p.suffix.lower() in LIDAR_EXTENSIONS]

    print(f"Directories: {len(dirs)}")
    print(f"Files: {len(files)}")
    print(f"Annotation-like files: {len(annotation_files)}")
    print(f"Image files: {len(image_files)}")
    print(f"LiDAR-like files: {len(lidar_files)}")
    print("\nFile extensions:")
    for suffix, count in suffix_counts.most_common():
        print(f"- {suffix}: {count}")

    print("\nTop-level folders:")
    for path in sorted([p for p in root.iterdir() if p.is_dir()])[:30]:
        print(f"- {path}")

    print("\nSample files:")
    for path in sorted(files)[:30]:
        print(f"- {path}")

    if is_imptc_root(root):
        print("\nIMPTC sequence summary:")
        for sequence_dir in sorted(child for child in root.iterdir() if child.is_dir())[:20]:
            vehicles = list((sequence_dir / "vehicles").glob("*/track.json"))
            vrus = list((sequence_dir / "vrus").glob("*/track.json"))
            context = list((sequence_dir / "context").glob("*.json"))
            if vehicles or vrus:
                print(
                    f"- {sequence_dir.name}: vehicles={len(vehicles)}, vrus={len(vrus)}, "
                    f"context_files={len(context)}"
                )

    imptc_detected = is_imptc_root(root)

    if annotation_files:
        sample = annotation_files[0]
        print(f"\nExample annotation: {sample}")
        data = load_structured_file(sample)
        if data is not None:
            print("Example annotation keys:")
            for line in summarize_keys(data):
                print(line)

    if imptc_detected:
        print("\nParsed scenes: use scripts/preprocess_sample.py for IMPTC frame assembly.")
    else:
        dataset = FlexibleSceneDataset(root, max_files=args.max_files)
        scenes = dataset.load_scenes()
        print(f"\nParsed scenes: {len(scenes)}")
        for scene in scenes[:5]:
            object_count = sum(len(frame.objects) for frame in scene.frames)
            print(f"- scene_id={scene.scene_id}, frames={len(scene.frames)}, objects={object_count}, source={scene.source_path}")
            if scene.frames:
                first = scene.frames[0]
                print(f"  first_frame={first.frame_id}, timestamp={first.timestamp}, objects={len(first.objects)}")


if __name__ == "__main__":
    main()
