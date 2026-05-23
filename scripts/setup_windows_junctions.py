#!/usr/bin/env python3
"""
Windows-compatible replacement for the ln -s step in run_imptc_sets_experiments.sh.
Creates directory junctions (mklink /J) which don't require admin or Developer Mode.

Usage:
  python scripts/setup_windows_junctions.py --sets 01 02 03 04 05 \
      --source-root data --selected-root data/imptc_selected/set01_set02_set03_set04_set05
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def make_junction(source: Path, target: Path) -> bool:
    """Create a Windows directory junction. Returns True on success."""
    if target.exists() or target.is_symlink():
        return True
    try:
        result = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(target), str(source)],
            capture_output=True, text=True
        )
        return result.returncode == 0
    except Exception as e:
        print(f"  [WARN] mklink failed for {target.name}: {e}", file=sys.stderr)
        return False


def link_sequence_dir(seq_dir: Path, selected_root: Path) -> None:
    """Create a junction for one sequence directory."""
    if not (seq_dir / "vehicles").is_dir() or not (seq_dir / "vrus").is_dir():
        return
    target = selected_root / seq_dir.name
    if target.exists():
        return
    abs_source = seq_dir.resolve()
    ok = make_junction(abs_source, target)
    if ok:
        print(f"  [J] {seq_dir.name}")
    else:
        print(f"  [COPY] {seq_dir.name} (junction failed, copying...)")
        import shutil
        shutil.copytree(str(abs_source), str(target))


def main() -> None:
    parser = argparse.ArgumentParser(description="Set up imptc_selected via Windows junctions.")
    parser.add_argument("--sets", nargs="+", default=["01", "02", "03", "04", "05"],
                        help="Set IDs to include, e.g. 01 02 03")
    parser.add_argument("--source-root", default="data",
                        help="Root containing imptc_set_XX directories")
    parser.add_argument("--selected-root", default=None,
                        help="Target directory (auto-derived from sets if not given)")
    args = parser.parse_args()

    set_ids = [f"{int(s):02d}" for s in args.sets]
    source_root = (PROJECT_ROOT / args.source_root).resolve()

    if args.selected_root is None:
        tag = "set" + "_set".join(set_ids)
        selected_root = PROJECT_ROOT / "data" / "imptc_selected" / tag
    else:
        selected_root = (PROJECT_ROOT / args.selected_root).resolve()

    selected_root.mkdir(parents=True, exist_ok=True)
    print(f"Source root  : {source_root}")
    print(f"Selected root: {selected_root}")
    print(f"Sets         : {set_ids}")

    total = 0
    for set_id in set_ids:
        set_dir = source_root / f"imptc_set_{set_id}"
        if not set_dir.is_dir():
            print(f"[WARN] {set_dir} not found, skipping")
            continue

        print(f"\n[set_{set_id}] scanning {set_dir} ...")
        # Check if set_dir itself has vehicles/vrus (unlikely but handle it)
        if (set_dir / "vehicles").is_dir() and (set_dir / "vrus").is_dir():
            link_sequence_dir(set_dir, selected_root)
            total += 1
        else:
            # Each subdirectory is a sequence
            for seq_dir in sorted(set_dir.iterdir()):
                if seq_dir.is_dir():
                    link_sequence_dir(seq_dir, selected_root)
                    total += 1

    count = len(list(selected_root.iterdir()))
    print(f"\n[OK] {count} sequences in {selected_root}")
    print(f"     SELECTED_ROOT={selected_root}")


if __name__ == "__main__":
    main()
