#!/usr/bin/env python3
"""Compatibility wrapper for the MR-GCN single-frame experiment."""

from __future__ import annotations

import sys

from train_single_frame_gat import main


if __name__ == "__main__":
    if "--model" not in sys.argv:
        sys.argv.extend(["--model", "mrgcn"])
    main()
