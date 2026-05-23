#!/usr/bin/env python3
"""Compatibility wrapper for the spatio-temporal GNN experiment."""

from __future__ import annotations

import sys

from train_temporal_gat import main


if __name__ == "__main__":
    if "--temporal-model" not in sys.argv:
        sys.argv.extend(["--temporal-model", "social_stgcn"])
    main()
