#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUTPUT_DIR = Path("outputs/figures")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 10)
    ax.axis("off")
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#f8fafc")

    draw_title(ax)
    draw_lane_labels(ax)

    nodes = {
        "pkl": box(ax, 0.8, 7.2, 3.3, 1.25, "Generated graph_dataset.pkl", "virtual scenes\ntrain/val/test split\n8 node / 5 edge features", "#dbeafe", "#1d4ed8"),
        "canonical": box(ax, 5.1, 7.2, 3.4, 1.25, "Canonicalize pkl", "convert to IMPTC contract\n14 node / 6 edge features\nTTC added", "#e0f2fe", "#0369a1"),
        "pkl_train": box(ax, 9.5, 7.2, 3.2, 1.25, "Single-frame GAT input", "FrameGraphSample\nblind-zone node target\nblind_y label", "#dcfce7", "#15803d"),
        "imptc_raw": box(ax, 0.8, 4.35, 3.3, 1.25, "IMPTC sequences", "official Zenodo chunks\nset_01: 50 sequences\nreal vehicle/VRU tracks", "#fef3c7", "#b45309"),
        "preprocess": box(ax, 5.1, 4.35, 3.4, 1.25, "Spatial graph preprocessing", "track.json -> frames\nreference vehicle\noccluder blind-zone nodes", "#ffedd5", "#c2410c"),
        "scene_split": box(ax, 9.5, 4.35, 3.2, 1.25, "Scene-level splits", "80/20 repeated splits\npositive-scene stratified\nno frame leakage", "#fae8ff", "#a21caf"),
        "align": box(ax, 13.7, 5.8, 3.3, 1.25, "Alignment + normalization", "feature aliases\ntrain-fitted scaling\nbinary features preserved", "#ede9fe", "#6d28d9"),
        "metrics": box(ax, 13.7, 3.35, 3.3, 1.25, "Evaluation outputs", "AUPRC / AUROC / F1\nbest threshold\nresearch_results.md", "#e2e8f0", "#334155"),
        "temporal": box(ax, 9.5, 1.45, 3.2, 1.25, "Temporal GAT path", "history frames\nprediction horizon\nGRU aggregation", "#fee2e2", "#b91c1c"),
    }

    arrow(ax, nodes["pkl"], nodes["canonical"], "#2563eb")
    arrow(ax, nodes["canonical"], nodes["pkl_train"], "#2563eb")
    arrow(ax, nodes["imptc_raw"], nodes["preprocess"], "#ea580c")
    arrow(ax, nodes["preprocess"], nodes["scene_split"], "#ea580c")
    arrow(ax, nodes["scene_split"], nodes["align"], "#9333ea")
    arrow(ax, nodes["pkl_train"], nodes["align"], "#16a34a")
    arrow(ax, nodes["align"], nodes["metrics"], "#475569")
    arrow(ax, nodes["preprocess"], nodes["temporal"], "#dc2626", rad=-0.18)
    arrow(ax, nodes["temporal"], nodes["metrics"], "#dc2626", rad=-0.12)

    add_metric_callout(ax)
    add_footer(ax)

    png_path = OUTPUT_DIR / "dataset_pipeline.png"
    pdf_path = OUTPUT_DIR / "dataset_pipeline.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(pdf_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"[OK] saved {png_path}")
    print(f"[OK] saved {pdf_path}")


def draw_title(ax) -> None:
    ax.text(0.8, 9.55, "Blind-Zone Risk Dataset Pipeline", fontsize=24, weight="bold", color="#0f172a")
    ax.text(
        0.8,
        9.16,
        "How generated graphs and real IMPTC sequences are aligned, split, and evaluated for EIGAT training",
        fontsize=12.5,
        color="#475569",
    )


def draw_lane_labels(ax) -> None:
    ax.text(0.35, 8.05, "Generated", fontsize=11, color="#1d4ed8", weight="bold", rotation=90, va="center")
    ax.text(0.35, 4.95, "Real IMPTC", fontsize=11, color="#b45309", weight="bold", rotation=90, va="center")
    ax.text(0.35, 1.95, "Temporal", fontsize=11, color="#b91c1c", weight="bold", rotation=90, va="center")


def box(ax, x, y, w, h, title, body, fill, edge):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.03,rounding_size=0.08",
        linewidth=1.8,
        edgecolor=edge,
        facecolor=fill,
    )
    ax.add_patch(patch)
    ax.text(x + 0.18, y + h - 0.32, title, fontsize=12.2, weight="bold", color="#0f172a", va="top")
    ax.text(x + 0.18, y + h - 0.68, body, fontsize=9.6, color="#334155", va="top", linespacing=1.25)
    return (x, y, w, h)


def arrow(ax, src, dst, color, rad=0.0):
    x1, y1, w1, h1 = src
    x2, y2, w2, h2 = dst
    start = (x1 + w1, y1 + h1 / 2)
    end = (x2, y2 + h2 / 2)
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=16,
        linewidth=2.0,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=8,
        shrinkB=8,
    )
    ax.add_patch(patch)


def add_metric_callout(ax) -> None:
    x, y, w, h = 13.7, 7.75, 3.3, 0.78
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.03,rounding_size=0.08",
        linewidth=1.4,
        edgecolor="#0f766e",
        facecolor="#ccfbf1",
    )
    ax.add_patch(patch)
    ax.text(x + 0.16, y + h - 0.22, "Main scoring", fontsize=11.3, weight="bold", color="#134e4a", va="top")
    ax.text(x + 0.16, y + h - 0.52, "AUPRC · AUROC · F1-score", fontsize=10, color="#134e4a", va="top")


def add_footer(ax) -> None:
    ax.text(
        0.8,
        0.58,
        "Key safety choice: repeated splits are scene-level, not frame-random, to avoid leakage from adjacent frames.",
        fontsize=11,
        color="#475569",
    )


if __name__ == "__main__":
    main()
