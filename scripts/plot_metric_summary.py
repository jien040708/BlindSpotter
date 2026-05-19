#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = Path("outputs/figures")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_metric_rows()
    plot_grouped_bars(rows)
    plot_repeated_split_summary()


def load_metric_rows() -> list[dict[str, float | str]]:
    specs = [
        (
            "Generated\naligned 8f",
            "outputs/models/single_frame_gat_aligned_1layer_3ep.metrics.json",
            "test_metrics",
        ),
        (
            "Generated\ncanonical EIGAT",
            "outputs/models/single_frame_gat_canonical_stable_1layer_5ep.metrics.json",
            "test_metrics",
        ),
        (
            "Generated\ncanonical no-edge",
            "outputs/models/single_frame_gat_canonical_no_edge_1layer_5ep.metrics.json",
            "test_metrics",
        ),
        (
            "IMPTC set01\nfull train",
            "outputs/models/single_frame_gat_imptc_set01_1layer_10ep.metrics.json",
            "test_metrics",
        ),
        (
            "IMPTC set01\nneg sample",
            "outputs/models/single_frame_gat_imptc_set01_balanced_20ep.metrics.json",
            "test_metrics",
        ),
        (
            "Temporal\nhorizon +1",
            "outputs/models/temporal_gat_h1.metrics.json",
            "test_metrics",
        ),
    ]
    rows = []
    for label, path, key in specs:
        metric_path = Path(path)
        if not metric_path.exists():
            continue
        data = json.loads(metric_path.read_text(encoding="utf-8"))
        metrics = data.get(key, {})
        rows.append(
            {
                "label": label,
                "auprc": float(metrics.get("auprc", 0.0)),
                "auroc": float(metrics.get("auroc", 0.0)),
                "f1": float(metrics.get("f1", 0.0)),
                "best_f1": float(metrics.get("best_f1", 0.0)),
            }
        )
    return rows


def plot_grouped_bars(rows: list[dict[str, float | str]]) -> None:
    labels = [str(row["label"]) for row in rows]
    metrics = ["auprc", "auroc", "f1", "best_f1"]
    metric_labels = ["AUPRC", "AUROC", "F1@0.5", "Best F1"]
    colors = ["#2563eb", "#059669", "#dc2626", "#7c3aed"]

    x = np.arange(len(labels))
    width = 0.18

    fig, ax = plt.subplots(figsize=(16, 8.5))
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#f8fafc")

    for idx, (metric, metric_label, color) in enumerate(zip(metrics, metric_labels, colors)):
        values = [float(row[metric]) for row in rows]
        offset = (idx - 1.5) * width
        bars = ax.bar(x + offset, values, width, label=metric_label, color=color, alpha=0.88)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.012,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color="#334155",
            )

    ax.set_title("Blind-Zone Risk Model Metrics", fontsize=20, weight="bold", color="#0f172a", pad=18)
    ax.set_ylabel("Score", fontsize=12, color="#334155")
    ax.set_ylim(0, 0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.grid(axis="y", color="#cbd5e1", linewidth=0.8, alpha=0.7)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#94a3b8")
    ax.legend(ncols=4, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)
    ax.text(
        0,
        -0.23,
        "Note: F1@0.5 uses a fixed threshold; Best F1 is threshold-tuned on the evaluated split. Rare IMPTC events need calibration.",
        transform=ax.transAxes,
        fontsize=10,
        color="#64748b",
    )

    png_path = OUTPUT_DIR / "model_metric_summary.png"
    pdf_path = OUTPUT_DIR / "model_metric_summary.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(pdf_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"[OK] saved {png_path}")
    print(f"[OK] saved {pdf_path}")


def plot_repeated_split_summary() -> None:
    path = Path("outputs/models/repeated_single_frame_set01_summary.json")
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data["rows"]
    repeats = [f"R{row['repeat']}" for row in rows]
    metrics = ["auprc", "auroc", "best_f1"]
    metric_labels = ["AUPRC", "AUROC", "Best F1"]
    colors = ["#2563eb", "#059669", "#7c3aed"]

    x = np.arange(len(repeats))
    width = 0.23
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#f8fafc")

    for idx, (metric, metric_label, color) in enumerate(zip(metrics, metric_labels, colors)):
        values = [float(row[metric]) for row in rows]
        bars = ax.bar(x + (idx - 1) * width, values, width, label=metric_label, color=color, alpha=0.88)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.012,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color="#334155",
            )

    ax.set_title("Repeated Scene-Level 80/20 Validation", fontsize=18, weight="bold", color="#0f172a", pad=14)
    ax.set_ylabel("Score", fontsize=12, color="#334155")
    ax.set_ylim(0, 0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(repeats, fontsize=11)
    ax.grid(axis="y", color="#cbd5e1", linewidth=0.8, alpha=0.7)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#94a3b8")
    ax.legend(ncols=3, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)

    png_path = OUTPUT_DIR / "repeated_split_metric_summary.png"
    pdf_path = OUTPUT_DIR / "repeated_split_metric_summary.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(pdf_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"[OK] saved {png_path}")
    print(f"[OK] saved {pdf_path}")


if __name__ == "__main__":
    main()
