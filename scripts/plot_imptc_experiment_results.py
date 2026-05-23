#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


DISPLAY_METRICS = [
    ("auprc", "AUPRC"),
    ("auroc", "AUROC"),
    ("f1", "F1@0.5"),
    ("best_f1", "Best F1"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot IMPTC model comparison metrics and validation curves.")
    parser.add_argument("metrics", nargs="+", help="Metric JSON files")
    parser.add_argument("--output-dir", default="outputs/figures/imptc_set01_set02")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = [load_run(Path(path)) for path in args.metrics]
    plot_metric_bars(runs, output_dir / "main_metric_comparison.png")
    plot_validation_curves(runs, output_dir / "validation_curves.png")
    print(f"[OK] wrote {output_dir / 'main_metric_comparison.png'}")
    print(f"[OK] wrote {output_dir / 'validation_curves.png'}")


def load_run(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload.get("test_metrics") or payload.get("eval_metrics") or {}
    label = path.stem.replace(".metrics", "")
    return {"label": label, "metrics": metrics, "history": payload.get("history", [])}


def plot_metric_bars(runs: list[dict], output: Path) -> None:
    labels = [run["label"] for run in runs]
    x = range(len(labels))
    width = 0.18
    colors = ["#2563eb", "#16a34a", "#dc2626", "#7c3aed"]

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 2.2), 5.2))
    for metric_idx, (metric_key, metric_label) in enumerate(DISPLAY_METRICS):
        values = [float(run["metrics"].get(metric_key, 0.0) or 0.0) for run in runs]
        offsets = [idx + (metric_idx - 1.5) * width for idx in x]
        bars = ax.bar(offsets, values, width=width, label=metric_label, color=colors[metric_idx], alpha=0.88)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_title("IMPTC Set01+Set02 Test Metrics")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_validation_curves(runs: list[dict], output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=False)
    curve_specs = [("auprc", "Validation AUPRC"), ("auroc", "Validation AUROC"), ("best_f1", "Validation Best F1")]
    for ax, (metric_key, title) in zip(axes, curve_specs):
        for run in runs:
            history = run["history"]
            epochs = [row["epoch"] for row in history]
            values = [float(row["val_metrics"].get(metric_key, 0.0) or 0.0) for row in history]
            if epochs:
                ax.plot(epochs, values, marker="o", linewidth=1.8, label=run["label"])
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Score")
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
    fig.suptitle("Validation Curves During Training", y=1.03)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
