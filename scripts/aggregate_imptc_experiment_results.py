#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METRICS = ("auprc", "auroc", "f1", "best_f1", "precision", "recall")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate IMPTC experiment metric JSON files into CSV/Markdown.")
    parser.add_argument("metrics", nargs="+", help="Metric JSON paths")
    parser.add_argument("--output-csv", default="outputs/results/imptc_set01_set02_results.csv")
    parser.add_argument("--output-md", default="outputs/results/imptc_set01_set02_results.md")
    args = parser.parse_args()

    rows = []
    for metric_path in args.metrics:
        path = Path(metric_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        metric_block = payload.get("test_metrics") or payload.get("eval_metrics") or {}
        row = {
            "run": path.stem.replace(".metrics", ""),
            "path": str(path),
            "selection_metric": payload.get("selection_metric", ""),
            "best_val_score": payload.get("best_val_score", ""),
        }
        for metric in METRICS:
            row[metric] = metric_block.get(metric, "")
        rows.append(row)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["run", "selection_metric", "best_val_score", *METRICS, "path"]
    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| run | AUPRC | AUROC | F1@0.5 | Best F1 | Precision | Recall |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {run} | {auprc} | {auroc} | {f1} | {best_f1} | {precision} | {recall} |".format(
                run=row["run"],
                auprc=format_metric(row["auprc"]),
                auroc=format_metric(row["auroc"]),
                f1=format_metric(row["f1"]),
                best_f1=format_metric(row["best_f1"]),
                precision=format_metric(row["precision"]),
                recall=format_metric(row["recall"]),
            )
        )
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] wrote {output_csv}")
    print(f"[OK] wrote {output_md}")


def format_metric(value: object) -> str:
    if value == "":
        return ""
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


if __name__ == "__main__":
    main()
