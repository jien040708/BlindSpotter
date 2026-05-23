#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle


NODE_STYLE = {
    "ego_vehicle": {"color": "#2563eb", "marker": "s", "size": 210, "label": "reference vehicle"},
    "car": {"color": "#64748b", "marker": "o", "size": 90, "label": "vehicle / occluder"},
    "truck": {"color": "#475569", "marker": "o", "size": 110, "label": "truck / occluder"},
    "bus": {"color": "#475569", "marker": "o", "size": 110, "label": "bus / occluder"},
    "vehicle": {"color": "#64748b", "marker": "o", "size": 90, "label": "vehicle / occluder"},
    "pedestrian": {"color": "#16a34a", "marker": "^", "size": 95, "label": "VRU"},
    "cyclist": {"color": "#22c55e", "marker": "^", "size": 95, "label": "VRU"},
    "e_scooter": {"color": "#ef4444", "marker": "^", "size": 120, "label": "scooter / PM"},
    "occlusion_zone": {"color": "#f97316", "marker": "D", "size": 125, "label": "blind-zone node"},
}

EDGE_STYLE = {
    "spatial_near": {"color": "#94a3b8", "alpha": 0.22, "width": 0.8},
    "potential_conflict": {"color": "#f59e0b", "alpha": 0.58, "width": 1.4},
    "occludes": {"color": "#dc2626", "alpha": 0.75, "width": 1.8},
    "blind_zone_relation": {"color": "#ea580c", "alpha": 0.64, "width": 1.4},
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a paper-style GNN training graph sample.")
    parser.add_argument("--graphs", default="outputs/graphs_imptc_set01")
    parser.add_argument("--output", default="outputs/figures/training_graph_sample.png")
    parser.add_argument("--pdf-output", default="outputs/figures/training_graph_sample.pdf")
    parser.add_argument("--prefer-positive", action="store_true", default=True)
    args = parser.parse_args()

    graph, frame = select_frame(Path(args.graphs), prefer_positive=args.prefer_positive)
    output = Path(args.output)
    pdf_output = Path(args.pdf_output)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16, 9.5), facecolor="#f8fafc")
    grid = fig.add_gridspec(2, 3, width_ratios=[1.35, 1.0, 0.78], height_ratios=[0.22, 1.0])
    title_ax = fig.add_subplot(grid[0, :])
    spatial_ax = fig.add_subplot(grid[1, 0])
    graph_ax = fig.add_subplot(grid[1, 1])
    info_ax = fig.add_subplot(grid[1, 2])

    draw_title(title_ax, graph, frame)
    draw_spatial_graph(spatial_ax, frame)
    draw_abstract_graph(graph_ax, frame)
    draw_info_panel(info_ax, graph, frame)

    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(pdf_output, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"[OK] saved {output}")
    print(f"[OK] saved {pdf_output}")


def select_frame(graph_dir: Path, prefer_positive: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    paths = sorted(
        path
        for path in graph_dir.glob("*.json")
        if path.name not in {"preprocess_summary.json", "validation_summary.json"}
    )
    fallback = None
    for path in paths:
        graph = json.loads(path.read_text(encoding="utf-8"))
        for frame in graph.get("frames", []):
            if frame.get("blind_node_indices"):
                fallback = fallback or (graph, frame)
            if prefer_positive and sum(int(v) for v in frame.get("blind_y", [])) > 0:
                return graph, frame
    if fallback is None:
        raise SystemExit(f"No frame with blind-zone nodes found in {graph_dir}")
    return fallback


def draw_title(ax, graph: dict[str, Any], frame: dict[str, Any]) -> None:
    ax.axis("off")
    positives = sum(int(v) for v in frame.get("blind_y", []))
    targets = len(frame.get("blind_y", []))
    ax.text(0.0, 0.72, "Training Graph Sample for Blind-Zone Risk Prediction", fontsize=22, weight="bold", color="#0f172a")
    ax.text(
        0.0,
        0.25,
        f"scene={graph.get('scene_id')}  |  frame={frame.get('frame_id')}  |  blind-zone targets={targets}  |  positive labels={positives}",
        fontsize=11.5,
        color="#475569",
    )


def draw_spatial_graph(ax, frame: dict[str, Any]) -> None:
    ax.set_title("A. Spatial frame graph", loc="left", fontsize=14, weight="bold", color="#0f172a")
    ax.set_facecolor("#ffffff")
    ax.grid(True, color="#e2e8f0", linewidth=0.8)
    x = frame["x"]
    node_types = frame["node_types"]
    edge_index = frame["edge_index"]
    edge_type = frame.get("edge_type", ["spatial_near"] * len(edge_index[0]))

    for src, dst, etype in zip(edge_index[0], edge_index[1], edge_type):
        sx, sy = x[src][0], x[src][1]
        dx, dy = x[dst][0], x[dst][1]
        style = EDGE_STYLE.get(etype, EDGE_STYLE["spatial_near"])
        ax.plot([sx, dx], [sy, dy], color=style["color"], alpha=style["alpha"], linewidth=style["width"], zorder=1)

    target_to_label = {
        int(idx): int(label)
        for idx, label in zip(frame.get("blind_node_indices", []), frame.get("blind_y", []))
    }
    for idx, (features, node_type) in enumerate(zip(x, node_types)):
        style = NODE_STYLE.get(node_type, {"color": "#64748b", "marker": "o", "size": 80})
        edge_color = "#991b1b" if target_to_label.get(idx, 0) == 1 else "#ffffff"
        linewidth = 2.8 if idx in target_to_label else 0.9
        ax.scatter(
            [features[0]],
            [features[1]],
            s=style["size"],
            marker=style["marker"],
            c=style["color"],
            edgecolors=edge_color,
            linewidths=linewidth,
            zorder=3,
        )
        if idx in target_to_label:
            ax.text(features[0] + 0.55, features[1] + 0.55, f"y={target_to_label[idx]}", fontsize=9, weight="bold", color="#991b1b")
        elif idx == 0:
            ax.text(features[0] + 0.55, features[1] + 0.55, "ref", fontsize=9, weight="bold", color="#1d4ed8")

    ax.set_xlabel("x position [m]", color="#334155")
    ax.set_ylabel("y position [m]", color="#334155")
    ax.set_aspect("equal", adjustable="datalim")
    add_node_legend(ax)


def draw_abstract_graph(ax, frame: dict[str, Any]) -> None:
    ax.set_title("B. GNN view: nodes, relations, target", loc="left", fontsize=14, weight="bold", color="#0f172a")
    ax.set_facecolor("#ffffff")
    ax.axis("off")
    x = frame["x"]
    node_types = frame["node_types"]
    edge_index = frame["edge_index"]
    edge_type = frame.get("edge_type", ["spatial_near"] * len(edge_index[0]))

    pos = normalized_positions(x)
    for src, dst, etype in zip(edge_index[0], edge_index[1], edge_type):
        if etype == "spatial_near" and (src + dst) % 3 != 0:
            continue
        style = EDGE_STYLE.get(etype, EDGE_STYLE["spatial_near"])
        patch = FancyArrowPatch(
            pos[src],
            pos[dst],
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=style["width"],
            color=style["color"],
            alpha=min(style["alpha"] + 0.12, 0.85),
            shrinkA=10,
            shrinkB=10,
            zorder=1,
        )
        ax.add_patch(patch)

    target_to_label = {
        int(idx): int(label)
        for idx, label in zip(frame.get("blind_node_indices", []), frame.get("blind_y", []))
    }
    for idx, node_type in enumerate(node_types):
        px, py = pos[idx]
        style = NODE_STYLE.get(node_type, {"color": "#64748b", "marker": "o", "size": 80})
        ring = "#991b1b" if target_to_label.get(idx, 0) == 1 else "#ffffff"
        ax.scatter([px], [py], s=style["size"] * 0.9, marker=style["marker"], c=style["color"], edgecolors=ring, linewidths=2.4, zorder=3)
        ax.text(px, py - 0.055, str(idx), ha="center", va="top", fontsize=8, color="#0f172a", zorder=4)

    ax.text(0.03, 0.04, "Target: occlusion_zone node embedding → blind_y", fontsize=10.5, color="#334155", transform=ax.transAxes)
    add_edge_legend(ax)


def draw_info_panel(ax, graph: dict[str, Any], frame: dict[str, Any]) -> None:
    ax.set_title("C. Tensor contract", loc="left", fontsize=14, weight="bold", color="#0f172a")
    ax.set_facecolor("#ffffff")
    ax.axis("off")

    node_features = graph.get("node_feature_names", [])
    edge_features = graph.get("edge_feature_names", [])
    node_count = len(frame.get("node_ids", []))
    edge_count = len(frame.get("edge_index", [[], []])[0])
    target_count = len(frame.get("blind_node_indices", []))
    positives = sum(int(v) for v in frame.get("blind_y", []))

    lines = [
        ("Node matrix X", f"{node_count} nodes × {len(node_features)} features"),
        ("Edge index", f"2 × {edge_count} directed edges"),
        ("Edge attributes", f"{edge_count} edges × {len(edge_features)} expert features"),
        ("Targets", f"{target_count} blind-zone nodes"),
        ("Positive labels", f"{positives} emergence risks"),
    ]
    y = 0.93
    for title, value in lines:
        ax.text(0.05, y, title, fontsize=10.8, weight="bold", color="#0f172a", transform=ax.transAxes)
        ax.text(0.05, y - 0.05, value, fontsize=10, color="#475569", transform=ax.transAxes)
        y -= 0.14

    draw_feature_block(ax, 0.05, 0.22, "Expert edge features", edge_features[:6])
    draw_feature_block(ax, 0.05, 0.02, "Core node features", node_features[:8])


def draw_feature_block(ax, x: float, y: float, title: str, features: list[str]) -> None:
    ax.add_patch(Rectangle((x, y), 0.9, 0.16, transform=ax.transAxes, facecolor="#f1f5f9", edgecolor="#cbd5e1", linewidth=1))
    ax.text(x + 0.02, y + 0.115, title, fontsize=9.5, weight="bold", color="#334155", transform=ax.transAxes)
    ax.text(x + 0.02, y + 0.045, ", ".join(features), fontsize=8.4, color="#475569", transform=ax.transAxes, wrap=True)


def normalized_positions(x: list[list[float]]) -> dict[int, tuple[float, float]]:
    xs = [row[0] for row in x]
    ys = [row[1] for row in x]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(max_x - min_x, 1e-6)
    span_y = max(max_y - min_y, 1e-6)
    return {
        idx: (0.08 + 0.84 * ((row[0] - min_x) / span_x), 0.13 + 0.74 * ((row[1] - min_y) / span_y))
        for idx, row in enumerate(x)
    }


def add_node_legend(ax) -> None:
    handles = []
    seen = set()
    for key in ["ego_vehicle", "car", "e_scooter", "pedestrian", "cyclist", "occlusion_zone"]:
        style = NODE_STYLE[key]
        label = style["label"]
        if label in seen:
            continue
        seen.add(label)
        handles.append(
            Line2D([0], [0], marker=style["marker"], color="w", label=label, markerfacecolor=style["color"], markersize=8)
        )
    ax.legend(handles=handles, loc="upper right", frameon=True, framealpha=0.95, fontsize=8.5)


def add_edge_legend(ax) -> None:
    handles = []
    labels = {
        "spatial_near": "spatial near",
        "potential_conflict": "potential conflict",
        "occludes": "occludes",
        "blind_zone_relation": "blind-zone relation",
    }
    for key, label in labels.items():
        style = EDGE_STYLE[key]
        handles.append(Line2D([0], [0], color=style["color"], linewidth=2, label=label))
    ax.legend(handles=handles, loc="upper right", frameon=True, framealpha=0.95, fontsize=8)


if __name__ == "__main__":
    main()
