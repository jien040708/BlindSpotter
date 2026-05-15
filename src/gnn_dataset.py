from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass
class FrameGraphSample:
    scene_id: str
    frame_id: str
    timestamp: float
    node_ids: list[str]
    node_types: list[str]
    x: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    target_indices: torch.Tensor
    y: torch.Tensor


@dataclass
class TemporalGraphSample:
    scene_id: str
    frame_id: str
    timestamp: float
    frames: list[FrameGraphSample]
    target_node_ids: list[str]
    y: torch.Tensor


def load_graph_jsons(graph_dir: str | Path) -> list[dict[str, Any]]:
    graph_dir = Path(graph_dir)
    paths = sorted(
        path
        for path in graph_dir.glob("*.json")
        if path.name not in {"preprocess_summary.json", "validation_summary.json"}
    )
    graphs = []
    for path in paths:
        graphs.append(json.loads(path.read_text(encoding="utf-8")))
    return graphs


def load_frame_samples(graph_dir: str | Path) -> tuple[list[FrameGraphSample], dict[str, list[str]]]:
    graphs = load_graph_jsons(graph_dir)
    samples: list[FrameGraphSample] = []
    metadata = {"node_feature_names": [], "edge_feature_names": []}
    for graph in graphs:
        metadata["node_feature_names"] = graph.get("node_feature_names", metadata["node_feature_names"])
        metadata["edge_feature_names"] = graph.get("edge_feature_names", metadata["edge_feature_names"])
        scene_id = str(graph.get("scene_id", "unknown"))
        for frame in graph.get("frames", []):
            blind_indices = frame.get("blind_node_indices", [])
            blind_y = frame.get("blind_y", [])
            if not blind_indices:
                continue
            samples.append(frame_to_sample(scene_id, frame, blind_indices, blind_y))
    return samples, metadata


def load_temporal_samples(graph_dir: str | Path, history: int = 5) -> tuple[list[TemporalGraphSample], dict[str, list[str]]]:
    graphs = load_graph_jsons(graph_dir)
    temporal_samples: list[TemporalGraphSample] = []
    metadata = {"node_feature_names": [], "edge_feature_names": []}
    for graph in graphs:
        metadata["node_feature_names"] = graph.get("node_feature_names", metadata["node_feature_names"])
        metadata["edge_feature_names"] = graph.get("edge_feature_names", metadata["edge_feature_names"])
        scene_id = str(graph.get("scene_id", "unknown"))
        frame_samples = []
        for frame in graph.get("frames", []):
            frame_samples.append(
                frame_to_sample(scene_id, frame, frame.get("blind_node_indices", []), frame.get("blind_y", []))
            )
        for idx, current in enumerate(frame_samples):
            if current.target_indices.numel() == 0:
                continue
            start = max(0, idx - history + 1)
            window = frame_samples[start : idx + 1]
            target_node_ids = [current.node_ids[int(i)] for i in current.target_indices.tolist()]
            temporal_samples.append(
                TemporalGraphSample(
                    scene_id=current.scene_id,
                    frame_id=current.frame_id,
                    timestamp=current.timestamp,
                    frames=window,
                    target_node_ids=target_node_ids,
                    y=current.y,
                )
            )
    return temporal_samples, metadata


def frame_to_sample(scene_id: str, frame: dict[str, Any], target_indices: list[int], y: list[int]) -> FrameGraphSample:
    edge_index = frame.get("edge_index", [[], []])
    edge_attr = frame.get("edge_attr", [])
    if not edge_attr:
        edge_attr = []
    return FrameGraphSample(
        scene_id=scene_id,
        frame_id=str(frame.get("frame_id", "")),
        timestamp=float(frame.get("timestamp", 0.0)),
        node_ids=[str(v) for v in frame.get("node_ids", [])],
        node_types=[str(v) for v in frame.get("node_types", [])],
        x=torch.tensor(frame.get("x", []), dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
        target_indices=torch.tensor(target_indices, dtype=torch.long),
        y=torch.tensor(y, dtype=torch.float32),
    )


def split_samples(samples: list[Any], val_ratio: float = 0.2, seed: int = 7) -> tuple[list[Any], list[Any]]:
    rng = random.Random(seed)
    indices = list(range(len(samples)))
    rng.shuffle(indices)
    val_size = max(1, int(len(indices) * val_ratio)) if len(indices) > 1 else 0
    val_indices = set(indices[:val_size])
    train = [sample for idx, sample in enumerate(samples) if idx not in val_indices]
    val = [sample for idx, sample in enumerate(samples) if idx in val_indices]
    return train, val


def estimate_pos_weight(samples: list[Any]) -> torch.Tensor:
    positives = 0.0
    total = 0.0
    for sample in samples:
        y = sample.y
        positives += float(y.sum().item())
        total += float(y.numel())
    negatives = max(total - positives, 1.0)
    positives = max(positives, 1.0)
    return torch.tensor([negatives / positives], dtype=torch.float32)


def move_frame_sample(sample: FrameGraphSample, device: torch.device | str) -> FrameGraphSample:
    return FrameGraphSample(
        scene_id=sample.scene_id,
        frame_id=sample.frame_id,
        timestamp=sample.timestamp,
        node_ids=sample.node_ids,
        node_types=sample.node_types,
        x=sample.x.to(device),
        edge_index=sample.edge_index.to(device),
        edge_attr=sample.edge_attr.to(device),
        target_indices=sample.target_indices.to(device),
        y=sample.y.to(device),
    )


def move_temporal_sample(sample: TemporalGraphSample, device: torch.device | str) -> TemporalGraphSample:
    return TemporalGraphSample(
        scene_id=sample.scene_id,
        frame_id=sample.frame_id,
        timestamp=sample.timestamp,
        frames=[move_frame_sample(frame, device) for frame in sample.frames],
        target_node_ids=sample.target_node_ids,
        y=sample.y.to(device),
    )
