from __future__ import annotations

import json
import pickle
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
    edge_types: list[str] | None = None


@dataclass
class TemporalGraphSample:
    scene_id: str
    frame_id: str
    timestamp: float
    frames: list[FrameGraphSample]
    target_node_ids: list[str]
    y: torch.Tensor


@dataclass
class FeatureNormalizer:
    node_mean: torch.Tensor
    node_std: torch.Tensor
    edge_mean: torch.Tensor
    edge_std: torch.Tensor
    node_feature_names: list[str]
    edge_feature_names: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_mean": self.node_mean.tolist(),
            "node_std": self.node_std.tolist(),
            "edge_mean": self.edge_mean.tolist(),
            "edge_std": self.edge_std.tolist(),
            "node_feature_names": self.node_feature_names,
            "edge_feature_names": self.edge_feature_names,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FeatureNormalizer":
        return cls(
            node_mean=torch.tensor(data["node_mean"], dtype=torch.float32),
            node_std=torch.tensor(data["node_std"], dtype=torch.float32),
            edge_mean=torch.tensor(data["edge_mean"], dtype=torch.float32),
            edge_std=torch.tensor(data["edge_std"], dtype=torch.float32),
            node_feature_names=list(data.get("node_feature_names", [])),
            edge_feature_names=list(data.get("edge_feature_names", [])),
        )


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


def load_frame_sample_splits(data_path: str | Path) -> tuple[dict[str, list[FrameGraphSample]], dict[str, list[str]]]:
    data_path = Path(data_path)
    if data_path.is_file() and data_path.suffix == ".pkl":
        return load_frame_sample_splits_from_pickle(data_path)

    samples, metadata = load_frame_samples(data_path)
    train_samples, val_samples = split_samples(samples)
    return {"train": train_samples, "val": val_samples, "test": []}, metadata


def load_frame_sample_splits_from_pickle(pkl_path: str | Path) -> tuple[dict[str, list[FrameGraphSample]], dict[str, list[str]]]:
    with Path(pkl_path).open("rb") as file:
        payload = pickle.load(file)

    raw_splits = payload.get("dataset", payload)
    if not isinstance(raw_splits, dict):
        raise ValueError(f"Expected graph dataset pickle to contain split dicts, got {type(raw_splits)!r}")

    metadata = {
        "node_feature_names": list(payload.get("node_feature_names", [])) if isinstance(payload, dict) else [],
        "edge_feature_names": list(payload.get("edge_feature_names", [])) if isinstance(payload, dict) else [],
    }
    sample_splits: dict[str, list[FrameGraphSample]] = {}
    for split_name in ("train", "val", "test"):
        raw_samples = raw_splits.get(split_name, [])
        sample_splits[split_name] = [
            pkl_record_to_frame_sample(record, fallback_frame_id=f"{split_name}_{idx}")
            for idx, record in enumerate(raw_samples)
        ]
    return sample_splits, metadata


def pkl_record_to_frame_sample(record: dict[str, Any], fallback_frame_id: str) -> FrameGraphSample:
    node_count = len(record.get("x", []))
    target_index = int(record.get("bz_node_idx", node_count - 1))
    scene_id = str(record.get("scene_id", "unknown"))
    meta = record.get("meta", {})
    timestamp = float(meta.get("timestamp", meta.get("frame_time", 0.0))) if isinstance(meta, dict) else 0.0
    node_ids = [f"{scene_id}_node_{idx}" for idx in range(node_count)]

    return FrameGraphSample(
        scene_id=scene_id,
        frame_id=str(record.get("frame_id", fallback_frame_id)),
        timestamp=timestamp,
        node_ids=node_ids,
        node_types=[str(value) for value in record.get("node_types", ["unknown"] * node_count)],
        x=torch.tensor(record.get("x", []), dtype=torch.float32),
        edge_index=torch.tensor(record.get("edge_index", [[], []]), dtype=torch.long),
        edge_attr=torch.tensor(record.get("edge_attr", []), dtype=torch.float32),
        target_indices=torch.tensor([target_index], dtype=torch.long),
        y=torch.tensor([float(record.get("label", 0))], dtype=torch.float32),
        edge_types=[str(value) for value in record.get("edge_type", [])],
    )


def load_frame_samples(graph_dir: str | Path) -> tuple[list[FrameGraphSample], dict[str, list[str]]]:
    graph_dir = Path(graph_dir)
    if graph_dir.is_file() and graph_dir.suffix == ".pkl":
        splits, metadata = load_frame_sample_splits_from_pickle(graph_dir)
        samples = splits["train"] + splits["val"] + splits["test"]
        return samples, metadata

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


def align_frame_samples(
    samples: list[FrameGraphSample],
    metadata: dict[str, list[str]],
    target_metadata: dict[str, list[str]],
) -> list[FrameGraphSample]:
    node_indices = feature_indices(metadata.get("node_feature_names", []), target_metadata.get("node_feature_names", []), NODE_ALIASES)
    edge_indices = feature_indices(metadata.get("edge_feature_names", []), target_metadata.get("edge_feature_names", []), EDGE_ALIASES)
    return [align_frame_sample(sample, node_indices, edge_indices) for sample in samples]


def alignment_report(metadata: dict[str, list[str]], target_metadata: dict[str, list[str]]) -> dict[str, Any]:
    source_node_names = metadata.get("node_feature_names", [])
    source_edge_names = metadata.get("edge_feature_names", [])
    target_node_names = target_metadata.get("node_feature_names", [])
    target_edge_names = target_metadata.get("edge_feature_names", [])
    node_indices = feature_indices(source_node_names, target_node_names, NODE_ALIASES)
    edge_indices = feature_indices(source_edge_names, target_edge_names, EDGE_ALIASES)
    return {
        "source_node_feature_names": source_node_names,
        "source_edge_feature_names": source_edge_names,
        "target_node_feature_names": target_node_names,
        "target_edge_feature_names": target_edge_names,
        "missing_node_features": [name for name, index in zip(target_node_names, node_indices) if index is None],
        "missing_edge_features": [name for name, index in zip(target_edge_names, edge_indices) if index is None],
    }


def feature_indices(source_names: list[str], target_names: list[str], aliases: dict[str, str]) -> list[int | None]:
    source_lookup = {name: idx for idx, name in enumerate(source_names)}
    indices: list[int | None] = []
    for target_name in target_names:
        source_name = target_name if target_name in source_lookup else aliases.get(target_name, target_name)
        indices.append(source_lookup.get(source_name))
    return indices


def align_frame_sample(
    sample: FrameGraphSample,
    node_indices: list[int | None],
    edge_indices: list[int | None],
) -> FrameGraphSample:
    x = select_feature_columns(sample.x, node_indices)
    edge_attr = select_feature_columns(sample.edge_attr, edge_indices)
    return FrameGraphSample(
        scene_id=sample.scene_id,
        frame_id=sample.frame_id,
        timestamp=sample.timestamp,
        node_ids=sample.node_ids,
        node_types=sample.node_types,
        x=x,
        edge_index=sample.edge_index,
        edge_attr=edge_attr,
        target_indices=sample.target_indices,
        y=sample.y,
        edge_types=sample.edge_types,
    )


def select_feature_columns(values: torch.Tensor, indices: list[int | None]) -> torch.Tensor:
    if values.ndim != 2:
        return values
    columns = []
    for index in indices:
        if index is None:
            columns.append(torch.zeros(values.size(0), 1, dtype=values.dtype))
        else:
            columns.append(values[:, index : index + 1])
    if not columns:
        return torch.empty(values.size(0), 0, dtype=values.dtype)
    return torch.cat(columns, dim=1)


def fit_feature_normalizer(samples: list[FrameGraphSample], metadata: dict[str, list[str]]) -> FeatureNormalizer:
    node_dim = samples[0].x.size(1)
    edge_dim = samples[0].edge_attr.size(1) if samples[0].edge_attr.ndim == 2 else 0
    node_values = torch.cat([sample.x for sample in samples if sample.x.numel()], dim=0)
    edge_values = torch.cat(
        [sample.edge_attr for sample in samples if sample.edge_attr.ndim == 2 and sample.edge_attr.numel()],
        dim=0,
    ) if edge_dim else torch.empty(0, 0)

    node_mean = torch.zeros(node_dim, dtype=torch.float32)
    node_std = torch.ones(node_dim, dtype=torch.float32)
    edge_mean = torch.zeros(edge_dim, dtype=torch.float32)
    edge_std = torch.ones(edge_dim, dtype=torch.float32)

    normalize_node = normalizable_feature_mask(metadata.get("node_feature_names", []), NODE_NON_NORMALIZED_FEATURES)
    normalize_edge = normalizable_feature_mask(metadata.get("edge_feature_names", []), EDGE_NON_NORMALIZED_FEATURES)
    if node_values.numel():
        node_mean[normalize_node] = node_values[:, normalize_node].mean(dim=0)
        node_std[normalize_node] = stable_std(node_values[:, normalize_node].std(dim=0))
    if edge_values.numel() and edge_dim:
        edge_mean[normalize_edge] = edge_values[:, normalize_edge].mean(dim=0)
        edge_std[normalize_edge] = stable_std(edge_values[:, normalize_edge].std(dim=0))

    return FeatureNormalizer(
        node_mean=node_mean,
        node_std=node_std,
        edge_mean=edge_mean,
        edge_std=edge_std,
        node_feature_names=list(metadata.get("node_feature_names", [])),
        edge_feature_names=list(metadata.get("edge_feature_names", [])),
    )


def stable_std(std: torch.Tensor) -> torch.Tensor:
    return torch.where(std < 1e-6, torch.ones_like(std), std)


def normalizable_feature_mask(feature_names: list[str], excluded_names: set[str]) -> torch.Tensor:
    return torch.tensor([name not in excluded_names for name in feature_names], dtype=torch.bool)


def normalize_frame_samples(samples: list[FrameGraphSample], normalizer: FeatureNormalizer) -> list[FrameGraphSample]:
    return [normalize_frame_sample(sample, normalizer) for sample in samples]


def normalize_frame_sample(sample: FrameGraphSample, normalizer: FeatureNormalizer) -> FrameGraphSample:
    x = (sample.x - normalizer.node_mean) / normalizer.node_std
    edge_attr = sample.edge_attr
    if edge_attr.ndim == 2 and edge_attr.numel():
        edge_attr = (edge_attr - normalizer.edge_mean) / normalizer.edge_std
    return FrameGraphSample(
        scene_id=sample.scene_id,
        frame_id=sample.frame_id,
        timestamp=sample.timestamp,
        node_ids=sample.node_ids,
        node_types=sample.node_types,
        x=x,
        edge_index=sample.edge_index,
        edge_attr=edge_attr,
        target_indices=sample.target_indices,
        y=sample.y,
        edge_types=sample.edge_types,
    )


def stabilize_expert_features(samples: list[FrameGraphSample], metadata: dict[str, list[str]]) -> list[FrameGraphSample]:
    edge_names = metadata.get("edge_feature_names", [])
    try:
        ttc_idx = edge_names.index("time_to_collision")
    except ValueError:
        return samples
    return [stabilize_frame_expert_features(sample, ttc_idx) for sample in samples]


def stabilize_frame_expert_features(sample: FrameGraphSample, ttc_idx: int) -> FrameGraphSample:
    edge_attr = sample.edge_attr.clone()
    if edge_attr.ndim == 2 and edge_attr.numel() and ttc_idx < edge_attr.size(1):
        edge_attr[:, ttc_idx] = torch.log1p(torch.clamp(edge_attr[:, ttc_idx], min=0.0, max=30.0))
    return FrameGraphSample(
        scene_id=sample.scene_id,
        frame_id=sample.frame_id,
        timestamp=sample.timestamp,
        node_ids=sample.node_ids,
        node_types=sample.node_types,
        x=sample.x,
        edge_index=sample.edge_index,
        edge_attr=edge_attr,
        target_indices=sample.target_indices,
        y=sample.y,
        edge_types=sample.edge_types,
    )


def normalize_temporal_samples(samples: list[TemporalGraphSample], normalizer: FeatureNormalizer) -> list[TemporalGraphSample]:
    return [
        TemporalGraphSample(
            scene_id=sample.scene_id,
            frame_id=sample.frame_id,
            timestamp=sample.timestamp,
            frames=normalize_frame_samples(sample.frames, normalizer),
            target_node_ids=sample.target_node_ids,
            y=sample.y,
        )
        for sample in samples
    ]


def stabilize_temporal_expert_features(
    samples: list[TemporalGraphSample],
    metadata: dict[str, list[str]],
) -> list[TemporalGraphSample]:
    return [
        TemporalGraphSample(
            scene_id=sample.scene_id,
            frame_id=sample.frame_id,
            timestamp=sample.timestamp,
            frames=stabilize_expert_features(sample.frames, metadata),
            target_node_ids=sample.target_node_ids,
            y=sample.y,
        )
        for sample in samples
    ]


NODE_ALIASES = {
    "type_id": "object_type_id",
    "object_type_id": "type_id",
}


EDGE_ALIASES = {
    "rel_vx": "relative_velocity_x",
    "rel_vy": "relative_velocity_y",
    "rel_heading": "relative_heading",
    "relative_velocity_x": "rel_vx",
    "relative_velocity_y": "rel_vy",
    "relative_heading": "rel_heading",
}


NODE_NON_NORMALIZED_FEATURES = {
    "object_type_id",
    "type_id",
    "visibility",
    "is_occluder",
    "is_vulnerable_road_user",
}


EDGE_NON_NORMALIZED_FEATURES = {
    "distance",
    "visibility_blocked",
}


def load_temporal_samples(
    graph_dir: str | Path,
    history: int = 5,
    prediction_horizon: int = 0,
) -> tuple[list[TemporalGraphSample], dict[str, list[str]]]:
    graph_dir = Path(graph_dir)
    cache_path = graph_dir.parent / f".temporal_cache_h{history}_t{prediction_horizon}.pkl"

    # Try loading from cache
    if cache_path.exists():
        print(f"Loading from cache: {cache_path.name}", flush=True)
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

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
            target_frame_idx = idx + prediction_horizon
            if target_frame_idx >= len(frame_samples):
                continue
            target_frame = frame_samples[target_frame_idx]
            start = max(0, idx - history + 1)
            window = frame_samples[start : idx + 1]
            target_node_ids = [current.node_ids[int(i)] for i in current.target_indices.tolist()]
            if prediction_horizon > 0:
                future_id_to_target = {
                    target_frame.node_ids[int(node_idx)]: target_idx
                    for target_idx, node_idx in enumerate(target_frame.target_indices.tolist())
                }
                aligned_y = []
                aligned_target_node_ids = []
                for target_node_id in target_node_ids:
                    if target_node_id not in future_id_to_target:
                        continue
                    aligned_target_node_ids.append(target_node_id)
                    aligned_y.append(target_frame.y[future_id_to_target[target_node_id]])
                if not aligned_y:
                    continue
                y = torch.stack(aligned_y).float()
                target_node_ids = aligned_target_node_ids
            else:
                y = current.y
            temporal_samples.append(
                TemporalGraphSample(
                    scene_id=current.scene_id,
                    frame_id=current.frame_id,
                    timestamp=current.timestamp,
                    frames=window,
                    target_node_ids=target_node_ids,
                    y=y,
                )
            )

    # Save cache
    print(f"Saving cache: {cache_path.name}", flush=True)
    with open(cache_path, 'wb') as f:
        pickle.dump((temporal_samples, metadata), f)

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
        edge_types=[str(value) for value in frame.get("edge_type", [])],
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
        edge_types=sample.edge_types,
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
