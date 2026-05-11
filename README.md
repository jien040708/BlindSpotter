# E-Scooter Blind-Zone Risk Prediction

This project builds a preprocessing pipeline for predicting the emergence risk of e-scooters from blind spots using spatiotemporal graph learning.

## Current Goal

We only use a small sample dataset to verify the preprocessing pipeline.

## Pipeline

1. Inspect dataset structure
2. Extract ego and object trajectories
3. Build graph representation
4. Generate heuristic risk labels
5. Visualize sample scenes

## Project Structure

```text
project/
├── data/
│   └── sample/
├── scripts/
│   ├── inspect_dataset.py
│   ├── preprocess_sample.py
│   └── visualize_sample.py
├── src/
│   ├── dataset.py
│   ├── graph_builder.py
│   ├── label_builder.py
│   └── utils.py
└── outputs/
    ├── graphs/
    └── figures/
```

## Usage

Install dependencies:

```bash
pip install -r requirements.txt
```

Download and extract the IMPTC sample package:

```bash
bash scripts/download_imptc_sample.sh
```

Then run:

```bash
python scripts/inspect_dataset.py --root data/sample
python scripts/preprocess_sample.py --root data/sample --output outputs/graphs
python scripts/visualize_sample.py --root data/sample --output outputs/figures
```

## Colab Quick Start

Open the project notebook:

```text
notebooks/IMPTC_BlindZone_GraphML.ipynb
```

Or run the same workflow manually in Colab:

```python
!git clone https://github.com/jien040708/BlindSpotter.git
%cd BlindSpotter
!pip install -r requirements.txt
!bash scripts/download_imptc_sample.sh
!python scripts/inspect_dataset.py --root data/sample
!python scripts/preprocess_sample.py --root data/sample --output outputs/graphs --max-sequences 1 --max-frames 120 --frame-stride 10
```

`data/` and `outputs/` are intentionally ignored by Git. Each teammate should download the data and regenerate outputs locally or in Colab.

For the IMPTC sample package, the scripts automatically detect sequence folders with `vehicles/` and `vrus/` tracks:

```bash
python scripts/preprocess_sample.py --root data/sample --max-sequences 1 --max-frames 300 --frame-stride 5
```

The parser does not assume a fixed dataset schema. It searches for common scene, frame, ego pose, and object fields. If a required field does not exist, the scripts print a warning and skip or use a conservative fallback instead of crashing.

## Graph Output

Each scene graph is saved as JSON by default. A graph contains frame-level node features, spatial edges, temporal edges, and a heuristic risk label.
For IMPTC, the loader chooses the longest vehicle track as a temporary `reference_vehicle` and adds `occlusion_zone` nodes behind nearby vehicle occluders. Each frame graph includes `blind_node_indices` and `blind_y`, where `blind_y = 1` means a scooter-like VRU appears near that blind-zone within the future time window.

Node features currently include:

```text
x, y, vx, vy, heading, object_type_id, distance_to_ego,
relative_angle_to_ego, visibility, is_occluder, is_vulnerable_road_user
```

Edge features currently include:

```text
distance, relative_velocity_x, relative_velocity_y,
relative_heading, time_to_collision, visibility_blocked
```

## Heuristic Label

The initial label is intentionally simple:

```text
risk = 1
```

when an e-scooter appears near an occluder and comes within the ego vehicle risk distance.

```text
risk = 0
```

otherwise.

## Future Work

- Improve occlusion modeling
- Add road geometry and map features
- Export directly to PyTorch Geometric `Data`
- Train Spatiotemporal GNN
- Evaluate early warning performance
