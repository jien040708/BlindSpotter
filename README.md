# BlindSpotter

BlindSpotter is a graph neural network research project for predicting blind-zone risk from IMPTC intersection trajectory data.

The main research question is:

```text
Given the current traffic scene or a short sequence of past frames,
can we predict whether a blind-zone region will become risky for vulnerable road users?
```

Unlike standard object detection, this project does not focus only on already visible pedestrians, cyclists, or e-scooter riders. Instead, we represent blind-zone regions as graph nodes and predict whether a VRU is likely to emerge near those regions in the near future.

## Project Overview

The project converts IMPTC trajectory sequences into frame-level scene graphs. Each graph contains traffic agents, reference vehicles, and blind-zone nodes. We then train and compare graph neural network models for blind-zone risk classification.

The final experiment compares three models under the same data split and evaluation setting:

```text
EIGAT single-frame model
MR-GCN single-frame model
Social-STGCNN / ST-GCN temporal model
```

The final evaluation uses IMPTC set01 to set05.

## Task Definition

Each frame is converted into a graph:

```text
traffic scene
-> frame graph
-> blind-zone node prediction
-> blind_y classification
```

For temporal modeling, the model uses a short history of previous frames:

```text
past 5 frames
-> graph encoder
-> temporal aggregation
-> blind-zone risk prediction
```

The target label `blind_y` indicates whether a blind-zone node becomes associated with future VRU emergence.

## Evaluation Metrics

The main metrics are:

```text
1. AUPRC
2. Best F1 / F1-score
3. AUROC
```

AUPRC is treated as the most important metric because blind-zone risk labels are highly imbalanced. Accuracy is not used as the main metric because a model can achieve high accuracy by predicting the majority negative class.

## Final Results

The final tracked result files are:

```text
outputs/results/three_models_comparison.md
outputs/results/three_models_comparison.csv
```

Current test-set summary:

| Model | AUPRC | AUROC | F1@0.5 | Best F1 | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| EIGAT single-frame | 0.3183 | 0.8574 | 0.3271 | 0.4658 | 0.5696 | 0.2295 |
| MR-GCN single-frame | 0.0147 | 0.5647 | 0.0000 | 0.0338 | 0.0000 | 0.0000 |
| Social-STGCNN temporal | 0.4573 | 0.8801 | 0.5145 | 0.5481 | 0.4708 | 0.5671 |

Overall, the temporal Social-STGCNN model achieved the strongest performance, especially on AUPRC, AUROC, Best F1, and Recall. EIGAT served as a strong single-frame baseline, while MR-GCN underperformed in the current relation setup.

These results should be interpreted as research baseline results, not as production-ready autonomous driving performance.

## Key Visualizations

Main comparison figures:

```text
outputs/figures/final_comparison/main_metric_comparison.png
outputs/figures/final_comparison/validation_curves.png
```

Research diagnostic figures:

```text
outputs/figures/imptc_set01_set02_set03_set04_set05_vru/pr_roc_curves.png
outputs/figures/imptc_set01_set02_set03_set04_set05_vru/threshold_sweep.png
outputs/figures/imptc_set01_set02_set03_set04_set05_vru/prediction_score_histogram.png
outputs/figures/imptc_set01_set02_set03_set04_set05_vru/scene_positive_distribution.png
outputs/figures/imptc_set01_set02_set03_set04_set05_vru/representative_graph_sample.png
outputs/figures/imptc_set01_set02_set03_set04_set05_vru/temporal_event_timeline.png
```

Poster-specific figures:

```text
outputs/figures/poster_set01_05/poster_panel6_model_evidence_v2.png
outputs/figures/poster_set01_05/poster_panel7_threshold_temporal_evidence_v3.png
```

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

The experiments were mainly run on CPU:

```bash
export KMP_DUPLICATE_LIB_OK=TRUE
export DEVICE=cpu
```

## Quick Reproduction

To run the full IMPTC set01-set05 experiment:

```bash
KMP_DUPLICATE_LIB_OK=TRUE \
DEVICE=cpu \
SETS="1 2 3 4 5" \
EPOCHS=20 \
TEMPORAL_EPOCHS=20 \
MAX_FRAMES=500 \
FRAME_STRIDE=5 \
./scripts/run_imptc_sets_experiments.sh
```

For a faster smoke test:

```bash
KMP_DUPLICATE_LIB_OK=TRUE \
DEVICE=cpu \
SETS="1 2" \
EPOCHS=2 \
TEMPORAL_EPOCHS=1 \
MAX_FRAMES=100 \
FRAME_STRIDE=10 \
./scripts/run_imptc_sets_experiments.sh
```

Supported set formats:

```bash
SETS="1 2 3"
SETS="01 02 03"
SETS="set1,set2,set3"
```

To skip temporal model training:

```bash
RUN_STGCN=0 SETS="1 2 3" EPOCHS=5 ./scripts/run_imptc_sets_experiments.sh
```

## Data Layout

Official IMPTC archives are downloaded into:

```text
data/downloads/
```

Extracted sequence data is placed under:

```text
data/imptc_sequences/
```

The runner creates a selected sequence root for each experiment:

```text
data/imptc_selected/set01_set02_set03_set04_set05/
```

If using manually prepared data, use the following structure:

```text
data/sample/
├── set1/
│   ├── 0000_...
│   └── 0001_...
├── set2/
│   └── 0050_...
└── set3/
    └── 0100_...
```

Each sequence folder should contain:

```text
<sequence>/
├── vehicles/
│   └── <track_id>/track.json
└── vrus/
    └── <track_id>/track.json
```

To run with local sample data:

```bash
SOURCE_ROOT=data/sample \
DOWNLOAD_IMPTC=0 \
SETS="1 2 3" \
./scripts/run_imptc_sets_experiments.sh
```

## Pipeline

The main runner is:

```text
scripts/run_imptc_sets_experiments.sh
```

It performs the following steps:

```text
1. Download selected IMPTC archives if needed
2. Build the selected sequence root
3. Convert trajectories into frame graph JSON files
4. Create a scene-level train/validation/test split
5. Train the EIGAT single-frame baseline
6. Train the MR-GCN single-frame comparison model
7. Train the ST-GCN temporal model
8. Aggregate metrics and generate figures
```

The split is scene-level rather than frame-level:

```text
train = 70%
validation = 15%
test = 15%
seed = 7
```

Scene-level splitting prevents frames from the same scene from appearing in both training and test sets.

## Graph Representation

Each frame graph stores:

```text
node_ids
node_types
x
edge_index
edge_attr
edge_type
blind_node_indices
blind_y
```

Main node types:

```text
reference_vehicle
vehicle
pedestrian
cyclist
e_scooter
occlusion_zone
```

Node features include:

```text
position
velocity
heading
object type
distance and angle to reference vehicle
visibility
occluder flag
VRU flag
```

Edge features include:

```text
distance
relative velocity
relative heading
time-to-collision
visibility blocking
```

## Known Limitations

The current implementation is a research baseline and has several limitations.

First, the reference vehicle is selected heuristically because IMPTC does not provide an explicit ego vehicle for this task.

Second, blind-zone nodes are generated using a simplified occluder heuristic rather than a full camera or LiDAR visibility model.

Third, `blind_y` is a proxy label for future VRU emergence near a blind zone. It is not a direct collision-risk or injury-severity label.

Fourth, positive labels are highly imbalanced, so threshold selection is important. The project uses validation threshold sweeps and reports Best F1 in addition to F1@0.5.

Finally, the current temporal model uses a short 5-frame history. Longer temporal horizons, map-aware features, lane context, traffic signals, and better label definitions remain important future work.

## Important Files

```text
src/imptc_dataset.py                  # IMPTC track.json parsing
src/graph_builder.py                  # graph node, edge, and blind-zone generation
src/label_builder.py                  # blind-zone label generation
src/gnn_dataset.py                    # graph JSON to training samples
src/gnn_models.py                     # EIGAT, MR-GCN, and ST-GCN models
src/training_utils.py                 # metrics, seeds, and training helpers

scripts/run_imptc_sets_experiments.sh # full configurable experiment runner
scripts/download_imptc_sequences.sh   # IMPTC archive downloader
scripts/preprocess_sample.py          # graph preprocessing
scripts/create_scene_split.py         # scene-level split
scripts/train_single_frame_gat.py     # EIGAT trainer
scripts/train_mrgcn.py                # MR-GCN wrapper
scripts/train_stgcn.py                # ST-GCN wrapper
scripts/aggregate_imptc_experiment_results.py
scripts/plot_imptc_experiment_results.py
scripts/plot_imptc_research_visualizations.py
```

## Submission Notes

The submission zip is intentionally lightweight. It includes the latest tracked code, README, documentation, summary results, and key visualizations from the `main` branch.

The following large files and folders are excluded:

```text
raw IMPTC data
downloaded archives
intermediate graph JSON files
model checkpoints
graph_dataset.pkl
large generated output folders
```

The full experiment can be reproduced by cloning the GitHub repository and running the commands described above.
```