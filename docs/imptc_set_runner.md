# IMPTC Set Runner

`scripts/run_imptc_sets_experiments.sh` runs the full blind-zone GNN experiment for any selected IMPTC sequence sets:

1. download selected IMPTC archives if missing
2. collect only the requested sequence folders into `data/imptc_selected/<tag>`
3. preprocess sequences into graph JSON
4. create a shared scene-level train/val/test split
5. train EIGAT, MR-GCN, and optionally Social-STGCNN / MR-GCN+Transformer temporal models
6. aggregate AUROC, AUPRC, F1-score results
7. save metric plots and research visualizations

## Basic Commands

Run set01 + set02:

```bash
SETS="1 2" KMP_DUPLICATE_LIB_OK=TRUE DEVICE=cpu EPOCHS=12 TEMPORAL_EPOCHS=4 ./scripts/run_imptc_sets_experiments.sh
```

Run set01 + set02 + set03:

```bash
SETS="1 2 3" KMP_DUPLICATE_LIB_OK=TRUE DEVICE=cpu EPOCHS=12 TEMPORAL_EPOCHS=4 ./scripts/run_imptc_sets_experiments.sh
```

Run one set only:

```bash
SETS="3" ./scripts/run_imptc_sets_experiments.sh
```

The `SETS` value accepts these equivalent forms:

```bash
SETS="1 2 3"
SETS="01 02 03"
SETS="set1,set2,set3"
```

If you manually put data under `data/sample`, point the runner there and skip official downloading:

```bash
SOURCE_ROOT=data/sample DOWNLOAD_IMPTC=0 SETS="1" ./scripts/run_imptc_sets_experiments.sh
```

This also works when `data/sample` contains named set folders:

```text
data/sample/set1/<sequence folders>
data/sample/set2/<sequence folders>
data/sample/set3/<sequence folders>
```

Then run:

```bash
SOURCE_ROOT=data/sample DOWNLOAD_IMPTC=0 SETS="1 2 3" ./scripts/run_imptc_sets_experiments.sh
```

## Faster Debug Runs

Skip temporal training:

```bash
SETS="1 2 3" RUN_STGCN=0 EPOCHS=5 ./scripts/run_imptc_sets_experiments.sh
```

Use fewer frames per sequence:

```bash
SETS="1 2" MAX_FRAMES=100 FRAME_STRIDE=10 RUN_STGCN=0 EPOCHS=2 ./scripts/run_imptc_sets_experiments.sh
```

Use more parallel download parts for large sets:

```bash
SETS="1 2 3 4 5" IMPTC_PARALLEL_PARTS=16 ./scripts/run_imptc_sets_experiments.sh
```

## Outputs

For `SETS="1 2 3"`, the tag is `set01_set02_set03`.

```text
data/imptc_selected/set01_set02_set03/
outputs/graphs_imptc_set01_set02_set03/
outputs/splits/imptc_set01_set02_set03_scene_split.json
outputs/models/imptc_set01_set02_set03/
outputs/results/imptc_set01_set02_set03_results.md
outputs/results/imptc_set01_set02_set03_results.csv
outputs/figures/imptc_set01_set02_set03/
outputs/figures/imptc_set01_set02_set03/research/
```

The research figure folder contains:

```text
pr_roc_curves.png
threshold_sweep.png
prediction_score_histogram.png
scene_positive_distribution.png
representative_graph_sample.png
temporal_event_timeline.png
```

## Main Parameters

```bash
SETS="1 2"              # selected IMPTC sets, supported official chunks: 1..5
SOURCE_ROOT=data/imptc_sequences
DOWNLOAD_IMPTC=1        # set 0 when using manually copied data under data/sample
LABEL_TARGET=scooter    # scooter or vru
EPOCHS=12              # EIGAT and MR-GCN epochs
TEMPORAL_EPOCHS=4      # ST-GCN epochs
RUN_STGCN=1            # set 0 to skip temporal model
TEMPORAL_MODEL=social_stgcn  # gat_gru, social_stgcn, or mrgcn_transformer
RUN_RESEARCH_PLOTS=1   # set 0 to skip paper-style visualizations
DEVICE=cpu             # cpu, mps, or cuda depending on local environment
NEGATIVE_RATIO=20      # negative blind-zone samples per positive sample
POS_WEIGHT=1           # BCE positive class weight override
MAX_FRAMES=500         # max frames per scene during preprocessing
FRAME_STRIDE=5         # frame sampling stride
NEIGHBOR_RADIUS=30     # graph edge radius in meters
```

## Assumption

The runner assumes each official IMPTC set contributes 50 extracted sequence folders and that extracted sequence folders begin with zero-padded numeric prefixes:

```text
set01 -> 0000..0049
set02 -> 0050..0099
set03 -> 0100..0149
set04 -> 0150..0199
set05 -> 0200..0249
```

If the extracted folder naming changes, update the selection block in `scripts/run_imptc_sets_experiments.sh`.
