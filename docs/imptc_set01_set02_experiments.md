# IMPTC Set01+Set02 Graph Experiments

This workflow trains the current EIGAT baseline, an MR-GCN comparison model, and the temporal ST-GCN/ST-GAT path on only official IMPTC set01 and set02 sequence data.

## Data

- Source chunks: `imptc_set_01.tar.gz`, `imptc_set_02.tar.gz`
- Extracted root: `data/imptc_sequences`
- Preprocessed graph output: `outputs/graphs_imptc_set01_set02`
- Split unit: scene, not frame
- Default split: 70% train, 15% validation, 15% test
- Stratification: positive-scene ratio is preserved separately for positive and negative scenes

## One-command Reproduction

```bash
KMP_DUPLICATE_LIB_OK=TRUE ./scripts/run_imptc_set01_set02_experiments.sh
```

Useful overrides:

```bash
DEVICE=mps EPOCHS=20 TEMPORAL_EPOCHS=6 ./scripts/run_imptc_set01_set02_experiments.sh
DEVICE=cpu EPOCHS=12 TEMPORAL_EPOCHS=4 ./scripts/run_imptc_set01_set02_experiments.sh
IMPTC_PARALLEL_PARTS=16 ./scripts/run_imptc_set01_set02_experiments.sh
```

## Models

- `eigat_single_frame`: frame graph -> edge-aware GAT -> blind-zone node embedding -> `blind_y`
- `mrgcn_single_frame`: frame graph -> relation-specific message passing -> blind-zone node embedding -> `blind_y`
- `stgcn_temporal_h5_t1`: past 5 frames -> per-frame graph encoder -> GRU temporal aggregation -> `blind_y` at `t+1`

## Shared Training Conditions

- Same preprocessed graph JSONs
- Same scene-level train/validation/test split
- Same train-fitted feature normalization
- Same TTC stabilization: `log1p(clamp(TTC, 0, 30))`
- Same class-imbalance strategy by default:
  - train-time negative sampling ratio: `20` negatives per positive
  - BCE positive weight override: `1`
- Same main selection metric: validation AUPRC
- Main reported metrics: AUPRC, AUROC, F1@0.5, best-F1

## Outputs

- Checkpoints and metric JSONs: `outputs/models/imptc_set01_set02`
- Aggregated CSV: `outputs/results/imptc_set01_set02_results.csv`
- Aggregated Markdown table: `outputs/results/imptc_set01_set02_results.md`
- Main metric bar chart: `outputs/figures/imptc_set01_set02/main_metric_comparison.png`
- Validation curve chart: `outputs/figures/imptc_set01_set02/validation_curves.png`
- Research diagnostic figures: `outputs/figures/imptc_set01_set02/research`

Current partial results from the interrupted local run:

- Partial Markdown table: `outputs/results/imptc_set01_set02_partial_results.md`
- Partial metric bar chart: `outputs/figures/imptc_set01_set02_partial/main_metric_comparison.png`
- Partial validation curve chart: `outputs/figures/imptc_set01_set02_partial/validation_curves.png`

## Individual Commands

```bash
IMPTC_PARALLEL_PARTS=12 ./scripts/download_imptc_sequences.sh imptc_set_01.tar.gz imptc_set_02.tar.gz
KMP_DUPLICATE_LIB_OK=TRUE python scripts/preprocess_sample.py --root data/imptc_sequences --output outputs/graphs_imptc_set01_set02 --max-frames 500 --frame-stride 5 --neighbor-radius 30
python scripts/create_scene_split.py --summary outputs/graphs_imptc_set01_set02/preprocess_summary.json --output outputs/splits/imptc_set01_set02_scene_split.json --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15 --seed 7
```

```bash
KMP_DUPLICATE_LIB_OK=TRUE python scripts/train_single_frame_gat.py --model eigat --graphs outputs/graphs_imptc_set01_set02 --scene-split outputs/splits/imptc_set01_set02_scene_split.json --output outputs/models/imptc_set01_set02/eigat_single_frame.pt --metrics-output outputs/models/imptc_set01_set02/eigat_single_frame.metrics.json --epochs 12 --hidden-dim 32 --layers 1 --heads 2 --negative-ratio 20 --pos-weight 1 --selection-metric auprc --device cpu
KMP_DUPLICATE_LIB_OK=TRUE python scripts/train_mrgcn.py --graphs outputs/graphs_imptc_set01_set02 --scene-split outputs/splits/imptc_set01_set02_scene_split.json --output outputs/models/imptc_set01_set02/mrgcn_single_frame.pt --metrics-output outputs/models/imptc_set01_set02/mrgcn_single_frame.metrics.json --epochs 12 --hidden-dim 32 --layers 1 --negative-ratio 20 --pos-weight 1 --selection-metric auprc --device cpu
KMP_DUPLICATE_LIB_OK=TRUE python scripts/train_stgcn.py --graphs outputs/graphs_imptc_set01_set02 --scene-split outputs/splits/imptc_set01_set02_scene_split.json --output outputs/models/imptc_set01_set02/stgcn_temporal_h5_t1.pt --metrics-output outputs/models/imptc_set01_set02/stgcn_temporal_h5_t1.metrics.json --history 5 --prediction-horizon 1 --epochs 4 --hidden-dim 32 --temporal-hidden-dim 32 --layers 1 --heads 2 --negative-ratio 20 --pos-weight 1 --selection-metric auprc --device cpu
```

After all three metric files exist, regenerate the final table and plots:

```bash
python scripts/aggregate_imptc_experiment_results.py outputs/models/imptc_set01_set02/eigat_single_frame.metrics.json outputs/models/imptc_set01_set02/mrgcn_single_frame.metrics.json outputs/models/imptc_set01_set02/stgcn_temporal_h5_t1.metrics.json --output-csv outputs/results/imptc_set01_set02_results.csv --output-md outputs/results/imptc_set01_set02_results.md
python scripts/plot_imptc_experiment_results.py outputs/models/imptc_set01_set02/eigat_single_frame.metrics.json outputs/models/imptc_set01_set02/mrgcn_single_frame.metrics.json outputs/models/imptc_set01_set02/stgcn_temporal_h5_t1.metrics.json --output-dir outputs/figures/imptc_set01_set02
```

To regenerate only the current EIGAT/MR-GCN partial plots:

```bash
python scripts/aggregate_imptc_experiment_results.py outputs/models/imptc_set01_set02/eigat_single_frame.metrics.json outputs/models/imptc_set01_set02/mrgcn_single_frame.metrics.json --output-csv outputs/results/imptc_set01_set02_partial_results.csv --output-md outputs/results/imptc_set01_set02_partial_results.md
python scripts/plot_imptc_experiment_results.py outputs/models/imptc_set01_set02/eigat_single_frame.metrics.json outputs/models/imptc_set01_set02/mrgcn_single_frame.metrics.json --output-dir outputs/figures/imptc_set01_set02_partial
```

## Research Visualizations

After the three checkpoints exist, generate the presentation/paper-oriented diagnostics:

```bash
KMP_DUPLICATE_LIB_OK=TRUE python scripts/plot_imptc_research_visualizations.py --graphs outputs/graphs_imptc_set01_set02 --scene-split outputs/splits/imptc_set01_set02_scene_split.json --output-dir outputs/figures/imptc_set01_set02/research
```

This writes:

- `pr_roc_curves.png`
- `threshold_sweep.png`
- `prediction_score_histogram.png`
- `scene_positive_distribution.png`
- `representative_graph_sample.png`
- `temporal_event_timeline.png`
