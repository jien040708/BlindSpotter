# BlindSpotter

BlindSpotter is a preprocessing prototype for predicting whether a scooter-like vulnerable road user (VRU) may emerge from a blind zone within a short future time window.

Our current research question is:

```text
P(VRU emerges from blind-zone within 3 seconds)
```

We use the IMPTC sample dataset first, not the full dataset, so that the whole team can run and debug the pipeline quickly.

## Team Workflow

Use this rule:

```text
GitHub = code sharing
Colab = running experiments
data/ = downloaded locally or in Colab, not committed
outputs/ = generated locally or in Colab, not committed
```

Do not upload dataset files, zip/tar files, generated graphs, or generated figures to GitHub. They are intentionally ignored by `.gitignore`.

## What To Edit

Most project work should happen in `src/` and `scripts/`.

```text
src/
├── imptc_dataset.py    # Reads IMPTC tracks and turns them into Scene/Frame/ObjectState
├── dataset.py          # Common data classes: Scene, Frame, ObjectState
├── graph_builder.py    # Builds graph nodes, edges, blind-zone nodes, graph features
├── label_builder.py    # Builds risk labels and blind-zone emergence labels
└── utils.py            # Shared helper functions

scripts/
├── download_imptc_sample.sh  # Downloads the IMPTC sample data
├── inspect_dataset.py        # Checks dataset structure
├── preprocess_sample.py      # Converts sample data into graph JSON files
└── visualize_sample.py       # Saves top-view PNG visualizations

notebooks/
└── IMPTC_BlindZone_GraphML.ipynb  # Colab draft / experiment notebook
```

If you want to change how IMPTC data is read, edit:

```text
src/imptc_dataset.py
```

If you want to change graph nodes, edges, or blind-zone candidate generation, edit:

```text
src/graph_builder.py
```

If you want to change the risk label, edit:

```text
src/label_builder.py
```

If you want to change command-line options or how preprocessing is executed, edit:

```text
scripts/preprocess_sample.py
```

If you only want to test ideas or make plots, use:

```text
notebooks/IMPTC_BlindZone_GraphML.ipynb
```

But once an idea becomes important, move the logic into `src/` or `scripts/` so everyone can reuse it.

## Colab Quick Start

In a new Colab notebook, run:

```python
!git clone https://github.com/jien040708/BlindSpotter.git
%cd BlindSpotter
!pip install -r requirements.txt
!bash scripts/download_imptc_sample.sh
```

Inspect the dataset:

```python
!python scripts/inspect_dataset.py --root data/sample
```

Run a small preprocessing test:

```python
!python scripts/preprocess_sample.py \
  --root data/sample \
  --output outputs/graphs \
  --max-sequences 1 \
  --max-frames 120 \
  --frame-stride 10
```

Create top-view figures:

```python
!python scripts/visualize_sample.py \
  --root data/sample \
  --output outputs/figures \
  --max-files 1 \
  --max-frames 20
```

Generated files will appear inside Colab:

```text
outputs/graphs/    # graph JSON files for later GNN input
outputs/figures/   # PNG visualizations for checking results
```

These outputs are not pushed to GitHub.

## Updating Code In Colab

If someone changed GitHub and you want the newest code in Colab:

```python
%cd /content/BlindSpotter
!git pull
```

Then rerun the script you need.

If you edited files directly inside Colab, download/copy those changes carefully or push them from a branch. For beginners, it is easier to edit code locally or ask Codex to edit files, then push to GitHub.

## Beginner Git Workflow

Before starting work:

```bash
git pull
```

Create your own branch:

```bash
git checkout -b feature/your-task-name
```

Examples:

```bash
git checkout -b feature/imptc-loader
git checkout -b feature/blind-zone-label
git checkout -b feature/visualization
git checkout -b feature/gnn-model
```

After editing code:

```bash
git status
git add src scripts notebooks README.md requirements.txt
git commit -m "Describe what you changed"
git push origin feature/your-task-name
```

Then open a Pull Request on GitHub and merge it into `main` after review.

Avoid committing these folders:

```text
data/
outputs/
```

They are ignored by Git, but still avoid manually forcing them into a commit.

## Recommended Team Split

Team member 1: Data and preprocessing

```text
src/imptc_dataset.py
scripts/inspect_dataset.py
scripts/preprocess_sample.py
```

Team member 2: Blind-zone and labels

```text
src/graph_builder.py
src/label_builder.py
scripts/visualize_sample.py
```

Team member 3: Model and experiments

```text
notebooks/
future src/model.py
future scripts/train.py
```

## Local Usage

If you run this project on your laptop instead of Colab:

```bash
git clone https://github.com/jien040708/BlindSpotter.git
cd BlindSpotter
pip install -r requirements.txt
bash scripts/download_imptc_sample.sh
python scripts/inspect_dataset.py --root data/sample
python scripts/preprocess_sample.py --root data/sample --output outputs/graphs --max-sequences 1 --max-frames 120 --frame-stride 10
```

## Pipeline

The current pipeline is:

```text
1. Download IMPTC sample data
2. Inspect dataset structure
3. Load vehicle and VRU trajectories
4. Select a temporary reference vehicle
5. Build frame-level graph representations
6. Add blind-zone candidate nodes
7. Generate heuristic blind-zone labels
8. Save graph JSON files and optional visualizations
```

## Graph Output

Each scene graph is saved as JSON by default.

For IMPTC, the loader chooses the longest vehicle track as a temporary `reference_vehicle`. The graph builder then adds `occlusion_zone` nodes behind nearby vehicle occluders.

Each frame graph includes:

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

`blind_y = 1` means a scooter-like VRU appears near that blind-zone within the future time window.

Current node features:

```text
x, y, vx, vy, heading, object_type_id, distance_to_ego,
relative_angle_to_ego, visibility, is_occluder, is_vulnerable_road_user
```

Current edge features:

```text
distance, relative_velocity_x, relative_velocity_y,
relative_heading, time_to_collision, visibility_blocked
```

## Current Limitations

The current blind-zone logic is a simple heuristic. It places candidate blind-zone nodes behind nearby vehicle occluders. It is not yet a full geometric occlusion model.

Future work:

- Improve occlusion geometry
- Add road geometry and map features
- Add traffic light and weather context
- Export directly to PyTorch Geometric `Data`
- Train a spatiotemporal GNN
- Evaluate early-warning performance
