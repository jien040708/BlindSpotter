#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

export KMP_DUPLICATE_LIB_OK="${KMP_DUPLICATE_LIB_OK:-TRUE}"
export IMPTC_PARALLEL_PARTS="${IMPTC_PARALLEL_PARTS:-12}"

SETS="${SETS:-01 02}"
SOURCE_ROOT="${SOURCE_ROOT:-data/imptc_sequences}"
DOWNLOAD_IMPTC="${DOWNLOAD_IMPTC:-1}"
EPOCHS="${EPOCHS:-12}"
TEMPORAL_EPOCHS="${TEMPORAL_EPOCHS:-4}"
HIDDEN_DIM="${HIDDEN_DIM:-32}"
LAYERS="${LAYERS:-1}"
HEADS="${HEADS:-2}"
NEGATIVE_RATIO="${NEGATIVE_RATIO:-20}"
POS_WEIGHT="${POS_WEIGHT:-1}"
SEED="${SEED:-7}"
DEVICE="${DEVICE:-cpu}"
MAX_FRAMES="${MAX_FRAMES:-500}"
FRAME_STRIDE="${FRAME_STRIDE:-5}"
NEIGHBOR_RADIUS="${NEIGHBOR_RADIUS:-30}"
RUN_STGCN="${RUN_STGCN:-1}"
RUN_RESEARCH_PLOTS="${RUN_RESEARCH_PLOTS:-1}"
TEMPORAL_MODEL="${TEMPORAL_MODEL:-social_stgcn}"
LABEL_TARGET="${LABEL_TARGET:-scooter}"

normalize_sets() {
  python - "$SETS" <<'PY'
import re
import sys

tokens = re.split(r"[\s,]+", sys.argv[1].strip())
sets = []
for token in tokens:
    if not token:
        continue
    token = token.lower().replace("imptc_set_", "").replace("set", "")
    number = int(token)
    if number < 1 or number > 5:
        raise SystemExit(f"Unsupported IMPTC set: {number}. Expected 1..5.")
    sets.append(f"{number:02d}")
print(" ".join(dict.fromkeys(sets)))
PY
}

SET_IDS="$(normalize_sets)"
if [[ -z "${SET_IDS}" ]]; then
  echo "[ERROR] SETS is empty. Example: SETS=\"01 02\" $0"
  exit 1
fi

SET_TAG="set$(echo "${SET_IDS}" | sed 's/ /_set/g')"
LABEL_SUFFIX=""
if [[ "${LABEL_TARGET:-scooter}" != "scooter" ]]; then
  LABEL_SUFFIX="_${LABEL_TARGET}"
fi
TAG="${SET_TAG}${LABEL_SUFFIX}"
SELECTED_ROOT="${SELECTED_ROOT:-data/imptc_selected/${TAG}}"
GRAPHS_DIR="${GRAPHS_DIR:-outputs/graphs_imptc_${TAG}}"
SPLIT_PATH="${SPLIT_PATH:-outputs/splits/imptc_${TAG}_scene_split.json}"
MODEL_DIR="${MODEL_DIR:-outputs/models/imptc_${TAG}}"
RESULT_DIR="${RESULT_DIR:-outputs/results}"
FIGURE_DIR="${FIGURE_DIR:-outputs/figures/imptc_${TAG}}"

mkdir -p "${MODEL_DIR}" "${RESULT_DIR}" "${FIGURE_DIR}"

archives=()
for set_id in ${SET_IDS}; do
  archives+=("imptc_set_${set_id}.tar.gz")
done

echo "[1/8] Downloading IMPTC ${SET_IDS} if needed"
if [[ "${DOWNLOAD_IMPTC}" == "1" && "${SOURCE_ROOT}" == "data/imptc_sequences" ]]; then
  ./scripts/download_imptc_sequences.sh "${archives[@]}"
else
  echo "[INFO] Skipping official download because DOWNLOAD_IMPTC=${DOWNLOAD_IMPTC}, SOURCE_ROOT=${SOURCE_ROOT}"
fi

echo "[2/8] Preparing selected sequence root: ${SELECTED_ROOT}"
rm -rf "${SELECTED_ROOT}"
mkdir -p "${SELECTED_ROOT}"

link_sequence_dir() {
  local sequence_dir="$1"
  local basename
  basename="$(basename "${sequence_dir}")"
  if [[ -d "${sequence_dir}/vehicles" && -d "${sequence_dir}/vrus" && ! -e "${SELECTED_ROOT}/${basename}" ]]; then
    ln -s "$(python - "${sequence_dir}" "${SELECTED_ROOT}" <<'PY'
from pathlib import Path
import os
import sys

source = Path(sys.argv[1]).resolve()
target_parent = Path(sys.argv[2]).resolve()
print(os.path.relpath(source, target_parent))
PY
)" "${SELECTED_ROOT}/${basename}"
  fi
}

for set_id in ${SET_IDS}; do
  set_num=$((10#${set_id}))
  found_named_set=0
  for set_dir in \
    "${SOURCE_ROOT}/set${set_num}" \
    "${SOURCE_ROOT}/set${set_id}" \
    "${SOURCE_ROOT}/imptc_set_${set_id}" \
    "${SOURCE_ROOT}/imptc_set_${set_num}"; do
    if [[ -d "${set_dir}" ]]; then
      found_named_set=1
      if [[ -d "${set_dir}/vehicles" && -d "${set_dir}/vrus" ]]; then
        link_sequence_dir "${set_dir}"
      else
        while IFS= read -r sequence_dir; do
          link_sequence_dir "${sequence_dir}"
        done < <(find "${set_dir}" -mindepth 1 -maxdepth 2 -type d -name vehicles -print | sed 's#/vehicles$##' | sort)
      fi
    fi
  done
  if [[ "${found_named_set}" == "1" ]]; then
    continue
  fi

  start=$(((set_num - 1) * 50))
  end=$((start + 49))
  for seq_idx in $(seq -f "%04g" "${start}" "${end}"); do
    matches=("${SOURCE_ROOT}/${seq_idx}"_*)
    if [[ -e "${matches[0]}" ]]; then
      link_sequence_dir "${matches[0]}"
    fi
  done
done
sequence_count="$(find "${SELECTED_ROOT}" -maxdepth 1 -type l | wc -l | tr -d ' ')"
if [[ "${sequence_count}" == "0" ]]; then
  echo "[ERROR] No selected sequences found under ${SELECTED_ROOT}."
  exit 1
fi
echo "[OK] selected_sequences=${sequence_count}, tag=${TAG}"

echo "[3/8] Preprocessing selected sequences into graph JSON"
python scripts/preprocess_sample.py \
  --root "${SELECTED_ROOT}" \
  --output "${GRAPHS_DIR}" \
  --max-frames "${MAX_FRAMES}" \
  --frame-stride "${FRAME_STRIDE}" \
  --neighbor-radius "${NEIGHBOR_RADIUS}" \
  --label-target "${LABEL_TARGET}"

echo "[4/8] Creating shared scene split"
python scripts/create_scene_split.py \
  --summary "${GRAPHS_DIR}/preprocess_summary.json" \
  --output "${SPLIT_PATH}" \
  --train-ratio 0.7 \
  --val-ratio 0.15 \
  --test-ratio 0.15 \
  --seed "${SEED}"

echo "[5/8] Training EIGAT single-frame baseline"
python scripts/train_single_frame_gat.py \
  --model eigat \
  --graphs "${GRAPHS_DIR}" \
  --scene-split "${SPLIT_PATH}" \
  --output "${MODEL_DIR}/eigat_single_frame.pt" \
  --metrics-output "${MODEL_DIR}/eigat_single_frame.metrics.json" \
  --epochs "${EPOCHS}" \
  --hidden-dim "${HIDDEN_DIM}" \
  --layers "${LAYERS}" \
  --heads "${HEADS}" \
  --negative-ratio "${NEGATIVE_RATIO}" \
  --pos-weight "${POS_WEIGHT}" \
  --selection-metric auprc \
  --seed "${SEED}" \
  --device "${DEVICE}"

echo "[6/8] Training MR-GCN single-frame comparison"
python scripts/train_mrgcn.py \
  --graphs "${GRAPHS_DIR}" \
  --scene-split "${SPLIT_PATH}" \
  --output "${MODEL_DIR}/mrgcn_single_frame.pt" \
  --metrics-output "${MODEL_DIR}/mrgcn_single_frame.metrics.json" \
  --epochs "${EPOCHS}" \
  --hidden-dim "${HIDDEN_DIM}" \
  --layers "${LAYERS}" \
  --negative-ratio "${NEGATIVE_RATIO}" \
  --pos-weight "${POS_WEIGHT}" \
  --selection-metric auprc \
  --seed "${SEED}" \
  --device "${DEVICE}"

metric_files=(
  "${MODEL_DIR}/eigat_single_frame.metrics.json"
  "${MODEL_DIR}/mrgcn_single_frame.metrics.json"
)

if [[ "${RUN_STGCN}" == "1" ]]; then
  echo "[7/8] Training ST-GCN/ST-GAT temporal comparison"
  python scripts/train_stgcn.py \
    --graphs "${GRAPHS_DIR}" \
    --scene-split "${SPLIT_PATH}" \
    --output "${MODEL_DIR}/stgcn_temporal_h5_t1.pt" \
    --metrics-output "${MODEL_DIR}/stgcn_temporal_h5_t1.metrics.json" \
    --temporal-model "${TEMPORAL_MODEL}" \
    --history 5 \
    --prediction-horizon 1 \
    --epochs "${TEMPORAL_EPOCHS}" \
    --hidden-dim "${HIDDEN_DIM}" \
    --temporal-hidden-dim "${HIDDEN_DIM}" \
    --layers "${LAYERS}" \
    --heads "${HEADS}" \
    --negative-ratio "${NEGATIVE_RATIO}" \
    --pos-weight "${POS_WEIGHT}" \
    --selection-metric auprc \
    --seed "${SEED}" \
    --device "${DEVICE}"
  metric_files+=("${MODEL_DIR}/stgcn_temporal_h5_t1.metrics.json")
else
  echo "[7/8] Skipping ST-GCN/ST-GAT because RUN_STGCN=${RUN_STGCN}"
fi

echo "[8/8] Aggregating metrics and plotting figures"
python scripts/aggregate_imptc_experiment_results.py \
  "${metric_files[@]}" \
  --output-csv "${RESULT_DIR}/imptc_${TAG}_results.csv" \
  --output-md "${RESULT_DIR}/imptc_${TAG}_results.md"

python scripts/plot_imptc_experiment_results.py \
  "${metric_files[@]}" \
  --output-dir "${FIGURE_DIR}"

if [[ "${RUN_RESEARCH_PLOTS}" == "1" && "${RUN_STGCN}" == "1" ]]; then
  python scripts/plot_imptc_research_visualizations.py \
    --graphs "${GRAPHS_DIR}" \
    --scene-split "${SPLIT_PATH}" \
    --eigat "${MODEL_DIR}/eigat_single_frame.pt" \
    --mrgcn "${MODEL_DIR}/mrgcn_single_frame.pt" \
    --stgcn "${MODEL_DIR}/stgcn_temporal_h5_t1.pt" \
    --output-dir "${FIGURE_DIR}/research"
elif [[ "${RUN_RESEARCH_PLOTS}" == "1" ]]; then
  python scripts/plot_imptc_research_visualizations.py \
    --graphs "${GRAPHS_DIR}" \
    --scene-split "${SPLIT_PATH}" \
    --eigat "${MODEL_DIR}/eigat_single_frame.pt" \
    --mrgcn "${MODEL_DIR}/mrgcn_single_frame.pt" \
    --stgcn "${MODEL_DIR}/missing_stgcn.pt" \
    --output-dir "${FIGURE_DIR}/research"
fi

echo "[OK] Tag: ${TAG}"
echo "[OK] Results: ${RESULT_DIR}/imptc_${TAG}_results.md"
echo "[OK] Figures: ${FIGURE_DIR}"
