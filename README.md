# BlindSpotter

BlindSpotter는 IMPTC 교차로 trajectory 데이터에서 blind-zone 위험을 graph neural network로 예측하는 연구 코드입니다.

현재 연구 질문은 다음과 같습니다.

```text
현재 frame 또는 과거 frame sequence를 보고,
blind-zone node에서 VRU가 등장할 위험을 예측할 수 있는가?
```

즉, 이미 보이는 전동킥보드나 자전거를 탐지하는 것이 아니라, **차량 뒤쪽 occlusion/blind-zone 자체가 위험한지**를 예측하는 것이 목표입니다.

## 현재 구현된 실험

현재 branch의 핵심 실험은 IMPTC set 단위로 같은 조건에서 세 모델을 비교하는 것입니다.

```text
IMPTC set01, set02, ...
-> frame graph 전처리
-> scene-level train/val/test split
-> EIGAT single-frame classifier
-> MR-GCN single-frame comparison
-> ST-GCN/ST-GAT temporal comparison
-> AUROC, AUPRC, F1-score 평가
-> 논문용 diagnostic figure 생성
```

평가 metric의 우선순위는 다음과 같습니다.

```text
1. AUPRC
2. F1-score / best F1
3. AUROC
```

데이터 불균형이 매우 크기 때문에, 일반 accuracy는 main metric으로 보지 않습니다.

## 빠른 실행

로컬에서 set01 + set02를 그대로 재실행하려면:

```bash
KMP_DUPLICATE_LIB_OK=TRUE DEVICE=cpu EPOCHS=12 TEMPORAL_EPOCHS=4 SETS="1 2" ./scripts/run_imptc_sets_experiments.sh
```

set01 + set02 + set03을 실행하려면:

```bash
KMP_DUPLICATE_LIB_OK=TRUE DEVICE=cpu EPOCHS=12 TEMPORAL_EPOCHS=4 SETS="1 2 3" ./scripts/run_imptc_sets_experiments.sh
```

set 번호 표기는 아래 형식을 모두 지원합니다.

```bash
SETS="1 2 3"
SETS="01 02 03"
SETS="set1,set2,set3"
```

빠르게 single-frame 모델만 확인하려면 ST-GCN을 끌 수 있습니다.

```bash
SETS="1 2 3" RUN_STGCN=0 EPOCHS=5 ./scripts/run_imptc_sets_experiments.sh
```

## 데이터 추가 위치

공식 IMPTC set을 사용하는 경우에는 직접 데이터를 옮길 필요가 없습니다. 아래 명령을 실행하면 필요한 archive를 `data/downloads/`에 받고, 압축을 `data/imptc_sequences/`에 풉니다.

```bash
SETS="1 2 3" ./scripts/run_imptc_sets_experiments.sh
```

직접 받은 IMPTC sequence를 sample 폴더에 넣어 실험하려면 다음 구조를 사용합니다.

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

각 sequence 폴더에는 최소한 아래 구조가 있어야 합니다.

```text
<sequence>/
├── vehicles/
│   └── <track_id>/track.json
└── vrus/
    └── <track_id>/track.json
```

`data/sample`의 데이터를 사용하려면 공식 다운로드를 끄고 `SOURCE_ROOT`를 지정합니다.

```bash
SOURCE_ROOT=data/sample DOWNLOAD_IMPTC=0 SETS="1 2 3" ./scripts/run_imptc_sets_experiments.sh
```

기존 sample처럼 `data/sample` 바로 아래에 sequence 폴더를 넣은 경우에도 실행할 수 있습니다.

```bash
SOURCE_ROOT=data/sample DOWNLOAD_IMPTC=0 SETS="1" ./scripts/run_imptc_sets_experiments.sh
```

## 전체 파이프라인

범용 실행 스크립트는 아래 순서로 동작합니다.

```text
1. 선택한 IMPTC set archive 다운로드
2. 선택한 sequence만 data/imptc_selected/<tag>/로 symlink 구성
3. IMPTC track.json 로드
4. reference vehicle 선택
5. frame 단위 scene graph 생성
6. vehicle/VRU/blind-zone node 생성
7. expert-informed edge feature 생성
8. blind-zone emergence label 생성
9. scene-level stratified train/val/test split 생성
10. EIGAT single-frame 모델 학습
11. MR-GCN single-frame 모델 학습
12. ST-GCN/ST-GAT temporal 모델 학습
13. metric JSON, CSV, Markdown 저장
14. metric plot과 research visualization 저장
```

### Graph Input

각 frame graph에는 다음 정보가 저장됩니다.

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

주요 node 종류는 다음과 같습니다.

```text
reference_vehicle
vehicle
pedestrian
cyclist
e_scooter
occlusion_zone
```

현재 node feature는 위치, 속도, heading, object type, reference vehicle 기준 거리/각도, visibility, occluder 여부, VRU 여부 등을 포함합니다.

현재 edge feature는 다음 값을 포함합니다.

```text
distance
relative_velocity_x
relative_velocity_y
relative_heading
time_to_collision
visibility_blocked
```

MR-GCN은 `edge_type`을 relation id로 사용하고, EIGAT/ST-GCN은 edge feature를 attention/message passing에 사용합니다.

## Train/Validation/Test Split

split은 frame 단위 random split이 아니라 **scene-level split**입니다. 같은 scene의 frame이 train과 test에 동시에 들어가면 leakage가 생기기 때문입니다.

기본 비율은 다음과 같습니다.

```text
train = 70%
validation = 15%
test = 15%
seed = 7
positive scene 비율 유지
```

관련 스크립트:

```bash
python scripts/create_scene_split.py \
  --summary outputs/graphs_imptc_set01_set02/preprocess_summary.json \
  --output outputs/splits/imptc_set01_set02_scene_split.json \
  --train-ratio 0.7 \
  --val-ratio 0.15 \
  --test-ratio 0.15 \
  --seed 7
```

작은 sample 데이터에서는 scene 수가 부족할 수 있으므로, split script가 train/val/test가 완전히 비지 않도록 최소 보정을 수행합니다.

## 결과물 위치

`SETS="1 2 3"`이면 tag는 `set01_set02_set03`입니다.

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

주요 figure는 다음과 같습니다.

```text
main_metric_comparison.png
validation_curves.png
research/pr_roc_curves.png
research/threshold_sweep.png
research/prediction_score_histogram.png
research/scene_positive_distribution.png
research/representative_graph_sample.png
research/temporal_event_timeline.png
```

`RUN_STGCN=0`으로 실행하면 temporal model과 `temporal_event_timeline.png`는 생략됩니다.

## 주요 명령어 모음

다운로드만 따로 실행:

```bash
IMPTC_PARALLEL_PARTS=12 ./scripts/download_imptc_sequences.sh imptc_set_01.tar.gz imptc_set_02.tar.gz
```

전처리만 실행:

```bash
python scripts/preprocess_sample.py \
  --root data/imptc_sequences \
  --output outputs/graphs_imptc_set01_set02 \
  --max-frames 500 \
  --frame-stride 5 \
  --neighbor-radius 30
```

EIGAT만 학습:

```bash
python scripts/train_single_frame_gat.py \
  --model eigat \
  --graphs outputs/graphs_imptc_set01_set02 \
  --scene-split outputs/splits/imptc_set01_set02_scene_split.json \
  --output outputs/models/imptc_set01_set02/eigat_single_frame.pt \
  --metrics-output outputs/models/imptc_set01_set02/eigat_single_frame.metrics.json \
  --epochs 12 \
  --selection-metric auprc \
  --device cpu
```

MR-GCN만 학습:

```bash
python scripts/train_mrgcn.py \
  --graphs outputs/graphs_imptc_set01_set02 \
  --scene-split outputs/splits/imptc_set01_set02_scene_split.json \
  --output outputs/models/imptc_set01_set02/mrgcn_single_frame.pt \
  --metrics-output outputs/models/imptc_set01_set02/mrgcn_single_frame.metrics.json \
  --epochs 12 \
  --selection-metric auprc \
  --device cpu
```

ST-GCN/ST-GAT만 학습:

```bash
python scripts/train_stgcn.py \
  --graphs outputs/graphs_imptc_set01_set02 \
  --scene-split outputs/splits/imptc_set01_set02_scene_split.json \
  --output outputs/models/imptc_set01_set02/stgcn_temporal_h5_t1.pt \
  --metrics-output outputs/models/imptc_set01_set02/stgcn_temporal_h5_t1.metrics.json \
  --history 5 \
  --prediction-horizon 1 \
  --epochs 4 \
  --selection-metric auprc \
  --device cpu
```

결과 집계:

```bash
python scripts/aggregate_imptc_experiment_results.py \
  outputs/models/imptc_set01_set02/eigat_single_frame.metrics.json \
  outputs/models/imptc_set01_set02/mrgcn_single_frame.metrics.json \
  outputs/models/imptc_set01_set02/stgcn_temporal_h5_t1.metrics.json \
  --output-csv outputs/results/imptc_set01_set02_results.csv \
  --output-md outputs/results/imptc_set01_set02_results.md
```

기본 metric plot:

```bash
python scripts/plot_imptc_experiment_results.py \
  outputs/models/imptc_set01_set02/eigat_single_frame.metrics.json \
  outputs/models/imptc_set01_set02/mrgcn_single_frame.metrics.json \
  outputs/models/imptc_set01_set02/stgcn_temporal_h5_t1.metrics.json \
  --output-dir outputs/figures/imptc_set01_set02
```

논문용 diagnostic figure:

```bash
python scripts/plot_imptc_research_visualizations.py \
  --graphs outputs/graphs_imptc_set01_set02 \
  --scene-split outputs/splits/imptc_set01_set02_scene_split.json \
  --eigat outputs/models/imptc_set01_set02/eigat_single_frame.pt \
  --mrgcn outputs/models/imptc_set01_set02/mrgcn_single_frame.pt \
  --stgcn outputs/models/imptc_set01_set02/stgcn_temporal_h5_t1.pt \
  --output-dir outputs/figures/imptc_set01_set02/research
```

## 현재 알고리즘의 문제점

현재 구현은 연구 baseline으로는 쓸 수 있지만, 아직 “좋은 모델”이라고 말하기 어렵습니다.

1. Positive label이 매우 적습니다.

   set01 + set02 기준으로 positive blind-zone scene은 100개 중 20개 정도이고, blind-zone target 단위 positive 비율은 더 낮습니다. 그래서 threshold 0.5에서는 recall/F1이 0으로 떨어지는 경우가 많습니다. 이 때문에 AUPRC와 best-F1을 같이 봐야 합니다.

2. Reference vehicle 선택이 heuristic입니다.

   IMPTC에는 명확한 ego vehicle 정의가 없어서, 현재는 가장 길게 관측된 vehicle track을 `reference_vehicle`로 사용합니다. 실제 운전자 시야 기준의 blind-zone과 다를 수 있습니다.

3. Blind-zone 생성이 단순 occluder heuristic입니다.

   차량 뒤쪽에 occlusion-zone candidate를 만드는 방식이며, LiDAR/camera visibility polygon이나 precise map geometry를 사용한 진짜 geometric occlusion model은 아닙니다.

4. Label이 emergence proxy입니다.

   `blind_y=1`은 미래 window 안에서 scooter-like VRU가 blind-zone 근처에 등장했다는 proxy label입니다. 실제 사고 위험, 운전자 반응 가능성, 충돌 가능성과는 아직 다릅니다.

5. Scene-level generalization이 어렵습니다.

   같은 교차로/시간대/traffic pattern이 많으면 모델이 구조적 위험을 배우기보다 dataset bias를 배울 수 있습니다. 반드시 scene-level split을 유지해야 합니다.

6. Temporal model은 아직 짧은 history baseline입니다.

   현재 ST-GCN/ST-GAT는 `history=5`, `prediction_horizon=1` 기준의 baseline입니다. 연구 목적에 맞게 1초 전후 history와 미래 horizon을 더 체계적으로 sweep해야 합니다.

7. Map/context feature가 부족합니다.

   crosswalk, lane proximity, traffic light, weather, road geometry가 아직 충분히 들어가지 않았습니다. Expert-informed attention의 설득력을 높이려면 이 feature들이 중요합니다.

## 현재 set01 + set02 baseline 해석

기존 실행 결과에서는 test 기준으로 대략 아래 경향이 나왔습니다.

```text
EIGAT  : AUPRC 0.035, AUROC 0.504, best F1 0.089
MR-GCN : AUPRC 0.015, AUROC 0.479, best F1 0.035
ST-GCN : AUPRC 0.027, AUROC 0.569, best F1 0.073
```

이 결과는 “현재 데이터와 label로 학습 파이프라인은 작동한다”는 baseline 의미가 큽니다. 하지만 점수 자체는 강하지 않습니다. 특히 AUPRC가 낮기 때문에, 데이터 확장, positive scene 확보, label 정교화, map/context feature 추가가 다음 개선 우선순위입니다.

## 중요한 파일

```text
src/imptc_dataset.py                 # IMPTC track.json -> Scene/Frame/ObjectState
src/graph_builder.py                 # graph node/edge/blind-zone 생성
src/label_builder.py                 # blind-zone label 생성
src/gnn_dataset.py                   # graph JSON -> 학습 sample
src/gnn_models.py                    # EIGAT, MR-GCN, Temporal GAT 모델
src/training_utils.py                # metric, seed, 학습 helper

scripts/run_imptc_sets_experiments.sh       # 전체 set 실험 runner
scripts/run_imptc_set01_set02_experiments.sh # set01+set02 호환 wrapper
scripts/download_imptc_sequences.sh         # 공식 IMPTC archive 다운로드
scripts/preprocess_sample.py                # graph JSON 전처리
scripts/create_scene_split.py               # scene-level split 생성
scripts/train_single_frame_gat.py           # EIGAT/MR-GCN single-frame trainer
scripts/train_mrgcn.py                      # MR-GCN wrapper
scripts/train_temporal_gat.py               # temporal trainer
scripts/train_stgcn.py                      # ST-GCN wrapper
scripts/aggregate_imptc_experiment_results.py
scripts/plot_imptc_experiment_results.py
scripts/plot_imptc_research_visualizations.py
```

자세한 set runner 설명은 아래 문서에도 정리되어 있습니다.

```text
docs/imptc_set_runner.md
docs/imptc_set01_set02_experiments.md
```

## Git 작업 규칙

아래 폴더는 Git에 올리지 않습니다.

```text
data/
outputs/
*.pkl
```

코드와 문서만 commit합니다.

```bash
git status
git add README.md docs src scripts
git commit -m "Add configurable IMPTC experiment runner"
git push origin feature/eigcn
```

Colab 또는 다른 PC에서 최신 코드를 받으려면:

```bash
git pull
```

그 다음 원하는 set으로 `scripts/run_imptc_sets_experiments.sh`를 다시 실행하면 됩니다.
