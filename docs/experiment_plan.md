# MR-GCN / EIGCN / STGCN 실험 분기 계획

이 프로젝트에서는 세 명이 각각 다른 GNN 계열 모델을 맡아 비교합니다.

```text
MR-GCN
EIGCN / Expert-informed GCN
STGCN / Spatio-Temporal GCN
```

중요한 원칙은 **raw preprocessing을 모델마다 다르게 만들지 않는 것**입니다.

모델마다 node/edge를 완전히 따로 만들면, 나중에 성능 차이가 모델 때문인지 전처리 차이 때문인지 비교하기 어렵습니다.

따라서 아래 구조를 사용합니다.

```text
IMPTC raw data
  ↓
공통 canonical graph dataset
  ↓
model adapter
  ├── MR-GCN input
  ├── EIGCN input
  └── STGCN input
  ↓
각자 모델 학습
```

## 어디까지 공통으로 만들고 branch를 나눌까?

공통 dataset은 여기까지 고정합니다.

```text
outputs/graphs/*.json
```

공통 graph JSON에는 다음이 들어 있어야 합니다.

```text
node_ids
node_types
x
edge_index
edge_attr
edge_type
blind_node_indices
blind_y
node_feature_names
edge_feature_names
```

즉, **blind-zone node까지 생성하고, blind_y label까지 붙인 graph JSON**이 공통 출발점입니다.

branch는 이 이후에 나누는 것이 좋습니다.

```text
main
  └── 공통 preprocessing / graph JSON contract

feature/mrgcn
  └── MR-GCN용 relation graph adapter + model

feature/eigcn
  └── expert edge feature / attention-bias model

feature/stgcn
  └── temporal window dataset + temporal model
```

## 모델별로 다르게 해도 되는 것

모델 input adapter는 다르게 만들어도 됩니다.

### MR-GCN

MR-GCN은 edge relation type이 중요합니다.

공통 graph의 `edge_type`을 relation id로 바꿉니다.

```text
spatial_near
potential_conflict
occludes
blind_zone_relation
temporal_next
```

MR-GCN 담당자는 다음을 주로 수정합니다.

```text
src/model_adapters.py
future src/models/mrgcn.py
future scripts/train_mrgcn.py
```

### EIGCN

EIGCN은 expert-informed edge feature가 중요합니다.

공통 graph의 `edge_attr`를 그대로 사용하거나, 추가 expert feature를 만듭니다.

현재 edge feature:

```text
distance
relative_velocity_x
relative_velocity_y
relative_heading
time_to_collision
visibility_blocked
```

EIGCN 담당자는 다음을 주로 수정합니다.

```text
src/model_adapters.py
src/gnn_models.py
future src/models/eigcn.py
future scripts/train_eigcn.py
```

### STGCN

STGCN은 시간 window가 중요합니다.

여러 frame을 묶어서 sequence sample을 만듭니다.

```text
t-4, t-3, t-2, t-1, t
  ↓
current blind_y 예측
```

STGCN 담당자는 다음을 주로 수정합니다.

```text
src/model_adapters.py
src/gnn_dataset.py
future src/models/stgcn.py
future scripts/train_stgcn.py
```

## 지금 기준 recommended branch 순서

현재 로컬은 GitHub와 diverged 상태입니다.

```text
local main: ahead 1
origin/main: behind 4
```

그래서 바로 push하거나 branch를 원격에 올리기보다, 먼저 GitHub의 최신 변경을 확인하고 충돌을 정리해야 합니다.

권장 순서:

```bash
git fetch origin
git status
git pull --rebase origin main
```

충돌이 나면 특히 아래 파일을 조심해서 해결합니다.

```text
notebooks/IMPTC_BlindZone_GraphML.ipynb
README.md
```

그 다음 branch를 만듭니다.

```bash
git checkout -b feature/mrgcn
git checkout -b feature/eigcn
git checkout -b feature/stgcn
```

팀원별로 한 branch씩 담당하면 됩니다.

## 비교를 위해 고정해야 할 것

모델 비교를 공정하게 하려면 아래는 고정합니다.

```text
same graph JSON files
same train/val split seed
same blind_y labels
same evaluation metrics
same max_sequences / max_frames / frame_stride setting
```

추천 기본 설정:

```text
max_sequences = 4
max_frames = 200
frame_stride = 10
history = 5
seed = 7
```

공통 metric:

```text
accuracy
precision
recall
F1
positive_rate
```

이 연구에서는 false negative가 위험하므로, accuracy보다 **recall과 F1**을 더 중요하게 봅니다.

## feature/eigcn: graph_dataset.pkl 기준 Step 1

가공된 pickle dataset을 바로 사용할 때는 `feature/eigcn` 브랜치에서 아래처럼 실행합니다.

```bash
KMP_DUPLICATE_LIB_OK=TRUE python scripts/train_single_frame_gat.py \
  --graphs graph_dataset.pkl \
  --output outputs/models/single_frame_gat.pt \
  --epochs 20 \
  --hidden-dim 64 \
  --heads 2 \
  --layers 2
```

현재 `graph_dataset.pkl`은 이미 `train`, `val`, `test` split을 포함하므로, 스크립트는 random split을 다시 만들지 않고 pickle의 split을 그대로 사용합니다.

디버그용 smoke test는 다음처럼 작게 돌릴 수 있습니다.

```bash
KMP_DUPLICATE_LIB_OK=TRUE python scripts/train_single_frame_gat.py \
  --graphs graph_dataset.pkl \
  --output outputs/models/single_frame_gat_smoke.pt \
  --epochs 1 \
  --hidden-dim 16 \
  --heads 2 \
  --layers 1 \
  --max-train-samples 256 \
  --max-val-samples 128 \
  --max-test-samples 128
```

pickle의 현재 feature는 다음과 같습니다.

```text
node_feature_names:
x, y, vx, vy, heading, type_id, speed, is_occluder

edge_feature_names:
distance, rel_vx, rel_vy, rel_heading, visibility_blocked
```

## Dataset Alignment Fix

`graph_dataset.pkl`과 IMPTC canonical graph JSON은 서로 다른 feature contract에서 출발합니다.

```text
graph_dataset.pkl:
node 8 features, edge 5 features

IMPTC canonical graph JSON:
node 14 features, edge 6 features
```

따라서 raw tensor를 그대로 비교하지 않고, 학습 checkpoint의 feature contract에 맞춰 IMPTC graph를 정렬합니다.

```text
object_type_id ↔ type_id
relative_velocity_x ↔ rel_vx
relative_velocity_y ↔ rel_vy
relative_heading ↔ rel_heading
```

또한 train split에서 feature normalization 통계를 fit하고, val/test/eval graph에 같은 통계를 적용합니다. 이때 categorical/binary feature는 normalization하지 않습니다.

```text
not normalized:
type_id, object_type_id, visibility, is_occluder,
is_vulnerable_road_user, visibility_blocked
```

재평가는 다음처럼 합니다.

```bash
KMP_DUPLICATE_LIB_OK=TRUE python scripts/evaluate_single_frame_gat.py \
  --checkpoint outputs/models/single_frame_gat_aligned_1layer_3ep.pt \
  --graphs graph_dataset.pkl \
  --split test

KMP_DUPLICATE_LIB_OK=TRUE python scripts/evaluate_single_frame_gat.py \
  --checkpoint outputs/models/single_frame_gat_aligned_1layer_3ep.pt \
  --graphs outputs/graphs_validation_all \
  --split all
```
