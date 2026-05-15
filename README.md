# BlindSpotter

BlindSpotter는 전동킥보드와 같은 scooter-like VRU(Vulnerable Road User)가 짧은 미래 시간 안에 blind zone에서 등장할 위험을 예측하기 위한 전처리 프로토타입입니다.

현재 연구 질문은 다음과 같습니다.

```text
P(VRU emerges from blind-zone within 3 seconds)
```

즉, 이미 보이는 전동킥보드를 탐지하는 것이 아니라, **현재 보이지 않는 blind-zone 자체가 위험한지**를 graph 기반으로 예측하는 것이 목표입니다.

처음부터 전체 IMPTC dataset을 사용하지 않고, 팀원 모두가 빠르게 실행하고 디버깅할 수 있도록 IMPTC sample dataset부터 사용합니다.

## 팀 작업 방식

이 프로젝트에서는 아래 규칙을 따릅니다.

```text
GitHub = 코드 공유
Colab = 실험 실행
data/ = 각자 로컬 또는 Colab에서 다운로드, GitHub에 올리지 않음
outputs/ = 실행 결과물, GitHub에 올리지 않음
```

dataset 파일, zip/tar 압축 파일, 생성된 graph, 생성된 figure는 GitHub에 올리지 않습니다.  
이 파일들은 `.gitignore`에 의해 자동으로 무시되도록 설정되어 있습니다.

## 어떤 파일을 수정해야 하나요?

대부분의 프로젝트 코드는 `src/`와 `scripts/` 안에 있습니다.

```text
src/
├── imptc_dataset.py    # IMPTC track을 읽어서 Scene/Frame/ObjectState로 변환
├── dataset.py          # 공통 데이터 구조: Scene, Frame, ObjectState
├── graph_builder.py    # graph node, edge, blind-zone node, graph feature 생성
├── label_builder.py    # risk label, blind-zone emergence label 생성
├── gnn_dataset.py      # graph JSON을 GAT 학습 sample로 변환
├── gnn_models.py       # Expert-informed GAT / Temporal GAT 모델
├── training_utils.py   # 학습 metric, seed 고정 helper
└── utils.py            # 여러 파일에서 함께 쓰는 helper 함수

scripts/
├── download_imptc_sample.sh  # IMPTC sample data 다운로드
├── inspect_dataset.py        # dataset 구조 확인
├── preprocess_sample.py      # sample data를 graph JSON으로 변환
├── visualize_sample.py       # top-view PNG 시각화 저장
├── validate_preprocessing.py # 생성된 graph JSON 검증
├── train_single_frame_gat.py # Step 1: single-frame GAT baseline 학습
└── train_temporal_gat.py     # Step 2~3: temporal + expert-informed GAT 학습

notebooks/
└── IMPTC_BlindZone_GraphML.ipynb  # scripts/src를 실행하는 Colab notebook
```

IMPTC 데이터를 읽는 방식을 바꾸고 싶다면:

```text
src/imptc_dataset.py
```

graph node, edge, blind-zone 후보 생성 방식을 바꾸고 싶다면:

```text
src/graph_builder.py
```

risk label 또는 blind-zone label을 바꾸고 싶다면:

```text
src/label_builder.py
```

전처리 실행 옵션이나 실행 흐름을 바꾸고 싶다면:

```text
scripts/preprocess_sample.py
```

아이디어를 빠르게 테스트하거나 그림을 보고 싶다면:

```text
notebooks/IMPTC_BlindZone_GraphML.ipynb
```

단, 중요한 로직은 notebook 안에만 두지 말고 나중에 `src/` 또는 `scripts/`로 옮겨야 합니다. 그래야 팀원 모두가 같은 코드를 재사용할 수 있습니다.

현재 Colab notebook은 전처리 함수를 직접 길게 들고 있지 않고, `scripts/`와 `src/`를 실행하도록 정리되어 있습니다. 따라서 전처리 로직을 바꾸려면 notebook보다 `src/` 또는 `scripts/` 파일을 먼저 수정하는 것이 좋습니다.

## Colab에서 처음 실행하는 방법

새 Colab notebook에서 아래 셀을 실행합니다.

```python
!git clone https://github.com/jien040708/BlindSpotter.git
%cd BlindSpotter
!pip install -r requirements.txt
!bash scripts/download_imptc_sample.sh
```

dataset 구조를 확인합니다.

```python
!python scripts/inspect_dataset.py --root data/sample
```

작은 전처리 테스트를 실행합니다.

```python
!python scripts/preprocess_sample.py \
  --root data/sample \
  --output outputs/graphs \
  --max-sequences 1 \
  --max-frames 120 \
  --frame-stride 10
```

top-view figure를 생성합니다.

```python
!python scripts/visualize_sample.py \
  --root data/sample \
  --output outputs/figures \
  --max-files 1 \
  --max-frames 20
```

생성된 graph가 전처리 형식에 맞는지 검증합니다.

```python
!python scripts/validate_preprocessing.py --graphs outputs/graphs --write-summary
```

Step 1 baseline인 single-frame GAT를 학습합니다.

```python
!python scripts/train_single_frame_gat.py \
  --graphs outputs/graphs \
  --epochs 10 \
  --output outputs/models/single_frame_gat.pt
```

Step 2~3 temporal expert-informed GAT를 학습합니다.

```python
!python scripts/train_temporal_gat.py \
  --graphs outputs/graphs \
  --history 5 \
  --epochs 10 \
  --output outputs/models/temporal_gat.pt
```

실행 결과는 Colab 안에서 아래 폴더에 생성됩니다.

```text
outputs/graphs/    # 나중에 GNN 입력으로 사용할 graph JSON 파일
outputs/figures/   # 결과 확인용 PNG 시각화 파일
```

이 결과물들은 GitHub에 push하지 않습니다.

## Colab에서 최신 코드 받기

다른 팀원이 GitHub에 코드를 올렸고, Colab에서 최신 코드를 받고 싶다면:

```python
%cd /content/BlindSpotter
!git pull
```

그 다음 필요한 script를 다시 실행하면 됩니다.

Colab 안에서 직접 파일을 수정할 수도 있지만, 초보자에게는 추천하지 않습니다.  
Colab에서 수정한 내용은 런타임이 초기화되면 사라질 수 있고, GitHub에 반영하려면 별도로 commit/push를 해야 합니다.

초보자에게 추천하는 방식은:

```text
1. 로컬 또는 Codex에서 코드 파일 수정
2. GitHub에 push
3. Colab에서 git pull
4. Colab에서 실행
```

## Git 초보자용 작업 흐름

작업을 시작하기 전에 항상 최신 코드를 받습니다.

```bash
git pull
```

자기 작업용 branch를 만듭니다.

```bash
git checkout -b feature/your-task-name
```

예시:

```bash
git checkout -b feature/imptc-loader
git checkout -b feature/blind-zone-label
git checkout -b feature/visualization
git checkout -b feature/gnn-model
```

코드를 수정한 뒤:

```bash
git status
git add src scripts notebooks README.md requirements.txt
git commit -m "Describe what you changed"
git push origin feature/your-task-name
```

그 다음 GitHub에서 Pull Request를 열고, 확인 후 `main` branch에 merge합니다.

아래 폴더는 commit하지 않습니다.

```text
data/
outputs/
```

이 폴더들은 `.gitignore`로 무시되지만, 실수로 강제로 추가하지 않도록 주의합니다.

## 팀원 역할 분담 예시

팀원 1: 데이터 로딩 / 전처리

```text
src/imptc_dataset.py
scripts/inspect_dataset.py
scripts/preprocess_sample.py
```

팀원 2: blind-zone / label

```text
src/graph_builder.py
src/label_builder.py
scripts/visualize_sample.py
```

팀원 3: 모델 / 실험

```text
notebooks/
future src/model.py
future scripts/train.py
```

## 로컬에서 실행하는 방법

Colab이 아니라 노트북/PC에서 실행하고 싶다면:

```bash
git clone https://github.com/jien040708/BlindSpotter.git
cd BlindSpotter
pip install -r requirements.txt
bash scripts/download_imptc_sample.sh
python scripts/inspect_dataset.py --root data/sample
python scripts/preprocess_sample.py --root data/sample --output outputs/graphs --max-sequences 1 --max-frames 120 --frame-stride 10
```

## 현재 파이프라인

현재 구현된 pipeline은 다음과 같습니다.

```text
1. IMPTC sample data 다운로드
2. dataset 구조 확인
3. vehicle / VRU trajectory 로드
4. 임시 reference vehicle 선택
5. frame 단위 graph 생성
6. blind-zone 후보 node 추가
7. heuristic blind-zone label 생성
8. graph JSON과 선택적 visualization 저장
9. single-frame GAT baseline 학습
10. temporal expert-informed GAT 학습
```

## Graph Output

각 scene graph는 기본적으로 JSON으로 저장됩니다.

IMPTC에는 자율주행 dataset처럼 명확한 ego vehicle이 없기 때문에, 현재 loader는 가장 길게 관측된 vehicle track을 임시 `reference_vehicle`로 선택합니다.

그 다음 graph builder가 주변 vehicle occluder 뒤쪽에 `occlusion_zone` node를 추가합니다.

각 frame graph에는 다음 정보가 들어갑니다.

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

`blind_y = 1`은 해당 blind-zone 근처에 미래 시간 window 안에서 scooter-like VRU가 등장했다는 뜻입니다.

현재 node feature는 다음과 같습니다.

```text
x, y, vx, vy, heading, object_type_id, distance_to_ego,
relative_angle_to_ego, visibility, is_occluder, is_vulnerable_road_user
```

현재 edge feature는 다음과 같습니다.

```text
distance, relative_velocity_x, relative_velocity_y,
relative_heading, time_to_collision, visibility_blocked
```

## 현재 한계

현재 blind-zone logic은 단순 heuristic입니다.  
주변 vehicle occluder 뒤쪽에 candidate blind-zone node를 배치하는 방식이며, 아직 정확한 geometric occlusion model은 아닙니다.

앞으로 할 일:

- road geometry와 map feature 추가
- traffic light / weather context 추가
- PyTorch Geometric `Data` 형식으로 직접 export
- 모델 성능 비교 및 ablation
- early-warning 성능 평가
- uncertainty / MDN head 추가
