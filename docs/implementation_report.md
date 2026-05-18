# BlindSpotter — MR-GCN 구현 보고서

> **Task**: 교차로 사각지대(Blind Zone)에 숨어 있는 PM(Personal Mobility: 킥보드/자전거)이  
> 향후 **3초 이내에 도로로 진입할 위험**이 있는지를 장면 그래프(scene graph)로 이진 분류한다.

---

## 1. 프로젝트 개요

| 항목 | 내용 |
|------|------|
| 데이터셋 | IMPTC (Intersection Multi-modal Perception and Trajectory Collection) |
| 문제 유형 | 이진 분류 (label=1: 3초 내 출현 위험, label=0: 비위험) |
| 모델 | MR-GCN (Multi-Relational Graph Convolutional Network) |
| 참고 논문 | Liu et al., *"Learning From Interaction-Enhanced Scene Graph for Pedestrian Collision Risk Assessment"*, IEEE Transactions on Intelligent Vehicles, Vol.8, No.9, 2023 |
| 구현 파일 | `notebooks/MRGCN_Training.ipynb` (로컬), `notebooks/MRGCN_Colab.ipynb` (Colab) |

### 배경 및 문제 정의

IMPTC 데이터셋은 원래 VRU(Vulnerable Road User) 클래스(사람, 자전거, 유모차)만 포함하며 차량 데이터가 없다.  
이를 보완하기 위해 **샘플 차량 데이터를 합성**하여 추가하였고, 전처리된 결과를 `data/graph_dataset.pkl`에 저장하였다.

---

## 2. 데이터셋 구조

### 2.1 분할 통계

| Split | 전체 샘플 | Positive (위험) | Negative (비위험) | 양성 비율 |
|-------|-----------|-----------------|-------------------|----------|
| Train | 28,190    | ~11,500         | ~16,700           | ~40.8%   |
| Val   | 1,928     | -               | -                 | -        |
| Test  | 7,822     | -               | -                 | -        |

### 2.2 각 샘플 필드

```
scene_id      : 씬 식별자 (예: "0000_20230322_081506")
split         : 'train' / 'val' / 'test'
label         : 0 또는 1
x             : 노드 피처 행렬 [N, NODE_DIM]   (NODE_DIM=8)
edge_index    : 간선 인덱스 [2, E]
edge_attr     : 간선 피처 [E, EDGE_DIM]         (EDGE_DIM=5)
edge_type     : 간선 관계 타입 (문자열 리스트)
node_types    : 각 노드의 타입 ('ego', 'vehicle', 'pm', ...)
bz_node_idx   : 사각지대 노드(PM)의 로컬 인덱스
occ_node_idx  : 가리는 물체(occluder) 노드 인덱스
meta
  dist_to_ego   : PM과 에고 차량 사이의 거리 (m)
  hidden_sec    : PM이 사각지대에 숨어 있던 시간 (초) — EWT 분석에 사용
  ego_speed     : 에고 차량 속도 (m/s)
  pm_speed      : PM 속도 (m/s)
```

### 2.3 관계(Relation) 타입 — 4종

| ID | 이름 | 의미 |
|----|------|------|
| 0 | `spatial_near` | 두 객체가 공간적으로 가까움 |
| 1 | `occludes` | 한 객체가 다른 객체를 가림 |
| 2 | `potential_conflict` | 충돌 가능성이 있는 경로 교차 |
| 3 | `blind_zone_relation` | PM이 사각지대에 존재하는 관계 |

---

## 3. 모델 아키텍처: BlindSpotterRGCN

참고 논문(Liu et al. 2023)의 핵심 기여를 모두 반영하였다.

```
입력 노드 피처 x  [N, 8]
        ↓
[Paper Eq.1] Degree Embedding
  h = x + in_deg_emb(in_deg) + out_deg_emb(out_deg)   [N, 8]
        ↓
RGCNConv Layer 1 (8 → 64) + BatchNorm + ReLU + Dropout(0.3)
        ↓
RGCNConv Layer 2 (64 → 64) + BatchNorm + ReLU + Dropout(0.3)
        ↓
RGCNConv Layer 3 (64 → 64) + BatchNorm + ReLU
        ↓
  ┌─────────────────────┐
  │ BZ-node embed [B,64]│  ← bz_idx로 사각지대 노드 임베딩 추출
  │ + Global Mean Pool  │  ← 전체 씬 그래프 평균 임베딩 [B,64]
  └─────────────────────┘
        concat → [B, 128]
        ↓
Linear(128→64) + ReLU + Dropout
        ↓
Linear(64→1) → logit → sigmoid → 위험 확률
```

### 3.1 논문 기여 반영 내역

#### (1) Degree Embedding (논문 Eq. 1)
각 노드의 in-degree / out-degree를 임베딩 테이블로 변환하여 노드 피처에 더한다.  
그래프 내 구조적 위치 정보(얼마나 많이 연결되어 있는가)를 피처로 인코딩하는 효과가 있다.

```python
h = x + in_deg_emb(in_deg) + out_deg_emb(out_deg)
```

> 논문 보고 효과: Accuracy +2%, AUC +1.4%

#### (2) 3-layer MR-GCN
논문은 2레이어 대신 3레이어를 사용한다. 레이어를 깊게 쌓을수록 더 멀리 있는 노드의 정보까지 집약된다.

#### (3) Basis Decomposition
관계(relation)별 가중치 행렬을 직접 학습하면 파라미터 수가 `num_relations × 매트릭스 크기`만큼 늘어난다.  
Basis Decomposition은 이를 공유 기저(shared bases) `V_b`로 표현하여 파라미터를 줄인다:

```
W_r = Σ_b ( a_rb · V_b )

V_b : 공유 기저 행렬 (N_BASES=4개)
a_rb : 관계 r의 기저 b에 대한 계수 (학습 파라미터)
```

#### (4) Combined Readout
- **BZ-node 임베딩**: 사각지대에 숨어 있는 PM 노드의 임베딩 → 국소적 위험 맥락
- **Global Mean Pool**: 씬 전체의 평균 임베딩 → 전역적 씬 맥락
- 두 가지를 **concat**하여 분류기에 입력

### 3.2 하이퍼파라미터

| 파라미터 | 값 | 비고 |
|----------|-----|------|
| Hidden dim | 64 | |
| Dropout | 0.3 | |
| N_bases | 4 | Basis decomposition |
| Max degree | 16 | Degree embedding 클램핑 상한 |
| Batch size | 64 | 논문 동일 |
| Learning rate | 1e-4 | 논문은 1e-5; 데이터셋 규모 고려해 상향 |
| Weight decay | 5e-4 | 논문 Eq.5 동일 |
| Epochs | 50 | |
| Scheduler | CosineAnnealingLR | T_max=50 |

---

## 4. 학습 설정

### 4.1 손실 함수: BCEWithLogitsLoss + pos_weight

클래스 불균형(Positive ~40.8%, Negative ~59.2%) 보정을 위해 pos_weight를 적용한다.

```python
pos_weight = n_neg / n_pos   # ≈ 1.45
criterion  = BCEWithLogitsLoss(pos_weight=pos_weight)
```

pos_weight = k이면 양성 샘플의 loss가 k배 더 크게 반영되어,  
모델이 양성(위험) 케이스를 더 적극적으로 감지하도록 유도한다.

### 4.2 최적화

```python
optimizer = Adam(lr=1e-4, weight_decay=5e-4)
scheduler = CosineAnnealingLR(T_max=50)
clip_grad_norm_(model.parameters(), max_norm=1.0)   # 그래디언트 폭발 방지
```

### 4.3 모델 저장 기준

Validation AUPRC가 가장 높은 에포크의 체크포인트를 `best_mrgcn.pt`로 저장한다.  
(AUROC가 아닌 AUPRC를 기준으로 사용하는 이유: 불균형 데이터에서 AUPRC가 더 엄격한 지표)

---

## 5. 평가 지표 정의 및 해석

### 5.1 AUROC (Area Under ROC Curve)

**ROC 커브**: FPR(False Positive Rate)을 X축, TPR(Recall)을 Y축으로 그린 곡선.  
각 임계값(threshold)별로 (FPR, TPR) 점을 찍어 연결한 것.

- AUROC = 0.5 → 랜덤 분류기 수준
- AUROC = 1.0 → 완벽한 분류기
- **해석**: "임의의 양성 샘플이 임의의 음성 샘플보다 더 높은 예측 확률을 받을 확률"

> 적합한 상황: 양성/음성 비율이 균형 잡혀 있을 때. 불균형이 심하면 낙관적으로 보일 수 있음.

### 5.2 AUPRC (Area Under Precision-Recall Curve)

**PR 커브**: Recall을 X축, Precision을 Y축으로 그린 곡선.

- AUPRC = 양성 비율(prior) → 랜덤 수준
- AUPRC = 1.0 → 완벽
- **해석**: 모델이 양성을 얼마나 정확하게, 얼마나 많이 잡아내는가의 종합 점수

> **불균형 데이터에서 AUROC보다 엄격하고 신뢰할 수 있는 지표**

### 5.3 F1 Score

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

- Precision: 양성으로 예측한 것 중 실제 양성의 비율 (오탐 억제)
- Recall: 실제 양성 중 맞게 잡아낸 비율 (누락 억제)
- F1은 두 지표의 조화 평균

**threshold 선택 방법 (팀 통일 필요):**

세 모델(GAT / ST-GCNN / MR-GCN)을 공정하게 비교하려면 threshold를 반드시 **val set 기준**으로 선택해야 한다.

```
1. val set에서 F1이 최대가 되는 threshold 선택
2. 그 threshold를 고정 → test set에서 F1 평가
```

> ⚠️ 현재 코드(`threshold_analysis` 셀)는 **test set에서** threshold를 sweep해서 best를 고르고 있어 데이터 leakage 문제가 있다.  
> 팀 전체가 같은 방식으로 통일한 후 코드를 수정할 것.

### 5.4 Recall@FPR (FPR 제약 하에서의 재현율)

ROC 커브에서 특정 FPR 지점에서의 TPR(=Recall) 값.

| 지표 | 의미 |
|------|------|
| Recall@FPR=0.05 | "거짓 경고 5% 이하를 유지하면서 잡을 수 있는 위험 비율" |
| Recall@FPR=0.10 | "거짓 경고 10% 이하를 유지하면서 잡을 수 있는 위험 비율" |

> 자율주행/안전 시스템에서 중요한 지표: 실제 운용 중 허용 가능한 오탐률을 먼저 정하고, 그 제약 내에서 얼마나 많은 위험을 탐지하는지 측정함.

### 5.5 Early Warning Time (EWT)

**⚠️ 팀 통일 필요: EWT 계산 방식이 두 가지 정의로 혼재함**

#### 현재 코드 방식 (hidden_sec 기반)

```
EWT_proxy = TP 샘플의 hidden_sec 평균
```

`hidden_sec`은 PM이 사각지대에 숨은 시점부터 현재 프레임까지의 경과 시간(초).

| 구분 | 해석 |
|------|------|
| TP의 mean hidden_sec **낮을수록** 좋음 | PM이 막 숨었을 때(초기)에도 위험을 탐지함 = 조기 경보 |
| FN의 mean hidden_sec이 높으면 | 오래 숨어 있던 케이스를 놓침 = 만성 누락 |
| TP mean < FN mean | 모델이 숨은 초기 단계에서 더 잘 탐지함 (이상적) |

#### 팀 문서(metric_info.md) 정의

```
EWT = t_event - t_detect
  t_event  : PM이 실제로 도로에 출현한 시각
  t_detect : 모델이 처음으로 위험(threshold 이상)을 예측한 시각
```

- **값이 클수록 좋음** (1.3초 = 등장 1.3초 전에 감지)
- 현재 코드와 **방향이 반대** (hidden_sec은 낮을수록 좋음)

> 팀 내 GAT / ST-GCNN이 어떤 방식으로 EWT를 계산하는지 확인 후 통일할 것.  
> 통일 방향이 정해지면 코드 수정 예정.

---

## 6. 출력 파일

학습 후 `data/` 디렉토리(로컬) 또는 Google Drive `outputs/`(Colab)에 저장되는 파일:

| 파일명 | 내용 |
|--------|------|
| `best_mrgcn.pt` | Val AUPRC 기준 최고 성능 체크포인트 (model state_dict) |
| `mrgcn_final.pt` | 모델 가중치 + 설정(config) + 지표(metrics) 통합 저장 |
| `mrgcn_metrics.json` | 테스트 지표를 JSON으로 저장 (가독성 높음) |
| `training_curves.png` | Loss / AUROC / AUPRC 학습 곡선 (Train vs Val) |
| `test_pr_roc.png` | PR 커브 + ROC 커브 (FPR 동작점 마커 포함) |
| `ewt_analysis.png` | hidden_sec 분포 히스토그램 + TP vs FN 박스플롯 |
| `threshold_analysis.png` | 임계값별 F1 / Recall 곡선 |

### mrgcn_metrics.json 예시 구조

```json
{
  "test_auroc": 0.XXXX,
  "test_auprc": 0.XXXX,
  "test_f1": 0.XXXX,
  "recall_at_fpr05": 0.XXXX,
  "recall_at_fpr10": 0.XXXX,
  "ewt_tp_mean": X.XXX,
  "ewt_fn_mean": X.XXX,
  "best_f1_threshold": 0.XX,
  "best_f1_value": 0.XXXX,
  "recall85_threshold": 0.XX
}
```

---

## 7. 결과 해석 가이드

### 7.1 숫자가 출력될 때 확인 순서

```
1. Val AUPRC가 에포크가 지남에 따라 꾸준히 올라가는가?
   → 올라가다 꺾이면 그 지점이 최적 에포크 (early stopping 고려)

2. Train loss vs Val loss 차이가 커지는가?
   → 크게 벌어지면 과적합(overfitting) → Dropout 높이거나 weight_decay 증가

3. 최종 Test AUPRC가 0.5보다 얼마나 높은가?
   → 0.5 = 랜덤 수준. 0.7+ 이면 유의미, 0.85+ 이면 우수

4. Recall@FPR=0.05 값이 Recall@FPR=0.10보다 낮다
   → 당연함. FPR 허용치가 낮을수록 잡을 수 있는 위험이 줄어듦

5. EWT: TP mean hidden_sec이 FN mean보다 낮은가?
   → 낮으면 "초기 감지가 잘 됨" (바람직)
   → 비슷하거나 높으면 모델이 숨은 시간에 무관하게 예측
```

### 7.2 Threshold 조정 전략

기본 threshold = 0.5는 F1을 어느 정도 균형 있게 유지하지만,  
안전 시스템에서는 **Recall 우선** (놓치는 것이 오탐보다 위험)이 중요하다.

- `threshold_analysis.png`의 "Recall vs Threshold" 그래프를 참고
- Recall >= 0.85를 유지하는 최대 threshold를 `recall85_threshold`로 저장
- 실제 배포 시 이 threshold를 사용하면 위험 누락을 최소화할 수 있음

---

## 8. 노트북 사용 방법

### 로컬 (MRGCN_Training.ipynb)

```
1. 가상환경 활성화: .venv\Scripts\activate
2. jupyter notebook notebooks/MRGCN_Training.ipynb
3. 셀 순서대로 실행 (Shift+Enter)
4. DATA_PKL = Path('../data/graph_dataset.pkl') 경로 확인
5. 결과물은 ../data/ 폴더에 저장됨
```

### Google Colab (MRGCN_Colab.ipynb)

두 가지 데이터 소스 모드를 지원한다. 셀 0번에서 `DATA_SOURCE`를 선택한다.

#### 모드 A: PKL 파일 사용 (빠름)

```
DATA_SOURCE = 'pkl'

1. Google Drive에 graph_dataset.pkl 업로드
2. 노트북 업로드 후 Runtime → Change runtime type → T4 GPU 선택
3. 셀 0번에서 DRIVE_DIR 경로 수정
   예: DRIVE_DIR = '/content/drive/MyDrive/BlindSpotter'
4. 순서대로 실행 (총 19개 셀)
5. 결과물은 Drive의 outputs/ 폴더에 자동 저장됨
```

#### 모드 B: 원본 IMPTC 폴더 직접 사용 (raw)

```
DATA_SOURCE = 'raw'

1. Google Drive에 data/ 폴더 통째로 업로드
   (각 시퀀스 폴더 안에 vehicles/ 와 vrus/ 가 있는 구조)
2. 셀 0번에서 RAW_DATA_DIR 경로 수정
   예: RAW_DATA_DIR = '/content/drive/MyDrive/BlindSpotter/data'
3. 실행 시 자동으로 그래프 샘플 빌드 → 학습 → 평가 진행
4. CACHE_PKL 경로를 지정하면 빌드 결과를 pkl로 저장 (다음 실행 시 재사용 가능)
```

> raw 모드는 src/ 폴더 없이 Colab에서 단독 실행 가능 (모든 전처리 코드 인라인 포함).  
> 빌드 시간이 추가로 수 분 소요되므로 반복 실행 시 CACHE_PKL을 활용할 것.

---

## 9. 향후 개선 방향

### Temporal Graph Transformer (논문 제안)

현재 모델은 각 프레임을 독립적으로 처리한다.  
시계열 정보(연속된 프레임의 변화 추이)를 반영하면 성능이 향상될 수 있다.

```
frame_t-2 → MR-GCN → h_{t-2} ─┐
frame_t-1 → MR-GCN → h_{t-1} ─┤→ Transformer Encoder → classifier
frame_t   → MR-GCN → h_t     ─┘
```

> 논문 예상 효과: 3프레임 이상 시퀀스에서 AUROC +1~2%

### 기타 실험 아이디어

| 아이디어 | 예상 효과 |
|----------|-----------|
| SAGPool을 추가하여 불필요한 노드 제거 후 readout | 노이즈 감소, 속도 향상 |
| 에지 피처(edge_attr)를 메시지 패싱에 포함 | 거리/속도 관계 더 잘 반영 |
| Focal Loss 사용 (pos_weight 대체) | 어려운 샘플에 더 집중 |
| Ensemble (GAT + MR-GCN) 결합 | 다양한 관점 통합 |

---

## 10. 참고

- **원본 논문**: Liu et al., IEEE Transactions on Intelligent Vehicles, Vol.8, No.9, Sep. 2023  
  *"Learning From Interaction-Enhanced Scene Graph for Pedestrian Collision Risk Assessment"*
- **데이터셋**: [IMPTC Dataset](https://github.com/FranciscoMartinezGarcia/IMPTC)
- **PyTorch Geometric RGCNConv**: https://pytorch-geometric.readthedocs.io/
