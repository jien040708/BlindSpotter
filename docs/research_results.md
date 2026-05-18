# Blind-Zone Risk Classifier Research Results

## Summary

현재 가장 좋은 제출 후보는 canonical feature contract로 학습한 single-frame EIGAT입니다.

```text
checkpoint:
outputs/models/single_frame_gat_canonical_stable_1layer_5ep.pt

graph_dataset.pkl test:
AUPRC 0.592
AUROC 0.677
F1@0.5 0.620
best-F1 0.628
```

## Dataset Alignment

초기 `graph_dataset.pkl`은 node 8개, edge 5개 feature를 사용했고, IMPTC canonical graph는 node 14개, edge 6개 feature를 사용했습니다. 이 차이 때문에 IMPTC evaluation에서 domain/feature mismatch가 컸습니다.

해결한 내용:

```text
1. graph_dataset.pkl 로더 추가
2. feature alias alignment 추가
3. train split 기준 feature normalization 추가
4. generated pkl을 IMPTC canonical 14/6 feature contract로 변환
5. TTC는 log1p(min(TTC, 30s))로 bounded transform 적용
6. train에서 constant인 feature는 std=1 처리하여 eval 폭발 방지
```

생성한 canonical dataset:

```text
outputs/models/graph_dataset_canonical.pkl
```

## Step 1: Single-Frame GAT

### Main Dataset Test

| model | AUPRC | AUROC | F1@0.5 | best-F1 |
| --- | ---: | ---: | ---: | ---: |
| aligned 8-feature EIGAT | 0.570 | 0.673 | 0.601 | 0.624 |
| canonical EIGAT | 0.592 | 0.677 | 0.620 | 0.628 |
| canonical no-edge GAT | 0.591 | 0.675 | 0.592 | 0.621 |
| canonical + IMPTC augmentation | 0.560 | 0.652 | 0.532 | 0.617 |

결론:

```text
canonical EIGAT이 main dataset test에서 가장 균형이 좋음.
edge feature attention은 F1 기준으로 no-edge보다 이득이 있음.
```

## Step 2: Spatio-Temporal GAT

IMPTC canonical graph 4개 sequence만 temporal training에 사용할 수 있었습니다. pkl은 frame order/timestamp가 없어 temporal sample로 쓰기 어렵습니다.

Scene split:

```text
train: 0000_20230322_081506, 0002_20230523_111106
val:   0001_20230523_105516
test:  0003_20230523_111514
```

| temporal setting | test AUPRC | test AUROC | F1@0.5 | best-F1 |
| --- | ---: | ---: | ---: | ---: |
| history 5, horizon +1 | 0.058 | 0.387 | 0.000 | 0.139 |
| history 10, horizon +5 | 0.057 | 0.360 | 0.000 | 0.156 |

결론:

```text
Temporal model implementation works, but current IMPTC sample count is too small for stable scene-level generalization.
Train F1 rises, but held-out scene test remains weak.
```

## Step 3: Expert-Informed Attention

Edge feature attention uses:

```text
distance
relative_velocity_x
relative_velocity_y
relative_heading
time_to_collision
visibility_blocked
```

Ablation result:

```text
canonical EIGAT:
F1@0.5 0.620

canonical no-edge GAT:
F1@0.5 0.592
```

결론:

```text
expert-informed edge bias improves main-test F1 after feature contract alignment.
```

## IMPTC Generalization

Held-out IMPTC scene `0003_20230523_111514` remains difficult.

| model | AUPRC | AUROC | F1@0.5 | best-F1 |
| --- | ---: | ---: | ---: | ---: |
| aligned 8-feature EIGAT | 0.070 | 0.480 | 0.138 | 0.155 |
| canonical EIGAT | 0.169 | 0.500 | 0.143 | 0.395 |
| canonical no-edge GAT | 0.080 | 0.502 | 0.143 | 0.161 |
| canonical + IMPTC augmentation | 0.077 | 0.502 | 0.000 | 0.155 |

해석:

```text
canonicalization improves AUPRC and possible threshold-tuned F1 on IMPTC held-out scene,
but AUROC remains near random because only 4 IMPTC sequences are available and scene distribution varies strongly.
```

## Final Recommendation

Main scoring 후보:

```text
outputs/models/single_frame_gat_canonical_stable_1layer_5ep.pt
```

보고서/발표에서 주장할 수 있는 내용:

```text
1. Generated graph dataset is learnable by GNN.
2. Canonical feature alignment improves Step 1 performance.
3. Expert-informed edge attention improves F1 after alignment.
4. Spatio-temporal GAT is implemented, but current IMPTC sequence count is insufficient for robust scene-level temporal generalization.
5. The biggest remaining issue is not model architecture but real/generated domain gap and limited IMPTC positive scenes.
```

다음으로 성능을 실제로 올리려면:

```text
1. more IMPTC sequences
2. scene-level train/val/test split
3. generated scenes rebuilt directly with canonical 14/6 feature contract
4. positive scene oversampling or focal loss
5. temporal training after enough real sequence coverage
```

## Additional IMPTC Set 01 Experiment

추가 IMPTC sequence 확보를 위해 official sequence dataset 중 `imptc_set_01.tar.gz`를 다운로드했습니다.

```text
downloaded chunk:
data/downloads/imptc_set_01.tar.gz

extracted root:
data/imptc_sequences

graph output:
outputs/graphs_imptc_set01

scene split:
outputs/splits/imptc_set01_scene_split.json
```

`set_01`에는 50개 sequence가 포함되어 있고, 전처리 결과 positive blind label이 있는 scene은 9개였습니다.

Scene-level split target 수:

| split | frame samples | blind-zone targets | positives |
| --- | ---: | ---: | ---: |
| train | 12,537 | 53,979 | 709 |
| val | 2,442 | 10,062 | 105 |
| test | 2,842 | 9,579 | 275 |

### Single-Frame GAT on Real IMPTC Set 01

| training setup | test AUPRC | test AUROC | F1@0.5 | best-F1 |
| --- | ---: | ---: | ---: | ---: |
| full train, pos_weight | 0.021 | 0.376 | 0.003 | 0.064 |
| negative sampling, pos_weight=75 | 0.093 | 0.790 | 0.000 | 0.225 |
| negative sampling, pos_weight=1 | 0.096 | 0.777 | 0.000 | 0.196 |

해석:

```text
more IMPTC sequences clearly improve ranking metrics compared with the 4-sequence sample,
but threshold 0.5 is not calibrated for rare-event prediction.
The next required step is threshold calibration / focal loss / calibrated precision-recall operating point.
```

### Temporal Set 01 Status

Temporal GAT on set01 was started with:

```text
history=5
prediction_horizon=1
train=12,472 temporal samples
val=2,431 temporal samples
test=2,819 temporal samples
```

The current implementation processes graphs one-by-one and took about 5 minutes per epoch on CPU. The run was stopped after confirming that a full temporal training pass requires batching/caching optimization.
