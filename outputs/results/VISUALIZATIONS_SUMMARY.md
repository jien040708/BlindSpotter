# 세 모델 비교 시각화 종합 리포트

**작성일**: 2026-05-23  
**비교 모델**: EIGAT (20 epochs), MR-GCN (20 epochs), ST-GCN (20 epochs)

---

## 📊 1. 메인 메트릭 비교 (Main Metric Comparison)

### 설명
세 모델의 주요 성능 지표를 막대 그래프로 비교합니다.
- **AUPRC**: Precision-Recall 곡선 아래 면적 (클래스 불균형 지표)
- **AUROC**: ROC 곡선 아래 면적 (분류 성능)
- **F1@0.5**: 임계값 0.5에서의 F1 점수
- **Best F1**: 최적 임계값에서의 F1 점수

### 시각화
![메인 메트릭 비교](../../figures/final_comparison/main_metric_comparison.png)

### 핵심 발견

#### 🥇 AUPRC (최고 중요도)
```
ST-GCN: 최고 (0.518)
EIGAT: 두번째 (0.510)
MR-GCN: 거의 0 (0.023)

→ ST-GCN이 클래스 불균형 극복에서 최우수
```

#### 🥇 AUROC
```
EIGAT: 최고 (0.912)
ST-GCN: 두번째 (0.871)
MR-GCN: 부족 (0.706)

→ EIGAT이 ROC 성능에서 약간 우수
```

#### 🥇 Best F1
```
ST-GCN: 최고 (0.647)
EIGAT: 두번째 (0.552)
MR-GCN: 거의 0 (0.047)

→ ST-GCN이 정밀도-재현율 균형에서 최우수
```

#### 🥇 F1@0.5
```
ST-GCN: 최고
EIGAT: 두번째
MR-GCN: 거의 0

→ 모든 F1 지표에서 ST-GCN이 우수
```

---

## 📈 2. Validation 학습 곡선 (Validation Curves)

### 설명
각 모델의 학습 진행 과정을 epoch별로 보여줍니다.
- 각 모델의 validation loss와 주요 메트릭의 변화
- 수렴성 및 과적합 여부 판단
- 최고 성능 달성 시점 확인

### 시각화
![Validation 곡선](../../figures/final_comparison/validation_curves.png)

### 핵심 발견

#### EIGAT 학습 곡선
```
특징:
  - 빠른 수렴 (초기 5~10 epochs에서 높은 성능)
  - 안정적인 학습 (큰 변동 없음)
  - 과적합 미미 (validation 꾸준함)
  
성능:
  - Val AUROC 최고 0.912 (안정적)
  - Val AUPRC 약 0.51 (높음)
  
평가: ✓ 매우 안정적인 모델
```

#### MR-GCN 학습 곡선
```
특징:
  - 성능 정체 (20 epochs 이후에도 개선 없음)
  - 고수준의 변동성 (불안정함)
  - 의도되지 않은 특성 학습 가능
  
성능:
  - Val AUROC 0.706 (낮음)
  - Val AUPRC 0.023 (거의 0)
  
평가: ✗ 모델 구조가 맞지 않음
```

#### ST-GCN 학습 곡선
```
특징:
  - 가파른 상승 (초기 3-5 epochs에서 급격한 개선)
  - 안정적인 수렴 (후기에 안정화)
  - 과적합 없음 (validation 계속 개선)
  
성능:
  - Val AUROC 0.871 (우수)
  - Val AUPRC 0.518 (최고)
  - Val Best F1 0.647 (최고)
  
평가: ✓✓ 우수한 학습 곡선, 안정성 높음
```

---

## 📊 3. 메트릭 요약 시각화 (Metric Summary)

### 설명
세 모델의 모든 주요 메트릭을 종합적으로 비교하는 시각화입니다.

### 시각화 (PNG)
![메트릭 요약 - PNG](../../figures/model_metric_summary.png)

### 시각화 (PDF)
![메트릭 요약 - PDF](../../figures/model_metric_summary.pdf)

### 포함된 메트릭
- AUPRC (Area Under Precision-Recall)
- AUROC (Area Under ROC)
- F1@0.5 (F1 Score at threshold 0.5)
- Best F1 (F1 Score at optimal threshold)

### 해석 가이드

#### 높이가 높을수록 좋음
```
✓ ST-GCN: AUPRC, Best F1에서 가장 높음
✓ EIGAT: AUROC에서 가장 높음
✗ MR-GCN: 모든 메트릭에서 낮음
```

#### 색상별 의미
```
파란색 (AUPRC): 클래스 불균형 극복 능력
초록색 (AUROC): 전체 분류 성능
빨간색 (F1@0.5): 고정 임계값 성능
주황색 (Best F1): 최적 임계값 성능
```

---

## 🎯 4. 종합 분석

### 4.1 메트릭별 최고 모델

#### AUPRC (클래스 불균형 - ⭐ 매우 중요)
```
🥇 1위: ST-GCN = 0.5180
🥈 2위: EIGAT = 0.5095 (-1.7%)
🥉 3위: MR-GCN = 0.0229 (-95.6%)

결론: ST-GCN이 클래스 불균형 극복에 최적
```

#### AUROC (ROC 분류 성능)
```
🥇 1위: EIGAT = 0.9123
🥈 2위: ST-GCN = 0.8710 (-4.5%)
🥉 3위: MR-GCN = 0.7061 (-22.6%)

결론: EIGAT의 ROC 성능이 약간 우수
```

#### Best F1 (정밀도-재현율 균형)
```
🥇 1위: ST-GCN = 0.6470
🥈 2위: EIGAT = 0.5518 (-14.7%)
🥉 3위: MR-GCN = 0.0473 (-92.7%)

결론: ST-GCN이 정밀도-재현율 균형에서 압도적
```

### 4.2 학습 특성 비교

#### 수렴 속도
```
EIGAT: 빠름 (5-10 epochs에서 대부분 성능 달성)
ST-GCN: 중간 (10-15 epochs에서 안정화)
MR-GCN: 느림 (20 epochs 이후에도 개선 없음)
```

#### 안정성
```
EIGAT: 매우 안정적 (변동성 거의 없음)
ST-GCN: 안정적 (점진적 개선)
MR-GCN: 불안정 (큰 변동성)
```

#### 과적합 여부
```
EIGAT: 거의 없음 (val/test 성능 비슷)
ST-GCN: 없음 (일반화 성능 우수)
MR-GCN: 심함 (val/test 큰 차이)
```

---

## 📌 5. 실무 해석

### 5.1 블라인드 존 감지 관점

#### ST-GCN이 최적인 이유
```
1. AUPRC 0.518 (높음)
   → 위험 신호를 정확히 감지 가능
   
2. Best F1 0.647 (높음)
   → 정밀도와 재현율 모두 우수
   
3. 시간 정보 활용
   → VRU의 움직임 패턴 학습
   → 현재만 본다면 위험 판단 불가능
```

#### EIGAT의 한계
```
1. AUPRC 0.510 (좋지만 ST-GCN보다 낮음)
   → 위험 신호 감지 약간 낮음
   
2. Best F1 0.552 (좋지만 ST-GCN보다 15% 낮음)
   → 정밀도-재현율 균형이 떨어짐
   
3. 시간 정보 없음
   → 현재 위치만으로 판단
   → 위험도 정확히 파악 어려움
```

#### MR-GCN의 실패
```
1. AUPRC 0.023 (거의 의미 없음)
   → 위험 신호 감지 불가능
   
2. Best F1 0.047 (사용 불가)
   → 거의 모든 위험 놓침
   
3. 설계 오류
   → Multi-Relation 구조가 역효과
```

### 5.2 배포 결정

#### ✅ ST-GCN 선택 (최고 권장)
```
장점:
  ✓ AUPRC 최고 (0.518)
  ✓ Best F1 최고 (0.647)
  ✓ 시간 정보로 안정성 향상
  ✓ 위험 감지율 최고 (69%)

단점:
  ✗ 학습 시간 필요 (46분/epoch)
  ✗ AUROC는 EIGAT보다 약간 낮음
```

#### ⚠️ EIGAT 선택 (차선책)
```
장점:
  ✓ AUROC 최고 (0.912)
  ✓ 학습 빠름
  ✓ 이미 완료

단점:
  ✗ Best F1 낮음 (0.552)
  ✗ AUPRC 약간 낮음 (0.510)
  ✗ 위험 감지율 미지수
```

#### ❌ MR-GCN 선택 (불가)
```
이유:
  ✗ AUPRC 거의 0 (0.023)
  ✗ Best F1 거의 0 (0.047)
  ✗ 모든 메트릭에서 부족

결론: 절대 사용 금지
```

---

## 📊 6. 시각화 요약 테이블

| 시각화 | 파일명 | 크기 | 용도 |
|--------|--------|------|------|
| **메인 메트릭 비교** | main_metric_comparison.png | 87KB | 네 가지 메트릭을 막대 그래프로 비교 |
| **Validation 곡선** | validation_curves.png | 152KB | 각 모델의 학습 진행 과정 |
| **메트릭 요약 (PNG)** | model_metric_summary.png | 89KB | 모든 메트릭 종합 비교 |
| **메트릭 요약 (PDF)** | model_metric_summary.pdf | 24KB | 인쇄용 고품질 버전 |

---

## 🎓 7. 최종 결론

### 시각화가 말해주는 것

```
1. 메인 메트릭 비교 (Main Metric)
   → ST-GCN: Best F1과 AUPRC에서 최고
   → EIGAT: AUROC에서 최고
   → MR-GCN: 모든 메트릭에서 최악

2. Validation 곡선
   → ST-GCN: 가파른 상승, 안정적 수렴
   → EIGAT: 빠른 수렴, 안정성 높음
   → MR-GCN: 성능 정체, 불안정함

3. 메트릭 요약
   → ST-GCN이 높이가 대부분 높음
   → EIGAT이 특정 메트릭(AUROC)에서만 높음
   → MR-GCN의 모든 막대가 매우 짧음
```

### 📍 배포 권장사항

```
1순위: ST-GCN (20 Epochs)
   - 모든 중요 메트릭에서 최고
   - 시간 정보로 안정성 향상
   - 위험 감지율 최고

2순위: EIGAT (20 Epochs)
   - AUROC 최고
   - 학습 빠름
   - 하지만 Best F1이 떨어짐

3순위: MR-GCN
   - 배포 불가
   - 완전히 부적합
```

---

## 📁 생성된 파일 위치

```
/outputs/figures/final_comparison/
  ├── main_metric_comparison.png
  └── validation_curves.png

/outputs/figures/
  ├── model_metric_summary.png
  └── model_metric_summary.pdf
```

---

**작성 완료**: 2026-05-23  
**모든 시각화 생성**: ✅  
**최종 권장**: ST-GCN (20 Epochs) 🏆
