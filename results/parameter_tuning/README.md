# Parameter Tuning Results

이 디렉토리는 하이퍼파라미터 튜닝 및 모델 최적화 결과를 저장합니다.

## 디렉토리 구조

```
parameter_tuning/
├── best/                    # 최종 최적 결과
│   ├── best_feature_pipeline.csv
│   ├── best_model_evaluation.csv
│   └── best_n_trials_parameter.csv
├── figures/                 # 시각화 결과
│   └── meta_score_by_n_trials.png
├── meta_score.py            # 메타 스코어 계산 스크립트
├── plot_meta_score.py       # 시각화 스크립트
└── README.md                # 이 파일
```

## 파일 설명

### 최종 결과 (`best/`)

- **`best_feature_pipeline.csv`**: Feature selection/engineering 조합별 최고 성능 결과
- **`best_model_evaluation.csv`**: 최고 모델의 평가 지표 (ROC-AUC, PR-AUC, F1, Recall, Precision 등)
- **`best_n_trials_parameter.csv`**: Ensemble strategy별 최적 n_trials 파라미터

### 입력 데이터

튜닝에 사용된 입력 CSV 파일들은 `data/processed/parameter_tuning/` 디렉토리에 저장되어 있습니다:

- `n_trials_comparison.csv`: n_trials별 모델 성능 비교 데이터
- `past_data.csv`: 과거 실험 데이터

### 스크립트

- **`meta_score.py`**: F2-score 및 MetaScore 계산, 최적 파라미터 추출
- **`plot_meta_score.py`**: n_trials에 따른 meta_score 변화 시각화

## 사용 방법

### 메타 스코어 계산

```python
from results.parameter_tuning.meta_score import main

# 기본 경로 사용 (best/best_feature_pipeline.csv)
main()

# 또는 커스텀 경로 지정
main(input_file="path/to/input.csv", output_file="path/to/output.csv")
```

### 시각화 생성

```python
from results.parameter_tuning.plot_meta_score import main

# 기본 경로 사용
main()

# 또는 커스텀 경로 지정
from pathlib import Path
main(
    input_path=Path("data/processed/parameter_tuning/n_trials_comparison.csv"),
    output_path=Path("results/parameter_tuning/figures/meta_score_by_n_trials.png")
)
```

## 하이퍼파라미터 튜닝 실험

### 실험 개요

본 프로젝트에서는 **Optuna 베이지안 최적화**를 사용하여 앙상블 전략별 최적 하이퍼파라미터를 탐색했습니다. 특히 `n_trials` 파라미터가 모델 성능에 미치는 영향을 체계적으로 분석했습니다.

### 실험 설계

#### 1. 앙상블 전략 비교
- **Stacking Ensemble**: Base Models (Random Forest, XGBoost, LightGBM) + Meta-Learner (Logistic Regression)
- **Voting Ensemble**: Base Models (Random Forest, XGBoost, LightGBM) + Hard Voting

#### 2. n_trials 탐색 범위
- **범위**: 30 ~ 450 (30 간격으로 증가)
- **총 실험 수**: 각 앙상블 전략당 15개 실험
- **평가 방법**: Stratified K-Fold Cross-Validation (K=5)

#### 3. 평가 지표
- **주요 지표**: MetaScore (최종 선정 기준)
- **보조 지표**: ROC-AUC, PR-AUC, F2-Score, Recall, Precision, Accuracy

### 실험 결과

#### n_trials에 따른 성능 변화

![Meta Score by n_trials](figures/meta_score_by_n_trials.png)

**주요 발견사항:**

1. **Stacking 앙상블**
   - 최적 n_trials: **240**
   - 최고 MetaScore: **0.9479**
   - n_trials 증가에 따라 안정적인 성능 유지
   - n_trials=240 이후에도 0.946 이상의 높은 성능 유지

2. **Voting 앙상블**
   - 최적 n_trials: **120**
   - 최고 MetaScore: **0.9466**
   - n_trials 변화에 따른 성능 변동이 큼
   - n_trials=150, 250에서 성능 저하 관찰

3. **전략 간 비교**
   - Stacking이 Voting보다 전반적으로 우수한 성능
   - Stacking은 더 안정적이고 예측 가능한 성능 패턴
   - 두 전략 모두 n_trials=100 이후 높은 성능 달성

#### 최적 파라미터 선정

| 앙상블 전략 | 최적 n_trials | MetaScore | ROC-AUC | PR-AUC | F2-Score | Recall | Precision | Accuracy |
|------------|--------------|-----------|---------|--------|----------|--------|-----------|----------|
| **Stacking** | **240** | **0.9479** | 0.9918 | 0.9642 | 0.9206 | 0.9570 | 0.7992 | 0.9543 |
| Voting | 120 | 0.9466 | 0.9915 | 0.9626 | 0.9190 | 0.9551 | 0.7984 | 0.9539 |

**결론**: MetaScore 기준으로 **Stacking 앙상블 (n_trials=240)**이 최종 선정되었습니다.

### 실험 데이터

- **원본 데이터**: `data/processed/parameter_tuning/n_trials_comparison.csv`
- **최적 파라미터**: `best/best_n_trials_parameter.csv`
- **최종 모델 평가**: `best/best_model_evaluation.csv`

### 실험 재현 방법

```python
from main import run
from src.preprocessing import load_data

# 데이터 로드
df = load_data()

# Stacking 앙상블로 n_trials=240 튜닝
results = run(
    df=df,
    cv_strategy='stratified_kfold',
    tuning_strategy='optuna',
    ensemble_strategy='stacking',
    is_save=True
)
```

## MetaScore 공식

MetaScore는 다음 공식으로 계산됩니다:

```
MetaScore = 0.5 × F2 + 0.3 × PR-AUC + 0.2 × ROC-AUC
```

여기서 F2-score는 Recall에 더 높은 가중치를 부여한 지표입니다:

```
F2 = (1 + 2²) × (precision × recall) / (4×precision + recall)
```

**가중치 설계 이유:**
- **F2 (50%)**: 이탈 고객을 놓치지 않는 것이 최우선이므로 재현율에 높은 가중치
- **PR-AUC (30%)**: 불균형 데이터에서 정밀도-재현율 균형 평가
- **ROC-AUC (20%)**: 전반적인 분류 성능 평가

