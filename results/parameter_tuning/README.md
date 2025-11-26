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

## MetaScore 공식

MetaScore는 다음 공식으로 계산됩니다:

```
MetaScore = 0.5 × F2 + 0.3 × PR-AUC + 0.2 × ROC-AUC
```

여기서 F2-score는 Recall에 더 높은 가중치를 부여한 지표입니다:

```
F2 = (1 + 2²) × (precision × recall) / (4×precision + recall)
```

