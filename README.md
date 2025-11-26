# 📘 프로젝트 README

## 1️⃣ 팀원 및 담당 업무

| #   | 팀원   | 담당 업무                                            | 비고 |
| --- | ------ | ---------------------------------------------------- | ---- |
| 1   | 김준석 | 모듈화, 하이퍼파라미터 튜닝, 파이프라이닝, Streamlit UI 구현             |   👑  |
| 2   | 문지영 |  Feature Engineering, 앙상블 전략, Streamlit UI 구현     |      |
| 3   | 신병탁 | CV 전략 확립·모델 실험, Streamlit UI 구현               |      |
| 4   | 이명준 | eda, 모델 실험, 모델 평가, QA, 문서화                               |      |
| 5   | 손현우 | Feature Engineering, Streamlit UI 구현              |      |

---

## 2️⃣ 프로젝트 주제 및 선정 이유

### 2‑1. 주제

> “Credit Card사 고객 이탈(Churn) 예측”

### 2‑2. 주제를 선택한 이유

- **비즈니스 가치** – 이탈 고객을 사전에 파악해 맞춤형 마케팅/유지 전략을 수립할 수 있음.
- **데이터 가용성** – Kaggle에 풍부한 데이터가 존재(1만여명의 데이터).
- **기술 학습 목표** – Tuning, 앙상블 등 최신 ML 기법을 실전 프로젝트에 적용하고자 함.

---

## 3️⃣ 주요 기능

- **이탈 확률 예측** – 입력된 고객 정보(거래·인구통계)로 이탈 가능성을 실시간 예측.
- **이탈 예방 리텐션 마케팅(Streamlit)** – 이탈 위험도에 따라 각 회원의 이용 패턴에 최적화된 피드백 및 메시지발송

---

## 4️⃣ 프로젝트 디렉터리·파일 구조

### 4-1. 전체 구조

```
SKN21-2nd-3Team/
├─ .gitignore
├─ README.md
├─ requirements.txt
├─ main.py
├─ data/
│   ├─ images/
│   ├─ processed/
│   └─ raw/
├─ src/
│   ├─ __init__.py
│   ├─ preprocessing.py
│   ├─ cv.py
│   ├─ ensemble.py
│   └─ tuner.py
└─ results/
    ├─ Final_Model/
    ├─ Preprocessing/
    ├─ Modeling/
    ├─ parameter_tuning/
    └─ streamlit/
```

### 4-2. 루트 디렉터리 파일

| 파일명 | 설명 |
|--------|------|
| `main.py` | 프로젝트의 메인 진입점. 모델 학습, 평가, 저장 등의 전체 파이프라인을 실행 |
| `requirements.txt` | 프로젝트에 필요한 Python 패키지 및 버전 정보 |
| `README.md` | 프로젝트 개요, 사용법, 구조 등 프로젝트 전반에 대한 설명 문서 |
| `.gitignore` | Git 버전 관리에서 제외할 파일 및 디렉터리 목록 (venv, __pycache__, .pyc 등) |

### 4-3. data/ 디렉터리

데이터 저장 및 관리 디렉터리

```
data/
├─ images/                    # 프로젝트 문서화에 사용되는 이미지 파일들
│   ├─ 00_basic_preprocessing.png
│   ├─ 01_BASEMODE_difference.png
│   ├─ 02_validation_score.png
│   ├─ 03_SRCfolder.png
│   ├─ 04_columns_to_korean.png
│   ├─ 05_optuna_exp.png
│   └─ 06_exp_ntrial.png
├─ processed/                 # 전처리된 데이터 및 중간 결과물
│   └─ parameter_tuning/
│       ├─ n_trials_comparison.csv
│       └─ past_data.csv
└─ raw/                       # 원본 데이터
    └─ BankChurners.csv       # Kaggle에서 수집한 신용카드 고객 이탈 데이터
```

**주요 파일 설명:**
- `raw/BankChurners.csv`: 원본 고객 이탈 예측 데이터셋
- `processed/parameter_tuning/`: 하이퍼파라미터 튜닝 과정에서 생성된 중간 데이터
- `images/`: README 및 문서화에 사용되는 시각화 이미지

### 4-4. src/ 디렉터리

핵심 모듈 및 유틸리티 함수

```
src/
├─ __init__.py
├─ preprocessing.py           # 데이터 전처리 및 Feature Engineering
├─ cv.py                      # Cross-Validation 전략 구현
├─ ensemble.py                # 앙상블 모델 구현 (Stacking, Voting)
└─ tuner.py                   # 하이퍼파라미터 튜닝 (Optuna, GridSearch, RandomSearch)
```

**주요 모듈 설명:**

- **`preprocessing.py`**: 
  - 결측치 처리, 스케일링, 인코딩
  - Feature Engineering (파생 변수 생성, Feature Selection)
  
- **`cv.py`**: 
  - Simple Split, K-Fold, Stratified K-Fold 구현
  - 클래스 불균형 데이터를 위한 Stratified K-Fold 활용
  
- **`ensemble.py`**: 
  - Stacking 앙상블 (LGBM, XGBoost, RandomForest → LogisticRegression)
  - Voting 앙상블 구현
  
- **`tuner.py`**: 
  - Optuna를 활용한 베이지안 최적화
  - GridSearch, RandomSearch 구현

### 4-5. results/ 디렉터리

실험 결과, 모델, 시각화 자료 저장

```
results/
├─ Final_Model/               # 최종 선택된 모델 파일들
│   ├─ final_model.joblib
│   ├─ stacking_model.joblib
│   ├─ voting_model.joblib
│   └─ Exclude_Features_selection_*.joblib
├─ Preprocessing/             # 전처리 단계 결과물
│   ├─ EDA.ipynb              # 탐색적 데이터 분석 노트북
│   ├─ images/                # EDA 시각화 결과
│   └─ README.md
├─ Modeling/                  # 모델링 실험 결과
│   ├─ images/
│   │   ├─ exp_ntrial.png
│   │   └─ model_evaluation.png
│   └─ README.md
├─ parameter_tuning/          # 하이퍼파라미터 튜닝 결과
│   ├─ best/                  # 최적 파라미터 및 평가 결과
│   │   ├─ best_feature_pipeline.csv
│   │   ├─ best_model_evaluation.csv
│   │   └─ best_n_trials_parameter.csv
│   ├─ figures/
│   │   └─ meta_score_by_n_trials.png
│   ├─ meta_score.py
│   ├─ plot_meta_score.py
│   └─ README.md
└─ streamlit/                 # Streamlit 웹 애플리케이션
    ├─ main.py                # Streamlit 앱 진입점
    ├─ dashboard.py           # 대시보드 UI 구현
    ├─ data_chart.py          # 데이터 시각화 컴포넌트
    ├─ message_center.py      # 맞춤형 메시지 발송 기능
    └─ utils.py               # Streamlit 앱 유틸리티 함수
```

**주요 디렉터리 설명:**

- **`Final_Model/`**: 
  - 학습 완료된 최종 모델 파일 (`.joblib` 형식)
  - 다양한 Feature Selection 전략에 따른 모델 버전 저장
  
- **`Preprocessing/`**: 
  - EDA 노트북 및 전처리 과정에서 생성된 시각화 자료
  - 데이터 분포, 상관관계, 이상치 분석 결과
  
- **`Modeling/`**: 
  - 모델 평가 결과 및 실험 과정 시각화
  
- **`parameter_tuning/`**: 
  - Optuna 튜닝 결과 및 최적 하이퍼파라미터
  - n_trials에 따른 메타 스코어 비교 결과
  
- **`streamlit/`**: 
  - 고객 이탈 예측 및 리텐션 마케팅을 위한 웹 애플리케이션
  - 실시간 예측, 대시보드, 맞춤형 메시지 발송 기능

---

## 5️⃣ 수집 데이터 설명

| 데이터명           | 출처                         | 주요 컬럼                                                                                                                                                              | 비고 |
|--------------------|------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------|
| `BankChurners.csv` | Kaggle (Bank Customer Churn) | `Gender`, `Age`, `IncomeCategory`, `EducationLevel`, `CardCategory`, `MonthsOnBook`, `TotalTransactionCount`, `TotalTransactionAmount`, `Attrition_Flag` |      |

![columns_to_korean](./data/images/04_columns_to_korean.png)

## 6️⃣ 개발 과정

### 6-1. EDA 및 전략 수립
**데이터 특성 파악, 평가지표 선정, CV 전략 결정**

1. **EDA 진행**: `results/Preprocessing/EDA.ipynb`를 통해 데이터 특성 분석
2. **평가지표 선정**: 비즈니스 목적과 데이터 불균형 고려
    - **FN (놓친 이탈 고객)**: 고객을 완전히 잃음 → **큰 손실** (중요)
    - **FP (잘못된 이탈 예측)**: 불필요한 프로모션 비용 → 상대적으로 작은 비용
    ![평가점수](./data/images/02_validation_score.png)
3. **CV 전략 수립**: 과적합 방지 및 신뢰성 확보
    - *질문*: 타겟 데이터가 불균형한가? & 시계열 데이터인가?
    - *결정*: **Stratified K-Fold** 적용

### 6-2. 베이스라인 구축
**기본 전처리/FE 파이프라인 구축**

- 기본적인 인코딩 및 불필요한 컬럼 제거만 수행하는 **BASE PIPELINE** 제작
![basic_preprocess](./data/images/00_basic_preprocessing.png)

### 6-3. 모델 선택
**동일한 BasePipeline, CV 환경에서 여러 모델 비교 후 Top 3 선정**

- **선정 모델**: `LogReg`, `DTree`, `LGBM`, `XGBoost`, `RandomForest` → 최종 `LGBM`, `XGBoost`, `RandomForest` 선택
![베이스모델평가](./data/images/01_BASEMODE_difference.png)
이때에도 평가Metric 을 최대한 통일하려함.


### 6-4. 모듈화
**전처리, FE, 앙상블 전략을 `.py` 파일로 구조화**

![SRC](./data/images/03_SRCfolder.png)

- **`cv.py`**: Simple Split, K-Fold, Stratified K-Fold 구현
- **`ensemble.py`**: Stacking, Voting 앙상블 기법 구현
- **`preprocessing.py`**: 결측치 처리, 스케일링, Feature Engineering (Add/Select) 구현
- **`tuner.py`**: Optuna, GridSearch, RandomSearch 구현

### 6-5. 체계적 실험
**각 전략 조합을 실험하며 결과 추적 (Seed 관리)**

- 🔗 [Notion 실험 로그 보러가기](https://www.notion.so/2-2b6153f6ee1280e5bec5d62110449c73?source=copy_link)
- **실험 파이프라인 순서**:
  `CV 전략` → `전처리` → `Feature Engineering` → `튜닝` → `앙상블 전략` 순으로 고정하며 최적화
- **실험 환경**: Random Seed `42` 고정
![optuna_Exp](./data/images/05_optuna_exp.png)
![n_trials](./data/images/06_exp_ntrial.png)
### 6-6. 최종 모델 확정
**가장 안정적이고 높은 성능의 모델/앙상블 조합 도출**

| 구분 | 최종 결정 내용 |
| :--- | :--- |
| **전처리** | 결측치 제거, 불필요 컬럼 제거 |
| **FE** | 거래 변화율, 비활동 리스크 스코어, 고객 참여도(Engagement Score), Utilization 위험 구간화 |
| **CV** | Stratified K-Fold |
| **튜닝** | Optuna (n_splits=240) |
| **앙상블** | **Stacking** (LGBM, XGB, RF → LogisticRegression) |

## 7️⃣ Application 주요 기능

### 7‑1 제공 기능

- **예측 버튼** – 현재 고객들의 이탈확률을 체크
- **이탈률에 따라 맞춤형 메시지 발송 서비스** - 자동으로 회원의 정보에 맞는 메시지가 발송됨.
- **회원별 피드백 추천 페이지** – 신용 사용률, 최근 거래 빈도, 거래횟수 등을 고려하여 더욱 디테일한 피드백 제공.

### 7‑2 기술 스택

- **백엔드**: Python, Ensemble, Optuna
- **프론트엔드**: Streamlit

---

## 8️⃣ 회고

### 8‑1 구현 중 발생한 문제 & 해결 방안

| 문제                       | 원인                                        | 해결 방법                                                   |
| -------------------------- | ------------------------------------------- | ----------------------------------------------------------- |
| 결측치·이상치 처리         | 일부 컬럼(`Education_Level` 등) 데이터 누락 | 평균/최빈값 대체 + 이상치 IQR 필터링                        |
| 클래스 불균형 (양성 16%)   | 데이터 자체 비율                            | `scale_pos_weight` 조정 + Stratified K‑Fold CV              |
| 평가 점수 확립                          | F1-SCORE, ROC-AUC, PR-AUC, RECALL 등 다양한 평가점수 존재                       | …                                                           |

### 8‑2 회고록

- **김준석**: "이번 프로젝트를 통해 ML 개발의 체계적인 프로세스를 체득할 수 있었습니다. 특히 EDA → 전처리 → Feature Engineering → 튜닝 → 앙상블 순으로 진행되는 파이프라인을 직접 구축하면서, 각 단계가 왜 필요한지 명확히 이해하게 되었습니다. 처음에는 CV 평가 확립을 언제 해야 할지, 실험을 어떻게 셋팅해야 할지 막연했지만, 팀원들과 함께 체계적으로 실험하며 최적의 조합을 찾아가는 과정에서 자연스럽게 전체 흐름이 머릿속에 각인되었습니다. 또한 모듈화 작업을 맡으면서 cv.py, ensemble.py, tuner.py를 구조화하는 경험을 통해, 혼자 작업할 때보다 팀원들과 역할을 나눠 협업하는 것이 훨씬 효율적이라는 것을 체감했습니다. 앞으로 특정 로직을 선택할때 그 근거를 더 명확히 하는것이 좋은 ML개발자가 될 수 있다 라는 말을 곱씹으며 임하도록 하겠습니다." </br>
- **문지영**: "이번 이탈률 예측 프로젝트를 진행하면서, 단순히 모델을 학습시키는 것보다 데이터를 어떻게 다루고 설계하는지가 모델 성능에 더 큰 영향을 준다는 것을 느낄 수 있었습니다. 처음 모델을 학습시켰을 때는 클래스 불균형 문제가 나타났는데, 이를 해결하기 위해 모델 학습 시 하이퍼파라미터인 'scale_pos_weight'를 활용해 이탈 고객에 더 높은 가중치를 부여함으로써 모델이 두 클래스를 균형 있게 학습할 수 있도록 개선했습니다. 단순히 원본 컬럼만 사용했을 때보다, 파생 변수를 추가한 후 모델의 표현력이 확실히 개선되는 경험을 했으며, 최적화된 단일 모델도 한계를 가질 수 있기에 서로 다른 오류 패턴을 가진 모델(LightGBM, Random Forest, Logistic Regression 등)을 조화롭게 결합한 앙상블을 통해 단순히 성능을 높이는 것을 넘어 예측 결과가 극단적으로 흔들리는 것을 방지하는 안정성을 확보할 수 있음을 체감했습니다. 이번 프로젝트를 하면서 기술적인 성과뿐 아니라, 팀원들이 방향을 잡아주고 함께 고민하며 진행한 덕분에 다양한 시도와 배움을 얻을 수 있다는 협업의 의미도 다시 한 번 느꼈습니다." </br>
- **신병탁**: “??” </br>
- **이명준**: “git을 통한 협업에 대해 얘기를 하던 도중, 시니어 개발자이신 팀원 분의 이야기를 통해 git을 통한 협업 방식을 구축하는 데에 있어 큰 관심이 생겼고, 작은 기능의 구현보다 단위 프로젝트가 나아가야할 방향성에 대한 생각을 하고 중심을 잃지 않으려고 노력하며 프로젝트를 진행했습니다. 또한, 프로젝트 상에서 역할 분배에 미흡한 부분이 있는지를 능동적으로 확인하고, 해당 부분을 메꿔가며 팀원 분들이 진행중인 workflow에 필요하지 않은 부분이 없는지 확인하며 진행했습니다. 이를 통해 기술적인 관점에서 agent를 이용해 프로젝트를 진행을 한다면 충분히 혼자서도 할 수 있지만, 각자의 분야에서 agent를 사용하며 강점을 살리는 협업을 통한 워크플로우의 중요성을 더 크게 실감했습니다. 또한 기술적으로는 target인 Churners(고객이탈)의 특수성을 고려하여 recall을 충분히 반영하는 f2 score와 데이터셋의 불균형(Churners가 16.1%)문제에서 오는 불안정성을 해결하기 위한 PR-AUC, 그리고 ROC-AUC를 가중합하여 Recall 우선 전략을 위한 평가지표인 Metascore를 만들었습니다. 해석에 따라 달리할 수 있는 성능 지표를 어떻게 설정할지 논의하며, 모델 평가지표에 대한 관점이 다각적으로 변화하기도 하였습니다. QA 측면에서는 Random Seed 고정을 통한 실험 재현성 확보, CV Fold별 성능 편차 모니터링, Streamlit 앱의 추론 정상 동작 검증을 진행했고, 실험 결과를 CSV로 체계적으로 기록하고 meta_score.py를 통해 앙상블 전략별 최적 파라미터를 자동 추출하여 문서화하는 역할을 담당했습니다. 이후의 프로젝트에서도 flow에 논리적으로 모순이 없도록 노력하겠습니다.” </br>
- **손현우**: "이번 Feature Engineering 분야에 있으면서, 그 과정에 있어서 데이터를 정리하고 꼼꼼히 하나하나 분석하며 나아가는 것에 대한 중요성을 배웠습니다. 또한 Streamlit으로 통하여 비즈니스 서비스 제안에 있어서, Feature 기반을 더 확률성이 확실한것으로 만들어 나가는 것이 중효하다는 것을 느꼈습니다. EDA로 통하여 코드 직접지었을때, 데이터 분석을 깔짝해서 코드를 만들수 있는게 아니라, 코드하나라도 실용성이 있어요 modeling 쪽으로 인수인계를 할때 발랜스가 수월하며, 데이터값도 거희 호환성이 높은 값을 제공하는것이 통계분석에 있어 중요하다는 점을 다시한번 깨달았습니다."
