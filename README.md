# 📘 프로젝트 README

## 1️⃣ 팀원 및 담당 업무

| #   | 팀원   | 담당 업무                                            | 비고 |
| --- | ------ | ---------------------------------------------------- | ---- |
| 1   | 김준석 | 모듈화, 하이퍼파라미터 튜닝, Streamlit UI 구현             |      |
| 2   | 문지영 | 앙상블 전략, Feature Engineering, Streamlit UI 구현     |      |
| 3   | 신병탁 | CV 전략 확립·모델 실험, Streamlit UI 구현               |      |
| 4   | 이명준 | 앙상블 전략·모델 실험                                   |      |
| 5   | 손현우 | Feature Engineering, Streamlit UI 구현              |      |

---

## 2️⃣ 프로젝트 주제 및 선정 이유

### 2‑1️⃣ 주제

> “Credit Card사 고객 이탈(Churn) 예측”

### 2‑2️⃣ 주제를 선택한 이유

- **비즈니스 가치** – 이탈 고객을 사전에 파악해 맞춤형 마케팅/유지 전략을 수립할 수 있음.
- **데이터 가용성** – Kaggle에 풍부한 데이터가 존재(1만여명의 데이터).
- **기술 학습 목표** – Tuning, 앙상블 등 최신 ML 기법을 실전 프로젝트에 적용하고자 함.

---

## 3️⃣ 주요 기능

- **이탈 확률 예측** – 입력된 고객 정보(거래·인구통계)로 이탈 가능성을 실시간 예측.
- **시각화 대시보드 (Streamlit)** – 전체 고객 이탈 분포, 이탈 위험 고객 상세 리포트.

---

## 4️⃣ 프로젝트 디렉터리·파일 구조

```
skn-2nd-3team/
├─ .gitignore
├─ README.md
├─ requirements.txt
├─ main.py                       ← 프로젝트 진입점
├─ data/
│   └─ BankChurners.csv          # 원본 CSV 등
├─ src/
│   ├─ preprocessing.py          # 전처리 파이프라인
│   ├─ cv.py                     # Cross‑validation utils
│   ├─ ensemble.py               # 앙상블 로직
│   └─ tuner.py                  # 튜닝 로직
├─ results/
│   ├─ Final_Model/
│   │   └─ final_model.joblib    # 최종 모델
│   ├─ Preprocessing/
│   │   └─ README.md
│   ├─ Modeling/
│   │   └─ README.md
│   └─ streamlit/
│       ├─ main.py
│       ├─ dashboard.py
│       └─ utils.py
```

---

## 5️⃣ 수집 데이터 설명

| 데이터명           | 출처                         | 주요 컬럼                                                                                                                                                              | 비고 |
|--------------------|------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------|
| `BankChurners.csv` | Kaggle (Bank Customer Churn) | `CustomerID`, `Gender`, `Age`, `IncomeCategory`, `EducationLevel`, `CardCategory`, `MonthsOnBook`, `TotalTransactionCount`, `TotalTransactionAmount`, `Attrition_Flag` |      |

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

- **김준석**: “”
- **문지영**: “이번 이탈률 예측 프로젝트를 진행하면서, 단순히 모델을 학습시키는 것보다 데이터를 어떻게 다루고 설계하느냐가 모델 성능에 더 큰 영향을 준다는 것을 직접 체감할 수 있었습니다. 처음 모델을 학습시켰을 때는 클래스 불균형 문제가 나타났는데, 이를 해결하기 위해 모델 학습 시 하이퍼파라미터인 'scale_pos_weight'를 활용하여 Minority Class(이탈 고객)에 더 높은 가중치를 부여함으로써 모델이 두 클래스를 보다 균형 있게 학습할 수 있도록 개선했습니다. 또한, 단순히 원본 컬럼만 사용했을 때보다, 파생 변수를 추가한 후 모델의 표현력이 확실히 개선되는 경험을 했으며, 최적화된 단일 모델도 한계를 가질 수 있기에 서로 다른 오류 패턴을 가진 모델(LightGBM, Random Forest, Logistic Regression 등)을 조화롭게 결합한 앙상블을 통해 단순히 성능을 높이는 것을 넘어 예측 결과가 극단적으로 흔들리는 것을 방지하는 안정성을 확보할 수 있음을 체감했습니다. 이번 프로젝트를 하면서 기술적인 성과뿐 아니라, 팀원들이 방향을 잡아주고 함께 고민하며 진행한 덕분에 다양한 시도와 배움을 얻을 수 있다는 협업의 의미도 다시 한 번 느꼈습니다. ”
- **신병탁**: “??”
- **이명준**: “??”
- **손현우**: "??"
