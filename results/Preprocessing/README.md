# 데이터 전처리 결과서

## 수집한 데이터 설명

데이터 출처: 2020년 말 https://analyttica.com/leaps/ -> 웹사이트의 운영정책 변경으로 현재 데이터 접근 불가 -> 캐글에 있는 데이터셋 사용

## EDA 결과

### (1) 결측치

- 결측치가 있는 컬럼: Education_Level - 1519, Income_Category - 1112, Marital_Status - 749
- 어떻게 처리했는가?
- 왜 그렇게 처리 했는가?

### (2) 이상치

- 이상치 판정 기준이 무엇인가?
- 어떻게 처리했는가?
- 왜 그렇게 처리 했는가?

### (3) 기타 전처리 방법

- feature들과 target_col의 상관관계
- 상관관계 계수가 높은 feature 처리방법
- target_col과의 상관관계 계수 절대값이 높은 feature 처리방법
- 범주형 데이터 인코딩 방법 (인코딩 된 컬럼을 대신 사용)
- etc...

## 적용한 Feature Engineering 방식
  * 비율 (Ratio): 숫자가 크거나 작거나의 결과값으로 추정하는것이 아니라, 숫자가 얼마만큼 변했는지를 보는 것이 중요할때 사용. 
  * 조합기반 (Combination): 두 개 이상의 컬럼을 곱셈·가중합 형태로 결합해
  고객의 활동 특성을 단일 점수로 표현하는 방식. 
  * 구간 나누어 주기 (Binning): 숫자를 그대로 쓰는것이 아니라, 각자 라벨링을 줘서 등급을 표기하는 방식. 
    - Ex. 


### (1) Feature Scaling
  * Linear 모델(Logistic, SVM) : StandardScaler / MinMaxScaler 적용.
  * Tree 계열 모델(XGBoost/RandomForest) : Scaling 불필요.

### (2) Feature Selection
  * 상관계수 기반 필터링.
  * XGBoost Feature Importance 기반 선정.

### (3) Feature Engineering

- 거래 변화율 지표: 
  * 거래 급증 또는 급감 패턴이 이탈률과 유의한 상관관계를 가지는 점을 반영.
  * 고객의 총 거래 횟수 대비 분기 변화율을 정규화한 지표.

- 비활동 기간 리스크: 
  * 고객의 금융 행동 안정성 / 불균형 신호를 모델이 인식하도록 유도.
  * 비활동 개월 수 × 카드 사용률.

- 고객 활동 점수: 
  * 고객의 적극적인 활용성 점수가 낮다는 경향발견. 
  * 거래 금액 + 거래 횟수 - 비활동 패널.

- 카드 사용률 기반 위험 등급:
  * 비선형적 위험 구간을 “범주형 라벨”로 모델에 제공.