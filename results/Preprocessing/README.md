# 데이터 전처리 결과서

## 수집한 데이터 설명

데이터 출처: 2020년 말 https://analyttica.com/leaps/ -> 웹사이트의 운영정책 변경으로 현재 데이터 접근 불가 -> 캐글에 있는 데이터셋 사용

## EDA 결과

✓ Feature Engineering 완료!
생성된 파생 변수: Trans_Change_Ratio, Inactivity_Score, Engagement_Score, Utilization_Risk_Level
최종 데이터 shape: (10127, 34)

### (1) 결측치

- 결측치가 있는 컬럼: Education_Level - 1519, Income_Category - 1112,Marital_Status-749

- 어떻게 처리했는가?:

    * 범주형 컬럼: df[col].mode()[0]를 사용하여 최빈값(가장 많이 등장하는 값)으로 결측치 채움.
    
    * 수치형 컬럼: 0의 값을 "결측값" 으로 여기기. 0인 부분을 NaN으로 바꾼후에 평균 (mean_val)을 계산 후 평균값으로 채움. 

- 왜 그렇게 처리 했는가?
    * 범주형 컬럼 (Object 타입): 최빈값으로 채우며, 기존 데이터 패턴을 덜 깨뜨리면서, 새로운 "가짜 카테고리"를 생성할 필요성이 주러들음. 

    * 수치형 컬럼 (number 타입): 분포를 크게 왜곡하지 않으면서, 모델이 결측 때문에 튀는 패턴을 배우지 않도록 막아주는 안전한 선택.

### (2) 이상치

* 이상치 판정 기준이 무엇인가?: 
   - 코드 상에선 명시적인 이상치 탐지/제거 로직(IQR, z-score 등)'은 구현되어 있지 않음.

* 어떻게 처리했는가?: 
   - 명시적인 이상치가 구현이 안돼있다고 해도, 완전히 무시하지는 않고, 간접적으로 완화되는 부분은 감지: 
      - 0 → 평균값 대체 로직: 만약 어떤 컬럼에서 0이 비정상적으로 많다면, 그 값들을 평균으로 치환 되면서 분포가 조금 더 부드러워짐.

* 왜 그렇게 처리 했는가?:
    - 고객 행동 데이터에서는 이상치처럼 보이는 값도 실제로 중요한 “특이 행동 패턴”일 수 있기 때문.
        
        Ex:
         - 거래가 갑자기 폭증 → 사용률이 비정상적으로 높음 → 오히려 이탈 신호일 수도 있음.

            따라서:
                단순히 통계 기준(IQR, z-score)만으로 강하게 자르기보다는 모델이 스스로 학습하도록 두고,
                추후 모델 성능/Feature Importance를 보며 필요할 때만 추가로 처리하는 방향을 선택.

### (3) 기타 전처리 방법

* feature들과 target_col의 상관관계:

- Feature들과 target_col(Attrition_Binary)의 상관관계
  - 타겟: Attrition_Binary (이탈: 1, 유지: 0)
     - 코드: corr_with_target = df.corr(numeric_only=True)["Attrition_Binary"].sort_values(ascending=False)
print(corr_with_target)

- 이 상관계수를 통해:
   - 어떤 Feature가 이탈과 양(+)의 관계인지
   - 어떤 Feature가 이탈과 음(-)의 관계인지
   - 강하게 붙어 있는지(절대값이 큰지)를 확인.

- 이 결과를 참고해서: 
   - Engagement_Score, Inactivity_Score 같은 파생변수를 설계했고
   - 거래 관련 변수를 묶어주는 방향의 Feature Engineering을 진행

* 상관관계 계수가 높은 feature 처리방법: 
   - 두 Feature 간 상관계수가 |corr| > 0.85처럼 매우 높으면, 둘 다 거의 같은 정보를 가지고 있다고 볼 수 있음.
   - 그 경우:
        - 의미 해석이 더 쉬운 컬럼 하나만 남기거나
        - 둘을 합쳐서 요약 Feature (점수/비율) 로 바꾸는 전략 사용.

* target_col과의 상관관계 계수 절대값이 높은 feature 처리방법:
    - 타겟과의 상관관계 절대값이 높은 Feature는 기본적으로 모델에 포함하되,교차 검증을 통해 과적합 여부를 확인하고, 필요한 경우 정규화 및 규제를 통해 안정성을 확보.

* 범주형 데이터 인코딩 방법 (인코딩 된 컬럼을 대신 사용): 
    - 범주형 Feature는 pd.get_dummies(drop_first=True)를 통해 One-Hot Encoding 하였으며, 기준 카테고리를 하나 제거(k-1 인코딩)하여 다중공선성을 완화.인코딩 이후에는 원본 범주형 컬럼 대신 인코딩된 더미 컬럼을 사용.

## 적용한 Feature Engineering 방식

  * 비율 (Ratio): 숫자가 크거나 작거나의 결과값으로 추정하는것이 아니라, 숫자가 얼마만큼 변했는지를 보는 것이 중요할때 사용. 

  * 조합기반 (Combination): 두 개 이상의 컬럼을 곱셈·가중합 형태로 결합해
  고객의 활동 특성을 단일 점수로 표현하는 방식. 

  * 구간 나누어 주기 (Binning): 숫자를 그대로 쓰는것이 아니라, 각자 라벨링을 줘서 등급을 표기하는 방식. 

  * 데이터 의미 보정 (Meaningful Replace): 숫자 0이나 NaN의 오해를 방치하기 위해: 
   - 범주형 = 최빈값으로 채우고
   - 숫자형 = 0을 일단 NaN으로 보고 평균으로 채워서
     → 데이터 의미가 왜곡되지 않게 만듦. 

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