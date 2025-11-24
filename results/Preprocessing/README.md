# 데이터 전처리 결과서

## 수집한 데이터 설명
데이터 출처: 2020년 말 https://analyttica.com/leaps/ -> 웹사이트의 운영정책 변경으로 현재 데이터 접근 불가 -> 캐글에 있는 데이터셋 사용

## EDA 결과
### (1) 결측치

결측치가 있는 컬럼 ('Unknown' 포함):
                 Unknown_Count  Missing_Percent
Education_Level           1519           15.000
Income_Category           1112           10.981
Marital_Status             749            7.396

총 결측치 비율: 1.45%

### (2) 이상치

--- 로그 변환 결과 (Log_Total_Trans_Amt 열 추가) ---
|   Total_Trans_Amt |   Log_Total_Trans_Amt |
|------------------:|----------------------:|
|         1144.0000 |                7.0432 |
|         1291.0000 |                7.1639 |
|         1887.0000 |                7.5433 |
|         1171.0000 |                7.0665 |
|          816.0000 |                6.7056 |

### (3) 

## 결측치 처리 방법 및 이유


## 이상치 판정 기준과 처리 방법 및 이유

**Log 변환 이상치 처리**

* Total_Trans_Amt 로그 변환. 

data['Log_Total_Trans_Amt'] = np.log1p(data['Total_Trans_Amt'])

print("--- 로그 변환 결과 (Log_Total_Trans_Amt 열 추가) ---")
print(data[['Total_Trans_Amt', 'Log_Total_Trans_Amt']].head().to_markdown(index=False, floatfmt=".4f"))

* Total_Trans_Amt + Total_Revolving_Bal 로그 변환 (왜도 높은 컬럼처리)

log_cols = ["Total_Trans_Amt", "Total_Revolving_Bal"]
for col in log_cols:
    if col in data.columns:
        data[f"log_{col}"] = np.log1p(data[col])

로그전환으로 통한 이상치 판정값이 필요했던 경우, 극단값이 매우 많았습니다.

Ex.
- Total_Trans_Amt (거래량)
- Total_Revolving_Bal (잔액)

같은 값들의 경우, 원본 값을 쓰면, 머신러닝 모델이 극단값에 과도하게 끝려가서, 이탈예측률이 불안정할수도있고, 성능도 떨어질수있습니다. 

따라서 log 로 처리할경우, 튀어나오는 값이 없이 데이터 값에 분포가 부드러워져서 모델 균형을 줄수있습니다. 


## 기타 전처리 방법
feature들과 target_col의 상관관계
상관관계 계수가 높은 feature 처리방법
target_col과의 상관관계 계수 절대값이 높은 feature 처리방법
범주형 데이터 인코딩 방법 (인코딩 된 컬럼을 대신 사용)

## 적용한 Feature Engineering 방식

**Feature Engineering: 단일 지표로 설명되지 않는 데이터 보완**

고객 이탈(Churn) 패턴은 단일 차원 정보만으로는 충분히 설명되지 않는 경우가 많습니다.
예를 들어:

* Total_Trans_Amt → 총 거래 금액
* Total_Trans_Ct → 총 거래 횟수

두 변수는 각각 의미가 있지만, 개별적으로는 고객의 실제 활동성을 온전히 반영하기 어렵습니다.

**Why Combine Features?**

단일 정보만 보면 고객의 행동 패턴이 불확실할 수 있습니다.
예를 들어,

* 거래 금액이 높지만 횟수가 적은 고객
* 거래 횟수는 많지만 금액이 낮은 고객

이들의 **진짜 활동 강도(activity level)**를 정량적으로 표현하기 어렵습니다.

따라서 두 변수를 조합해 새로운 관점을 얻을 수 있습니다.

* New Engineered Feature 예시
Activity_Index = Total_Trans_Amt × Total_Trans_Ct

이 지표는 **전체적인 활동 강도(Activity Intensity)**를 의미하며,
단일 변수를 볼 때보다 이탈 가능성을 더 강하게 설명하는 특징(Feature)이 됩니다.

**효과: ML 모델 예측력 향상**

머신러닝 모델은 종종 조합된 Feature에서 더 높은 예측력을 보입니다.

예시 engineered features:

* Activity_Index
* Amt_per_Contact
* Risk_Score

이러한 새로운 Feature들은 데이터의 숨겨진 패턴을 강조하여
전체 모델 성능(Accuracy·Recall·AUC 등)을 크게 향상시키는 효과가 있습니다.
