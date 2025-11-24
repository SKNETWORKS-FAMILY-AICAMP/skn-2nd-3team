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



### (3) 

## 결측치 처리 방법 및 이유

## 이상치 판정 기준과 처리 방법 및 이유

## 기타 전처리 방법
feature들과 target_col의 상관관계
상관관계 계수가 높은 feature 처리방법
target_col과의 상관관계 계수 절대값이 높은 feature 처리방법
범주형 데이터 인코딩 방법 (인코딩 된 컬럼을 대신 사용)

## 적용한 Feature Engineering 방식