import pandas as pd
import numpy as np
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_data(filepath = PROJECT_ROOT + f"/data/raw/BankChurners.csv"):
	"""CSV 파일을 불러와 데이터프레임 반환"""
	df = pd.read_csv(filepath)
	return df

def drop_column(df):
	"""이탈 예측용 데이터 전처리"""
	df = df.copy()
	df['Attrition_Binary'] = (df['Attrition_Flag'] == 'Attrited Customer').astype(int)
	drop_cols = [col for col in [
		'CLIENTNUM',
		'Attrition_Flag',
		"Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2",
		"Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1"
	] if col in df.columns]
	df = df.drop(columns=drop_cols)

	return df

def replace_nan_value(df):
	"""Nan 값을 대체합니다. (범주형 칼럼: 최빈값, 숫자형 칼럼: 평균값)"""
	df = df.copy()
	for col in df.select_dtypes(include='object').columns:
		df[col] = df[col].fillna(df[col].mode()[0])
	for col in df.select_dtypes(include=np.number).columns:
		mean_val = df.loc[df[col] != 0, col].mean()  # 0이 아닌 값의 평균
		df[col] = df[col].replace(0, np.nan)         # 0을 NaN으로 변경
		df[col] = df[col].fillna(mean_val)
	df = pd.get_dummies(df, drop_first=True)
	return df

def add_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """ML 전처리용 Feature Engineering 컬럼 생성"""

    df = df.copy()

    # 1) 거래 변화율 (거래 횟수 대비 분기 변화 비율)
    df["Trans_Change_Ratio"] = (
        df["Total_Trans_Ct"] / (df["Total_Ct_Chng_Q4_Q1"] + 1)
    )

    # 2) 비활동 기반 리스크 스코어
    df["Inactivity_Score"] = (
        df["Months_Inactive_12_mon"] * df["Avg_Utilization_Ratio"]
    )

    # 3) 고객 참여도 스코어 (Engagement Score)
    df["Engagement_Score"] = (
        df["Total_Trans_Amt"] * 0.4 +
        df["Total_Trans_Ct"] * 0.4 -
        df["Months_Inactive_12_mon"] * 0.2
    )

    # 4) Utilization 기반 위험 구간화
    df["Utilization_Risk_Level"] = pd.cut(
        df["Avg_Utilization_Ratio"],
        bins=[0, 0.3, 0.6, 1.0],
        labels=[0, 1, 2]
    ).astype(int)

    return df 

def preprocess_data(df):
    df = df.copy()

	# 전처리 파이프라인
	# 1.필요없는 칼럼 드랍
	# 2. nan값 채우기
	# 3. engineered feature 추가
	
    df = drop_column(df)
    df = replace_nan_value(df)
    df = add_feature_engineering(df)

    return df

import pandas as pd

def find_numeric_nan_columns(df):
    """
    숫자형 컬럼 중 NaN 값을 가진 컬럼과 NaN 개수를 반환합니다.
    """
    numeric_cols = df.select_dtypes(include='number').columns
    nan_info = {}
    for col in numeric_cols:
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            nan_info[col] = nan_count
    return nan_info

# 사용 예시
# nan_columns = find_numeric_nan_columns(df)
# print(nan_columns)


if __name__ == "__main__":
    pass


	