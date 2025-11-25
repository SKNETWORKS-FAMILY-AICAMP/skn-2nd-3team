import pandas as pd
import numpy as np
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
csv_file_path = sys.path[-1] + r"/data/raw/BankChurners.csv"


def load_data(csv_file_path = csv_file_path):
    """CSV 파일을 불러와 데이터프레임 반환"""
    df = pd.read_csv(csv_file_path)
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

def replace_nan_value(df, target_col='Attrition_Binary'):
    """Nan 값을 대체합니다. (범주형 칼럼: 최빈값, 숫자형 칼럼: 평균값)"""
    categoriacal_cols = []
    numerical_cols = []
    df = df.copy()

    for col in df.select_dtypes(include='object').columns:
        if col not in categoriacal_cols:
            categoriacal_cols.append(col)
        df[col] = df[col].fillna(df[col].mode()[0])
    for col in df.select_dtypes(include=np.number).columns:
        if col not in numerical_cols:
            numerical_cols.append(col)
        if col == target_col:
            # 타깃 컬럼은 원래 분포를 유지해야 하므로 결측만 최빈값으로 처리
            df[col] = df[col].fillna(df[col].mode()[0])
            continue
        
        mean_val = df.loc[df[col] != 0, col].mean()  # 0이 아닌 값의 평균
        df[col] = df[col].replace(0, np.nan)         # 0을 NaN으로 변경
        df[col] = df[col].fillna(mean_val)
    df = pd.get_dummies(df, drop_first=True)
    # print(f"categoriacal_cols: {categoriacal_cols}")
    # print(f"numerical_cols: {numerical_cols}")
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

def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature Selection
    
    간단한 Feature Selection 예시:
    Variance(분산)가 거의 0인 컬럼 제거
    도메인 지식 기반 불필요 칼럼 제거 (예: ID 등)
	"""
	
    df = df.copy()
    # 분산이 거의 0인 컬럼 드랍
    low_variance_cols = df.loc[:, df.var() <= 1e-4].columns.tolist()
    
	#  타깃변수와 상관관계가 낮은 컬럼 드랍 (Correlation < 0.01)
    target_col = "Attrition_Binary"
    if target_col in df.columns:
        corr = df.corr()[target_col].abs().sort_values(ascending=False)
        low_corr_cols = corr[corr < 0.01].index.tolist()
        if target_col in low_corr_cols:
            low_corr_cols.remove(target_col)
    else:
        low_corr_cols = []
    drop_cols = list(set(low_variance_cols + low_corr_cols))
    df = df.drop(columns=drop_cols, errors='ignore')

    return df

def preprocess_pipeline(df, enginnered_feature_selection=True):
    # 전처리 파이프라인
    # 1. 필요없는 칼럼 드랍
    # 2. nan값 채우기

    df = drop_column(df)
    df = replace_nan_value(df)

    return df

def feature_engineering_pipeline(df):
	# Feature Engineering 파이프라인

    df = df.copy()
    df = add_feature_engineering(df)
    df = select_features(df)

    return df

import pandas as pd


if __name__ == "__main__":
    # df =load_data()
    # df = preprocess_pipeline(df)
    pass
