import pandas as pd
import numpy as np

def load_data(filepath="data/raw/BankChurners.csv"):
	"""CSV 파일을 불러와 데이터프레임 반환"""
	df = pd.read_csv(filepath)
	return df

def preprocess_data(df):
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
	for col in df.select_dtypes(include='object').columns:
		df[col] = df[col].fillna(df[col].mode()[0])
	for col in df.select_dtypes(include=np.number).columns:
		df[col] = df[col].fillna(df[col].mean())
	df = pd.get_dummies(df, drop_first=True)
	return df
