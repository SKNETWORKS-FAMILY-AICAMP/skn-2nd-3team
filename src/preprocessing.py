import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def load_data(filepath="data/raw/BankChurners.csv"):
	"""CSV 파일을 불러와 데이터프레임 반환"""
	df = pd.read_csv(filepath)
	return df


def preprocess_data(df):
	"""이탈 예측용 데이터 전처리 (기존 방식 - 하위 호환성)"""
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


def create_features(df):
	"""
	새로운 Feature 생성 (노트북 기반)
	
	Args:
		df: 원본 데이터프레임
		
	Returns:
		Feature가 추가된 데이터프레임
	"""
	df = df.copy()
	
	# 월 평균 거래 금액
	if 'Total_Trans_Amt' in df.columns and 'Total_Trans_Ct' in df.columns:
		df["Avg_Transaction_Amount"] = df["Total_Trans_Amt"] / (df["Total_Trans_Ct"] + 1)
	
	# 카드 사용 기간 대비 신용 한도 비율
	if 'Credit_Limit' in df.columns and 'Months_on_book' in df.columns:
		df["Credit_Utilization"] = df["Credit_Limit"] / (df["Months_on_book"] + 1)
	
	return df


def get_preprocessing_pipeline(X):
	"""
	ColumnTransformer 기반 전처리 파이프라인 생성 (노트북 기반)
	
	Args:
		X: 입력 데이터프레임 (타겟 변수 제외)
		
	Returns:
		전처리 파이프라인 (ColumnTransformer)
	"""
	# 컬럼 유형 분류
	numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
	categorical_cols = X.select_dtypes(include='object').columns.tolist()
	
	# 수치형 파이프라인
	numerical_pipeline = Pipeline([
		('imputer', SimpleImputer(strategy='mean')),
		('scaler', StandardScaler())
	])
	
	# 범주형 파이프라인
	categorical_pipeline = Pipeline([
		('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
	])
	
	# ColumnTransformer 생성
	preprocessor = ColumnTransformer(
		transformers=[
			('num', numerical_pipeline, numerical_cols),
			('cat', categorical_pipeline, categorical_cols)
		]
	)
	
	return preprocessor


def preprocess_with_pipeline(df, create_new_features=True):
	"""
	전처리 파이프라인을 사용한 데이터 전처리
	
	Args:
		df: 원본 데이터프레임
		create_new_features: 새로운 Feature 생성 여부
		
	Returns:
		전처리된 데이터프레임과 타겟 변수, 전처리 파이프라인
	"""
	df = df.copy()
	
	# 타겟 변수 생성
	df['Attrition_Binary'] = (df['Attrition_Flag'] == 'Attrited Customer').astype(int)
	
	# 불필요한 컬럼 제거
	drop_cols = [col for col in [
		'CLIENTNUM',
		'Attrition_Flag',
		"Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1",
		"Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2"
	] if col in df.columns]
	df = df.drop(columns=drop_cols)
	
	# 새로운 Feature 생성
	if create_new_features:
		df = create_features(df)
	
	# 타겟 변수 분리
	X = df.drop('Attrition_Binary', axis=1)
	y = df['Attrition_Binary']
	
	# 전처리 파이프라인 생성
	preprocessor = get_preprocessing_pipeline(X)
	
	return X, y, preprocessor
