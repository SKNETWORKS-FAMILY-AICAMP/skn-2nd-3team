import pandas as pd
import streamlit as st
import os
import sys

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.preprocessing import preprocess_pipeline, feature_engineering_pipeline

@st.cache_data
def load_data():
    """
    Loads the BankChurners data and removes specific columns.
    Columns to remove (1-based index from user request):
    - 2: Attrition_Flag
    - 22: Naive_Bayes_..._1
    - 23: Naive_Bayes_..._2
    """
    # Construct absolute path to data
    # The file is in results/streamlit/utils.py
    # Data is in data/raw/BankChurners.csv
    # So we need to go up two levels: ../../data/raw/BankChurners.csv
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(current_dir, '../../data/BankChurners.csv')
    
    if not os.path.exists(DATA_PATH):
        st.error(f"Data file not found at: {DATA_PATH}")
        return pd.DataFrame()

    df = pd.read_csv(DATA_PATH)

    # Reset CLIENTNUM to start from 1
    df['CLIENTNUM'] = range(1, len(df) + 1)
    
    # Columns to drop
    # 2 -> Attrition_Flag
    # 22 -> Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1
    # 23 -> Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2
    
    cols_to_drop = [
        'Attrition_Flag',
        'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
        'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'
    ]
    
    # Drop existing columns only
    existing_cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    df = df.drop(columns=existing_cols_to_drop)
    
    # Rename columns to Korean
    column_mapping = {
        "CLIENTNUM": "회원 ID",
        "Attrition_Flag": "이탈 여부",
        "Customer_Age": "나이",
        "Gender": "성별",
        "Dependent_count": "부양 가족 수",
        "Education_Level": "학력",
        "Marital_Status": "결혼 여부",
        "Income_Category": "소득 구간",
        "Card_Category": "카드 등급",
        "Months_on_book": "고객 관계 기간",
        "Total_Relationship_Count": "총 상품 수",
        "Months_Inactive_12_mon": "12개월 중 비활성화 개월 수",
        "Contacts_Count_12_mon": "12개월 중 Contact 횟수",
        "Credit_Limit": "신용한도",
        "Total_Revolving_Bal": "총 리볼빙 금액",
        "Avg_Open_To_Buy": "평균 사용가능 금액",
        "Total_Amt_Chng_Q4_Q1": "거래금액 변화율",
        "Total_Trans_Amt": "총 거래량",
        "Total_Trans_Ct": "총 거래 횟수",
        "Total_Ct_Chng_Q4_Q1": "거래 횟수 변화율",
        "Avg_Utilization_Ratio": "평균 신용 사용률",
        "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1": "예측 모델 1",
        "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2": "예측 모델 2"
    }
    
    df = df.rename(columns=column_mapping)
    
    return df

@st.cache_resource
def load_model():
    """
    Loads the trained Logistic Regression model.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, '../../results/Final_Model/final_model.joblib')
    
    if not os.path.exists(model_path):
        st.error(f"Model file not found at: {model_path}")
        return None
        
    import joblib
    return joblib.load(model_path)

def preprocess_for_inference(df, model):
    """
    Preprocesses the dataframe for inference using the main pipeline.
    """
    # 1. Revert Korean column names to English
    korean_to_english = {
        "회원 ID": "CLIENTNUM",
        "이탈 여부": "Attrition_Flag",
        "나이": "Customer_Age",
        "성별": "Gender",
        "부양 가족 수": "Dependent_count",
        "학력": "Education_Level",
        "결혼 여부": "Marital_Status",
        "소득 구간": "Income_Category",
        "카드 등급": "Card_Category",
        "고객 관계 기간": "Months_on_book",
        "총 상품 수": "Total_Relationship_Count",
        "12개월 중 비활성화 개월 수": "Months_Inactive_12_mon",
        "12개월 중 Contact 횟수": "Contacts_Count_12_mon",
        "신용한도": "Credit_Limit",
        "총 리볼빙 금액": "Total_Revolving_Bal",
        "평균 사용가능 금액": "Avg_Open_To_Buy",
        "거래금액 변화율": "Total_Amt_Chng_Q4_Q1",
        "총 거래량": "Total_Trans_Amt",
        "총 거래 횟수": "Total_Trans_Ct",
        "거래 횟수 변화율": "Total_Ct_Chng_Q4_Q1",
        "평균 신용 사용률": "Avg_Utilization_Ratio",
        "예측 모델 1": "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1",
        "예측 모델 2": "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2"
    }
    
    df_processed = df.rename(columns=korean_to_english)
    
    # 2. Apply Preprocessing Pipeline (from src/preprocessing.py)
    # This handles dropping columns and filling NaNs (and one-hot encoding)
    df_processed = preprocess_pipeline(df_processed)
    
    # 3. Apply Feature Engineering Pipeline (from src/preprocessing.py)
    # This adds new features and selects features
    df_processed = feature_engineering_pipeline(df_processed)
    
    # 4. Feature Alignment using model.feature_names_in_
    if hasattr(model, 'feature_names_in_'):
        # Drop extra columns
        extra_cols = set(df_processed.columns) - set(model.feature_names_in_)
        df_processed = df_processed.drop(columns=list(extra_cols))
        
        # Reorder columns
        df_processed = df_processed[model.feature_names_in_]
    
    return df_processed

def predict_churn(model, df):
    """
    Predicts churn probability and risk status.
    Returns the original dataframe with new columns: '이탈 확률', '이탈 위험'
    """
    df_out = df.copy()
    
    # Preprocess
    X = preprocess_for_inference(df, model)
    
    # Handle potential attribute errors in ensemble models

    # Predict
    probs = model.predict_proba(X)[:, 1] # Probability of class 1 (Attrition)

    # Add to dataframe
    df_out['이탈 확률'] = probs
    df_out['이탈 위험'] = probs > 0.8 # Threshold 0.8
    
    return df_out