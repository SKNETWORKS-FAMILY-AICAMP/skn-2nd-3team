import pandas as pd
import streamlit as st
import os

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
    DATA_PATH = os.path.join(current_dir, '../../data/raw/BankChurners.csv')
    
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
    
    return df
