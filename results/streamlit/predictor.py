import streamlit as st
import pandas as pd
import numpy as np
from utils import load_data

def predict_customer(df: pd.DataFrame):
    # 제목 및 설명
    st.title("💳 신용카드 고객 이탈 예측")
    st.markdown("""
    이 대시보드는 고객 정보를 입력받아 **이탈 가능성(Attrited)**을 예측합니다.\n
    데이터를 입력한 후 하단의 예측 버튼을 눌러주세요.
    """)

    st.divider()

    # --- 입력 폼 구성 (3단 레이아웃) ---
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("👤 인구통계 정보")
        customer_age = st.number_input("고객 나이 (Customer Age)", min_value=18, max_value=100, value=45)
        gender = st.selectbox("성별 (Gender)", ["M (남성)", "F (여성)"])
        dependent_count = st.slider("부양가족 수 (Dependent Count)", 0, 5, 2)
        education_level = st.selectbox("교육 수준 (Education Level)", 
                                    ["Uneducated", "High School", "College", "Graduate", "Post-Graduate", "Doctorate", "Unknown"])
        marital_status = st.selectbox("결혼 상태 (Marital Status)", 
                                    ["Single", "Married", "Divorced", "Unknown"])
        income_category = st.selectbox("소득 구간 (Income Category)", 
                                    ["Less than $40K", "$40K - $60K", "$60K - $80K", "$80K - $120K", "$120K +", "Unknown"])

    with col2:
        st.subheader("🏦 계좌 및 상품 정보")
        card_category = st.selectbox("카드 등급 (Card Category)", ["Blue", "Silver", "Gold", "Platinum"])
        months_on_book = st.number_input("가입 기간(개월) (Months on Book)", min_value=1, value=36)
        total_relationship_count = st.slider("보유 상품 수 (Total Relationship Count)", 1, 6, 3)
        credit_limit = st.number_input("신용 한도 (Credit Limit)", min_value=0.0, value=5000.0)
        total_revolving_bal = st.number_input("회전 신용 잔액 (Total Revolving Bal)", min_value=0, value=1000)
        avg_open_to_buy = st.number_input("사용 가능 한도 (Avg Open To Buy)", min_value=0.0, value=4000.0)

    with col3:
        st.subheader("📊 거래 및 활동 정보")
        total_trans_amt = st.number_input("총 거래 금액 (Total Trans Amt)", min_value=0, value=2000)
        total_trans_ct = st.number_input("총 거래 횟수 (Total Trans Ct)", min_value=0, value=50)
        total_amt_chng_q4_q1 = st.number_input("거래 금액 변동률 (Q4/Q1)", min_value=0.0, value=0.7, format="%.3f")
        total_ct_chng_q4_q1 = st.number_input("거래 횟수 변동률 (Q4/Q1)", min_value=0.0, value=0.6, format="%.3f")
        avg_utilization_ratio = st.slider("평균 한도 소진율 (Avg Utilization Ratio)", 0.0, 1.0, 0.3)
        months_inactive_12_mon = st.slider("지난 12개월 비활성 기간 (Months Inactive)", 0, 12, 2)
        contacts_count_12_mon = st.slider("지난 12개월 상담 횟수 (Contacts Count)", 0, 6, 2)

    st.divider()
    st.subheader("예측 시뮬레이션")
    # --- 예측 로직 --
    # 입력 데이터를 DataFrame으로 변환 (모델 학습 시 사용한 컬럼명과 일치해야 함)
    input_data = {
        'Customer_Age': [customer_age],
        'Gender': [gender[0]], # 'M (남성)' -> 'M'
        'Dependent_count': [dependent_count],
        'Education_Level': [education_level],
        'Marital_Status': [marital_status],
        'Income_Category': [income_category],
        'Card_Category': [card_category],
        'Months_on_book': [months_on_book],
        'Total_Relationship_Count': [total_relationship_count],
        'Months_Inactive_12_mon': [months_inactive_12_mon],
        'Contacts_Count_12_mon': [contacts_count_12_mon],
        'Credit_Limit': [credit_limit],
        'Total_Revolving_Bal': [total_revolving_bal],
        'Avg_Open_To_Buy': [avg_open_to_buy],
        'Total_Amt_Chng_Q4_Q1': [total_amt_chng_q4_q1],
        'Total_Trans_Amt': [total_trans_amt],
        'Total_Trans_Ct': [total_trans_ct],
        'Total_Ct_Chng_Q4_Q1': [total_ct_chng_q4_q1],
        'Avg_Utilization_Ratio': [avg_utilization_ratio]
    }

    df_input = pd.DataFrame(input_data)
    
    st.markdown("---")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        preprocess = st.selectbox("전처리", ["True", "False"])
    with col2:
        feature_engineering = st.selectbox("피쳐엔지니어링", ["True", "False"])
    with col3:
        cv = st.selectbox("크로스밸리데이션", ["True", "False"])
    with col4:
        tuning_strategy = st.selectbox("튜닝전략", [None, 'optuna', 'grid_search', 'random_search'])
    with col5:
        ensemble_strategy = st.selectbox("앙상블전략", ["voting", "stacking"])

    
    # 버튼 클릭 시 동작
    if st.button("🚀 이탈여부 예측(Predict)", type="secondary"):        
        st.subheader("입력 데이터 확인")
        st.dataframe(df_input)
        
        # ---------------------------------------------------------
        # [주의] 실제 모델 연동 부분
        # ---------------------------------------------------------
        # 1. 저장된 모델 불러오기 (예: joblib 또는 pickle 사용)
        # import joblib
        # model = joblib.load("my_best_xgboost_model.pkl")
        
        # 2. 전처리 (인코딩) 수행
        # 학습할 때 사용했던 OneHotEncoder나 LabelEncoder를 불러와서 transform 해야 합니다.
        # 예: df_input_processed = encoder.transform(df_input)
        
        # 3. 예측 수행 (여기서는 임시로 랜덤 값을 출력합니다)
        # prediction = model.predict(df_input_processed)
        # proba = model.predict_proba(df_input_processed)
        
        # --- 임시 결과 출력 (실제 모델 연결 시 삭제하세요) ---
        import random
        mock_pred = random.choice([0, 1])
        mock_proba = random.uniform(0.5, 0.99)
        # ---------------------------------------------------------

        st.subheader("예측 결과")

        # 실제 연결 시 if prediction[0] == 1: 로 변경
        if mock_pred == 1:
            st.error(f"⚠️ **이탈 위험 고객**입니다! (확률: {mock_proba*100:.2f}%)")
            st.write("제안: 고객 유지 프로모션을 제공하거나 상담을 진행하세요.")
        else:
            st.success(f"✅ **충성 고객 (유지)**으로 예상됩니다. (확률: {mock_proba*100:.2f}%)")

df = load_data()
predict_customer(df)