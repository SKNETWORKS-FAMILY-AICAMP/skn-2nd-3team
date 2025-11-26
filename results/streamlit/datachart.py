import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBClassifier


# =====================================================
# 1) 데이터 로드 & Soft Feature Engineering
# =====================================================
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\skn-2nd-3team\data\BankChurners.csv")

    df["Attrition_binary"] = df["Attrition_Flag"].map({
        "Existing Customer": 0,
        "Attrited Customer": 1
    })

    # Soft Feature Engineering (과적합 방지 목적)
    df["Activity_Index"] = np.log1p(df["Total_Trans_Amt"] * df["Total_Trans_Ct"])
    df["Avg_Transaction_Amount"] = np.log1p(df["Total_Trans_Amt"] / (df["Total_Trans_Ct"] + 1))
    df["Risk_Score"] = (
        df["Avg_Utilization_Ratio"] * 0.4 +
        np.log1p(df["Total_Revolving_Bal"]) * 0.6
    )
    df["Inactivity_Score"] = (
        df["Months_Inactive_12_mon"] * df["Avg_Utilization_Ratio"]
    )

    return df


# =====================================================
# 2) 모델 학습 (부드러운 확률 생성용)
# =====================================================
@st.cache_resource
def train_soft_model(df):
    X = df[[
        "Customer_Age",
        "Total_Trans_Amt",
        "Total_Trans_Ct",
        "Avg_Utilization_Ratio",
        "Total_Revolving_Bal",
        "Activity_Index",
        "Avg_Transaction_Amount",
        "Risk_Score",
        "Inactivity_Score"
    ]]
    y = df["Attrition_binary"]

    # 과적합 방지 + Probability smoothing
    model = XGBClassifier(
        n_estimators=120,
        learning_rate=0.08,
        max_depth=3,
        min_child_weight=5,
        subsample=0.75,
        colsample_bytree=0.75,
        reg_lambda=12,   # L2 regularization ↑
        reg_alpha=6,     # L1 regularization ↑
        gamma=3,         # node split penalty
        random_state=42
    )

    model.fit(X, y)
    return model, X.columns


# =====================================================
# 3) 확률 보정 함수 (Calibration)
# =====================================================
def calibrated_prediction(raw_prob):
    # Too sharp → soften
    return 0.15 + (raw_prob * 0.7)


# =====================================================
# 4) 게이지 그래프
# =====================================================
def churn_gauge(prob):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        title={'text': "이탈 확률 (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'steps': [
                {'range': [0, 30], 'color': 'lightgreen'},
                {'range': [30, 70], 'color': 'gold'},
                {'range': [70, 100], 'color': 'tomato'}
            ],
            'bar': {'color': "black"}
        }
    ))
    return fig


# =====================================================
# 5) Streamlit UI
# =====================================================
st.set_page_config(page_title="고객 단일 이탈 예측", layout="centered")
st.title("🔎 고객 단일 이탈 예측 대시보드 (Soft Prediction)")

df = load_data()
model, feature_cols = train_soft_model(df)

customer_idx = st.number_input(
    "고객 Row 선택 (0 ~ {}):".format(len(df)-1), 
    min_value=0, max_value=len(df)-1, value=0
)

customer = df.iloc[customer_idx][feature_cols].values.reshape(1, -1)

# 원본 예측
raw_prob = model.predict_proba(customer)[0][1]

# 보정된 예측
prob = calibrated_prediction(raw_prob)
prob = float(np.clip(prob, 0.01, 0.99))

# 그래프
st.plotly_chart(churn_gauge(prob), use_container_width=True)

# 결과 표시
if prob > 0.7:
    st.error(f"⚠ 고위험 고객 (예측 확률: {prob*100:.1f}%)")
elif prob > 0.4:
    st.warning(f"⚠ 중간 위험 고객 (예측 확률: {prob*100:.1f}%)")
else:
    st.success(f"✔ 낮은 위험 고객 (예측 확률: {prob*100:.1f}%)")



