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


# =====================================================
# 🔥 고객 피드백 생성 함수 (추가)
# =====================================================
def generate_feedback(row, prob):
    feedback = []

    # 1) 확률 기반 피드백
    if prob > 0.7:
        feedback.append("고객의 이탈 위험이 높습니다. 즉각적인 대응이 필요합니다.")
    elif prob > 0.4:
        feedback.append("중간 수준의 이탈 위험이 있습니다. 지속적인 관찰이 필요합니다.")
    else:
        feedback.append("이 고객은 비교적 안정적인 상태입니다.")

    # 2) 주요 행동 지표 기반 피드백
    if row["Avg_Utilization_Ratio"] > 0.6:
        feedback.append("· 신용 사용률이 높아 재정적 스트레스를 느낄 수 있습니다.")
    if row["Months_Inactive_12_mon"] > 3:
        feedback.append("· 비활성 기간이 길어 서비스에 대한 관심도가 낮을 수 있습니다.")
    if row["Total_Ct_Chng_Q4_Q1"] < 0.8:
        feedback.append("· 최근 거래 빈도 감소가 확인되었습니다.")
    if row["Contacts_Count_12_mon"] > 3:
        feedback.append("· 고객센터 문의 빈도가 높아 불만 가능성이 있습니다.")
    if row["Total_Trans_Ct"] > 60:
        feedback.append("· 거래 횟수가 매우 높아 충성도 잠재력이 있습니다.")

    return feedback


# =====================================================
# 🔥 고객 피드백 출력 (추가)
# =====================================================
st.markdown("---")
st.markdown("### 📝 고객 맞춤 피드백")

selected_row = df.iloc[customer_idx]
feedback_list = generate_feedback(selected_row, prob)

for fb in feedback_list:
    st.write(f"- {fb}")

# =====================================================
# 🔥 위험도별 직접적 해결 전략(추천 Action) 함수 추가
# =====================================================
def generate_action_plan(prob):
    actions = []

    if prob > 0.7:  # 고위험
        actions.append("🔴 즉시 VIP 상담 또는 개인 맞춤 제안 제공")
        actions.append("🔴 이탈 고객 패턴을 기반으로 한 리텐션 캠페인 발송")
        actions.append("🔴 높은 리볼빙 잔액/사용률 고객에게 금리 혜택 또는 한도 조정 제안")
        actions.append("🔴 최근 불만/문의 증가 시, 빠른 CS 담당자 배정 필요")

    elif prob > 0.4:  # 중간 위험
        actions.append("🟠 고객 활동성 회복을 위한 리마인드 마케팅 발송")
        actions.append("🟠 맞춤형 혜택(포인트/쿠폰/캐시백) 제공")
        actions.append("🟠 최근 거래 감소 고객 대상 서비스/상품 재참여 유도 메시지 발송")
        actions.append("🟠 비활성 고객에게 상품 추천 또는 사용 튜토리얼 제공")

    else:  # 저위험
        actions.append("🟢 고객 충성도 향상 프로그램 제공 (등급 상승 안내)")
        actions.append("🟢 최근 활동 패턴 기반 추천 서비스 자동 발송")
        actions.append("🟢 긍정적 고객 경험 유지: 알림/혜택 제공 적정 유지")
        actions.append("🟢 장기 고객 혜택으로 충성도 강화")

    return actions


# =====================================================
# 🔥 해결 전략(Action Plan) 출력
# =====================================================
st.markdown("### 🎯 고객 이탈 방지 전략 (Action Plan)")

action_plan = generate_action_plan(prob)

for act in action_plan:
    st.write(f"- {act}")
