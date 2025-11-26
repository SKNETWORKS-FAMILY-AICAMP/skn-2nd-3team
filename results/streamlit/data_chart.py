import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
import os


import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from xgboost import XGBClassifier
import os

if "idx" not in st.session_state:
    st.session_state.idx = 0

if "last_new_inputs" not in st.session_state:
    st.session_state.last_new_inputs = None

def stable_input(key, default):
    """입력값 유지하여 재랜더링 깜빡임 방지"""
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]


# =====================================================
# 1) 데이터 로드 + Soft Feature Engineering
# =====================================================
@st.cache_data
def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '../../data/BankChurners.csv')
    df = pd.read_csv(data_path)

    df["Attrition_binary"] = df["Attrition_Flag"].map({
        "Existing Customer": 0,
        "Attrited Customer": 1
    })

    # Soft Feature Engineering
    df["Activity_Index"] = np.log1p(df["Total_Trans_Amt"] * df["Total_Trans_Ct"])
    df["Avg_Transaction_Amount"] = np.log1p(df["Total_Trans_Amt"] / (df["Total_Trans_Ct"] + 1))
    df["Risk_Score"] = (
        df["Avg_Utilization_Ratio"] * 0.4 +
        np.log1p(df["Total_Revolving_Bal"]) * 0.6
    )
    df["Inactivity_Score"] = df["Months_Inactive_12_mon"] * df["Avg_Utilization_Ratio"]

    return df


# =====================================================
# 2) 모델 학습 (Soft Model)
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

    model = XGBClassifier(
        n_estimators=120,
        learning_rate=0.08,
        max_depth=3,
        min_child_weight=5,
        subsample=0.75,
        colsample_bytree=0.75,
        reg_lambda=12,
        reg_alpha=6,
        gamma=3,
        random_state=42
    )

    model.fit(X, y)
    return model, X.columns


# =====================================================
# 3) 확률 보정 함수
# =====================================================
def calibrated_prediction(raw_prob):
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
# 5) 피드백 생성 함수
# =====================================================
def generate_feedback(row, prob):
    fb = []

    if prob > 0.7:
        fb.append("고객의 이탈 위험이 높습니다. 즉각적인 대응이 필요합니다.")
    elif prob > 0.4:
        fb.append("중간 수준의 이탈 위험이 있습니다. 지속적인 관찰이 필요합니다.")
    else:
        fb.append("이 고객은 비교적 안정적인 상태입니다.")

    # 행동 지표 기반 피드백
    if row["Avg_Utilization_Ratio"] > 0.6:
        fb.append("· 신용 사용률이 높아 재정적 스트레스를 느낄 수 있습니다.")
    if row["Months_Inactive_12_mon"] > 3:
        fb.append("· 비활성 기간이 길어 서비스 관심도가 낮아졌습니다.")
    if row["Total_Ct_Chng_Q4_Q1"] < 0.8:
        fb.append("· 최근 거래 빈도 감소가 확인됩니다.")
    if row["Contacts_Count_12_mon"] > 3:
        fb.append("· 고객센터 문의 증가 → 불만 가능성 높음.")
    if row["Total_Trans_Ct"] > 60:
        fb.append("· 거래량 많음 → 충성고객 잠재력 높음.")

    return fb


# =====================================================
# 6) 액션 플랜 생성 함수
# =====================================================
def generate_action_plan(prob):
    act = []

    if prob > 0.7:
        act.append("🔴 즉시 VIP 전문 상담 배정")
        act.append("🔴 맞춤형 혜택 또는 한도 조정 제공")
        act.append("🔴 불만 해소를 위한 콜백/CS 강화")
        act.append("🔴 재참여 유도 캠페인 발송")
    elif prob > 0.4:
        act.append("🟠 리마인드 마케팅 발송")
        act.append("🟠 포인트/쿠폰 제공")
        act.append("🟠 사용량 회복을 위한 맞춤 추천 제공")
    else:
        act.append("🟢 충성도 프로그램 제공")
        act.append("🟢 맞춤 서비스 자동 추천")
        act.append("🟢 장기 혜택 유지로 만족도 강화")

    return act


# =====================================================
# 7) Streamlit UI (통합 버전)
# =====================================================
st.set_page_config(page_title="고객 이탈 예측 통합 대시보드", layout="centered")
st.title("🔎 고객 이탈 예측 통합 대시보드 (기존 + 신규 고객)")

df = load_data()
model, feature_cols = train_soft_model(df)

# -----------------------------------------------------
# 분석 유형 선택
# -----------------------------------------------------
mode = st.radio(
    "분석할 고객 유형 선택:",
    ("👥 기존 고객 분석", "🆕 신규 고객 분석")
)

# -----------------------------------------------------
# 기존 고객 분석
# -----------------------------------------------------
if mode == "👥 기존 고객 분석":

    idx = st.number_input(
    "기존 고객 Row 선택 (0 ~ {}):".format(len(df)-1),
    min_value=0, max_value=len(df)-1,
    key="idx"
)

    row = df.iloc[idx]
    model_input = row[feature_cols].values.reshape(1, -1)

    raw = model.predict_proba(model_input)[0][1]
    prob = float(np.clip(calibrated_prediction(raw), 0.01, 0.99))

    st.plotly_chart(churn_gauge(prob), use_container_width=True)

    # 위험도 표시
    if prob > 0.7:
        st.error(f"⚠ 고위험 고객 ({prob*100:.1f}%)")
    elif prob > 0.4:
        st.warning(f"⚠ 중간 위험 고객 ({prob*100:.1f}%)")
    else:
        st.success(f"✔ 낮은 위험 고객 ({prob*100:.1f}%)")

    # 피드백
    st.markdown("### 📝 고객 맞춤 피드백")
    for fb in generate_feedback(row, prob):
        st.write(f"- {fb}")

    # 전략
    st.markdown("### 🎯 고객 이탈 방지 전략")
    for ac in generate_action_plan(prob):
        st.write(f"- {ac}")


# -----------------------------------------------------
# 신규 고객 분석
# -----------------------------------------------------
else:

    st.markdown("### 🆕 신규 고객 정보 입력")

    col1, col2 = st.columns(2)
    with col1:

        # 신규 입력값 깜빡임 방지 + 기본값 지정
        if "age" not in st.session_state:
            st.session_state.age = 35

        age = st.number_input("나이", 18, 100, key="age")

        trans_amt = st.number_input("총 거래 금액", 0, 100000, 5000)
        trans_ct = st.number_input("총 거래 횟수", 0, 200, 50)
        util = st.number_input("평균 신용 사용률", 0.0, 1.0, 0.3)

    with col2:

        revolve = st.number_input("리볼빙 잔액", 0, 100000, 1200)
        inactive = st.number_input("비활성 개월수", 0, 12, 1)
        contact = st.number_input("문의 횟수", 0, 20, 1)
        ct_chg = st.number_input("거래 변화율", 0.0, 3.0, 1.0)

    # 신규 고객 Feature Engineering
    Aindex = np.log1p(trans_amt * trans_ct)
    Aavg = np.log1p(trans_amt / (trans_ct + 1))
    Rscore = (util * 0.4) + (np.log1p(revolve) * 0.6)
    Iscore = inactive * util

    new_input = np.array([
        age, trans_amt, trans_ct, util, revolve,
        Aindex, Aavg, Rscore, Iscore
    ]).reshape(1, -1)

    if st.button("신규 고객 예측하기"):
        raw_new = model.predict_proba(new_input)[0][1]
        prob_new = float(np.clip(calibrated_prediction(raw_new), 0.01, 0.99))

        st.plotly_chart(churn_gauge(prob_new), use_container_width=True)

        if prob_new > 0.7:
            st.error(f"⚠ 고위험 신규 고객 ({prob_new*100:.1f}%)")
        elif prob_new > 0.4:
            st.warning(f"⚠ 중간 위험 신규 고객 ({prob_new*100:.1f}%)")
        else:
            st.success(f"✔ 낮은 위험 신규 고객 ({prob_new*100:.1f}%)")

        # 신규 고객 피드백
        new_row = {
            "Avg_Utilization_Ratio": util,
            "Months_Inactive_12_mon": inactive,
            "Total_Ct_Chng_Q4_Q1": ct_chg,
            "Contacts_Count_12_mon": contact,
            "Total_Trans_Ct": trans_ct
        }

        st.markdown("### 📝 신규 고객 피드백")
        for fb in generate_feedback(new_row, prob_new):
            st.write(f"- {fb}")

        # 전략
        st.markdown("### 🎯 신규 고객 액션 플랜")
        for ac in generate_action_plan(prob_new):
            st.write(f"- {ac}")

# =====================================================
# 🔧 화면 깜빡임 최소화 패치
# =====================================================

# Streamlit은 입력값 변화마다 전체 페이지를 rerun 하므로
# session_state를 활용하여 rerun 횟수 최소화

if "last_idx" not in st.session_state:
    st.session_state.last_idx = None

if "last_new_inputs" not in st.session_state:
    st.session_state.last_new_inputs = None

def stable_input(key, default):
    """입력값이 변해도 UI 전체가 깜빡이지 않도록 안정적으로 저장"""
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]