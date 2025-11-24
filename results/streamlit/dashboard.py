import streamlit as st
import pandas as pd
from utils import load_model, predict_churn

def show_dashboard(df: pd.DataFrame):
    """
    Displays the main dashboard with customer data.
    """
    st.markdown("## 👥 고객 정보 관리")
    
    # Initialize session state for prediction results
    if 'prediction_done' not in st.session_state:
        st.session_state.prediction_done = False
        st.session_state.df_result = df
        
    # Inference Button
    if st.button("🔍 이탈 위험 예측 (AI Inference)"):
        with st.spinner("AI 모델이 고객 데이터를 분석 중입니다..."):
            model = load_model()
            if model:
                # Run prediction
                result_df = predict_churn(model, df)
                st.session_state.df_result = result_df
                st.session_state.prediction_done = True
                st.success("분석이 완료되었습니다!")
            else:
                st.error("모델을 불러올 수 없습니다.")

    # Top metrics
    total_customers = len(df)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="총 고객 수", value=f"{total_customers:,} 명")
        
    if st.session_state.prediction_done:
        df_display = st.session_state.df_result
        at_risk_count = df_display['이탈 위험'].sum()
        with col2:
            st.metric(label="⚠️ 이탈 위험 회원 수", value=f"{at_risk_count:,} 명", delta=f"{at_risk_count} 명 위험", delta_color="inverse")
    else:
        df_display = df

    st.markdown("---")
    
    # Data Table
    st.markdown("### 고객 데이터 목록")
    
    # Apply styling if prediction is done
    if st.session_state.prediction_done:
        # Highlight rows where '이탈 위험' is True
        def highlight_risk(row):
            if row['이탈 위험']:
                return ['background-color: #ffcdd2'] * len(row)
            return [''] * len(row)
            
        st.dataframe(
            df_display.style.apply(highlight_risk, axis=1),
            use_container_width=True,
            height=800,
            hide_index=True
        )
    else:
        st.dataframe(
            df_display,
            use_container_width=True,
            height=800,
            hide_index=True
        )
