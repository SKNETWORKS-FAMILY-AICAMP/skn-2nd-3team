import streamlit as st
import pandas as pd
from utils import load_data, load_model, predict_churn

def show_dashboard(df: pd.DataFrame):
    st.markdown("## 👥 고객 정보 관리")

    if 'prediction_done' not in st.session_state:
        st.session_state.prediction_done = False
        st.session_state.df_result = df
        
    if st.button("🔍 이탈 위험 예측 (AI Inference)", type="primary"):
        with st.spinner("AI 모델이 고객 데이터를 분석 중입니다..."):
            model = load_model()
            if model:
                result_df = predict_churn(model, df)
                st.session_state.df_result = result_df
                st.session_state.prediction_done = True
                st.success("분석이 완료되었습니다!")
            else:
                st.error("모델을 불러올 수 없습니다.")

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

    st.markdown("### 고객 데이터 목록")

    column_config = {
        "신용한도": st.column_config.NumberColumn(format="%.1f"),
        "평균 사용가능 금액": st.column_config.NumberColumn(format="%.1f"),
        "총 리볼빙 금액": st.column_config.NumberColumn(format="%d"),
        "총 거래량": st.column_config.NumberColumn(format="%d"),
        "총 거래 횟수": st.column_config.NumberColumn(format="%d"),
        "거래금액 변화율": st.column_config.NumberColumn(format="%.3f"),
        "거래 횟수 변화율": st.column_config.NumberColumn(format="%.3f"),
        "평균 신용 사용률": st.column_config.NumberColumn(format="%.3f"),
        "이탈 확률": st.column_config.NumberColumn(format="%.2%"),
    }

    PAGE_SIZE = 2000
    total_rows = len(df_display)
    total_pages = (total_rows - 1) // PAGE_SIZE + 1

    if 'current_page' not in st.session_state:
        st.session_state.current_page = 1
        
    if st.session_state.current_page > total_pages:
        st.session_state.current_page = total_pages
    if st.session_state.current_page < 1:
        st.session_state.current_page = 1
        
    current_page = st.session_state.current_page
        
    start_idx = (current_page - 1) * PAGE_SIZE
    end_idx = min(start_idx + PAGE_SIZE, total_rows)
    df_page = df_display.iloc[start_idx:end_idx]

    st.caption(f"총 {total_rows:,}개 데이터 중 {start_idx+1:,} - {end_idx:,} 표시")

    if st.session_state.prediction_done:
        def highlight_risk(row):
            if row['이탈 위험']:
                return ['background-color: #ffcdd2'] * len(row)
            return [''] * len(row)
            
        st.dataframe(
            df_page.style.apply(highlight_risk, axis=1),
            use_container_width=True,
            height=800,
            hide_index=True,
            column_config=column_config
        )
    else:
        st.dataframe(
            df_page,
            use_container_width=True,
            height=800,
            hide_index=True,
            column_config=column_config
        )

    if total_pages > 1:
        st.markdown("""
            <style>
                /* Target secondary buttons (pagination) */
                button[kind="secondary"] {
                    border: none !important;
                    background: transparent !important;
                    box-shadow: none !important;
                    padding: 0px 10px !important;
                    color: #555 !important;
                }
                button[kind="secondary"]:hover {
                    color: #ff4b4b !important;
                    background: transparent !important;
                }
                /* Disable button style (current page) */
                button[disabled] {
                    color: #000 !important;
                    font-weight: bold !important;
                    background: transparent !important;
                    border: none !important;
                    opacity: 1 !important;
                }
            </style>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True) 
        
        _, col_center, _ = st.columns([5, 3, 5])
        
        with col_center:
            max_buttons = 5
            start_page = ((current_page - 1) // max_buttons) * max_buttons + 1
            end_page = min(start_page + max_buttons - 1, total_pages)
            
            num_buttons = end_page - start_page + 1 + 4 
            cols = st.columns(num_buttons)
            
            if cols[0].button("⏮", disabled=(current_page == 1), key="first_page"):
                st.session_state.current_page = 1
                st.rerun()
            
            if cols[1].button("◀", disabled=(start_page == 1), key="prev_page"):
                st.session_state.current_page = max(1, start_page - 1)
                st.rerun()
            
            for i, page_num in enumerate(range(start_page, end_page + 1)):
                is_current = (page_num == current_page)
                label = f"{page_num}" 
                
                if cols[i+2].button(label, key=f"page_{page_num}", disabled=is_current):
                    st.session_state.current_page = page_num
                    st.rerun()
            
            if cols[-2].button("▶", disabled=(end_page == total_pages), key="next_page"):
                st.session_state.current_page = min(total_pages, end_page + 1)
                st.rerun()
            
            if cols[-1].button("⏭", disabled=(current_page == total_pages), key="last_page"):
                st.session_state.current_page = total_pages
                st.rerun()


df = load_data()
show_dashboard(df)