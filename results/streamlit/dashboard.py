import streamlit as st
import pandas as pd

def show_dashboard(df: pd.DataFrame):
    """
    Displays the main dashboard with customer data.
    """
    st.markdown("## 👥 고객 정보 관리")
    
    # Top metrics
    total_customers = len(df)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="총 고객 수", value=f"{total_customers:,} 명")
    
    st.markdown("---")
    
    # Data Table
    st.markdown("### 고객 데이터 목록")
    st.dataframe(
        df,
        use_container_width=True,
        height=800,
        hide_index=True
    )
