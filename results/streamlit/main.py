import streamlit as st
import sys
import os

# Add the project root to sys.path to allow imports from src
# Assuming this script is run from the project root or results/streamlit
# We need to ensure 'src' is importable.
# Current file: results/streamlit/main.py
# Project root: ../../
# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.abspath(os.path.join(current_dir, "../../"))
# if project_root not in sys.path:
#     sys.path.append(project_root)

from utils import load_data

# Page Configuration
st.set_page_config(
    page_title="SKN 2기 3팀 - 고객 관리 시스템",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.set_page_config(page_title="SKN 2기 3팀 - 고객 관리 시스템")
home    = st.Page("./dashboard.py", title="대시보드",   icon="🏠")
test_sample  = st.Page("./predictor.py",  title="고객이탈 예측",   icon="🚗")
message_center = st.Page("./message_center.py", title="고객 관리 메세지", icon="⚙️")
data_chart = st.Page("./data_chart.py", title="고객이탈 확률 피드백", icon="📊")

nav = st.navigation([home, test_sample, message_center, data_chart])
nav.run()

# def main():
#     # Load Data
#     with st.spinner('데이터를 불러오는 중입니다...'):
#         df = load_data()
    
#     if df.empty:
#         st.error("데이터를 불러올 수 없습니다.")
#         return

#     # Show Dashboard
#     # tester()

# if __name__ == "__main__":
#     main()
