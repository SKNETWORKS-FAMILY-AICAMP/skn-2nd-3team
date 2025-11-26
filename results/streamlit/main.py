import streamlit as st
import sys
import os
from utils import load_data

st.set_page_config(
    page_title="SKN 2기 3팀 - 고객 관리 시스템",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)
home    = st.Page("./dashboard.py", title="대시보드",   icon="🏠")
message_center = st.Page("./message_center.py", title="고객 관리 메세지", icon="⚙️")
data_chart = st.Page("./data_chart.py", title="고객이탈 확률 피드백", icon="📊")

nav = st.navigation([home, message_center, data_chart])
nav.run()