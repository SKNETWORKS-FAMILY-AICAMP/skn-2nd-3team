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
from dashboard import show_dashboard

# Page Configuration
st.set_page_config(
    page_title="SKN 2기 3팀 - 고객 관리 시스템",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for "Web App" feel
st.markdown("""
    <style>
        /* Hide Streamlit Header and Footer */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Custom Font (Optional - using system defaults for now but making it clean) */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        
        /* Card-like style for metrics */
        div[data-testid="stMetric"] {
            background-color: #f0f2f6;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        
        /* Table styling adjustments */
        .stDataFrame {
            border: 1px solid #e0e0e0;
            border-radius: 5px;
        }
    </style>
""", unsafe_allow_html=True)

def main():
    # Load Data
    with st.spinner('데이터를 불러오는 중입니다...'):
        df = load_data()
    
    if df.empty:
        st.error("데이터를 불러올 수 없습니다.")
        return

    # Show Dashboard
    show_dashboard(df)

if __name__ == "__main__":
    main()
