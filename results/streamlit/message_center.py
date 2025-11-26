import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import time
from utils import load_data, load_model, predict_churn


def show_message_center(df: pd.DataFrame):
    """
    고객 메시지 발송 센터
    """
    st.markdown("## 📱 고객 메시지 발송 센터")
    
    # Initialize session state
    if 'sent_messages' not in st.session_state:
        st.session_state.sent_messages = []
    
    # 이탈 위험 고객만 필터링
    if '이탈 위험' in df.columns:
        at_risk_df = df[df['이탈 위험'] == True].copy()
    else:
        st.warning("⚠️ 먼저 Dashboard에서 '이탈 위험 예측'을 실행해주세요!")
        return
    
    if len(at_risk_df) == 0:
        st.info("🎉 현재 이탈 위험 고객이 없습니다!")
        return
    
    # 통계 표시
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("이탈 위험 고객", f"{len(at_risk_df):,}명")
    with col2:
        high_risk = len(at_risk_df[at_risk_df['이탈 확률'] >= 0.7])
        st.metric("고위험 고객 (70%+)", f"{high_risk:,}명", delta="긴급", delta_color="inverse")
    with col3:
        st.metric("발송 완료", f"{len(st.session_state.sent_messages):,}건")
    
    st.markdown("---")
    
    # 탭 구성
    tab1, tab2, tab3 = st.tabs(["📤 메시지 발송", "📋 메시지 템플릿", "📊 발송 이력"])
    
    with tab1:
        show_send_message_tab(at_risk_df)
    
    with tab2:
        show_template_tab()
    
    with tab3:
        show_history_tab()


def show_send_message_tab(at_risk_df: pd.DataFrame):
    """메시지 발송 탭"""
    st.markdown("### 1️⃣ 타겟 고객 선택")
    
    col1, col2 = st.columns(2)
    
    with col1:
        risk_level = st.selectbox(
            "위험 등급",
            ["전체", "고위험 (70%+)", "중위험 (50-70%)", "저위험 (50% 미만)"]
        )
    
    with col2:
        customer_count = st.number_input(
            "발송 대상 수",
            min_value=1,
            max_value=len(at_risk_df),
            value=min(10, len(at_risk_df))
        )
    
    # 위험 등급별 필터링
    if risk_level == "고위험 (70%+)":
        filtered_df = at_risk_df[at_risk_df['이탈 확률'] >= 0.7]
    elif risk_level == "중위험 (50-70%)":
        filtered_df = at_risk_df[(at_risk_df['이탈 확률'] >= 0.5) & (at_risk_df['이탈 확률'] < 0.7)]
    elif risk_level == "저위험 (50% 미만)":
        filtered_df = at_risk_df[at_risk_df['이탈 확률'] < 0.5]
    else:
        filtered_df = at_risk_df
    
    # 이탈 확률 높은 순으로 정렬
    filtered_df = filtered_df.sort_values('이탈 확률', ascending=False).head(customer_count)
    
    st.markdown(f"**선택된 고객: {len(filtered_df)}명**")
    
    # 선택된 고객 미리보기
    with st.expander("👥 선택된 고객 목록 보기"):
        display_cols = ['CLIENTNUM', '이탈 확률', '신용한도', '총 거래량', '총 거래 횟수']
        st.dataframe(
            filtered_df[display_cols],
            use_container_width=True,
            hide_index=True
        )
    
    st.markdown("---")
    st.markdown("### 2️⃣ 메시지 작성")
    
    # 메시지 템플릿 선택
    template_options = {
        "직접 작성": "",
        "VIP 특별 혜택": "🎁 [고객명]님, VIP 회원님만을 위한 특별 혜택을 준비했습니다! 이번 달 신용카드 사용 시 최대 30% 캐시백 혜택을 받으세요. 자세한 내용은 앱에서 확인하세요!",
        "이용료 할인": "💳 [고객명]님, 소중한 고객님께 연회비 50% 할인 혜택을 드립니다. 지금 바로 앱에서 확인하시고 혜택을 받아가세요!",
        "포인트 적립 프로모션": "⭐ [고객명]님, 이번 주 특별 이벤트! 모든 결제 건에 포인트 2배 적립! 놓치지 마세요!",
        "맞춤형 추천": "✨ [고객명]님의 소비 패턴을 분석한 결과, 회원님께 딱 맞는 맞춤형 혜택을 준비했습니다. 지금 확인해보세요!"
    }
    
    selected_template = st.selectbox("메시지 템플릿", list(template_options.keys()))
    
    message_text = st.text_area(
        "메시지 내용",
        value=template_options[selected_template],
        height=150,
        help="[고객명]은 자동으로 고객 이름으로 치환됩니다."
    )
    
    # 발송 예약
    col1, col2 = st.columns(2)
    with col1:
        send_now = st.checkbox("즉시 발송", value=True)
    with col2:
        if not send_now:
            scheduled_time = st.time_input("예약 시간", value=datetime.now().time())
    
    st.markdown("---")
    
    # 발송 버튼
    if st.button("📤 메시지 발송", type="primary", use_container_width=True):
        if not message_text:
            st.error("메시지 내용을 입력해주세요!")
        else:
            with st.spinner("메시지 발송 중..."):
                # 시뮬레이션
                progress_bar = st.progress(0)
                for i in range(len(filtered_df)):
                    time.sleep(0.05)  # 발송 시뮬레이션
                    progress_bar.progress((i + 1) / len(filtered_df))
                
                # 발송 기록 저장
                send_time = datetime.now() if send_now else datetime.combine(
                    datetime.now().date(), scheduled_time
                )
                
                for _, row in filtered_df.iterrows():
                    st.session_state.sent_messages.append({
                        'customer_id': row['CLIENTNUM'],
                        'risk_level': row['이탈 확률'],
                        'message': message_text,
                        'sent_time': send_time,
                        'status': '발송 완료' if send_now else '예약됨'
                    })
                
                st.success(f"✅ {len(filtered_df)}명의 고객에게 메시지가 {'발송' if send_now else '예약'}되었습니다!")


def show_template_tab():
    """메시지 템플릿 관리 탭"""
    st.markdown("### 💬 메시지 템플릿 라이브러리")
    
    templates = [
        {
            "제목": "🎁 VIP 특별 혜택",
            "내용": "[고객명]님, VIP 회원님만을 위한 특별 혜택을 준비했습니다!",
            "카테고리": "프로모션",
            "예상 반응률": "23%"
        },
        {
            "제목": "💳 연회비 할인",
            "내용": "[고객명]님, 소중한 고객님께 연회비 50% 할인 혜택을 드립니다.",
            "카테고리": "리텐션",
            "예상 반응률": "31%"
        },
        {
            "제목": "⭐ 포인트 2배 적립",
            "내용": "[고객명]님, 이번 주 특별 이벤트! 모든 결제 건에 포인트 2배 적립!",
            "카테고리": "프로모션",
            "예상 반응률": "19%"
        },
        {
            "제목": "🏆 우수 고객 감사",
            "내용": "[고객명]님, 항상 저희 카드를 이용해주셔서 감사합니다. 특별 혜택을 드립니다.",
            "카테고리": "감사",
            "예상 반응률": "27%"
        }
    ]
    
    for i, template in enumerate(templates):
        with st.expander(f"{template['제목']}"):
            st.markdown(f"**카테고리:** {template['카테고리']}")
            st.markdown(f"**예상 반응률:** {template['예상 반응률']}")
            st.markdown(f"**내용:**\n{template['내용']}")
            
            col1, col2 = st.columns([1, 3])
            with col1:
                if st.button("사용하기", key=f"use_template_{i}"):
                    st.info("메시지 발송 탭으로 이동하여 이 템플릿을 선택하세요!")


def show_history_tab():
    """발송 이력 탭"""
    st.markdown("### 📊 메시지 발송 이력")
    
    if len(st.session_state.sent_messages) == 0:
        st.info("아직 발송된 메시지가 없습니다.")
        return
    
    # 발송 이력 데이터프레임 생성
    history_df = pd.DataFrame(st.session_state.sent_messages)
    
    # 통계
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("총 발송 건수", f"{len(history_df):,}건")
    with col2:
        completed = len(history_df[history_df['status'] == '발송 완료'])
        st.metric("발송 완료", f"{completed:,}건")
    with col3:
        scheduled = len(history_df[history_df['status'] == '예약됨'])
        st.metric("예약됨", f"{scheduled:,}건")
    with col4:
        # 시뮬레이션: 반응률 (실제로는 고객 반응 데이터 필요)
        response_rate = 23  # 예시
        st.metric("평균 반응률", f"{response_rate}%")
    
    st.markdown("---")
    
    # 발송 이력 테이블
    display_df = history_df.copy()
    display_df['sent_time'] = display_df['sent_time'].dt.strftime('%Y-%m-%d %H:%M')
    display_df['risk_level'] = display_df['risk_level'].apply(lambda x: f"{x:.1%}")
    
    st.dataframe(
        display_df[['customer_id', 'risk_level', 'sent_time', 'status', 'message']],
        use_container_width=True,
        hide_index=True,
        column_config={
            'customer_id': '고객 ID',
            'risk_level': '이탈 확률',
            'sent_time': '발송 시간',
            'status': '상태',
            'message': '메시지'
        }
    )
    
    # 이력 초기화 버튼
    if st.button("🗑️ 발송 이력 초기화", type="secondary"):
        st.session_state.sent_messages = []
        st.rerun()

df = load_data() # utils.py의 load_data 함수를 사용하여 데이터 로드
if not df.empty:
    show_message_center(df) # 정의된 메인 함수 호출