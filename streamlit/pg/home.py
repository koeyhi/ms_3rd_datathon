import streamlit as st

st.set_page_config(page_title="Home", layout="wide")

st.markdown(
    """
    <div class="lck-hero">
        <span class="lck-chip">LCK ANALYTICS CENTER</span>
        <h1 class="lck-hero-title">3기 데이터톤 | LCK 인사이트 대시보드</h1>
        <p class="lck-hero-subtitle">
            팀/선수 퍼포먼스 지표와 밴픽 기반 예측을 결합해
            실전 전략 수립에 필요한 핵심 데이터를 한 화면에서 확인합니다.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3)
col1.markdown(
    """
    <div class="lck-card">
        <h3 class="lck-card-title">전력 분석</h3>
        <p class="lck-card-body">LCK 팀별 지표를 비교해 강점/약점과 메타 적응 흐름을 파악합니다.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
col2.markdown(
    """
    <div class="lck-card">
        <h3 class="lck-card-title">사전 예측</h3>
        <p class="lck-card-body">리그, 진영, 밴픽 입력으로 경기 전 승률을 빠르게 추정합니다.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
col3.markdown(
    """
    <div class="lck-card">
        <h3 class="lck-card-title">데이터 기반 의사결정</h3>
        <p class="lck-card-body">코칭 스태프와 팬이 같은 지표를 공유하며 전략 토론에 활용할 수 있습니다.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
