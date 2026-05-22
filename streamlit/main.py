import streamlit as st
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
from pathlib import Path

font_path = "streamlit/fonts/NanumGothic-Regular.ttf"

font_manager.fontManager.addfont(font_path)
plt.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()
plt.rcParams["axes.unicode_minus"] = False


def apply_lck_theme():
    css_path = Path(__file__).with_name("lck_theme.css")
    if css_path.exists():
        theme_css = css_path.read_text(encoding="utf-8")
        st.markdown(f"<style>{theme_css}</style>", unsafe_allow_html=True)


apply_lck_theme()

pg_list = [
    st.Page("pg/home.py", title="홈"),
    st.Page("pg/dashboard.py", title="LCK 팀 지표 확인"),
    st.Page("pg/inference.py", title="경기 전 사전예측"),
    st.Page("pg/result.py", title="상관관계"),
    st.Page("pg/test.py", title="팀/선수 주요 지표 확인"),
]

pg = st.navigation(pg_list)
pg.run()
