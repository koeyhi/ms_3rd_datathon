import streamlit as st
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

font_path = "streamlit/fonts/NanumGothic-Regular.ttf"

font_manager.fontManager.addfont(font_path)
plt.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()
plt.rcParams["axes.unicode_minus"] = False

pg_list = [
    st.Page("pg/home.py", title="홈"),
    st.Page("pg/dashboard.py", title="LCK 팀 지표 확인"),
    st.Page("pg/inference.py", title="경기 전 사전예측"),
    st.Page("pg/result.py", title="상관관계"),
    st.Page("pg/test.py", title="팀/선수 주요 지표 확인"),
]

pg = st.navigation(pg_list)
pg.run()
