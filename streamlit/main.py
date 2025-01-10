import streamlit as st
import os
import streamlit as st

st.write(f"현재 작업 디렉토리: {os.getcwd()}")
st.write("디렉토리 내 파일 목록:")
st.write(os.listdir())

pg_list = [
    st.Page("pg/home.py", title="홈"),
    st.Page("pg/dashboard.py", title="대시보드"),
    st.Page("pg/inference.py", title="예측"),
]

pg = st.navigation(pg_list)
pg.run()
