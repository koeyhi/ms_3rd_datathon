import streamlit as st

pg_list = [
    st.Page("pg/home.py", title="홈"),
    st.Page("pg/dashboard.py", title="대시보드"),
    st.Page("pg/inference.py", title="예측"),
    st.Page("pg/result.py", title="결과"),
    st.Page("pg/player.py", title="player"),
    st.Page("pg/team.py", title="team"),
]

pg = st.navigation(pg_list)
pg.run()
