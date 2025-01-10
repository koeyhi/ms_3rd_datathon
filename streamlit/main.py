import streamlit as st
import subprocess
import sys

print(sys.version)

try:
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    subprocess.run([sys.executable, "-m", "pip", "install", "joblib"], check=True)
except subprocess.CalledProcessError as e:
    print(f"패키지 설치 실패: {e}")

pg_list = [
    st.Page("pg/home.py", title="홈"),
    st.Page("pg/dashboard.py", title="대시보드"),
    st.Page("pg/inference.py", title="예측"),
]

pg = st.navigation(pg_list)
pg.run()
