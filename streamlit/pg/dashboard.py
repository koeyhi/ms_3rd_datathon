import pandas as pd
import streamlit as st

st.set_page_config(page_title="Dashboard", layout="wide")
DATA_PATH = "../data/"

df = pd.read_csv(f"{DATA_PATH}featured_data.csv")

st.header("리그 오브 레전드 경기 데이터")
with st.expander("데이터 보기"):
    st.write(df)
