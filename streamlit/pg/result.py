import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 사용자 데이터 경로 지정
DATA_PATH = "streamlit/data/"
# DATA_PATH = "data/"
team_df = pd.read_csv(f"{DATA_PATH}team.csv")

team_df = team_df[team_df["year"] == 2024]  # 2024년 데이터만 선택

st.write(f"### 2024년 팀 데이터 (총 {team_df.shape[0]}개 경기)")
st.dataframe(team_df.head())

# 3. 문자열 타입 컬럼 제거
string_columns = team_df.select_dtypes(include=["object"]).columns
if not string_columns.empty:
    team_df = team_df.drop(columns=string_columns)

# 4. 특정 키워드 포함 컬럼 제거
keywords = ["opp", "10", "15"]
columns_to_drop = [
    col for col in team_df.columns if any(keyword in col for keyword in keywords)
]
if columns_to_drop:
    team_df = team_df.drop(columns=columns_to_drop)

# 5. 상관관계 분석
correlation = team_df.corr()  # 2024년 데이터로 전체 상관행렬 계산
result_correlation = correlation["result"].sort_values(
    ascending=False
)  # result와의 상관관계만 정렬

st.write("#### Result와의 상관계수")
st.dataframe(result_correlation)

# 히트맵 시각화
st.write("### 2024년 팀 승리 히트맵")
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(correlation, annot=False, cmap="coolwarm", cbar=True, ax=ax)
plt.title("2024년 Correlation Matrix")
st.pyplot(fig)

# ------------------------------------------
# 1. 데이터 로드 (경로 지정)
# 사용자 데이터 경로 지정
player_df = pd.read_csv(f"{DATA_PATH}player.csv")

player_df = player_df[player_df["year"] == 2024]  # 2024년 데이터만 선택

st.write(f"### 2024년 선수 데이터 (총 {player_df.shape[0]}개)")
st.dataframe(player_df.head())

# 3. 문자열 타입 컬럼 제거
string_columns = player_df.select_dtypes(include=["object"]).columns
if not string_columns.empty:
    player_df = player_df.drop(columns=string_columns)

# 4. 특정 키워드 포함 컬럼 제거
keywords = ["opp", "10", "15"]
columns_to_drop = [
    col for col in player_df.columns if any(keyword in col for keyword in keywords)
]
if columns_to_drop:
    player_df = player_df.drop(columns=columns_to_drop)

# 5. 상관관계 분석
correlation = player_df.corr()  # 2024년 데이터로 전체 상관행렬 계산
result_correlation = correlation["result"].sort_values(
    ascending=False
)  # result와의 상관관계만 정렬

st.write("#### Result와의 상관계수")
st.dataframe(result_correlation)

# 히트맵 시각화
st.write("### 2024년 선수 승리 히트맵")
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(correlation, annot=False, cmap="coolwarm", cbar=True, ax=ax)
plt.title("2024년 Correlation Matrix")
st.pyplot(fig)
