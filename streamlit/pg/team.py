import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# 데이터 경로와 데이터 로드
DATA_PATH = "streamlit/data/"
# DATA_PATH = "data/"
team_df = pd.read_csv(f"{DATA_PATH}team.csv")

# Streamlit 애플리케이션
# st.title("팀별 성과 방사형 그래프")
# st.sidebar.header("필터 설정")

# 사용자가 선택할 옵션 정의
available_years = sorted(team_df["year"].unique())
selected_year = st.selectbox("연도 선택", available_years)

available_leagues = team_df[team_df["year"] == selected_year]["league"].unique()
selected_league = st.selectbox("리그 선택", available_leagues)

available_splits = team_df[
    (team_df["year"] == selected_year) & (team_df["league"] == selected_league)
]["split"].unique()
selected_split = st.selectbox("스플릿 선택", available_splits)

available_teams = team_df[
    (team_df["year"] == selected_year)
    & (team_df["league"] == selected_league)
    & (team_df["split"] == selected_split)
]["teamname"].unique()
selected_team = st.selectbox("팀 선택", available_teams)

# 사용자가 선택할 컬럼 선택
metrics = ["firstblood", "firsttower", "void_grubs", "dragons", "barons"]
selected_metrics = st.multiselect("그래프에 표시할 컬럼 선택", metrics, default=metrics)

# 선택된 선수의 스플릿별 전체 평균 데이터 필터링
filtered_data = team_df[
    (team_df["year"] == selected_year)
    & (team_df["league"] == selected_league)
    & (team_df["split"] == selected_split)
    & (team_df["teamname"] == selected_team)
]

# 데이터 시각화
if selected_metrics and not filtered_data.empty:
    avg_values = filtered_data[selected_metrics].mean().tolist()

    if len(selected_metrics) == 1 or len(selected_metrics) == 2:
        # 1, 2개 선택 시: 가로형 바 차트
        st.write(f"{selected_team}의 {selected_split} 스플릿 평균 (바 차트)")
        bar_fig = go.Figure(
            go.Bar(
                x=avg_values,
                y=selected_metrics,
                orientation="h",
                text=avg_values,
                textposition="auto",
            )
        )
        bar_fig.update_layout(
            title="가로형 바 차트", xaxis_title="평균 값", yaxis_title="지표"
        )
        st.plotly_chart(bar_fig)

    elif len(selected_metrics) in [3, 4]:
        # 3, 4개 선택 시: 라인 그래프
        st.write(f"{selected_team}의 {selected_split} 스플릿 평균 (라인 그래프)")
        line_fig = go.Figure(
            go.Scatter(
                x=selected_metrics,
                y=avg_values,
                mode="lines+markers",
                line=dict(shape="spline"),
                marker=dict(size=10),
                text=avg_values,
                hovertemplate="<b>%{x}</b>: %{y:.2f}<extra></extra>",
            )
        )
        line_fig.update_layout(
            title="라인 그래프", xaxis_title="지표", yaxis_title="평균 값"
        )
        st.plotly_chart(line_fig)

    elif len(selected_metrics) == 5:
        # 5개 선택 시: 방사형 그래프
        max_values = [1, 1, 5, 5, 2]
        normalized_values = [v / max_val for v, max_val in zip(avg_values, max_values)]
        radar_fig = go.Figure(
            go.Scatterpolar(
                r=normalized_values + [normalized_values[0]],  # 순환하여 폐곡선 생성
                theta=selected_metrics + [selected_metrics[0]],
                fill="toself",
                name=f"{selected_team} (스플릿 전체 평균)",
                hovertemplate="<b>%{theta}</b><br>Value: %{customdata}<br>Normalized: %{r:.2f}<extra></extra>",
                customdata=avg_values + [avg_values[0]],
            )
        )
        radar_fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1])  # 비율로 그리므로 0~1 사이
            ),
            showlegend=True,
            title="방사형 그래프",
        )
        st.plotly_chart(radar_fig)
else:
    st.warning("선택한 조건에 해당하는 데이터가 없거나, 컬럼이 선택되지 않았습니다.")
