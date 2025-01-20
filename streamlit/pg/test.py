import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

# 데이터 생성 (샘플 데이터)
DATA_PATH = "LoLesports_data/"
team_df = pd.read_csv(f"{DATA_PATH}team.csv")
player_df = pd.read_csv(f"{DATA_PATH}player.csv")

st.title("팀/선수 주요 지표 방사형 그래프로 확인하기")

# 두 개의 열 생성
col1, col2 = st.columns(2)

# 첫 번째 그래프
with col1:
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
    selected_metrics = st.multiselect(
        "그래프에 표시할 컬럼 선택", metrics, default=metrics
    )

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
            normalized_values = [
                v / max_val for v, max_val in zip(avg_values, max_values)
            ]
            radar_fig = go.Figure(
                go.Scatterpolar(
                    r=normalized_values
                    + [normalized_values[0]],  # 순환하여 폐곡선 생성
                    theta=selected_metrics + [selected_metrics[0]],
                    fill="toself",
                    name=f"{selected_team} (스플릿 전체 평균)",
                    hovertemplate="<b>%{theta}</b><br>Value: %{customdata}<br>Normalized: %{r:.2f}<extra></extra>",
                    customdata=avg_values + [avg_values[0]],
                )
            )
            radar_fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True, range=[0, 1]  # 비율로 그리므로 0~1 사이
                    )
                ),
                showlegend=True,
                title="방사형 그래프",
            )
            st.plotly_chart(radar_fig)
    else:
        st.warning(
            "선택한 조건에 해당하는 데이터가 없거나, 컬럼이 선택되지 않았습니다."
        )

# 두 번째 그래프
with col2:
    available_years = sorted(player_df["year"].unique())
    selected_year = st.selectbox("연도 선택", available_years, key="player_year")

    available_leagues = player_df[player_df["year"] == selected_year]["league"].unique()
    selected_league = st.selectbox("리그 선택", available_leagues, key="player_league")

    available_splits = player_df[
        (player_df["year"] == selected_year) & (player_df["league"] == selected_league)
    ]["split"].unique()
    selected_split = st.selectbox("스플릿 선택", available_splits, key="player_split")

    available_players = player_df[
        (player_df["year"] == selected_year)
        & (player_df["league"] == selected_league)
        & (player_df["split"] == selected_split)
    ]["playername"].unique()
    selected_player = st.selectbox("선수 선택", available_players, key="player_name")

    # 사용자가 선택할 컬럼 선택
    metrics = ["kda", "dpm", "total cs", "totalgold", "visionscore"]
    selected_metrics = st.multiselect(
        "그래프에 표시할 컬럼 선택", metrics, default=metrics, key="player_metrics"
    )

    # 선택된 선수의 스플릿별 전체 평균 데이터 필터링
    filtered_data = player_df[
        (player_df["year"] == selected_year)
        & (player_df["league"] == selected_league)
        & (player_df["split"] == selected_split)
        & (player_df["playername"] == selected_player)
    ]

    # 데이터 시각화
    if selected_metrics and not filtered_data.empty:
        avg_values = filtered_data[selected_metrics].mean().tolist()

        if len(selected_metrics) == 1 or len(selected_metrics) == 2:
            # 1, 2개 선택 시: 가로형 바 차트
            st.write(f"{selected_player}의 {selected_split} 스플릿 평균 (바 차트)")
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
            st.write(f"{selected_player}의 {selected_split} 스플릿 평균 (라인 그래프)")
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
            max_values = [10, 800, 500, 15000, 100]
            normalized_values = [
                v / max_val for v, max_val in zip(avg_values, max_values)
            ]
            radar_fig = go.Figure(
                go.Scatterpolar(
                    r=normalized_values
                    + [normalized_values[0]],  # 순환하여 폐곡선 생성
                    theta=selected_metrics + [selected_metrics[0]],
                    fill="toself",
                    name=f"{selected_player} (스플릿 전체 평균)",
                    hovertemplate="<b>%{theta}</b><br>Value: %{customdata}<br>Normalized: %{r:.2f}<extra></extra>",
                    customdata=avg_values + [avg_values[0]],
                )
            )
            radar_fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True, range=[0, 1]  # 비율로 그리므로 0~1 사이
                    )
                ),
                showlegend=True,
                title="방사형 그래프",
            )
            st.plotly_chart(radar_fig)
    else:
        st.warning(
            "선택한 조건에 해당하는 데이터가 없거나, 컬럼이 선택되지 않았습니다."
        )
