from datetime import datetime
import streamlit as st
import pandas as pd
import json

st.set_page_config(layout="wide")
DATA_PATH = "streamlit/data/"

teams_train = pd.read_csv(f"{DATA_PATH}teams_train.csv")
teams_test = pd.read_csv(f"{DATA_PATH}teams_test.csv")
train_data = pd.concat([teams_train, teams_test], ignore_index=True)

featured_data = pd.read_csv(f"{DATA_PATH}featured_data.csv")
featured_data.drop("gameid", axis=1, inplace=True)

with open("data/teams.json", "r") as f:
    teams = json.load(f)

with open("data/champions.json", "r") as f:
    champions = json.load(f)

with open("data/leagues.json", "r") as f:
    leagues = json.load(f)

temp_opp_teams = (
    train_data.groupby("gameid")["teamname"]
    .transform(lambda x: x.iloc[::-1].values)
    .to_frame("opp_teamname")
)
train_data = pd.concat([train_data, temp_opp_teams], axis=1)
train_data.drop("gameid", axis=1, inplace=True)

train_data["date"] = pd.to_datetime(train_data["date"])
train_data["year"] = train_data["date"].dt.year
train_data["month"] = train_data["date"].dt.month
train_data["day"] = train_data["date"].dt.day
train_data["hour"] = train_data["date"].dt.hour
train_data["minute"] = train_data["date"].dt.minute

import joblib
from catboost import CatBoostClassifier, Pool

stacking_model = joblib.load("artifacts/stacking_0107.pkl")

with open("data/cat_features.json", "r") as f:
    cat_cols = json.load(f)

cat_model = CatBoostClassifier()
cat_model.load_model("artifacts/cat_0107.cbm")


def add_features(input_data, train_data):
    stats_columns = [
        "result",
        "gamelength",
        "kills",
        "deaths",
        "assists",
        "firstblood",
        "team kpm",
        "ckpm",
        "firstdragon",
        "firstherald",
        "void_grubs",
        "firstbaron",
        "firsttower",
        "towers",
        "firstmidtower",
        "firsttothreetowers",
        "turretplates",
        "inhibitors",
        "damagetochampions",
        "dpm",
        "damagetakenperminute",
        "damagemitigatedperminute",
        "wardsplaced",
        "wpm",
        "wardskilled",
        "wcpm",
        "controlwardsbought",
        "visionscore",
        "vspm",
    ]

    # 최근 10경기 데이터 추가
    input_team_data = (
        train_data[train_data["teamname"] == input_data["teamname"]]
        .sort_values(["year", "month", "day", "hour", "minute"])
        .reset_index(drop=True)
    )

    input_opp_data = (
        train_data[train_data["teamname"] == input_data["opp_teamname"]]
        .sort_values(["year", "month", "day", "hour", "minute"])
        .reset_index(drop=True)
    )

    recent10_stats = {}
    for col in stats_columns:
        if len(input_team_data) > 0:
            team_recent10 = (
                input_team_data[col].rolling(window=10, min_periods=1).mean().iloc[-1]
            )
        else:
            team_recent10 = 0.5 if col == "result" else 0
        recent10_stats[f"recent10_{col}"] = team_recent10

        if len(input_opp_data) > 0:
            opp_recent10 = (
                input_opp_data[col].rolling(window=10, min_periods=1).mean().iloc[-1]
            )
        else:
            opp_recent10 = 0.5 if col == "result" else 0
        recent10_stats[f"opp_recent10_{col}"] = opp_recent10

    for feature in recent10_stats:
        input_data[feature] = recent10_stats[feature]

    # 상대전적 추가
    head_to_head = train_data[
        (
            (train_data["teamname"] == input_data["teamname"])
            & (train_data["opp_teamname"] == input_data["opp_teamname"])
        )
        | (
            (train_data["teamname"] == input_data["opp_teamname"])
            & (train_data["opp_teamname"] == input_data["teamname"])
        )
    ].sort_values(["year", "month", "day", "hour", "minute"])

    if len(head_to_head) > 0:
        team_wins = head_to_head[
            (
                (head_to_head["teamname"] == input_data["teamname"])
                & (head_to_head["result"] == 1)
                | (head_to_head["teamname"] == input_data["opp_teamname"])
                & (head_to_head["result"] == 0)
            )
        ].shape[0]
        h2h_winrate = team_wins / len(head_to_head)
    else:
        h2h_winrate = 0.5

    input_data["h2h_winrate"] = h2h_winrate

    # 리그 승률
    team_league_games = (
        train_data[
            (train_data["teamname"] == input_data["teamname"])
            & (train_data["league"] == input_data["league"])
        ]
        .sort_values(["year", "month", "day", "hour", "minute"])
        .reset_index(drop=True)
    )
    if len(team_league_games) > 0:
        team_league_winrate = team_league_games["result"].mean()
    else:
        team_league_winrate = 0.5

    input_data["league_winrate"] = team_league_winrate

    input_data = pd.DataFrame([input_data.values()], columns=input_data.keys())

    return input_data


def split_data(input_data, featured_data):
    input_data["side"] = input_data["side"].map({"Blue": 0, "Red": 1})
    cat_input_data = input_data.copy()
    cat_featured_data = featured_data.copy()

    cat_cols = [
        "teamname",
        "opp_teamname",
        "ban1",
        "ban2",
        "ban3",
        "ban4",
        "ban5",
        "pick1",
        "pick2",
        "pick3",
        "pick4",
        "pick5",
    ]
    cat_input_data[cat_cols] = cat_input_data[cat_cols].astype("category")

    return input_data, cat_input_data, cat_featured_data


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


def preprocess(input_data, train_data, champions, teams):
    champions_df = pd.DataFrame({"champion": champions})
    champions_df = champions_df.dropna().reset_index(drop=True)

    le = LabelEncoder()
    le.fit(champions_df["champion"])

    for col in [
        "ban1",
        "ban2",
        "ban3",
        "ban4",
        "ban5",
        "pick1",
        "pick2",
        "pick3",
        "pick4",
        "pick5",
    ]:
        input_data[col] = le.transform(input_data[col])

    encoder = OneHotEncoder()
    encoder.fit(train_data[["league"]])
    league_encoded = encoder.transform(input_data[["league"]]).toarray()
    league_cols = [f"league_{col}" for col in encoder.categories_[0]]
    input_data = pd.concat(
        [input_data, pd.DataFrame(league_encoded, columns=league_cols)], axis=1
    )
    input_data.drop("league", axis=1, inplace=True)

    le_team = LabelEncoder()
    teams_df = pd.DataFrame(teams)
    le_team.fit(teams_df)

    input_data["teamname"] = le_team.transform(input_data["teamname"])
    input_data["opp_teamname"] = le_team.transform(input_data["opp_teamname"])

    return input_data


from sklearn.preprocessing import MinMaxScaler


def scale(input_data, featured_data):
    scaler = MinMaxScaler()
    numeric_cols = input_data.select_dtypes("number").columns

    scaler.fit(featured_data[numeric_cols])
    input_data[numeric_cols] = scaler.transform(input_data[numeric_cols])

    return input_data


with st.form("예측 폼", border=True):
    teamname = st.selectbox("팀 선택", teams)
    opp_teamname = st.selectbox("상대 팀 선택", teams)
    patch = st.number_input("패치 버전", value=14.23)
    league = st.selectbox("리그 선택", leagues)
    side = st.selectbox("진영 선택", ["Blue", "Red"])
    date = st.date_input("날짜 선택", value=datetime.today())
    time = st.time_input("시간 선택", value=datetime.now().time())
    ban1 = st.selectbox("밴 1", champions)
    ban2 = st.selectbox("밴 2", champions)
    ban3 = st.selectbox("밴 3", champions)
    ban4 = st.selectbox("밴 4", champions)
    ban5 = st.selectbox("밴 5", champions)
    pick1 = st.selectbox("픽 1", champions)
    pick2 = st.selectbox("픽 2", champions)
    pick3 = st.selectbox("픽 3", champions)
    pick4 = st.selectbox("픽 4", champions)
    pick5 = st.selectbox("픽 5", champions)
    submit_button = st.form_submit_button("예측")

    input_data = {
        "patch": patch,
        "side": side,
        "league": league,
        "teamname": teamname,
        "opp_teamname": opp_teamname,
        "ban1": ban1,
        "ban2": ban2,
        "ban3": ban3,
        "ban4": ban4,
        "ban5": ban5,
        "pick1": pick1,
        "pick2": pick2,
        "pick3": pick3,
        "pick4": pick4,
        "pick5": pick5,
        "year": date.year,
        "month": date.month,
        "day": date.day,
        "hour": time.hour,
        "minute": time.minute,
    }

    if submit_button:
        input_data = add_features(input_data, train_data)
        input_data, cat_input_data, cat_featured_data = split_data(
            input_data, featured_data
        )
        input_data = preprocess(input_data, train_data, champions, teams)
        featured_data = preprocess(featured_data, train_data, champions, teams)

        input_data = scale(input_data, featured_data)
        cat_input_data = scale(cat_input_data, featured_data)
        cat_featured_data = scale(cat_featured_data, featured_data)

        cat_input_data = Pool(cat_input_data, cat_features=cat_cols)

        pred_stacking = stacking_model.predict_proba(input_data)
        pred_cat = cat_model.predict_proba(cat_input_data)

        pred = (pred_stacking + pred_cat) / 2
        st.write(f"{teamname} 승리 확률: {pred[0][1] * 100:.1f}%")
        st.write(f"{opp_teamname} 승리 확률: {pred[0][0] * 100:.1f}%")
