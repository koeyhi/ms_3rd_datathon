from datetime import datetime
import numpy as np
import streamlit as st
import pandas as pd
import json
import pytz

st.set_page_config(layout="wide")
DATA_PATH = "streamlit/data/"
ARTIFACTS_PATH = "streamlit/artifacts/"
# DATA_PATH = "data/"
# ARTIFACTS_PATH = "artifacts/"

teams_train = pd.read_csv(f"{DATA_PATH}teams_train.csv")
teams_test = pd.read_csv(f"{DATA_PATH}teams_test.csv")
train_data = pd.concat([teams_train, teams_test], ignore_index=True)

jh_featured_data = pd.read_csv(f"{DATA_PATH}featured_data.csv")
jh_featured_data.drop("gameid", axis=1, inplace=True)

hj_featured_train = pd.read_csv(f"{DATA_PATH}TEST88_train.csv")
hj_featured_test = pd.read_csv(f"{DATA_PATH}TEST88_test.csv")
hj_featured_data = pd.concat([hj_featured_train, hj_featured_test], ignore_index=True)
hj_featured_data.drop("gameid", axis=1, inplace=True)
hj_featured_data["side"] = hj_featured_data["side"].map({"Blue": 0, "Red": 1})

with open(f"{DATA_PATH}teams.json", "r") as f:
    teams = json.load(f)

with open(f"{DATA_PATH}champions.json", "r") as f:
    champions = json.load(f)

with open(f"{DATA_PATH}leagues.json", "r") as f:
    leagues = json.load(f)

temp_opp_teams = (
    train_data.groupby("gameid")["teamname"]
    .transform(lambda x: x.iloc[::-1].values)
    .to_frame("opp_teamname")
)
train_data = pd.concat([train_data, temp_opp_teams], axis=1)
train_data.drop("gameid", axis=1, inplace=True)

import joblib
from catboost import CatBoostClassifier, Pool

jh_stacking = joblib.load(f"{ARTIFACTS_PATH}stacking_0107.pkl")

with open(f"{DATA_PATH}cat_features.json", "r") as f:
    cat_cols = json.load(f)

jh_cat = CatBoostClassifier()
jh_cat.load_model(f"{ARTIFACTS_PATH}cat_0107.cbm")

hj_stacking = joblib.load(f"{ARTIFACTS_PATH}5_stacking_model_0120.pkl")

train_data["date"] = pd.to_datetime(train_data["date"])
train_data["year"] = train_data["date"].dt.year
train_data["month"] = train_data["date"].dt.month
train_data["day"] = train_data["date"].dt.day
train_data["hour"] = train_data["date"].dt.hour
train_data["minute"] = train_data["date"].dt.minute
train_data.drop("date", axis=1, inplace=True)


def update_time(input_data):
    league_locations = {
        "LCK": "Asia/Seoul",
        "LEC": "Europe/Berlin",
        "LCS": "America/Los_Angeles",
        "CBLOL": "America/Sao_Paulo",
        "PCS": "Asia/Taipei",
        "VCS": "Asia/Ho_Chi_Minh",
        "MSI": {2022: "Asia/Seoul", 2023: "Europe/London", 2024: "Asia/Shanghai"},
        "WLDs": {
            2022: "America/Los_Angeles",
            2023: "Asia/Seoul",
            2024: "Europe/Berlin",
        },
    }

    date_str = f"{input_data['year']}-{input_data['month']}-{input_data['day']} {input_data['hour']}:{input_data['minute']}"
    if len(date_str.split(" ")[1].split(":")[0]) == 1:
        date_str = date_str.replace(" ", " 0", 1)

    if len(date_str.split(":")) == 2:
        date_str += ":00"

    input_data["date"] = date_str

    utc = pytz.timezone("UTC")

    local_times = []

    utc_time_str = input_data["date"]
    league = input_data["league"]

    utc_time = datetime.strptime(utc_time_str, "%Y-%m-%d %H:%M:%S")
    utc_time = utc.localize(utc_time)

    year = utc_time.year

    if league in league_locations:
        if isinstance(league_locations[league], dict):
            local_tz = pytz.timezone(league_locations[league].get(year, "UTC"))
        else:
            local_tz = pytz.timezone(league_locations[league])
        local_time = utc_time.astimezone(local_tz)
        local_time_str = local_time.strftime("%Y-%m-%d %H:%M:%S")
    else:
        local_time_str = utc_time_str

    local_times.append(local_time_str)

    input_data["local_time"] = local_times
    input_data["local_time"] = pd.to_datetime(input_data["local_time"])

    input_data["year"] = int(input_data["local_time"].year[0])
    input_data["month"] = int(input_data["local_time"].month[0])
    input_data["day"] = int(input_data["local_time"].day[0])
    input_data["hour"] = int(input_data["local_time"].hour[0])
    input_data["minute"] = int(input_data["local_time"].minute[0])

    input_data["hour_sin"] = np.sin(2 * np.pi * input_data["hour"] / 24)
    input_data["hour_cos"] = np.cos(2 * np.pi * input_data["hour"] / 24)

    del input_data["local_time"]
    del input_data["date"]

    if input_data["hour"] < 6:
        input_data["hour"] += 24
    input_data["time_period"] = int(
        pd.cut(
            [input_data["hour"]],
            bins=[-1, 12, 18, 30],
            labels=["0", "1", "2"],
        )[0]
    )

    return input_data


def add_recent10_stats(input_data, train_data):
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

    return input_data


def add_h2h_winrate(input_data, train_data):
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

    return input_data


def add_league_winrate(input_data, train_data):
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
        "league",
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
    teamname = st.selectbox("팀", teams)
    opp_teamname = st.selectbox("상대 팀", teams)
    patch = st.number_input("패치 버전", value=14.23)
    league = st.selectbox("리그", leagues)
    side = st.selectbox("진영", ["Blue", "Red"])
    date = st.date_input("날짜", value=datetime.today())
    time = st.time_input("시간", value=datetime.now().time())
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
        input_data_for_jh_model = add_recent10_stats(input_data, train_data)
        input_data_for_jh_model = add_h2h_winrate(input_data_for_jh_model, train_data)
        input_data_for_jh_model = add_league_winrate(
            input_data_for_jh_model, train_data
        )
        (
            input_data_for_jh_model,
            cat_input_data_for_jh_model,
            cat_featured_data_for_jh_model,
        ) = split_data(input_data_for_jh_model, jh_featured_data)

        input_data_for_hj_model = update_time(input_data)
        input_data_for_hj_model = add_recent10_stats(
            input_data_for_hj_model, train_data
        )
        input_data_for_hj_model = add_h2h_winrate(input_data_for_hj_model, train_data)
        input_data_for_hj_model = add_league_winrate(
            input_data_for_hj_model, train_data
        )
        (
            input_data_for_hj_model,
            cat_input_data_for_hj_model,
            cat_featured_data_for_hj_model,
        ) = split_data(input_data_for_hj_model, hj_featured_data)

        input_data_for_jh_model = preprocess(
            input_data_for_jh_model, train_data, champions, teams
        )
        cat_featured_data_for_jh_model = preprocess(
            cat_featured_data_for_jh_model, train_data, champions, teams
        )
        jh_featured_data = preprocess(jh_featured_data, train_data, champions, teams)

        input_data_for_hj_model = preprocess(
            input_data_for_hj_model, train_data, champions, teams
        )
        hj_featured_data = preprocess(hj_featured_data, train_data, champions, teams)

        input_data_for_jh_model = scale(input_data_for_jh_model, jh_featured_data)
        cat_input_data_for_jh_model = scale(
            cat_input_data_for_jh_model, jh_featured_data
        )
        cat_featured_data_for_jh_model = scale(
            cat_featured_data_for_jh_model, jh_featured_data
        )
        cat_input_data_for_jh_model = Pool(
            cat_input_data_for_jh_model, cat_features=cat_cols
        )

        input_data_for_hj_model = scale(input_data_for_hj_model, hj_featured_data)

        pred_jh_stacking = jh_stacking.predict_proba(input_data_for_jh_model)
        pred_jh_cat = jh_cat.predict_proba(cat_input_data_for_jh_model)

        input_data_for_hj_model.columns = hj_featured_data.columns
        pred_hj_stacking = hj_stacking.predict_proba(input_data_for_hj_model)

        pred = np.mean([pred_jh_stacking, pred_jh_cat, pred_hj_stacking], axis=0)
        st.write(f"{teamname} 승리 확률: {pred[0][1] * 100:.1f}%")
        st.write(f"{opp_teamname} 승리 확률: {pred[0][0] * 100:.1f}%")
