import glob
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st


logger = logging.getLogger("ipl_score_predictor")

TEAM_MAP = {
    "CSK": "Chennai Super Kings",
    "DC": "Delhi Capitals",
    "PBKS": "Punjab Kings",
    "KKR": "Kolkata Knight Riders",
    "MI": "Mumbai Indians",
    "RR": "Rajasthan Royals",
    "RCB": "Royal Challengers Bangalore",
    "SRH": "Sunrisers Hyderabad",
    "GT": "Gujarat Titans",
    "LSG": "Lucknow Super Giants",
    "GL": "Gujarat Lions",
    "RPS": "Rising Pune Supergiants",
    "DEC": "Deccan Chargers",
}

TEAM_ALIASES = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Punjab Kings": "Punjab Kings",
    "Royal Challengers Bengaluru": "Royal Challengers Bangalore",
    "Rising Pune Supergiant": "Rising Pune Supergiants",
}

TEAM_VALUES = sorted(set(TEAM_MAP.values()) | set(TEAM_ALIASES.values()))


def _normalize_team_name(name):
    if pd.isna(name):
        return name
    value = str(name).strip()
    return TEAM_ALIASES.get(value, value)


def _standardize_legacy_frame(raw_df):
    df = raw_df.copy()

    rename_map = {
        "batting_team": "bat_team",
        "bowling_team": "bowl_team",
        "over": "overs",
    }
    for old_col, new_col in rename_map.items():
        if old_col in df.columns and new_col not in df.columns:
            df.rename(columns={old_col: new_col}, inplace=True)

    required = ["bat_team", "bowl_team", "runs", "wickets", "overs", "runs_last_5", "wickets_last_5", "total"]
    for col in required:
        if col not in df.columns:
            df[col] = 0

    if "mid" not in df.columns:
        df["mid"] = np.arange(1, len(df) + 1)

    if "date" not in df.columns:
        df["date"] = datetime(2019, 1, 1)
    df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)

    if "venue" not in df.columns:
        df["venue"] = "Unknown Venue"

    if "toss_winner" not in df.columns:
        df["toss_winner"] = df["bat_team"]
    if "toss_decision" not in df.columns:
        df["toss_decision"] = "bat"

    df["bat_team"] = df["bat_team"].map(_normalize_team_name)
    df["bowl_team"] = df["bowl_team"].map(_normalize_team_name)
    df["toss_winner"] = df["toss_winner"].map(_normalize_team_name)

    core_cols = [
        "mid",
        "date",
        "venue",
        "bat_team",
        "bowl_team",
        "runs",
        "wickets",
        "overs",
        "runs_last_5",
        "wickets_last_5",
        "total",
        "toss_winner",
        "toss_decision",
    ]
    return df[core_cols].copy()


def _convert_ball_by_ball_frame(raw_df):
    df = raw_df.copy()

    if not {"match_id", "batting_team", "runs_batter", "runs_extras", "wicket_taken", "over"}.issubset(df.columns):
        return pd.DataFrame()

    df.rename(columns={"batting_team": "bat_team", "match_id": "mid", "over": "over_int"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)

    df["bat_team"] = df["bat_team"].map(_normalize_team_name)

    teams_per_match = df.groupby("mid")["bat_team"].agg(lambda s: list(pd.unique(s.dropna()))).to_dict()

    def _other_team(row):
        teams = teams_per_match.get(row["mid"], [])
        for team in teams:
            if team != row["bat_team"]:
                return team
        return "Unknown Opponent"

    df["bowl_team"] = df.apply(_other_team, axis=1)
    df["bowl_team"] = df["bowl_team"].map(_normalize_team_name)

    df["runs_event"] = pd.to_numeric(df["runs_batter"], errors="coerce").fillna(0) + pd.to_numeric(
        df["runs_extras"], errors="coerce"
    ).fillna(0)
    df["wicket_event"] = pd.to_numeric(df["wicket_taken"], errors="coerce").fillna(0)

    innings_key = ["mid", "bat_team"]
    df["ball_index"] = df.groupby(innings_key).cumcount() + 1
    df["runs"] = df.groupby(innings_key)["runs_event"].cumsum()
    df["wickets"] = df.groupby(innings_key)["wicket_event"].cumsum()

    # Uses 6-ball grouping to align with classic IPL over format used by existing app.
    df["overs"] = ((df["ball_index"] - 1) // 6) + (((df["ball_index"] - 1) % 6) + 1) / 10.0

    rolling_runs = (
        df.groupby(innings_key)["runs_event"]
        .rolling(window=30, min_periods=1)
        .sum()
        .reset_index(level=innings_key, drop=True)
    )
    rolling_wkts = (
        df.groupby(innings_key)["wicket_event"]
        .rolling(window=30, min_periods=1)
        .sum()
        .reset_index(level=innings_key, drop=True)
    )
    df["runs_last_5"] = rolling_runs
    df["wickets_last_5"] = rolling_wkts

    df["total"] = df.groupby(innings_key)["runs"].transform("max")
    if "venue" not in df.columns:
        df["venue"] = "Unknown Venue"
    else:
        df["venue"] = df["venue"].fillna("Unknown Venue")
    df["toss_winner"] = df["bat_team"]
    df["toss_decision"] = "bat"

    core_cols = [
        "mid",
        "date",
        "venue",
        "bat_team",
        "bowl_team",
        "runs",
        "wickets",
        "overs",
        "runs_last_5",
        "wickets_last_5",
        "total",
        "toss_winner",
        "toss_decision",
    ]
    return df[core_cols].copy()


@st.cache_data(show_spinner=False)
def load_data():
    csv_files = sorted(set(glob.glob("ipl_data*.csv") + glob.glob("all_matches_ball_by_ball*.csv")))
    if not csv_files:
        raise FileNotFoundError("No IPL dataset found. Expected at least ipl_data.csv")

    frames = []
    for path in csv_files:
        raw = pd.read_csv(path)
        if {"match_id", "batting_team", "runs_batter", "runs_extras", "over"}.issubset(raw.columns):
            converted = _convert_ball_by_ball_frame(raw)
            if not converted.empty:
                frames.append(converted)
        else:
            frames.append(_standardize_legacy_frame(raw))

    df = pd.concat(frames, ignore_index=True)

    df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)

    for col in ["bat_team", "bowl_team"]:
        if col not in df.columns:
            raise ValueError(f"Required column missing: {col}")

    df = df[(df["bat_team"].isin(TEAM_VALUES)) & (df["bowl_team"].isin(TEAM_VALUES))]
    df = df[df["overs"] >= 0.1].copy()

    if "venue" not in df.columns:
        df["venue"] = "Unknown Venue"
    df["venue"] = df["venue"].fillna("Unknown Venue")

    if "toss_winner" not in df.columns:
        df["toss_winner"] = df["bat_team"]
    df["toss_winner"] = df["toss_winner"].fillna(df["bat_team"])

    if "toss_decision" not in df.columns:
        df["toss_decision"] = "bat"
    df["toss_decision"] = df["toss_decision"].fillna("bat")

    # Keep useful core numeric columns available.
    for col in ["runs", "wickets", "runs_last_5", "wickets_last_5", "total"]:
        if col not in df.columns:
            df[col] = 0

    logger.info("Loaded %s rows from %s dataset files.", len(df), len(csv_files))
    return df


def _compute_balls(overs_value):
    over_part = int(np.floor(overs_value))
    ball_part = int(round((overs_value - over_part) * 10))
    ball_part = min(max(ball_part, 0), 5)
    return over_part * 6 + ball_part


@st.cache_data(show_spinner=False)
def build_enhanced_dataset(df):
    data = df.copy()

    data["balls_bowled"] = data["overs"].apply(_compute_balls)
    data["balls_remaining"] = (120 - data["balls_bowled"]).clip(lower=0)
    data["wickets_remaining"] = (10 - data["wickets"]).clip(lower=0)
    data["current_run_rate"] = data["runs"] / data["overs"].clip(lower=0.1)
    data["momentum_score"] = data["runs_last_5"] / 5.0
    data["pressure_index"] = data["wickets"] * (20.0 - data["overs"]).clip(lower=0)

    data["is_chasing"] = np.where(data["toss_decision"].str.lower().eq("field"), 1, 0)
    data["required_run_rate"] = np.where(
        data["is_chasing"] == 1,
        (data["total"] - data["runs"]).clip(lower=0) / (data["balls_remaining"].clip(lower=1) / 6.0),
        0.0,
    )
    data["batting_first_flag"] = np.where(data["is_chasing"] == 1, 0, 1)

    innings_cols = ["mid", "bat_team", "bowl_team", "date", "total", "venue"]
    available_innings_cols = [col for col in innings_cols if col in data.columns]
    innings_df = data[available_innings_cols].drop_duplicates()

    if "date" in innings_df.columns:
        innings_df = innings_df.sort_values("date")

    if {"bat_team", "total"}.issubset(innings_df.columns):
        innings_df["team_recent_form"] = (
            innings_df.groupby("bat_team")["total"]
            .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
            .fillna(innings_df["total"].mean())
        )
    else:
        innings_df["team_recent_form"] = data["total"].mean()

    if {"bowl_team", "total"}.issubset(innings_df.columns):
        bowl_strength = innings_df.groupby("bowl_team")["total"].mean()
        innings_df["opponent_strength"] = innings_df["bowl_team"].map(bowl_strength).fillna(bowl_strength.mean())
    else:
        innings_df["opponent_strength"] = data["total"].mean()

    if "venue" in innings_df.columns:
        venue_avg = innings_df.groupby("venue")["total"].mean()
        innings_df["venue_avg_score"] = innings_df["venue"].map(venue_avg).fillna(venue_avg.mean())
    else:
        innings_df["venue_avg_score"] = data["total"].mean()

    team_strength = innings_df.groupby("bat_team")["total"].mean()
    innings_df["team_strength_score"] = innings_df["bat_team"].map(team_strength).fillna(team_strength.mean())

    global_avg = max(float(innings_df["total"].mean()), 1.0)
    innings_df["venue_factor"] = innings_df["venue_avg_score"] / global_avg

    merge_keys = [c for c in ["mid", "bat_team", "bowl_team", "date", "venue"] if c in data.columns]
    if merge_keys:
        data = data.merge(
            innings_df[merge_keys + [
                "team_recent_form",
                "opponent_strength",
                "venue_avg_score",
                "team_strength_score",
                "venue_factor",
            ]],
            on=merge_keys,
            how="left",
        )

    data["phase"] = pd.cut(
        data["overs"],
        bins=[0, 6, 15, 20],
        labels=["powerplay", "middle", "death"],
        include_lowest=True,
    )

    phase_stats = (
        data.groupby("phase")["current_run_rate"].mean().to_dict() if "phase" in data.columns else {}
    )
    data["powerplay_score_rate"] = phase_stats.get("powerplay", data["current_run_rate"].mean())
    data["middle_score_rate"] = phase_stats.get("middle", data["current_run_rate"].mean())
    data["death_score_rate"] = phase_stats.get("death", data["current_run_rate"].mean())

    data["death_over_boost"] = np.where(data["overs"] >= 15, data["momentum_score"] * 1.2, data["momentum_score"] * 0.8)

    # Fill any leftovers from joins.
    for col in [
        "team_recent_form",
        "opponent_strength",
        "venue_avg_score",
        "team_strength_score",
        "venue_factor",
    ]:
        data[col] = data[col].fillna(data[col].median())

    return data
