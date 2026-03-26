import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler


NUMERIC_FEATURES = [
    "overs",
    "runs",
    "wickets",
    "runs_last_5",
    "wickets_last_5",
    "current_run_rate",
    "required_run_rate",
    "wickets_remaining",
    "balls_remaining",
    "pressure_index",
    "momentum_score",
    "team_recent_form",
    "opponent_strength",
    "venue_avg_score",
    "team_strength_score",
    "venue_factor",
    "death_over_boost",
    "powerplay_score_rate",
    "middle_score_rate",
    "death_score_rate",
]

CATEGORICAL_FEATURES = [
    "bat_team",
    "bowl_team",
    "venue",
    "toss_winner",
    "toss_decision",
]


@st.cache_data(show_spinner=False)
def remove_outliers_iqr(df, columns):
    filtered = df.copy()
    for col in columns:
        if col not in filtered.columns:
            continue
        q1 = filtered[col].quantile(0.25)
        q3 = filtered[col].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        low = q1 - 1.5 * iqr
        high = q3 + 1.5 * iqr
        filtered = filtered[(filtered[col] >= low) & (filtered[col] <= high)]
    return filtered


def _target_encode_fit(df, categorical_cols, target_col):
    stats = {}
    global_mean = float(df[target_col].mean())
    for col in categorical_cols:
        if col not in df.columns:
            continue
        mapping = df.groupby(col)[target_col].mean().to_dict()
        stats[col] = mapping
    return stats, global_mean


def _target_encode_transform(df, categorical_cols, stats, global_mean):
    transformed = df.copy()
    for col in categorical_cols:
        enc_col = f"{col}_te"
        if col in transformed.columns:
            transformed[enc_col] = transformed[col].map(stats.get(col, {})).fillna(global_mean)
    return transformed


@st.cache_data(show_spinner=False)
def preprocess_data(df, encode_method="target"):
    data = df.copy()

    for col in NUMERIC_FEATURES + ["total"]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    for col in NUMERIC_FEATURES:
        if col in data.columns:
            data[col] = data[col].fillna(data[col].median())

    for col in CATEGORICAL_FEATURES:
        if col in data.columns:
            data[col] = data[col].fillna("Unknown")

    cleaned = remove_outliers_iqr(data, ["total", "runs", "current_run_rate", "required_run_rate"])

    feature_cols = [c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES if c in cleaned.columns]
    working = cleaned[feature_cols + ["total"]].copy()

    if encode_method == "onehot":
        X = pd.get_dummies(working.drop(columns=["total"]), columns=[c for c in CATEGORICAL_FEATURES if c in working.columns])
        encoding_artifacts = {"method": "onehot", "columns": list(X.columns)}
    else:
        stats, global_mean = _target_encode_fit(working, [c for c in CATEGORICAL_FEATURES if c in working.columns], "total")
        encoded = _target_encode_transform(working, [c for c in CATEGORICAL_FEATURES if c in working.columns], stats, global_mean)
        te_cols = [c for c in encoded.columns if c.endswith("_te")]
        X = encoded[[c for c in NUMERIC_FEATURES if c in encoded.columns] + te_cols].copy()
        encoding_artifacts = {
            "method": "target",
            "stats": stats,
            "global_mean": global_mean,
            "feature_columns": list(X.columns),
            "categorical_cols": [c for c in CATEGORICAL_FEATURES if c in working.columns],
        }

    y = working["total"].values

    scaler = StandardScaler()
    X_scaled = X.copy()
    numeric_in_X = [c for c in NUMERIC_FEATURES if c in X_scaled.columns]
    X_scaled[numeric_in_X] = scaler.fit_transform(X_scaled[numeric_in_X])

    artifacts = {
        "scaler": scaler,
        "feature_columns": list(X_scaled.columns),
        "numeric_columns": numeric_in_X,
        "encoding": encoding_artifacts,
    }
    return X_scaled, y, artifacts


def transform_match_input(match_df, artifacts):
    encoding = artifacts["encoding"]
    transformed = match_df.copy()

    for col in artifacts["numeric_columns"]:
        if col not in transformed.columns:
            transformed[col] = 0.0

    if encoding["method"] == "onehot":
        transformed = pd.get_dummies(transformed)
        for col in artifacts["feature_columns"]:
            if col not in transformed.columns:
                transformed[col] = 0.0
        transformed = transformed[artifacts["feature_columns"]]
    else:
        global_mean = encoding["global_mean"]
        stats = encoding["stats"]
        for col in encoding["categorical_cols"]:
            transformed[f"{col}_te"] = transformed[col].map(stats.get(col, {})).fillna(global_mean)
        for col in artifacts["feature_columns"]:
            if col not in transformed.columns:
                transformed[col] = 0.0
        transformed = transformed[artifacts["feature_columns"]]

    scaled = transformed.copy()
    numeric = artifacts["numeric_columns"]
    scaled[numeric] = artifacts["scaler"].transform(scaled[numeric])
    return scaled
