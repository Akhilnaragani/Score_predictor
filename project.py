import logging
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger("ipl_score_predictor")

MODEL_BUNDLE_PATH = "ipl_score_predictor_model.pkl"
RANDOM_STATE = 42
FAST_MODE_DEFAULT = os.getenv("FAST_TRAINING", "1") == "1"

# Team mapping
team_map = {
    "CSK": "Chennai Super Kings",
    "DD": "Delhi Daredevils",
    "KXIP": "Kings XI Punjab",
    "KKR": "Kolkata Knight Riders",
    "MI": "Mumbai Indians",
    "RR": "Rajasthan Royals",
    "RCB": "Royal Challengers Bangalore",
    "SRH": "Sunrisers Hyderabad",
}
team_values = list(team_map.values())


@st.cache_data
def load_data():
    df = pd.read_csv("ipl_data.csv")
    df.drop(
        labels=["mid", "venue", "batsman", "bowler", "striker", "non-striker"],
        axis=1,
        inplace=True,
    )
    df = df[(df["bat_team"].isin(team_values)) & (df["bowl_team"].isin(team_values))]
    df = df[df["overs"] >= 5.0]
    df["date"] = df["date"].apply(lambda x: datetime.strptime(x, "%Y-%m-%d"))
    return df


@st.cache_data
def preprocess_data(df):
    processed_df = df.copy()

    # Feature engineering for stronger signal extraction.
    processed_df["current_run_rate"] = processed_df["runs"] / processed_df["overs"].clip(lower=0.1)
    processed_df["balls_remaining"] = (20.0 - processed_df["overs"]).clip(lower=0.0) * 6.0
    processed_df["wickets_remaining"] = (10 - processed_df["wickets"]).clip(lower=0)
    processed_df["momentum"] = processed_df["runs_last_5"] / 5.0

    encoded_df = pd.get_dummies(processed_df, columns=["bat_team", "bowl_team"])

    for team in team_values:
        bat_col = f"bat_team_{team}"
        bowl_col = f"bowl_team_{team}"
        if bat_col not in encoded_df.columns:
            encoded_df[bat_col] = 0
        if bowl_col not in encoded_df.columns:
            encoded_df[bowl_col] = 0

    ordered_columns = [
        "date",
        *[f"bat_team_{team}" for team in team_values],
        *[f"bowl_team_{team}" for team in team_values],
        "overs",
        "runs",
        "wickets",
        "runs_last_5",
        "wickets_last_5",
        "current_run_rate",
        "balls_remaining",
        "wickets_remaining",
        "momentum",
        "total",
    ]

    encoded_df = encoded_df[ordered_columns]
    X = encoded_df.drop(columns=["total", "date"])
    y = encoded_df["total"].values

    numeric_cols = [
        "overs",
        "runs",
        "wickets",
        "runs_last_5",
        "wickets_last_5",
        "current_run_rate",
        "balls_remaining",
        "wickets_remaining",
        "momentum",
    ]
    return X, y, numeric_cols


@st.cache_resource
def train_models(X, y, numeric_cols, fast_mode=True):
    logger.info("Starting model training and hyperparameter search. fast_mode=%s", fast_mode)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    n_iter = 4 if fast_mode else 10

    model_configs = {
        "RandomForest": {
            "model": RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1),
            "params": {
                "n_estimators": [120, 200, 300],
                "max_depth": [8, 12, 16, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            },
        },
        "XGBoost": {
            "model": XGBRegressor(
                objective="reg:squarederror",
                random_state=RANDOM_STATE,
                n_jobs=1,
                verbosity=0,
            ),
            "params": {
                "n_estimators": [150, 250, 400],
                "max_depth": [4, 6, 8],
                "learning_rate": [0.03, 0.05, 0.1],
                "subsample": [0.8, 0.9, 1.0],
                "colsample_bytree": [0.7, 0.85, 1.0],
            },
        },
        "LightGBM": {
            "model": LGBMRegressor(random_state=RANDOM_STATE, n_jobs=1, verbose=-1),
            "params": {
                "n_estimators": [150, 250, 400],
                "num_leaves": [31, 63, 127],
                "learning_rate": [0.03, 0.05, 0.1],
                "max_depth": [-1, 8, 12],
                "subsample": [0.8, 0.9, 1.0],
            },
        },
        "CatBoost": {
            "model": CatBoostRegressor(
                loss_function="RMSE",
                random_seed=RANDOM_STATE,
                thread_count=1,
                verbose=0,
            ),
            "params": {
                "iterations": [150, 300, 500],
                "depth": [4, 6, 8],
                "learning_rate": [0.03, 0.05, 0.1],
                "l2_leaf_reg": [3, 5, 7, 9],
            },
        },
    }

    trained_models = {}
    metrics = {}
    best_model_name = None
    best_mae = float("inf")

    for model_name, config in model_configs.items():
        logger.info("Tuning %s", model_name)
        try:
            search = RandomizedSearchCV(
                estimator=config["model"],
                param_distributions=config["params"],
                n_iter=n_iter,
                scoring="neg_mean_absolute_error",
                cv=cv,
                random_state=RANDOM_STATE,
                n_jobs=1,
                verbose=0,
            )
            search.fit(X_train_scaled, y_train)
            best_estimator = search.best_estimator_

            y_pred = best_estimator.predict(X_test_scaled)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            cv_mae = -search.best_score_

            trained_models[model_name] = best_estimator
            metrics[model_name] = {
                "MAE": float(mae),
                "RMSE": float(rmse),
                "CV_MAE": float(cv_mae),
            }

            logger.info(
                "%s completed | MAE: %.3f | RMSE: %.3f | CV_MAE: %.3f",
                model_name,
                mae,
                rmse,
                cv_mae,
            )

            if mae < best_mae:
                best_mae = mae
                best_model_name = model_name
        except Exception as model_exc:
            logger.exception("Model tuning failed for %s: %s", model_name, model_exc)

    if not trained_models:
        logger.warning("All advanced models failed; falling back to baseline RandomForest.")
        fallback_model = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=1)
        fallback_model.fit(X_train_scaled, y_train)
        y_pred = fallback_model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        trained_models["RandomForest"] = fallback_model
        metrics["RandomForest"] = {
            "MAE": float(mae),
            "RMSE": float(rmse),
            "CV_MAE": float(mae),
        }
        best_model_name = "RandomForest"

    best_model = trained_models[best_model_name]

    model_bundle = {
        "best_model_name": best_model_name,
        "best_model": best_model,
        "all_models": trained_models,
        "metrics": metrics,
        "feature_columns": list(X.columns),
        "numeric_cols": numeric_cols,
        "scaler": scaler,
    }

    joblib.dump(
        {
            "best_model_name": best_model_name,
            "best_model": best_model,
            "feature_columns": list(X.columns),
            "numeric_cols": numeric_cols,
            "scaler": scaler,
            "metrics": metrics,
        },
        MODEL_BUNDLE_PATH,
    )
    logger.info("Saved best model bundle: %s", best_model_name)

    return model_bundle


def load_saved_model_bundle():
    loaded_bundle = joblib.load(MODEL_BUNDLE_PATH)
    best_model_name = loaded_bundle["best_model_name"]
    best_model = loaded_bundle["best_model"]
    metrics = loaded_bundle.get("metrics", {})

    all_models = {best_model_name: best_model}
    if metrics:
        for name in metrics.keys():
            if name == best_model_name:
                all_models[name] = best_model

    logger.info("Loaded saved model bundle: %s", best_model_name)
    return {
        "best_model_name": best_model_name,
        "best_model": best_model,
        "all_models": all_models,
        "metrics": metrics,
        "feature_columns": loaded_bundle["feature_columns"],
        "numeric_cols": loaded_bundle["numeric_cols"],
        "scaler": loaded_bundle["scaler"],
    }


def evaluate_models(metrics):
    rows = []
    for model_name, model_metrics in metrics.items():
        rows.append(
            {
                "Model": model_name,
                "MAE": round(model_metrics["MAE"], 3),
                "RMSE": round(model_metrics["RMSE"], 3),
                "CV_MAE": round(model_metrics["CV_MAE"], 3),
            }
        )
    return pd.DataFrame(rows).sort_values(by="MAE")


def _build_feature_row(
    feature_columns,
    overs,
    runs,
    wickets,
    runs_last_5,
    wickets_last_5,
    batting_team,
    bowling_team,
):
    row = {col: 0.0 for col in feature_columns}
    row[f"bat_team_{team_map[batting_team]}"] = 1.0
    row[f"bowl_team_{team_map[bowling_team]}"] = 1.0
    row["overs"] = float(overs)
    row["runs"] = float(runs)
    row["wickets"] = float(wickets)
    row["runs_last_5"] = float(runs_last_5)
    row["wickets_last_5"] = float(wickets_last_5)
    row["current_run_rate"] = float(runs) / max(float(overs), 0.1)
    row["balls_remaining"] = max((20.0 - float(overs)) * 6.0, 0.0)
    row["wickets_remaining"] = max(10.0 - float(wickets), 0.0)
    row["momentum"] = float(runs_last_5) / 5.0
    return pd.DataFrame([row], columns=feature_columns)


@st.cache_data(show_spinner=False)
def predict_score(
    batting_team="CSK",
    bowling_team="MI",
    overs=5.1,
    runs=50,
    wickets=0,
    runs_last_5=50,
    wickets_last_5=0,
    selected_model_name="Best (Auto)",
):
    feature_columns = model_artifacts["feature_columns"]
    numeric_cols = model_artifacts["numeric_cols"]
    scaler = model_artifacts["scaler"]
    all_models = model_artifacts["all_models"]
    best_model_name = model_artifacts["best_model_name"]

    model_name = best_model_name if selected_model_name == "Best (Auto)" else selected_model_name
    model = all_models[model_name]

    sample_df = _build_feature_row(
        feature_columns,
        overs,
        runs,
        wickets,
        runs_last_5,
        wickets_last_5,
        batting_team,
        bowling_team,
    )
    sample_df[numeric_cols] = scaler.transform(sample_df[numeric_cols])

    predicted_score = float(model.predict(sample_df)[0])

    ensemble_predictions = np.array([mdl.predict(sample_df)[0] for mdl in all_models.values()])
    std_dev = float(np.std(ensemble_predictions))
    confidence_low = int(round(max(predicted_score - 1.96 * std_dev, 0)))
    confidence_high = int(round(predicted_score + 1.96 * std_dev))

    return {
        "predicted_score": int(round(predicted_score)),
        "confidence_low": confidence_low,
        "confidence_high": confidence_high,
        "std_dev": round(std_dev, 2),
        "model_name": model_name,
    }


@st.cache_data(show_spinner=False)
def get_feature_importance(model_name):
    model = model_artifacts["all_models"][model_name]
    feature_columns = model_artifacts["feature_columns"]
    if hasattr(model, "feature_importances_"):
        importance = pd.DataFrame(
            {
                "feature": feature_columns,
                "importance": model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)
        return importance.head(15)
    return pd.DataFrame(columns=["feature", "importance"])


df = load_data()
X, y, numeric_cols = preprocess_data(df)

force_retrain = st.sidebar.checkbox("Retrain Models", value=False)
fast_mode = st.sidebar.checkbox("Fast Training Mode", value=FAST_MODE_DEFAULT)

if (not force_retrain) and os.path.exists(MODEL_BUNDLE_PATH):
    model_artifacts = load_saved_model_bundle()
else:
    try:
        model_artifacts = train_models(X, y, numeric_cols, fast_mode)
    except Exception as exc:
        logger.exception("Training failed.")
        if os.path.exists(MODEL_BUNDLE_PATH):
            model_artifacts = load_saved_model_bundle()
            st.warning(
                f"Advanced training failed in this environment ({exc}). Loaded saved model: {model_artifacts['best_model_name']}."
            )
        else:
            st.error(
                "Model training failed and no saved model is available. Please redeploy with Fast Training Mode enabled."
            )
            st.stop()

metrics_df = evaluate_models(model_artifacts["metrics"]) if model_artifacts["metrics"] else pd.DataFrame()

# Streamlit UI
st.title("IPL Score Predictor")

st.sidebar.header("Input Match Details")
batting_team = st.sidebar.selectbox("Select Batting Team", options=team_map.keys())
bowling_team = st.sidebar.selectbox("Select Bowling Team", options=team_map.keys())
overs = st.sidebar.slider("Overs Completed", 5.0, 20.0, step=0.1)
runs = st.sidebar.number_input("Current Runs", min_value=0)
wickets = st.sidebar.number_input("Wickets Lost", min_value=0, max_value=10)
runs_last_5 = st.sidebar.number_input("Runs Scored in Last 5 Overs", min_value=0)
wickets_last_5 = st.sidebar.number_input("Wickets Lost in Last 5 Overs", min_value=0, max_value=10)

model_options = ["Best (Auto)"] + list(model_artifacts["all_models"].keys())
selected_model_name = st.sidebar.selectbox("Model Selection", options=model_options)

st.sidebar.markdown(f"**Best Model (Auto):** {model_artifacts['best_model_name']}")

if st.sidebar.button("Predict Score"):
    with st.spinner("Making Prediction..."):
        prediction = predict_score(
            batting_team,
            bowling_team,
            overs,
            runs,
            wickets,
            runs_last_5,
            wickets_last_5,
            selected_model_name,
        )
        st.subheader("Predicted Final Score")
        st.success(
            f"Using {prediction['model_name']}, predicted score is {prediction['predicted_score']} "
            f"with likely range {prediction['confidence_low']} to {prediction['confidence_high']}."
        )
        st.caption(f"Prediction uncertainty (std dev across models): {prediction['std_dev']}")

if not metrics_df.empty:
    st.markdown("## Model Performance")
    st.dataframe(metrics_df, use_container_width=True)

    selected_for_plot = (
        model_artifacts["best_model_name"]
        if selected_model_name == "Best (Auto)"
        else selected_model_name
    )
    feature_importance_df = get_feature_importance(selected_for_plot)
    if not feature_importance_df.empty:
        st.markdown(f"### Feature Importance ({selected_for_plot})")
        st.bar_chart(feature_importance_df.set_index("feature"))

# Sample Predictions Section
st.markdown("## Sample Predictions from IPL 2018")

sample_1 = predict_score("KKR", "DD", 9.2, 79, 2, 60, 1, selected_model_name)
st.markdown(
    f"""
### Prediction 1
- **Date:** 16th April 2018
- **IPL:** Season 11
- **Match number:** 13
- **Teams:** Kolkata Knight Riders vs. Delhi Daredevils
- **First Innings Final Score:** 200/9
- **Predicted Score Range:** {sample_1['confidence_low']} to {sample_1['confidence_high']}
"""
)

sample_2 = predict_score("SRH", "RCB", 10.5, 67, 3, 29, 1, selected_model_name)
st.markdown(
    f"""
### Prediction 2
- **Date:** 7th May 2018
- **IPL:** Season 11
- **Match number:** 39
- **Teams:** Sunrisers Hyderabad vs. Royal Challengers Bangalore
- **First Innings Final Score:** 146/10
- **Predicted Score Range:** {sample_2['confidence_low']} to {sample_2['confidence_high']}
"""
)

sample_3 = predict_score("MI", "KXIP", 14.1, 136, 4, 50, 0, selected_model_name)
st.markdown(
    f"""
### Prediction 3
- **Date:** 17th May 2018
- **IPL:** Season 11
- **Match number:** 50
- **Teams:** Mumbai Indians vs. Kings XI Punjab
- **First Innings Final Score:** 186/8
- **Predicted Score Range:** {sample_3['confidence_low']} to {sample_3['confidence_high']}
"""
)
