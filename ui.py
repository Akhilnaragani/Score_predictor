import logging

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_loader import TEAM_MAP, TEAM_VALUES, build_enhanced_dataset, load_data
from model import (
    evaluate_models_table,
    feature_importance,
    load_model_bundle,
    predict_match,
    save_model_bundle,
    train_models,
)
from preprocessing import preprocess_data, transform_match_input


logger = logging.getLogger("ipl_score_predictor")
FAST_MODE_DEFAULT = True


def _safe_stat(value, fallback):
    try:
        if value is None or np.isnan(value):
            return fallback
    except TypeError:
        return fallback
    return float(value)


THEME_CSS = """
<style>
    .stApp {
        background: radial-gradient(circle at 10% 10%, #111929 0%, #090f1c 42%, #060b16 100%);
        color: #f4f7ff;
    }
    .card {
        border: 1px solid rgba(90, 145, 255, 0.25);
        border-radius: 14px;
        background: linear-gradient(180deg, rgba(11,18,35,0.95), rgba(8,13,24,0.95));
        padding: 16px;
    }
    .score-main {
        font-size: 46px;
        font-weight: 800;
        color: #7ee3ff;
        line-height: 1.1;
    }
    .muted {
        color: #9bb1d4;
    }
    .accent-good {
        color: #49d17f;
        font-weight: 700;
    }
    .accent-pressure {
        color: #ff6f6f;
        font-weight: 700;
    }
</style>
"""


@st.cache_data(show_spinner=False)
def _build_training_artifacts():
    raw = load_data()
    enhanced = build_enhanced_dataset(raw)
    return enhanced


def _build_input_frame(inputs):
    df = pd.DataFrame(
        [
            {
                "bat_team": TEAM_MAP[inputs["batting_team"]],
                "bowl_team": TEAM_MAP[inputs["bowling_team"]],
                "venue": inputs["venue"],
                "toss_winner": TEAM_MAP[inputs["toss_winner"]],
                "toss_decision": inputs["toss_decision"],
                "overs": inputs["overs"],
                "runs": inputs["runs"],
                "wickets": inputs["wickets"],
                "runs_last_5": inputs["runs_last_5"],
                "wickets_last_5": inputs["wickets_last_5"],
                "total": max(inputs.get("target_score", 175), inputs["runs"] + 1),
            }
        ]
    )

    balls_bowled = int(np.floor(inputs["overs"])) * 6 + int(round((inputs["overs"] % 1) * 10))
    balls_bowled = min(max(balls_bowled, 0), 120)
    balls_remaining = 120 - balls_bowled

    df["balls_remaining"] = balls_remaining
    df["wickets_remaining"] = 10 - inputs["wickets"]
    df["current_run_rate"] = inputs["runs"] / max(inputs["overs"], 0.1)
    df["required_run_rate"] = (
        max(inputs.get("target_score", 0) - inputs["runs"], 0) / max(balls_remaining / 6.0, 1 / 6.0)
        if inputs["is_chasing"]
        else 0.0
    )
    df["pressure_index"] = inputs["wickets"] * max(20 - inputs["overs"], 0)
    df["momentum_score"] = inputs["runs_last_5"] / 5.0
    df["team_recent_form"] = inputs["team_recent_form"]
    df["opponent_strength"] = inputs["opponent_strength"]
    df["venue_avg_score"] = inputs["venue_avg_score"]
    df["team_strength_score"] = inputs["team_strength_score"]
    df["venue_factor"] = inputs["venue_factor"]
    df["death_over_boost"] = df["momentum_score"] * (1.2 if inputs["overs"] >= 15 else 0.8)
    df["powerplay_score_rate"] = inputs["powerplay_score_rate"]
    df["middle_score_rate"] = inputs["middle_score_rate"]
    df["death_score_rate"] = inputs["death_score_rate"]
    return df


def _run_rate_curve(overs, runs, momentum):
    x = np.arange(1, int(max(np.ceil(overs), 6)) + 1)
    base_rr = runs / max(overs, 0.1)
    y = base_rr + np.sin(x / 1.7) * 0.35 + (momentum - 8) * 0.05
    return pd.DataFrame({"over": x, "run_rate": np.clip(y, 2, 18)})


def _preset_values(name):
    presets = {
        "Balanced Start (50/2 in 6)": {"overs": 6.0, "runs": 50, "wickets": 2, "runs_last_5": 40, "wickets_last_5": 1},
        "Aggressive Start (78/1 in 6)": {"overs": 6.0, "runs": 78, "wickets": 1, "runs_last_5": 58, "wickets_last_5": 0},
        "Collapse (42/4 in 8)": {"overs": 8.0, "runs": 42, "wickets": 4, "runs_last_5": 24, "wickets_last_5": 2},
    }
    return presets.get(name)


def run_app():
    st.set_page_config(page_title="IPL AI Predictor Pro", page_icon="🏏", layout="wide")
    st.markdown(THEME_CSS, unsafe_allow_html=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

    enhanced_df = _build_training_artifacts()

    if "bundle" not in st.session_state:
        loaded = load_model_bundle()
        if loaded is not None:
            st.session_state.bundle = {
                "best_model_name": loaded["best_model_name"],
                "models": loaded["models"],
                "metrics": loaded["metrics"],
                "feature_columns": loaded["feature_columns"],
                "eval_frame": loaded.get("eval_frame", pd.DataFrame(columns=["actual", "predicted"])),
                "stack_model_name": loaded.get("stack_model_name"),
            }
            st.session_state.preprocess_artifacts = loaded["preprocess_artifacts"]
        else:
            X, y, prep = preprocess_data(enhanced_df, encode_method="target")
            bundle = train_models(X, y, list(X.columns), fast_mode=True)
            save_model_bundle(bundle, prep)
            st.session_state.bundle = bundle
            st.session_state.preprocess_artifacts = prep

    bundle = st.session_state.bundle
    preprocess_artifacts = st.session_state.preprocess_artifacts

    st.sidebar.markdown("### Match Controls")
    quick_mode = st.sidebar.toggle("Quick Prediction", value=True)
    advanced_mode = st.sidebar.toggle("Advanced Prediction", value=False)
    fast_mode = st.sidebar.toggle("Fast Training Mode", value=FAST_MODE_DEFAULT)
    simulate_live = st.sidebar.toggle("Live Simulation", value=False)

    preset = st.sidebar.selectbox(
        "Scenario Preset",
        ["None", "Balanced Start (50/2 in 6)", "Aggressive Start (78/1 in 6)", "Collapse (42/4 in 8)"],
    )

    if st.sidebar.button("Retrain Advanced Models"):
        with st.spinner("Retraining models with latest data..."):
            X, y, prep = preprocess_data(enhanced_df, encode_method="target")
            new_bundle = train_models(X, y, list(X.columns), fast_mode=fast_mode)
            save_model_bundle(new_bundle, prep)
            st.session_state.bundle = new_bundle
            st.session_state.preprocess_artifacts = prep
            bundle = new_bundle
            preprocess_artifacts = prep
            st.success(f"Retrained. Best model: {bundle['best_model_name']}")

    st.sidebar.markdown("### Input Match Details")
    batting_team = st.sidebar.selectbox("Batting Team", list(TEAM_MAP.keys()), index=0)
    bowling_team = st.sidebar.selectbox("Bowling Team", list(TEAM_MAP.keys()), index=4)

    venue_options = sorted(enhanced_df["venue"].astype(str).dropna().unique().tolist())
    venue = st.sidebar.selectbox("Venue", venue_options if venue_options else ["Unknown Venue"])

    toss_winner = st.sidebar.selectbox("Toss Winner", list(TEAM_MAP.keys()), index=0)
    toss_decision = st.sidebar.selectbox("Toss Decision", ["bat", "field"])

    is_chasing = st.sidebar.toggle("Batting Second (Chasing)", value=False)

    default_vals = {"overs": 6.0, "runs": 50, "wickets": 2, "runs_last_5": 40, "wickets_last_5": 1}
    preset_data = _preset_values(preset)
    if preset_data:
        default_vals.update(preset_data)

    overs = st.sidebar.slider("Overs Completed", 0.1, 20.0, float(default_vals["overs"]), 0.1)
    runs = st.sidebar.number_input("Current Runs", min_value=0, value=int(default_vals["runs"]))
    wickets = st.sidebar.number_input("Wickets Lost", min_value=0, max_value=10, value=int(default_vals["wickets"]))
    runs_last_5 = st.sidebar.number_input("Runs in Last 5 Overs", min_value=0, value=int(default_vals["runs_last_5"]))
    wickets_last_5 = st.sidebar.number_input("Wickets in Last 5 Overs", min_value=0, max_value=10, value=int(default_vals["wickets_last_5"]))
    target_score = st.sidebar.number_input("Target Score (if chasing)", min_value=0, value=180)

    model_options = ["Best (Auto)"] + list(bundle["models"].keys())
    selected_model = st.sidebar.selectbox("Model Selection", model_options)

    team_recent_form = _safe_stat(
        enhanced_df[enhanced_df["bat_team"] == TEAM_MAP[batting_team]]["team_recent_form"].median(),
        170.0,
    )
    opponent_strength = _safe_stat(
        enhanced_df[enhanced_df["bowl_team"] == TEAM_MAP[bowling_team]]["opponent_strength"].median(),
        170.0,
    )
    venue_avg_score = _safe_stat(
        enhanced_df[enhanced_df["venue"] == venue]["venue_avg_score"].median(),
        165.0,
    )
    team_strength_score = _safe_stat(
        enhanced_df[enhanced_df["bat_team"] == TEAM_MAP[batting_team]]["team_strength_score"].median(),
        170.0,
    )
    venue_factor = _safe_stat(
        enhanced_df[enhanced_df["venue"] == venue]["venue_factor"].median(),
        1.0,
    )

    powerplay_rate = float(enhanced_df["powerplay_score_rate"].mean())
    middle_rate = float(enhanced_df["middle_score_rate"].mean())
    death_rate = float(enhanced_df["death_score_rate"].mean())

    match_inputs = {
        "batting_team": batting_team,
        "bowling_team": bowling_team,
        "venue": venue,
        "toss_winner": toss_winner,
        "toss_decision": toss_decision,
        "overs": overs,
        "runs": runs,
        "wickets": wickets,
        "runs_last_5": runs_last_5,
        "wickets_last_5": wickets_last_5,
        "is_chasing": is_chasing,
        "target_score": target_score,
        "team_recent_form": team_recent_form,
        "opponent_strength": opponent_strength,
        "venue_avg_score": venue_avg_score,
        "team_strength_score": team_strength_score,
        "venue_factor": venue_factor,
        "powerplay_score_rate": powerplay_rate,
        "middle_score_rate": middle_rate,
        "death_score_rate": death_rate,
    }

    if simulate_live:
        overs = min(20.0, overs + 0.1)
        match_inputs["overs"] = overs
        st.sidebar.caption("Live simulation adjusted overs by +0.1")

    input_df = _build_input_frame(match_inputs)
    transformed_row = transform_match_input(input_df, preprocess_artifacts)

    prediction = predict_match(
        bundle,
        transformed_row,
        selected_model_name=selected_model,
        target_score=target_score,
        is_chasing=is_chasing,
    )

    current_rr = runs / max(overs, 0.1)
    required_rr = (max(target_score - runs, 0) / max((120 - int(overs * 6)) / 6, 1 / 6)) if is_chasing else 0.0
    innings_progress = min(max((overs / 20.0), 0.0), 1.0)

    st.title("IPL Score Predictor")

    top_left, top_mid, top_right = st.columns([1.2, 1.2, 1.2])
    with top_left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Main Score Card")
        st.markdown(f"<div class='score-main'>{runs}/{wickets}</div>", unsafe_allow_html=True)
        st.markdown(f"Overs: {overs:.1f}")
        st.markdown(f"Run Rate: {current_rr:.2f}")
        if is_chasing:
            st.markdown(f"Required RR: {required_rr:.2f}")
        st.progress(innings_progress)
        st.markdown("</div>", unsafe_allow_html=True)

    with top_mid:
        pressure_label = "Good Control" if prediction["confidence"] >= 70 else "High Pressure"
        pressure_class = "accent-good" if prediction["confidence"] >= 70 else "accent-pressure"

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### AI Prediction Panel")
        st.markdown(
            f"Predicted Range: **{prediction['range_low']} - {prediction['range_high']}**  \\\nModel: **{prediction['model']}**"
        )
        st.markdown(f"Confidence: **{prediction['confidence']}%**")
        st.markdown(f"Win Probability: **{prediction['win_probability']}%**")
        st.markdown(f"<span class='{pressure_class}'>{pressure_label}</span>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with top_right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Match Insights")
        st.write(f"Venue Avg Score: {venue_avg_score:.1f}")
        st.write(f"Team Form (Last 5 avg): {team_recent_form:.1f}")
        st.write(f"Opponent Strength: {opponent_strength:.1f}")
        st.write(f"Momentum: {input_df['momentum_score'].iloc[0]:.2f}")
        st.write(f"Pressure Index: {input_df['pressure_index'].iloc[0]:.2f}")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### Run Rate Graph")
    rr_df = _run_rate_curve(overs, runs, input_df["momentum_score"].iloc[0])
    rr_fig = px.line(rr_df, x="over", y="run_rate", markers=True)
    rr_fig.update_layout(height=280, template="plotly_dark", margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(rr_fig, use_container_width=True)

    st.markdown("### Momentum Indicator")
    momentum_value = float(input_df["momentum_score"].iloc[0])
    st.progress(min(max(momentum_value / 20.0, 0.0), 1.0))
    st.caption(f"Momentum score: {momentum_value:.2f}")

    metrics_df = evaluate_models_table(bundle["metrics"])
    st.markdown("### Model Performance")
    st.dataframe(metrics_df, use_container_width=True)

    st.markdown("### Predicted vs Actual")
    eval_frame = bundle.get("eval_frame", pd.DataFrame(columns=["actual", "predicted"]))
    if not eval_frame.empty:
        fig_eval = px.scatter(eval_frame.sample(min(len(eval_frame), 350), random_state=1), x="actual", y="predicted", opacity=0.65)
        fig_eval.add_trace(
            go.Scatter(
                x=[eval_frame["actual"].min(), eval_frame["actual"].max()],
                y=[eval_frame["actual"].min(), eval_frame["actual"].max()],
                mode="lines",
                name="Ideal",
            )
        )
        fig_eval.update_layout(height=320, template="plotly_dark", margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_eval, use_container_width=True)

    st.markdown("### Feature Importance")
    fi_df = feature_importance(bundle, selected_model)
    if not fi_df.empty:
        fi_fig = px.bar(fi_df.head(15), x="importance", y="feature", orientation="h")
        fi_fig.update_layout(height=420, template="plotly_dark", margin=dict(l=10, r=10, t=30, b=10), yaxis=dict(autorange="reversed"))
        st.plotly_chart(fi_fig, use_container_width=True)

    if quick_mode and not advanced_mode:
        st.caption("Quick Prediction mode enabled: optimized for lower latency.")
    if advanced_mode:
        st.caption("Advanced mode enabled: richer analytics and full model diagnostics.")
