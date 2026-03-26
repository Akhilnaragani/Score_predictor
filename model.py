import logging
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from xgboost import XGBRegressor


logger = logging.getLogger("ipl_score_predictor")
MODEL_BUNDLE_PATH = "ipl_score_predictor_model.pkl"
RANDOM_STATE = 42


def _model_configs(fast_mode):
    if fast_mode:
        n_iter = 3
        params = {
            "XGBoost": {
                "n_estimators": [120, 180, 240],
                "max_depth": [4, 6],
                "learning_rate": [0.05, 0.1],
                "subsample": [0.8, 1.0],
            },
            "LightGBM": {
                "n_estimators": [120, 180, 260],
                "num_leaves": [31, 63],
                "learning_rate": [0.05, 0.1],
                "max_depth": [-1, 8],
            },
            "CatBoost": {
                "iterations": [120, 200, 300],
                "depth": [4, 6],
                "learning_rate": [0.05, 0.1],
            },
        }
    else:
        n_iter = 8
        params = {
            "XGBoost": {
                "n_estimators": [200, 350, 500],
                "max_depth": [4, 6, 8],
                "learning_rate": [0.03, 0.05, 0.1],
                "subsample": [0.8, 0.9, 1.0],
                "colsample_bytree": [0.8, 1.0],
            },
            "LightGBM": {
                "n_estimators": [200, 350, 500],
                "num_leaves": [31, 63, 127],
                "learning_rate": [0.03, 0.05, 0.1],
                "max_depth": [-1, 8, 12],
            },
            "CatBoost": {
                "iterations": [250, 400, 600],
                "depth": [4, 6, 8],
                "learning_rate": [0.03, 0.05, 0.1],
                "l2_leaf_reg": [3, 5, 7],
            },
        }

    models = {
        "XGBoost": XGBRegressor(
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=1,
            verbosity=0,
        ),
        "LightGBM": LGBMRegressor(random_state=RANDOM_STATE, n_jobs=1, verbose=-1),
        "CatBoost": CatBoostRegressor(random_seed=RANDOM_STATE, thread_count=1, verbose=0),
    }
    return models, params, n_iter


def _evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return float(mae), float(rmse), float(r2)


@st.cache_resource(show_spinner=False)
def train_models(X, y, feature_columns, fast_mode=True):
    logger.info("Training advanced models | fast_mode=%s", fast_mode)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    models, params, n_iter = _model_configs(fast_mode)

    trained_models = {}
    metrics = {}

    for name, model in models.items():
        try:
            tuner = RandomizedSearchCV(
                estimator=model,
                param_distributions=params[name],
                n_iter=n_iter,
                scoring="neg_mean_absolute_error",
                cv=cv,
                random_state=RANDOM_STATE,
                n_jobs=1,
                verbose=0,
            )
            tuner.fit(X_train, y_train)
            best_model = tuner.best_estimator_
            pred = best_model.predict(X_test)
            mae, rmse, r2 = _evaluate(y_test, pred)
            trained_models[name] = best_model
            metrics[name] = {
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
                "CV_MAE": float(-tuner.best_score_),
            }
            logger.info("%s done | MAE %.3f RMSE %.3f R2 %.3f", name, mae, rmse, r2)
        except Exception as exc:
            logger.exception("Model training failed for %s: %s", name, exc)

    if not trained_models:
        fallback = RandomForestRegressor(n_estimators=180, max_depth=12, random_state=RANDOM_STATE, n_jobs=1)
        fallback.fit(X_train, y_train)
        pred = fallback.predict(X_test)
        mae, rmse, r2 = _evaluate(y_test, pred)
        trained_models = {"RandomForestFallback": fallback}
        metrics = {"RandomForestFallback": {"MAE": mae, "RMSE": rmse, "R2": r2, "CV_MAE": mae}}

    stack_model_name = None
    if len(trained_models) >= 2 and all(name in trained_models for name in ["XGBoost", "LightGBM", "CatBoost"]):
        stack = StackingRegressor(
            estimators=[
                ("xgb", trained_models["XGBoost"]),
                ("lgbm", trained_models["LightGBM"]),
                ("cat", trained_models["CatBoost"]),
            ],
            final_estimator=Ridge(alpha=1.0),
            n_jobs=1,
            passthrough=False,
        )
        stack.fit(X_train, y_train)
        pred = stack.predict(X_test)
        mae, rmse, r2 = _evaluate(y_test, pred)
        trained_models["StackingEnsemble"] = stack
        metrics["StackingEnsemble"] = {
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "CV_MAE": mae,
        }
        stack_model_name = "StackingEnsemble"

    best_model_name = min(metrics.keys(), key=lambda k: metrics[k]["MAE"])

    # Keep evaluation frame for diagnostics visualization.
    best_predictions = trained_models[best_model_name].predict(X_test)
    eval_frame = pd.DataFrame({"actual": y_test, "predicted": best_predictions})

    bundle = {
        "best_model_name": best_model_name,
        "models": trained_models,
        "metrics": metrics,
        "feature_columns": feature_columns,
        "eval_frame": eval_frame,
        "stack_model_name": stack_model_name,
    }
    return bundle


def save_model_bundle(bundle, preprocess_artifacts):
    serializable = {
        "best_model_name": bundle["best_model_name"],
        "models": bundle["models"],
        "metrics": bundle["metrics"],
        "feature_columns": bundle["feature_columns"],
        "eval_frame": bundle["eval_frame"],
        "stack_model_name": bundle["stack_model_name"],
        "preprocess_artifacts": preprocess_artifacts,
    }
    joblib.dump(serializable, MODEL_BUNDLE_PATH)


def load_model_bundle():
    if not os.path.exists(MODEL_BUNDLE_PATH):
        return None
    return joblib.load(MODEL_BUNDLE_PATH)


def evaluate_models_table(metrics):
    rows = []
    for name, vals in metrics.items():
        mae = float(vals.get("MAE", 0.0))
        rmse = float(vals.get("RMSE", mae))
        r2 = vals.get("R2")
        if r2 is None:
            r2 = vals.get("R_SQUARED", np.nan)

        rows.append(
            {
                "Model": name,
                "MAE": round(mae, 3),
                "RMSE": round(rmse, 3),
                "R2": round(float(r2), 3) if pd.notna(r2) else np.nan,
                "CV_MAE": round(float(vals.get("CV_MAE", mae)), 3),
            }
        )
    return pd.DataFrame(rows).sort_values("MAE")


def predict_match(bundle, transformed_row, selected_model_name="Best (Auto)", target_score=None, is_chasing=False):
    models = bundle["models"]
    best = bundle["best_model_name"]
    model_name = best if selected_model_name == "Best (Auto)" else selected_model_name
    if model_name not in models:
        model_name = best

    model = models[model_name]
    pred = float(model.predict(transformed_row)[0])

    ensemble_preds = np.array([mdl.predict(transformed_row)[0] for mdl in models.values()])
    spread = float(np.std(ensemble_preds)) if len(ensemble_preds) > 1 else 8.0

    low = int(round(max(pred - 1.65 * spread, 0)))
    high = int(round(pred + 1.65 * spread))

    confidence = float(np.clip(100.0 - spread * 3.5, 52.0, 95.0))

    if is_chasing and target_score is not None:
        chase_margin = pred - float(target_score)
        win_prob = float(1.0 / (1.0 + np.exp(-chase_margin / 12.0)))
    else:
        par = 175.0
        defend_margin = pred - par
        win_prob = float(1.0 / (1.0 + np.exp(-defend_margin / 14.0)))

    return {
        "model": model_name,
        "predicted_score": int(round(pred)),
        "range_low": low,
        "range_high": high,
        "confidence": round(confidence, 1),
        "win_probability": round(win_prob * 100.0, 1),
        "spread": round(spread, 2),
    }


def feature_importance(bundle, model_name):
    selected = model_name
    if selected == "Best (Auto)":
        selected = bundle["best_model_name"]

    model = bundle["models"].get(selected)
    if model is None:
        return pd.DataFrame(columns=["feature", "importance"])

    if hasattr(model, "feature_importances_"):
        imp = pd.DataFrame(
            {"feature": bundle["feature_columns"], "importance": model.feature_importances_}
        ).sort_values("importance", ascending=False)
        return imp.head(20)

    return pd.DataFrame(columns=["feature", "importance"])
