"""
Cached data loading. All pages call these instead of reading the CSV
directly, so the dataset and model only load once per session.
"""
import json
import os
import streamlit as st
import pandas as pd
import xgboost as xgb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "data", "historical_replay_dataset.csv")
META_PATH = os.path.join(ROOT, "data", "model_meta.json")
MODEL_PATH = os.path.join(ROOT, "models", "final_stock_model.json")

FEATURES = ["sent_zscore", "sent_ma_3", "returns", "volatility", "rsi", "gap_days"]
SHAP_COLS = [f"shap_{f}" for f in FEATURES]


@st.cache_data
def load_dataset():
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    return df


@st.cache_data
def load_meta():
    with open(META_PATH) as f:
        return json.load(f)


@st.cache_resource
def load_model():
    model = xgb.Booster()
    model.load_model(MODEL_PATH)
    return model


def verdict_for_row(row):
    """Returns (badge_text, css_class, sub_text) for a given dataset row."""
    if row["pred_label"] == -1:
        return (
            "NO TRADE",
            "verdict-amber",
            "Predicted probability landed inside the 48-60% hold zone — model abstained rather than guess.",
        )
    correct = row["pred_label"] == row["Target_Trend"]
    direction = "BUY SIGNAL" if row["pred_label"] == 1 else "SELL SIGNAL"
    if correct:
        return (f"{direction} — CORRECT", "verdict-teal", "Next-day direction matched the call.")
    return (f"{direction} — INCORRECT", "verdict-clay", "Next-day direction went the other way.")
