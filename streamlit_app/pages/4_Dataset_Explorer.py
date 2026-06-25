import sys
import os

import streamlit as st
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.theme import inject_base_css, disclaimer_banner, metric_card_html, FEATURE_LABELS, FEATURE_DESCRIPTIONS
from utils.data_loader import load_dataset, load_meta, FEATURES

st.set_page_config(page_title="Dataset Explorer", page_icon="🗂️", layout="wide")
inject_base_css()

df = load_dataset()
meta = load_meta()

st.title("Dataset Explorer & Model Insights")
st.markdown(disclaimer_banner(), unsafe_allow_html=True)

st.markdown("### Dataset")
st.markdown(
    f"""
    Daily OHLCV price data (via `yfinance`) merged with Twitter/news sentiment scored by a fine-tuned
    BERT sentiment model, for NASDAQ/NYSE tickers. Sentiment is reduced to a single daily
    `sentiment_score` per ticker, then turned into the model's actual input features below.

    - **Historical period covered:** {meta['dataset_date_min']} → {meta['dataset_date_max']}
    - **Raw rows loaded:** {meta['n_loaded']:,} across {meta['n_tickers_loaded']:,} tickers
    - **After removing forward-filled / stale rows:** {meta['n_loaded'] - meta['n_stale_removed']:,} rows
      ({meta['n_stale_removed']:,} removed)
    - **After the ≥20-row-per-ticker minimum-history filter:** {meta['n_tickers_after_minhist']:,} tickers
      kept (of {meta['n_tickers_before_minhist']:,})
    - **Final modeling dataset:** {meta['n_train'] + meta['n_test']:,} rows ({meta['n_train']:,} train /
      {meta['n_test']:,} test) across {meta['n_tickers_test']:,} tickers
    - **Test period:** {meta['date_min']} → {meta['date_max']} (per-ticker time split — each stock's own
      last 15% of dates held out, never seen during training)
    """
)

st.markdown("### Features used")
for feat in FEATURES:
    st.markdown(
        f"""
        <div class="feature-cell" style="margin-bottom:10px;">
          <div class="flabel">{FEATURE_LABELS[feat]}</div>
          <div style="color:#EDEFE7; font-size:14px; margin-top:4px;">{FEATURE_DESCRIPTIONS[feat]}</div>
          <div style="color:#9FB3AC; font-size:12px; margin-top:4px;">Gain importance: {meta['importance'][feat]:.2f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("### Why these particular features")
st.markdown(
    """
    `gap_days` and `returns` carry the most weight, ahead of the two sentiment features — the technical
    signal turned out to matter more than tweet sentiment for this dataset and time period. `gap_days`
    exists specifically *because* of the data-quality issues documented in the methodology (real gaps of
    up to 44 days were found): rather than letting those gaps silently distort the other rolling features,
    the model is explicitly told how stale each observation's context is, and apparently uses that
    information meaningfully.
    """
)

st.markdown("---")
st.markdown("### Browse the raw data")

c1, c2, c3 = st.columns([1, 1, 1])
with c1:
    split_filter = st.multiselect("Split", ["train", "test"], default=["test"])
with c2:
    ticker_filter = st.multiselect("Ticker (leave empty = all)", sorted(df["Stock Name"].unique().tolist()))
with c3:
    show_n = st.number_input("Rows to show", min_value=10, max_value=2000, value=100, step=10)

filtered = df[df["split"].isin(split_filter)] if split_filter else df.copy()
if ticker_filter:
    filtered = filtered[filtered["Stock Name"].isin(ticker_filter)]

st.dataframe(filtered.sort_values("Date", ascending=False).head(int(show_n)), width='stretch', height=420)
st.caption(f"{len(filtered):,} rows match the current filter (of {len(df):,} total).")

csv = filtered.to_csv(index=False).encode("utf-8")
st.download_button("Download filtered rows as CSV", csv, "filtered_historical_data.csv", "text/csv")
