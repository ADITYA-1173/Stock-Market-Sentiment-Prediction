import streamlit as st
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.theme import inject_base_css, metric_card_html, disclaimer_banner, TEAL_BRIGHT, CLAY_BRIGHT, AMBER_BRIGHT
from utils.data_loader import load_dataset, load_meta

st.set_page_config(
    page_title="The Abstaining Trader — Overview",
    page_icon="📉",
    layout="wide",
)
inject_base_css()

meta = load_meta()
df = load_dataset()

st.markdown(disclaimer_banner(), unsafe_allow_html=True)

st.markdown(
    """
    <p class="mono" style="color:#9FB3AC; letter-spacing:.12em; text-transform:uppercase; font-size:12px;">
    HISTORICAL REPLAY &nbsp;·&nbsp; OUT-OF-SAMPLE TEST SET &nbsp;·&nbsp; NOT A LIVE TRADING SYSTEM
    </p>
    """,
    unsafe_allow_html=True,
)
st.title("The Abstaining Trader")
st.markdown(
    f"""
    <p style="max-width:760px; font-size:16px; color:#9FB3AC;">
    An XGBoost classifier trained on BERT-scored Twitter sentiment and price technicals to predict
    next-day stock direction — and explicitly designed to <b style="color:#EDEFE7;">abstain rather than guess</b>
    when it isn't confident. This dashboard replays its real, historical, out-of-sample predictions.
    It does not connect to live markets and cannot be used for live trading — there is no live data
    pipeline here by design.
    </p>
    """,
    unsafe_allow_html=True,
)

st.markdown("### Headline results")
cols = st.columns(6)
metrics = [
    (f"{meta['roc_auc']:.4f}", "Test AUC (all rows)"),
    (f"{meta['accuracy']*100:.1f}%", "Accuracy (signals)"),
    (f"{meta['coverage_pct']:.1f}%", "Signal coverage"),
    (meta["n_signals"], "Trades taken"),
    (meta["n_tickers_test"], "Tickers in test"),
    (f"{meta['n_train']:,}", "Train rows"),
]
for col, (num, label) in zip(cols, metrics):
    col.markdown(metric_card_html(num, label), unsafe_allow_html=True)

st.markdown("---")

c1, c2 = st.columns([1.3, 1])
with c1:
    st.markdown("### What makes this project different")
    st.markdown(
        """
        Most "Twitter sentiment predicts stocks" projects report inflated accuracy because of subtle
        evaluation leakage. Before trusting any number here, four specific bugs were found and fixed:

        - **Train/test split** was alphabetical by ticker, not by time — 476 of 477 test tickers had
          never appeared in training. Fixed with a per-ticker time split.
        - **Rolling windows** counted rows, not days — a "7-day" window silently spanned up to 44 real
          days during data gaps. Fixed with calendar-time-aware rolling (`'7D'`, `'14D'`).
        - **Stale rows**: forward-filled weekend/holiday prices were treated as genuine flat trading
          days. 15,485 of them were detected and removed.
        - **Sparse tickers** with too little history vanished silently through a blanket `dropna()`.
          Now filtered explicitly, with the dropped count logged (2,999 of 3,646 tickers).

        See the **Model Explainability** and **Dataset Explorer** pages for the full breakdown, and the
        repository README for the complete before/after methodology.
        """
    )
with c2:
    st.markdown("### Navigate")
    st.markdown(
        """
        - **Historical Replay** — step day-by-day through a ticker's history, watching sentiment,
          technicals, and the model's signal evolve.
        - **Trade Simulation** — an educational paper-trading run over any date range.
        - **Model Explainability** — SHAP-driven, plain-English reasoning for any individual prediction.
        - **Dataset Explorer** — the raw numbers, metrics, and the leakage-fix methodology in detail.
        """
    )

st.markdown("---")
st.markdown(
    f"""
    <p class="mono" style="font-size:12px; color:#9FB3AC;">
    Dataset coverage: {meta['dataset_date_min']} → {meta['dataset_date_max']} &nbsp;|&nbsp;
    Test period: {meta['date_min']} → {meta['date_max']} &nbsp;|&nbsp;
    {meta['n_tickers_after_minhist']} tickers with sufficient history (of {meta['n_tickers_loaded']:,} originally)
    </p>
    """,
    unsafe_allow_html=True,
)
