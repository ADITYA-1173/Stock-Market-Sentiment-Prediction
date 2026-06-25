import sys
import os

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.theme import inject_base_css, disclaimer_banner, metric_card_html, FEATURE_LABELS, TEAL_BRIGHT, CLAY_BRIGHT, AMBER_BRIGHT
from utils.data_loader import load_dataset, load_meta, FEATURES, verdict_for_row
from utils.explain import generate_explanation, feature_breakdown

st.set_page_config(page_title="Model Explainability", page_icon="🔍", layout="wide")
inject_base_css()

df = load_dataset()
meta = load_meta()
test_df = df[df["split"] == "test"].copy()
signals_df = test_df[test_df["pred_label"] != -1].copy()

st.title("Model Explainability")
st.markdown(disclaimer_banner(), unsafe_allow_html=True)
st.markdown("Why the model makes the calls it makes — globally, across all out-of-sample signals, and for any single prediction.")

st.markdown("### Model metrics (test set)")
cols = st.columns(5)
m = [
    (f"{meta['accuracy']*100:.1f}%", "Accuracy"),
    (f"{meta['precision']*100:.1f}%", "Precision"),
    (f"{meta['recall']*100:.1f}%", "Recall"),
    (f"{meta['f1']*100:.1f}%", "F1 score"),
    (f"{meta['roc_auc']:.4f}", "ROC AUC"),
]
for col, (num, label) in zip(cols, m):
    col.markdown(metric_card_html(num, label), unsafe_allow_html=True)
st.caption("Accuracy / Precision / Recall / F1 are computed on the 188 high-confidence signals only. ROC AUC is computed across all 3,109 out-of-sample rows.")

st.markdown("---")
left, right = st.columns([1, 1])

with left:
    st.markdown("### Feature importance (gain)")
    imp = pd.DataFrame(
        [(FEATURE_LABELS[k], v) for k, v in sorted(meta["importance"].items(), key=lambda x: -x[1])],
        columns=["Feature", "Gain"],
    )
    fig = px.bar(imp, x="Gain", y="Feature", orientation="h", color_discrete_sequence=[TEAL_BRIGHT])
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0E1F1B", plot_bgcolor="#0E1F1B",
        font=dict(family="IBM Plex Mono, monospace", color="#EDEFE7"),
        yaxis=dict(autorange="reversed"), height=320, margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig, width='stretch')

with right:
    st.markdown("### Confusion matrix (188 signals)")
    cm = meta["cm"]
    z = [[cm[0][0], cm[0][1]], [cm[1][0], cm[1][1]]]
    fig_cm = go.Figure(data=go.Heatmap(
        z=[[cm[1][0], cm[1][1]], [cm[0][0], cm[0][1]]],
        x=["Pred. Down", "Pred. Up"], y=["Actual Up", "Actual Down"],
        colorscale=[[0, "#16302A"], [1, TEAL_BRIGHT]],
        text=[[cm[1][0], cm[1][1]], [cm[0][0], cm[0][1]]], texttemplate="%{text}",
        textfont=dict(size=20, family="IBM Plex Mono, monospace"), showscale=False,
    ))
    fig_cm.update_layout(
        template="plotly_dark", paper_bgcolor="#0E1F1B", plot_bgcolor="#0E1F1B",
        font=dict(family="IBM Plex Mono, monospace", color="#EDEFE7"),
        height=320, margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig_cm, width='stretch')

st.markdown("---")
left2, right2 = st.columns([1, 1])

with left2:
    st.markdown("### Confidence distribution (all 3,109 test rows)")
    fig_conf = go.Figure()
    fig_conf.add_trace(go.Histogram(x=test_df["pred_prob"], nbinsx=40, marker_color=TEAL_BRIGHT, opacity=0.85))
    fig_conf.add_vrect(x0=0.48, x1=0.60, fillcolor=AMBER_BRIGHT, opacity=0.15, line_width=0,
                        annotation_text="hold zone", annotation_position="top")
    fig_conf.update_layout(
        template="plotly_dark", paper_bgcolor="#0E1F1B", plot_bgcolor="#0E1F1B",
        font=dict(family="IBM Plex Mono, monospace", color="#EDEFE7"),
        xaxis_title="P(up)", yaxis_title="Count", height=300, margin=dict(l=10, r=10, t=30, b=10),
    )
    st.plotly_chart(fig_conf, width='stretch')

with right2:
    st.markdown("### Prediction distribution")
    counts = signals_df["pred_label"].map({1: "BUY", 0: "SELL"}).value_counts()
    n_abstain = len(test_df) - len(signals_df)
    labels = list(counts.index) + ["NO TRADE (hold zone)"]
    values = list(counts.values) + [n_abstain]
    colors = [TEAL_BRIGHT if l == "BUY" else CLAY_BRIGHT if l == "SELL" else AMBER_BRIGHT for l in labels]
    fig_pred = go.Figure(data=[go.Bar(x=labels, y=values, marker_color=colors)])
    fig_pred.update_layout(
        template="plotly_dark", paper_bgcolor="#0E1F1B", plot_bgcolor="#0E1F1B",
        font=dict(family="IBM Plex Mono, monospace", color="#EDEFE7"),
        yaxis_title="Count", height=300, margin=dict(l=10, r=10, t=30, b=10),
    )
    st.plotly_chart(fig_pred, width='stretch')

st.markdown("---")
st.markdown("### Explain a single prediction")
st.caption("Pick any out-of-sample signal and see exactly which features pushed the call, and by how much (SHAP values, log-odds space).")

c1, c2 = st.columns([1, 1])
with c1:
    sel_ticker = st.selectbox("Ticker", sorted(signals_df["Stock Name"].unique().tolist()), key="explain_ticker")
ticker_signals = signals_df[signals_df["Stock Name"] == sel_ticker].sort_values("Date")
with c2:
    sel_date = st.selectbox("Date", ticker_signals["Date"].dt.date.tolist(), key="explain_date")

row = ticker_signals[ticker_signals["Date"].dt.date == sel_date].iloc[0]
badge, cls, sub = verdict_for_row(row)
st.markdown(
    f'<div class="verdict-box {cls}"><span class="badge {cls.replace("verdict-","badge-")}">{badge}</span>'
    f'<span style="color:#9FB3AC; font-size:13.5px;">{sub}</span></div>',
    unsafe_allow_html=True,
)
st.info(generate_explanation(row))

breakdown = feature_breakdown(row)
bd_df = pd.DataFrame(breakdown)
bd_df["color"] = bd_df["pushes_up"].map({True: TEAL_BRIGHT, False: CLAY_BRIGHT})
fig_shap = go.Figure(go.Bar(
    x=bd_df["shap"], y=bd_df["label"], orientation="h", marker_color=bd_df["color"],
))
fig_shap.add_vline(x=0, line_color="#9FB3AC")
fig_shap.update_layout(
    template="plotly_dark", paper_bgcolor="#0E1F1B", plot_bgcolor="#0E1F1B",
    font=dict(family="IBM Plex Mono, monospace", color="#EDEFE7"),
    xaxis_title="SHAP contribution (log-odds, → pushes UP, ← pushes DOWN)",
    height=280, margin=dict(l=10, r=10, t=20, b=10),
)
st.plotly_chart(fig_shap, width='stretch')
