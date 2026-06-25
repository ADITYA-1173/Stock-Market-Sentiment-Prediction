import sys
import os
import time

import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.theme import inject_base_css, disclaimer_banner
from utils.data_loader import load_dataset, verdict_for_row
from utils.gauge import gauge_svg
from utils.explain import generate_explanation

st.set_page_config(page_title="Historical Replay", page_icon="🎞️", layout="wide")
inject_base_css()

df = load_dataset()
test_df = df[df["split"] == "test"].copy()

st.title("Historical Replay")
st.markdown(disclaimer_banner(), unsafe_allow_html=True)
st.markdown(
    "Step through a ticker's out-of-sample history day by day, exactly as the model saw it — "
    "sentiment, technicals, the model's confidence, and what actually happened next."
)

# ---- Controls ----
tickers = sorted(test_df["Stock Name"].unique().tolist())
default_idx = tickers.index("AAPL") if "AAPL" in tickers else 0

c1, c2, c3 = st.columns([1, 1, 1])
with c1:
    ticker = st.selectbox("Ticker", tickers, index=default_idx)
ticker_df = test_df[test_df["Stock Name"] == ticker].sort_values("Date").reset_index(drop=True)
min_date, max_date = ticker_df["Date"].min().date(), ticker_df["Date"].max().date()

with c2:
    date_range = st.date_input(
        "Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date
    )
with c3:
    speed = st.select_slider("Replay speed", options=["Slow", "Medium", "Fast"], value="Medium")
speed_map = {"Slow": 1.2, "Medium": 0.6, "Fast": 0.15}

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_d, end_d = date_range
else:
    start_d, end_d = min_date, max_date

mask = (ticker_df["Date"].dt.date >= start_d) & (ticker_df["Date"].dt.date <= end_d)
window_df = ticker_df[mask].reset_index(drop=True)

if len(window_df) == 0:
    st.warning("No test-set rows for that ticker in this date range. Widen the range.")
    st.stop()

if "replay_idx" not in st.session_state or st.session_state.get("replay_ticker") != ticker:
    st.session_state.replay_idx = 0
    st.session_state.replay_ticker = ticker

st.session_state.replay_idx = min(st.session_state.replay_idx, len(window_df) - 1)

nav1, nav2, nav3, nav4 = st.columns([1, 1, 1, 3])
with nav1:
    if st.button("⏮ Reset"):
        st.session_state.replay_idx = 0
with nav2:
    if st.button("◀ Step back") and st.session_state.replay_idx > 0:
        st.session_state.replay_idx -= 1
with nav3:
    if st.button("Step forward ▶") and st.session_state.replay_idx < len(window_df) - 1:
        st.session_state.replay_idx += 1
with nav4:
    autoplay = st.button("▶▶ Autoplay to end")

slider_idx = st.slider(
    "Scrub", min_value=0, max_value=max(len(window_df) - 1, 0),
    value=st.session_state.replay_idx, key="scrub_slider",
)
st.session_state.replay_idx = slider_idx

placeholder = st.empty()


def render_day(idx):
    row = window_df.iloc[idx]
    with placeholder.container():
        st.markdown(f"**{ticker}** &nbsp;·&nbsp; `{row['Date'].date()}` &nbsp;·&nbsp; day {idx+1} of {len(window_df)}")
        pc1, pc2 = st.columns([1, 1])
        with pc1:
            st.metric("Close", f"${row['Close']:.2f}", f"{row['price_move_pct']*100:+.2f}% next day")
        with pc2:
            st.metric("Next-day close", f"${row['Target_Next_Day_Close']:.2f}")

        st.markdown(gauge_svg(row["pred_prob"]), unsafe_allow_html=True)

        badge, cls, sub = verdict_for_row(row)
        st.markdown(
            f'<div class="verdict-box {cls}"><span class="badge {cls.replace("verdict-","badge-")}">{badge}</span>'
            f'<span style="color:#9FB3AC; font-size:13.5px;">{sub}</span></div>',
            unsafe_allow_html=True,
        )

        fcols = st.columns(3)
        feats = [
            ("Sentiment Z-score", f"{row['sent_zscore']:.2f}"),
            ("Sentiment 3D avg", f"{row['sent_ma_3']:.3f}"),
            ("Return (last obs.)", f"{row['returns']*100:.2f}%"),
            ("Volatility (7D)", f"{row['volatility']*100:.2f}%"),
            ("RSI (14D)", f"{row['rsi']:.1f}"),
            ("Gap since prior obs.", f"{int(row['gap_days'])}d"),
        ]
        for i, (label, val) in enumerate(feats):
            with fcols[i % 3]:
                st.markdown(
                    f'<div class="feature-cell"><div class="flabel">{label}</div>'
                    f'<div class="fvalue">{val}</div></div>',
                    unsafe_allow_html=True,
                )

        st.info(generate_explanation(row))


if autoplay:
    start_at = st.session_state.replay_idx
    for i in range(start_at, len(window_df)):
        st.session_state.replay_idx = i
        render_day(i)
        time.sleep(speed_map[speed])
else:
    render_day(st.session_state.replay_idx)
