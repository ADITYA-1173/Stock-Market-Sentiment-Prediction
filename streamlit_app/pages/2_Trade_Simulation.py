import sys
import os

import streamlit as st
import plotly.graph_objects as go

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.theme import inject_base_css, disclaimer_banner, metric_card_html, TEAL_BRIGHT, CLAY_BRIGHT
from utils.data_loader import load_dataset
from utils.trade_sim import run_simulation

st.set_page_config(page_title="Trade Simulation", page_icon="💰", layout="wide")
inject_base_css()

df = load_dataset()
test_df = df[df["split"] == "test"].copy()

st.title("Trade Simulation")
st.markdown(disclaimer_banner("EDUCATIONAL HISTORICAL SIMULATION ONLY — NOT INVESTMENT ADVICE"), unsafe_allow_html=True)
st.markdown(
    """
    A simple paper-trading walk-through of the model's actual out-of-sample signals. **Rules:** start
    with \\$10,000 virtual capital; on a BUY signal, go long for one day and let capital compound by the
    stock's real next-day return; on a SELL signal, stay in cash (long-only — no shorting); the model's
    NO TRADE / hold-zone days are excluded since no decision was made. No fees, slippage, or position
    sizing beyond "all capital, one trade at a time" are modeled — this is a simplification for an
    educational demo, not a realistic tradeable backtest.
    """
)

min_date, max_date = test_df["Date"].min().date(), test_df["Date"].max().date()
all_tickers = sorted(test_df["Stock Name"].unique().tolist())

c1, c2, c3 = st.columns([1, 1, 1])
with c1:
    date_range = st.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
with c2:
    ticker_mode = st.radio("Ticker scope", ["All tickers", "Choose tickers"], horizontal=True)
with c3:
    starting_capital = st.number_input("Starting capital ($)", value=10000, step=1000, min_value=1000)

if ticker_mode == "Choose tickers":
    chosen = st.multiselect("Tickers", all_tickers, default=all_tickers[:10])
else:
    chosen = all_tickers

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_d, end_d = date_range
else:
    start_d, end_d = min_date, max_date

sim_df = test_df[
    (test_df["Date"].dt.date >= start_d)
    & (test_df["Date"].dt.date <= end_d)
    & (test_df["Stock Name"].isin(chosen))
]

if len(sim_df) == 0 or len(sim_df[sim_df["pred_label"] != -1]) == 0:
    st.warning("No signals in this selection. Widen the date range or ticker scope.")
    st.stop()

result = run_simulation(sim_df, starting_capital=starting_capital)

st.markdown("### Results")
cols = st.columns(5)
return_color = "▲" if result["total_return_pct"] >= 0 else "▼"
m = [
    (f"${result['final_capital']:,.0f}", "Final capital"),
    (f"{return_color} {result['total_return_pct']:+.1f}%", "Total return"),
    (f"{result['win_rate_pct']:.1f}%", "Win rate (BUY trades)"),
    (result["num_buy_trades"], "BUY trades taken"),
    (f"{result['max_drawdown_pct']:.1f}%", "Max drawdown"),
]
for col, (num, label) in zip(cols, m):
    col.markdown(metric_card_html(num, label), unsafe_allow_html=True)

st.caption(
    f"{result['num_sell_skips']} additional SELL signals stayed in cash (no capital at risk) · "
    f"directional accuracy across all {result['num_buy_trades'] + result['num_sell_skips']} signals: "
    f"{result['directional_accuracy_pct']:.1f}%"
)
st.markdown(
    "<span style='color:#9FB3AC; font-size:12.5px;'>⚠ The return above reflects "
    "<b style='color:#EDEFE7;'>all capital reinvested into every single trade sequentially</b> with no "
    "diversification — a deliberately unrealistic sizing assumption that amplifies both gains and the "
    "drawdown shown below. It demonstrates the model's directional edge, not a realistic risk-adjusted "
    "return.</span>",
    unsafe_allow_html=True,
)

st.markdown("### Equity curve")
fig = go.Figure()
fig.add_trace(go.Scatter(
    y=result["equity_curve"], mode="lines",
    line=dict(color=TEAL_BRIGHT, width=2),
    fill="tozeroy", fillcolor="rgba(63,179,158,0.08)",
    name="Equity",
))
fig.add_hline(y=starting_capital, line_dash="dot", line_color="#9FB3AC", annotation_text="Starting capital")
fig.update_layout(
    template="plotly_dark", paper_bgcolor="#0E1F1B", plot_bgcolor="#0E1F1B",
    font=dict(family="IBM Plex Mono, monospace", color="#EDEFE7"),
    xaxis_title="Trade #", yaxis_title="Equity ($)",
    height=380, margin=dict(l=10, r=10, t=10, b=10),
)
st.plotly_chart(fig, width='stretch')

st.markdown("### Trade log")
log = result["log"].copy()
log["Date"] = log["Date"].dt.date
log["Equity After"] = log["Equity After"].round(2)
log["Actual Move %"] = log["Actual Move %"].round(2)
st.dataframe(log, width='stretch', height=320)
