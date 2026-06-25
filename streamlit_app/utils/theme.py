"""
Shared visual identity for the app — same palette and type system as the
companion historical-replay HTML demo, so the Streamlit dashboard, the
static demo, and the README all read as one coherent project rather than
three disconnected artifacts.
"""
import streamlit as st

INK = "#0E1F1B"
INK_SOFT = "#16302A"
INK_LINE = "#2B453E"
PAPER = "#EDEFE7"
CARD = "#16302A"
TEAL = "#1F7A6C"
TEAL_BRIGHT = "#3FB39E"
TEAL_SOFT = "#DCEDE8"
CLAY = "#B5502D"
CLAY_BRIGHT = "#E07A4F"
CLAY_SOFT = "#F3E1D7"
AMBER = "#9C7B12"
AMBER_BRIGHT = "#E0B83A"
AMBER_SOFT = "#F4EBCC"
TEXT_PAPER = "#EDEFE7"
TEXT_DIM = "#9FB3AC"

FEATURE_LABELS = {
    "sent_zscore": "Sentiment Z-score",
    "sent_ma_3": "Sentiment 3D avg",
    "returns": "Returns",
    "volatility": "Volatility (7D)",
    "rsi": "RSI (14D)",
    "gap_days": "Gap since prior obs.",
}

FEATURE_DESCRIPTIONS = {
    "sent_zscore": "How far today's tweet sentiment sits from that stock's own trailing 14-day sentiment baseline.",
    "sent_ma_3": "3-day rolling average of BERT-scored tweet sentiment — short-term sentiment momentum.",
    "returns": "Most recent daily price return (% change in Close vs. the prior available observation).",
    "volatility": "7-calendar-day rolling standard deviation of daily returns — how choppy the stock has been.",
    "rsi": "14-calendar-day Relative Strength Index — classic momentum indicator; <30 oversold, >70 overbought.",
    "gap_days": "Calendar days since this ticker's previous observation — flags stale or sparse data.",
}


def inject_base_css():
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Serif:ital,wght@0,400;0,500;0,600;1,400&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

        html, body, [class*="css"] {{
            font-family: 'IBM Plex Sans', sans-serif;
        }}
        h1, h2, h3 {{
            font-family: 'IBM Plex Serif', serif !important;
            font-weight: 500 !important;
        }}
        .stApp {{
            background-color: {INK};
            color: {TEXT_PAPER};
        }}
        section[data-testid="stSidebar"] {{
            background-color: {INK_SOFT};
            border-right: 1px solid {INK_LINE};
        }}
        .mono {{ font-family: 'IBM Plex Mono', monospace; }}

        .metric-card {{
            background: {INK_SOFT};
            border: 1px solid {INK_LINE};
            border-radius: 10px;
            padding: 16px 18px;
        }}
        .metric-card .num {{
            font-family: 'IBM Plex Mono', monospace;
            font-size: 26px;
            font-weight: 600;
            color: {TEXT_PAPER};
        }}
        .metric-card .label {{
            font-family: 'IBM Plex Mono', monospace;
            font-size: 11px;
            letter-spacing: .04em;
            text-transform: uppercase;
            color: {TEXT_DIM};
            margin-top: 4px;
        }}

        .badge {{
            display: inline-block;
            font-family: 'IBM Plex Mono', monospace;
            font-size: 12.5px;
            font-weight: 600;
            letter-spacing: .04em;
            padding: 5px 10px;
            border-radius: 5px;
        }}
        .badge-teal {{ background: {TEAL_BRIGHT}; color: {INK}; }}
        .badge-clay {{ background: {CLAY_BRIGHT}; color: {INK}; }}
        .badge-amber {{ background: {AMBER_BRIGHT}; color: {INK}; }}

        .verdict-box {{
            border-radius: 9px;
            padding: 14px 16px;
            border: 1px solid;
            display: flex;
            align-items: center;
            gap: 12px;
            margin: 10px 0 18px;
        }}
        .verdict-teal {{ background: rgba(63,179,158,.12); border-color: {TEAL_BRIGHT}; }}
        .verdict-clay {{ background: rgba(224,122,79,.12); border-color: {CLAY_BRIGHT}; }}
        .verdict-amber {{ background: rgba(224,184,58,.12); border-color: {AMBER_BRIGHT}; }}

        .disclaimer {{
            font-family: 'IBM Plex Mono', monospace;
            font-size: 12px;
            color: {AMBER_BRIGHT};
            border: 1px dashed {AMBER_BRIGHT};
            border-radius: 7px;
            padding: 8px 12px;
            margin-bottom: 14px;
        }}

        .feature-cell {{
            background: {INK_SOFT};
            border: 1px solid {INK_LINE};
            border-radius: 8px;
            padding: 12px 14px;
            margin-bottom: 8px;
        }}
        .feature-cell .flabel {{
            font-family: 'IBM Plex Mono', monospace;
            font-size: 11px;
            color: {TEXT_DIM};
            text-transform: uppercase;
            letter-spacing: .04em;
        }}
        .feature-cell .fvalue {{
            font-family: 'IBM Plex Mono', monospace;
            font-size: 16px;
            color: {TEXT_PAPER};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def metric_card_html(num, label):
    return f'<div class="metric-card"><div class="num">{num}</div><div class="label">{label}</div></div>'


def disclaimer_banner(text="HISTORICAL SIMULATION ONLY — NOT LIVE TRADING, NOT INVESTMENT ADVICE"):
    return f'<div class="disclaimer">⚠ {text}</div>'
