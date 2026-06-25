"""
Educational, historical-only trade simulation. Walks chronologically through
a sequence of model signals and tracks a single virtual capital balance.

Rules (explicit, since this is "production-quality" code others will read):
  - Only TEST-split rows are eligible (out-of-sample, never seen in training).
  - BUY signal (pred_label == 1): go long for one day, capital compounds by
    the stock's actual next-day return. This is the only way capital is put
    at risk.
  - SELL signal (pred_label == 0): no short position is taken (long-only,
    matching the original "sell / stay cash" spec) -- capital is simply held
    flat for that step.
  - NO TRADE (pred_label == -1, the abstain/hold zone) is excluded entirely;
    it's not a trading decision.
  - No transaction fees, slippage, or position sizing beyond "all capital,
    one trade at a time" are modeled. This is a simplification appropriate
    for an educational demo, not a realistic backtest of tradeable returns.
"""
import numpy as np
import pandas as pd


def run_simulation(df, starting_capital=10000.0):
    """
    df: dataframe filtered to the rows you want to walk through
        (e.g. split == 'test', within a date range), with columns
        Date, Stock Name, pred_label, Target_Trend, price_move_pct.
    Returns a dict with the equity curve and summary stats.
    """
    trades = df[df["pred_label"] != -1].sort_values("Date").reset_index(drop=True)

    capital = starting_capital
    equity_curve = [starting_capital]
    dates = [trades["Date"].min() if len(trades) else None]
    log = []

    for _, row in trades.iterrows():
        if row["pred_label"] == 1:  # BUY -- capital at risk
            capital = capital * (1 + row["price_move_pct"])
            traded = True
        else:  # SELL / DOWN signal -- stay in cash
            traded = False
        equity_curve.append(capital)
        dates.append(row["Date"])
        log.append({
            "Date": row["Date"],
            "Ticker": row["Stock Name"],
            "Signal": "BUY" if row["pred_label"] == 1 else "SELL (cash)",
            "Actual Move %": row["price_move_pct"] * 100,
            "Correct": bool(row["pred_label"] == row["Target_Trend"]),
            "Equity After": capital,
        })

    equity_series = pd.Series(equity_curve)
    running_max = equity_series.cummax()
    drawdown = (equity_series - running_max) / running_max
    max_drawdown = drawdown.min() * 100 if len(drawdown) else 0.0

    buy_trades = trades[trades["pred_label"] == 1]
    win_rate = float((buy_trades["pred_label"] == buy_trades["Target_Trend"]).mean() * 100) if len(buy_trades) else 0.0

    total_return = (capital / starting_capital - 1) * 100

    return {
        "equity_curve": equity_curve,
        "dates": dates,
        "log": pd.DataFrame(log),
        "final_capital": capital,
        "total_return_pct": total_return,
        "win_rate_pct": win_rate,
        "num_buy_trades": int(len(buy_trades)),
        "num_sell_skips": int(len(trades) - len(buy_trades)),
        "max_drawdown_pct": max_drawdown,
        "directional_accuracy_pct": float((trades["pred_label"] == trades["Target_Trend"]).mean() * 100) if len(trades) else 0.0,
    }
