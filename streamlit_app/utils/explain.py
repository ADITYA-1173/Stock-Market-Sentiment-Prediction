"""
Turns a row's raw feature values + SHAP contributions into a short
natural-language explanation, plus a structured breakdown for charting.

SHAP values here are in margin (log-odds) space, which is the only space
in which they are mathematically additive for a binary:logistic XGBoost
model. We use their sign and rank for the narrative (which features pushed
toward UP vs DOWN, and by how much relative to each other) rather than
converting to probability units, since that conversion is not additive and
would overstate precision.
"""
from utils.data_loader import FEATURES, SHAP_COLS
from utils.theme import FEATURE_LABELS


def qualify_sent_zscore(v):
    if v > 0.5:
        return "well above its recent baseline (a bullish sentiment shift)"
    if v < -0.5:
        return "well below its recent baseline (a bearish sentiment shift)"
    return "close to its own recent baseline (a neutral sentiment shift)"


def qualify_sent_ma3(v):
    if v > 0.05:
        return "net positive over the past 3 days"
    if v < -0.05:
        return "net negative over the past 3 days"
    return "roughly neutral over the past 3 days"


def qualify_returns(v):
    if v > 0.01:
        return "the stock had just risen sharply"
    if v < -0.01:
        return "the stock had just fallen sharply"
    return "the stock had seen little recent movement"


def qualify_volatility(v):
    if v > 0.03:
        return "7-day volatility was elevated"
    if v < 0.01:
        return "7-day volatility was unusually calm"
    return "7-day volatility was fairly typical"


def qualify_rsi(v):
    if v < 30:
        return f"RSI was oversold ({v:.0f})"
    if v > 70:
        return f"RSI was overbought ({v:.0f})"
    return f"RSI showed neutral momentum ({v:.0f})"


def qualify_gap(v):
    if v > 7:
        return f"there was a {int(v)}-day gap in available data before this point (a stale-signal risk)"
    return "the data leading into this point was fresh, with no significant gap"


QUALIFIERS = {
    "sent_zscore": qualify_sent_zscore,
    "sent_ma_3": qualify_sent_ma3,
    "returns": qualify_returns,
    "volatility": qualify_volatility,
    "rsi": qualify_rsi,
    "gap_days": qualify_gap,
}


def feature_breakdown(row):
    """Returns a list of dicts, one per feature, sorted by |SHAP| descending."""
    items = []
    for feat in FEATURES:
        shap_val = row[f"shap_{feat}"]
        items.append({
            "feature": feat,
            "label": FEATURE_LABELS[feat],
            "value": row[feat],
            "shap": shap_val,
            "pushes_up": shap_val > 0,
            "qualifier": QUALIFIERS[feat](row[feat]),
        })
    items.sort(key=lambda d: abs(d["shap"]), reverse=True)
    return items


def generate_explanation(row, top_n=3):
    breakdown = feature_breakdown(row)
    lean = "bullish" if row["pred_prob"] >= 0.5 else "bearish"

    top = breakdown[:top_n]
    clauses = []
    for item in top:
        push = "toward UP" if item["pushes_up"] else "toward DOWN"
        clauses.append(f"{item['qualifier']}, pushing {push}")

    if len(clauses) == 1:
        body = clauses[0]
    elif len(clauses) == 2:
        body = f"{clauses[0]}; and {clauses[1]}"
    else:
        body = f"{'; '.join(clauses[:-1])}; and {clauses[-1]}"

    baseline_note = (
        " (the model's baseline historical up-rate across all stocks also factors into every call, "
        "which is why the top individual features don't always have to unanimously agree with the final lean)"
    )

    if row["pred_label"] == -1:
        opener = f"The model leaned {lean} overall (P(up) = {row['pred_prob']*100:.1f}%), but stayed flat since that's inside the hold zone. Strongest factors: "
    else:
        action = "a BUY" if row["pred_label"] == 1 else "a SELL"
        opener = f"The model fired {action} signal — leaning {lean} overall at P(up) = {row['pred_prob']*100:.1f}%. Strongest factors: "

    return opener + body + "." + (baseline_note if any(
        (i["pushes_up"] and lean == "bearish") or (not i["pushes_up"] and lean == "bullish") for i in top
    ) else "")
