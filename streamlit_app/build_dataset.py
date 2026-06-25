"""
Rebuilds the fixed feature pipeline (same logic validated earlier), runs the
real trained model, computes SHAP values, and saves one clean dataset that
the Streamlit app loads directly (no live recomputation needed at app runtime).
"""
import os
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix)
import json

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'raw', 'final_prediction_dataset.csv')
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'final_stock_model.json')
OUT_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'historical_replay_dataset.csv')
OUT_META = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'model_meta.json')

df = pd.read_csv(DATA_PATH)

def parse_dates_robust(series):
    for fmt in ('%Y-%m-%d', '%d-%m-%Y', '%m/%d/%Y', '%d/%m/%Y'):
        try:
            return pd.to_datetime(series, format=fmt)
        except (ValueError, TypeError):
            continue
    return pd.to_datetime(series, format='mixed', dayfirst=True)

df['Date'] = parse_dates_robust(df['Date'])
df = df.sort_values(['Stock Name', 'Date']).reset_index(drop=True)
n_loaded, n_tickers_loaded = len(df), df['Stock Name'].nunique()

OHLCV_COLS = ['Open', 'High', 'Low', 'Close', 'Volume']
prev_ohlcv = df.groupby('Stock Name')[OHLCV_COLS].shift(1)
is_stale = (df[OHLCV_COLS] == prev_ohlcv).all(axis=1)
n_stale = int(is_stale.sum())
df = df[~is_stale].copy().reset_index(drop=True)

MIN_ROWS_PER_TICKER = 20
ticker_counts = df['Stock Name'].value_counts()
valid_tickers = ticker_counts[ticker_counts >= MIN_ROWS_PER_TICKER].index
n_tickers_before_minhist = ticker_counts.size
df = df[df['Stock Name'].isin(valid_tickers)].copy().reset_index(drop=True)
n_tickers_after_minhist = df['Stock Name'].nunique()

df['gap_days'] = df.groupby('Stock Name')['Date'].diff().dt.days.fillna(0)
df['returns'] = df.groupby('Stock Name')['Close'].pct_change()
df['volatility'] = df.groupby('Stock Name').rolling('7D', on='Date', min_periods=2)['returns'].std().values

def add_calendar_rsi(frame, period_days=14):
    delta = frame.groupby('Stock Name')['Close'].diff()
    frame['_gain'] = delta.where(delta > 0, 0.0)
    frame['_loss'] = (-delta.where(delta < 0, 0.0))
    avg_gain = frame.groupby('Stock Name').rolling(f'{period_days}D', on='Date', min_periods=2)['_gain'].mean().values
    avg_loss = frame.groupby('Stock Name').rolling(f'{period_days}D', on='Date', min_periods=2)['_loss'].mean().values
    frame['rsi'] = 100 - (100 / (1 + (avg_gain / (avg_loss + 1e-9))))
    frame.drop(columns=['_gain', '_loss'], inplace=True)
    return frame

df = add_calendar_rsi(df, 14)
df['sent_ma_14'] = df.groupby('Stock Name').rolling('14D', on='Date', min_periods=2)['sentiment_score'].mean().values
df['sent_std_14'] = df.groupby('Stock Name').rolling('14D', on='Date', min_periods=2)['sentiment_score'].std().values
df['sent_zscore'] = (df['sentiment_score'] - df['sent_ma_14']) / (df['sent_std_14'] + 1e-9)
df['sent_ma_3'] = df.groupby('Stock Name').rolling('3D', on='Date', min_periods=1)['sentiment_score'].mean().values

df['price_move_pct'] = (df['Target_Next_Day_Close'] - df['Close']) / df['Close']
df = df[df['price_move_pct'].abs() > 0.005].copy()
df['Target_Trend'] = (df['price_move_pct'] > 0).astype(int)
df.dropna(inplace=True)
df = df.sort_values(['Stock Name', 'Date']).reset_index(drop=True)

def per_ticker_time_split(frame, test_frac=0.15):
    train_idx, test_idx = [], []
    for _, g in frame.groupby('Stock Name', sort=False):
        n = len(g)
        cut = max(1, int(np.floor(n * (1 - test_frac))))
        cut = min(cut, n - 1) if n > 1 else n
        train_idx.append(g.index[:cut])
        test_idx.append(g.index[cut:])
    return np.concatenate(train_idx), np.concatenate(test_idx)

train_idx, test_idx = per_ticker_time_split(df, 0.15)
df['split'] = 'train'
df.loc[test_idx, 'split'] = 'test'

FEATURES = ['sent_zscore', 'sent_ma_3', 'returns', 'volatility', 'rsi', 'gap_days']

model = xgb.Booster()
model.load_model(MODEL_PATH)
dall = xgb.DMatrix(df[FEATURES], feature_names=FEATURES)
df['pred_prob'] = model.predict(dall)
df['pred_label'] = np.where(df['pred_prob'] > 0.60, 1, np.where(df['pred_prob'] < 0.48, 0, -1))

print("Computing SHAP values (TreeExplainer on the real trained booster)...")
# Important: the saved model carries a 'best_iteration' attribute from early
# stopping (iteration 38 of 139), but Booster.predict() ignores it by default
# and uses the FULL 139-tree model (verified: that's what produces pred_prob
# above, matching every previously-reported metric). shap.TreeExplainer,
# however, DOES respect that attribute by default and would silently explain
# only the 39-tree checkpoint -- a different model than the one actually
# making predictions. Stripping the attribute first keeps SHAP and the
# deployed model's predictions consistent with each other.
model.set_attr(best_iteration=None, best_score=None)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(df[FEATURES])
for i, feat in enumerate(FEATURES):
    df[f'shap_{feat}'] = shap_values[:, i]
df['shap_base_value'] = explainer.expected_value

# ---- Metrics on TEST split ----
test_df = df[df['split'] == 'test'].copy()
mask = test_df['pred_label'] != -1

acc = accuracy_score(test_df.loc[mask, 'Target_Trend'], test_df.loc[mask, 'pred_label'])
prec = precision_score(test_df.loc[mask, 'Target_Trend'], test_df.loc[mask, 'pred_label'])
rec = recall_score(test_df.loc[mask, 'Target_Trend'], test_df.loc[mask, 'pred_label'])
f1 = f1_score(test_df.loc[mask, 'Target_Trend'], test_df.loc[mask, 'pred_label'])
auc_all = roc_auc_score(test_df['Target_Trend'], test_df['pred_prob'])
cm = confusion_matrix(test_df.loc[mask, 'Target_Trend'], test_df.loc[mask, 'pred_label'])
importance = model.get_score(importance_type='gain')

meta = {
    "n_loaded": n_loaded, "n_tickers_loaded": n_tickers_loaded,
    "n_stale_removed": n_stale,
    "n_tickers_before_minhist": int(n_tickers_before_minhist),
    "n_tickers_after_minhist": int(n_tickers_after_minhist),
    "n_train": int(len(train_idx)), "n_test": int(len(test_idx)),
    "n_tickers_test": int(test_df['Stock Name'].nunique()),
    "n_signals": int(mask.sum()), "coverage_pct": round(float(mask.mean()*100), 2),
    "accuracy": round(float(acc), 4), "precision": round(float(prec), 4),
    "recall": round(float(rec), 4), "f1": round(float(f1), 4),
    "roc_auc": round(float(auc_all), 4),
    "cm": cm.tolist(),
    "importance": {k: round(float(v), 4) for k, v in importance.items()},
    "date_min": str(test_df['Date'].min().date()), "date_max": str(test_df['Date'].max().date()),
    "dataset_date_min": str(df['Date'].min().date()), "dataset_date_max": str(df['Date'].max().date()),
    "features": FEATURES,
    "shap_base_value": float(explainer.expected_value),
}

with open(OUT_META, 'w') as f:
    json.dump(meta, f, indent=2)

keep_cols = ['Stock Name', 'Date', 'Close', 'Target_Next_Day_Close', 'price_move_pct',
             'Target_Trend', 'split', 'pred_prob', 'pred_label'] + FEATURES + [f'shap_{f}' for f in FEATURES]
df[keep_cols].to_csv(OUT_DATA, index=False)

print("\nSaved dataset:", OUT_DATA, "rows:", len(df))
print("Saved meta:", OUT_META)
print(json.dumps(meta, indent=2))
