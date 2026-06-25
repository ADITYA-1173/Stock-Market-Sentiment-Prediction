# Stock Market Prediction Using Twitter Sentiment Analysis

A complete ML pipeline predicting next-day stock price direction using fine-tuned FinBERT sentiment scoring and XGBoost classification — evaluated honestly, with five data-leakage bugs found and fixed before reporting any result.

## Live Demo (Streamlit)

https://stock-market-sentiment-prediction-8bjulbx8eweqpzpakuffmu.streamlit.app/

## Architecture

Historical Tweets (3M+)
↓
Fine-tuned BERT
↓
Daily Sentiment Score per Stock
↓
Yahoo Finance OHLCV
↓
Feature Engineering (RSI, Volatility, Returns, Z-score, Gap)
↓
XGBoost Classifier
↓
SHAP Explainability + Streamlit Dashboard

## Results

| Metric | Value |
|---|---|
| BERT sentiment accuracy (Financial PhraseBank) | ~93% |
| XGBoost directional accuracy (high-confidence signals) | 58% |
| Signal coverage | 6% of test days |
| ROC AUC (all test rows) | 0.52 |
| Stocks covered | 647 tickers |

## Five Evaluation Bugs Found and Fixed

1. **Train/test split**: Alphabetical sort caused 476 of 477 test tickers to be completely absent from training → fixed with per-ticker time-based split.
2. **Rolling windows**: `.rolling(7)` counted rows, not days → fixed with `.rolling('7D', on='Date')` (calendar-time aware).
3. **Stale rows**: 15,485 forward-filled weekend/holiday rows removed before feature computation.
4. **Sparse tickers**: 2,999 of 3,646 tickers had insufficient history and were silently dropped via `dropna()` → now explicitly filtered and logged.
5. **SHAP/model mismatch**: SHAP `TreeExplainer` was explaining a 39-tree checkpoint while real predictions used the full 139-tree model → fixed by clearing `best_iteration` attribute before explaining.

## Project Structure

```text
├── my-project.ipynb                 # Full training pipeline
├── final_prediction_dataset.csv     # Merged OHLCV + sentiment (after pipeline)
├── master_sentiment_features.csv    # Daily sentiment scores per ticker
├── Sentences_75Agree.txt            # Financial PhraseBank (BERT fine-tuning)
├── Sentences_AllAgree.txt           # Financial PhraseBank (BERT fine-tuning)
├── BERT_Finetuned_gpu/              # Tokenizer config (model weights on HuggingFace)
├── XGBoost Model/                   # Trained model + evaluation plots
├── streamlit_app/                   # 5-page interactive historical replay dashboard
└── requirements.txt
```

## Data Sources

- **Tweets**: [Stock Market Tweets (Kaggle)](https://www.kaggle.com/) — raw CSVs not included due to size (~500 MB); download directly from Kaggle.
- **BERT model weights**: Not included due to GitHub's 100 MB file limit (~420 MB). Download from: [HuggingFace Hub](https://huggingface.co/ELLIOT73/stock-sentiment-finbert)
- **Price data**: Yahoo Finance via yfinance API — fetched dynamically.

## How to Run

```bash
pip install -r requirements.txt
```

### To run the Streamlit dashboard:
```bash
cd streamlit_app
streamlit run app.py
```

### To retrain from scratch:
Open `my-project.ipynb` in Jupyter and run all cells.

## Tech Stack

Python, HuggingFace Transformers, FinBERT, XGBoost, SHAP, Streamlit, YFinance, Pandas, scikit-learn, Plotly, CUDA (Kaggle GPU)
