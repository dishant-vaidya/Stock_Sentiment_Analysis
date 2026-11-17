# YES Bank Stock Forecasting & News Sentiment System

A Streamlit dashboard that forecasts the price of **YES Bank (`YESBANK.NS`)** using two time-series approaches (Auto-ARIMA and Prophet), evaluates them honestly on unseen data, and adds a **FinBERT-based news sentiment layer** that can optionally nudge the forecast.

> **Disclaimer:** This is an educational/portfolio project. It is not financial advice and must not be used for real trading decisions.

---

## Table of Contents

1. [Features](#features)
2. [How it works](#how-it-works)
3. [Project structure](#project-structure)
4. [Installation](#installation)
5. [Configuring the NewsAPI key](#configuring-the-newsapi-key)
6. [Running the app](#running-the-app)
7. [Using the dashboard](#using-the-dashboard)
8. [Methodology in detail](#methodology-in-detail)
9. [What was fixed from the original version](#what-was-fixed-from-the-original-version)
10. [Limitations](#limitations)
11. [Troubleshooting](#troubleshooting)
12. [Future improvements](#future-improvements)
13. [Tech stack](#tech-stack)

---

## Features

- **Live data:** 5 years of daily prices from Yahoo Finance via `yfinance`.
- **Two forecasting models:**
  - **Auto-ARIMA** on log-returns (order chosen automatically).
  - **Prophet** on closing prices (weekly + yearly seasonality).
- **Honest evaluation:** true out-of-sample hold-out on the last N trading days, with RMSE, MAE, MAPE, directional accuracy, and a **naive baseline** for comparison.
- **True future forecast:** models are refit on all data and forecast the next N business days with correct future dates.
- **News sentiment:** recent headlines (title + description) from NewsAPI classified by **FinBERT** (`ProsusAI/finbert`) into Positive / Neutral / Negative with probabilities.
- **Sentiment-adjusted forecast:** an adjustable heuristic that adds a small, decaying, volatility-scaled drift to the ARIMA forecast based on aggregate sentiment.
- **Interactive sidebar:** change ticker, news query, horizon, number of articles, and sentiment strength.
- **Secure key handling:** NewsAPI key from `st.secrets`, an environment variable, or the sidebar (never hardcoded).

---

## How it works

```
Yahoo Finance ──► Close price ──► Log returns ──┬──► Auto-ARIMA ──┐
                                                │                 ├──► Hold-out metrics (RMSE, MAE, MAPE, direction)
                                                └──► Prophet ─────┤
                                                                  └──► Future forecast (next N business days)
                                                                                   ▲
NewsAPI ──► Headlines ──► FinBERT ──► Aggregate sentiment score ──► drift overlay ─┘ (ARIMA only, optional)
```

---

## Project structure

```
.
├── StockSentimentAnalysis.py   # Streamlit app (all logic)
├── requirements.txt            # Python dependencies
└── README.md                   # This file

```

---

## Installation

### Prerequisites

- **Python 3.9 - 3.11** recommended (some of `pmdarima` / `prophet` wheels lag behind the newest Python versions).
- ~2 GB free disk space (PyTorch + FinBERT weights).
- Internet access (Yahoo Finance, NewsAPI, Hugging Face model download on first run).

### Steps

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

The first run downloads the FinBERT model (~440 MB) from Hugging Face and caches it locally.

---

## Configuring the NewsAPI key

Get a free key at [https://newsapi.org](https://newsapi.org). The free plan has request limits and returns limited/delayed articles, which is fine for this project.

Provide the key using **any one** of these (checked in this order):

**Option 1: Streamlit secrets (recommended)**

Create `.streamlit/secrets.toml`:

```toml
NEWS_API_KEY = "your_key_here"
```

**Option 2: Environment variable**

```bash
export NEWS_API_KEY="your_key_here"      # Windows PowerShell: $env:NEWS_API_KEY="your_key_here"
```

**Option 3: Sidebar field**

Paste it into the "NewsAPI key" box in the app sidebar.

> Add `.streamlit/secrets.toml` to `.gitignore` so your key is never pushed to GitHub.

If no key is supplied, the price forecasts still work; only the sentiment section is skipped.

---

## Running the app

```bash
streamlit run StockSentimentAnalysis.py
```

Open the local URL shown in the terminal (usually `http://localhost:8501`). The first load is slow (model fitting + FinBERT download); later runs are faster thanks to caching.

---

## Using the dashboard

**Sidebar controls**

| Control | Purpose |
|---|---|
| Ticker | Any Yahoo Finance symbol (default `YESBANK.NS`) |
| News search query | Keywords sent to NewsAPI (default `YES Bank`) |
| Forecast horizon | Number of trading days to evaluate and forecast (10-60) |
| Number of news articles | How many recent articles to score (5-30) |
| Sentiment influence | 0 disables the sentiment overlay; higher values push the forecast more |
| NewsAPI key | Only shown if no key is found in secrets/env |

**Dashboard sections**

1. **News Sentiment:** aggregate score (-1 to +1), counts per class, and a table of each article with class probabilities.
2. **Model Comparison:** metrics table (Auto-ARIMA vs Prophet vs Naive) and a hold-out chart of forecast vs actual.
3. **Future Forecast:** chart of the next N business days, with Prophet's uncertainty band and (optionally) the sentiment-adjusted ARIMA path, plus a summary table.

**Reading the metrics**

- **RMSE / MAE:** average error in rupees (lower is better). RMSE penalises large misses more.
- **MAPE %:** average percentage error, comparable across price levels.
- **Direction acc. %:** how often the predicted day-to-day move had the right sign (around 50% = coin flip).
- **Naive baseline:** if a model does not beat this, it adds no real predictive value.

---

## Methodology in detail

### 1. Data preparation
- Downloads 5 years of adjusted daily prices.
- Flattens `yfinance`'s MultiIndex columns (newer versions) and strips timezone info.
- Computes log returns: `r_t = ln(P_t / P_{t-1})`, which are closer to stationary than raw prices.

### 2. Hold-out evaluation (no look-ahead)
- The last *N* trading days are held out as a test set.
- Both models are trained **only** on the data before that window.
- ARIMA's return forecasts are converted to prices by compounding from the **last training price** (not the last price in the dataset).
- Prophet is predicted on exactly the test dates.
- Metrics are computed against the real prices of the test window.

### 3. Future forecast
- ARIMA is refit with the selected order on **all** data; Prophet is refit on all data.
- Future dates are the next *N* **business days** (weekends skipped), so the plot extends past the last real date.

### 4. Sentiment with FinBERT
- Text = headline + description, truncated to 256 tokens.
- The label mapping is read from the model's own `id2label` config (FinBERT's order is *positive, negative, neutral*).
- Per-article score = `P(positive) - P(negative)`; the aggregate is the mean over articles.

### 5. Sentiment-adjusted forecast (heuristic)
```
drift_t = alpha x sentiment_score x recent_daily_volatility x exp(-t / 10)
```
- Scaled by the last 60 days' volatility so the effect stays realistic.
- Decays over time because the price impact of news fades.
- This is an **experimental overlay, not a trained or back-tested model**. Set the slider to 0 to ignore it.

---

## What was fixed from the original version

| # | Issue in original | Fix |
|---|---|---|
| 1 | ARIMA forecast was anchored to the last close, then compared with the same last 30 days (invalid RMSE) | Anchored to the last **training** price and compared with the true hold-out period |
| 2 | Prophet was trained on all data, so its RMSE was in-sample | Prophet is trained on the training split only for evaluation |
| 3 | Forecast was plotted over existing dates, not the future | Separate hold-out plot and a real future plot with future dates |
| 4 | Prophet added 30 calendar days | Uses business days; horizon is configurable |
| 5 | **FinBERT labels were mapped in the wrong order** (`Negative, Neutral, Positive`) | Labels read from `model.config.id2label` |
| 6 | Sentiment was disconnected from the forecasts | Aggregate sentiment feeds an optional drift overlay |
| 7 | No baseline to judge model quality | Naive baseline + MAE, MAPE, directional accuracy added |
| 8 | `yfinance` MultiIndex columns could break `['Close']` | Columns flattened; timezone removed |
| 9 | Hardcoded API key | `st.secrets` / env var / sidebar input |
| 10 | No error handling | Handles empty data, missing key, and API failures |
| 11 | Sentiment on title only, no inference hygiene | Title + description, `model.eval()`, `torch.no_grad()` |
| 12 | Heavy work re-run on every interaction | `st.cache_data` / `st.cache_resource` with TTLs |

---

## Limitations

- Stock prices are close to a random walk; neither ARIMA nor Prophet should be expected to beat the naive baseline consistently, and the app shows this openly.
- ARIMA on log-returns forecasts returns near zero, so its price path is almost flat.
- Prophet is designed for series with strong seasonality/trend and can extrapolate trends aggressively on volatile equities.
- The sentiment overlay is heuristic and unvalidated; NewsAPI's free tier gives few, sometimes irrelevant articles.
- NSE holidays are not modelled in future dates.
- Single-stock, single-split evaluation. A rolling-origin back-test would be more robust.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `pmdarima` / `numpy` binary incompatibility error | Use `numpy<2` (as pinned) and Python 3.9-3.11 |
| `prophet` fails to install or `cmdstan` errors | Upgrade pip, try `pip install prophet --upgrade`, or use conda: `conda install -c conda-forge prophet` |
| "No price data found" | Check the ticker (NSE stocks need `.NS`), or retry; Yahoo may rate-limit |
| NewsAPI `apiKeyInvalid` / `rateLimited` | Verify the key; free tier has daily request limits |
| First run very slow | FinBERT is downloading and models are fitting; subsequent runs are cached |
| Out-of-memory with torch | Reduce number of articles, or install the CPU-only build of PyTorch |

---

## Future improvements

- Rolling-origin (walk-forward) cross-validation across multiple windows.
- Train a model that uses sentiment as a real feature (e.g. Prophet regressor, LSTM, or gradient boosting) instead of a heuristic drift.
- Historical news/sentiment time series aligned to dates for back-testing the sentiment effect.
- Add GARCH for volatility forecasting and proper prediction intervals for ARIMA.
- NSE holiday calendar, additional tickers, and a model-comparison dashboard across stocks.
- Dockerfile and CI for reproducible deployment.

---

## Tech stack

Python, Streamlit, yfinance, pandas, NumPy, pmdarima (Auto-ARIMA), Prophet, PyTorch, Hugging Face Transformers (FinBERT), NewsAPI, Matplotlib, scikit-learn.
