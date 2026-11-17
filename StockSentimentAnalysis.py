

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
import yfinance as yf
from newsapi import NewsApiClient
from pmdarima import ARIMA, auto_arima
from prophet import Prophet
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(page_title="YES Bank Stock Intelligence System", layout="wide")
st.title("YES Bank Stock Forecasting & News Sentiment System")
st.caption("Educational project. Not financial advice.")

# ----------------------------------------------------------------------------
# SIDEBAR / CONFIG
# ----------------------------------------------------------------------------
st.sidebar.header("Settings")
STOCK_SYMBOL = st.sidebar.text_input("Ticker (Yahoo Finance)", "YESBANK.NS")
NEWS_QUERY = st.sidebar.text_input("News search query", "YES Bank")
HORIZON = st.sidebar.slider("Forecast horizon (trading days)", 10, 60, 30)
N_ARTICLES = st.sidebar.slider("Number of news articles", 5, 30, 15)
ALPHA = st.sidebar.slider(
    "Sentiment influence (0 = off)", 0.0, 1.0, 0.2, 0.05,
    help="Strength of the sentiment-driven drift added to the ARIMA forecast.",
)


def get_news_api_key() -> str:
    """Priority: st.secrets -> environment variable -> sidebar input."""
    try:
        key = st.secrets["NEWS_API_KEY"]
        if key:
            return key
    except Exception:
        pass
    key = os.getenv("NEWS_API_KEY", "")
    if key:
        return key
    return st.sidebar.text_input("NewsAPI key", type="password")


NEWS_API_KEY = get_news_api_key()


# ----------------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------------
def rmse(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.mean(np.abs(a - b)))


def mape(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.mean(np.abs((a - b) / a)) * 100)


def directional_accuracy(actual: pd.Series, pred: pd.Series, anchor: float) -> float:
    """% of days where predicted day-over-day direction matches the actual one."""
    a = np.diff(np.concatenate([[anchor], actual.values]))
    p = np.diff(np.concatenate([[anchor], pred.values]))
    return float(np.mean(np.sign(a) == np.sign(p)) * 100)


# ----------------------------------------------------------------------------
# DATA
# ----------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner="Downloading price data...")
def load_price_data(symbol: str) -> pd.DataFrame:
    raw = yf.download(symbol, period="5y", auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        return pd.DataFrame()

    # Newer yfinance returns MultiIndex columns like ("Close", "YESBANK.NS")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[["Close"]].copy()
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    df.index.name = "Date"
    df["LogReturn"] = np.log(df["Close"] / df["Close"].shift(1))
    df = df.dropna()
    return df


df = load_price_data(STOCK_SYMBOL)
if df.empty:
    st.error(f"No price data found for '{STOCK_SYMBOL}'. Check the ticker symbol.")
    st.stop()
if len(df) < HORIZON * 5:
    st.error("Not enough price history for the chosen horizon.")
    st.stop()

# ----------------------------------------------------------------------------
# FORECASTING FUNCTIONS
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Running hold-out evaluation (ARIMA + Prophet)...")
def run_holdout(data: pd.DataFrame, h: int):
    """
    Train on everything EXCEPT the last h trading days, forecast those h days,
    and compare with what actually happened (true out-of-sample evaluation).
    """
    train, test = data.iloc[:-h], data.iloc[-h:]
    anchor = float(train["Close"].iloc[-1])  # last price the models are allowed to see

    # --- ARIMA on log-returns, converted back to price from the anchor ---
    arima = auto_arima(
        train["LogReturn"], seasonal=False, stepwise=True,
        suppress_warnings=True, error_action="ignore",
    )
    ret_fc = np.asarray(arima.predict(n_periods=h))
    arima_pred = pd.Series(anchor * np.exp(np.cumsum(ret_fc)), index=test.index)

    # --- Prophet on prices, trained on train only ---
    p_train = train.reset_index()[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
    prophet = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
    prophet.fit(p_train)
    p_fc = prophet.predict(pd.DataFrame({"ds": test.index}))
    prophet_pred = pd.Series(p_fc["yhat"].values, index=test.index)

    # --- Naive baseline: tomorrow = today (hard to beat for stocks) ---
    naive_pred = pd.Series(anchor, index=test.index)

    actual = test["Close"]
    rows = []
    for name, pred in [("Auto-ARIMA", arima_pred), ("Prophet", prophet_pred), ("Naive (last price)", naive_pred)]:
        rows.append({
            "Model": name,
            "RMSE": rmse(actual, pred),
            "MAE": mae(actual, pred),
            "MAPE %": mape(actual, pred),
            "Direction acc. %": directional_accuracy(actual, pred, anchor),
        })

    return {
        "arima_pred": arima_pred,
        "prophet_pred": prophet_pred,
        "naive_pred": naive_pred,
        "metrics": pd.DataFrame(rows).set_index("Model"),
        "arima_order": arima.order,
    }


@st.cache_data(show_spinner="Refitting on full data and forecasting the future...")
def run_future(data: pd.DataFrame, h: int, arima_order: tuple):
    """Refit on ALL data and forecast the next h business days."""
    last_date = data.index[-1]
    future_dates = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=h)
    last_price = float(data["Close"].iloc[-1])

    arima = ARIMA(order=arima_order, suppress_warnings=True)
    arima.fit(data["LogReturn"])
    ret_fc = np.asarray(arima.predict(n_periods=h))

    p_all = data.reset_index()[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
    prophet = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
    prophet.fit(p_all)
    p_fc = prophet.predict(pd.DataFrame({"ds": future_dates}))

    return {
        "dates": future_dates,
        "last_price": last_price,
        "arima_returns": ret_fc,
        "prophet": p_fc[["ds", "yhat", "yhat_lower", "yhat_upper"]].set_index("ds"),
    }


# ----------------------------------------------------------------------------
# SENTIMENT (FinBERT)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading FinBERT...")
def load_finbert():
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    model.eval()
    return tokenizer, model


@st.cache_data(ttl=1800, show_spinner="Fetching news...")
def fetch_news(api_key: str, query: str, n: int):
    client = NewsApiClient(api_key=api_key)
    resp = client.get_everything(query, language="en", page_size=n, sort_by="publishedAt")
    return resp.get("articles", [])


def analyze_sentiment(articles) -> pd.DataFrame:
    tokenizer, model = load_finbert()
    # FinBERT's label order is NOT negative/neutral/positive: read it from the config.
    id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}

    rows = []
    for art in articles:
        title = (art.get("title") or "").strip()
        desc = (art.get("description") or "").strip()
        text = f"{title}. {desc}" if desc else title
        if not text:
            continue
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            probs = torch.softmax(model(**inputs).logits, dim=1)[0]
        p = {id2label[i]: float(probs[i]) for i in range(len(probs))}
        label = max(p, key=p.get)
        rows.append({
            "Published": (art.get("publishedAt") or "")[:10],
            "Source": (art.get("source") or {}).get("name", ""),
            "Headline": title,
            "Sentiment": label.capitalize(),
            "Positive": p.get("positive", 0.0),
            "Neutral": p.get("neutral", 0.0),
            "Negative": p.get("negative", 0.0),
            "Score": p.get("positive", 0.0) - p.get("negative", 0.0),  # range -1..+1
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# SECTION 1: SENTIMENT (run first so it can feed the forecast)
# ----------------------------------------------------------------------------
st.header("1. News Sentiment (FinBERT)")

sent_df = pd.DataFrame()
sentiment_score = 0.0

if not NEWS_API_KEY:
    st.warning(
        "No NewsAPI key found. Add `NEWS_API_KEY` to `.streamlit/secrets.toml`, "
        "set it as an environment variable, or paste it in the sidebar. "
        "Forecasts below still work without it (sentiment influence = 0)."
    )
else:
    try:
        articles = fetch_news(NEWS_API_KEY, NEWS_QUERY, N_ARTICLES)
        if not articles:
            st.info("No articles returned for this query.")
        else:
            sent_df = analyze_sentiment(articles)
    except Exception as e:
        st.error(f"News/sentiment step failed: {e}")

if not sent_df.empty:
    sentiment_score = float(sent_df["Score"].mean())
    counts = sent_df["Sentiment"].value_counts()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Aggregate sentiment (-1 to +1)", f"{sentiment_score:+.3f}")
    c2.metric("Positive", int(counts.get("Positive", 0)))
    c3.metric("Neutral", int(counts.get("Neutral", 0)))
    c4.metric("Negative", int(counts.get("Negative", 0)))
    st.dataframe(
        sent_df.style.format({"Positive": "{:.2f}", "Neutral": "{:.2f}", "Negative": "{:.2f}", "Score": "{:+.2f}"}),
        use_container_width=True,
        hide_index=True,
    )

# ----------------------------------------------------------------------------
# SECTION 2: HOLD-OUT EVALUATION
# ----------------------------------------------------------------------------
st.header("2. Model Comparison (out-of-sample hold-out)")
st.write(
    f"Models are trained on data **up to {df.index[-HORIZON - 1].date()}** and asked to forecast the "
    f"next **{HORIZON} trading days**, which they have never seen. "
    "The naive baseline (price stays flat) is included because a model that cannot beat it has no real skill."
)

res = run_holdout(df, HORIZON)
st.dataframe(res["metrics"].style.format("{:.2f}"), use_container_width=True)
st.caption(f"Auto-ARIMA selected order (p, d, q) = {res['arima_order']} on log-returns.")

test_idx = df.index[-HORIZON:]
hist = df["Close"].iloc[-(HORIZON + 90):]

fig1, ax1 = plt.subplots(figsize=(11, 4.5))
ax1.plot(hist.index, hist.values, label="Actual", color="black")
ax1.plot(test_idx, res["arima_pred"], label="ARIMA", linestyle="--")
ax1.plot(test_idx, res["prophet_pred"], label="Prophet", linestyle="--")
ax1.plot(test_idx, res["naive_pred"], label="Naive", linestyle=":")
ax1.axvline(test_idx[0], color="grey", alpha=0.5)
ax1.set_title("Hold-out forecast vs actual")
ax1.set_ylabel("Close price")
ax1.legend()
fig1.autofmt_xdate()
st.pyplot(fig1)

# ----------------------------------------------------------------------------
# SECTION 3: TRUE FUTURE FORECAST
# ----------------------------------------------------------------------------
st.header("3. Future Forecast")

fut = run_future(df, HORIZON, res["arima_order"])
dates = fut["dates"]
last_price = fut["last_price"]
base_returns = fut["arima_returns"]

# Heuristic sentiment overlay: a small drift (scaled by recent volatility) that
# fades over time, since news impact decays.
daily_vol = float(df["LogReturn"].iloc[-60:].std())
decay = np.exp(-np.arange(1, HORIZON + 1) / 10.0)
sentiment_drift = ALPHA * sentiment_score * daily_vol * decay

arima_future = pd.Series(last_price * np.exp(np.cumsum(base_returns)), index=dates)
arima_sent_future = pd.Series(last_price * np.exp(np.cumsum(base_returns + sentiment_drift)), index=dates)
prophet_future = fut["prophet"]

fig2, ax2 = plt.subplots(figsize=(11, 4.5))
hist2 = df["Close"].iloc[-120:]
ax2.plot(hist2.index, hist2.values, label="Historical", color="black")
ax2.plot(dates, arima_future, label="ARIMA", linestyle="--")
if ALPHA > 0 and not sent_df.empty:
    ax2.plot(dates, arima_sent_future, label=f"ARIMA + sentiment (score {sentiment_score:+.2f})", linestyle="--")
ax2.plot(prophet_future.index, prophet_future["yhat"], label="Prophet", linestyle="--")
ax2.fill_between(
    prophet_future.index, prophet_future["yhat_lower"], prophet_future["yhat_upper"],
    alpha=0.15, label="Prophet 80% interval",
)
ax2.set_title(f"Forecast for the next {HORIZON} business days")
ax2.set_ylabel("Close price")
ax2.legend()
fig2.autofmt_xdate()
st.pyplot(fig2)

summary = pd.DataFrame({
    "Last close": [last_price],
    f"ARIMA (day {HORIZON})": [arima_future.iloc[-1]],
    f"ARIMA + sentiment (day {HORIZON})": [arima_sent_future.iloc[-1]],
    f"Prophet (day {HORIZON})": [prophet_future["yhat"].iloc[-1]],
})
st.dataframe(summary.style.format("{:.2f}"), use_container_width=True, hide_index=True)

with st.expander("About the sentiment adjustment and limitations"):
    st.markdown(
        """
- The sentiment overlay is a **simple heuristic**, not a trained model: daily drift =
  `alpha x sentiment score x recent daily volatility x decay`. It has not been
  validated by back-testing; treat it as an experiment.
- ARIMA on log-returns usually forecasts returns close to zero, so its price path is
  nearly flat. That is expected for noisy financial series.
- Future dates use business days and ignore NSE holidays.
- Stock prices are driven by many factors not modelled here. This is not investment advice.
        """
    )
