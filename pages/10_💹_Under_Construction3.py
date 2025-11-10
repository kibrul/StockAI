import bdshare
import streamlit as st
import pandas as pd
from bdshare import get_current_trade_data, get_hist_data, get_current_trading_code, get_market_inf_more_data, get_hist_data
import datetime as dt
from datetime import datetime, timedelta, timezone
import numpy as np
from ta.trend import sma_indicator
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

from utils import init
init()
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import datetime as dt
from datetime import timedelta

st.set_page_config(page_title="WeinKulBee - S&P500 Screener (Prefilter)", layout="wide")
st.title("📈 WeinKulBee - S&P500 Screener (Prefilter)")

# ----------------------
# User parameters
# ----------------------
TRADING_DAYS = 300  # history length for full screening
PREFILTER_DAYS = 60  # history window to compute avg volume and recent price
MIN_PRICE = 10.0
MIN_AVG_VOLUME = 600_000
MIN_MARKETCAP = 2_000_000_000  # $2B

MA_STAGE_WINDOW = 200
RESISTANCE_LOOKBACK = 50
VOLUME_MA_WINDOW = 50
MIN_DATA_POINTS_FOR_200DMA = 200

st.sidebar.markdown("## Prefilter settings")
st.sidebar.write(f"Price > ${MIN_PRICE}")
st.sidebar.write(f"Avg Daily Volume (last {PREFILTER_DAYS} days) > {MIN_AVG_VOLUME:,}")
st.sidebar.write(f"Market Cap > ${MIN_MARKETCAP:,}")
run_btn = st.sidebar.button("Run Prefilter + WeinKulBee")

# ----------------------
# Helpers
# ----------------------
@st.cache_data(ttl=24*3600)
def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    r = requests.get(url, headers=headers)
    r.raise_for_status()
    tables = pd.read_html(r.text)
    df = tables[0]
    tickers = df['Symbol'].str.replace('.', '-', regex=False).tolist()
    return tickers, df

def safe_yf_download(ticker_list, period_str):
    """Download with yfinance and handle exceptions, return raw df or empty."""
    try:
        raw = yf.download(tickers=ticker_list, period=period_str, interval="1d", group_by='ticker', threads=True, progress=False)
        return raw
    except Exception as e:
        st.warning(f"yfinance download error: {e}")
        return None

def compute_prefilter_stats(raw_bulk, tickers):
    """
    From a bulk yfinance download, compute last_close and avg_volume for each ticker.
    raw_bulk may be MultiIndex columns or single ticker.
    """
    stats = []
    for t in tickers:
        try:
            if isinstance(raw_bulk.columns, pd.MultiIndex):
                if t not in raw_bulk.columns.get_level_values(0):
                    continue
                df_t = raw_bulk[t].dropna(subset=['Close']).copy()
            else:
                df_t = raw_bulk.copy().dropna(subset=['Close'])
            if df_t.empty:
                continue
            last_close = df_t['Close'].iloc[-1]
            avg_vol = df_t['Volume'].tail(PREFILTER_DAYS).mean()
            stats.append((t, float(last_close), float(avg_vol)))
        except Exception:
            continue
    stats_df = pd.DataFrame(stats, columns=['symbol', 'last_close', 'avg_volume'])
    return stats_df

def fetch_marketcap_for_list(ticker_list):
    """Fetch marketCap using yf.Ticker.info for given tickers. Returns dict symbol->marketcap (or NaN)."""
    mc = {}
    for t in ticker_list:
        try:
            info = yf.Ticker(t).info
            mc_val = info.get('marketCap', np.nan)
            mc[t] = mc_val if mc_val is not None else np.nan
        except Exception:
            mc[t] = np.nan
    return mc

# WeinKulBee logic functions (EOD daily series; sorted ascending)
def is_stage_2(close_series):
    ma = close_series.rolling(window=MA_STAGE_WINDOW).mean()
    try:
        return bool((close_series.iloc[-1] > ma.iloc[-1]) and (ma.iloc[-1] > ma.iloc[-20]))
    except Exception:
        return False

def is_breakout(close_series):
    if len(close_series) < RESISTANCE_LOOKBACK + 2:
        return False
    recent_max = close_series.iloc[-RESISTANCE_LOOKBACK-1:-1].max()
    return bool(close_series.iloc[-1] > recent_max)

def is_stockbee_momentum(close_series, volume_series):
    if len(close_series) < 2 or len(volume_series) < VOLUME_MA_WINDOW:
        return False
    pct_change = (close_series.iloc[-1] - close_series.iloc[-2]) / close_series.iloc[-2]
    vol_ma = volume_series.rolling(window=VOLUME_MA_WINDOW).mean()
    vol_spike = False
    if not np.isnan(vol_ma.iloc[-1]):
        vol_spike = volume_series.iloc[-1] > 1.5 * vol_ma.iloc[-1]
    return bool((pct_change > 0.04) and vol_spike)

# ----------------------
# Main flow
# ----------------------
tickers, _ = get_sp500_tickers()
st.write(f"Loaded S&P 500 tickers: {len(tickers)} symbols.")

if not run_btn:
    st.info("Press **Run Prefilter + WeinKulBee** in the sidebar to start the run.")
    st.stop()

# 1) FAST prefilter using last PREFILTER_DAYS
st.info("Step 1: Prefilter — downloading recent data for all S&P500 tickers...")
period_prefilter = f"{PREFILTER_DAYS}d"
raw_pref = safe_yf_download(tickers, period_prefilter)
if raw_pref is None:
    st.error("Failed to download prefilter data. Try again later.")
    st.stop()

st.info("Computing last close and average volume...")
pref_stats_df = compute_prefilter_stats(raw_pref, tickers)

# Apply price & avg volume filters
pref_candidates = pref_stats_df[
    (pref_stats_df['last_close'] > MIN_PRICE) &
    (pref_stats_df['avg_volume'] > MIN_AVG_VOLUME)
].copy()

st.write(f"After price & avg-volume filters: {len(pref_candidates)} tickers remain.")

if pref_candidates.empty:
    st.warning("No tickers passed the initial price + avg-volume filters. Adjust thresholds and try again.")
    st.stop()

# 2) Fetch market caps for the prefilter survivors (slower but limited list)
st.info("Fetching market cap for prefiltered tickers (this loops per-ticker)...")
candidate_list = pref_candidates['symbol'].tolist()
marketcaps = fetch_marketcap_for_list(candidate_list)
pref_candidates['marketCap'] = pref_candidates['symbol'].map(marketcaps).fillna(np.nan)

# Apply market cap filter
pref_candidates = pref_candidates[pref_candidates['marketCap'] > MIN_MARKETCAP].copy()
st.write(f"After market cap filter (> ${MIN_MARKETCAP:,}): {len(pref_candidates)} tickers remain.")

if pref_candidates.empty:
    st.warning("No tickers passed the market cap filter. Adjust thresholds and try again.")
    st.stop()

st.dataframe(pref_candidates.reset_index(drop=True))

# 3) Download full TRADING_DAYS history for survivors (bulk)
st.info(f"Step 2: Downloading {TRADING_DAYS} trading days history for {len(pref_candidates)} tickers...")
period_full = f"{TRADING_DAYS}d"
survivors = pref_candidates['symbol'].tolist()
raw_full = safe_yf_download(survivors, period_full)
if raw_full is None:
    st.error("Failed to download full history. Try again later.")
    st.stop()

# 4) Run WeinKulBee screening on survivors
st.info("Step 3: Running WeinKulBee screening logic on survivors...")
results = []
final_results = []
progress = st.progress(0)
status = st.empty()

for idx, symbol in enumerate(survivors):
    status.text(f"Processing {symbol} ({idx+1}/{len(survivors)})")
    progress.progress((idx+1)/len(survivors))
    try:
        if isinstance(raw_full.columns, pd.MultiIndex):
            if symbol not in raw_full.columns.get_level_values(0):
                continue
            df_sym = raw_full[symbol].dropna(subset=['Close']).copy()
        else:
            df_sym = raw_full.copy().dropna(subset=['Close'])
        if df_sym.empty:
            continue

        df_sym = df_sym[['Close', 'Volume']].rename(columns={'Close': 'close', 'Volume': 'volume'}).sort_index()
        if len(df_sym) < MIN_DATA_POINTS_FOR_200DMA:
            continue

        close = df_sym['close'].astype(float)
        volume = df_sym['volume'].astype(float)

        in_stage_2 = is_stage_2(close)
        breakout = is_breakout(close)
        momentum = is_stockbee_momentum(close, volume)

        row = {
            "Symbol": symbol,
            "Price": float(close.iloc[-1]),
            "Stage 2": in_stage_2,
            "Breakout": breakout,
            "Momentum": momentum,
            "Latest Volume": int(volume.iloc[-1]),
            "Avg Volume (prefilter 60d)": int(pref_candidates.loc[pref_candidates['symbol'] == symbol, 'avg_volume'].values[0]),
            "MarketCap": int(pref_candidates.loc[pref_candidates['symbol'] == symbol, 'marketCap'].values[0])
        }

        if (in_stage_2 and breakout) or (in_stage_2 and momentum) or (breakout and momentum):
            results.append(row)

        if in_stage_2 and breakout and momentum:
            final_results.append(row)

    except Exception as e:
        # continue on error
        st.write(f"Error processing {symbol}: {e}")
        continue

status.empty()
progress.empty()

screener_df = pd.DataFrame(results).sort_values(by=['Stage 2', 'Breakout', 'Momentum', 'Price'], ascending=[False, False, False, False])
final_df = pd.DataFrame(final_results).sort_values(by=['Price'], ascending=False)

st.subheader("🎯 WeinKulBee — Matches (any 2 of 3 criteria)")
st.write(f"Found {len(screener_df)} matches.")
st.dataframe(screener_df.reset_index(drop=True))

st.subheader("🔥 WeinKulBee — Strong Matches (all 3 criteria)")
st.write(f"Found {len(final_df)} strong matches.")
st.dataframe(final_df.reset_index(drop=True))

# Download buttons
if not screener_df.empty:
    st.download_button("Download matches CSV", data=screener_df.to_csv(index=False).encode('utf-8'),
                       file_name='weinkulbee_matches.csv', mime='text/csv')
if not final_df.empty:
    st.download_button("Download strong matches CSV", data=final_df.to_csv(index=False).encode('utf-8'),
                       file_name='weinkulbee_strong_matches.csv', mime='text/csv')

st.success("Prefilter + WeinKulBee scan complete.")
