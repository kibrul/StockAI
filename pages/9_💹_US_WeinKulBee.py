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
from io import StringIO

from utils import init
init()

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
import requests
from datetime import datetime, timedelta, timezone

st.set_page_config(page_title="WeinKulBee - S&P500 Screener", layout="wide")
st.title("📈 WeinKulBee - S&P500 Screener (yfinance, EOD)")

#
# CONFIG
#
TRADING_DAYS = 300  # user requested 300 trading days
MIN_DATA_POINTS_FOR_200DMA = 200
# WeinKulBee parameters (same logic as your DSE code)
MA_STAGE_WINDOW = 200  # 200-day MA for Stage 2
RESISTANCE_LOOKBACK = 50  # 50-day resistance for breakout
VOLUME_MA_WINDOW = 50  # 50-day volume MA for momentum
MIN_VOLUME_FOR_ACTIVE = 100000  # optional filter to skip extremely illiquid names

# Simple UI controls
st.sidebar.markdown("## Settings")
st.sidebar.write(f"History length (trading days): **{TRADING_DAYS}**")
st.sidebar.write("Universe: **S&P 500**")
run_btn = st.sidebar.button("Run WeinKulBee Screener")

#st.markdown("""
#This screener:
#- downloads historical **daily** data for S&P 500 tickers (300 trading days),
#- computes Stage-2 (200-day MA), 50-day breakout, and StockBee momentum (≥4% move + volume spike),
#- shows all matches and the strongest matches (all three criteria).
#""")

#
# Helper: get S&P500 tickers from Wikipedia
#
@st.cache_data(ttl=24*3600)
def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    # 1. Fetch HTML content
    r = requests.get(url, headers=headers)
    r.raise_for_status()  # raise if still blocked

    # 2. FIX: Wrap the HTML string in StringIO object
    html_buffer = StringIO(r.text)

    # 3. Read HTML tables from the StringIO buffer
    tables = pd.read_html(html_buffer) # <-- CHANGED: passing html_buffer instead of r.text
    
    df = tables[1]
    tickers = df['Symbol'].str.replace('.', '-', regex=False).tolist()
    return tickers, df


#
# Screening functions (WeinKulBee logic)
#
def is_stage_2(close_series):
    # close_series is a pd.Series sorted by date ascending
    ma = close_series.rolling(window=MA_STAGE_WINDOW).mean()
    try:
        cond = (close_series.iloc[-1] > ma.iloc[-1]) and (ma.iloc[-1] > ma.iloc[-20])
        return bool(cond)
    except Exception:
        return False

def is_breakout(close_series):
    # check breakout vs recent resistance (exclude current bar)
    if len(close_series) < RESISTANCE_LOOKBACK + 2:
        return False
    recent_max = close_series.iloc[-RESISTANCE_LOOKBACK-1:-1].max()
    return bool(close_series.iloc[-1] > recent_max)

def is_stockbee_momentum(close_series, volume_series):
    # price pct change (last vs previous) + volume spike vs 50-day avg
    if len(close_series) < 2 or len(volume_series) < VOLUME_MA_WINDOW:
        return False
    pct_change = (close_series.iloc[-1] - close_series.iloc[-2]) / close_series.iloc[-2]
    vol_ma = volume_series.rolling(window=VOLUME_MA_WINDOW).mean()
    vol_spike = volume_series.iloc[-1] > 1.5 * vol_ma.iloc[-1] if not np.isnan(vol_ma.iloc[-1]) else False
    return bool((pct_change > 0.04) and vol_spike)

#
# Main app logic
#
tickers, sp500_df = get_sp500_tickers()
st.write(f"Loaded S&P 500 tickers: {len(tickers)} symbols.")

if not run_btn:
    st.info("Press **Run WeinKulBee Screener** in the sidebar to start.")
    st.stop()

# Download bulk historical data with yfinance
with st.spinner("Downloading historical data (this may take ~20-60s)..."):
    end_date = dt.datetime.now().date()
    # yfinance 'period' supports trading-day like '300d'
    period_str = f"{TRADING_DAYS}d"
    # Use yf.download for many tickers — it returns a multi-level column dataframe
    try:
        raw = yf.download(tickers=tickers, period=period_str, interval="1d", group_by='ticker', threads=True, progress=False)
    except Exception as e:
        st.error(f"yfinance download error: {e}")
        st.stop()

# Prepare results containers
results = []
final_result = []

# UI elements for progress
status_text = st.empty()
main_progress = st.progress(0)

# If yfinance returns a single-ticker DataFrame the shape is different.
# We'll iterate tickers and extract data safely.
for idx, symbol in enumerate(tickers):
    status_text.text(f"Processing {symbol} — {idx+1}/{len(tickers)}")
    main_progress.progress((idx + 1) / len(tickers))

    try:
        # Extract per-symbol data from the bulk download output.
        if isinstance(raw.columns, pd.MultiIndex):
            if symbol in raw.columns.get_level_values(0):
                df_sym = raw[symbol].copy()
            else:
                # Sometimes yfinance uses a different ticker format; skip if missing
                continue
        else:
            # Single symbol result (rare when only one ticker), assume raw is the df
            df_sym = raw.copy()

        # Clean & drop rows with NaNs in Close
        df_sym = df_sym.dropna(subset=['Close']).copy()
        if df_sym.empty:
            continue

        # Ensure columns named 'Close' and 'Volume' exist (yfinance standard)
        df_sym = df_sym[['Close', 'Volume']].rename(columns={'Close': 'close', 'Volume': 'volume'})
        df_sym = df_sym.sort_index()  # ascending by date

        # Basic liquidity filter (optional)
        if df_sym['volume'].iloc[-1] < MIN_VOLUME_FOR_ACTIVE:
            # Skip extremely illiquid names to speed up results; comment this out if you want all symbols
            continue

        # Need enough data for 200dma
        if len(df_sym) < MIN_DATA_POINTS_FOR_200DMA:
            continue

        close = df_sym['close'].astype(float)
        volume = df_sym['volume'].astype(float)

        in_stage_2 = is_stage_2(close)
        breakout = is_breakout(close)
        momentum = is_stockbee_momentum(close, volume)

        if (in_stage_2 and breakout) or (in_stage_2 and momentum) or (breakout and momentum):
            results.append({
                "Symbol": symbol,
                "Price": float(close.iloc[-1]),
                "Stage 2": in_stage_2,
                "Breakout": breakout,
                "Momentum": momentum,
                "Latest Volume": int(volume.iloc[-1])
            })

        if in_stage_2 and breakout and momentum:
            final_result.append({
                "Symbol": symbol,
                "Price": float(close.iloc[-1]),
                "Stage 2": in_stage_2,
                "Breakout": breakout,
                "Momentum": momentum,
                "Latest Volume": int(volume.iloc[-1])
            })

    except Exception as e:
        # non-fatal: show and continue
        st.write(f"Error processing {symbol}: {e}")
        continue
        
if not screener_df.empty:
    # Convert to DataFrame and display
    screener_df = pd.DataFrame(results).sort_values(by=['Stage 2', 'Breakout', 'Momentum', 'Price'], ascending=[False, False, False, False])
if not fscreener_df.empty:
    fscreener_df = pd.DataFrame(final_result).sort_values(by=['Price'], ascending=False)

st.subheader("🎯 WeinKulBee - All Matches (any 2 of 3 criteria)")
st.write(f"Found {len(screener_df)} matches (any 2 of 3 criteria).")
st.dataframe(screener_df.reset_index(drop=True))

st.subheader("🔥 WeinKulBee - Strong Matches (all 3 criteria)")
st.write(f"Found {len(fscreener_df)} strong matches (Stage2 + Breakout + Momentum).")
st.dataframe(fscreener_df.reset_index(drop=True))

# Quick download buttons
if not screener_df.empty:
    csv = screener_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download matches CSV", data=csv, file_name='weinkulbee_matches.csv', mime='text/csv')

if not fscreener_df.empty:
    csv2 = fscreener_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download strong matches CSV", data=csv2, file_name='weinkulbee_strong_matches.csv', mime='text/csv')

st.success("Scan complete.")
status_text.empty()
main_progress.empty()



