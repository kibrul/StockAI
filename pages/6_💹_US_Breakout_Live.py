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
import datetime as dt
import requests
from datetime import datetime, timezone, date, timedelta

st.set_page_config(page_title="📈 Breakout-Tani US Screener", layout="wide")
st.title("📈 Breakout-Tani US Screener (yfinance, Live)")

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

# 1. Set the end date for HISTORICAL data to TODAY's date.
#    yfinance will typically fetch data UP TO (but not including) this date,
#    which means we get data up to yesterday's close.
end_date_hist = date.today()  #- timedelta(days=3)

# Simple UI controls
st.sidebar.markdown("## Settings")
st.sidebar.write(f"History length (trading days): **{TRADING_DAYS}**")
st.sidebar.write("Universe: **S&P 500**")
run_btn = st.sidebar.button("Run BreakOut Screener")


def get_reliable_latest_quote_data(ticker):
    """
    Fetches the most reliable latest price, volume, and trade time
    data for a given ticker, including necessary fallbacks.
    """
    stock = yf.Ticker(ticker)
    info = stock.info

    # --- Price Retrieval ---
    price = info.get('regularMarketPrice')
    if price is None:
        price = info.get('currentPrice')
    if price is None and info.get('marketState') != 'REGULAR':
        price = info.get('preMarketPrice') or info.get('postMarketPrice')
    if price is None:
        price = info.get('previousClose')

    # --- Volume Retrieval ---
    volume = info.get('regularMarketVolume')
    if volume is None:
        volume = info.get('volume')
    if volume is None:
        # We need the previous day's close for the 'Open' field of today's row
        prev_close = info.get('previousClose')
    if volume is not None:
        volume = int(volume)

    # --- Date/Time Retrieval ---
    # timestamp = info.get('regularMarketTime') or info.get('exchangeDataDelayedBy')

    # date_time_str = None
    # if timestamp is not None and isinstance(timestamp, (int, float)):
    #    # Convert to datetime object (UTC)
    #    dt_object = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    # We only need the date for the index of the DataFrame
    #     trade_date = dt_object.date()
    # else:
    # Fallback to today's date if timestamp is unavailable
    trade_date = date.today()

    # --- Return Results Dictionary (Aligned for DataFrame) ---
    return {
        'DateIndex': pd.to_datetime(trade_date),  # Use this for DataFrame index
        'Open': info.get('regularMarketOpen', price),  # Use today's open, fallback to latest price
        'High': info.get('regularMarketDayHigh', price),  # Use day high, fallback to latest price
        'Low': info.get('regularMarketDayLow', price),  # Use day low, fallback to latest price
        'Close': price,
        'Adj Close': price,  # Assume Adj Close = Close for the current, live bar
        'Volume': volume
    }


#
# Helper: get S&P500 tickers from Wikipedia
#
@st.cache_data(ttl=24*3600)
def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    r = requests.get(url, headers=headers)
    r.raise_for_status()  # raise if still blocked

    tables = pd.read_html(r.text)
    df = tables[0]
    tickers = df['Symbol'].str.replace('.', '-', regex=False).tolist()
    return tickers, df


#
# Main app logic
#
tickers, sp500_df = get_sp500_tickers()
st.write(f"Loaded S&P 500 tickers: {len(tickers)} symbols.")

if not run_btn:
    st.info("Press **Run BreakOut Screener** in the sidebar to start.")
    st.stop()

# Download bulk historical data with yfinance
with st.spinner("Downloading historical data (this may take ~20-60s)..."):
    end_date = dt.datetime.now().date()
    # yfinance 'period' supports trading-day like '300d'
    period_str = f"{TRADING_DAYS}d"
    # Use yf.download for many tickers — it returns a multi-level column dataframe
    try:
        raw = yf.download(tickers=tickers, period=period_str, interval="1d", group_by='ticker', threads=True, progress=False)

        # 2. Download Historical Data (now correctly using the 'end' parameter)
        #with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        #    raw_hist = yf.download(
        #        tickers=[TICKER],
        #        period=HISTORY_PERIOD,
        #        interval="1d",
        #        # --- FIX IS HERE ---
        #        end=end_date_hist,
        #        # -------------------
        #        group_by='ticker',
        #        threads=True,
        #        progress=False
        #    ).droplevel(0, axis=1)


    except Exception as e:
        st.error(f"yfinance download error: {e}")
        st.stop()

# Prepare results containers
breakout = []

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

        # 3. Get Today's Live Quote Data (as before)
        live_quote_dict = get_reliable_latest_quote_data(symbol)
        #print(live_quote_dict)

        # 4. Convert Live Quote to DataFrame Row and Align Index
        df_live_row = pd.DataFrame(
            [live_quote_dict],
            index=[live_quote_dict['DateIndex']]
        ).drop(columns=['DateIndex'])

        # Ensure columns align perfectly
        df_live_row = df_live_row.reindex(columns=df_sym.columns)

        # 5. Concatenate the two DataFrames
        df_final = pd.concat([df_sym, df_live_row], axis=0)


        # Ensure columns named 'Close' and 'Volume' exist (yfinance standard)
        df_sym = df_final[['Close', 'Volume']].rename(columns={'Close': 'close', 'Volume': 'volume'})
        df_sym = df_sym.sort_index()  # ascending by date
        #print(df_sym.tail(3))
        # Basic liquidity filter (optional)
        if df_sym['volume'].iloc[-1] < MIN_VOLUME_FOR_ACTIVE:
            # Skip extremely illiquid names to speed up results; comment this out if you want all symbols
            continue

        # Basic Price filter (optional)
        if df_sym['close'].iloc[-1] > 300:
            continue

        # Need enough data for 200dma
        if len(df_sym) < MIN_DATA_POINTS_FOR_200DMA:
            continue





        #st.write(df_sym)
        close = df_sym['close'].astype(float)
        volume = df_sym['volume'].astype(float)

        #in_stage_2 = is_stage_2(close)
        #breakout = is_breakout(close)
        #momentum = is_stockbee_momentum(close, volume)
        avg9 = close.tail(6).mean()
        avg66 = close.tail(66).mean()

        # Condition 1: Short-term > long-term by 5%
        ratio_condition = (avg9 / avg66) > 1.06

        # Condition 2: Today's % change between -1% and +1%
        db_today_close = close.iloc[-2]
        db_yesterday_close = close.iloc[-3]
        pct_change = ((db_today_close - db_yesterday_close) / db_yesterday_close) * 100
        flat_condition = -1 <= pct_change <= 1

        # Bullish 4% Break Out
        today_close = close.iloc[-1]
        yesterday_close = close.iloc[-2]
        bullish4 = (today_close / yesterday_close) >= 1.04

        # if bullish4 and ratio_condition and flat_condition:
        if bullish4 and ratio_condition:
            breakout.append({
                'Symbol': symbol,
                '9d/66d Ratio': round(avg9 / avg66, 3),
                '% Change Today': round(pct_change, 2),
                'Last Price': today_close
            })
    except Exception as e:
        # non-fatal: show and continue
        st.write(f"Error processing {symbol}: {e}")
        continue

# Convert to DataFrame and display
#screener_df = pd.DataFrame(results).sort_values(by=['Stage 2', 'Breakout', 'Momentum', 'Price'], ascending=[False, False, False, False])
fscreener_df = pd.DataFrame(breakout)

st.subheader("🔥 Breakout Watchlist (Consolidating after strength):")
st.write(f"Found {len(fscreener_df)} Strong Matches Breakout.")
st.dataframe(fscreener_df.reset_index(drop=True))

if not fscreener_df.empty:
    csv2 = fscreener_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download strong matches CSV", data=csv2, file_name='Breakout_Matches.csv', mime='text/csv')

st.success("Scan complete.")
status_text.empty()
main_progress.empty()
