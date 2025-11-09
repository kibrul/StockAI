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
import yfinance as yf
import requests

from utils import init
init()

st.set_page_config(page_title="📈 Breakout-Tani US Screener", layout="wide")
st.title("📈 Breakout-Tani US Screener (yfinance, EOD)")

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
run_btn = st.sidebar.button("Run BreakOut Screener")




import yfinance as yf

def get_reliable_latest_price_and_volume(ticker):
    """
    Fetches the most reliable latest price and volume data for a given ticker,
    using fallback mechanisms for price.
    """
    stock = yf.Ticker(ticker)
    info = stock.info
    
    # --- Price Retrieval (Same Robust Logic) ---
    
    # 1. Try the most reliable field for the latest price
    price = info.get('regularMarketPrice')
    
    # 2. Fallback to 'currentPrice'
    if price is None:
        price = info.get('currentPrice')
        
    # 3. If market is closed, check for pre/post-market price
    if price is None and info.get('marketState') != 'REGULAR':
        price = info.get('preMarketPrice') or info.get('postMarketPrice')
    
    # 4. Final fallback to the previous day's close
    if price is None:
        price = info.get('previousClose')

    # --- Volume Retrieval ---
    
    # 1. Try the most reliable field for the latest volume (current session)
    volume = info.get('regularMarketVolume')
    
    # 2. Fallback to the 'volume' field (often the same, but good for redundancy)
    if volume is None:
        volume = info.get('volume')
        
    # 3. Final fallback to the previous day's volume if current volume is None
    #    This ensures a volume figure is returned, even if stale.
    if volume is None:
        volume = info.get('previousVolume')
    
    # Convert volume to integer if it's not None
    if volume is not None:
        volume = int(volume)
        
    # --- Return Results ---
    
    # Return a dictionary for easy access
    return {
        'Price': price,
        'Volume': volume
    }

# Example usage:
#latest_qcom_data = get_reliable_latest_price_and_volume('QCOM')

#print(f"Latest QCOM Data:")
#print(f"  Price: ${latest_qcom_data['Price']:.2f}")
#print(f"  Volume: {latest_qcom_data['Volume']:,}")






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

        # Ensure columns named 'Close' and 'Volume' exist (yfinance standard)
        df_sym = df_sym[['Close', 'Volume']].rename(columns={'Close': 'close', 'Volume': 'volume'})
        df_sym = df_sym.sort_index()  # ascending by date

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

        st.write(df_sym)
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
