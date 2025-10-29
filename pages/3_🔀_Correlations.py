from utils import init
init()
# weinkulbee_sp500_nearreal.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import datetime as dt
from datetime import timedelta

st.set_page_config(page_title="WeinKulBee - S&P500 Near-Real-Time", layout="wide")
st.title("📈 WeinKulBee - S&P500 Near-Real-Time (yfinance, Manual Run)")

# ----------------------
# User parameters (your chosen filters)
# ----------------------
TRADING_DAYS = 300
PREFILTER_DAYS = 60
MIN_PRICE = 10.0
MIN_AVG_VOLUME = 600_000
MIN_MARKETCAP = 2_000_000_000  # $2B

MA_STAGE_WINDOW = 200
RESISTANCE_LOOKBACK = 50
VOLUME_MA_WINDOW = 50
MIN_DATA_POINTS_FOR_200DMA = 200

st.sidebar.markdown("## Settings (Near-Real-Time, Manual Run)")
st.sidebar.write("Data source: yfinance (near-real-time via fast_info when available)")
st.sidebar.write(f"History: {TRADING_DAYS} trading days")
st.sidebar.write(f"Prefilter: last {PREFILTER_DAYS} days for avg volume & price")
st.sidebar.write(f"Price > ${MIN_PRICE}, AvgVol > {MIN_AVG_VOLUME:,}, MarketCap > ${MIN_MARKETCAP:,}")
run_btn = st.sidebar.button("Run WeinKulBee (Manual)")

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
    try:
        raw = yf.download(tickers=ticker_list, period=period_str, interval="1d",
                          group_by='ticker', threads=True, progress=False)
        return raw
    except Exception as e:
        st.warning(f"yfinance download error: {e}")
        return None

def compute_prefilter_stats(raw_bulk, tickers):
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
    mc = {}
    for t in ticker_list:
        try:
            info = yf.Ticker(t).info
            mc_val = info.get('marketCap', np.nan)
            mc[t] = mc_val if mc_val is not None else np.nan
        except Exception:
            mc[t] = np.nan
    return mc

def try_get_latest_from_fastinfo(ticker_obj):
    """
    Attempt to get near-real-time values from yfinance's fast_info or info.
    Returns dict with keys: last_price, last_volume (may be None)
    """
    out = {'last_price': None, 'last_volume': None}
    try:
        fast = getattr(ticker_obj, "fast_info", None)
        if fast:
            # fast_info field names differ by yfinance versions; try common ones
            lp = fast.get("lastPrice") or fast.get("last_price") or fast.get("last_close")
            lv = fast.get("lastVolume") or fast.get("last_volume")
            if lp is not None:
                out['last_price'] = float(lp)
            if lv is not None:
                out['last_volume'] = int(lv)
        # fallback to info's marketPrice fields
        if out['last_price'] is None:
            info = ticker_obj.info
            rp = info.get('regularMarketPrice') or info.get('previousClose') or info.get('currentPrice')
            rv = info.get('regularMarketVolume') or info.get('volume')
            if rp is not None:
                out['last_price'] = float(rp)
            if rv is not None:
                try:
                    out['last_volume'] = int(rv)
                except Exception:
                    out['last_volume'] = None
    except Exception:
        pass
    return out

# WeinKulBee logic (EOD series; if latest price present we append it)
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
    st.info("Press **Run WeinKulBee (Manual)** in the sidebar to start the scan.")
    st.stop()

# Step 1 - Prefilter (fast)
st.info("Step 1: Prefiltering using last 60 days (price + avg volume)...")
period_prefilter = f"{PREFILTER_DAYS}d"
raw_pref = safe_yf_download(tickers, period_prefilter)
if raw_pref is None:
    st.error("Failed to download prefilter data. Try again later.")
    st.stop()

pref_stats_df = compute_prefilter_stats(raw_pref, tickers)
pref_candidates = pref_stats_df[
    (pref_stats_df['last_close'] > MIN_PRICE) &
    (pref_stats_df['avg_volume'] > MIN_AVG_VOLUME)
].copy()
st.write(f"After price & avg-volume filters: {len(pref_candidates)} tickers remain.")
if pref_candidates.empty:
    st.warning("No tickers passed prefilter. Try lowering thresholds.")
    st.stop()

# Step 2 - Market cap filter
st.info("Fetching market caps for prefiltered tickers...")
candidate_list = pref_candidates['symbol'].tolist()
marketcaps = fetch_marketcap_for_list(candidate_list)
pref_candidates['marketCap'] = pref_candidates['symbol'].map(marketcaps).fillna(np.nan)
pref_candidates = pref_candidates[pref_candidates['marketCap'] > MIN_MARKETCAP].copy()
st.write(f"After market cap filter: {len(pref_candidates)} tickers remain.")
if pref_candidates.empty:
    st.warning("No tickers passed market cap filter.")
    st.stop()

st.dataframe(pref_candidates.reset_index(drop=True))

# Step 3 - Download full history for survivors
st.info(f"Step 2: Downloading {TRADING_DAYS} trading days history for survivors...")
survivors = pref_candidates['symbol'].tolist()
raw_full = safe_yf_download(survivors, f"{TRADING_DAYS}d")
if raw_full is None:
    st.error("Failed to download full history.")
    st.stop()

# Step 4 - Run WeinKulBee with near-real-time augmentation
st.info("Step 3: Running WeinKulBee (EOD series + near-real-time latest price where available)...")
results = []
final_results = []
progress = st.progress(0)
status = st.empty()

for i, symbol in enumerate(survivors):
    status.text(f"Processing {symbol} ({i+1}/{len(survivors)})")
    progress.progress((i+1)/len(survivors))
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

        # Try to augment with latest near-real-time price from yfinance.fast_info / info
        ticker_obj = yf.Ticker(symbol)
        latest = try_get_latest_from_fastinfo(ticker_obj)
        appended = False
        try:
            if latest['last_price'] is not None:
                # only append if last_price is different from latest close (and not NaN)
                last_ts = df_sym.index[-1]
                # Make synthetic row dated "now" to represent latest tick (note: index must be datetime like)
                now_dt = pd.Timestamp(dt.datetime.now(tz=dt.timezone.utc))
                # If last recorded close equals last_price (within small tolerance) skip
                if not np.isclose(float(df_sym['close'].iloc[-1]), float(latest['last_price']), atol=1e-6):
                    row = pd.DataFrame({
                        'close': [float(latest['last_price'])],
                        'volume': [int(latest['last_volume']) if latest['last_volume'] is not None else int(df_sym['volume'].iloc[-1])]
                    }, index=[now_dt])
                    # Append but keep chronological order expected by logic (we'll sort)
                    df_sym = pd.concat([df_sym, row])
                    appended = True
        except Exception:
            # ignore augment failures
            appended = False

        df_sym = df_sym.sort_index()  # ensure ascending dates

        # Need enough data
        if len(df_sym) < MIN_DATA_POINTS_FOR_200DMA:
            continue

        close = df_sym['close'].astype(float)
        volume = df_sym['volume'].astype(float)

        in_stage_2 = is_stage_2(close)
        breakout = is_breakout(close)
        momentum = is_stockbee_momentum(close, volume)

        # Build result row (note if appended, 'LatestType' = 'near-real-time', else 'EOD')
        latest_type = 'near-real-time' if appended else 'EOD'
        row = {
            "Symbol": symbol,
            "Price": float(close.iloc[-1]),
            "LatestType": latest_type,
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

if not screener_df.empty:
    st.download_button("Download matches CSV", data=screener_df.to_csv(index=False).encode('utf-8'),
                       file_name='weinkulbee_matches_nearreal.csv', mime='text/csv')
if not final_df.empty:
    st.download_button("Download strong matches CSV", data=final_df.to_csv(index=False).encode('utf-8'),
                       file_name='weinkulbee_strong_nearreal.csv', mime='text/csv')

st.success("Near-real-time WeinKulBee scan complete.")
