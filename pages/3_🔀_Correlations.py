from utils import init
init()


# weinkulbee_sp500_realtime_refresh.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import datetime as dt
from datetime import timedelta
import matplotlib.pyplot as plt
import io
import time

st.set_page_config(page_title="WeinKulBee - Near-RealTime Refresh + Sparklines", layout="wide")
st.title("📈 WeinKulBee - Near-RealTime Refresh + Sparklines (S&P500)")

# ----------------------
# Parameters (you chose earlier)
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

# caching TTLs
CACHE_TTL_FINVIZ = 6 * 3600  # 6 hours
CACHE_TTL_SP500 = 24 * 3600  # 24 hours

# ----------------------
# Helpers
# ----------------------
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

@st.cache_data(ttl=CACHE_TTL_SP500)
def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    tables = pd.read_html(r.text)
    df = tables[0]
    tickers = df['Symbol'].str.replace('.', '-', regex=False).tolist()
    return tickers, df

@st.cache_data(ttl=CACHE_TTL_FINVIZ)
def fetch_marketcaps_finviz():
    """
    Scrape Finviz screener pages filtered for S&P500 (idx_sp500) and return dict symbol->marketCap (int or NaN).
    Finviz paginates results; we iterate until no new rows.
    """
    base = "https://finviz.com/screener.ashx?v=111&f=idx_sp500"
    page = 1
    all_rows = []
    while True:
        url = base + f"&r={(page-1)*20+1}"
        r = requests.get(url, headers=HEADERS, timeout=20)
        if r.status_code != 200:
            break
        # find table via pandas.read_html on r.text
        try:
            tables = pd.read_html(r.text)
        except Exception:
            break
        # Finviz screener primary table is usually the 1st or 4th; find the one with 'No.' column
        found = None
        for t in tables:
            if 'No.' in t.columns or 'Ticker' in t.columns or 'Ticker' in t.columns.tolist():
                found = t
                break
        if found is None:
            # try to guess the largest table
            found = tables[0]
        # Finviz columns often: No., Ticker, Company, Sector, Industry, Country, Market Cap, P/E, etc
        # normalize
        cols = [c.lower() for c in found.columns]
        # If there's 'ticker' column
        if 'Ticker' in found.columns:
            # rename to consistent
            df = found.rename(columns={c: c for c in found.columns})
        else:
            df = found
        # Attempt to extract Ticker and Market Cap columns by name matching
        # flexible matching:
        ticker_col = None
        mcap_col = None
        for c in df.columns:
            cl = str(c).lower()
            if 'ticker' in cl or c == 'No.':
                # skip 'No.' as ticker, but prefer explicit Ticker
                if 'ticker' in cl:
                    ticker_col = c
            if 'market cap' in cl or 'marketcap' in cl or 'mkt.cap' in cl:
                mcap_col = c
        # fallback heuristics
        if ticker_col is None:
            # column of length 1-5 uppercase tickers
            for c in df.columns:
                sample = df[c].astype(str).head(6).tolist()
                if all(s.isupper() and len(s) <= 5 for s in sample):
                    ticker_col = c
                    break
        if mcap_col is None:
            for c in df.columns:
                if any(word in str(c).lower() for word in ['market', 'mkt']):
                    mcap_col = c
                    break
        # if still not found, try index-based guess (Ticker often at position 1, MarketCap near the end)
        if ticker_col is None:
            ticker_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
        if mcap_col is None:
            # try last 5 columns for something with '$' or 'B'/'M'
            found_mc = None
            for c in df.columns[-5:]:
                sample = df[c].astype(str).head(6).tolist()
                if any(('$' in s) or ('B' in s) or ('M' in s) for s in sample):
                    found_mc = c
                    break
            if found_mc:
                mcap_col = found_mc
        # Now parse rows
        if ticker_col not in df.columns:
            break
        for _, row in df.iterrows():
            try:
                sym = str(row[ticker_col]).strip()
                mcap_raw = row[mcap_col] if mcap_col in df.columns else np.nan
                # Convert mcap strings like '12.3B' or '$12.3B' or '123M' to integer
                mcap_val = np.nan
                try:
                    s = str(mcap_raw).strip().replace('$','').replace(',','')
                    if s.endswith('B') or s.endswith('b'):
                        mcap_val = float(s[:-1]) * 1_000_000_000
                    elif s.endswith('M') or s.endswith('m'):
                        mcap_val = float(s[:-1]) * 1_000_000
                    else:
                        # number
                        mcap_val = float(s)
                except Exception:
                    mcap_val = np.nan
                all_rows.append((sym, mcap_val))
            except Exception:
                continue
        # Finviz shows 20 rows per page; if less than 20 rows we reached last page
        if len(df) < 20:
            break
        page += 1
        time.sleep(0.5)  # be polite
    df_all = pd.DataFrame(all_rows, columns=['symbol', 'marketCap'])
    # remove duplicates keep first
    df_all = df_all.drop_duplicates(subset=['symbol']).set_index('symbol')
    return df_all['marketCap'].to_dict()

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

def try_get_latest_from_fastinfo(ticker_obj):
    out = {'last_price': None, 'last_volume': None}
    try:
        fast = getattr(ticker_obj, "fast_info", None)
        if fast:
            lp = fast.get("lastPrice") or fast.get("last_price") or fast.get("last_close")
            lv = fast.get("lastVolume") or fast.get("last_volume")
            if lp is not None:
                out['last_price'] = float(lp)
            if lv is not None:
                out['last_volume'] = int(lv)
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

# WeinKulBee logic
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
# UI and main flow
# ----------------------
tickers, _ = get_sp500_tickers()
st.write(f"Loaded S&P 500 tickers: {len(tickers)} symbols.")

# Buttons
col1, col2 = st.columns(2)
with col1:
    run_full = st.button("Run Full Prefilter + WeinKulBee")
with col2:
    refresh_btn = st.button("Refresh Latest Prices for Survivors (Fast)")

# Storage in session state
if 'survivors' not in st.session_state:
    st.session_state.survivors = []  # list of symbols that survived prefilter+marketcap
if 'raw_full' not in st.session_state:
    st.session_state.raw_full = None
if 'screener_df' not in st.session_state:
    st.session_state.screener_df = pd.DataFrame()
if 'final_df' not in st.session_state:
    st.session_state.final_df = pd.DataFrame()
if 'marketcaps' not in st.session_state:
    st.session_state.marketcaps = {}

# Full run
if run_full:
    st.info("Step 1: Prefilter using last 60 days")
    raw_pref = safe_yf_download(tickers, f"{PREFILTER_DAYS}d")
    if raw_pref is None:
        st.error("Prefilter download failed.")
        st.stop()
    pref_stats_df = compute_prefilter_stats(raw_pref, tickers)
    pref_candidates = pref_stats_df[
        (pref_stats_df['last_close'] > MIN_PRICE) &
        (pref_stats_df['avg_volume'] > MIN_AVG_VOLUME)
    ].copy()
    st.write(f"After price & avg-volume filters: {len(pref_candidates)} tickers remain.")
    if pref_candidates.empty:
        st.warning("No tickers passed prefilter")
        st.stop()

    st.info("Fetching bulk market caps from Finviz (fast)")
    finviz_mc = fetch_marketcaps_finviz()  # cached for 6 hours
    st.session_state.marketcaps = finviz_mc
    pref_candidates['marketCap'] = pref_candidates['symbol'].map(finviz_mc).fillna(np.nan)
    pref_candidates = pref_candidates[pref_candidates['marketCap'] > MIN_MARKETCAP].copy()
    st.write(f"After market cap filter: {len(pref_candidates)} tickers remain.")
    st.dataframe(pref_candidates.reset_index(drop=True))

    # Download full history for survivors
    survivors = pref_candidates['symbol'].tolist()
    st.session_state.survivors = survivors
    st.info(f"Downloading {TRADING_DAYS} trading days for {len(survivors)} survivors...")
    raw_full = safe_yf_download(survivors, f"{TRADING_DAYS}d")
    if raw_full is None:
        st.error("Full history download failed.")
        st.stop()
    st.session_state.raw_full = raw_full

    # Compute screening results
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
            df_sym = df_sym[['Close', 'Volume']].rename(columns={'Close':'close','Volume':'volume'}).sort_index()
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
                "Avg Volume (prefilter 60d)": int(pref_candidates.loc[pref_candidates['symbol']==symbol,'avg_volume'].values[0]),
                "MarketCap": int(pref_candidates.loc[pref_candidates['symbol']==symbol,'marketCap'].values[0])
            }
            if (in_stage_2 and breakout) or (in_stage_2 and momentum) or (breakout and momentum):
                results.append(row)
            if in_stage_2 and breakout and momentum:
                final_results.append(row)
        except Exception as e:
            st.write(f"Error {symbol}: {e}")
            continue
    status.empty()
    progress.empty()
    st.session_state.screener_df = pd.DataFrame(results)
    st.session_state.final_df = pd.DataFrame(final_results)
    st.success("Full run complete. Use the Refresh button to update latest prices later.")

# Refresh (fast) - only updates survivors & recomputes signals
if refresh_btn:
    if not st.session_state.survivors or st.session_state.raw_full is None:
        st.warning("No survivors cached. Run full prefilter first.")
    else:
        st.info("Refreshing latest prices for survivors (fast). This updates signals using latest price if available.")
        survivors = st.session_state.survivors
        raw_full = st.session_state.raw_full.copy()
        # We'll try to augment each survivor with latest price via yf.Ticker.fast_info/info and recompute signals
        updated_results = []
        updated_final = []
        prog = st.progress(0)
        stat = st.empty()
        for i, symbol in enumerate(survivors):
            stat.text(f"Refreshing {symbol} ({i+1}/{len(survivors)})")
            prog.progress((i+1)/len(survivors))
            try:
                # extract historic df
                if isinstance(raw_full.columns, pd.MultiIndex):
                    if symbol not in raw_full.columns.get_level_values(0):
                        continue
                    df_sym = raw_full[symbol].dropna(subset=['Close']).copy()
                else:
                    df_sym = raw_full.copy().dropna(subset=['Close'])
                if df_sym.empty:
                    continue
                df_sym = df_sym[['Close','Volume']].rename(columns={'Close':'close','Volume':'volume'}).sort_index()
                # get latest via yfinance fast_info/info
                ticker_obj = yf.Ticker(symbol)
                latest = try_get_latest_from_fastinfo(ticker_obj)
                appended = False
                try:
                    if latest['last_price'] is not None:
                        if not np.isclose(float(df_sym['close'].iloc[-1]), float(latest['last_price']), atol=1e-6):
                            now_dt = pd.Timestamp(dt.datetime.now(tz=dt.timezone.utc))
                            row = pd.DataFrame({
                                'close':[float(latest['last_price'])],
                                'volume':[int(latest['last_volume']) if latest['last_volume'] is not None else int(df_sym['volume'].iloc[-1])]
                            }, index=[now_dt])
                            df_sym = pd.concat([df_sym, row])
                            appended = True
                except Exception:
                    appended = False
                df_sym = df_sym.sort_index()
                if len(df_sym) < MIN_DATA_POINTS_FOR_200DMA:
                    continue
                close = df_sym['close'].astype(float)
                volume = df_sym['volume'].astype(float)
                in_stage_2 = is_stage_2(close)
                breakout = is_breakout(close)
                momentum = is_stockbee_momentum(close, volume)
                latest_type = 'near-real-time' if appended else 'EOD'
                row = {
                    "Symbol": symbol,
                    "Price": float(close.iloc[-1]),
                    "LatestType": latest_type,
                    "Stage 2": in_stage_2,
                    "Breakout": breakout,
                    "Momentum": momentum,
                    "Latest Volume": int(volume.iloc[-1]),
                    "Avg Volume (prefilter 60d)": np.nan,
                    "MarketCap": st.session_state.marketcaps.get(symbol, np.nan)
                }
                # fill avg volume (if initial prefilter stats known, otherwise recompute quick)
                row["Avg Volume (prefilter 60d)"] = int(df_sym['volume'].tail(PREFILTER_DAYS).mean())
                if (in_stage_2 and breakout) or (in_stage_2 and momentum) or (breakout and momentum):
                    updated_results.append(row)
                if in_stage_2 and breakout and momentum:
                    updated_final.append(row)
            except Exception as e:
                st.write(f"Refresh error {symbol}: {e}")
                continue
        stat.empty()
        prog.empty()
        st.session_state.screener_df = pd.DataFrame(updated_results)
        st.session_state.final_df = pd.DataFrame(updated_final)
        st.success("Refresh complete. Signals updated with latest prices.")

# Display current results (from session_state)
st.subheader("🎯 WeinKulBee — Matches (any 2 of 3 criteria)")
if not st.session_state.screener_df.empty:
    # Show small sparkline + 180MA for each symbol row
    df_show = st.session_state.screener_df.copy().reset_index(drop=True)
    st.write(f"{len(df_show)} matches (from cached or refreshed run).")
    # We'll construct a single display with symbol, price, flags, marketcap, sparkline
    for idx, r in df_show.iterrows():
        sym = r['Symbol']
        cols = st.columns([1,1,1,1,2])  # symbol, price, flags, marketcap, sparkline
        cols[0].markdown(f"**{sym}**")
        cols[1].write(f"${r['Price']:.2f}")
        flags = ""
        flags += "S2 " if r.get('Stage 2') else ""
        flags += "B " if r.get('Breakout') else ""
        flags += "M " if r.get('Momentum') else ""
        cols[2].write(flags.strip())
        mc = r.get('MarketCap', np.nan)
        cols[3].write(f"${mc/1e9:.2f}B" if pd.notna(mc) else "N/A")
        # sparkline
        try:
            # prepare df from raw_full if available, else try yfinance quick fetch
            if st.session_state.raw_full is not None and isinstance(st.session_state.raw_full.columns, pd.MultiIndex) and sym in st.session_state.raw_full.columns.get_level_values(0):
                hist = st.session_state.raw_full[sym].dropna(subset=['Close']).copy()
                hist = hist[['Close']].rename(columns={'Close':'close'})
            else:
                # fallback small fetch
                quick = yf.download(tickers=sym, period="240d", interval="1d", progress=False)
                hist = quick[['Close']].rename(columns={'Close':'close'})
            hist = hist.sort_index()
            hist_close = hist['close'].astype(float)
            ma180 = hist_close.rolling(window=180, min_periods=1).mean()
            # create sparkline tiny chart
            fig, ax = plt.subplots(figsize=(4,0.9))
            ax.plot(hist_close.values[-60:], linewidth=0.9)  # last 60 days
            ax.plot(ma180.values[-60:], linestyle='--', linewidth=0.7)
            ax.axis('off')  # minimal sparkline (no axes)
            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches='tight', dpi=80)
            plt.close(fig)
            buf.seek(0)
            cols[4].image(buf)
        except Exception as e:
            cols[4].write("No chart")
    # offer CSV download
    st.download_button("Download matches CSV", data=st.session_state.screener_df.to_csv(index=False).encode('utf-8'),
                       file_name='weinkulbee_matches_realtime.csv', mime='text/csv')
else:
    st.write("No matches to show. Run full prefiler or refresh.")

st.subheader("🔥 WeinKulBee — Strong Matches (all 3 criteria)")
if not st.session_state.final_df.empty:
    st.dataframe(st.session_state.final_df.reset_index(drop=True))
    st.download_button("Download strong matches CSV", data=st.session_state.final_df.to_csv(index=False).encode('utf-8'),
                       file_name='weinkulbee_strong_realtime.csv', mime='text/csv')
else:
    st.write("No strong matches currently.")
