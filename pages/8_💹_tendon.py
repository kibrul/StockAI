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


st.title("📈 WeinKulBee Dynamic Parabola Screener (DSE via bdshare)")
st.write("""
This screener identifies DSE stocks showing a **3-phase pattern**:
1️⃣ **Uptrend** — sustained higher highs  
2️⃣ **Sideways / Consolidation** — range-bound phase  
3️⃣ **Parabolic Curve** — arch or topping parabola  
""")



# Step 1: Fetch real-time DSE data
live_data = get_current_trade_data()
# Optional filter to focus only on active stocks
live_data = live_data[pd.to_numeric(live_data['volume']) > 100000]

# --- Configuration ---# 🔍 Watchlist
watchlist = live_data['symbol'].unique()  # Update this list with DSE tickers


# --- Default Watchlist ---
#watchlist = ['MJLBD', 'BSC', 'KBPPWBIL', 'LOVELLO', 'MALEKSPIN',  'SEAPEARL', 'CITYBANK', 'SQURPHARMA', 'BRACBANK', 'FINEFOODS']


# --- Date Range ---
start_date = st.date_input("Start Date", datetime.today() - timedelta(days=365))
end_date = st.date_input("End Date", datetime.today())


def get_stock_price(Start_Date,End_Date,Symbol):
    df1 = pd.DataFrame(get_hist_data(Start_Date,End_Date,Symbol))
    df = df1.reset_index()
    df = df[['date','symbol', 'close','volume']]
    df.set_index('date', inplace=False)

    # UTC + 6:00 (Bangladesh time zone)
    currentHour = datetime.now(tz=timezone(timedelta(hours=6))).strftime("%H")
    if int(currentHour)>=10 and int(currentHour) <=15: #Consider Trading Hours
        df2 = pd.DataFrame(get_current_trade_data(Symbol))
        df2['date'] = pd.to_datetime('today').date()
        df2['close'] = df2['ltp'].values
        df3 = df2[['date','symbol', 'close','volume']]
        df4 = pd.concat([df, df3])
        df4['date'] = pd.to_datetime(df4['date'])
        df6 = df4.sort_values('date', ascending=False)
    else:
        df6 = df
    return df6

# -------------------- FUNCTIONS --------------------
def load_dse_data(symbol, start, end):
    """Load historical DSE data using bdshare"""
    try:
        df3 = get_stock_price(start, end, symbol)

        if df3 is None or df3.empty:
            return pd.DataFrame()

        df = df3.reset_index()
        df = df[['date', 'symbol', 'close', 'volume']].sort_values('date')
        df.set_index('date', inplace=True)

        return df
    except Exception:
        return pd.DataFrame()

def slope_of(prices, window=20):
    """Calculate rolling regression slope"""
    slopes = []
    for i in range(len(prices)):
        if i < window:
            slopes.append(0)
        else:
            y = prices[i - window:i]
            X = np.arange(window).reshape(-1, 1)
            model = LinearRegression().fit(X, y)
            slopes.append(model.coef_[0])
    return np.array(slopes)

def parabola(x, a, b, c):
    return a * (x - b) ** 2 + c

def detect_parabolic_top(df):
    """Detect uptrend → sideways → parabolic top"""
    if df.empty or len(df) < 50:
        return None
    df['slope'] = slope_of(pd.to_numeric(df['close'].values))
    prices = pd.to_numeric(df['close'].values)
    slopes = df['slope'].values

    # Detect phases
    uptrend_zone = np.where(slopes > np.percentile(slopes, 75))[0]
    flat_zone = np.where((slopes < np.percentile(slopes, 60)) & (slopes > np.percentile(slopes, 40)))[0]

    if len(uptrend_zone) == 0 or len(flat_zone) == 0:
        return None

    start_idx = max(uptrend_zone[-1] - 30, 0)
    end_idx = len(prices)
    x_seg = np.arange(end_idx - start_idx)
    y_seg = prices[start_idx:end_idx]

    try:
        popt, _ = curve_fit(parabola, x_seg, y_seg)
        a, b, c = popt
        # Must be downward parabola (arch)
        if a < 0:
            fit = parabola(x_seg, *popt)
            curvature = abs(a)
            return {
                'start': df.index[start_idx],
                'end': df.index[end_idx - 1],
                'a': a,
                'b': b,
                'c': c,
                'curvature': curvature,
                'fit': fit,
                'price_start': y_seg[0],
                'price_end': y_seg[-1]
            }
    except Exception:
        pass
    return None

# -------------------- MAIN LOOP --------------------
results = []
for symbol in watchlist:
    df = load_dse_data(symbol, str(start_date), str(end_date))
    pattern = detect_parabolic_top(df)
    if pattern:
        results.append({
            'Symbol': symbol,
            'Start': pattern['start'],
            'End': pattern['end'],
            'Curvature': round(pattern['curvature'], 6),
            'Start Price': round(pattern['price_start'], 2),
            'End Price': round(pattern['price_end'], 2),
            'Trend': "🔻 Parabolic Top"
        })
    else:
        results.append({
            'Symbol': symbol,
            'Start': '-',
            'End': '-',
            'Curvature': '-',
            'Start Price': '-',
            'End Price': '-',
            'Trend': '⚪ Neutral / No Parabola'
        })

results_df = pd.DataFrame(results)

# -------------------- DISPLAY --------------------
st.subheader("📊 Screener Results (WeinKulBee Dynamic Parabola)")
st.dataframe(results_df, use_container_width=True)

# -------------------- CHART SECTION --------------------
selected_symbol = st.selectbox("Select symbol to view chart:", watchlist)
df_selected = load_dse_data(selected_symbol, str(start_date), str(end_date))
pattern = detect_parabolic_top(df_selected)

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df_selected.index, df_selected['close'], label='close', color='blue')

if pattern:
    start, end = pattern['start'], pattern['end']
    segment = df_selected.loc[start:end]
    ax.plot(segment.index, pattern['fit'], 'r--', linewidth=2, label='Parabola Fit')
    ax.axvspan(start, end, color='red', alpha=0.15, label='Parabolic Zone')
    st.success(f"✅ {selected_symbol}: Parabolic Top between {start} → {end}")
else:
    st.warning(f"No parabolic top detected for {selected_symbol}.")

ax.set_title(f"{selected_symbol} — Price with Dynamic Parabola Detection")
ax.legend()
st.pyplot(fig)

# -------------------- RAW DATA --------------------
with st.expander("🔍 View Recent Data"):
    st.dataframe(df_selected.tail(30))