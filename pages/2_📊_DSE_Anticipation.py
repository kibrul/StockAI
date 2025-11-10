import streamlit as st
from utils import init

import pandas as pd
from bdshare import get_current_trade_data, get_hist_data, get_current_trading_code, get_market_inf_more_data
import datetime as dt
from datetime import datetime, timedelta, timezone
import numpy as np
init()

st.title("📈 Anticipation-Tani DSE Screener")

def get_stock_price(Start_Date, End_Date, Symbol):
    df1 = pd.DataFrame(get_hist_data(Start_Date, End_Date, Symbol))
    df = df1.reset_index()
    df = df[['date', 'symbol', 'close', 'volume']]
    df.set_index('date', inplace=False)

    # UTC + 6:00 (Bangladesh time zone)
    currentHour = datetime.now(tz=timezone(timedelta(hours=6))).strftime("%H")
    if int(currentHour) >= 10 and int(currentHour) <= 15:  # Consider Trading Hours
        df2 = pd.DataFrame(get_current_trade_data(Symbol))
        df2['date'] = pd.to_datetime('today').date()
        df2['close'] = df2['ltp'].values
        df3 = df2[['date', 'symbol', 'close', 'volume']]
        df4 = pd.concat([df, df3])
        df4['date'] = pd.to_datetime(df4['date'])
        df6 = df4.sort_values('date', ascending=False)
    else:
        df6 = df
    return df6


Start_Date = dt.datetime.now().date() - timedelta(days=300)
End_Date = dt.datetime.now().date()

# Step 1: Fetch real-time DSE data
live_data = get_current_trade_data()

# Optional filter to focus only on active stocks
live_data = live_data[pd.to_numeric(live_data['volume']) > 600000]

# --- Configuration ---# 🔍 Watchlist
symbol_list = live_data['symbol'].unique()  # Update this list with DSE tickers
#print(symbol_list)
st.write(symbol_list)
anticipation = []
latest_iteration = st.empty()
prg = st.progress(0)

for idx, symbol in enumerate(symbol_list):
    latest_iteration.text(f'{symbol} Items left {len(symbol_list) - (idx + 1)}')
    prg.progress((idx + 1) / len(symbol_list))
    
    try:
        # df = get_hist_data(symbol, start_date=str(today - datetime.timedelta(days=days_of_history)))
        # df.dropna(inplace=True)
        # df.reset_index(drop=True, inplace=True)
        df3 = get_stock_price(Start_Date, End_Date, symbol)
        # Reset the index date
        df = df3.reset_index()
        df = df[['date', 'symbol', 'close', 'volume']].sort_values('date')
        df.set_index('date', inplace=True)

        if len(df) < 66:
            continue  # Not enough data for 66dma

        # Moving averages
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        avg9 = df['close'].tail(6).mean()
        avg66 = df['close'].tail(66).mean()

        # Condition 1: Short-term > long-term by 5%
        ratio_condition = (avg9 / avg66) > 1.06
    
        # Condition 2: Today's % change between -1% and +1%
        today_close = pd.to_numeric(df['close'].iloc[-1])
        yesterday_close = pd.to_numeric(df['close'].iloc[-2])
        pct_change = ((today_close - yesterday_close) / yesterday_close) * 100
        flat_condition = -1 <= pct_change <= 1

        if ratio_condition and flat_condition:
                anticipation.append({
                    'Symbol': symbol,
                    '9d/66d Ratio': round(avg9 / avg66, 3),
                    '% Change Today': round(pct_change, 2),
                    'Last Price': today_close
                })
    except Exception as e:
         st.write(f"Error processing {symbol}: {e}")

# --- Output Result ---
watchlist = pd.DataFrame(anticipation)
if len(watchlist) > 0:
     st.write("📊 Anticipation Watchlist (Consolidating after strength):")
     st.write(watchlist)
else:
     st.write("⚠️ No stocks met the filter criteria today.")


