import streamlit as st
import pandas as pd
from bdshare import get_current_trade_data, get_hist_data, get_current_trading_code, get_market_inf_more_data
import datetime as dt
from datetime import datetime, timedelta, timezone
import numpy as np
from ta.trend import sma_indicator
from utils import init
init()

st.title("📈 WeinKulBee-Tani DSE Screener")


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


Start_Date = dt.datetime.now().date() - timedelta(days=600)
End_Date = dt.datetime.now().date()

# Step 1: Fetch real-time DSE data
live_data = get_current_trade_data()

# Optional filter to focus only on active stocks
live_data = live_data[pd.to_numeric(live_data['volume']) > 1000000]

# --- Configuration ---# 🔍 Watchlist
symbol_list = live_data['symbol'].unique()  # Update this list with DSE tickers
st.write(symbol_list)

# --- Results Container ---
results = []

final_result = []




# --- Strategy Functions ---
def is_stage_2(df):
    """Check if stock is in Stan Weinstein's Stage 2"""
    df['200dma'] = df['close'].rolling(window=180).mean()
    if pd.to_numeric(df['close'].iloc[-1]) > pd.to_numeric(df['200dma'].iloc[-1]) and pd.to_numeric(
            df['200dma'].iloc[-1]) > pd.to_numeric(df['200dma'].iloc[-20]):
        return True
    return False


def is_breakout(df):
    """Check if current close breaks past 50-day resistance"""
    recent_max = pd.to_numeric(df['close'].iloc[-49:-1].max())
    if pd.to_numeric(df['close'].iloc[-1]) > recent_max:
        return True
    return False


def is_stockbee_momentum(df):
    """StockBee momentum filter: price % change + volume spike"""
    df['volume_50ma'] = df['volume'].rolling(window=48).mean()
    pct_change = (pd.to_numeric(df['close'].iloc[-1]) - pd.to_numeric(df['close'].iloc[-2])) / pd.to_numeric(
        df['close'].iloc[-2])
    vol_spike = pd.to_numeric(df['volume'].iloc[-1]) > 1.5 * pd.to_numeric(df['volume_50ma'].iloc[-1])
    return pct_change > 0.04 and vol_spike


latest_iteration = st.empty()
prg = st.progress(0)
# --- Main Screening Loop ---
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

        if len(df) < 200:
            continue  # Not enough data for 200dma

        in_stage_2 = is_stage_2(df)
        breakout = is_breakout(df)
        momentum = is_stockbee_momentum(df)

        # if in_stage_2 or breakout or momentum:
        if (in_stage_2 and breakout) or (in_stage_2 and momentum) or (breakout and momentum):
            results.append({
                "Symbol": symbol,
                "Price": df['close'].iloc[-1],
                "Stage 2": in_stage_2,
                "Breakout": breakout,
                "Momentum": momentum
            })

        if (in_stage_2 and breakout and  momentum):
            final_result.append({
                "Symbol": symbol,
                "Price": df['close'].iloc[-1],
                "Stage 2": in_stage_2,
                "Breakout": breakout,
                "Momentum": momentum
            })
    except Exception as e:
        st.write(f"Error processing {symbol}: {e}")

# --- Output Result ---
screener_df = pd.DataFrame(results)
fscreener_df = pd.DataFrame(final_result)
st.write("\n==🎯 WeinKulBee Triple Strategy Screener Result 🎯==")
st.write(screener_df)
st.write('\n')
st.write(fscreener_df)
st.write('Hello world! WeinKulBee- Kibrul Bary Tani')
