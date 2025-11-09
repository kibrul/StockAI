import streamlit as st
from utils import init

import pandas as pd
from bdshare import get_current_trade_data, get_hist_data, get_current_trading_code, get_market_inf_more_data
import datetime as dt
from datetime import datetime, timedelta, timezone
import numpy as np
init()

st.title("📈 Breakout-Tani DSE Screener")

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

breakout = []
latest_iteration = st.empty()
prg = st.progress(0)

for idx, symbol in enumerate(symbol_list):
    latest_iteration.text(f'{symbol} Items left {len(symbol_list) - (idx + 1)}')
    prg.progress((idx + 1) / len(symbol_list))
    
    try:
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
        db_today_close = pd.to_numeric(df['close'].iloc[-2])
        db_yesterday_close = pd.to_numeric(df['close'].iloc[-3])
        pct_change = ((db_today_close - db_yesterday_close) / db_yesterday_close) * 100
        flat_condition = -1 <= pct_change <= 1

        #Bullish 4% Break Out
        today_close = pd.to_numeric(df['close'].iloc[-1])
        yesterday_close = pd.to_numeric(df['close'].iloc[-2])
        bullish4 = (today_close/yesterday_close) >= 1.04
        
        
        #if bullish4 and ratio_condition and flat_condition:
        if bullish4 and ratio_condition :
                breakout.append({
                    'Symbol': symbol,
                    '9d/66d Ratio': round(avg9 / avg66, 3),
                    '% Change Today': round(pct_change, 2),
                    'Last Price': today_close
                })
    except Exception as e:
         st.write(f"Error processing {symbol}: {e}")

# --- Output Result ---
watchlist = pd.DataFrame(breakout)
if len(watchlist) > 0:
     st.write("📊 Breakout Watchlist (Breakout After Consolidating):")
     st.write(watchlist)
else:
     st.write("⚠️ No stocks met the filter criteria today.")

"""
stock should have range expansion on breakout day
volume on breakout day should be higher
day before breakout should be narrow range day or negative day
pre breakout there should not be many 4% breakdowns
stock should have linearity in prior action
correction or consolidation should be orderly
volume during consolidation should be preferably orderly
stock should close near high on breakout day
A good momentum burst candidate will have following characters:
The day prior to range expansion day will be narrow range day or negative day
The stock will have 3 to 20 days consolidation prior to range expansion day
The stock will have series of narrow range days prior to breakout
On breakout day volume is higher than previous day
On breakout day stock closes at or near its high for the day (preferred)
Stock is not extended. First or second breakout at start of an up trend is preferred. 
Stock should have linear and orderly move
A very volatile stock exhibiting drunken man walk kind moves should be avoided
Low float below 25 million is good. Below 10 million float leads to explosive moves
Low priced stocks (below 5 dollar) tend to make very explosive moves of 40% kind in 3 to 5 days. 


2 not up 2 days in a row Asmall up day before b/o is fine.

L linearity of prior move

Y young trend 1 to 3 rd b/o from consolidation is low risk. As trend ages risk of failure increases.

N narrow range day or negative day pre breakout

C consolidation/pullback is shallow, orderly and compact with narrow range bars and low volume . no more than one 4% b/d in consolidation

H close near high of the day
"""
