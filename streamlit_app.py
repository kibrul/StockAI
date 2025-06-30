import streamlit as st
import pandas as pd
from bdshare import get_current_trade_data, get_hist_data, get_current_trading_code, get_market_inf_more_data
import datetime as dt
from datetime import datetime, timedelta, timezone
import numpy as np
from ta.trend import sma_indicator

st.title("📈 WeinKulBee-Tani DSE Screener")

def get_stock_price(Start_Date,End_Date,Symbol):
    df1 = pd.DataFrame(get_hist_data(Start_Date,End_Date,Symbol))

    # Reset the index date
    df = df1.reset_index()

    # 🔹 Ensure 'close' column exists and is sorted by date
    #df = df[['date', 'close']].sort_values('date')
    df.set_index('date', inplace=True)

    df['change'] = round(df1.apply(lambda x: pd.to_numeric(x['close'],downcast='float') -  pd.to_numeric(x['ycp'],downcast='float'), axis=1),2)  

    # UTC + 6:00 (Bangladesh time zone) 
    currentHour = datetime.now(tz=timezone(timedelta(hours=6))).strftime("%H")
    if int(currentHour)>=10 and int(currentHour) <=15: #Consider Trading Hours
        df2 = pd.DataFrame(get_current_trade_data(Symbol))
        df2['date'] = pd.to_datetime(End_Date)
        df2['open'] = df2['ycp'].values
        df3 = pd.concat([df, df2])
    else:
        df3 = df
        
    return df3
  
Start_Date = dt.datetime.now().date() - timedelta(days=600)
End_Date = dt.datetime.now().date()

# Step 1: Fetch real-time DSE data
live_data = get_current_trade_data()

# Optional filter to focus only on active stocks
live_data = live_data[pd.to_numeric(live_data['volume']) > 1000000]



# 🔍 Watchlist
watchlist = live_data['symbol'].unique()
#watchlist = ['CENTRALINS','LOVELLO','MJLBD','BSC','KBPPWBIL','LOVELLO','MALEKSPIN','SEAPEARL','CITYBANK','SQURPHARMA','BRACBANK','FINEFOODS']

#watchlist = ['CENTRALINS']
st.write(watchlist)
# 📦 Result container
results = []

# ⚙️ Screener logic per stock
for symbol in watchlist:
    try:
        #print(symbol)
        #df3 = get_hist_data(Start_Date,End_Date,symbol)
        df3 = get_stock_price(Start_Date,End_Date,symbol)
        # Reset the index date
        df = df3.reset_index()
        df = df[['date', 'close', 'volume']].sort_values('date')
        df.set_index('date', inplace=True)
    
        # 📈 Moving Averages
        df['MA_50'] = sma_indicator(df['close'], window=48)
        df['MA_150'] = sma_indicator(df['close'], window=180)

        # 🎯 Add 50-day high for Kristjan Breakout
        df['50d_high'] = df['close'].rolling(window=50).max()

        #print(df.tail(10))

        # 📌 Basic checks
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        slope_check = df['MA_150'][-5:].is_monotonic_increasing
    
        # 📊 Stan Weinstein Stage 2
        is_stage_2 = (
            pd.to_numeric(latest['close']) > pd.to_numeric(latest['MA_150']) and
            pd.to_numeric(slope_check)
        )

        # 💥 Kristjan Breakout
        rs_rank = pd.to_numeric(latest['close'])/pd.to_numeric(latest['50d_high'])
        is_breakout = rs_rank >= 0.99 and pd.to_numeric(latest['close']) > pd.to_numeric(prev['close'])
    
        # ⚡ StockBee Momentum
        df['range'] = pd.to_numeric(df['close']) - pd.to_numeric(df['close'].shift(1))
        nr7 = df['range'].rolling(window=7).apply(lambda x: x[-1] == min(x), raw=True)
        is_nr7 = nr7.iloc[-1] == 1
        price_jump = (pd.to_numeric(latest['close']) - pd.to_numeric(prev['close'])) / pd.to_numeric(prev['close']) >= 0.04
        is_stockbee = is_nr7 and price_jump

        # 📋 Append results
        if (is_stage_2 and is_breakout) or (is_stage_2 and is_stockbee) or (is_breakout and is_stockbee):
         st.write('Findout:', symbol)   
         results.append({
            'symbol': symbol,
            'close': round(pd.to_numeric(latest['close']), 2),
            'MA_50': round(pd.to_numeric(latest['MA_50']), 2),
            'MA_150': round(pd.to_numeric(latest['MA_150']), 2),
            'Stage_2': is_stage_2,
            'Kristjan_Breakout': is_breakout,
            'StockBee_Momentum': is_stockbee,
            'WeinKulBee_Triple': is_stage_2 and is_breakout and is_stockbee
         })
    except Exception as e:
        st.write(f"⚠️ Error with {symbol}: {e}")

# 📊 Show results
df_screen = pd.DataFrame(results)
df_screen = df_screen.sort_values('WeinKulBee_Triple', ascending=False)
st.write("\n🎯 WeinKulBee Screener Output:\n")
st.write(df_screen)

st.write('Hello world! Kibrul')
