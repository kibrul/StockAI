import streamlit as st
import altair as alt
from utils import init
init()

import pandas as pd
from bdshare import get_current_trade_data, get_hist_data, get_current_trading_code, get_market_inf_more_data
import datetime as dt
from datetime import datetime, timedelta, timezone
import time

df_outstanding_share = pd.read_excel('D:\\stock_analysis\\all_company_info.xlsx',usecols=['symbol', 'free_float'])
df_market_cap = pd.read_excel('D:\\stock_analysis\\market_cap.xlsx',usecols=['symbol', 'mCap'])

#selected_mCap = st.selectbox(f'Select Market Cap 👇', ['Micro', 'Small','Mid','Large'],index=0, key='slist')
#df_market_cap = df_market_cap[df_market_cap['mCap'] == f"{selected_mCap}"]
symbols = df_market_cap['symbol'].tolist()


start_dt = dt.datetime.now().date() - timedelta(days=8)
end_dt = dt.datetime.now().date() - timedelta(days=1)

col1, col2 = st.columns(2)
with col1:
    Start_Date = st.date_input("Start Date", start_dt)
with col2:
    End_Date = st.date_input("End Date", end_dt)

latest_iteration = st.empty()
prg = st.progress(0)

Stock_DF = pd.DataFrame()

if st.button('Data Prepare For Mean DSR'):
    if symbols:
        for i in range(len(symbols)):
            latest_iteration.text(f'{len(symbols) - i} Items left')
            prg.progress(round((i/3.66)+1))
            #prg.progress((100/len(symbols))*i)
            df3 = pd.DataFrame(get_hist_data(start_dt,end_dt,symbols[i]))
            df4 = pd.merge(df3, df_outstanding_share, on='symbol')
            df4['dsr'] = round((pd.to_numeric(df4['volume']) / pd.to_numeric(df4['free_float'])) * 100, 2)
            df4['mean_dsr'] = round(pd.to_numeric(df4['dsr']).mean(),2)-round(pd.to_numeric(df4['dsr']).std() * 0.40,2)
            df5 = df4[['symbol','mean_dsr']]
            df6 = df5.groupby('symbol')['mean_dsr'].max()

            Stock_DF = pd.concat([Stock_DF, df6])



        #st.bar_chart(Stock_DF, x="symbol", y=["dsr","mean_dsr"])
        Stock_DF.to_excel( 'stock_dsr.xlsx')


if st.button('Data Prepare For Mean DSR - STD '):
    if symbols:
        for i in range(len(symbols)):
            latest_iteration.text(f'{len(symbols) - i} Items left')
            prg.progress(round((i/3.66)+1))
            #prg.progress((100/len(symbols))*i)
            df3 = pd.DataFrame(get_hist_data(start_dt,end_dt,symbols[i]))
            df4 = pd.merge(df3, df_outstanding_share, on='symbol')
            df4['dsr'] = round((pd.to_numeric(df4['volume']) / pd.to_numeric(df4['free_float'])) * 100, 2)
            df4['mean_dsr'] = round(pd.to_numeric(df4['dsr']).mean(),2)-round(pd.to_numeric(df4['dsr']).std(),2)
            df5 = df4[['symbol','mean_dsr']]
            df6 = df5.groupby('symbol')['mean_dsr'].max()

            Stock_DF = pd.concat([Stock_DF, df6])



        #st.bar_chart(Stock_DF, x="symbol", y=["dsr","mean_dsr"])
        Stock_DF.to_excel( 'stock_dsr_std.xlsx')
