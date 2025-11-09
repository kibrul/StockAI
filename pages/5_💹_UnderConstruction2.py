import streamlit as st
import pandas as pd
from bdshare import get_current_trade_data, get_hist_data, get_current_trading_code, get_market_inf_more_data
import datetime as dt
from datetime import datetime, timedelta, timezone
import numpy as np

from utils import init
init()

df_outstanding_share = pd.read_excel('h:\\stock_analysis\\all_company_info.xlsx',usecols=['symbol', 'free_float'])


def get_stock_price(Start_Date, End_Date, Symbol):
    df1 = pd.DataFrame(get_hist_data(Start_Date, End_Date, Symbol))
    df1['dates'] = pd.to_datetime(df1.index)
    df1['change'] = round(
        df1.apply(lambda x: pd.to_numeric(x['close'], downcast='float') - pd.to_numeric(x['ycp'], downcast='float'),
                  axis=1), 2)

    # UTC + 6:00 (Bangladesh time zone)
    currentHour = datetime.now(tz=timezone(timedelta(hours=6))).strftime("%H")
    if int(currentHour) >= 10 and int(currentHour) <= 14:  # Consider Trading Hours
        df2 = pd.DataFrame(get_current_trade_data(Symbol))
        df2['dates'] = pd.to_datetime(End_Date)
        df2['open'] = df2['ycp'].values
        df3 = pd.concat([df1, df2])
    else:
        df3 = df1

    df4 = get_market_inf_more_data(Start_Date, End_Date)  # get historical market data #,index = 'date'
    df4['Date'] = pd.to_datetime(df4['Date'], infer_datetime_format=True)
    df5 = df4[['Date', 'Total Volume']]
    df5 = df5.rename(columns={'Date': 'dates', 'Total Volume': 'TotalMRKTVolume'})

    if int(currentHour) >= 10 and int(currentHour) <= 14:  # Consider Trading Hours
        df6 = get_current_trade_data()
        df6['volume'] = pd.to_numeric(df6['volume'], errors='coerce')
        new_row = {'dates': pd.to_datetime(End_Date), 'TotalMRKTVolume': df6['volume'].sum()}
        # df7 = df5.append(new_row, ignore_index=True)
        df7 = pd.concat([df5, pd.DataFrame([new_row])])
    else:
        df7 = df5

    df8 = pd.merge(df3, df7, on='dates')

    df9 = pd.merge(df8, df_outstanding_share, on='symbol')

    # DSR stands for Demand to Supply Ratio. The DSR is a score, larger ratio of demand to supply for a equity market. The higher the DSR, the more demand exceeds supply. One of the most fundamental laws of economics is that prices rise when demand exceeds supply.
    # Share Turnover(DSR) = (Trading Volume /  Outstanding Shares(free_float))*100
    df9['dsr'] = round((pd.to_numeric(df9['volume']) / pd.to_numeric(df9['free_float'])) * 100, 3)

    df9 = df9.sort_values(by='dates', ascending=True)  # index='date'
    df9.index = range(len(df9.index))
    # volume breakout = (average volume + 40% of maximum volume)
    dfmean = df9.iloc[0:len(df9) - 1].copy()
    average_mean_volume = round(pd.to_numeric(dfmean['volume']).mean())
    breakout_volume = round(pd.to_numeric(dfmean['volume']).mean()) + round(
        pd.to_numeric(dfmean['volume']).max() * 0.40)

    df9['mean_value'] =  average_mean_volume

    for i in (range(0, len(df9))):
        if pd.to_numeric((df9.at[i, 'volume'])) >= average_mean_volume and pd.to_numeric((df9.at[i, 'volume'])) < breakout_volume:
            df9.at[i, 'v_status'] = 'Average'
        elif pd.to_numeric((df9.at[i, 'volume'])) >= breakout_volume:
            df9.at[i, 'v_status'] = 'BreakOut'
        else:
            df9.at[i, 'v_status'] = 'Normal'

    return df9


df_all_stock_name = get_current_trading_code()
start_dt = dt.datetime.now().date() - timedelta(days=8)
end_dt = dt.datetime.now().date()



# correlation of Pearson's coefficient
# consider indivisual stock volume against total market volume
# so that perticular specific stock contribution correlation to the hole market volume)
# for correlation time frame consider previous 7 wrokiding days data
# https://learn.robinhood.com/articles/7APpaAyA7UoOBXdVfBYkKj/what-is-correlation/

def get_stock_correlation_volume(Start_Date, End_Date, Symbol):
    # pd.to_datetime(startdate) + pd.DateOffset(days=5)
    # print (pd.to_datetime(Start_Date) - timedelta(days=30))

    date_difference = (pd.to_datetime(End_Date) - pd.to_datetime(Start_Date)).days

    df = get_stock_price(pd.to_datetime(Start_Date) - timedelta(days=date_difference * 3), End_Date, Symbol)

    df.index = range(len(df.index))

    df_row_count = len(df[df.dates.between(pd.to_datetime(Start_Date, format="%Y-%m-%d"),
                                           pd.to_datetime(End_Date, format="%Y-%m-%d"))])
    df_start_count = len(df) - df_row_count
    df_end_count = len(df) + 1  # Max_value
    # print(df_row_count)

    for i in reversed(range(df_start_count, df_end_count)):
        # print(i)
        df_crr = df.iloc[df_start_count:i].copy()
        # print(df_crr['dates'])
        df.at[i - 1, 'correlation'] = round(
            pd.to_numeric(df_crr['volume']).corr(pd.to_numeric(df_crr['TotalMRKTVolume'])), 3)
        df_start_count = df_start_count - 1

    df1 = df[df.dates.between(pd.to_datetime(Start_Date, format="%Y-%m-%d"), pd.to_datetime(End_Date, format="%Y-%m-%d"))]
    df1.index = range(len(df1.index))
    return df1

selectbox_Stock_Name = st.selectbox(f'Select Stock 👇', df_all_stock_name,index=323, key='all_stock_list')

col1, col2 = st.columns(2)
with col1:
    Start_Date = st.date_input("Start Date", start_dt)
with col2:
    End_Date = st.date_input("End Date", end_dt)

st.dataframe (get_stock_correlation_volume(Start_Date, End_Date, selectbox_Stock_Name))
