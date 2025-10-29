import bdshare
import streamlit as st
import pandas as pd
from bdshare import get_current_trade_data, get_hist_data, get_current_trading_code, get_market_inf_more_data
import datetime as dt
from datetime import datetime, timedelta, timezone
import numpy as np
from ta.trend import sma_indicator
from utils import init
init()

import matplotlib.pyplot as plt


st.title("📈 Wyckoff accumulation → markup → distribution → markdown DSE Screener")

# ---------------------------
# Step 1: Load data
# ---------------------------
df = bdshare.get_hist_data( "2023-01-01", "2025-01-01","SEAPEARL")
df.reset_index(inplace=True)
df = df[["date", "close", "volume"]]
df["date"] = pd.to_datetime(df["date"])
df.set_index("date", inplace=True)
print(df.tail(10))


# ✅ Ensure numeric types
df["close"] = pd.to_numeric(df["close"], errors="coerce")
df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
# ---------------------------
# Step 2: Wyckoff Phase Detection
# ---------------------------
def detect_wyckoff_phase(df, window=20):
    df["returns"] = df["close"].pct_change()
    df["vol_ma"] = df["volume"].rolling(window).mean()
    df["price_ma"] = df["close"].rolling(window).mean()
    df["volatility"] = df["returns"].rolling(window).std()

    conditions = []
    signals = []

    for i in range(len(df)):
        phase = "Neutral"
        signal = None

        if df["volatility"].iloc[i] < df["volatility"].mean() * 0.7 and df["volume"].iloc[i] > df["vol_ma"].iloc[i]:
            phase = "Accumulation"
        elif df["close"].iloc[i] > df["price_ma"].iloc[i] * 1.05:
            phase = "Markup"
        elif df["volatility"].iloc[i] < df["volatility"].mean() * 0.7 and df["volume"].iloc[i] > df["vol_ma"].iloc[i] and df["close"].iloc[i] > df["price_ma"].iloc[i]:
            phase = "Distribution"
        elif df["close"].iloc[i] < df["price_ma"].iloc[i] * 0.95:
            phase = "Markdown"

        # ---------------------------
        # Buy/Sell Rules
        # ---------------------------
        if phase == "Accumulation" and df["close"].iloc[i] > df["price_ma"].iloc[i]:
            signal = "BUY"
        elif phase == "Distribution" or phase == "Markdown":
            signal = "SELL"

        conditions.append(phase)
        signals.append(signal)

    df["Wyckoff_Phase"] = conditions
    df["Signal"] = signals
    return df

df = detect_wyckoff_phase(df)

# ---------------------------
# Step 3: Visualization
# ---------------------------
phase_colors = {
    "Accumulation": "green",
    "Markup": "blue",
    "Distribution": "orange",
    "Markdown": "red",
    "Neutral": "gray"
}

plt.figure(figsize=(14,7))
plt.plot(df.index, df["close"], label="Close Price", color="black")

# Plot Wyckoff phases
for phase, color in phase_colors.items():
    mask = df["Wyckoff_Phase"] == phase
    plt.scatter(df.index[mask], df["close"][mask], label=phase, s=20, c=color)

# Plot Buy/Sell signals
buy_mask = df["Signal"] == "BUY"
sell_mask = df["Signal"] == "SELL"
plt.scatter(df.index[buy_mask], df["close"][buy_mask], marker="^", color="lime", s=100, label="BUY Signal")
plt.scatter(df.index[sell_mask], df["close"][sell_mask], marker="v", color="red", s=100, label="SELL Signal")

plt.title("Wyckoff Model with Buy/Sell Signals (SEAPEARL)")
plt.legend()
plt.show()

# ---------------------------
# Step 4: Show last signals
# ---------------------------
print(df.tail(10)[["close", "Wyckoff_Phase", "Signal"]])