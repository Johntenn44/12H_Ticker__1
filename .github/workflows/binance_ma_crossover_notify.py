import os
import ccxt
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime
import traceback

# --- TELEGRAM CONFIGURATION ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    raise ValueError("Telegram bot token or chat ID not set in environment variables.")

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML"
    }
    try:
        resp = requests.post(url, data=payload)
        resp.raise_for_status()
        print(f"Telegram message sent: {message}")
    except Exception as e:
        print(f"Failed to send Telegram message: {e}")

# --- FETCH OHLCV DATA ---

def fetch_latest_ohlcv(symbol='EUR/USD', timeframe='15m', limit=100):
    try:
        exchange = ccxt.kraken()
        exchange.load_markets()
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df.astype(float)
    except Exception as e:
        print(f"Error fetching OHLCV data: {e}")
        traceback.print_exc()
        return None

# --- INDICATOR CALCULATIONS ---

def calculate_rsi(series, period=13):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period-1, adjust=False).mean()
    avg_loss = loss.ewm(com=period-1, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_stoch_rsi(df, rsi_len=13, stoch_len=8, smooth_k=5, smooth_d=3):
    rsi = calculate_rsi(df['close'], rsi_len)
    min_rsi = rsi.rolling(window=stoch_len).min()
    max_rsi = rsi.rolling(window=stoch_len).max()
    denom = max_rsi - min_rsi
    denom = denom.replace(0, np.nan)
    stoch_rsi = (rsi - min_rsi) / denom * 100
    stoch_rsi = stoch_rsi.fillna(method='ffill')
    k = stoch_rsi.rolling(window=smooth_k).mean()
    d = k.rolling(window=smooth_d).mean()
    return k, d

def calculate_wr(df, length=14):
    highest_high = df['high'].rolling(window=length).max()
    lowest_low = df['low'].rolling(window=length).min()
    denom = highest_high - lowest_low
    denom = denom.replace(0, np.nan)
    wr = (highest_high - df['close']) / denom * -100
    return wr.fillna(method='ffill')

def calculate_kdj(df, length=5, ma1=8, ma2=8):
    low_min = df['low'].rolling(window=length, min_periods=1).min()
    high_max = df['high'].rolling(window=length, min_periods=1).max()
    rsv = (df['close'] - low_min) / (high_max - low_min) * 100
    k = rsv.ewm(span=ma1, adjust=False).mean()
    d = k.ewm(span=ma2, adjust=False).mean()
    j = 3 * k - 2 * d
    return k, d, j

# --- TREND ANALYSIS FUNCTIONS ---

def analyze_stoch_rsi_trend(k, d):
    if len(k) < 2 or pd.isna(k.iloc[-2]) or pd.isna(d.iloc[-2]) or pd.isna(k.iloc[-1]) or pd.isna(d.iloc[-1]):
        return None
    if k.iloc[-2] < d.iloc[-2] and k.iloc[-1] > d.iloc[-1] and k.iloc[-1] < 80:
        return "up"
    elif k.iloc[-2] > d.iloc[-2] and k.iloc[-1] < d.iloc[-1] and k.iloc[-1] > 20:
        return "down"
    else:
        return None

def analyze_wr_trend(wr):
    if len(wr) < 2 or pd.isna(wr.iloc[-2]) or pd.isna(wr.iloc[-1]):
        return None
    prev, curr = wr.iloc[-2], wr.iloc[-1]
    if prev > -80 and curr <= -80:
        return "up"
    elif prev < -20 and curr >= -20:
        return "down"
    else:
        return None

def analyze_rsi_trend(rsi8, rsi13, rsi21):
    if rsi8 > rsi13 > rsi21:
        return "up"
    elif rsi8 < rsi13 < rsi21:
        return "down"
    else:
        return None

def analyze_kdj_trend(k, d, j):
    if len(k) < 2 or len(d) < 2 or len(j) < 2:
        return None
    k_prev, k_curr = k.iloc[-2], k.iloc[-1]
    d_prev, d_curr = d.iloc[-2], d.iloc[-1]
    j_prev, j_curr = j.iloc[-2], j.iloc[-1]
    if k_prev < d_prev and k_curr > d_curr and j_curr > k_curr and j_curr > d_curr:
        return "up"
    elif k_prev > d_prev and k_curr < d_curr and j_curr < k_curr and j_curr < d_curr:
        return "down"
    else:
        return None

# --- SIGNAL CHECK ---

def check_signal(df):
    k, d = calculate_stoch_rsi(df)
    wr = calculate_wr(df)
    rsi8 = calculate_rsi(df['close'], 8).iloc[-1]
    rsi13 = calculate_rsi(df['close'], 13).iloc[-1]
    rsi21 = calculate_rsi(df['close'], 21).iloc[-1]
    kdj_k, kdj_d, kdj_j = calculate_kdj(df)

    stoch_trend = analyze_stoch_rsi_trend(k, d)
    wr_trend = analyze_wr_trend(wr)
    rsi_trend = analyze_rsi_trend(rsi8, rsi13, rsi21)
    kdj_trend = analyze_kdj_trend(kdj_k, kdj_d, kdj_j)

    # Aggregate signals: buy if majority "up", sell if majority "down"
    signals = [stoch_trend, wr_trend, rsi_trend, kdj_trend]
    up_signals = signals.count("up")
    down_signals = signals.count("down")

    if up_signals > down_signals:
        return "buy"
    elif down_signals > up_signals:
        return "sell"
    else:
        return None

# --- MAIN LOOP ---

def main():
    while True:
        print(f"Checking signals at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        df = fetch_latest_ohlcv()
        if df is None or df.empty:
            print("No data fetched, retrying in 5 minutes...")
            time.sleep(300)
            continue

        signal = check_signal(df)
        if signal == "buy":
            last_close_time = df.index[-1].strftime('%Y-%m-%d %H:%M UTC')
            message = f"<b>Buy Signal Detected for EUR/USD</b>\nTime: {last_close_time}\nIndicators aligned for buy."
            send_telegram_message(message)
        elif signal == "sell":
            last_close_time = df.index[-1].strftime('%Y-%m-%d %H:%M UTC')
            message = f"<b>Sell Signal Detected for EUR/USD</b>\nTime: {last_close_time}\nIndicators aligned for sell."
            send_telegram_message(message)
        else:
            print("No clear buy or sell signal detected.")

        time.sleep(300)  # wait 5 minutes before next check

if __name__ == "__main__":
    main()
