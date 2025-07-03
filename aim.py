import ccxt
import pandas as pd
import numpy as np
import os
import time
import requests
from datetime import datetime

# --- Telegram notification ---
def send_telegram_message(message):
    TELEGRAM_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
    TELEGRAM_CHANNEL_ID = os.environ.get('TELEGRAM_CHANNEL_ID')
    if not TELEGRAM_TOKEN or not TELEGRAM_CHANNEL_ID:
        print("Telegram bot token or channel ID missing!")
        return
    url = f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage'
    payload = {
        'chat_id': TELEGRAM_CHANNEL_ID,
        'text': message,
        'parse_mode': 'HTML'
    }
    try:
        response = requests.post(url, data=payload, timeout=10)
        if not response.ok:
            print(f"Failed to send message: {response.text}")
    except Exception as e:
        print(f"Error sending Telegram message: {e}")

# --- Indicator functions ---

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

def calculate_multi_wr(df, lengths=[3, 13, 144, 8, 233, 55]):
    wr_dict = {}
    for length in lengths:
        highest_high = df['high'].rolling(window=length).max()
        lowest_low = df['low'].rolling(window=length).min()
        denom = highest_high - lowest_low
        denom = denom.replace(0, np.nan)
        wr = (highest_high - df['close']) / denom * -100
        wr_dict[length] = wr.fillna(method='ffill')
    return wr_dict

def analyze_wr_relative_positions(wr_dict):
    try:
        wr_8 = wr_dict[8].iloc[-1]
        wr_3 = wr_dict[3].iloc[-1]
        wr_144 = wr_dict[144].iloc[-1]
        wr_233 = wr_dict[233].iloc[-1]
        wr_55 = wr_dict[55].iloc[-1]
    except (KeyError, IndexError):
        return None

    if wr_8 > wr_233 and wr_3 > wr_233 and wr_8 > wr_144 and wr_3 > wr_144:
        return "up"
    elif wr_8 < wr_55 and wr_3 < wr_55:
        return "down"
    else:
        return None

def calculate_kdj(df, length=5, ma1=8, ma2=8):
    low_min = df['low'].rolling(window=length, min_periods=1).min()
    high_max = df['high'].rolling(window=length, min_periods=1).max()
    rsv = (df['close'] - low_min) / (high_max - low_min) * 100
    k = rsv.ewm(span=ma1, adjust=False).mean()
    d = k.ewm(span=ma2, adjust=False).mean()
    j = 3 * k - 2 * d
    return k, d, j

def analyze_stoch_rsi_trend(k, d):
    if len(k) < 2 or pd.isna(k.iloc[-2]) or pd.isna(d.iloc[-2]) or pd.isna(k.iloc[-1]) or pd.isna(d.iloc[-1]):
        return None
    if k.iloc[-2] < d.iloc[-2] and k.iloc[-1] > d.iloc[-1] and k.iloc[-1] < 80:
        return "up"
    elif k.iloc[-2] > d.iloc[-2] and k.iloc[-1] < d.iloc[-1] and k.iloc[-1] > 20:
        return "down"
    else:
        return None

def analyze_rsi_trend(rsi5, rsi13, rsi21):
    if rsi5 > rsi13 > rsi21:
        return "up"
    elif rsi5 < rsi13 < rsi21:
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

def calculate_macd(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line

def calculate_bollinger_bands(close, window=20, num_std=2):
    sma = close.rolling(window=window).mean()
    std = close.rolling(window=window).std()
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    return upper_band, lower_band

def check_signal(df):
    k, d = calculate_stoch_rsi(df)
    wr_dict = calculate_multi_wr(df)
    wr_trend = analyze_wr_relative_positions(wr_dict)

    rsi5 = calculate_rsi(df['close'], 5).iloc[-1]
    rsi13 = calculate_rsi(df['close'], 13).iloc[-1]
    rsi21 = calculate_rsi(df['close'], 21).iloc[-1]
    kdj_k, kdj_d, kdj_j = calculate_kdj(df)

    stoch_trend = analyze_stoch_rsi_trend(k, d)
    rsi_trend = analyze_rsi_trend(rsi5, rsi13, rsi21)
    kdj_trend = analyze_kdj_trend(kdj_k, kdj_d, kdj_j)

    signals = [stoch_trend, wr_trend, rsi_trend, kdj_trend]

    macd_line, macd_signal = calculate_macd(df['close'])
    macd_trend = None
    if len(macd_line) > 1 and macd_line.iloc[-2] < macd_signal.iloc[-2] and macd_line.iloc[-1] > macd_signal.iloc[-1]:
        macd_trend = "up"
    elif len(macd_line) > 1 and macd_line.iloc[-2] > macd_signal.iloc[-2] and macd_line.iloc[-1] < macd_signal.iloc[-1]:
        macd_trend = "down"
    if macd_trend:
        signals.append(macd_trend)

    upper_band, lower_band = calculate_bollinger_bands(df['close'])
    price = df['close'].iloc[-1]
    bb_trend = None
    if price < lower_band.iloc[-1]:
        bb_trend = "up"
    elif price > upper_band.iloc[-1]:
        bb_trend = "down"
    if bb_trend:
        signals.append(bb_trend)

    up_signals = signals.count("up")
    down_signals = signals.count("down")

    if up_signals > down_signals:
        return "buy"
    elif down_signals > up_signals:
        return "sell"
    else:
        return None

def fetch_latest_ohlcv(symbol='EUR/USD', timeframe='15m', limit=100):
    try:
        exchange = ccxt.kraken()
        exchange.load_markets()
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df.astype(float)
    except Exception as e:
        print(f"Error fetching OHLCV data: {e}")
        return None

def main():
    while True:
        now = datetime.utcnow()
        if now.minute % 3 == 0:
            print(f"Checking signals at {now.strftime('%Y-%m-%d %H:%M:%S UTC')}")
            df = fetch_latest_ohlcv()
            if df is None or df.empty:
                print("No data fetched, retrying in 5 minutes...")
                time.sleep(300)
                continue

            # Skip signals outside 6:00 - 22:59 UTC
            last_time = df.index[-1]
            if last_time.hour >= 23 or last_time.hour < 6:
                print("Signal detected outside trading hours (11 PM - 6 AM). Skipping.")
                time.sleep(60)
                continue

            signal = check_signal(df)
            if signal == "buy":
                last_close_time = df.index[-1].strftime('%Y-%m-%d %H:%M UTC')
                message = f"🚀 <b>Buy (IV) Signal Detected for EUR/USD</b>\n🕒 Time: {last_close_time}\n✅ Majority indicators aligned for buy."
                send_telegram_message(message)
                print(message)
            elif signal == "sell":
                last_close_time = df.index[-1].strftime('%Y-%m-%d %H:%M UTC')
                message = f"🔥 <b>Sell (IV) Signal Detected for EUR/USD</b>\n🕒 Time: {last_close_time}\n⚠️ Majority indicators aligned for sell."
                send_telegram_message(message)
                print(message)
            else:
                print("No clear buy or sell signal detected.")

            # Sleep 60 seconds to avoid multiple checks in the same minute
            time.sleep(180)
        else:
            # Sleep 10 seconds and check again
            time.sleep(10)

if __name__ == "__main__":
    main()