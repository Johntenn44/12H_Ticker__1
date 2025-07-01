import os
import ccxt
import pandas as pd
import numpy as np
import requests
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
        print("Telegram message sent successfully.")
    except Exception as e:
        print(f"Failed to send Telegram message: {e}")

# --- FETCH OHLCV DATA FROM KRAKEN ---
def fetch_ohlcv_kraken(symbol='EUR/USD', timeframe='15m', limit=1350):
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
        raise

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
        return False
    return k.iloc[-2] < d.iloc[-2] and k.iloc[-1] > d.iloc[-1] and k.iloc[-1] < 80

def analyze_wr_oversold(wr):
    if len(wr) < 2 or pd.isna(wr.iloc[-2]) or pd.isna(wr.iloc[-1]):
        return False
    return wr.iloc[-2] > -80 and wr.iloc[-1] <= -80

def analyze_rsi_trend(rsi8, rsi13, rsi21):
    if rsi8 > rsi13 > rsi21:
        return "Uptrend"
    elif rsi8 < rsi13 < rsi21:
        return "Downtrend"
    else:
        return "No clear RSI trend"

def analyze_kdj_trend(k, d, j):
    if len(k) < 2 or len(d) < 2 or len(j) < 2:
        return "No clear KDJ trend"
    k_prev, k_curr = k.iloc[-2], k.iloc[-1]
    d_prev, d_curr = d.iloc[-2], d.iloc[-1]
    j_prev, j_curr = j.iloc[-2], j.iloc[-1]
    if k_prev < d_prev and k_curr > d_curr and j_curr > k_curr and j_curr > d_curr:
        return "Bullish KDJ crossover"
    elif k_prev > d_prev and k_curr < d_curr and j_curr < k_curr and j_curr < d_curr:
        return "Bearish KDJ crossover"
    else:
        return "No clear KDJ trend"

# --- SIGNAL GENERATION ---

def generate_combined_signal(df):
    k, d = calculate_stoch_rsi(df)
    wr = calculate_wr(df, length=14)
    rsi8 = calculate_rsi(df['close'], 8).iloc[-1]
    rsi13 = calculate_rsi(df['close'], 13).iloc[-1]
    rsi21 = calculate_rsi(df['close'], 21).iloc[-1]
    kdj_k, kdj_d, kdj_j = calculate_kdj(df)

    stoch_signal = analyze_stoch_rsi_trend(k, d)
    wr_signal = analyze_wr_oversold(wr)
    rsi_trend = analyze_rsi_trend(rsi8, rsi13, rsi21)
    kdj_trend = analyze_kdj_trend(kdj_k, kdj_d, kdj_j)

    # Buy signal without MA filters:
    buy_signal = (stoch_signal or wr_signal) and rsi_trend == "Uptrend" and kdj_trend == "Bullish KDJ crossover"
    return buy_signal

# --- BACKTESTING LOGIC ---

def backtest(df):
    position_size = 100
    trades = []
    position = None
    closes = df['close'].values

    for i in range(len(df)):
        df_slice = df.iloc[:i+1]
        if len(df_slice) < 200:
            continue
        if position is None:
            if generate_combined_signal(df_slice):
                position = {'entry_index': i, 'entry_price': closes[i], 'size': position_size}
        else:
            if i == position['entry_index'] + 3:  # Close after 3 candles (45 min)
                exit_price = closes[i]
                pnl = (exit_price - position['entry_price']) * position['size']
                win = pnl > 0
                trades.append({
                    'entry_time': df.index[position['entry_index']],
                    'exit_time': df.index[i],
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'size': position['size'],
                    'pnl': pnl,
                    'win': win
                })
                position_size += 4 if win else -5
                position_size = max(0, position_size)
                position = None

    wins = sum(t['win'] for t in trades)
    total = len(trades)
    accuracy = (wins / total * 100) if total > 0 else 0
    final_size = trades[-1]['size'] if trades else 100

    return trades, accuracy, final_size

# --- MAIN EXECUTION ---

def main():
    print("Fetching EUR/USD 15m data from Kraken for the past ~14 days...")
    df = fetch_ohlcv_kraken(limit=1350)

    print("Running backtest...")
    trades, accuracy, final_size = backtest(df)

    summary = (
        f"<b>EUR/USD Kraken 15m Backtest Summary (Last ~14 days)</b>\n"
        f"Total trades: {len(trades)}\n"
        f"Winning trades: {sum(t['win'] for t in trades)}\n"
        f"Accuracy: {accuracy:.2f}%\n"
        f"Final position size (units): {final_size}"
    )

    print(summary)
    send_telegram_message(summary)

    for t in trades:
        print(f"Entry: {t['entry_time']} @ {t['entry_price']:.5f} | Exit: {t['exit_time']} @ {t['exit_price']:.5f} | "
              f"PnL: {t['pnl']:.2f} | {'WIN' if t['win'] else 'LOSS'}")

if __name__ == "__main__":
    main()
