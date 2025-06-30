import os
import ccxt
import pandas as pd
import numpy as np
import requests
from datetime import datetime

# --- CONFIGURATION ---

COINS = ["USDT/EUR"]     # KuCoin symbol for USDT/EUR
EXCHANGE_ID = 'kucoin'
INTERVAL = '15m'
LOOKBACK = 672           # 7 days * 96 candles/day = 672 candles

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    raise ValueError("Telegram bot token or chat ID not set in environment variables.")

# --- INDICATOR CALCULATION ---

def add_indicators(df):
    df['EMA200'] = df['close'].ewm(span=200, adjust=False).mean()
    df['MA50'] = df['close'].rolling(window=50).mean()
    df['MA200'] = df['close'].rolling(window=200).mean()
    return df

def calculate_rsi(series, period):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_kdj(df, length=5, ma1=8, ma2=8):
    low_min = df['low'].rolling(window=length, min_periods=1).min()
    high_max = df['high'].rolling(window=length, min_periods=1).max()
    rsv = (df['close'] - low_min) / (high_max - low_min) * 100
    k = rsv.ewm(span=ma1, adjust=False).mean()
    d = k.ewm(span=ma2, adjust=False).mean()
    j = 3 * k - 2 * d
    return k, d, j

# --- TREND LOGIC ---

def analyze_trend(df):
    cp = df['close'].iloc[-1]
    ma50 = df['MA50'].iloc[-1]
    ema200 = df['EMA200'].iloc[-1]
    ma200 = df['MA200'].iloc[-1]

    low = min(ma50, ema200, ma200)
    high = max(ma50, ema200, ma200)

    return {'price_between_mas': low <= cp <= high}

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

# --- DATA FETCHING AND CONVERSION ---

def fetch_ohlcv_and_convert(symbol, timeframe, limit):
    exchange_class = getattr(ccxt, EXCHANGE_ID)
    exchange = exchange_class()
    exchange.load_markets()

    if symbol not in exchange.symbols:
        raise ValueError(f"Symbol {symbol} not available on {EXCHANGE_ID}")

    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df.astype(float)

    # Convert USDT/EUR to EUR/USDT by taking reciprocal of prices
    df['open'] = 1 / df['open']
    df['high'] = 1 / df['low']
    df['low'] = 1 / df['high']
    df['close'] = 1 / df['close']

    # Convert volume approx to EUR (base currency of EUR/USDT)
    df['volume'] = df['volume'] * df['close']

    return df

# --- TELEGRAM NOTIFICATION ---

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML"
    }
    resp = requests.post(url, data=payload)
    resp.raise_for_status()

# --- BACKTEST FUNCTION ---

def backtest(df):
    signals = []
    # Start at 200 to have enough data for indicators
    for i in range(200, len(df)):
        window = df.iloc[:i+1].copy()
        window = add_indicators(window)

        trend = analyze_trend(window)
        if not trend.get('price_between_mas'):
            continue

        rsi8 = calculate_rsi(window['close'], 8).iloc[-1]
        rsi13 = calculate_rsi(window['close'], 13).iloc[-1]
        rsi21 = calculate_rsi(window['close'], 21).iloc[-1]
        if np.isclose(rsi8, rsi13) and np.isclose(rsi13, rsi21):
            continue
        rsi_trend = analyze_rsi_trend(rsi8, rsi13, rsi21)

        k, d, j = calculate_kdj(window, length=5, ma1=8, ma2=8)
        if np.isclose(k.iloc[-1], d.iloc[-1]) and np.isclose(d.iloc[-1], j.iloc[-1]):
            continue
        kdj_trend = analyze_kdj_trend(k, d, j)

        if rsi_trend == "No clear RSI trend" and kdj_trend == "No clear KDJ trend":
            continue

        signals.append({
            'timestamp': window.index[-1],
            'rsi_trend': rsi_trend,
            'kdj_trend': kdj_trend,
            'close': window['close'].iloc[-1]
        })

    return signals

# --- MAIN LOGIC ---

def main():
    dt = datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')

    try:
        df = fetch_ohlcv_and_convert("USDT/EUR", INTERVAL, LOOKBACK)
        if len(df) < 200:
            print("Not enough data")
            return

        signals = backtest(df)

        if signals:
            msg_lines = [f"<b>KuCoin {INTERVAL.upper()} Backtest RSI & KDJ Signals for EUR/USDT ({dt})</b>"]
            for s in signals:
                ts = s['timestamp'].strftime('%Y-%m-%d %H:%M')
                msg_lines.append(f"{ts} - Close: {s['close']:.6f} | RSI: {s['rsi_trend']} | KDJ: {s['kdj_trend']}")
            msg = "\n".join(msg_lines)
            send_telegram_message(msg)
            print("Backtest signals sent.")
        else:
            print("No signals generated in backtest.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
