import os
import ccxt
import pandas as pd
import numpy as np
import requests
from datetime import datetime
import traceback

# --- CONFIGURATION ---

EXCHANGE_ID = 'kraken'
SYMBOL = 'EUR/USD'
INTERVAL = '15m'       # 15-minute candles
LOOKBACK = 50         # 3 days (96 candles per day * 3)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    raise ValueError("Telegram bot token or chat ID not set in environment variables.")

# --- INDICATOR CALCULATIONS ---

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_kdj(df, length=5, ma1=3, ma2=3):
    low_min = df['low'].rolling(window=length, min_periods=1).min()
    high_max = df['high'].rolling(window=length, min_periods=1).max()
    rsv = (df['close'] - low_min) / (high_max - low_min) * 100
    k = rsv.ewm(span=ma1, adjust=False).mean()
    d = k.ewm(span=ma2, adjust=False).mean()
    j = 3 * k - 2 * d
    return k, d, j

# --- SIGNAL GENERATION ---

def generate_signals(df):
    rsi = calculate_rsi(df['close'])
    k, d, j = calculate_kdj(df)

    signals = []
    position = None  # None or 'long'
    net_pnl = 0.0

    for i in range(1, len(df)):
        signal = None

        # RSI-based signals
        if rsi.iloc[i-1] < 30 and rsi.iloc[i] >= 30:
            signal = 'buy'  # RSI crossing above oversold
        elif rsi.iloc[i-1] > 70 and rsi.iloc[i] <= 70:
            signal = 'sell'  # RSI crossing below overbought

        # KDJ-based signals (bullish/bearish crossover)
        if (k.iloc[i-1] < d.iloc[i-1] and k.iloc[i] > d.iloc[i] and j.iloc[i] > k.iloc[i] and j.iloc[i] > d.iloc[i]):
            signal = 'buy'
        elif (k.iloc[i-1] > d.iloc[i-1] and k.iloc[i] < d.iloc[i] and j.iloc[i] < k.iloc[i] and j.iloc[i] < d.iloc[i]):
            signal = 'sell'

        close_price = df['close'].iloc[i]

        # Simple backtest logic
        if position is None and signal == 'buy':
            position = {'entry_price': close_price, 'entry_index': i}
        elif position is not None and signal == 'sell':
            trade_pnl = close_price - position['entry_price']
            net_pnl += trade_pnl
            position = None

        signals.append({
            'timestamp': df.index[i],
            'signal': signal,
            'close': close_price,
            'net_pnl': net_pnl
        })

    # Close any open position at last price
    if position is not None:
        trade_pnl = df['close'].iloc[-1] - position['entry_price']
        net_pnl += trade_pnl
        position = None

    return signals, net_pnl

# --- DATA FETCHING ---

def fetch_ohlcv(symbol, timeframe, limit):
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
    return df

# --- TELEGRAM NOTIFICATION ---

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML"
    }
    resp = requests.post(url, json=payload)
    resp.raise_for_status()

# --- MAIN ---

def main():
    dt = datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')
    try:
        df = fetch_ohlcv(SYMBOL, INTERVAL, LOOKBACK)
        if len(df) < LOOKBACK:
            print(f"Not enough data: have {len(df)} candles, need {LOOKBACK}")
            return

        signals, net_pnl = generate_signals(df)

        msg_lines = [
            f"<b>{EXCHANGE_ID.capitalize()} {SYMBOL} RSI & KDJ Backtest ({dt})</b>",
            f"Current Price: {df['close'].iloc[-1]:.6f}",
            f"Backtest Period: {len(df)} candles ({INTERVAL})",
            f"<b>Net Profit/Loss: {net_pnl:.6f} per unit traded</b>",
            "",
            "<b>Signals:</b>"
        ]

        for s in signals:
            if s['signal'] is not None:
                ts = s['timestamp'].strftime('%Y-%m-%d %H:%M')
                msg_lines.append(f"{ts} | Signal: {s['signal'].capitalize()} | Close: {s['close']:.6f} | Net PnL: {s['net_pnl']:.6f}")

        send_telegram_message("\n".join(msg_lines))
        print("Backtest report sent.")

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    # Define constants here
    SYMBOL = 'EUR/USD'
    INTERVAL = '15m'
    LOOKBACK = 288  # 3 days of 15m candles
    main()
