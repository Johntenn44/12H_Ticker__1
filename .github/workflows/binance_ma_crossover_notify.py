import os
import ccxt
import pandas as pd
import numpy as np
import requests
from datetime import datetime
import traceback

# --- CONFIGURATION ---

EXCHANGE_ID = 'kraken'
SYMBOL = 'EUR/USD'  # Kraken symbol for EUR/USD
INTERVAL = '15m'    # 15-minute candles
LOOKBACK = 500      # Number of candles to fetch

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# --- INDICATOR CALCULATIONS ---

def calculate_rsi(series, period=13):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_stoch_rsi(df, rsi_length=13, stoch_length=8, smooth_k=5, smooth_d=3):
    rsi = calculate_rsi(df['close'], rsi_length)
    min_rsi = rsi.rolling(window=stoch_length).min()
    max_rsi = rsi.rolling(window=stoch_length).max()
    denominator = max_rsi - min_rsi
    denominator = denominator.replace(0, np.nan)  # avoid division by zero
    stoch_rsi = (rsi - min_rsi) / denominator * 100
    stoch_rsi = stoch_rsi.fillna(method='ffill')
    k = stoch_rsi.rolling(window=smooth_k).mean()
    d = k.rolling(window=smooth_d).mean()
    return k, d

def calculate_wr(df, length):
    highest_high = df['high'].rolling(window=length).max()
    lowest_low = df['low'].rolling(window=length).min()
    denominator = highest_high - lowest_low
    denominator = denominator.replace(0, np.nan)  # avoid division by zero
    wr = (highest_high - df['close']) / denominator * -100
    return wr.fillna(method='ffill')

# --- TREND LOGIC ---

def analyze_stoch_rsi_trend(k, d):
    if len(k) < 2 or pd.isna(k.iloc[-2]) or pd.isna(d.iloc[-2]) or pd.isna(k.iloc[-1]) or pd.isna(d.iloc[-1]):
        return "No clear Stoch RSI trend"
    if k.iloc[-2] < d.iloc[-2] and k.iloc[-1] > d.iloc[-1] and k.iloc[-1] < 80:
        return "Uptrend"
    elif k.iloc[-2] > d.iloc[-2] and k.iloc[-1] < d.iloc[-1] and k.iloc[-1] > 20:
        return "Downtrend"
    else:
        return "No clear Stoch RSI trend"

def analyze_wr_trend(wr_series):
    if len(wr_series) < 2 or pd.isna(wr_series.iloc[-2]) or pd.isna(wr_series.iloc[-1]):
        return "No clear WR trend"
    prev, curr = wr_series.iloc[-2], wr_series.iloc[-1]
    if prev > -80 and curr <= -80:
        return "WR Oversold - Buy signal"
    elif prev < -20 and curr >= -20:
        return "WR Overbought - Sell signal"
    else:
        return "No clear WR trend"

# --- DATA FETCHING ---

def fetch_ohlcv_ccxt(symbol, timeframe, limit):
    exchange = getattr(ccxt, EXCHANGE_ID)()
    exchange.load_markets()
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(
        ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
    )
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df[['open','high','low','close','volume']] = df[['open','high','low','close','volume']].astype(float)
    return df

# --- TELEGRAM NOTIFICATION ---

def send_telegram_message(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram bot token or chat ID not set. Skipping Telegram message.")
        return
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

# --- SIGNAL GENERATION ---

def generate_signals(df):
    """
    Generate buy signals based on Stoch RSI Uptrend or Williams %R oversold signals.
    Returns a list of signals per candle: 'buy' or None.
    """
    signals = [None] * len(df)

    # Calculate indicators
    k, d = calculate_stoch_rsi(df)
    stoch_trends = []
    for i in range(len(df)):
        if i < 2:
            stoch_trends.append("No clear Stoch RSI trend")
            continue
        k_slice = k.iloc[:i+1]
        d_slice = d.iloc[:i+1]
        trend = analyze_stoch_rsi_trend(k_slice, d_slice)
        stoch_trends.append(trend)

    wr = calculate_wr(df, length=8)  # Using 8 as example period for WR
    wr_trends = []
    for i in range(len(df)):
        if i < 2:
            wr_trends.append("No clear WR trend")
            continue
        wr_slice = wr.iloc[:i+1]
        trend = analyze_wr_trend(wr_slice)
        wr_trends.append(trend)

    # Generate buy signals if Stoch RSI Uptrend or WR Oversold Buy signal
    for i in range(len(df)):
        if stoch_trends[i] == "Uptrend" or wr_trends[i] == "WR Oversold - Buy signal":
            signals[i] = 'buy'

    return signals

# --- BACKTESTING LOGIC ---

def backtest(df):
    position_size = 100
    trades = []
    position = None  # dict with entry info or None

    signals = generate_signals(df)
    closes = df['close'].values

    for i in range(len(df)):
        signal = signals[i]
        price = closes[i]

        # Entry: open position on buy signal if no open position
        if position is None and signal == 'buy':
            position = {
                'entry_index': i,
                'entry_price': price,
                'size': position_size
            }

        # Exit: close position 3 candles (45 minutes) after entry
        if position is not None and i == position['entry_index'] + 3:
            exit_price = price
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
            # Update position size
            if win:
                position_size += 4
            else:
                position_size = max(0, position_size - 5)
            position = None

    # Calculate accuracy
    wins = sum(t['win'] for t in trades)
    total = len(trades)
    accuracy = (wins / total * 100) if total > 0 else 0

    return trades, accuracy

# --- MAIN ---

def main():
    print(f"Fetching {SYMBOL} data from {EXCHANGE_ID}...")
    try:
        df = fetch_ohlcv_ccxt(SYMBOL, INTERVAL, LOOKBACK)
    except Exception as e:
        print(f"Error fetching data: {e}")
        traceback.print_exc()
        return

    if len(df) < 20:
        print("Not enough data to run backtest")
        return

    trades, accuracy = backtest(df)

    total_trades = len(trades)
    wins = sum(t['win'] for t in trades)
    final_position_size = trades[-1]['size'] if trades else 100

    print(f"Backtest completed on {SYMBOL} ({INTERVAL} timeframe)")
    print(f"Total trades: {total_trades}")
    print(f"Winning trades: {wins}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Final position size: {final_position_size}")

    # Prepare message for Telegram
    message = (
        f"<b>Backtest Summary for {SYMBOL} ({INTERVAL})</b>\n"
        f"Total trades: {total_trades}\n"
        f"Winning trades: {wins}\n"
        f"Accuracy: {accuracy:.2f}%\n"
        f"Final position size (units): {final_position_size}"
    )

    send_telegram_message(message)

    # Optional: print trade details
    for t in trades:
        print(f"Entry: {t['entry_time']} @ {t['entry_price']:.5f} | Exit: {t['exit_time']} @ {t['exit_price']:.5f} | PnL: {t['pnl']:.2f} | {'WIN' if t['win'] else 'LOSS'}")

if __name__ == "__main__":
    main()
