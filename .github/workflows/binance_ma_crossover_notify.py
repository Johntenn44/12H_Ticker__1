import os
import ccxt
import pandas as pd
import numpy as np
import requests
from datetime import datetime
import traceback

# --- CONFIGURATION ---
COINS = ["USDT/EUR"]  # KuCoin symbol for USDT/EUR
EXCHANGE_ID = 'kucoin'
INTERVAL = '15m'
LOOKBACK = 210

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    raise ValueError("Telegram credentials not set")

# --- INDICATOR CALCULATIONS ---
def calculate_rsi(series, period=13):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period-1, adjust=False).mean()
    avg_loss = loss.ewm(com=period-1, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_stoch_rsi(df, rsi_length=13, stoch_length=8, smooth_k=5, smooth_d=3):
    rsi = calculate_rsi(df['close'], rsi_length)
    min_rsi = rsi.rolling(stoch_length).min()
    max_rsi = rsi.rolling(stoch_length).max()
    stoch_rsi = (rsi - min_rsi) / (max_rsi - min_rsi).replace(0, np.nan) * 100
    stoch_rsi = stoch_rsi.fillna(method='ffill')
    k = stoch_rsi.rolling(smooth_k).mean()
    d = k.rolling(smooth_d).mean()
    return k, d

def calculate_wr(df, length):
    highest_high = df['high'].rolling(length).max()
    lowest_low = df['low'].rolling(length).min()
    wr = (highest_high - df['close']) / (highest_high - lowest_low).replace(0, np.nan) * -100
    return wr.fillna(method='ffill')

# --- TREND LOGIC ---
def analyze_stoch_rsi_trend(k, d):
    if len(k) < 2 or any(pd.isna(x) for x in [k.iloc[-2], d.iloc[-2], k.iloc[-1], d.iloc[-1]]):
        return "No clear trend"
    if k.iloc[-2] < d.iloc[-2] and k.iloc[-1] > d.iloc[-1] and k.iloc[-1] < 80:
        return "Uptrend"
    elif k.iloc[-2] > d.iloc[-2] and k.iloc[-1] < d.iloc[-1] and k.iloc[-1] > 20:
        return "Downtrend"
    return "No clear trend"

def analyze_wr_trend(wr_series):
    if len(wr_series) < 2 or any(pd.isna(wr_series.iloc[i]) for i in [-2, -1]):
        return "No clear WR trend"
    prev, curr = wr_series.iloc[-2], wr_series.iloc[-1]
    if prev > -80 and curr <= -80: return "Oversold - Buy"
    if prev < -20 and curr >= -20: return "Overbought - Sell"
    return "No clear WR trend"

# --- DATA HANDLING ---
def fetch_ohlcv_and_convert(symbol, timeframe, limit):
    exchange = getattr(ccxt, EXCHANGE_ID)()
    exchange.load_markets()
    
    # Fetch current price first
    ticker = exchange.fetch_ticker(symbol)
    current_price = 1 / ticker['last']  # Convert USDT/EUR to EUR/USDT
    
    # Fetch historical data
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df.astype(float)
    
    # Convert prices to EUR/USDT
    for col in ['open', 'high', 'low', 'close']:
        df[col] = 1 / df[col]
    df['volume'] *= df['close']
    
    return df, current_price

# --- BACKTESTING ---
def backtest(df):
    signals = []
    WR_PERIODS = [3, 8, 13, 55, 144, 233]
    start_index = max(20, max(WR_PERIODS))
    
    for i in range(start_index, len(df)):
        window = df.iloc[:i+1].copy()
        k, d = calculate_stoch_rsi(window)
        stoch_trend = analyze_stoch_rsi_trend(k, d)
        
        wr_signals = []
        for period in WR_PERIODS:
            wr = calculate_wr(window, period)
            trend = analyze_wr_trend(wr)
            if "No clear" not in trend:
                wr_signals.append(f"WR{period}: {trend}")
        
        if "No clear" not in stoch_trend or wr_signals:
            signals.append({
                'timestamp': window.index[-1],
                'price': window['close'].iloc[-1],
                'stoch_trend': stoch_trend,
                'wr_signals': ", ".join(wr_signals) if wr_signals else "No WR signals"
            })
    
    return signals

# --- TELEGRAM INTEGRATION ---
def send_telegram_message(message):
    requests.post(
        f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
        json={'chat_id': TELEGRAM_CHAT_ID, 'text': message, 'parse_mode': 'HTML'}
    ).raise_for_status()

# --- MAIN EXECUTION ---
def main():
    dt = datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')
    try:
        # Fetch data and current price
        df, current_price = fetch_ohlcv_and_convert(COINS[0], INTERVAL, LOOKBACK)
        if len(df) < LOOKBACK:
            print("Insufficient data")
            return
        
        # Get signals
        signals = backtest(df)
        
        # Prepare Telegram message
        msg_lines = [
            f"<b>KuCoin EUR/USDT Report ({dt})</b>",
            f"Current Price: {current_price:.6f}",
            f"Backtest Period: {len(df)} candles ({INTERVAL})"
        ]
        
        if signals:
            msg_lines.append("\n<b>Signals Found:</b>")
            for s in signals:
                ts = s['timestamp'].strftime('%Y-%m-%d %H:%M')
                msg_lines.append(
                    f"{ts} | Price: {s['price']:.6f} | "
                    f"StochRSI: {s['stoch_trend']} | {s['wr_signals']}"
                )
        else:
            msg_lines.append("\nNo trading signals detected")
        
        send_telegram_message("\n".join(msg_lines))
        print("Report sent")
        
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
