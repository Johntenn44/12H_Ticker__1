import os
import ccxt
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import traceback

# --- CONFIGURATION ---
EXCHANGE_ID = 'kraken'
SYMBOL = 'EUR/USD'
INTERVAL = '15m'
HOLD_CANDLES = 3  # Close positions after 3 candles (45 minutes)
LOOKBACK = 2880    # 30 days

STARTING_UNITS = 10000
PROFIT_UNIT_INCREASE = 9
LOSS_UNIT_DECREASE = 5

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
    return 100 - (100 / (1 + rs))

def calculate_stoch_rsi(df, rsi_length=14, stoch_length=14, smooth_k=3, smooth_d=3):
    rsi = calculate_rsi(df['close'], rsi_length)
    min_rsi = rsi.rolling(window=stoch_length).min()
    max_rsi = rsi.rolling(window=stoch_length).max()
    denominator = max_rsi - min_rsi
    denominator = denominator.replace(0, np.nan)
    stoch_rsi = (rsi - min_rsi) / denominator * 100
    stoch_rsi = stoch_rsi.fillna(method='ffill')
    k = stoch_rsi.rolling(window=smooth_k).mean()
    d = k.rolling(window=smooth_d).mean()
    return k, d

def calculate_wr(df, length=14):
    highest_high = df['high'].rolling(window=length).max()
    lowest_low = df['low'].rolling(window=length).min()
    denominator = highest_high - lowest_low
    denominator = denominator.replace(0, np.nan)
    return ((highest_high - df['close']) / denominator * -100).fillna(method='ffill')

# --- TREND ANALYSIS ---
def analyze_stoch_rsi_trend(k, d):
    if len(k) < 2 or pd.isna(k.iloc[-2]) or pd.isna(d.iloc[-2]) or pd.isna(k.iloc[-1]) or pd.isna(d.iloc[-1]):
        return None
    if k.iloc[-2] < d.iloc[-2] and k.iloc[-1] > d.iloc[-1] and k.iloc[-1] < 80:
        return 'buy'
    elif k.iloc[-2] > d.iloc[-2] and k.iloc[-1] < d.iloc[-1] and k.iloc[-1] > 20:
        return 'sell'
    return None

def analyze_wr_trend(wr_series):
    if len(wr_series) < 2 or pd.isna(wr_series.iloc[-2]) or pd.isna(wr_series.iloc[-1]):
        return None
    prev, curr = wr_series.iloc[-2], wr_series.iloc[-1]
    if prev > -80 and curr <= -80:
        return 'buy'
    elif prev < -20 and curr >= -20:
        return 'sell'
    return None

# --- DATA FETCHING ---
def fetch_ohlcv_paginated(symbol, timeframe, since, limit=1000):
    exchange_class = getattr(ccxt, EXCHANGE_ID)
    exchange = exchange_class()
    exchange.load_markets()
    all_ohlcv = []
    while True:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
        if not ohlcv: break
        all_ohlcv += ohlcv
        since = ohlcv[-1][0] + 1
        if len(ohlcv) < limit: break
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df.set_index('timestamp').astype(float)

def fetch_ohlcv_30days(symbol, timeframe):
    since = int((datetime.utcnow() - timedelta(days=30)).timestamp() * 1000)
    return fetch_ohlcv_paginated(symbol, timeframe, since)

# --- BACKTEST WITH TIMED EXIT ---
def backtest(df):
    k_stoch, d_stoch = calculate_stoch_rsi(df)
    wr = calculate_wr(df)
    
    position = None
    units = STARTING_UNITS
    wins = losses = 0

    for i in range(1, len(df)):
        # Check if we need to close a timed position
        if position and i >= position['exit_index']:
            trade_pnl = df['close'].iloc[i] - position['entry_price']
            if trade_pnl > 0:
                units += PROFIT_UNIT_INCREASE
                wins += 1
            else:
                units -= LOSS_UNIT_DECREASE
                losses += 1
            position = None
        
        # Generate signals only if no active position
        if not position:
            stoch_signal = analyze_stoch_rsi_trend(k_stoch.iloc[:i+1], d_stoch.iloc[:i+1])
            wr_signal = analyze_wr_trend(wr.iloc[:i+1])
            
            # Enter position on buy signal
            if 'buy' in [stoch_signal, wr_signal]:
                position = {
                    'entry_price': df['close'].iloc[i],
                    'entry_index': i,
                    'exit_index': i + HOLD_CANDLES
                }

    # Close any remaining position at last price
    if position:
        trade_pnl = df['close'].iloc[-1] - position['entry_price']
        if trade_pnl > 0:
            units += PROFIT_UNIT_INCREASE
            wins += 1
        else:
            units -= LOSS_UNIT_DECREASE
            losses += 1

    total_trades = wins + losses
    accuracy = (wins / total_trades * 100) if total_trades > 0 else 0
    return units, accuracy, total_trades

# --- TELEGRAM NOTIFICATION ---
def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    resp = requests.post(url, json={
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML"
    })
    resp.raise_for_status()

# --- MAIN EXECUTION ---
def main():
    dt = datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')
    try:
        df = fetch_ohlcv_30days(SYMBOL, INTERVAL)
        if len(df) < LOOKBACK:
            print(f"Not enough data: have {len(df)} candles, need {LOOKBACK}")
            return

        final_units, accuracy, total_trades = backtest(df)
        current_price = df['close'].iloc[-1]

        msg = (
            f"<b>{EXCHANGE_ID.capitalize()} {SYMBOL} Backtest Summary ({dt})</b>\n"
            f"Period: {len(df)} candles ({INTERVAL})\n"
            f"Current Price: {current_price:.6f}\n"
            f"Position Hold: {HOLD_CANDLES * 15} minutes\n"
            f"Starting Units: {STARTING_UNITS}\n"
            f"Final Units: {final_units}\n"
            f"Total Trades: {total_trades}\n"
            f"Accuracy: {accuracy:.2f}%"
        )

        send_telegram_message(msg)
        print("Report sent to Telegram")

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
