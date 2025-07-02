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
    TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram credentials are missing!")
        return
    url = f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage'
    payload = {
        'chat_id': TELEGRAM_CHAT_ID,
        'text': message,
        'parse_mode': 'HTML'
    }
    try:
        requests.post(url, data=payload, timeout=10)
    except Exception as e:
        print(f"Failed to send Telegram message: {e}")

# --- Indicator functions (same as before) ---
# (Include your existing indicator functions here: calculate_rsi, calculate_stoch_rsi, etc.)

# For brevity, assume all indicator functions and check_signal() are defined here exactly as you provided.

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

def wait_until_next_15min():
    """Sleep until the next quarter hour (15-min multiple) UTC."""
    now = datetime.utcnow()
    minutes = now.minute
    seconds = now.second
    # Calculate minutes to sleep until next 15-min multiple
    next_quarter = (minutes // 15 + 1) * 15
    if next_quarter == 60:
        next_quarter = 0
        # Sleep until next hour + 0 minutes
        sleep_minutes = (60 - minutes - 1)
    else:
        sleep_minutes = next_quarter - minutes - 1
    sleep_seconds = 60 - seconds
    total_sleep = sleep_minutes * 60 + sleep_seconds
    if total_sleep <= 0:
        total_sleep = 1  # minimum sleep to avoid busy loop
    print(f"Sleeping for {total_sleep} seconds until next 15-min interval...")
    time.sleep(total_sleep)

def main():
    while True:
        now = datetime.utcnow()
        if now.minute % 15 == 0:
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
                wait_until_next_15min()
                continue

            signal = check_signal(df)
            if signal == "buy":
                last_close_time = df.index[-1].strftime('%Y-%m-%d %H:%M UTC')
                message = f"🚀 <b>Buy Signal Detected for EUR/USD</b>\n🕒 Time: {last_close_time}\n✅ Majority indicators aligned for buy."
                send_telegram_message(message)
                print(message)
            elif signal == "sell":
                last_close_time = df.index[-1].strftime('%Y-%m-%d %H:%M UTC')
                message = f"🔥 <b>Sell Signal Detected for EUR/USD</b>\n🕒 Time: {last_close_time}\n⚠️ Majority indicators aligned for sell."
                send_telegram_message(message)
                print(message)
            else:
                print("No clear buy or sell signal detected.")

            wait_until_next_15min()
        else:
            wait_until_next_15min()

if __name__ == "__main__":
    main()