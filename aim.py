import subprocess
import sys
import os
import atexit
import aiohttp
import asyncio
import ccxt.async_support as ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict, deque
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from concurrent.futures import ProcessPoolExecutor, as_completed
import threading
import time
import math
import pickle
import json
from typing import Dict, List, Optional, Tuple, Any

# --- Launch Webserver Subprocess ---
webserver_path = os.path.join(os.path.dirname(__file__), 'webserver.py')
webserver_process = subprocess.Popen([sys.executable, webserver_path])
def cleanup():
    print("Terminating webserver subprocess...")
    webserver_process.terminate()
atexit.register(cleanup)

# --- Telegram Configuration ---
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
TELEGRAM_CHANNEL_ID = os.environ.get("TELEGRAM_CHANNEL_ID")

# --- Async Telegram messaging ---
async def send_telegram_message_async(message: str, chat_id_override: Optional[str] = None):
    if not TELEGRAM_BOT_TOKEN:
        print("TELEGRAM_BOT_TOKEN not set! Cannot send messages.")
        return
    
    MAX_LEN = 4096
    target_ids = [chat_id_override] if chat_id_override else list(filter(None, [TELEGRAM_CHAT_ID, TELEGRAM_CHANNEL_ID]))
    
    parts = []
    if len(message) > MAX_LEN:
        curr = ""
        for line in message.split('\n'):
            if len(curr) + len(line) + 1 <= MAX_LEN:
                curr += line + '\n'
            else:
                parts.append(curr)
                curr = line + '\n'
        if curr: 
            parts.append(curr)
    else:
        parts = [message]
    
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
        tasks = []
        for part in parts:
            for chat_id in target_ids:
                url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
                data = {
                    'chat_id': chat_id,
                    'text': part,
                    'parse_mode': 'HTML',
                    'disable_web_page_preview': True
                }
                tasks.append(session.post(url, data=data))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                print(f"Failed to send message: {result}")
            elif hasattr(result, 'status') and result.status != 200:
                print(f"Failed to send message: {result.status}")

# --- CCXT Exchange Initialization (Async) ---
async def create_exchange() -> ccxt.Exchange:
    exchange = ccxt.kucoin({'enableRateLimit': True, 'timeout': 30000})
    await exchange.load_markets()
    return exchange

CRYPTO_SYMBOLS = [
    "XRP/USDT", "XMR/USDT", "GMX/USDT", "LUNA/USDT", "TRX/USDT", "EIGEN/USDT",
    "APE/USDT", "WAVES/USDT", "PLUME/USDT", "SUSHI/USDT", "DOGE/USDT", "VIRTUAL/USDT",
    "CAKE/USDT", "GRASS/USDT", "AAVE/USDT", "SUI/USDT", "ARB/USDT", "XLM/USDT",
    "MNT/USDT", "LTC/USDT", "NEAR/USDT"
]

# --- Indicator Utilities --- 
# [Same as original, omitted here for brevity; use your existing functions]

# --- Indicator Signal Analysis ---
# [Same as original]

# --- Machine Learning Components with Calibration ---
# [Same as original]

# --- Trailing Take Profit Logic ---
# [Same as original]

# --- Async Fetch OHLCV Data ---
async def fetch_latest_ohlcv_async(exchange: ccxt.Exchange, symbol: str, timeframe: str = '4h', limit: int = 750):
    try:
        if symbol not in exchange.symbols:
            print(f"{symbol} not on exchange.")
            return None

        data = await exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df.dropna(subset=['close'], inplace=True)
        return df

    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None

# --- Position Sizing ---
MAX_UNLEVERAGED_NOTIONAL = 1.0
LEVERAGE = 15

def calculate_position_size_dollars(price: float) -> Tuple[float, float]:
    units = MAX_UNLEVERAGED_NOTIONAL / price
    notional = units * price
    return notional, units

# --- Process a Single Symbol (Multiprocessing Worker) ---
def process_symbol_worker(symbol_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    # [Same as original, just processes symbol_data with 4h candles data]

# --- Message Formatting ---
# [Same as original]

# --- Scheduling Utilities ---
def seconds_until_next_4h_utc():
    now = datetime.utcnow()
    next_hour = ((now.hour // 4) + 1) * 4 % 24
    next_run = datetime.combine(now.date(), datetime.min.time()) + timedelta(hours=next_hour)
    if next_run <= now:
        next_run += timedelta(days=1)
    return (next_run - now).total_seconds()

# --- Main Async Function ---
async def async_main_scan():
    start = datetime.utcnow()
    timestamp = start.strftime('%Y-%m-%d %H:%M UTC')
    print(f"\nStarting scan at {timestamp}")
    
    exchange = await create_exchange()
    try:
        print("Fetching market data...")
        fetch_tasks = [fetch_latest_ohlcv_async(exchange, symbol, timeframe='4h', limit=750) for symbol in CRYPTO_SYMBOLS]
        fetch_results = await asyncio.gather(*fetch_tasks, return_exceptions=True)

        symbol_data_list = []
        for i, result in enumerate(fetch_results):
            if isinstance(result, Exception):
                print(f"Error fetching {CRYPTO_SYMBOLS[i]}: {result}")
                continue
            if result is not None:
                df_data = {
                    'data': result.values.tolist(),
                    'columns': result.columns.tolist(),
                    'index': result.index.strftime('%Y-%m-%d %H:%M:%S').tolist()
                }
                symbol_data_list.append({'symbol': CRYPTO_SYMBOLS[i], 'df_data': df_data})
        
        print(f"Successfully fetched data for {len(symbol_data_list)} symbols")

        coin_results = []
        if symbol_data_list:
            max_workers = min(os.cpu_count() or 1, len(symbol_data_list))
            print(f"Processing with {max_workers} workers...")
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(process_symbol_worker, data): data['symbol'] for data in symbol_data_list}
                for future in as_completed(futures):
                    symbol = futures[future]
                    try:
                        result = future.result()
                        if result:
                            coin_results.append(result)
                    except Exception as e:
                        print(f"Error processing {symbol}: {e}")
        
        print(f"Processing complete: {len(coin_results)} signals detected.")

        print("Training ML model...")
        train_ml_model()

        msg = format_summary_message(coin_results, timestamp)
        await send_telegram_message_async(msg)

        print(f"Scan completed in {(datetime.utcnow() - start).total_seconds():.1f} seconds")
    finally:
        await exchange.close()

# --- Main Loop ---
def main():
    try:
        while True:
            asyncio.run(async_main_scan())
            sleep_sec = seconds_until_next_4h_utc()
            print(f"Sleeping {sleep_sec:.0f}s until next 4h UTC.")
            time.sleep(sleep_sec)
    except KeyboardInterrupt:
        print("Interrupted. Exiting.")
        cleanup()
    except Exception as e:
        print(f"Main loop error: {e}")
        cleanup()

if __name__ == "__main__":
    main()
