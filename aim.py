import os
import sys
import time
import math
import random
import atexit
import threading
import requests
import subprocess
import ccxt
import schedule
import pandas as pd
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from collections import defaultdict, deque

# === CONFIG ===
CONFIG = {
    "6h": {"desc": "6h TF", "interval_hours": 6},
    "8h": {"desc": "8h TF", "interval_hours": 8}
}

CRYPTO_SYMBOLS = [
    "XRP/USDT", "SUI/USDT", "DOGE/USDT", "XLM/USDT", "TRX/USDT",
    "AAVE/USDT", "LTC/USDT", "ARB/USDT", "NEAR/USDT", "MNT/USDT",
    "VIRTUAL/USDT", "XMR/USDT", "EIGEN/USDT", "CAKE/USDT", "SUSHI/USDT",
    "GRASS/USDT", "WAVES/USDT", "APE/USDT", "SKL/USDT", "PLUME/USDT", "LUNA/USDT"
]

MAX_RISK_PER_TRADE_USD = 10
BASE_MAX_LEVERAGE = 20
MAX_WORKERS = 4

# === TELEGRAM ===
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
TELEGRAM_CHANNEL_ID = os.getenv("TELEGRAM_CHANNEL_ID")
TARGET_CHAT_IDS = list(filter(None, [TELEGRAM_CHANNEL_ID, TELEGRAM_CHAT_ID]))

def send_telegram_message(msg):
    for cid in TARGET_CHAT_IDS:
        try:
            r = requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                data={"chat_id": cid, "text": msg, "parse_mode": "HTML", "disable_web_page_preview": True},
                timeout=15
            )
            if not r.ok:
                print(f"Telegram error: {r.text}")
        except Exception as e:
            print(f"Telegram exception: {e}")

# === START WEBSERVER SUBPROCESS ===
webserver_proc = None

def start_webserver():
    global webserver_proc
    try:
        path = os.path.join(os.path.dirname(__file__), "webserver.py")
        print("🚀 Starting webserver.py")
        webserver_proc = subprocess.Popen([sys.executable, path])
    except Exception as e:
        print(f"❌ Could not start webserver.py: {e}")

def stop_webserver():
    global webserver_proc
    if webserver_proc:
        print("🛑 Stopping webserver.py")
        webserver_proc.terminate()
        webserver_proc.wait()

atexit.register(stop_webserver)

# === EXCHANGE ===
exchange = None
exchange_lock = threading.Lock()

def initialize_exchange():
    global exchange
    exchange = ccxt.kucoin()
    exchange.load_markets()
    print("✅ Exchange initialized.")

initialize_exchange()

# === INDICATOR CALCULATIONS ===

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_stoch_rsi(df, rsi_len=14, stoch_len=14, smooth_k=3, smooth_d=3):
    rsi = calculate_rsi(df["close"], rsi_len)
    min_rsi = rsi.rolling(stoch_len).min()
    max_rsi = rsi.rolling(stoch_len).max()
    k = ((rsi - min_rsi) / (max_rsi - min_rsi)) * 100
    k = k.rolling(smooth_k).mean()
    d = k.rolling(smooth_d).mean()
    return k, d

def calculate_macd(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def calculate_bollinger_bands(close, window=20, num_std=2):
    sma = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    return upper, lower

def calculate_indicators(df):
    df["rsi5"] = calculate_rsi(df["close"], 5)
    df["rsi13"] = calculate_rsi(df["close"], 13)
    df["rsi21"] = calculate_rsi(df["close"], 21)
    k, d = calculate_stoch_rsi(df)
    df["stoch_k"] = k
    df["stoch_d"] = d
    macd, macd_signal = calculate_macd(df["close"])
    df["macd"], df["macd_signal"] = macd, macd_signal
    upper, lower = calculate_bollinger_bands(df["close"])
    df["bb_upper"], df["bb_lower"] = upper, lower
    return df

# === DATA FETCHING & SIGNAL PROCESSING ===

def fetch_latest_ohlcv(symbol, timeframe='6h', limit=200):
    try:
        with exchange_lock:
            data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        return df.dropna()
    except Exception as e:
        print(f"{symbol} fetch error: {e}")
        return None

def detect_signal(df):
    signals = {}
    try:
        idx = -1
        if df["stoch_k"].iloc[idx] > df["stoch_d"].iloc[idx]:
            signals["Stoch"] = "up"
        elif df["stoch_k"].iloc[idx] < df["stoch_d"].iloc[idx]:
            signals["Stoch"] = "down"

        if df["rsi5"].iloc[idx] > df["rsi13"].iloc[idx] > df["rsi21"].iloc[idx]:
            signals["RSI"] = "up"
        elif df["rsi5"].iloc[idx] < df["rsi13"].iloc[idx] < df["rsi21"].iloc[idx]:
            signals["RSI"] = "down"

        if df["macd"].iloc[idx] > df["macd_signal"].iloc[idx]:
            signals["MACD"] = "up"
        elif df["macd"].iloc[idx] < df["macd_signal"].iloc[idx]:
            signals["MACD"] = "down"

        close = df["close"].iloc[idx]
        if close > df["bb_upper"].iloc[idx]:
            signals["Bollinger"] = "down"
        elif close < df["bb_lower"].iloc[idx]:
            signals["Bollinger"] = "up"
    except:
        pass
    return signals

def infer_direction(signals):
    ups = sum(1 for v in signals.values() if v == "up")
    downs = sum(1 for v in signals.values() if v == "down")
    conf = (max(ups, downs) / len(signals)) * 100 if signals else 0.0
    if conf < 55.0: return None, 0.0
    return ("buy" if ups > downs else "sell"), conf

def process_symbol(symbol, timeframe):
    df = fetch_latest_ohlcv(symbol, timeframe=timeframe)
    if df is None or len(df) < 50:
        return None
    df = calculate_indicators(df)
    ind = detect_signal(df)
    signal, conf = infer_direction(ind)
    if signal is None:
        return None
    price = df["close"].iloc[-1]
    return {
        "symbol": symbol,
        "signal": signal,
        "confidence": conf,
        "indicator_states": ind,
        "price": price,
        "leverage": 10,
        "ttp_pct": 3.0,
        "stop_loss_pct": 1.0,
        "units": 1,
        "pos_usd": price,
        "comment": "Multi-indicator alignment detected."
    }

def format_indicator_states(states):
    return "\n".join(
        f"  • {k}: {'✅ Up' if v=='up' else '❌ Down' if v=='down' else '⚪ Neutral'}"
        for k,v in states.items()
    )

def format_signal_detail(res, highlight=False):
    emoji = "🟢 BUY" if res["signal"] == "buy" else "🔴 SELL"
    if highlight: emoji = "🌟 " + emoji
    return (
        f"<b>{res['symbol']} | {emoji} ({res['confidence']:.1f}%)</b>\n"
        f"  <b>Price:</b> <code>{res['price']:.2f}</code>\n"
        f"  <b>TP:</b> <code>{res['ttp_pct']}%</code> | SL: <code>{res['stop_loss_pct']}%</code>\n"
        f"  <b>Exposure:</b> ${res['pos_usd']:.2f} @ {res['leverage']}x\n"
        f"  <i>{res['comment']}</i>\n"
        f"{format_indicator_states(res['indicator_states'])}\n"
    )

def format_summary(results, timestamp, matches, main_tf, compare_tf):
    header = (
        f"📊 <b>{main_tf.upper()} Scan @ {timestamp}</b>\n"
        f"<i>Compared against {compare_tf.upper()} timeframe</i>\n"
    )
    if matches:
        header += "\n<u>🔁 Confirmed across timeframes:</u>\n"
        for m in matches:
            header += f"  • {m} ✅\n"

    sorted_res = sorted(results, key=lambda x: x["confidence"], reverse=True)
    body = "<b>⭐ Top Entries:</b>\n" + "".join(
        format_signal_detail(r, r["symbol"] in matches) for r in sorted_res[:5]
    )
    return header + "\n" + body

# === SCHEDULED TASKS ===

def run_scan_batch(tf):
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(process_symbol, sym, tf): sym for sym in CRYPTO_SYMBOLS}
        for f in as_completed(futures):
            res = f.result()
            if res: results.append(res)
    return results

def compare_and_send(main_tf, compare_tf):
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    results_main = run_scan_batch(main_tf)
    results_comp = run_scan_batch(compare_tf)
    main_syms = set(r["symbol"] for r in sorted(results_main, key=lambda x: x["confidence"], reverse=True)[:5])
    comp_syms = set(r["symbol"] for r in sorted(results_comp, key=lambda x: x["confidence"], reverse=True)[:5])
    matches = main_syms & comp_syms
    msg = format_summary(results_main, timestamp, matches, main_tf, compare_tf)
    send_telegram_message(msg)

def schedule_job_00():
    compare_and_send("8h", "6h")

def schedule_job_08():
    compare_and_send("8h", "6h")

def schedule_job_18():
    compare_and_send("8h", "6h")

# === MAIN LOOP ===

def main():
    start_webserver()
    schedule.every().day.at("00:00").do(schedule_job_00)
    schedule.every().day.at("08:00").do(schedule_job_08)
    schedule.every().day.at("18:00").do(schedule_job_18)
    print("⏳ Scheduler active. Waiting...")

    try:
        while True:
            schedule.run_pending()
            time.sleep(20)
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user. Goodbye!")

if __name__ == "__main__":
    main()