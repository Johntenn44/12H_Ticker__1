import os
import sys
import subprocess
import requests
import ccxt
import pandas as pd
import numpy as np
import time
import math
import random
import schedule
from datetime import datetime, timedelta
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
import atexit
import threading

# =============================
# CONFIGURATION
# =============================
CONFIG = {
    '6h': {'desc': '6h TF', 'interval_hours': 6},
    '8h': {'desc': '8h TF', 'interval_hours': 8},
}

MAX_RISK_PER_TRADE_USD = 10
BASE_MAX_LEVERAGE = 20
MAX_WORKERS = 4

# =============================
# GLOBAL INIT
# =============================
exchange = None
exchange_lock = threading.Lock()

TIMEFRAME = '8h'
CANDLE_DESC = CONFIG[TIMEFRAME]['desc']

CRYPTO_SYMBOLS = [
    "XRP/USDT", "SUI/USDT", "DOGE/USDT", "XLM/USDT", "TRX/USDT",
    "AAVE/USDT", "LTC/USDT", "ARB/USDT", "NEAR/USDT", "MNT/USDT",
    "VIRTUAL/USDT", "XMR/USDT", "EIGEN/USDT", "CAKE/USDT", "SUSHI/USDT",
    "GRASS/USDT", "WAVES/USDT", "APE/USDT", "SKL/USDT", "PLUME/USDT",
    "LUNA/USDT",
]

# =============================
# TELEGRAM CONFIG
# =============================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
TELEGRAM_CHANNEL_ID = os.getenv("TELEGRAM_CHANNEL_ID")

def send_telegram_message(msg):
    if not TELEGRAM_BOT_TOKEN:
        print("Telegram BOT TOKEN missing")
        return
    TARGET_IDS = list(filter(None, [TELEGRAM_CHAT_ID, TELEGRAM_CHANNEL_ID]))
    for chat_id in TARGET_IDS:
        try:
            r = requests.post(
                f'https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage',
                data={
                    'chat_id': chat_id,
                    'text': msg,
                    'parse_mode': 'HTML',
                    'disable_web_page_preview': True
                }
            )
            if not r.ok:
                print(f"❌ Failed to send message: {r.text}")
        except Exception as e:
            print(f"Error sending message: {e}")

# =============================
# EXCHANGE INIT
# =============================
def initialize_exchange():
    global exchange
    if exchange is None:
        try:
            exchange = ccxt.kucoin()
            exchange.load_markets()
            print("✅ Exchange initialized.")
        except Exception as e:
            print(f"Exchange init fail: {e}")
            sys.exit(1)

initialize_exchange()

# =============================
# INDICATOR FUNCTION SUITE
# =============================

def calculate_rsi(series, period=13):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_stoch_rsi(df, rsi_len=13, stoch_len=8, smooth_k=5, smooth_d=3):
    rsi = calculate_rsi(df["close"], rsi_len)
    min_rsi = rsi.rolling(window=stoch_len).min()
    max_rsi = rsi.rolling(window=stoch_len).max()
    pct = (rsi - min_rsi) / (max_rsi - min_rsi) * 100
    k = pct.rolling(window=smooth_k).mean()
    d = k.rolling(window=smooth_d).mean()
    return k, d

def calculate_macd(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    sig_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, sig_line

def calculate_bollinger_bands(close, window=20, num_std=2):
    sma = close.rolling(window).mean()
    std = close.rolling(window).std()
    return sma + num_std * std, sma - num_std * std

def calculate_indicators(df):
    df = df.copy()
    df["close"] = df["close"].astype(float)
    df["rsi5"] = calculate_rsi(df["close"], 5)
    df["rsi13"] = calculate_rsi(df["close"], 13)
    df["rsi21"] = calculate_rsi(df["close"], 21)
    k, d = calculate_stoch_rsi(df)
    df["stochrsi_k"], df["stochrsi_d"] = k, d
    macd, sig = calculate_macd(df["close"])
    df["macd_line"], df["macd_signal"] = macd, sig
    upper, lower = calculate_bollinger_bands(df["close"])
    df["bb_upper"], df["bb_lower"] = upper, lower
    return df

# =============================
# FETCH DATA AND PROCESS SYMBOL
# =============================

def fetch_latest_ohlcv(symbol, timeframe='6h', limit=200):
    try:
        with exchange_lock:
            data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        df.dropna(inplace=True)
        return df
    except Exception as e:
        print(f"{symbol} fetch error: {e}")
        return None

def evaluate_indicators(df):
    signals = {}
    try:
        idx = -1
        if df["stochrsi_k"].iloc[idx] > df["stochrsi_d"].iloc[idx]:
            signals["StochRSI"] = "up"
        elif df["stochrsi_k"].iloc[idx] < df["stochrsi_d"].iloc[idx]:
            signals["StochRSI"] = "down"

        if df["rsi5"].iloc[idx] > df["rsi13"].iloc[idx] > df["rsi21"].iloc[idx]:
            signals["RSI"] = "up"
        elif df["rsi5"].iloc[idx] < df["rsi13"].iloc[idx] < df["rsi21"].iloc[idx]:
            signals["RSI"] = "down"

        if df["macd_line"].iloc[idx] > df["macd_signal"].iloc[idx]:
            signals["MACD"] = "up"
        elif df["macd_line"].iloc[idx] < df["macd_signal"].iloc[idx]:
            signals["MACD"] = "down"

        close = df["close"].iloc[idx]
        if close > df["bb_upper"].iloc[idx]:
            signals["Boll"] = "down"
        elif close < df["bb_lower"].iloc[idx]:
            signals["Boll"] = "up"
    except:
        pass
    return signals

def adaptive_signal(df):
    signals = evaluate_indicators(df)
    ups = sum(1 for s in signals.values() if s == "up")
    downs = sum(1 for s in signals.values() if s == "down")
    total = len(signals)

    if total == 0:
        return None, 0.0, signals

    conf = max(ups, downs) / total * 100
    direction = "buy" if ups > downs else "sell" if downs > ups else None
    return direction, conf, signals

def process_symbol(symbol, timeframe):
    df = fetch_latest_ohlcv(symbol, timeframe=timeframe)
    if df is None or len(df) < 50:
        return None
    df = calculate_indicators(df)
    direction, confidence, indicators = adaptive_signal(df)
    if direction and confidence >= 55.0:
        return {
            "symbol": symbol,
            "signal": direction,
            "confidence": confidence,
            "indicator_states": indicators,
            "price": df["close"].iloc[-1],
            "regime": "high_vol",  # placeholder
            "ttp_pct": 3.0,
            "stop_loss_pct": 1.5,
            "leverage": 10,
            "units": 1,
            "pos_usd": df["close"].iloc[-1],
            "comment": "Dynamic signal",
        }
    return None

def format_indicator_states(states):
    return "\n".join(
        f"  • {k}: {'✅ Up' if v == 'up' else '❌ Down' if v == 'down' else '⚪ Neutral'}"
        for k, v in states.items()
    )

def format_single_signal_detail(res, highlight=False):
    emoji = "🟢 BUY" if res["signal"] == "buy" else "🔴 SELL"
    if highlight:
        emoji = "🌟 " + emoji
    price_str = f"{res['price']:.4f}" if res["price"] < 10 else f"{res['price']:.2f}"
    conf = res["confidence"]
    return (
        f"<b>{res['symbol']} | {emoji} ({conf:.1f}%)</b>\n"
        f"  <b>Price:</b> <code>{price_str}</code>\n"
        f"  <b>Leverage:</b> <code>{res['leverage']}x</code>\n"
        f"  <b>TP:</b> <code>{res['ttp_pct']}%</code> | SL: <code>{res['stop_loss_pct']}%</code>\n"
        f"  <i>{res['comment']}</i>\n"
        f"{format_indicator_states(res['indicator_states'])}\n"
    )

def format_summary_message_with_cross_tf(signal_results, timestamp_str, matched_symbols, main_tf, compare_tf):
    if not signal_results:
        return f"<b>📊 No strong signals on {main_tf.upper()} at {timestamp_str}</b>"

    sorted_results = sorted(signal_results, key=lambda x: x["confidence"], reverse=True)
    greeting = random.choice(["🌅 Good morning", "🌞 Good afternoon", "🌙 Good evening"])
    header = f"{greeting}, trader!\n<b>📊 Scan @ {timestamp_str} ({main_tf.upper()})</b>\nCompared to {compare_tf.upper()}\n"

    if matched_symbols:
        header += "\n<u>🔗 Confirmed across timeframes:</u>\n" + "\n".join(f"  • {s} ✅" for s in matched_symbols)

    details = "<b>⭐ Top Entries:</b>\n" + "".join(
        format_single_signal_detail(r, highlight=r["symbol"] in matched_symbols)
        for r in sorted_results[:5]
    )
    return header + "\n" + details

# =============================
# SCHEDULING
# =============================

def run_scan_batch(tf):
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_symbol, sym, tf): sym for sym in CRYPTO_SYMBOLS}
        for future in as_completed(futures):
            res = future.result()
            if res:
                results.append(res)
    return results

def compare_and_send(main_tf, compare_tf):
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    print(f"⏳ Running {main_tf} scan @ {ts}")
    res_main = run_scan_batch(main_tf)
    res_compare = run_scan_batch(compare_tf)

    top_main = set(r["symbol"] for r in sorted(res_main, key=lambda x: x["confidence"], reverse=True)[:5])
    top_other = set(r["symbol"] for r in sorted(res_compare, key=lambda x: x["confidence"], reverse=True)[:5])
    matches = top_main & top_other

    msg = format_summary_message_with_cross_tf(res_main, ts, matches, main_tf, compare_tf)
    send_telegram_message(msg)

def job_at_0000():
    compare_and_send("8h", "6h")

def job_at_0800():
    compare_and_send("8h", "6h")

def job_at_1800():
    compare_and_send("8h", "6h")

def sleep_in_chunks(seconds):
    while seconds > 0:
        block = min(60, seconds)
        time.sleep(block)
        seconds -= block

def main():
    schedule.every().day.at("00:00").do(job_at_0000)
    schedule.every().day.at("08:00").do(job_at_0800)
    schedule.every().day.at("18:00").do(job_at_1800)

    print("🔔 Scheduler active · Waiting for next trigger...")
    while True:
        schedule.run_pending()
        time.sleep(20)

if __name__ == "__main__":
    main()