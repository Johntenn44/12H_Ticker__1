import os
import sys
import subprocess
import time
import math
import random
import atexit
import threading
import requests
import ccxt
import pandas as pd
import numpy as np
import schedule
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict, deque
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

# =============================
# Configuration and Constants
# =============================

CONFIG = {
    '6h': {'desc': '6h TF', 'interval_hours': 6, 'bars_per_month': 120},
    '8h': {'desc': '8h TF', 'interval_hours': 8, 'bars_per_month': 90},
}

BACKTEST_PERIODS = {
    '1m': 1,
    '2m': 2,
    '3m': 3,
    '5m': 5,
}

MAX_WORKERS = 4
MAX_RISK_PER_TRADE_USD = 10
BASE_MAX_LEVERAGE = 20

CRYPTO_SYMBOLS = [
    "XRP/USDT", "SUI/USDT", "DOGE/USDT", "XLM/USDT", "TRX/USDT",
    "AAVE/USDT", "LTC/USDT", "ARB/USDT", "NEAR/USDT", "MNT/USDT",
    "VIRTUAL/USDT", "XMR/USDT", "EIGEN/USDT", "CAKE/USDT", "SUSHI/USDT",
    "GRASS/USDT", "WAVES/USDT", "APE/USDT", "SKL/USDT", "PLUME/USDT",
    "LUNA/USDT"
]

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
TELEGRAM_CHANNEL_ID = os.getenv("TELEGRAM_CHANNEL_ID")
TARGET_CHAT_IDS = list(filter(None, [TELEGRAM_CHANNEL_ID, TELEGRAM_CHAT_ID]))

# =============================
# Webserver subprocess management
# =============================
webserver_proc = None

def start_webserver():
    global webserver_proc
    try:
        webserver_path = os.path.join(os.path.dirname(__file__), 'webserver.py')
        print("▶️ Starting webserver subprocess...")
        webserver_proc = subprocess.Popen([sys.executable, webserver_path])
    except Exception as e:
        print(f"⚠️ Could not start webserver.py: {e}")

def stop_webserver():
    global webserver_proc
    if webserver_proc:
        print("🛑 Terminating webserver subprocess...")
        webserver_proc.terminate()
        webserver_proc.wait()

atexit.register(stop_webserver)

# =============================
# Exchange Initialization
# =============================
exchange = None
exchange_lock = threading.Lock()

def initialize_exchange():
    global exchange
    if exchange is None:
        try:
            exchange = ccxt.kucoin()
            exchange.load_markets()
            print("✅ Exchange initialized.")
        except Exception as e:
            print(f"❌ Exchange initialization failed: {e}")
            sys.exit(1)

initialize_exchange()

# =============================
# Telegram Messaging
# =============================
def send_telegram_message(msg):
    if not TELEGRAM_BOT_TOKEN:
        print("❌ Telegram BOT TOKEN missing.")
        return
    for chat_id in TARGET_CHAT_IDS:
        try:
            r = requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                data={
                    "chat_id": chat_id,
                    "text": msg,
                    "parse_mode": "HTML",
                    "disable_web_page_preview": True,
                },
                timeout=15,
            )
            if not r.ok:
                print(f"Telegram API error: {r.text}")
        except Exception as e:
            print(f"Telegram message send exception: {e}")

# =============================
# Indicator Calculations
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
    pct = pct.fillna(method="ffill")
    k = pct.rolling(window=smooth_k).mean()
    d = k.rolling(window=smooth_d).mean()
    return k, d

def calculate_multi_wr(df, lengths=(3, 8, 13, 55, 144, 233)):
    wr = {}
    for L in lengths:
        hh = df["high"].rolling(L).max()
        ll = df["low"].rolling(L).min()
        wr_val = (hh - df["close"]) / (hh - ll)
        wr_val = wr_val.replace([np.inf, -np.inf], np.nan).fillna(method="ffill")
        wr[L] = (wr_val * -100).fillna(method="ffill")
    return wr

def calculate_kdj(df, length=5, ma1=8, ma2=8):
    if len(df) < length:
        nan_series = pd.Series(np.nan, index=df.index)
        return nan_series, nan_series, nan_series
    low_min = df["low"].rolling(length, min_periods=1).min()
    high_max = df["high"].rolling(length, min_periods=1).max()
    denom = (high_max - low_min).replace(0, np.nan)
    rsv_raw = (df["close"] - low_min) / denom * 100
    rsv = rsv_raw.copy().fillna(method="ffill").fillna(50)
    k = rsv.ewm(span=ma1, adjust=False).mean()
    d = k.ewm(span=ma2, adjust=False).mean()
    j = 3 * k - 2 * d
    return k, d, j

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
    wrs = calculate_multi_wr(df)
    for L, series in wrs.items():
        df[f"wr_{L}"] = series
    kdj_k, kdj_d, kdj_j = calculate_kdj(df)
    df["kdj_k"], df["kdj_d"], df["kdj_j"] = kdj_k, kdj_d, kdj_j
    macd_line, macd_signal = calculate_macd(df["close"])
    df["macd_line"], df["macd_signal"] = macd_line, macd_signal
    bb_upper, bb_lower = calculate_bollinger_bands(df["close"])
    df["bb_upper"], df["bb_lower"] = bb_upper, bb_lower
    return df

# =============================
# Indicator signal interpretation
# =============================
def _trend(series1, series2, idx):
    if idx < 1 or pd.isna(series1.iloc[idx]) or pd.isna(series2.iloc[idx]):
        return None
    if series1.iloc[idx] > series2.iloc[idx]:
        return "up"
    if series1.iloc[idx] < series2.iloc[idx]:
        return "down"
    return None

def _wr_position(wr_dict, idx):
    try:
        w3, w8, w55, w144, w233 = (wr_dict[x].iloc[idx] for x in (3, 8, 55, 144, 233))
    except Exception:
        return None
    if w8 > w233 and w3 > w233 and w3 > w144 and w8 > w144:
        return "up"
    if w8 < w55 and w3 < w55:
        return "down"
    return None

INDICATOR_FUNCTIONS = {
    "Stoch RSI": lambda df, i: _trend(df["stochrsi_k"], df["stochrsi_d"], i),
    "Williams %R": lambda df, i: _wr_position({L: df[f"wr_{L}"] for L in (3, 8, 55, 144, 233)}, i),
    "RSI": lambda df, i: (
        "up" if df["rsi5"].iloc[i] > df["rsi13"].iloc[i] > df["rsi21"].iloc[i]
        else "down" if df["rsi5"].iloc[i] < df["rsi13"].iloc[i] < df["rsi21"].iloc[i]
        else None),
    "KDJ": lambda df, i: _trend(df["kdj_j"], df["kdj_d"], i),
    "MACD": lambda df, i: _trend(df["macd_line"], df["macd_signal"], i),
    "Bollinger": lambda df, i: (
        "up" if df["close"].iloc[i] < df["bb_lower"].iloc[i]
        else "down" if df["close"].iloc[i] > df["bb_upper"].iloc[i]
        else None),
}

# =============================
# Performance and ML State: Track full combos and single-indicator signals separately
# =============================

performance_tracker = {
    tf: {p: {"high_vol": {}, "low_vol": {}, "unknown": {}} for p in BACKTEST_PERIODS.keys()}
    for tf in CONFIG.keys()
}

single_perf_tracker = {
    tf: {p: {"high_vol": {}, "low_vol": {}, "unknown": {}} for p in BACKTEST_PERIODS.keys()}
    for tf in CONFIG.keys()
}

ml_X = deque(maxlen=1000)
ml_y = deque(maxlen=1000)

base_ml_model = LogisticRegression(solver="liblinear", random_state=42)
calibrated_ml_model = None
ml_trained = False
ml_lock = threading.Lock()

# Market regime detection
def _market_regime(df, window=50, threshold=0.02):
    if len(df) < window:
        return "unknown"
    vol = df["close"].pct_change().rolling(window).std().iloc[-1]
    return "high_vol" if vol > threshold else "low_vol"

# Update combo performance
def _update_indicator_performance(timeframe, period_label, combo_key, regime, correct):
    if regime == "unknown":
        return
    perf = performance_tracker[timeframe][period_label][regime]
    if combo_key not in perf:
        perf[combo_key] = [0, 0]  # [correct, total]
    perf[combo_key][1] += 1
    if correct:
        perf[combo_key][0] += 1

# Update single indicator signal performance
def _update_single_indicator_performance(timeframe, period_label, ind_name, ind_signal, regime, correct):
    if regime == "unknown":
        return
    perf = single_perf_tracker[timeframe][period_label][regime]
    key = (ind_name, ind_signal)
    if key not in perf:
        perf[key] = [0, 0]
    perf[key][1] += 1
    if correct:
        perf[key][0] += 1

def _add_ml_sample(signals, price_move):
    feats = [(1 if s == "up" else -1 if s == "down" else 0) for s in signals.values()]
    ml_X.append(feats)
    ml_y.append(1 if price_move > 0 else 0)

def _train_ml_model():
    global ml_trained, calibrated_ml_model
    with ml_lock:
        if len(ml_X) > len(INDICATOR_FUNCTIONS) * 5:
            try:
                base_ml_model.fit(np.array(ml_X), np.array(ml_y))
                calibrated_ml_model = CalibratedClassifierCV(base_ml_model, cv="prefit", method="sigmoid")
                calibrated_ml_model.fit(np.array(ml_X), np.array(ml_y))
                ml_trained = True
                print("🧠 ML model trained and calibrated.")
            except Exception as e:
                print(f"ML training failed: {e}")
                ml_trained = False
                calibrated_ml_model = None
        else:
            ml_trained = False
            calibrated_ml_model = None

def _ml_predict(signals):
    if not ml_trained or calibrated_ml_model is None:
        return None, 0.0
    feats = np.array([(1 if s == "up" else -1 if s == "down" else 0)
                      for s in signals.values()]).reshape(1, -1)
    try:
        proba = calibrated_ml_model.predict_proba(feats)[0]
        max_proba = max(proba)
        if max_proba < 0.55:
            return None, 0.0
        return ("buy", proba[1] * 100) if proba[1] > proba[0] else ("sell", proba[0] * 100)
    except Exception as e:
        print(f"ML prediction error: {e}")
        return None, 0.0

# Get performance weight for full combo (used for rating)
def _get_combo_performance_weight(timeframe, period_label, combo_key, regime):
    perf = performance_tracker.get(timeframe, {}).get(period_label, {}).get(regime, {})
    if combo_key not in perf:
        return None
    correct, total = perf[combo_key]
    if total < 10:  # minimum samples for trust
        return None
    return correct / total

# Get performance weight for single indicator signal (used in adaptive voting)
def _get_single_perf_weight(timeframe, period_label, ind_key, regime):
    perf = single_perf_tracker.get(timeframe, {}).get(period_label, {}).get(regime, {})
    if ind_key not in perf:
        return None
    correct, total = perf[ind_key]
    if total < 10:  # minimum samples for trust
        return None
    return correct / total

# Adaptive signal with weighting from single_perf_tracker and ML
def adaptive_signal(df, timeframe):
    regime = _market_regime(df)
    signals = {n: fn(df, -1) for n, fn in INDICATOR_FUNCTIONS.items()}
    combo = tuple(sorted((k, v) for k, v in signals.items() if v is not None))

    # Compute weighted average combo performance rating
    weighted_combo_perf = []
    total_w = 0
    for period_label, period_months in BACKTEST_PERIODS.items():
        w = 1.0 / period_months
        weight_val = _get_combo_performance_weight(timeframe, period_label, combo, regime)
        if weight_val is not None:
            weighted_combo_perf.append(w * weight_val)
            total_w += w
    weight_combo = (sum(weighted_combo_perf) / total_w) if total_w > 0 else 0.5

    ml_sig, ml_conf = _ml_predict(signals)
    if ml_sig:
        return ml_sig, ml_conf, signals, regime, weight_combo

    # Adaptive voting weighted by single indicator performance
    up_w = 0
    down_w = 0
    total = 0
    for n, s in signals.items():
        ind_key = (n, s)
        weight_votes = []
        total_ind_w = 0
        for period_label, period_months in BACKTEST_PERIODS.items():
            w = 1.0 / period_months
            perf_weight = _get_single_perf_weight(timeframe, period_label, ind_key, regime)
            if perf_weight is not None:
                weight_votes.append(w * perf_weight)
                total_ind_w += w
        weight_indicator = (sum(weight_votes) / total_ind_w) if total_ind_w > 0 else 0.5

        if s == 'up':
            up_w += weight_indicator
            total += weight_indicator
        elif s == 'down':
            down_w += weight_indicator
            total += weight_indicator

    if total > 0.1:
        if up_w > down_w * 1.2:
            return "buy", (up_w / total) * 100, signals, regime, weight_combo
        if down_w > up_w * 1.2:
            return "sell", (down_w / total) * 100, signals, regime, weight_combo

    if all(s == "up" for s in signals.values() if s is not None):
        avg_conf = weight_combo * 100
        return "buy", avg_conf, signals, regime, weight_combo
    if all(s == "down" for s in signals.values() if s is not None):
        avg_conf = weight_combo * 100
        return "sell", avg_conf, signals, regime, weight_combo

    return None, 0.0, signals, regime, weight_combo

# Backtesting update for combos and single indicators
def backtest_update_multi_period(df, timeframe):
    for period_label, months in BACKTEST_PERIODS.items():
        bars_needed = months * CONFIG[timeframe]['bars_per_month']
        if len(df) < bars_needed + 2:
            continue
        sub_df = df.tail(bars_needed + 2)
        regime = _market_regime(sub_df)
        for i in range(50, len(sub_df) - 1):
            sigs = {n: fn(sub_df, i) for n, fn in INDICATOR_FUNCTIONS.items()}
            combo_key = tuple(sorted((k, v) for k, v in sigs.items() if v is not None))
            p0, p1 = sub_df["close"].iloc[i], sub_df["close"].iloc[i + 1]
            ups = sum(1 for v in sigs.values() if v == "up")
            downs = sum(1 for v in sigs.values() if v == "down")
            correct = None
            if ups > downs:
                correct = p1 > p0
            elif downs > ups:
                correct = p1 < p0
            if correct is None:
                continue

            # Update full combo performance
            _update_indicator_performance(timeframe, period_label, combo_key, regime, correct)
            # Update individual indicators performance
            for ind_name, ind_sig in sigs.items():
                if ind_sig is not None:
                    _update_single_indicator_performance(timeframe, period_label, ind_name, ind_sig, regime, correct)
            _add_ml_sample(sigs, p1 - p0)
    _train_ml_model()

# =============================
# OHLCV fetch and process symbols
# =============================
def fetch_latest_ohlcv(symbol, timeframe='6h', limit=750):
    try:
        with exchange_lock:
            if symbol not in exchange.symbols:
                print(f"{symbol} not on exchange - skipping")
                return None
            data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        df.dropna(subset=["close"], inplace=True)
        return df
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None

def process_symbol(symbol, timeframe):
    print(f"Processing {symbol} for {timeframe} timeframe…")
    start_time = time.time()

    df = fetch_latest_ohlcv(symbol, timeframe=timeframe)
    if df is None or len(df) < 100:
        print(f"{symbol}: insufficient data")
        return None

    df = calculate_indicators(df)
    sub_df = df.tail(750)
    if len(sub_df) > 100:
        backtest_update_multi_period(sub_df, timeframe)

    sig, conf, states, regime, rating = adaptive_signal(df, timeframe)
    if sig not in ("buy", "sell") or conf < 55.0:
        print(f"{symbol}: weak signal ({sig}/{conf:.1f}%), skipping")
        return None

    price = df["close"].iloc[-1]
    leverage = BASE_MAX_LEVERAGE
    if regime == "high_vol":
        leverage = int(BASE_MAX_LEVERAGE * 0.5)
    elif regime == "low_vol":
        leverage = int(BASE_MAX_LEVERAGE * 1.0)

    ttp = 3.0 * (1 + rating/100 * 0.5)
    sl = 1.0 * (1 - rating/150)

    risk_per_unit = price * sl / 100
    if risk_per_unit == 0:
        print(f"{symbol}: zero risk per unit")
        return None

    units = math.floor(MAX_RISK_PER_TRADE_USD / risk_per_unit)
    if units == 0:
        print(f"{symbol}: position size too small for risk tolerance")
        return None

    pos_usd = units * price
    margin_needed = pos_usd / leverage

    comment = (f"ML conf: {conf:.1f}%, perf rating: {rating:.1f}%, regime: {regime}, "
               f"{leverage}x lev, TTP {ttp:.2f}%, SL {sl:.2f}%")

    print(f"{symbol} finished in {time.time() - start_time:.1f}s, signal={sig} ({conf:.1f}%), rating={rating:.1f}%")

    return {
        "symbol": symbol,
        "signal": sig,
        "confidence": conf,
        "indicator_states": states,
        "regime": regime,
        "price": price,
        "ttp_pct": ttp,
        "stop_loss_pct": sl,
        "leverage": leverage,
        "units": units,
        "pos_usd": pos_usd,
        "margin": margin_needed,
        "comment": comment,
        "rating": rating,
    }

# =============================
# Telegram message formatting
# =============================
def format_indicator_states(states):
    lines = []
    for name, s in states.items():
        symbol = '✅ Up' if s == 'up' else '❌ Down' if s == 'down' else '⚪ Neutral'
        lines.append(f"  • {name}: {symbol}")
    return "\n".join(lines)

def format_single_signal_detail(res, highlight=False):
    emoji = "🟢 BUY" if res["signal"] == "buy" else "🔴 SELL"
    if highlight:
        emoji = "🌟 " + emoji
    rating_str = f"• <b>Perf Rating:</b> <code>{res['rating']:.1f}%</code>\n" if res.get("rating") else ""
    return (
        f"<b>{res['symbol']} | {emoji} ({res['confidence']:.1f}%)</b>\n"
        f"  <b>Price:</b> <code>{res['price']:.4f}</code>\n"
        f"  <b>Leverage:</b> <code>{res['leverage']}x</code>\n"
        f"  <b>Trailing TP:</b> <code>{res['ttp_pct']:.2f}%</code>\n"
        f"  <b>Stop Loss:</b> <code>{res['stop_loss_pct']:.2f}%</code>\n"
        f"  <b>Position:</b> <code>{res['units']} units ≈ ${res['pos_usd']:.2f}</code>\n"
        f"{rating_str}"
        f"  <i>{res['comment']}</i>\n"
        f"  <u>Indicators:</u>\n{format_indicator_states(res['indicator_states'])}\n"
    )

def format_summary_message_with_cross_tf(signal_results, timestamp_str, matched_symbols, main_tf, compare_tf):
    if not signal_results:
        return f"<b>📊 No strong signals on {main_tf.upper()} at {timestamp_str}</b>"
    sorted_results = sorted(signal_results, key=lambda d: d["confidence"], reverse=True)
    greeting = ("🌅 Good morning" if 5 <= datetime.utcnow().hour < 12 else
                "🌞 Good afternoon" if 12 <= datetime.utcnow().hour < 18 else
                "🌙 Good evening")

    header = (
        f"{greeting}, trader!\n"
        f"<b>📊 {main_tf.upper()} Scan @ {timestamp_str}</b>\n"
        f"<i>Compared against {compare_tf.upper()} timeframe</i>\n"
    )
    if matched_symbols:
        header += f"\n<u>🔗 Dual-Timeframe Matches ({len(matched_symbols)}):</u>\n"
        for sym in sorted(matched_symbols):
            header += f"  • {sym} ✅\n"

    details = "<b>⭐ Top Signals:</b>\n" + "".join(
        format_single_signal_detail(r, highlight=(r["symbol"] in matched_symbols))
        for r in sorted_results[:5]
    )
    return header + "\n" + details

# =============================
# Scheduled scan runner and comparison
# =============================
def run_scan(timeframe):
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_symbol, sym, timeframe): sym for sym in CRYPTO_SYMBOLS}
        for fut in as_completed(futures):
            res = fut.result()
            if res:
                results.append(res)
    return results

def compare_and_send(main_tf, compare_tf):
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    main_results = run_scan(main_tf)
    compare_results = run_scan(compare_tf)

    top_main_syms = set(r["symbol"] for r in sorted(main_results, key=lambda x: x["confidence"], reverse=True)[:5])
    top_compare_syms = set(r["symbol"] for r in sorted(compare_results, key=lambda x: x["confidence"], reverse=True)[:5])
    matched = top_main_syms & top_compare_syms

    message = format_summary_message_with_cross_tf(main_results, timestamp, matched, main_tf, compare_tf)
    send_telegram_message(message)

# =============================
# Jobs at scheduled UTC times
# =============================
def job_0000():
    compare_and_send("8h", "6h")

def job_0800():
    compare_and_send("8h", "6h")

def job_1800():
    compare_and_send("8h", "6h")

# =============================
# Main Loop
# =============================
def main():
    start_webserver()

    schedule.every().day.at("00:00").do(job_0000)
    schedule.every().day.at("08:00").do(job_0800)
    schedule.every().day.at("18:00").do(job_1800)

    print("⏳ Scheduler started. Waiting for scheduled scans at 00:00, 08:00, and 18:00 UTC.")

    # Optional: Run a scan immediately on start, comment if undesired
    job_0000()

    try:
        while True:
            schedule.run_pending()
            time.sleep(15)
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user. Exiting...")

if __name__ == "__main__":
    main()
