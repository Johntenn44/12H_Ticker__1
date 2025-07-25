import subprocess
import sys
import os
import atexit
import requests
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict, deque
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
import math
import random

# ========================================================================
# Global Settings for 12-hour candles
# ========================================================================
TIMEFRAME = '12h'            # 12-hour candle timeframe
CANDLE_DESC = '12h TF'       # human label for messages
MAX_WORKERS = 4              # thread-pool workers
MAX_RISK_PER_TRADE_USD = 10  # risk capital per trade
BASE_MAX_LEVERAGE = 20       # hard cap on leverage

# Launch webserver subprocess if needed
webserver_path = os.path.join(os.path.dirname(__file__), 'webserver.py')
webserver_proc = subprocess.Popen([sys.executable, webserver_path])

def _cleanup():
    print("Terminating webserver subprocess …")
    webserver_proc.terminate()

atexit.register(_cleanup)

# Telegram config
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
TELEGRAM_CHANNEL_ID = os.environ.get("TELEGRAM_CHANNEL_ID")

def send_telegram_message(msg, chat_id_override=None):
    if not TELEGRAM_BOT_TOKEN:
        print("Telegram token not set, cannot send messages.")
        return
    MAX_LEN = 4096
    target_ids = [chat_id_override] if chat_id_override else list(
        filter(None, [TELEGRAM_CHAT_ID, TELEGRAM_CHANNEL_ID])
    )
    parts = []
    if len(msg) > MAX_LEN:
        buf = ""
        for line in msg.splitlines():
            if len(buf) + len(line) + 1 <= MAX_LEN:
                buf += line + "\n"
            else:
                parts.append(buf)
                buf = line + "\n"
        if buf:
            parts.append(buf)
    else:
        parts = [msg]

    for part in parts:
        for chat_id in target_ids:
            try:
                r = requests.post(
                    f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                    data={
                        "chat_id": chat_id,
                        "text": part,
                        "parse_mode": "HTML",
                        "disable_web_page_preview": True,
                    },
                    timeout=15,
                )
                if not r.ok:
                    print(f"Failed to send Telegram message to {chat_id}: {r.text}")
            except Exception as e:
                print(f"Exception sending Telegram message: {e}")

# Exchange Initialization
exchange = None
exchange_lock = threading.Lock()

def initialize_exchange():
    global exchange
    if exchange is None:
        try:
            exchange = ccxt.kucoin()
            exchange.load_markets()
            print("CCXT exchange initialized.")
        except Exception as e:
            print(f"Exchange initialization error: {e}")
            sys.exit(1)

initialize_exchange()

# Crypto symbols list
CRYPTO_SYMBOLS = [
    "XRP/USDT", "SUI/USDT", "DOGE/USDT", "XLM/USDT", "TRX/USDT",
    "AAVE/USDT", "LTC/USDT", "ARB/USDT", "NEAR/USDT", "MNT/USDT",
    "VIRTUAL/USDT", "XMR/USDT", "EIGEN/USDT", "CAKE/USDT", "SUSHI/USDT",
    "GRASS/USDT", "WAVES/USDT", "APE/USDT", "SKL/USDT", "PLUME/USDT",
    "LUNA/USDT",
]

# ========================================================================
# Indicator calculations
# ========================================================================
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

# ========================================================================
# Indicator signal analysis
# ========================================================================
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

# ========================================================================
# Machine learning performance trackers
# ========================================================================
indicator_perf = defaultdict(lambda: {"high_vol": deque(maxlen=200),
                                      "low_vol": deque(maxlen=200)})
ml_X = deque(maxlen=1000)
ml_y = deque(maxlen=1000)
base_ml_model = LogisticRegression(solver="liblinear", random_state=42)
calibrated_ml_model = None
ml_trained = False
ml_lock = threading.Lock()

def _market_regime(df, window=50, threshold=0.02):
    if len(df) < window:
        return "unknown"
    vol = df["close"].pct_change().rolling(window).std().iloc[-1]
    return "high_vol" if vol > threshold else "low_vol"

def _update_indicator_performance(name, regime, correct):
    if regime == "unknown":
        return
    indicator_perf[name][regime].append(1 if correct else 0)

def _get_indicator_weight(name, regime):
    if regime == "unknown":
        return 0.5
    perf = indicator_perf[name][regime]
    return sum(perf) / len(perf) if perf else 0.5

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
                print("ML model trained and calibrated.")
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

def adaptive_signal(df):
    regime = _market_regime(df)
    signals = {n: fn(df, -1) for n, fn in INDICATOR_FUNCTIONS.items()}
    ml_sig, ml_conf = _ml_predict(signals)
    if ml_sig:
        return ml_sig, ml_conf, signals, regime

    up_w = down_w = total = 0
    for n, s in signals.items():
        w = _get_indicator_weight(n, regime)
        if s == "up":
            up_w += w
            total += w
        if s == "down":
            down_w += w
            total += w
    if total > 0.1:
        if up_w > down_w * 1.2:
            return "buy", (up_w / total) * 100, signals, regime
        if down_w > up_w * 1.2:
            return "sell", (down_w / total) * 100, signals, regime
    if all(s == "up" for s in signals.values() if s is not None):
        avg_conf = np.mean([_get_indicator_weight(n, regime) for n in INDICATOR_FUNCTIONS]) * 100
        return "buy", avg_conf, signals, regime
    if all(s == "down" for s in signals.values() if s is not None):
        avg_conf = np.mean([_get_indicator_weight(n, regime) for n in INDICATOR_FUNCTIONS]) * 100
        return "sell", avg_conf, signals, regime
    return None, 0.0, signals, regime

def backtest_update(df):
    start = max(50, 0)
    if len(df) < start + 2:
        return
    for i in range(start, len(df) - 1):
        regime = _market_regime(df[: i + 1])
        if regime == "unknown":
            continue
        sigs = {n: fn(df, i) for n, fn in INDICATOR_FUNCTIONS.items()}
        p0, p1 = df["close"].iloc[i], df["close"].iloc[i + 1]
        for n, s in sigs.items():
            if s == "up":
                _update_indicator_performance(n, regime, p1 > p0)
            if s == "down":
                _update_indicator_performance(n, regime, p1 < p0)
        _add_ml_sample(sigs, p1 - p0)
    _train_ml_model()

# ========================================================================
# Volatility, risk & adaptive trade parameters
# ========================================================================
def calculate_recent_volatility(df, window=20):
    if len(df) < window + 1:
        return 0.0
    log_returns = np.log(df["close"] / df["close"].shift(1)).dropna()
    vol = log_returns.rolling(window).std().iloc[-1] * np.sqrt(365)
    return vol

def determine_adaptive_leverage(regime, ml_conf, price):
    leverage = BASE_MAX_LEVERAGE * (0.5 if regime == "high_vol"
                                    else 0.8 if regime == "unknown"
                                    else 1.0)
    leverage *= 1 + max(0, ml_conf - 50) / 100
    if price < 1:
        leverage = min(leverage, 5)
    return int(min(max(leverage, 1), BASE_MAX_LEVERAGE))

def determine_dynamic_ttp_and_sl(df, regime, ml_conf, indicator_states):
    base_map = {
        '12h': (6.0, 3.0),
        '1d': (8.0, 4.0),
        '4h': (5.0, 2.5),
        '1h': (3.0, 1.5),
        '8h': (5.5, 2.75),
        '6h': (4.0, 2.0)
    }
    base_ttp, base_sl = base_map.get(TIMEFRAME, (4.0, 2.0))
    if regime == "high_vol":
        base_ttp *= 1.3
        base_sl *= 1.3
    elif regime == "low_vol":
        base_ttp *= 0.8
        base_sl *= 0.8
    vol = min(calculate_recent_volatility(df, 20), 2.0)
    ttp = base_ttp * (1 + vol / 2)
    sl = base_sl * (1 + vol / 2)
    total_ind = len(indicator_states)
    agree = sum(1 for s in indicator_states.values()
                if s == ("up" if ml_conf >= 50 else "down"))
    ind_ratio = agree / total_ind if total_ind else 0.5
    ttp *= 1 + 0.4 * ind_ratio
    sl *= 1 + 0.4 * ind_ratio
    sl *= 1 - 0.3 * min(ml_conf / 100, 1.0)
    ttp = max(1.0, min(ttp, 15.0))
    sl = max(0.5, min(sl, ttp * 0.9))
    return ttp, sl

def calculate_position_size(price, stop_loss_pct, leverage):
    risk_per_unit = price * stop_loss_pct / 100
    if risk_per_unit == 0:
        return 0.0, 0, 0.0
    units = math.floor(MAX_RISK_PER_TRADE_USD / risk_per_unit)
    if units == 0:
        return 0.0, 0, 0.0
    notional = units * price
    margin_needed = notional / leverage
    return notional, units, margin_needed

# ========================================================================
# Dynamic comment generator
# ========================================================================
def generate_dynamic_comment(signal, confidence, leverage,
                             position_size_usd, ttp_percent,
                             regime, indicator_states):
    parts = []
    if signal == "buy":
        parts.append("Strong bullish momentum detected 🚀" if confidence >= 80
                     else "Bullish setup forming 📈" if confidence >= 65
                     else "Cautious buy signal ⚠️")
    else:
        parts.append("Strong bearish momentum detected 🔻" if confidence >= 80
                     else "Bearish setup forming ⚠️" if confidence >= 65
                     else "Cautious sell signal ⚠️")
    parts.append("Ultra-high leverage! Tight risk control." if leverage >= 18
                 else "High leverage in play; manage risk." if leverage >= 10
                 else "Moderate leverage." if leverage >= 5
                 else "Low leverage, conservative exposure.")
    parts.append("Micro position size." if position_size_usd < 1
                 else "Large position — ensure you can absorb draw-downs." if position_size_usd > 50
                 else "Balanced position size.")
    parts.append(f"Trailing TP set at {ttp_percent:.1f}%.")
    total_ind = len(indicator_states)
    agree = sum(1 for s in indicator_states.values()
                if s == ("up" if signal == "buy" else "down"))
    ratio = agree / total_ind if total_ind else 0.0
    parts.append("Indicators strongly align 🔥" if ratio > 0.75
                 else "Most indicators agree." if ratio > 0.5
                 else "Mixed indicator alignment — caution.")
    parts.append("High volatility regime." if regime == "high_vol"
                 else "Low volatility environment." if regime == "low_vol"
                 else "Volatility unclear.")
    return " ".join(parts)

# ========================================================================
# Symbol processing
# ========================================================================
def fetch_latest_ohlcv(symbol, timeframe=TIMEFRAME, limit=750):
    try:
        with exchange_lock:
            if symbol not in exchange.symbols:
                print(f"{symbol} not on exchange - skipping")
                return None
            data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df.dropna(subset=["close"], inplace=True)
        return df
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None

def format_indicator_states(states):
    return "\n".join(f"  • {n}: {'✅ Up' if s == 'up' else '❌ Down' if s == 'down' else '⚪ Neutral'}"
                     for n, s in states.items())

def process_symbol(symbol):
    print(f"Processing {symbol} …")
    start_time = time.time()
    df = fetch_latest_ohlcv(symbol)
    if df is None or len(df) < 100:
        print(f"{symbol}: insufficient data - skipping")
        return None
    df = calculate_indicators(df)
    sub_df = df.tail(750)
    if len(sub_df) > 50:
        backtest_update(sub_df)
    sig, conf, states, regime = adaptive_signal(df)
    if sig not in ("buy", "sell") or conf < 55.0:
        print(f"{symbol}: no strong signal ({sig} / {conf:.1f}%), skipping")
        return None
    price = df["close"].iloc[-1]
    leverage = determine_adaptive_leverage(regime, conf, price)
    ttp, sl_pct = determine_dynamic_ttp_and_sl(df, regime, conf, states)
    pos_usd, units, margin = calculate_position_size(price, sl_pct, leverage)
    if units == 0:
        print(f"{symbol}: position size too small, skipping")
        return None
    comment = generate_dynamic_comment(sig, conf, leverage, pos_usd,
                                       ttp, regime, states)
    print(f"{symbol}: processed in {time.time() - start_time:.2f}s, "
          f"signal {sig} ({conf:.1f}%), lev {leverage}x")
    return {
        "symbol": symbol,
        "signal": sig,
        "confidence": conf,
        "indicator_states": states,
        "regime": regime,
        "price": price,
        "ttp_pct": ttp,
        "stop_loss_pct": sl_pct,
        "units": units,
        "pos_usd": pos_usd,
        "leverage": leverage,
        "comment": comment,
    }

def format_single_signal_detail(res):
    emoji = "🟢 BUY" if res["signal"] == "buy" else "🔴 SELL"
    volemoji = "📈 High Volatility" if res["regime"] == "high_vol" else \
               "📉 Low Volatility" if res["regime"] == "low_vol" else "❓ Unknown"
    price_str = f"{res['price']:.4f}" if res["price"] < 10 else f"{res['price']:.2f}"
    confidence_str = min(res["confidence"], 99.9)
    return (
        f"<b>{res['symbol']} | {emoji} ({confidence_str:.1f}%)</b>\n"
        f"  <b>Price:</b> <code>{price_str}</code>\n"
        f"  <b>Market:</b> {volemoji}\n"
        f"  <b>Leverage:</b> <code>{res['leverage']}x</code>\n"
        f"  <b>Trailing TP:</b> <code>{res['ttp_pct']:.2f}%</code>\n"
        f"  <b>Stop Loss:</b> <code>{res['stop_loss_pct']:.2f}%</code>\n"
        f"  <b>Position:</b> <code>{res['units']} units ≈ ${res['pos_usd']:.2f}</code>\n"
        f"  <i>{res['comment']}</i>\n"
        f"  <u>Indicators:</u>\n{format_indicator_states(res['indicator_states'])}\n"
    )

def format_summary_message(signal_results, timestamp_str):
    if not signal_results:
        return (f"<b>📊 Market Scan @ {timestamp_str} ({CANDLE_DESC})</b>\n\n"
                "<i>No strong BUY/SELL signals detected.</i>")
    sorted_results = sorted(signal_results, key=lambda d: d["confidence"], reverse=True)
    hour = datetime.utcnow().hour
    greeting = ("🌅 Good morning" if 5 <= hour < 12 else
                "🌞 Good afternoon" if 12 <= hour < 18 else
                "🌙 Good evening")
    header = (
        f"{greeting}, trader!\n"
        f"<b>📊 Crypto Signals Scan @ {timestamp_str} ({CANDLE_DESC})</b>\n"
        f"<i>Top {min(5, len(sorted_results))} opportunities:</i>\n"
    )
    perf_lines = "".join(
        f"  • {n}: High Vol {_get_indicator_weight(n, 'high_vol')*100:.0f}% / "
        f"Low Vol {_get_indicator_weight(n, 'low_vol')*100:.0f}%\n"
        for n in INDICATOR_FUNCTIONS
    )
    system_perf = (f"<b>🧠 System Overview:</b>\n{perf_lines}"
                   f"  • ML Model: {'✅ Ready' if ml_trained else '⏳ Training…'}\n\n")
    top_details = "<b>⭐ Top Signals:</b>\n" + \
                  "".join(format_single_signal_detail(r) for r in sorted_results[:5])
    closing_remarks = random.choice([
        "<i>Trade smart, manage risk!</i>",
        "<i>Markets are dynamic. Stay sharp!</i>",
        "<i>Not financial advice.</i>"
    ])
    return header + system_perf + top_details + closing_remarks

# ========================================================================
# Sleep utility to avoid oversleeping and support small chunks
# ========================================================================
def sleep_in_chunks(total_sleep_seconds, chunk_seconds=60):
    remaining = total_sleep_seconds
    while remaining > 0:
        sleep_time = min(chunk_seconds, remaining)
        time.sleep(sleep_time)
        remaining -= sleep_time

def main():
    try:
        while True:
            start_time = datetime.utcnow()
            timestamp = start_time.strftime("%Y-%m-%d %H:%M UTC")
            print(f"\nStarting scan at {timestamp}")
            results = []
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_map = {executor.submit(process_symbol, sym): sym for sym in CRYPTO_SYMBOLS}
                for future in as_completed(future_map):
                    res = future.result()
                    if res is not None:
                        results.append(res)
            print(f"Scan completed with {len(results)} signals.")
            send_telegram_message(format_summary_message(results, timestamp))

            elapsed = (datetime.utcnow() - start_time).total_seconds()
            sleep_seconds = 12 * 3600 - elapsed

            if sleep_seconds > 0:
                print(f"Sleeping for {sleep_seconds / 3600:.2f} hours until next run in smaller chunks.")
                sleep_in_chunks(sleep_seconds, chunk_seconds=60)  # Chunk size 60 seconds; adjust if needed
            else:
                print("Processing took longer than 12 hours, starting next run immediately.")
    except KeyboardInterrupt:
        print("Interrupted by user, exiting.")
        _cleanup()

if __name__ == "__main__":
    main()
