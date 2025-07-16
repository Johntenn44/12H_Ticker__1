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

# --- Launch Webserver Subprocess ---
webserver_path = os.path.join(os.path.dirname(__file__), 'webserver.py')
webserver_process = subprocess.Popen([sys.executable, webserver_path])
def cleanup():
    print("Terminating webserver subprocess...")
    webserver_process.terminate()
atexit.register(cleanup)

# --- Telegram Configuration ---
TELEGRAM_BOT_TOKEN   = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID     = os.environ.get("TELEGRAM_CHAT_ID")
TELEGRAM_CHANNEL_ID  = os.environ.get("TELEGRAM_CHANNEL_ID")

def send_telegram_message(message, chat_id_override=None):
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
        if curr: parts.append(curr)
    else:
        parts = [message]
    for part in parts:
        for chat_id in target_ids:
            resp = requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                data={'chat_id': chat_id, 'text': part, 'parse_mode': 'HTML', 'disable_web_page_preview': True},
                timeout=15
            )
            if not resp.ok:
                print(f"Failed to send message to {chat_id}: {resp.text}")

# --- CCXT Exchange Initialization ---
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
            print(f"Error initializing CCXT: {e}")
            sys.exit(1)
initialize_exchange()

CRYPTO_SYMBOLS = [
    "XRP/USDT", "SUI/USDT", "DOGE/USDT", "XLM/USDT", "TRX/USDT",
    "AAVE/USDT", "LTC/USDT", "ARB/USDT", "NEAR/USDT", "MNT/USDT",
    "VIRTUAL/USDT", "XMR/USDT", "EIGEN/USDT", "CAKE/USDT", "SUSHI/USDT",
    "GRASS/USDT", "WAVES/USDT", "APE/USDT", "SKL/USDT", "PLUME/USDT",
    "LUNA/USDT"
]

# --- Indicator Utilities ---
def calculate_rsi(series, period=13):
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period-1, adjust=False).mean()
    avg_loss = loss.ewm(com=period-1, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_stoch_rsi(df, rsi_len=13, stoch_len=8, smooth_k=5, smooth_d=3):
    rsi = calculate_rsi(df['close'], rsi_len)
    min_rsi = rsi.rolling(window=stoch_len).min()
    max_rsi = rsi.rolling(window=stoch_len).max()
    pct = (rsi - min_rsi) / (max_rsi - min_rsi) * 100
    pct = pct.fillna(method='ffill')
    k = pct.rolling(window=smooth_k).mean()
    d = k.rolling(window=smooth_d).mean()
    return k, d

def calculate_multi_wr(df, lengths=[3,13,144,8,233,55]):
    wr = {}
    for L in lengths:
        hh = df['high'].rolling(L).max()
        ll = df['low'].rolling(L).min()
        wr[L] = ((hh - df['close']) / (hh - ll) * -100).fillna(method='ffill')
    return wr

def calculate_kdj(df, length=5, ma1=8, ma2=8):
    low_min  = df['low'].rolling(length, min_periods=1).min()
    high_max = df['high'].rolling(length, min_periods=1).max()
    rsv = ((df['close'] - low_min) / (high_max - low_min) * 100).fillna(method='ffill')
    k  = rsv.ewm(span=ma1, adjust=False).mean()
    d  = k.ewm(span=ma2, adjust=False).mean()
    j  = 3*k - 2*d
    return k, d, j

def calculate_macd(close, fast=12, slow=26, signal=9):
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_f - ema_s
    sig_line  = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, sig_line

def calculate_bollinger_bands(close, window=20, num_std=2):
    sma = close.rolling(window).mean()
    std = close.rolling(window).std()
    return sma + num_std*std, sma - num_std*std

def calculate_indicators(df):
    df['close'] = df['close'].astype(float)
    df['rsi5']  = calculate_rsi(df['close'], 5)
    df['rsi13'] = calculate_rsi(df['close'], 13)
    df['rsi21'] = calculate_rsi(df['close'], 21)
    k, d = calculate_stoch_rsi(df)
    df['stochrsi_k'], df['stochrsi_d'] = k, d
    wr = calculate_multi_wr(df)
    for L, series in wr.items(): df[f'wr_{L}'] = series
    kdj_k, kdj_d, kdj_j = calculate_kdj(df)
    df['kdj_k'], df['kdj_d'], df['kdj_j'] = kdj_k, kdj_d, kdj_j
    macd_line, macd_signal = calculate_macd(df['close'])
    df['macd_line'], df['macd_signal'] = macd_line, macd_signal
    ub, lb = calculate_bollinger_bands(df['close'])
    df['bb_upper'], df['bb_lower'] = ub, lb
    return df

# --- Indicator Signal Analysis ---
def analyze_trend(series1, series2, idx):
    if idx < 1 or pd.isna(series1.iloc[idx]) or pd.isna(series2.iloc[idx]): return None
    if series1.iloc[idx] > series2.iloc[idx]: return "up"
    if series1.iloc[idx] < series2.iloc[idx]: return "down"
    return None

def analyze_wr_positions(wr_dict, idx):
    try:
        w8, w3, w144, w233, w55 = (wr_dict[x].iloc[idx] for x in (8,3,144,233,55))
    except: return None
    if w8 > w233 and w3 > w233 and w8 > w144 and w3 > w144: return "up"
    if w8 < w55 and w3 < w55: return "down"
    return None

INDICATOR_FUNCTIONS = {
    'Stoch RSI': lambda df, idx: analyze_trend(df['stochrsi_k'], df['stochrsi_d'], idx),
    'Williams %R': lambda df, idx: analyze_wr_positions({L:df[f'wr_{L}'] for L in [3,13,144,8,233,55]}, idx),
    'RSI': lambda df, idx: ("up" if df['rsi5'].iloc[idx] > df['rsi13'].iloc[idx] > df['rsi21'].iloc[idx]
                            else "down" if df['rsi5'].iloc[idx] < df['rsi13'].iloc[idx] < df['rsi21'].iloc[idx]
                            else None),
    'KDJ': lambda df, idx: analyze_trend(df['kdj_j'], df['kdj_d'], idx),
    'MACD': lambda df, idx: analyze_trend(df['macd_line'], df['macd_signal'], idx),
    'Bollinger': lambda df, idx: ("up" if df['close'].iloc[idx] < df['bb_lower'].iloc[idx]
                                  else "down" if df['close'].iloc[idx] > df['bb_upper'].iloc[idx]
                                  else None),
}

# --- Machine Learning Components with Calibration ---
indicator_performance_regime = defaultdict(lambda: {'high_vol': deque(maxlen=200), 'low_vol': deque(maxlen=200)})
ml_X = deque(maxlen=1000)
ml_y = deque(maxlen=1000)
base_ml_model = LogisticRegression(solver='liblinear', random_state=42)
calibrated_ml_model = None
ml_trained = False
ml_lock = threading.Lock()

def get_market_regime(df, window=50, threshold=0.02):
    if len(df) < window: return "unknown"
    vol = df['close'].pct_change().rolling(window).std().iloc[-1]
    return "high_vol" if vol > threshold else "low_vol"

def update_indicator_performance(name, regime, correct):
    if regime == "unknown": return
    indicator_performance_regime[name][regime].append(1 if correct else 0)

def get_indicator_weight(name, regime):
    if regime == "unknown": return 0.5
    perf = indicator_performance_regime[name][regime]
    return sum(perf)/len(perf) if perf else 0.5

def add_ml_sample(signals, price_move):
    feats = [(1 if s=="up" else -1 if s=="down" else 0) for s in signals.values()]
    ml_X.append(feats)
    ml_y.append(1 if price_move>0 else 0)

def train_ml_model():
    global ml_trained, calibrated_ml_model
    with ml_lock:
        if len(ml_X) > len(INDICATOR_FUNCTIONS)*5:
            try:
                base_ml_model.fit(np.array(ml_X), np.array(ml_y))
                calibrated_ml_model = CalibratedClassifierCV(base_ml_model, cv='prefit', method='sigmoid')
                calibrated_ml_model.fit(np.array(ml_X), np.array(ml_y))
                ml_trained = True
                print("ML model trained and calibrated.")
            except Exception as e:
                print(f"ML training/calibration failed: {e}")
                ml_trained = False
                calibrated_ml_model = None
        else:
            ml_trained = False
            calibrated_ml_model = None

def ml_predict(signals):
    if not ml_trained or calibrated_ml_model is None:
        return None, 0.0
    feats = np.array([(1 if s=="up" else -1 if s=="down" else 0) for s in signals.values()]).reshape(1,-1)
    try:
        proba = calibrated_ml_model.predict_proba(feats)[0]
        max_proba = max(proba)
        if max_proba < 0.55:
            return None, 0.0
        if proba[1] > proba[0]:
            return "buy", proba[1]*100
        else:
            return "sell", proba[0]*100
    except Exception as e:
        print(f"ML prediction error: {e}")
        return None, 0.0

def adaptive_signal(df):
    regime = get_market_regime(df)
    signals = {n: fn(df, -1) for n, fn in INDICATOR_FUNCTIONS.items()}
    ml_sig, ml_conf = ml_predict(signals)
    if ml_sig:
        return ml_sig, ml_conf, signals, regime
    up_w = down_w = total = 0
    for n, s in signals.items():
        w = get_indicator_weight(n, regime)
        if s=="up": up_w+=w; total+=w
        if s=="down": down_w+=w; total+=w
    if total > 0.1:
        if up_w > down_w * 1.2:
            return "buy", (up_w / total) * 100, signals, regime
        if down_w > up_w * 1.2:
            return "sell", (down_w / total) * 100, signals, regime
    if all(s == "up" for s in signals.values() if s is not None):
        avg_conf = np.mean([get_indicator_weight(n, regime) for n in INDICATOR_FUNCTIONS]) * 100
        return "buy", avg_conf, signals, regime
    if all(s == "down" for s in signals.values() if s is not None):
        avg_conf = np.mean([get_indicator_weight(n, regime) for n in INDICATOR_FUNCTIONS]) * 100
        return "sell", avg_conf, signals, regime
    return None, 0.0, signals, regime

def backtest_update(df):
    start = max(50, 0)
    if len(df) < start + 2:
        return
    for i in range(start, len(df) - 1):
        regime = get_market_regime(df[:i + 1])
        if regime == "unknown":
            continue
        sigs = {n: fn(df, i) for n, fn in INDICATOR_FUNCTIONS.items()}
        p0, p1 = df['close'].iloc[i], df['close'].iloc[i + 1]
        for n, s in sigs.items():
            if s == "up":
                update_indicator_performance(n, regime, p1 > p0)
            if s == "down":
                update_indicator_performance(n, regime, p1 < p0)
        add_ml_sample(sigs, p1 - p0)
    train_ml_model()

# --- Trailing Take Profit Logic ---
def determine_trailing_take_profit(regime, accuracy):
    base = 3.0 if regime == "high_vol" else 1.0
    adjusted = base * (1.5 - accuracy)
    return max(0.5, min(adjusted, 4.0))

# --- Fetch OHLCV Data ---
def fetch_latest_ohlcv(symbol, timeframe='12h', limit=250):
    try:
        with exchange_lock:
            if symbol not in exchange.symbols:
                print(f"{symbol} not on exchange.")
                return None
            data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
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

# --- Position Sizing for $1 max trade with 5x leverage ---
MAX_UNLEVERAGED_NOTIONAL = 1.0  # $1 max notional per trade
LEVERAGE = 5                    # 5x leverage (adjusted for 12h candles)

def calculate_position_size_dollars(price):
    units = math.floor(MAX_UNLEVERAGED_NOTIONAL / price)
    if units == 0:
        return 0.0, 0
    notional = units * price
    return notional, units

# --- Process a Single Symbol with Position Sizing ---
def process_symbol(symbol):
    df = fetch_latest_ohlcv(symbol)
    if df is None or len(df) < 100:
        return None
    df = calculate_indicators(df.copy())
    sub = df.iloc[-250:]
    if len(sub) > 50:
        backtest_update(sub)
    sig, conf, states, regime = adaptive_signal(df)
    if sig in ("buy", "sell") and conf >= 55.0:
        price = df['close'].iloc[-1]
        weights = [get_indicator_weight(n, regime) for n in INDICATOR_FUNCTIONS]
        avg_perf = sum(weights) / len(weights) if weights else 0.5
        ttp = determine_trailing_take_profit(regime, avg_perf)
        stop_loss_pct = (ttp / 100) * 0.5  # stop loss at half trailing TP

        notional_usd, units = calculate_position_size_dollars(price)
        if units == 0:
            return None  # price too high for $1 max trade

        return {
            "symbol": symbol,
            "signal": sig,
            "confidence": conf,
            "indicator_states": states,
            "regime": regime,
            "current_price": price,
            "ttp_percent": ttp,
            "stop_loss_pct": stop_loss_pct * 100,
            "position_size_units": units,
            "position_size_usd": notional_usd,
            "leverage": LEVERAGE
        }
    return None

# --- Message Formatting ---
def format_single_coin_detail(symbol, signal, confidence, indicator_states,
                              regime, price_str, ttp_percent=None,
                              stop_loss_pct=None, position_size_units=None,
                              position_size_usd=None, leverage=None):
    emo = "🟢 BUY" if signal == "buy" else "🔴 SELL"
    reg_emo = "📈 High Volatility" if regime == "high_vol" else "📉 Low Volatility" if regime == "low_vol" else "❓ Unknown"
    commentary = {
        "buy": ["Momentum building! 🚀", "Breakout ahead! 📈", "Bulls taking charge! 🐂"],
        "sell": ["Caution downtrend! ⚠️", "Bears in control. 🐻", "Possible pullback 🔻"]
    }
    comment = np.random.choice(commentary[signal])
    details = "\n".join(
        f"  • {n}: {'✅ Up' if s == 'up' else '❌ Down' if s == 'down' else '⚪ Neutral'}"
        for n, s in indicator_states.items()
    )
    display_confidence = min(confidence, 99.9)
    ttp_line = f"  <b>Trailing Take Profit:</b> <code>{ttp_percent:.2f}%</code>\n" if ttp_percent else ""
    sl_line = f"  <b>Stop Loss Distance:</b> <code>{stop_loss_pct:.2f}%</code>\n" if stop_loss_pct else ""
    pos_line = ""
    if position_size_units is not None and position_size_usd is not None and leverage is not None:
        pos_line = (f"  <b>Position Size:</b> <code>{position_size_units} units ≈ ${position_size_usd:.2f}</code>\n"
                    f"  <b>Leverage:</b> <code>{leverage}x</code>\n")
    return (
        f"<b>{symbol} | {emo} ({display_confidence:.1f}%)</b>\n"
        f"  <b>Price:</b> <code>{price_str}</code>\n"
        f"  <b>Market:</b> {reg_emo}\n"
        f"{ttp_line}{sl_line}{pos_line}"
        f"  <i>{comment}</i>\n"
        f"  <u>Indicators:</u>\n{details}\n"
    )

def format_summary_message(coin_results, timestamp_str):
    if not coin_results:
        return (
            f"<b>📊 Market Scan @ {timestamp_str} (12h TF)</b>\n\n"
            f"<i>No strong BUY/SELL signals detected.</i>"
        )
    sorted_results = sorted(coin_results, key=lambda x: x['confidence'], reverse=True)
    hour = datetime.utcnow().hour
    greeting = ("🌅 Good morning" if 5 <= hour < 12 else "🌞 Good afternoon" if 12 <= hour < 18 else "🌙 Good evening")
    header = (
        f"{greeting}, trader!\n"
        f"<b>📊 Crypto Signals Scan @ {timestamp_str} (12h TF)</b>\n"
        f"<i>Top {min(5, len(sorted_results))} opportunities:</i>\n"
    )
    perf = "<b>🧠 System Overview:</b>\n" + "".join(
        f"  • {n}: High Vol: {get_indicator_weight(n, 'high_vol'):.1%}, Low Vol: {get_indicator_weight(n, 'low_vol'):.1%}\n"
        for n in INDICATOR_FUNCTIONS
    ) + f"  • ML Model: {'✅ Ready' if ml_trained else '⏳ Training...'}\n\n"
    details = "<b>⭐ Top Signals:</b>\n" + "".join(
        format_single_coin_detail(
            res['symbol'], res['signal'], res['confidence'],
            res['indicator_states'], res['regime'],
            f"{res['current_price']:.4f}" if res['current_price'] < 10 else f"{res['current_price']:.2f}",
            ttp_percent=res.get('ttp_percent'),
            stop_loss_pct=res.get('stop_loss_pct'),
            position_size_units=res.get('position_size_units'),
            position_size_usd=res.get('position_size_usd'),
            leverage=res.get('leverage')
        ) for res in sorted_results[:5]
    )
    closing = np.random.choice([
        "<i>Trade smart, manage risk!</i>",
        "<i>Markets are dynamic. Stay sharp!</i>",
        "<i>Not financial advice.</i>"
    ])
    return header + perf + details + closing

# --- Scheduling Utilities ---
def seconds_until_next_12h_utc():
    now = datetime.utcnow()
    next_hour = ((now.hour // 12) + 1) * 12 % 24
    next_run = datetime.combine(now.date(), datetime.min.time()) + timedelta(hours=next_hour)
    if next_run <= now:
        next_run += timedelta(days=1)
    return (next_run - now).total_seconds()

# --- Main Loop ---
def main():
    try:
        while True:
            start = datetime.utcnow()
            timestamp = start.strftime('%Y-%m-%d %H:%M UTC')
            print(f"\nStarting scan at {timestamp}")
            coin_results = []
            workers = min(16, (os.cpu_count() or 1) * 2 + 1)
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(process_symbol, sym): sym for sym in CRYPTO_SYMBOLS}
                for fut in as_completed(futures):
                    res = fut.result()
                    if res:
                        coin_results.append(res)
            print(f"Scan complete: {len(coin_results)} signals detected.")
            msg = format_summary_message(coin_results, timestamp)
            send_telegram_message(msg)
            sleep = seconds_until_next_12h_utc()
            print(f"Sleeping {sleep:.0f}s until next 12h UTC.")
            time.sleep(sleep)
    except KeyboardInterrupt:
        print("Interrupted. Exiting.")
        cleanup()

if __name__ == "__main__":
    main()
