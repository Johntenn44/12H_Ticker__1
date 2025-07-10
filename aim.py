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
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
import random

# --- Global Configurations ---
webserver_path = os.path.join(os.path.dirname(__file__), 'webserver.py')
webserver_process = subprocess.Popen([sys.executable, webserver_path])

def cleanup():
    print("Terminating webserver subprocess...")
    webserver_process.terminate()

atexit.register(cleanup)

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
TELEGRAM_CHANNEL_ID = os.environ.get("TELEGRAM_CHANNEL_ID")

exchange = None
exchange_lock = threading.Lock()

def initialize_exchange():
    global exchange
    if exchange is None:
        try:
            exchange = ccxt.kucoin()
            exchange.load_markets()
            print("CCXT exchange initialized and markets loaded.")
        except Exception as e:
            print(f"Error initializing CCXT exchange: {e}")
            sys.exit(1)

initialize_exchange()

CRYPTO_SYMBOLS = [
    "XRP/USDT", "XMR/USDT", "GMX/USDT", "LUNA/USDT", "TRX/USDT", "EIGEN/USDT",
    "APE/USDT", "WAVES/USDT", "PLUME/USDT", "SUSHI/USDT", "DOGE/USDT", "VIRTUAL/USDT",
    "CAKE/USDT", "GRASS/USDT", "AAVE/USDT", "SUI/USDT", "ARB/USDT", "XLM/USDT",
    "MNT/USDT", "LTC/USDT", "NEAR/USDT"
]

# --- Telegram Functions ---
def send_telegram_message(message, chat_id_override=None):
    if not TELEGRAM_BOT_TOKEN:
        print("TELEGRAM_BOT_TOKEN not set! Cannot send messages.")
        return

    MAX_MESSAGE_LENGTH = 4096

    target_chat_ids = []
    if chat_id_override:
        target_chat_ids = [chat_id_override]
    else:
        if TELEGRAM_CHAT_ID:
            target_chat_ids.append(TELEGRAM_CHAT_ID)
        if TELEGRAM_CHANNEL_ID:
            target_chat_ids.append(TELEGRAM_CHANNEL_ID)

    if not target_chat_ids:
        print("No TELEGRAM_CHAT_ID or TELEGRAM_CHANNEL_ID set, message not sent!")
        return

    messages_to_send = []
    if len(message) > MAX_MESSAGE_LENGTH:
        parts = []
        current_part = ""
        for line in message.split('\n'):
            if len(current_part) + len(line) + 1 <= MAX_MESSAGE_LENGTH:
                current_part += line + '\n'
            else:
                parts.append(current_part)
                current_part = line + '\n'
        if current_part:
            parts.append(current_part)
        messages_to_send = parts
        print(f"Message too long ({len(message)} chars), split into {len(messages_to_send)} parts.")
    else:
        messages_to_send = [message]

    sent_any = False
    for msg_part in messages_to_send:
        for chat_id in target_chat_ids:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            payload = {
                'chat_id': chat_id,
                'text': msg_part,
                'parse_mode': 'HTML',
                'disable_web_page_preview': True
            }
            try:
                response = requests.post(url, data=payload, timeout=15)
                if not response.ok:
                    print(f"Failed to send message part to {chat_id}: {response.text}")
                else:
                    sent_any = True
            except Exception as e:
                print(f"Error sending Telegram message part to {chat_id}: {e}")
    if not sent_any and not messages_to_send:
        print("No messages were generated or sent.")

# --- Indicator Calculations ---
def calculate_rsi(series, period=13):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period-1, adjust=False).mean()
    avg_loss = loss.ewm(com=period-1, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_stoch_rsi(df, rsi_len=13, stoch_len=8, smooth_k=5, smooth_d=3):
    rsi = calculate_rsi(df['close'], rsi_len)
    min_rsi = rsi.rolling(window=stoch_len).min()
    max_rsi = rsi.rolling(window=stoch_len).max()
    denom = max_rsi - min_rsi
    denom = denom.replace(0, np.nan)
    stoch_rsi = (rsi - min_rsi) / denom * 100
    stoch_rsi = stoch_rsi.fillna(method='ffill')
    k = stoch_rsi.rolling(window=smooth_k).mean()
    d = k.rolling(window=smooth_d).mean()
    return k, d

def calculate_multi_wr(df, lengths=[3, 13, 144, 8, 233, 55]):
    wr_dict = {}
    for length in lengths:
        highest_high = df['high'].rolling(window=length).max()
        lowest_low = df['low'].rolling(window=length).min()
        denom = highest_high - lowest_low
        denom = denom.replace(0, np.nan)
        wr = (highest_high - df['close']) / denom * -100
        wr_dict[length] = wr.fillna(method='ffill')
    return wr_dict

def analyze_wr_relative_positions(wr_dict, idx=-1):
    try:
        wr_8 = wr_dict[8].iloc[idx]
        wr_3 = wr_dict[3].iloc[idx]
        wr_144 = wr_dict[144].iloc[idx]
        wr_233 = wr_dict[233].iloc[idx]
        wr_55 = wr_dict[55].iloc[idx]
        if any(pd.isna([wr_8, wr_3, wr_144, wr_233, wr_55])):
            return None
    except (KeyError, IndexError, AttributeError):
        return None

    if wr_8 > wr_233 and wr_3 > wr_233 and wr_8 > wr_144 and wr_3 > wr_144:
        return "up"
    elif wr_8 < wr_55 and wr_3 < wr_55:
        return "down"
    else:
        return None

def calculate_kdj(df, length=5, ma1=8, ma2=8):
    low_min = df['low'].rolling(window=length, min_periods=1).min()
    high_max = df['high'].rolling(window=length, min_periods=1).max()
    denom = (high_max - low_min)
    denom = denom.replace(0, np.nan)
    rsv = (df['close'] - low_min) / denom * 100
    rsv = rsv.fillna(method='ffill')
    k = rsv.ewm(span=ma1, adjust=False).mean()
    d = k.ewm(span=ma2, adjust=False).mean()
    j = 3 * k - 2 * d
    return k, d, j

def analyze_stoch_rsi_trend(k, d, idx=-1):
    if pd.isna(k.iloc[idx]) or pd.isna(d.iloc[idx]):
        return None
    if k.iloc[idx] > d.iloc[idx]:
        return "up"
    elif k.iloc[idx] < d.iloc[idx]:
        return "down"
    else:
        return None

def analyze_rsi_trend(rsi5, rsi13, rsi21, idx=-1):
    try:
        if pd.isna(rsi5.iloc[idx]) or pd.isna(rsi13.iloc[idx]) or pd.isna(rsi21.iloc[idx]):
            return None
        if rsi5.iloc[idx] > rsi13.iloc[idx] > rsi21.iloc[idx]:
            return "up"
        elif rsi5.iloc[idx] < rsi13.iloc[idx] < rsi21.iloc[idx]:
            return "down"
        else:
            return None
    except IndexError:
        return None

def analyze_kdj_trend(k, d, j, idx=-1):
    try:
        if pd.isna(k.iloc[idx]) or pd.isna(d.iloc[idx]) or pd.isna(j.iloc[idx]):
            return None
        if j.iloc[idx] > k.iloc[idx] and j.iloc[idx] > d.iloc[idx]:
            return "up"
        elif j.iloc[idx] < k.iloc[idx] and j.iloc[idx] < d.iloc[idx]:
            return "down"
        else:
            return None
    except IndexError:
        return None

def calculate_macd(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line

def calculate_bollinger_bands(close, window=20, num_std=2):
    sma = close.rolling(window=window).mean()
    std = close.rolling(window=window).std()
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    return upper_band, lower_band

def calculate_indicators(df):
    df['close'] = df['close'].astype(float)
    df['rsi5'] = calculate_rsi(df['close'], 5)
    df['rsi13'] = calculate_rsi(df['close'], 13)
    df['rsi21'] = calculate_rsi(df['close'], 21)
    k, d = calculate_stoch_rsi(df, rsi_len=13, stoch_len=8, smooth_k=5, smooth_d=3)
    df['stochrsi_k'] = k
    df['stochrsi_d'] = d
    wr_dict = calculate_multi_wr(df, lengths=[3, 13, 144, 8, 233, 55])
    for length, series in wr_dict.items():
        df[f'wr_{length}'] = series
    kdj_k, kdj_d, kdj_j = calculate_kdj(df, length=5, ma1=8, ma2=8)
    df['kdj_k'] = kdj_k
    df['kdj_d'] = kdj_d
    df['kdj_j'] = kdj_j
    macd_line, macd_signal = calculate_macd(df['close'])
    df['macd_line'] = macd_line
    df['macd_signal'] = macd_signal
    upper_band, lower_band = calculate_bollinger_bands(df['close'])
    df['bb_upper'] = upper_band
    df['bb_lower'] = lower_band
    return df

def indicator_signal_stoch_rsi(df, idx):
    return analyze_stoch_rsi_trend(df['stochrsi_k'], df['stochrsi_d'], idx)

def indicator_signal_wr(df, idx):
    wr_dict = {length: df[f'wr_{length}'] for length in [3, 13, 144, 8, 233, 55]}
    return analyze_wr_relative_positions(wr_dict, idx)

def indicator_signal_rsi(df, idx):
    return analyze_rsi_trend(df['rsi5'], df['rsi13'], df['rsi21'], idx)

def indicator_signal_kdj(df, idx):
    return analyze_kdj_trend(df['kdj_k'], df['kdj_d'], df['kdj_j'], idx)

def indicator_signal_macd(df, idx):
    macd_line = df['macd_line']
    macd_signal = df['macd_signal']
    if idx < 1 or len(macd_line) <= idx or pd.isna(macd_line.iloc[idx]) or pd.isna(macd_signal.iloc[idx]):
        return None
    if macd_line.iloc[idx-1] < macd_signal.iloc[idx-1] and macd_line.iloc[idx] > macd_signal.iloc[idx]:
        return "up"
    elif macd_line.iloc[idx-1] > macd_signal.iloc[idx-1] and macd_line.iloc[idx] < macd_signal.iloc[idx]:
        return "down"
    else:
        return None

def indicator_signal_bollinger(df, idx):
    price = df['close'].iloc[idx]
    upper_band = df['bb_upper']
    lower_band = df['bb_lower']
    if np.isnan(upper_band.iloc[idx]) or np.isnan(lower_band.iloc[idx]) or np.isnan(price):
        return None
    if price < lower_band.iloc[idx]:
        return "up"
    elif price > upper_band.iloc[idx]:
        return "down"
    else:
        return None

INDICATOR_FUNCTIONS = {
    'Stoch RSI': indicator_signal_stoch_rsi,
    'Williams %R': indicator_signal_wr,
    'RSI': indicator_signal_rsi,
    'KDJ': indicator_signal_kdj,
    'MACD': indicator_signal_macd,
    'Bollinger': indicator_signal_bollinger,
}

def get_market_regime(df, window=50, threshold=0.02):
    if len(df) < window:
        return "unknown"
    returns = df['close'].pct_change()
    vol = returns.rolling(window=window).std()
    current_vol = vol.iloc[-1]
    return "high_vol" if current_vol > threshold else "low_vol"

indicator_performance_regime = defaultdict(lambda: {'high_vol': deque(maxlen=200), 'low_vol': deque(maxlen=200)})

def update_indicator_performance_regime(indicator_name, regime, was_correct):
    if regime == "unknown": return
    indicator_performance_regime[indicator_name][regime].append(1 if was_correct else 0)

def get_indicator_weight_regime(indicator_name, regime):
    if regime == "unknown": return 0.5
    perf = indicator_performance_regime[indicator_name][regime]
    return sum(perf) / len(perf) if perf else 0.5

ml_training_X = deque(maxlen=1000)
ml_training_y = deque(maxlen=1000)
ml_model = LogisticRegression(solver='liblinear', random_state=42)
ml_model_trained = False
ml_model_lock = threading.Lock()

def indicator_signals_to_features(indicator_signals):
    features = []
    for name in INDICATOR_FUNCTIONS.keys():
        val = indicator_signals.get(name)
        if val == "up":
            features.append(1)
        elif val == "down":
            features.append(-1)
        else:
            features.append(0)
    return np.array(features).reshape(1, -1)

def update_ml_model():
    global ml_model_trained
    with ml_model_lock:
        if len(ml_training_X) > len(INDICATOR_FUNCTIONS) * 5:
            try:
                ml_model.fit(np.array(list(ml_training_X)), np.array(list(ml_training_y)))
                ml_model_trained = True
            except ValueError:
                ml_model_trained = False
        else:
            ml_model_trained = False

def add_ml_training_sample(indicator_signals, price_move):
    features = []
    for name in INDICATOR_FUNCTIONS.keys():
        val = indicator_signals.get(name)
        if val == "up": features.append(1)
        elif val == "down": features.append(-1)
        else: features.append(0)
    ml_training_X.append(features)
    ml_training_y.append(1 if price_move > 0 else 0)

def ml_predict_signal(indicator_signals):
    if not ml_model_trained:
        return None, 0.0
    features = indicator_signals_to_features(indicator_signals)
    try:
        proba = ml_model.predict_proba(features)[0]
        if proba[1] > 0.55:
            signal = "buy"
        elif proba[0] > 0.55:
            signal = "sell"
        else:
            signal = None
        confidence = max(proba)
        return signal, confidence * 100
    except Exception:
        return None, 0.0

def adaptive_check_signal_with_regime(df):
    current_regime = get_market_regime(df)
    current_indicator_signals = {name: func(df, -1) for name, func in INDICATOR_FUNCTIONS.items()}
    ml_signal, ml_confidence = ml_predict_signal(current_indicator_signals)
    if ml_signal:
        return ml_signal, ml_confidence, current_indicator_signals, current_regime

    up_weight, down_weight = 0.0, 0.0
    total_relevant_weight = 0.0
    for name, signal in current_indicator_signals.items():
        weight = get_indicator_weight_regime(name, current_regime)
        if signal == "up":
            up_weight += weight
            total_relevant_weight += weight
        elif signal == "down":
            down_weight += weight
            total_relevant_weight += weight

    fallback_signal = None
    fallback_confidence = 0.0
    if total_relevant_weight > 0.1:
        if up_weight > down_weight * 1.2:
            fallback_signal = "buy"
            fallback_confidence = (up_weight / total_relevant_weight) * 100
        elif down_weight > up_weight * 1.2:
            fallback_signal = "sell"
            fallback_confidence = (down_weight / total_relevant_weight) * 100

    if fallback_signal is None:
        all_up = all(s == "up" for s in current_indicator_signals.values() if s is not None)
        all_down = all(s == "down" for s in current_indicator_signals.values() if s is not None)
        if all_up and len([s for s in current_indicator_signals.values() if s is not None]) > 0:
            fallback_signal = "buy"
            fallback_confidence = 75.0
        elif all_down and len([s for s in current_indicator_signals.values() if s is not None]) > 0:
            fallback_signal = "sell"
            fallback_confidence = 75.0

    return fallback_signal, fallback_confidence, current_indicator_signals, current_regime

def update_performance_and_ml_from_backtest(df):
    start_idx = max(50, 0)
    if len(df) < start_idx + 2:
        return
    for idx in range(start_idx, len(df) - 1):
        regime = get_market_regime(df.iloc[:idx+1])
        if regime == "unknown": continue
        current_indicator_signals = {name: func(df, idx) for name, func in INDICATOR_FUNCTIONS.items()}
        price_now = df['close'].iloc[idx]
        price_next = df['close'].iloc[idx+1]
        price_move = price_next - price_now
        for name, signal in current_indicator_signals.items():
            if signal == "up":
                was_correct = price_next > price_now
                update_indicator_performance_regime(name, regime, was_correct)
            elif signal == "down":
                was_correct = price_next < price_now
                update_indicator_performance_regime(name, regime, was_correct)
        add_ml_training_sample(current_indicator_signals, price_move)
    update_ml_model()

def fetch_latest_ohlcv(symbol, timeframe='4h', limit=750):
    try:
        with exchange_lock:
            if symbol not in exchange.symbols:
                print(f"Symbol {symbol} not available on {exchange.id}.")
                return None
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        if not ohlcv:
            print(f"No OHLCV data returned for {symbol}.")
            return None
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=['close'], inplace=True)
        return df
    except ccxt.NetworkError as e:
        print(f"Network error fetching OHLCV for {symbol}: {e}. Retrying might help.")
        return None
    except ccxt.ExchangeError as e:
        print(f"Exchange error fetching OHLCV for {symbol}: {e}. Check symbol or exchange status.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred fetching OHLCV for {symbol}: {e}")
        return None

# --- Unicode Font Generators ---
def to_squared(text):
    squared = {
        'A': '🄰', 'B': '🄱', 'C': '🄲', 'D': '🄳', 'E': '🄴', 'F': '🄵', 'G': '🄶',
        'H': '🄷', 'I': '🄸', 'J': '🄹', 'K': '🄺', 'L': '🄻', 'M': '🄼', 'N': '🄽',
        'O': '🄾', 'P': '🄿', 'Q': '🅀', 'R': '🅁', 'S': '🅂', 'T': '🅃', 'U': '🅄',
        'V': '🅅', 'W': '🅆', 'X': '🅇', 'Y': '🅈', 'Z': '🅉',
        ' ': ' '
    }
    return ''.join(squared.get(c.upper(), c) for c in text)

def to_bold_squared(text):
    bold_squared = {
        'A': '🅰', 'B': '🅱', 'C': '🅲', 'D': '🅳', 'E': '🅴', 'F': '🅵', 'G': '🅶',
        'H': '🅷', 'I': '🅸', 'J': '🅹', 'K': '🅺', 'L': '🅻', 'M': '🅼', 'N': '🅽',
        'O': '🅾', 'P': '🅿', 'Q': '🆀', 'R': '🆁', 'S': '🆂', 'T': '🆃', 'U': '🆄',
        'V': '🆅', 'W': '🆆', 'X': '🆇', 'Y': '🆈', 'Z': '🆉',
        ' ': ' '
    }
    return ''.join(bold_squared.get(c.upper(), c) for c in text)

def to_double_outline(text):
    double_outline = {
        'A': '🄰', 'B': '🄱', 'C': '🄲', 'D': '🄳', 'E': '🄴', 'F': '🄵', 'G': '🄶',
        'H': '🄷', 'I': '🄸', 'J': '🄹', 'K': '🄺', 'L': '🄻', 'M': '🄼', 'N': '🄽',
        'O': '🄾', 'P': '🄿', 'Q': '🅀', 'R': '🅁', 'S': '🅂', 'T': '🅃', 'U': '🅄',
        'V': '🅅', 'W': '🅆', 'X': '🅇', 'Y': '🅈', 'Z': '🅉',
        ' ': ' '
    }
    return ''.join(double_outline.get(c.upper(), c) for c in text)

def to_circled(text):
    circled = {
        'A': 'Ⓐ', 'B': 'Ⓑ', 'C': 'Ⓒ', 'D': 'Ⓓ', 'E': 'Ⓔ', 'F': 'Ⓕ', 'G': 'Ⓖ',
        'H': 'Ⓗ', 'I': 'Ⓘ', 'J': 'Ⓙ', 'K': 'Ⓚ', 'L': 'Ⓛ', 'M': 'Ⓜ', 'N': 'Ⓝ',
        'O': 'Ⓞ', 'P': 'Ⓟ', 'Q': 'Ⓠ', 'R': 'Ⓡ', 'S': 'Ⓢ', 'T': 'Ⓣ', 'U': 'Ⓤ',
        'V': 'Ⓥ', 'W': 'Ⓦ', 'X': 'Ⓧ', 'Y': 'Ⓨ', 'Z': 'Ⓩ',
        ' ': ' '
    }
    return ''.join(circled.get(c.upper(), c) for c in text)

def random_unicode_font(text):
    fonts = [to_squared, to_bold_squared, to_double_outline, to_circled]
    font_func = random.choice(fonts)
    return font_func(text)

# --- Message Formatting with Unicode Fonts ---
def format_single_coin_detail(symbol, signal, confidence, indicator_states, regime, current_price_str):
    signal_emoji = "🟢 BUY" if signal == 'buy' else "🔴 SELL"
    regime_emoji = "📈 High Volatility" if regime == "high_vol" else "📉 Low Volatility" if regime == "low_vol" else "❓ Unknown"
    commentary = {
        "buy": [
            "Momentum is building up! 🚀",
            "Potential breakout ahead! 📈",
            "Bulls are taking charge! 🐂"
        ],
        "sell": [
            "Caution: Downtrend detected! ⚠️",
            "Bears are in control. 🐻",
            "Possible pullback incoming. 🔻"
        ]
    }
    comment = random.choice(commentary[signal]) if signal in commentary else ""

    indicator_details = []
    for name, state in indicator_states.items():
        if state == "up":
            indicator_details.append(f"  • {name}: ✅ Up")
        elif state == "down":
            indicator_details.append(f"  • {name}: ❌ Down")
        else:
            indicator_details.append(f"  • {name}: ⚪ Neutral")
    details_str = "\n".join(indicator_details)
    return (
        f"<b>{symbol} | {signal_emoji} ({confidence:.1f}%)</b>\n"
        f"  <b>Price:</b> <code>{current_price_str}</code>\n"
        f"  <b>Market:</b> {regime_emoji}\n"
        f"  <i>{comment}</i>\n"
        f"  <u>Indicators:</u>\n{details_str}\n"
    )

def format_summary_message(coin_results, timestamp_str):
    # Insert a random styled phrase at the top
    styled_phrase = random_unicode_font("I LIKE YOU")

    if not coin_results:
        return (
            f"{styled_phrase}\n\n"
            f"<b>📊 Market Scan @ {timestamp_str} (4h TF)</b>\n\n"
            f"<i>No strong BUY/SELL signals detected across monitored assets.</i>\n"
            f"<i>Market appears to be indecisive or flat. Stay patient and manage risk!</i>"
        )
    sorted_results = sorted(coin_results, key=lambda x: x['confidence'], reverse=True)

    hour = datetime.utcnow().hour
    if 5 <= hour < 12:
        greeting = "🌅 Good morning, trader!"
    elif 12 <= hour < 18:
        greeting = "🌞 Good afternoon, trader!"
    else:
        greeting = "🌙 Good evening, trader!"

    header = (
        f"{styled_phrase}\n\n"
        f"{greeting}\n"
        f"<b>📊 Crypto Signals Scan @ {timestamp_str} (4h TF)</b>\n"
        f"<i>Top 5 trading opportunities, ranked by confidence.</i>\n\n"
        f"🚀 <b>{len(sorted_results)}</b> active signals detected.\n"
    )

    global_performance_summary = "<b>🧠 System Overview:</b>\n"
    for name, regimes in indicator_performance_regime.items():
        high_vol_acc = get_indicator_weight_regime(name, 'high_vol')
        low_vol_acc = get_indicator_weight_regime(name, 'low_vol')
        global_performance_summary += (
            f"  • {name}: High Vol: {high_vol_acc:.1%}, Low Vol: {low_vol_acc:.1%}\n"
        )
    global_performance_summary += f"  • ML Model: {'✅ Ready' if ml_model_trained else '⏳ Training...'}\n\n"

    top_n = 5
    detailed_signals_sections = []
    for i in range(min(top_n, len(sorted_results))):
        res = sorted_results[i]
        price_str = f"{res['current_price']:.4f}" if res['current_price'] < 10 else f"{res['current_price']:.2f}"
        detailed_signals_sections.append(
            format_single_coin_detail(
                res['symbol'], res['signal'], res['confidence'],
                res['indicator_states'], res['regime'], price_str
            )
        )
    detailed_signals_block = "<b>⭐ Top 5 Signals:</b>\n" + "\n".join(detailed_signals_sections) + "\n"

    closing_notes = [
        "<i>Remember: Markets are dynamic. Stay sharp and manage your risk!</i>",
        "<i>Trade smart, stay safe. This is not financial advice.</i>",
        "<i>Opportunities come and go. Patience is key!</i>"
    ]
    closing_note = random.choice(closing_notes)

    full_message = header + global_performance_summary + detailed_signals_block + closing_note
    return full_message

def process_symbol(symbol):
    print(f"Processing {symbol}...")
    df = fetch_latest_ohlcv(symbol, timeframe='4h', limit=750)
    if df is None or len(df) < 100:
        print(f"Not enough recent data for {symbol} after filtering ({len(df) if df is not None else 0} bars), skipping.")
        return None
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_cols):
        print(f"Missing required OHLCV columns for {symbol}, skipping.")
        return None
    df = calculate_indicators(df.copy())
    backtest_data_start_idx = max(0, len(df) - 750)
    backtest_df_subset = df.iloc[backtest_data_start_idx:]
    if len(backtest_df_subset) > 50:
        update_performance_and_ml_from_backtest(backtest_df_subset)
    else:
        print(f"Skipping backtest update for {symbol}, not enough data in backtest window.")
    signal, confidence, indicator_states, regime = adaptive_check_signal_with_regime(df)
    if signal in ("buy", "sell") and confidence >= 55.0:
        current_price = df['close'].iloc[-1]
        print(f"Signal detected for {symbol}: {signal} with {confidence:.1f}% confidence.")
        return {
            "symbol": symbol,
            "signal": signal,
            "confidence": confidence,
            "indicator_states": indicator_states,
            "regime": regime,
            "current_price": current_price
        }
    else:
        print(f"No strong signal for {symbol} (Signal: {signal}, Confidence: {confidence:.1f}%).")
        return None

def seconds_until_next_4h_utc():
    now = datetime.utcnow()
    next_hour = (now.hour // 4 + 1) * 4
    if next_hour >= 24:
        next_day = now.date() + timedelta(days=1)
        next_run = datetime.combine(next_day, datetime.min.time()) + timedelta(hours=0)
    else:
        next_run = datetime.combine(now.date(), datetime.min.time()) + timedelta(hours=next_hour)
    delta = next_run - now
    return max(delta.total_seconds(), 0)

def main():
    try:
        while True:
            start_time = datetime.utcnow()
            print(f"\nInitiating crypto signal scan at {start_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")

            MAX_WORKERS = min(16, os.cpu_count() * 2 + 1)
            print(f"Using {MAX_WORKERS} threads for parallel processing.")

            coin_results = []
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_to_symbol = {executor.submit(process_symbol, symbol): symbol for symbol in CRYPTO_SYMBOLS}
                for future in as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    try:
                        result = future.result()
                        if result:
                            coin_results.append(result)
                    except Exception as exc:
                        print(f'{symbol} generated an exception: {exc}')

            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()
            print(f"\nScan completed in {duration:.2f} seconds.")

            summary_message = format_summary_message(coin_results, start_time.strftime('%Y-%m-%d %H:%M UTC'))
            send_telegram_message(summary_message)
            print("\nFinal summary message sent to Telegram (or printed if not configured).")

            sleep_seconds = seconds_until_next_4h_utc()
            print(f"Sleeping for {sleep_seconds:.0f} seconds until next 4-hour UTC boundary.")
            time.sleep(sleep_seconds)

    except KeyboardInterrupt:
        print("\nExecution interrupted by user. Exiting gracefully.")
        cleanup()

if __name__ == "__main__":
    main()
