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

# --- ML and Backtest Globals ---
ml_training_X = deque(maxlen=1000)
ml_training_y = deque(maxlen=1000)
ml_model = LogisticRegression(solver='liblinear', random_state=42)
ml_model_trained = False
ml_model_lock = threading.Lock()

ml_retrain_count = 0  # Count how many times retrained
ml_retrain_performance = deque(maxlen=10)  # Store last 10 retrain accuracies (0-1 scale)

# --- Indicator functions mapping ---
INDICATOR_FUNCTIONS = {
    'Stoch RSI': lambda df, idx: analyze_stoch_rsi_trend(df['stochrsi_k'], df['stochrsi_d'], idx),
    'Williams %R': lambda df, idx: analyze_wr_relative_positions({length: df[f'wr_{length}'] for length in [3,13,144,8,233,55]}, idx),
    'RSI': lambda df, idx: analyze_rsi_trend(df['rsi5'], df['rsi13'], df['rsi21'], idx),
    'KDJ': lambda df, idx: analyze_kdj_trend(df['kdj_k'], df['kdj_d'], df['kdj_j'], idx),
    'MACD': lambda df, idx: indicator_signal_macd(df, idx),
    'Bollinger': lambda df, idx: indicator_signal_bollinger(df, idx),
}

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

# --- ML Functions ---
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
    global ml_model_trained, ml_retrain_count
    with ml_model_lock:
        if len(ml_training_X) > len(INDICATOR_FUNCTIONS) * 5:
            try:
                X = np.array(list(ml_training_X))
                y = np.array(list(ml_training_y))
                ml_model.fit(X, y)
                ml_model_trained = True
                ml_retrain_count += 1
                accuracy = ml_model.score(X, y)
                ml_retrain_performance.append(accuracy)
                print(f"ML model retrained {ml_retrain_count} times. Latest accuracy: {accuracy:.2%}")
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

# --- Backtest and Performance Update ---
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

# --- Market Regime and Indicator Performance ---
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

# --- Fetch OHLCV ---
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
    except Exception as e:
        print(f"Error fetching OHLCV for {symbol}: {e}")
        return None

# --- Retrain Summary Message ---
def format_retrain_summary_message(timestamp_str):
    if ml_retrain_count == 0:
        retrain_msg = "<i>No ML retraining has occurred yet.</i>"
    else:
        avg_perf = np.mean(ml_retrain_performance) if ml_retrain_performance else 0.0
        improvement_pct = avg_perf * 100
        retrain_msg = (
            f"<b>🤖 ML Retrain Summary @ {timestamp_str} (4h TF)</b>\n\n"
            f"• Total Retrains: <b>{ml_retrain_count}</b>\n"
            f"• Average Training Accuracy: <b>{improvement_pct:.2f}%</b>\n\n"
            f"<i>Model adapts as new data arrives. Monitor retrain frequency and accuracy.</i>"
        )
    return retrain_msg

# --- Time until next 4h UTC ---
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

# --- Main loop ---
def main():
    try:
        while True:
            start_time = datetime.utcnow()
            print(f"\nStarting backtest & retrain summary at {start_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")

            MAX_WORKERS = min(16, os.cpu_count() * 2 + 1)
            print(f"Using {MAX_WORKERS} threads for parallel processing.")

            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_to_symbol = {executor.submit(fetch_latest_ohlcv, symbol, '4h', 750): symbol for symbol in CRYPTO_SYMBOLS}
                for future in as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    try:
                        df = future.result()
                        if df is not None and len(df) >= 100:
                            df = calculate_indicators(df.copy())
                            backtest_data_start_idx = max(0, len(df) - 750)
                            backtest_df_subset = df.iloc[backtest_data_start_idx:]
                            if len(backtest_df_subset) > 50:
                                update_performance_and_ml_from_backtest(backtest_df_subset)
                            else:
                                print(f"Skipping backtest for {symbol}, insufficient data.")
                        else:
                            print(f"Insufficient data for {symbol}, skipping backtest.")
                    except Exception as exc:
                        print(f"{symbol} backtest exception: {exc}")

            summary_message = format_retrain_summary_message(start_time.strftime('%Y-%m-%d %H:%M UTC'))
            send_telegram_message(summary_message)
            print("Retrain summary sent to Telegram.")

            sleep_seconds = seconds_until_next_4h_utc()
            print(f"Sleeping {sleep_seconds:.0f} seconds until next 4-hour UTC boundary.")
            time.sleep(sleep_seconds)

    except KeyboardInterrupt:
        print("Interrupted by user. Exiting.")
        cleanup()

if __name__ == "__main__":
    main()
