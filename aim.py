import subprocess
import sys
import os
import atexit
import requests
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict, deque
from sklearn.linear_model import LogisticRegression

# Start webserver.py subprocess for Render port binding
webserver_path = os.path.join(os.path.dirname(__file__), 'webserver.py')
webserver_process = subprocess.Popen([sys.executable, webserver_path])
def cleanup():
    print("Terminating webserver subprocess...")
    webserver_process.terminate()
atexit.register(cleanup)

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
TELEGRAM_CHANNEL_ID = os.environ.get("TELEGRAM_CHANNEL_ID")

def send_telegram_message(message):
    if not TELEGRAM_BOT_TOKEN:
        print("TELEGRAM_BOT_TOKEN not set!")
        return
    sent = False
    for chat_id in [TELEGRAM_CHAT_ID, TELEGRAM_CHANNEL_ID]:
        if chat_id:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            payload = {
                'chat_id': chat_id,
                'text': message,
                'parse_mode': 'HTML',
                'disable_web_page_preview': True
            }
            try:
                response = requests.post(url, data=payload, timeout=10)
                if not response.ok:
                    print(f"Failed to send message to {chat_id}: {response.text}")
                else:
                    sent = True
            except Exception as e:
                print(f"Error sending Telegram message to {chat_id}: {e}")
    if not sent:
        print("No TELEGRAM_CHAT_ID or TELEGRAM_CHANNEL_ID set, message not sent!")

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
    except (KeyError, IndexError):
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
    rsv = (df['close'] - low_min) / (high_max - low_min) * 100
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
    if rsi5.iloc[idx] > rsi13.iloc[idx] > rsi21.iloc[idx]:
        return "up"
    elif rsi5.iloc[idx] < rsi13.iloc[idx] < rsi21.iloc[idx]:
        return "down"
    else:
        return None

def analyze_kdj_trend(k, d, j, idx=-1):
    if pd.isna(k.iloc[idx]) or pd.isna(d.iloc[idx]) or pd.isna(j.iloc[idx]):
        return None
    if j.iloc[idx] > k.iloc[idx] and j.iloc[idx] > d.iloc[idx]:
        return "up"
    elif j.iloc[idx] < k.iloc[idx] and j.iloc[idx] < d.iloc[idx]:
        return "down"
    else:
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
    k = df['stochrsi_k']
    d = df['stochrsi_d']
    return analyze_stoch_rsi_trend(k, d, idx)

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
    if idx < 1 or len(macd_line) <= idx:
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
    if np.isnan(upper_band.iloc[idx]) or np.isnan(lower_band.iloc[idx]):
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

# --- Regime Filter ---
def get_market_regime(df, window=50, threshold=0.02):
    returns = df['close'].pct_change()
    vol = returns.rolling(window=window).std()
    current_vol = vol.iloc[-1]
    return "high_vol" if current_vol > threshold else "low_vol"

# --- Regime-dependent Performance Tracking ---
indicator_performance_regime = defaultdict(lambda: {'high_vol': deque(maxlen=100), 'low_vol': deque(maxlen=100)})

def update_indicator_performance_regime(indicator_name, regime, was_correct):
    indicator_performance_regime[indicator_name][regime].append(1 if was_correct else 0)

def get_indicator_weight_regime(indicator_name, regime):
    perf = indicator_performance_regime[indicator_name][regime]
    return sum(perf) / len(perf) if perf else 0.5

# --- ML Model for Adaptive Signal Combination ---
ml_training_X = []
ml_training_y = []
ml_model = LogisticRegression()

def indicator_signals_to_features(indicator_signals):
    return np.array([
        1 if v == "up" else -1 if v == "down" else 0
        for v in indicator_signals.values()
    ]).reshape(1, -1)

def update_ml_model():
    if len(ml_training_X) > 50:
        ml_model.fit(np.array(ml_training_X), np.array(ml_training_y))

def add_ml_training_sample(indicator_signals, price_move):
    ml_training_X.append([
        1 if v == "up" else -1 if v == "down" else 0
        for v in indicator_signals.values()
    ])
    ml_training_y.append(1 if price_move > 0 else 0)
    if len(ml_training_X) > 500:
        del ml_training_X[0]
        del ml_training_y[0]
    update_ml_model()

def ml_predict_signal(indicator_signals):
    if len(ml_training_X) < 50:
        return None, 0.0
    features = indicator_signals_to_features(indicator_signals)
    proba = ml_model.predict_proba(features)[0]
    signal = "buy" if proba[1] > 0.55 else "sell" if proba[1] < 0.45 else None
    confidence = max(proba)
    return signal, confidence * 100

# --- Adaptive Signal Combination with Regime and ML ---
def adaptive_check_signal_with_regime(df):
    regime = get_market_regime(df)
    signals = {name: func(df, -1) for name, func in INDICATOR_FUNCTIONS.items()}
    # Regime-weighted voting (fallback if ML not ready)
    up_weight, down_weight = 0.0, 0.0
    for name, signal in signals.items():
        weight = get_indicator_weight_regime(name, regime)
        if signal == "up":
            up_weight += weight
        elif signal == "down":
            down_weight += weight
    fallback_signal = "buy" if up_weight > down_weight else "sell" if down_weight > up_weight else None
    fallback_confidence = max(up_weight, down_weight) / (up_weight + down_weight) * 100 if (up_weight + down_weight) > 0 else 0

    # ML-based signal
    ml_signal, ml_confidence = ml_predict_signal(signals)
    if ml_signal:
        return ml_signal, ml_confidence, signals, regime
    else:
        return fallback_signal, fallback_confidence, signals, regime

def update_performance_and_ml_from_backtest(df):
    for name, func in INDICATOR_FUNCTIONS.items():
        for idx in range(1, len(df) - 2):
            regime = get_market_regime(df.iloc[:idx+1])
            signal = func(df, idx)
            price_now = df['close'].iloc[idx]
            price_next = df['close'].iloc[idx+1]
            if signal == "up":
                was_correct = price_next > price_now
            elif signal == "down":
                was_correct = price_next < price_now
            else:
                continue
            update_indicator_performance_regime(name, regime, was_correct)
            # ML training
            indicator_signals = {n: f(df, idx) for n, f in INDICATOR_FUNCTIONS.items()}
            price_move = price_next - price_now
            add_ml_training_sample(indicator_signals, price_move)

CRYPTO_SYMBOLS = [
    "XRP/USDT", "XMR/USDT", "GMX/USDT", "LUNA/USDT", "TRX/USDT", "EIGEN/USDT",
    "APE/USDT", "WAVES/USDT", "PLUME/USDT", "SUSHI/USDT", "DOGE/USDT", "VIRTUAL/USDT",
    "CAKE/USDT", "GRASS/USDT", "AAVE/USDT", "SUI/USDT", "ARB/USDT", "XLM/USDT",
    "MNT/USDT", "LTC/USDT", "NEAR/USDT"
]

def fetch_latest_ohlcv(symbol, timeframe='6h', limit=750):
    try:
        exchange = ccxt.kucoin()
        exchange.load_markets()
        if symbol not in exchange.symbols:
            print(f"Symbol {symbol} not available on this exchange.")
            return None
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df.astype(float)
    except Exception as e:
        print(f"Error fetching OHLCV data for {symbol}: {e}")
        return None

def format_signal_message(symbol, signal, confidence, indicator_states, regime, time_str):
    indicator_rows = []
    for name in INDICATOR_FUNCTIONS.keys():
        val = indicator_states.get(name)
        if val == "up":
            val_disp = "🟢 up"
        elif val == "down":
            val_disp = "🔴 down"
        else:
            val_disp = "—"
        indicator_rows.append(f"{name:<13} {val_disp:<10}")
    indicator_table = "\n".join(indicator_rows)
    message = (
        f"{'🚀' if signal == 'buy' else '🔥'} <b>{'BUY' if signal == 'buy' else 'SELL'} Signal Detected</b>\n"
        f"<b>Pair:</b> <code>{symbol}</code>\n"
        f"<b>Time:</b> <code>{time_str}</code>\n"
        f"<b>Market Regime:</b> <code>{regime}</code>\n\n"
        f"<b>Confidence:</b> <code>{confidence:.1f}%</code>\n\n"
        f"📊 <b>Indicator States</b>\n"
        f"<pre>{indicator_table}</pre>"
    )
    return message

def format_ranking_message(coin_results):
    header = "<b>📈 Ranked Coin Signals</b>\n\n"
    if not coin_results:
        return header + "No active signals found for ranking."
    rows = []
    for rank, result in enumerate(coin_results, 1):
        rows.append(
            f"<b>{rank}.</b> <code>{result['symbol']}</code> | {result['signal'].upper()} | "
            f"Confidence: <b>{result['confidence']:.1f}%</b> | Regime: <b>{result['regime']}</b>"
        )
    return header + "\n".join(rows)

def main():
    now = datetime.utcnow()
    print(f"\nChecking signals for all symbols at {now.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    coin_results = []
    for symbol in CRYPTO_SYMBOLS:
        print(f"\n--- Checking {symbol} ---")
        df = fetch_latest_ohlcv(symbol, timeframe='6h', limit=750)
        if df is None or len(df) < 700:
            print(f"Not enough data for {symbol}, skipping.")
            continue
        df = calculate_indicators(df)
        backtest_df = df.iloc[-751:-1]
        update_performance_and_ml_from_backtest(backtest_df)
        signal, confidence, indicator_states, regime = adaptive_check_signal_with_regime(df)
        if signal in ("buy", "sell"):
            last_close_time = df.index[-1].strftime('%Y-%m-%d %H:%M UTC')
            message = format_signal_message(symbol, signal, confidence, indicator_states, regime, last_close_time)
            send_telegram_message(message)
            print(message)
            coin_results.append({
                "symbol": symbol,
                "signal": signal,
                "confidence": confidence,
                "regime": regime,
                "message": message
            })
        else:
            print(f"No clear buy or sell signal detected for {symbol}.")
    coin_results.sort(key=lambda x: (x['confidence']), reverse=True)
    ranking_message = format_ranking_message(coin_results)
    send_telegram_message(ranking_message)
    print("\n" + ranking_message)

if __name__ == "__main__":
    main()
