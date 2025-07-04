import subprocess
import sys
import os
import atexit
import time
import requests
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime

# --- Start webserver.py subprocess for Render port binding ---
webserver_path = os.path.join(os.path.dirname(__file__), 'webserver.py')
webserver_process = subprocess.Popen([sys.executable, webserver_path])
def cleanup():
    print("Terminating webserver subprocess...")
    webserver_process.terminate()
atexit.register(cleanup)

# --- Telegram config from environment ---
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
            url = "https://api.telegram.org/bot{}/sendMessage".format(TELEGRAM_BOT_TOKEN)
            payload = {
                'chat_id': chat_id,
                'text': message,
                'parse_mode': 'HTML',
                'disable_web_page_preview': True
            }
            try:
                response = requests.post(url, data=payload, timeout=10)
                if not response.ok:
                    print("Failed to send message to {}: {}".format(chat_id, response.text))
                else:
                    sent = True
            except Exception as e:
                print("Error sending Telegram message to {}: {}".format(chat_id, e))
    if not sent:
        print("No TELEGRAM_CHAT_ID or TELEGRAM_CHANNEL_ID set, message not sent!")

# --- Indicator functions ---
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
    if idx < 1 or pd.isna(k.iloc[idx-1]) or pd.isna(d.iloc[idx-1]) or pd.isna(k.iloc[idx]) or pd.isna(d.iloc[idx]):
        return None
    if k.iloc[idx-1] < d.iloc[idx-1] and k.iloc[idx] > d.iloc[idx] and k.iloc[idx] < 80:
        return "up"
    elif k.iloc[idx-1] > d.iloc[idx-1] and k.iloc[idx] < d.iloc[idx] and k.iloc[idx] > 20:
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
    if idx < 1:
        return None
    k_prev, k_curr = k.iloc[idx-1], k.iloc[idx]
    d_prev, d_curr = d.iloc[idx-1], d.iloc[idx]
    j_prev, j_curr = j.iloc[idx-1], j.iloc[idx]
    if k_prev < d_prev and k_curr > d_curr and j_curr > k_curr and j_curr > d_curr:
        return "up"
    elif k_prev > d_prev and k_curr < d_curr and j_curr < k_curr and j_curr < d_curr:
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

# --- Indicator backtest (accuracy) functions ---
def indicator_signal_stoch_rsi(df, idx):
    k, d = calculate_stoch_rsi(df)
    return analyze_stoch_rsi_trend(k, d, idx)

def indicator_signal_wr(df, idx):
    wr_dict = calculate_multi_wr(df)
    return analyze_wr_relative_positions(wr_dict, idx)

def indicator_signal_rsi(df, idx):
    rsi5 = calculate_rsi(df['close'], 5)
    rsi13 = calculate_rsi(df['close'], 13)
    rsi21 = calculate_rsi(df['close'], 21)
    return analyze_rsi_trend(rsi5, rsi13, rsi21, idx)

def indicator_signal_kdj(df, idx):
    k, d, j = calculate_kdj(df)
    return analyze_kdj_trend(k, d, j, idx)

def indicator_signal_macd(df, idx):
    macd_line, macd_signal = calculate_macd(df['close'])
    if idx < 1 or len(macd_line) <= idx:
        return None
    if macd_line.iloc[idx-1] < macd_signal.iloc[idx-1] and macd_line.iloc[idx] > macd_signal.iloc[idx]:
        return "up"
    elif macd_line.iloc[idx-1] > macd_signal.iloc[idx-1] and macd_line.iloc[idx] < macd_signal.iloc[idx]:
        return "down"
    else:
        return None

def indicator_signal_bollinger(df, idx):
    upper_band, lower_band = calculate_bollinger_bands(df['close'])
    price = df['close'].iloc[idx]
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

def indicator_stats(df, indicator_func):
    correct = 0
    total = 0
    for idx in range(1, len(df)-1):  # avoid first and last candle
        signal = indicator_func(df, idx)
        if signal not in ("up", "down"):
            continue
        price_now = df['close'].iloc[idx]
        price_next = df['close'].iloc[idx+1]
        if signal == "up" and price_next > price_now:
            correct += 1
        elif signal == "down" and price_next < price_now:
            correct += 1
        total += 1
    accuracy = (correct / total) if total > 0 else 0.5
    return accuracy, total, correct

def get_indicator_stats(df):
    stats = {}
    for name, func in INDICATOR_FUNCTIONS.items():
        acc, total, correct = indicator_stats(df, func)
        stats[name] = {'accuracy': acc, 'total': total, 'correct': correct}
    return stats

def check_signal_with_confidence(df, stats):
    signals = {}
    for name, func in INDICATOR_FUNCTIONS.items():
        signals[name] = func(df, -1)

    up_weight = 0.0
    down_weight = 0.0
    total_weight = 0.0
    for name, signal in signals.items():
        acc = stats.get(name, {}).get('accuracy', 0.5)
        if signal == "up":
            up_weight += acc
            total_weight += acc
        elif signal == "down":
            down_weight += acc
            total_weight += acc
    if total_weight == 0:
        return None, 0, signals

    if up_weight > down_weight:
        confidence = (up_weight / total_weight) * 100
        return "buy", confidence, signals
    elif down_weight > up_weight:
        confidence = (down_weight / total_weight) * 100
        return "sell", confidence, signals
    else:
        return None, 0, signals

CRYPTO_SYMBOLS = [
    "XRP/USDT", "XMR/USDT", "GMX/USDT", "LUNA/USDT", "TRX/USDT", "EIGEN/USDT",
    "APE/USDT", "WAVES/USDT", "PLUME/USDT", "SUSHI/USDT", "DOGE/USDT", "VIRTUAL/USDT",
    "CAKE/USDT", "GRASS/USDT", "AAVE/USDT", "SUI/USDT", "ARB/USDT", "XLM/USDT",
    "MNT/USDT", "LTC/USDT", "NEAR/USDT"
]

def fetch_latest_ohlcv(symbol, timeframe='6h', limit=130):
    try:
        exchange = ccxt.kucoin()
        exchange.load_markets()
        if symbol not in exchange.symbols:
            print("Symbol {} not available on this exchange.".format(symbol))
            return None
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df.astype(float)
    except Exception as e:
        print("Error fetching OHLCV data for {}: {}".format(symbol, e))
        return None

def format_signal_message(symbol, signal, confidence, indicator_states, time_str, stats):
    if signal == "buy":
        sig_emoji = "🚀"
        sig_word = "<b>BUY</b>"
        sig_line = "✅ <b>Majority indicators aligned for:</b> <b>BUY</b>"
    else:
        sig_emoji = "🔥"
        sig_word = "<b>SELL</b>"
        sig_line = "⚠️ <b>Majority indicators aligned for:</b> <b>SELL</b>"

    indicator_rows = []
    for name in INDICATOR_FUNCTIONS.keys():
        val = indicator_states.get(name)
        if val == "up":
            val_disp = "🟢 up"
        elif val == "down":
            val_disp = "🔴 down"
        else:
            val_disp = "—"
        s = stats.get(name, {})
        hits = f"{s.get('correct',0)}/{s.get('total',0)}"
        indicator_rows.append("{:<13} {:<8} ({})".format(name + ":", val_disp, hits))
    indicator_table = "\n".join(indicator_rows)

    message = (
        f"{sig_emoji} <b>{sig_word} Signal Detected</b>\n"
        f"<b>Pair:</b> <code>{symbol}</code>\n"
        f"<b>Time:</b> <code>{time_str}</code>\n\n"
        f"{sig_line}\n"
        f"<b>Confidence:</b> <code>{confidence:.1f}%</code>\n\n"
        f"📊 <b>Indicator Breakdown</b>\n"
        f"<pre>{indicator_table}</pre>"
        f"\n<i>Hits = correct predictions / total entries (past month)</i>"
    )
    return message

def main():
    last_checked_hour = None
    while True:
        now = datetime.utcnow()
        if now.hour % 6 == 0 and (last_checked_hour != now.hour or last_checked_hour is None):
            last_checked_hour = now.hour
            print("\nChecking signals for all symbols at {}".format(now.strftime('%Y-%m-%d %H:%M:%S UTC')))
            for symbol in CRYPTO_SYMBOLS:
                print("\n--- Checking {} ---".format(symbol))
                df = fetch_latest_ohlcv(symbol, timeframe='6h', limit=130)
                if df is None or len(df) < 50:
                    print("No data for {}, skipping.".format(symbol))
                    continue

                backtest_df = df.iloc[-121:-1]  # last 120 before current
                stats = get_indicator_stats(backtest_df)
                signal, confidence, indicator_states = check_signal_with_confidence(df, stats)
                if signal in ("buy", "sell"):
                    last_close_time = df.index[-1].strftime('%Y-%m-%d %H:%M UTC')
                    message = format_signal_message(symbol, signal, confidence, indicator_states, last_close_time, stats)
                    send_telegram_message(message)
                    print(message)
                else:
                    print("No clear buy or sell signal detected for {}.".format(symbol))

            now = datetime.utcnow()
            minutes_until_next_6h = ((6 - (now.hour % 6)) % 6) * 60 + (60 - now.minute)
            sleep_seconds = minutes_until_next_6h * 60
            print("\nSleeping for {} minutes until next 6-hour candle...".format(int(sleep_seconds // 60)))
            time.sleep(sleep_seconds)
        else:
            time.sleep(60)

if __name__ == "__main__":
    main()
