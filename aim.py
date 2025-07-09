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
from collections import deque
from itertools import combinations

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

# --- Indicator calculations ---

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

def calculate_atr(df, window=14):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr = true_range.rolling(window=window).mean()
    return atr

def detect_regime(df, ma_window=50, atr_window=14, trend_threshold=0.0001, volatility_threshold=0.002):
    close = df['close']
    ma = close.rolling(window=ma_window).mean()
    if len(ma) < ma_window + 2:
        return 'range'
    slope = (ma.iloc[-1] - ma.iloc[-ma_window]) / ma_window
    atr = calculate_atr(df, window=atr_window)
    recent_atr = atr.iloc[-1]
    avg_price = close.iloc[-ma_window:].mean()
    rel_atr = recent_atr / avg_price if avg_price else 0
    if abs(slope) > trend_threshold and rel_atr < volatility_threshold:
        return 'trend'
    elif rel_atr > volatility_threshold:
        return 'volatile'
    else:
        return 'range'

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

# --- Indicator signal functions ---

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

INDICATOR_REGIME_PREF = {
    'Stoch RSI': ['trend', 'volatile'],
    'Williams %R': ['trend', 'volatile'],
    'RSI': ['range'],
    'KDJ': ['trend', 'volatile'],
    'MACD': ['trend'],
    'Bollinger': ['range', 'volatile'],
}

ROLLING_WINDOW = 100
indicator_performance = {name: deque(maxlen=ROLLING_WINDOW) for name in INDICATOR_FUNCTIONS}

def update_performance(indicator_name, correct):
    indicator_performance[indicator_name].append(correct)

def get_weight(indicator_name, regime):
    perf = indicator_performance[indicator_name]
    base_weight = max(0.1, sum(perf) / len(perf)) if perf else 1.0
    regime_boost = 1.5 if regime in INDICATOR_REGIME_PREF.get(indicator_name, []) else 1.0
    return base_weight * regime_boost

def calculate_hits(df, indicator_func, indicator_name=None, update_perf=False):
    hits = 0
    total_entries = 0
    last_signal = None
    for idx in range(1, len(df) - 2):
        signal = indicator_func(df, idx)
        if signal in ("up", "down") and signal != last_signal:
            total_entries += 1
            price_now = df['close'].iloc[idx]
            price_1 = df['close'].iloc[idx + 1]
            price_2 = df['close'].iloc[idx + 2]
            correct = False
            if signal == "up":
                if price_1 > price_now and price_2 > price_1:
                    hits += 1
                    correct = True
            elif signal == "down":
                if price_1 < price_now and price_2 < price_1:
                    hits += 1
                    correct = True
            if update_perf and indicator_name:
                update_performance(indicator_name, int(correct))
        last_signal = signal
    hit_rate = hits / total_entries if total_entries > 0 else 0
    return hits, total_entries, hit_rate

def indicator_accuracy_and_hits(df, indicator_func, indicator_name, update_perf=False):
    correct = 0
    total = 0
    hits, total_entries, hit_rate = calculate_hits(df, indicator_func, indicator_name, update_perf)
    for idx in range(1, len(df) - 1):
        signal = indicator_func(df, idx)
        if signal not in ("up", "down"):
            continue
        price_now = df['close'].iloc[idx]
        price_next = df['close'].iloc[idx + 1]
        if signal == "up" and price_next > price_now:
            correct += 1
        elif signal == "down" and price_next < price_now:
            correct += 1
        total += 1
    accuracy = (correct / total) if total > 0 else 0.5
    idx = len(df) - 3
    current_signal = indicator_func(df, idx)
    is_ongoing = False
    if current_signal == "up":
        is_ongoing = (df['close'].iloc[idx + 1] > df['close'].iloc[idx] and
                      df['close'].iloc[idx + 2] > df['close'].iloc[idx + 1])
    elif current_signal == "down":
        is_ongoing = (df['close'].iloc[idx + 1] < df['close'].iloc[idx] and
                      df['close'].iloc[idx + 2] < df['close'].iloc[idx + 1])
    return accuracy, hits, total_entries, is_ongoing

def get_indicator_accuracies_and_hits(df, update_perf=False):
    accuracies = {}
    hits = {}
    totals = {}
    ongoing = {}
    for name, func in INDICATOR_FUNCTIONS.items():
        acc, hit, total, is_ongoing = indicator_accuracy_and_hits(df, func, name, update_perf)
        accuracies[name] = acc
        hits[name] = hit
        totals[name] = total
        ongoing[name] = is_ongoing
    return accuracies, hits, totals, ongoing

def adaptive_regime_check_signal(df, regime):
    signals = {}
    up_weight = 0.0
    down_weight = 0.0
    total_weight = 0.0
    for name, func in INDICATOR_FUNCTIONS.items():
        signal = func(df, -1)
        signals[name] = signal
        weight = get_weight(name, regime)
        if signal == "up":
            up_weight += weight
            total_weight += weight
        elif signal == "down":
            down_weight += weight
            total_weight += weight
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

def format_signal_message(symbol, signal, confidence, indicator_states, indicator_hits, indicator_totals, indicator_ongoing, time_str, regime, combo_accuracy=None, combo_hits=None, combo_total=None, best_combo=None, best_accuracy=None, best_correct=None):
    indicator_stats = []
    for name in INDICATOR_FUNCTIONS.keys():
        val = indicator_states.get(name)
        hit = indicator_hits.get(name, 0)
        total = indicator_totals.get(name, 0)
        hit_rate = (hit / total * 100) if total > 0 else 0.0
        agrees = 1 if ((signal == "buy" and val == "up") or (signal == "sell" and val == "down")) else 0
        ongoing = indicator_ongoing.get(name, False)
        ongoing_mark = "✅" if ongoing else "❌"
        if val == "up":
            val_disp = "🟢 up"
        elif val == "down":
            val_disp = "🔴 down"
        else:
            val_disp = "—"
        indicator_stats.append({
            "name": name,
            "val_disp": val_disp,
            "hit": hit,
            "total": total,
            "hit_rate": hit_rate,
            "agrees": agrees,
            "ongoing_mark": ongoing_mark
        })
    indicator_stats.sort(key=lambda x: (x['agrees'], x['hit_rate'], x['hit']), reverse=True)
    indicator_rows = [
        f"{x['name'] + ':':<13} {x['val_disp']:<10} ({x['hit']}/{x['total']})  {x['hit_rate']:5.1f}%  {x['ongoing_mark']}"
        for x in indicator_stats
    ]
    indicator_table = "\n".join(indicator_rows)
    regime_disp = {
        "trend": "📈 Trending",
        "range": "🔄 Ranging",
        "volatile": "⚡ Volatile"
    }.get(regime, regime)
    combo_msg = ""
    if combo_accuracy is not None and combo_total is not None and combo_total > 0:
        combo_msg = f"\n\n<b>Current Combination Accuracy:</b> {combo_accuracy:.1%} ({combo_hits}/{combo_total})"
    best_combo_str = ', '.join(sorted(best_combo)) if best_combo else "N/A"
    correctness_str = "✅ Correct" if best_correct else ("❌ Incorrect" if best_correct is False else "❓ Unknown")
    message = (
        f"{'🚀' if signal == 'buy' else '🔥'} <b>{'BUY' if signal == 'buy' else 'SELL'} Signal Detected</b>\n"
        f"<b>Pair:</b> <code>{symbol}</code>\n"
        f"<b>Time:</b> <code>{time_str}</code>\n"
        f"<b>Market Regime:</b> <code>{regime_disp}</code>\n\n"
        f"{'✅' if signal == 'buy' else '⚠️'} <b>Majority indicators aligned for:</b> <b>{'BUY' if signal == 'buy' else 'SELL'}</b>\n"
        f"<b>Confidence:</b> <code>{confidence:.1f}%</code>\n\n"
        f"📊 <b>Indicator Breakdown</b>\n"
        f"<pre>{indicator_table}</pre>"
        f"{combo_msg}\n"
        f"\n<b>Best Combo:</b> <code>{best_combo_str}</code>\n"
        f"<b>Best Combo Accuracy:</b> {best_accuracy:.1%}\n"
        f"<b>Latest Signal Correctness:</b> {correctness_str}\n"
        f"<i>Hits = correct predictions / total entries (past 6 months)</i>\n"
        f"<i>Ongoing trend: ✅ = yes, ❌ = no</i>"
    )
    return message

def fetch_latest_ohlcv(symbol, timeframe='15m', limit=750):
    try:
        exchange = ccxt.kraken()
        exchange.load_markets()
        if symbol not in exchange.symbols:
            print(f"Symbol {symbol} not available on Kraken.")
            return None
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df.astype(float)
    except Exception as e:
        print(f"Error fetching OHLCV data for {symbol}: {e}")
        return None

# --- Combination evaluation functions ---

def get_indicator_combinations(indicators, max_size=4):
    combos = []
    for size in range(2, max_size + 1):
        combos.extend(combinations(indicators, size))
    return combos

def compute_regime_series(df):
    regimes = []
    window = 60
    for i in range(len(df)):
        if i < window:
            regimes.append('range')
        else:
            sub_df = df.iloc[i-window:i+1]
            regimes.append(detect_regime(sub_df))
    return pd.Series(regimes, index=df.index)

def backtest_combination(df, indicator_funcs, combo, regime, regime_series):
    hits = 0
    total = 0
    for idx in range(1, len(df) - 2):
        if regime_series.iloc[idx] != regime:
            continue
        signals = []
        for ind_name in combo:
            signal = indicator_funcs[ind_name](df, idx)
            if signal not in ("up", "down"):
                break
            signals.append(signal)
        else:
            if len(set(signals)) == 1:
                total += 1
                signal = signals[0]
                price_now = df['close'].iloc[idx]
                price_1 = df['close'].iloc[idx + 1]
                price_2 = df['close'].iloc[idx + 2]
                if signal == "up" and price_1 > price_now and price_2 > price_1:
                    hits += 1
                elif signal == "down" and price_1 < price_now and price_2 < price_1:
                    hits += 1
    accuracy = hits / total if total > 0 else 0
    return hits, total, accuracy

def evaluate_all_combinations(df, regime_series, max_combo_size=4):
    combo_stats = {}
    indicator_names = list(INDICATOR_FUNCTIONS.keys())
    combos = get_indicator_combinations(indicator_names, max_size=max_combo_size)
    regimes = ['trend', 'range', 'volatile']
    for regime in regimes:
        for combo in combos:
            hits, total, accuracy = backtest_combination(df, INDICATOR_FUNCTIONS, combo, regime, regime_series)
            combo_stats[(regime, frozenset(combo))] = (hits, total, accuracy)
    return combo_stats

def rate_current_combination(df, idx, regime, combo_stats):
    active_inds = []
    for name, func in INDICATOR_FUNCTIONS.items():
        signal = func(df, idx)
        if signal in ("up", "down"):
            active_inds.append(name)
    if len(active_inds) < 2:
        return None, 0, 0
    combo_key = (regime, frozenset(active_inds))
    hits, total, accuracy = combo_stats.get(combo_key, (0, 0, 0))
    return accuracy, hits, total

def find_best_combination(combo_stats, regime):
    filtered = {k: v for k, v in combo_stats.items() if k[0] == regime}
    if not filtered:
        return None, 0, 0, 0
    best_combo_key = max(filtered, key=lambda k: filtered[k][2])
    hits, total, accuracy = filtered[best_combo_key]
    return best_combo_key[1], hits, total, accuracy

def check_combo_latest_signal_correctness(df, combo, idx=-1):
    if idx < 0:
        idx = len(df) - 1
    signals = []
    for ind_name in combo:
        signal = INDICATOR_FUNCTIONS[ind_name](df, idx)
        if signal not in ("up", "down"):
            return None
        signals.append(signal)
    if len(set(signals)) != 1:
        return None
    combo_signal = signals[0]
    price_now = df['close'].iloc[idx]
    if idx + 2 >= len(df):
        return None
    price_1 = df['close'].iloc[idx + 1]
    price_2 = df['close'].iloc[idx + 2]
    if combo_signal == "up" and price_1 > price_now and price_2 > price_1:
        return True
    elif combo_signal == "down" and price_1 < price_now and price_2 < price_1:
        return True
    else:
        return False

# --- Main loop ---

def main():
    symbol = "EUR/USD"
    last_checked_minute = None
    combo_stats = None
    regime_series = None

    while True:
        now = datetime.utcnow()
        if now.minute % 15 == 0 and (last_checked_minute != now.minute or last_checked_minute is None):
            last_checked_minute = now.minute
            print(f"\nChecking signals for {symbol} at {now.strftime('%Y-%m-%d %H:%M:%S UTC')}")
            df = fetch_latest_ohlcv(symbol, timeframe='15m', limit=750)
            if df is None or len(df) < 700:
                print(f"Not enough data for {symbol}, skipping.")
            else:
                df = calculate_indicators(df)
                regime_series = compute_regime_series(df)
                backtest_df = df.iloc[-751:-1]
                backtest_regime_series = regime_series.loc[backtest_df.index]

                combo_stats = evaluate_all_combinations(backtest_df, backtest_regime_series, max_combo_size=4)

                accuracies, hits, totals, ongoing = get_indicator_accuracies_and_hits(backtest_df, update_perf=True)
                regime = detect_regime(df)
                signal, confidence, indicator_states = adaptive_regime_check_signal(df, regime)

                best_combo, best_hits, best_total, best_accuracy = find_best_combination(combo_stats, regime)
                best_combo_correct = None
                if best_combo is not None:
                    best_combo_correct = check_combo_latest_signal_correctness(df, best_combo, idx=-1)

                if signal in ("buy", "sell"):
                    last_close_time = df.index[-1].strftime('%Y-%m-%d %H:%M UTC')
                    combo_accuracy, combo_hits, combo_total = rate_current_combination(df, -1, regime, combo_stats)
                    message = format_signal_message(
                        symbol, signal, confidence, indicator_states, hits, totals, ongoing,
                        last_close_time, regime, combo_accuracy, combo_hits, combo_total,
                        best_combo=best_combo, best_accuracy=best_accuracy, best_correct=best_combo_correct
                    )
                    send_telegram_message(message)
                    print(message)
                else:
                    print(f"No clear buy or sell signal detected for {symbol}. Regime: {regime}")

            now = datetime.utcnow()
            minutes_until_next_15 = (15 - (now.minute % 15)) % 15
            if minutes_until_next_15 == 0:
                minutes_until_next_15 = 15
            sleep_seconds = minutes_until_next_15 * 60 - now.second
            print(f"\nSleeping for {int(sleep_seconds // 60)} minutes until next 15-minute candle...")
            time.sleep(sleep_seconds)
        else:
            time.sleep(10)

if __name__ == "__main__":
    main()