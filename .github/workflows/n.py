import ccxt
import pandas as pd
import numpy as np
import traceback
from datetime import datetime, timedelta

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

def analyze_wr_relative_positions(wr_dict):
    try:
        wr_8 = wr_dict[8].iloc[-1]
        wr_3 = wr_dict[3].iloc[-1]
        wr_144 = wr_dict[144].iloc[-1]
        wr_233 = wr_dict[233].iloc[-1]
        wr_55 = wr_dict[55].iloc[-1]
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

def analyze_stoch_rsi_trend(k, d):
    if len(k) < 2 or pd.isna(k.iloc[-2]) or pd.isna(d.iloc[-2]) or pd.isna(k.iloc[-1]) or pd.isna(d.iloc[-1]):
        return None
    if k.iloc[-2] < d.iloc[-2] and k.iloc[-1] > d.iloc[-1] and k.iloc[-1] < 80:
        return "up"
    elif k.iloc[-2] > d.iloc[-2] and k.iloc[-1] < d.iloc[-1] and k.iloc[-1] > 20:
        return "down"
    else:
        return None

def analyze_rsi_trend(rsi5, rsi13, rsi21):
    if rsi5 > rsi13 > rsi21:
        return "up"
    elif rsi5 < rsi13 < rsi21:
        return "down"
    else:
        return None

def analyze_kdj_trend(k, d, j):
    if len(k) < 2 or len(d) < 2 or len(j) < 2:
        return None
    k_prev, k_curr = k.iloc[-2], k.iloc[-1]
    d_prev, d_curr = d.iloc[-2], d.iloc[-1]
    j_prev, j_curr = j.iloc[-2], j.iloc[-1]
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

def check_signal(df):
    k, d = calculate_stoch_rsi(df)
    wr_dict = calculate_multi_wr(df)
    wr_trend = analyze_wr_relative_positions(wr_dict)

    rsi5 = calculate_rsi(df['close'], 5).iloc[-1]
    rsi13 = calculate_rsi(df['close'], 13).iloc[-1]
    rsi21 = calculate_rsi(df['close'], 21).iloc[-1]
    kdj_k, kdj_d, kdj_j = calculate_kdj(df)

    stoch_trend = analyze_stoch_rsi_trend(k, d)
    rsi_trend = analyze_rsi_trend(rsi5, rsi13, rsi21)
    kdj_trend = analyze_kdj_trend(kdj_k, kdj_d, kdj_j)

    signals = [stoch_trend, wr_trend, rsi_trend, kdj_trend]

    macd_line, macd_signal = calculate_macd(df['close'])
    macd_trend = None
    if len(macd_line) > 1 and macd_line.iloc[-2] < macd_signal.iloc[-2] and macd_line.iloc[-1] > macd_signal.iloc[-1]:
        macd_trend = "up"
    elif len(macd_line) > 1 and macd_line.iloc[-2] > macd_signal.iloc[-2] and macd_line.iloc[-1] < macd_signal.iloc[-1]:
        macd_trend = "down"
    if macd_trend:
        signals.append(macd_trend)

    upper_band, lower_band = calculate_bollinger_bands(df['close'])
    price = df['close'].iloc[-1]
    bb_trend = None
    if price < lower_band.iloc[-1]:
        bb_trend = "up"
    elif price > upper_band.iloc[-1]:
        bb_trend = "down"
    if bb_trend:
        signals.append(bb_trend)

    up_signals = signals.count("up")
    down_signals = signals.count("down")

    if up_signals > down_signals:
        return "buy"
    elif down_signals > up_signals:
        return "sell"
    else:
        return None

def fetch_ohlcv(symbol='EUR/USD', timeframe='15m', since=None, limit=350):
    try:
        exchange = ccxt.kraken()
        exchange.load_markets()
        since_ms = int(since.timestamp() * 1000) if since else None
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since_ms, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df.astype(float)
    except Exception as e:
        print(f"Error fetching OHLCV data: {e}")
        traceback.print_exc()
        return None

def fetch_higher_ohlcv(symbol='EUR/USD', timeframe='1h', since=None, limit=100):
    try:
        exchange = ccxt.kraken()
        exchange.load_markets()
        since_ms = int(since.timestamp() * 1000) if since else None
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since_ms, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df.astype(float)
    except Exception as e:
        print(f"Error fetching higher timeframe OHLCV data: {e}")
        traceback.print_exc()
        return None

def fetch_higher_ohlcv_4h(symbol='EUR/USD', timeframe='4h', since=None, limit=30):
    try:
        exchange = ccxt.kraken()
        exchange.load_markets()
        since_ms = int(since.timestamp() * 1000) if since else None
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since_ms, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df.astype(float)
    except Exception as e:
        print(f"Error fetching 4h OHLCV data: {e}")
        traceback.print_exc()
        return None

def higher_timeframe_trend(df_higher):
    ema50 = df_higher['close'].ewm(span=50, adjust=False).mean()
    ema200 = df_higher['close'].ewm(span=200, adjust=False).mean()
    if ema50.iloc[-1] > ema200.iloc[-1]:
        return "up"
    elif ema50.iloc[-1] < ema200.iloc[-1]:
        return "down"
    else:
        return None

def check_signal_with_multi_htf(df_15m, trend_1h, trend_4h):
    signal = check_signal(df_15m)
    if signal is None or trend_1h is None or trend_4h is None:
        return None
    if signal == "buy" and trend_1h == "up" and trend_4h == "up":
        return "buy"
    elif signal == "sell" and (trend_1h == "down" or trend_4h == "down"):
        return "sell"
    else:
        return None

def backtest_signal_persistence_with_multi_htf(df_15m, df_1h, df_4h, persistence_lengths=[1, 2, 3, 4]):
    results = {p: {'buy_success': 0, 'buy_total': 0, 'sell_success': 0, 'sell_total': 0} for p in persistence_lengths}

    trend_1h = higher_timeframe_trend(df_1h)
    trend_4h = higher_timeframe_trend(df_4h)

    for i in range(50, len(df_15m) - max(persistence_lengths) - 1):
        timestamp = df_15m.index[i]
        hour = timestamp.hour
        # Skip trades between 11 PM and 6 AM
        if hour >= 23 or hour < 6:
            continue

        window_df = df_15m.iloc[:i + 1]
        signal = check_signal_with_multi_htf(window_df, trend_1h, trend_4h)
        if signal in ['buy', 'sell']:
            signal_close = df_15m['close'].iloc[i]
            print(f"Signal: {signal.upper()} at {timestamp} (Price: {signal_close})")  # Print time of entry
            for p in persistence_lengths:
                future_closes = df_15m['close'].iloc[i + 1:i + 1 + p]
                if len(future_closes) < p:
                    continue
                if signal == 'buy':
                    if all(future_closes > signal_close):
                        results[p]['buy_success'] += 1
                    results[p]['buy_total'] += 1
                elif signal == 'sell':
                    if all(future_closes < signal_close):
                        results[p]['sell_success'] += 1
                    results[p]['sell_total'] += 1

    for p in persistence_lengths:
        buy_total = results[p]['buy_total']
        sell_total = results[p]['sell_total']
        buy_success = results[p]['buy_success']
        sell_success = results[p]['sell_success']
        buy_rate = (buy_success / buy_total * 100) if buy_total > 0 else 0
        sell_rate = (sell_success / sell_total * 100) if sell_total > 0 else 0
        print(f"Persistence {p} candle(s):")
        print(f"  Buy signals: {buy_total} total, {buy_success} successful ({buy_rate:.2f}%)")
        print(f"  Sell signals: {sell_total} total, {sell_success} successful ({sell_rate:.2f}%)")
        print("")

def main():
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=3)  # Backtest past 3 days

    print(f"Fetching 15m data from {start_time} to {end_time} ...")
    df_15m = fetch_ohlcv('EUR/USD', '15m', since=start_time, limit=350)

    print(f"Fetching 1h data from {start_time} to {end_time} ...")
    df_1h = fetch_higher_ohlcv('EUR/USD', '1h', since=start_time, limit=100)

    print(f"Fetching 4h data from {start_time} to {end_time} ...")
    df_4h = fetch_higher_ohlcv_4h('EUR/USD', '4h', since=start_time, limit=30)

    if df_15m is None or df_15m.empty or df_1h is None or df_1h.empty or df_4h is None or df_4h.empty:
        print("Failed to fetch required data for 3-day backtest.")
        return

    print(f"Running backtest for last 3 days with multi-indicator, multi-timeframe confirmation, and time filter...")
    backtest_signal_persistence_with_multi_htf(df_15m, df_1h, df_4h)

if __name__ == "__main__":
    main()