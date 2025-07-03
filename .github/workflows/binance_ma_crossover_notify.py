import ccxt
import pandas as pd
import numpy as np
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

def analyze_wr_relative_positions(wr_dict, idx):
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

def analyze_stoch_rsi_trend(k, d, idx):
    if idx < 1 or pd.isna(k.iloc[idx-1]) or pd.isna(d.iloc[idx-1]) or pd.isna(k.iloc[idx]) or pd.isna(d.iloc[idx]):
        return None
    if k.iloc[idx-1] < d.iloc[idx-1] and k.iloc[idx] > d.iloc[idx] and k.iloc[idx] < 80:
        return "up"
    elif k.iloc[idx-1] > d.iloc[idx-1] and k.iloc[idx] < d.iloc[idx] and k.iloc[idx] > 20:
        return "down"
    else:
        return None

def analyze_rsi_trend(rsi5, rsi13, rsi21, idx):
    if rsi5.iloc[idx] > rsi13.iloc[idx] > rsi21.iloc[idx]:
        return "up"
    elif rsi5.iloc[idx] < rsi13.iloc[idx] < rsi21.iloc[idx]:
        return "down"
    else:
        return None

def analyze_kdj_trend(k, d, j, idx):
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

def check_signal(df, idx):
    k, d = calculate_stoch_rsi(df)
    wr_dict = calculate_multi_wr(df)
    wr_trend = analyze_wr_relative_positions(wr_dict, idx)

    rsi5 = calculate_rsi(df['close'], 5)
    rsi13 = calculate_rsi(df['close'], 13)
    rsi21 = calculate_rsi(df['close'], 21)
    kdj_k, kdj_d, kdj_j = calculate_kdj(df)

    stoch_trend = analyze_stoch_rsi_trend(k, d, idx)
    rsi_trend = analyze_rsi_trend(rsi5, rsi13, rsi21, idx)
    kdj_trend = analyze_kdj_trend(kdj_k, kdj_d, kdj_j, idx)

    signals = [stoch_trend, wr_trend, rsi_trend, kdj_trend]

    macd_line, macd_signal = calculate_macd(df['close'])
    macd_trend = None
    if idx > 0:
        if macd_line.iloc[idx-1] < macd_signal.iloc[idx-1] and macd_line.iloc[idx] > macd_signal.iloc[idx]:
            macd_trend = "up"
        elif macd_line.iloc[idx-1] > macd_signal.iloc[idx-1] and macd_line.iloc[idx] < macd_signal.iloc[idx]:
            macd_trend = "down"
    if macd_trend:
        signals.append(macd_trend)

    upper_band, lower_band = calculate_bollinger_bands(df['close'])
    price = df['close'].iloc[idx]
    bb_trend = None
    if not np.isnan(lower_band.iloc[idx]) and price < lower_band.iloc[idx]:
        bb_trend = "up"
    elif not np.isnan(upper_band.iloc[idx]) and price > upper_band.iloc[idx]:
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

# --- Crypto pairs ---
CRYPTO_SYMBOLS = [
    "XRP/USDT", "XMR/USDT", "GMX/USDT", "LUNA/USDT", "TRX/USDT", "EIGEN/USDT",
    "APE/USDT", "WAVES/USDT", "PLUME/USDT", "SUSHI/USDT", "DOGE/USDT", "VIRTUAL/USDT",
    "CAKE/USDT", "GRASS/USDT", "AAVE/USDT", "SUI/USDT", "ARB/USDT", "XLM/USDT",
    "MNT/USDT", "LTC/USDT", "NEAR/USDT"
]

def fetch_latest_ohlcv(symbol, timeframe='4h', limit=40):
    try:
        exchange = ccxt.binance()
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

def backtest():
    print(f"Backtesting for the last 4 days on 4h candles ({6 candles per day}, 24 candles)...\n")
    for symbol in CRYPTO_SYMBOLS:
        print(f"\n=== {symbol} ===")
        df = fetch_latest_ohlcv(symbol, timeframe='4h', limit=40)
        if df is None or df.empty:
            print("No data, skipping.")
            continue

        # Only last 24 candles (4 days)
        df = df.iloc[-24:]
        signals = []
        for idx in range(len(df)):
            signal = check_signal(df, idx)
            if signal in ["buy", "sell"]:
                signals.append((df.index[idx], signal, df['close'].iloc[idx]))
        if signals:
            for ts, sig, price in signals:
                print(f"{ts.strftime('%Y-%m-%d %H:%M')} | {sig.upper()} at price {price}")
        else:
            print("No signals in last 4 days.")

if __name__ == "__main__":
    backtest()
