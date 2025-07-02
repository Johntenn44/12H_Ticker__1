import ccxt
import pandas as pd
import numpy as np
import traceback
from datetime import datetime

# --- INDICATOR CALCULATIONS ---

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

# --- TREND ANALYSIS FUNCTIONS ---

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

# --- SIGNAL CHECK (MAJORITY VOTING) ---

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

    up_signals = signals.count("up")
    down_signals = signals.count("down")

    if up_signals > down_signals:
        return "buy"
    elif down_signals > up_signals:
        return "sell"
    else:
        return None

# --- FETCH OHLCV DATA ---

def fetch_ohlcv(symbol='EUR/USD', timeframe='15m', limit=1000):
    try:
        exchange = ccxt.kraken()
        exchange.load_markets()
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df.astype(float)
    except Exception as e:
        print(f"Error fetching OHLCV data: {e}")
        traceback.print_exc()
        return None

# --- HIGHER TIMEFRAME TREND (1H EMA CROSSOVER) ---

def fetch_higher_ohlcv(symbol='EUR/USD', timeframe='1h', limit=500):
    try:
        exchange = ccxt.kraken()
        exchange.load_markets()
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df.astype(float)
    except Exception as e:
        print(f"Error fetching higher timeframe OHLCV data: {e}")
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

# --- SIGNAL CHECK WITH HTF FILTER ---

def check_signal_with_htf_filter(df_15m, htf_trend):
    signal = check_signal(df_15m)
    if signal is None or htf_trend is None:
        return None
    if signal == "buy" and htf_trend == "up":
        return "buy"
    elif signal == "sell" and htf_trend == "down":
        return "sell"
    else:
        return None

# --- MAIN FUNCTION ---

def main():
    print(f"Fetching data at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')} ...")
    df_15m = fetch_ohlcv()
    df_1h = fetch_higher_ohlcv()
    if df_15m is None or df_15m.empty or df_1h is None or df_1h.empty:
        print("Failed to fetch data.")
        return
    htf_trend = higher_timeframe_trend(df_1h)
    signal = check_signal_with_htf_filter(df_15m, htf_trend)
    last_close_time = df_15m.index[-1].strftime('%Y-%m-%d %H:%M UTC')
    if signal == "buy":
        print(f"🚀 Buy Signal confirmed by 1h trend at {last_close_time}")
    elif signal == "sell":
        print(f"🔥 Sell Signal confirmed by 1h trend at {last_close_time}")
    else:
        print(f"No confirmed signal at {last_close_time}")

if __name__ == "__main__":
    main()
