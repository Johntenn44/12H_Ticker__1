import ccxt
import pandas as pd
import numpy as np
import traceback
from datetime import datetime, timedelta

# (Include your indicator and signal functions here: calculate_rsi, calculate_stoch_rsi, calculate_multi_wr, analyze_wr_relative_positions,
# calculate_kdj, analyze_stoch_rsi_trend, analyze_rsi_trend, analyze_kdj_trend, check_signal, higher_timeframe_trend, etc.)

def fetch_ohlcv_for_period(symbol='EUR/USD', timeframe='15m', since=None, limit=1000):
    try:
        exchange = ccxt.kraken()
        exchange.load_markets()
        # ccxt expects since in milliseconds
        since_ms = int(since.timestamp() * 1000) if since else None
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since_ms, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df.astype(float)
    except Exception as e:
        print(f"Error fetching OHLCV data: {e}")
        traceback.print_exc()
        return None

def backtest_over_period(df_15m, df_1h):
    # Calculate higher timeframe trend once
    htf_trend = higher_timeframe_trend(df_1h)
    
    results = []
    # Start from index where indicators are valid (e.g., 50)
    for i in range(50, len(df_15m)):
        window_df = df_15m.iloc[:i+1]
        signal = check_signal_with_htf_filter(window_df, htf_trend)
        timestamp = df_15m.index[i]
        results.append({'timestamp': timestamp, 'signal': signal})
        print(f"{timestamp} - Signal: {signal}")
    return pd.DataFrame(results)

def main():
    # Define the period: last 2 days from now
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=2)
    
    print(f"Fetching 15m data from {start_time} to {end_time} ...")
    df_15m = fetch_ohlcv_for_period('EUR/USD', '15m', since=start_time, limit=1000)
    
    print(f"Fetching 1h data from {start_time} to {end_time} ...")
    df_1h = fetch_ohlcv_for_period('EUR/USD', '1h', since=start_time, limit=500)
    
    if df_15m is None or df_15m.empty or df_1h is None or df_1h.empty:
        print("Failed to fetch data for backtest.")
        return
    
    print(f"Running backtest for last 2 days...")
    backtest_results = backtest_over_period(df_15m, df_1h)
    
    # You can extend here to analyze results, calculate success rates, plot, etc.

if __name__ == "__main__":
    main()
