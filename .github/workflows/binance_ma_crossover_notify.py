import ccxt
import pandas as pd
import numpy as np
from datetime import datetime
import time

# (Include your indicator functions here: calculate_rsi, calculate_stoch_rsi, calculate_multi_wr, analyze_wr_relative_positions,
# calculate_kdj, analyze_stoch_rsi_trend, analyze_rsi_trend, analyze_kdj_trend, check_signal)

# For brevity, assume all your functions from the original script are defined here or imported.

def fetch_historical_ohlcv(symbol='EUR/USD', timeframe='15m', limit=1000):
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
        return None

def backtest_signal_persistence(df, persistence_candles=[3,4]):
    results = {p: {'buy_success':0, 'buy_total':0, 'sell_success':0, 'sell_total':0} for p in persistence_candles}
    
    # Iterate over dataframe starting from index where indicators can be calculated
    for i in range(50, len(df) - max(persistence_candles) - 1):  # 50 to ensure indicator windows are valid
        window_df = df.iloc[:i+1]  # data up to current candle
        signal = check_signal(window_df)
        if signal in ['buy', 'sell']:
            close_price = df['close'].iloc[i]
            for p in persistence_candles:
                future_closes = df['close'].iloc[i+1:i+1+p]
                if len(future_closes) < p:
                    continue  # Not enough future data
                
                if signal == 'buy':
                    # Check if price generally increased or stayed above signal close for all next p candles
                    if all(future_closes > close_price):
                        results[p]['buy_success'] += 1
                    results[p]['buy_total'] += 1
                elif signal == 'sell':
                    # Check if price generally decreased or stayed below signal close for all next p candles
                    if all(future_closes < close_price):
                        results[p]['sell_success'] += 1
                    results[p]['sell_total'] += 1

    # Calculate success rates
    for p in persistence_candles:
        buy_rate = (results[p]['buy_success'] / results[p]['buy_total'] * 100) if results[p]['buy_total'] > 0 else 0
        sell_rate = (results[p]['sell_success'] / results[p]['sell_total'] * 100) if results[p]['sell_total'] > 0 else 0
        print(f"Persistence {p} candles:")
        print(f"  Buy signals: {results[p]['buy_total']} total, {results[p]['buy_success']} successful ({buy_rate:.2f}%)")
        print(f"  Sell signals: {results[p]['sell_total']} total, {results[p]['sell_success']} successful ({sell_rate:.2f}%)")
        print("")

def main():
    df = fetch_historical_ohlcv()
    if df is None or df.empty:
        print("Failed to fetch data for backtest.")
        return
    backtest_signal_persistence(df)

if __name__ == "__main__":
    main()
