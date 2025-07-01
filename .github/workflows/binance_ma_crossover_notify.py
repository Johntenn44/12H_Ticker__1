import os
import ccxt
import pandas as pd
import numpy as np
import requests
from datetime import datetime
import traceback

# --- CONFIGURATION ---

EXCHANGE_ID = 'kraken'   # or 'kucoin'
INTERVAL = '15m'
LOOKBACK = 192  # 2 days
hours)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    raise ValueError("Telegram bot token or chat ID not set in environment variables.")

# --- INDICATOR CALCULATIONS ---

def calculate_rsi(series, period=13):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_stoch_rsi(df, rsi_length=13, stoch_length=8, smooth_k=5, smooth_d=3):
    rsi = calculate_rsi(df['close'], rsi_length)
    min_rsi = rsi.rolling(window=stoch_length).min()
    max_rsi = rsi.rolling(window=stoch_length).max()
    denominator = max_rsi - min_rsi
    denominator = denominator.replace(0, np.nan)
    stoch_rsi = (rsi - min_rsi) / denominator * 100
    stoch_rsi = stoch_rsi.fillna(method='ffill')
    k = stoch_rsi.rolling(window=smooth_k).mean()
    d = k.rolling(window=smooth_d).mean()
    return k, d

def calculate_wr(df, length):
    highest_high = df['high'].rolling(window=length).max()
    lowest_low = df['low'].rolling(window=length).min()
    denominator = highest_high - lowest_low
    denominator = denominator.replace(0, np.nan)
    wr = (highest_high - df['close']) / denominator * -100
    return wr.fillna(method='ffill')

# --- TREND LOGIC ---

def analyze_stoch_rsi_trend(k, d):
    if len(k) < 2 or pd.isna(k.iloc[-2]) or pd.isna(d.iloc[-2]) or pd.isna(k.iloc[-1]) or pd.isna(d.iloc[-1]):
        return "No clear Stoch RSI trend"
    if k.iloc[-2] < d.iloc[-2] and k.iloc[-1] > d.iloc[-1] and k.iloc[-1] < 80:
        return "Uptrend"
    elif k.iloc[-2] > d.iloc[-2] and k.iloc[-1] < d.iloc[-1] and k.iloc[-1] > 20:
        return "Downtrend"
    else:
        return "No clear Stoch RSI trend"

def analyze_wr_trend(wr_series):
    if len(wr_series) < 2 or pd.isna(wr_series.iloc[-2]) or pd.isna(wr_series.iloc[-1]):
        return "No clear WR trend"
    prev, curr = wr_series.iloc[-2], wr_series.iloc[-1]
    if prev > -80 and curr <= -80:
        return "WR Oversold - Buy signal"
    elif prev < -20 and curr >= -20:
        return "WR Overbought - Sell signal"
    else:
        return "No clear WR trend"

# --- DATA FETCHING AND EUR/USD CALCULATION ---

def fetch_ohlcv(symbol, timeframe, limit):
    exchange_class = getattr(ccxt, EXCHANGE_ID)
    exchange = exchange_class()
    exchange.load_markets()
    if symbol not in exchange.symbols:
        raise ValueError(f"Symbol {symbol} not available on {EXCHANGE_ID}")
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df.astype(float)
    return df

def get_eur_usd_from_usdt_pairs(timeframe, limit):
    # Fetch USDT/USD and USDT/EUR OHLCV data
    df_usdt_usd = fetch_ohlcv('USDT/USD', timeframe, limit)
    df_usdt_eur = fetch_ohlcv('USDT/EUR', timeframe, limit)

    # Align timestamps (inner join)
    df = df_usdt_usd.join(df_usdt_eur, lsuffix='_usd', rsuffix='_eur', how='inner')

    # Calculate EUR/USD prices = USDT/USD price / USDT/EUR price
    df['open'] = df['open_usd'] / df['open_eur']
    df['high'] = df['high_usd'] / df['high_eur']
    df['low'] = df['low_usd'] / df['low_eur']
    df['close'] = df['close_usd'] / df['close_eur']

    # Volume: average of both volumes as proxy
    df['volume'] = (df['volume_usd'] + df['volume_eur']) / 2

    # Keep only relevant columns
    df = df[['open', 'high', 'low', 'close', 'volume']]

    return df

# --- BACKTEST FUNCTION WITH NET PnL ---

def backtest(df):
    signals = []
    WR_PERIODS = [3, 8, 13, 55, 144, 233]
    start_index = max(20, max(WR_PERIODS))

    position = None
    net_pnl = 0.0

    for i in range(start_index, len(df)):
        window = df.iloc[:i+1].copy()

        k, d = calculate_stoch_rsi(window)
        stoch_trend = analyze_stoch_rsi_trend(k, d)

        wr_signals = []
        for period in WR_PERIODS:
            wr = calculate_wr(window, period)
            trend = analyze_wr_trend(wr)
            if "No clear" not in trend:
                wr_signals.append(f"WR{period}: {trend}")

        # Simple trade logic: enter long on Uptrend, exit on Downtrend
        if position is None and stoch_trend == "Uptrend":
            position = {'entry_price': window['close'].iloc[-1]}
        elif position is not None and stoch_trend == "Downtrend":
            exit_price = window['close'].iloc[-1]
            trade_pnl = exit_price - position['entry_price']
            net_pnl += trade_pnl
            position = None

        if "No clear" not in stoch_trend or wr_signals:
            signals.append({
                'timestamp': window.index[-1],
                'stoch_trend': stoch_trend,
                'wr_signals': ", ".join(wr_signals) if wr_signals else "No WR signals",
                'net_pnl': net_pnl
            })

    # Close open position at last price
    if position is not None:
        exit_price = df['close'].iloc[-1]
        trade_pnl = exit_price - position['entry_price']
        net_pnl += trade_pnl
        position = None

    return signals, net_pnl

# --- TELEGRAM NOTIFICATION ---

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML"
    }
    resp = requests.post(url, json=payload)
    resp.raise_for_status()

# --- MAIN LOGIC ---

def main():
    dt = datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')
    try:
        df = get_eur_usd_from_usdt_pairs(INTERVAL, LOOKBACK)
        if len(df) < LOOKBACK:
            print(f"Not enough data for backtest: got {len(df)} candles, need {LOOKBACK}")
            return

        current_price = df['close'].iloc[-1]

        signals, net_pnl = backtest(df)

        msg_lines = [
            f"<b>{EXCHANGE_ID.capitalize()} EUR/USD Backtest Report ({dt})</b>",
            f"Current Price: {current_price:.6f} EUR/USD",
            f"Backtest Period: {len(df)} candles ({INTERVAL})",
            f"<b>Net Profit/Loss: {net_pnl:.6f} per unit traded</b>",
            ""
        ]

        if signals:
            msg_lines.append("<b>Signals:</b>")
            for s in signals:
                ts = s['timestamp'].strftime('%Y-%m-%d %H:%M')
                msg_lines.append(f"{ts} | StochRSI: {s['stoch_trend']} | WR: {s['wr_signals']} | Net PnL: {s['net_pnl']:.6f}")
        else:
            msg_lines.append("No trading signals detected.")

        send_telegram_message("\n".join(msg_lines))
        print("Backtest report sent.")

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
