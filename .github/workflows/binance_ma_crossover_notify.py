import os
import ccxt
import pandas as pd
import requests
from datetime import datetime

# --- CONFIGURATION ---

COINS = [
    "XRP/USDT",  # Kraken uses XBT for Bitcoin
    "XMR/USDT",
    "DOGE/USDT",
    "VIRTUAL/USDT",
    "DOT/USDT",
    "SUI/USDT",
    "XLM/USDT",
    "MNT/USDT",
    "LTC/USDT",
    "NEAR/USDT",
    # Add more Kraken symbols here
]

EXCHANGE_ID = 'kucoin'
INTERVAL = '12h'      # <-- Changed from '1d' to '6h'
LOOKBACK = 210       # Number of candles to fetch (must be >= 200)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# --- INDICATOR CALCULATION ---

def add_indicators(df):
    df['EMA8'] = df['close'].ewm(span=8, adjust=False).mean()
    df['EMA13'] = df['close'].ewm(span=13, adjust=False).mean()
    df['EMA21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['EMA50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['EMA200'] = df['close'].ewm(span=200, adjust=False).mean()
    df['MA50'] = df['close'].rolling(window=50).mean()
    df['MA200'] = df['close'].rolling(window=200).mean()
    return df

# --- TREND LOGIC ---

def analyze_trend(df):
    results = {}
    cp = df['close'].iloc[-1]
    A = df['EMA8'].iloc[-1]
    B = df['EMA13'].iloc[-1]
    C = df['EMA21'].iloc[-1]
    D = df['EMA50'].iloc[-1]
    E = df['EMA200'].iloc[-1]
    MA50 = df['MA50'].iloc[-1]
    MA200 = df['MA200'].iloc[-1]

    # --- Start Conditions ---
    if (E > cp > A > B > C > D > MA50) and (cp < MA200):
        results['start'] = 'uptrend'
    elif (E < cp < A < B < C < D < MA50) and (cp > MA200):
        results['start'] = 'downtrend'

    # --- Continue Conditions ---
    if results.get('start') == 'uptrend':
        if abs(cp - E) < 1e-8 and abs(cp - MA200) > 1e-8:
            results['continue'] = 'up_a'
        elif abs(cp - MA200) < 1e-8 and abs(cp - E) > 1e-8:
            results['continue'] = 'up_b'
    elif results.get('start') == 'downtrend':
        if abs(cp - E) < 1e-8 and abs(cp - MA200) > 1e-8:
            results['continue'] = 'down_a'
        elif abs(cp - MA200) < 1e-8 and abs(cp - E) > 1e-8:
            results['continue'] = 'down_b'

    # --- Reversal/Reset Conditions ---
    if results.get('start') == 'uptrend':
        if (cp < min(A, B, C, D)) or not (A > B > C > D) or abs(cp - MA50) < 1e-8:
            results['warn'] = 'uptrend reversal/reset'
    elif results.get('start') == 'downtrend':
        if (cp > max(A, B, C, D)) or not (A < B < C < D) or abs(cp - MA50) < 1e-8:
            results['warn'] = 'downtrend reversal/reset'

    # --- End Conditions ---
    if cp >= MA200 >= E:
        results['end'] = 'uptrend end'
    elif cp <= MA200 <= E:
        results['end'] = 'downtrend end'

    # --- Save values for reporting ---
    results['values'] = {
        'cp': cp, 'EMA8': A, 'EMA13': B, 'EMA21': C, 'EMA50': D, 'EMA200': E,
        'MA50': MA50, 'MA200': MA200
    }
    return results

# --- DATA FETCHING ---

def fetch_ohlcv_ccxt(symbol, timeframe, limit):
    exchange = getattr(ccxt, EXCHANGE_ID)()
    exchange.load_markets()
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(
        ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
    )
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df['close'] = df['close'].astype(float)
    return df

# --- TELEGRAM NOTIFICATION ---

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML"
    }
    resp = requests.post(url, data=payload)
    resp.raise_for_status()

# --- MAIN LOGIC ---

def main():
    dt = datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')
    messages = []
    for symbol in COINS:
        try:
            df = fetch_ohlcv_ccxt(symbol, INTERVAL, LOOKBACK)
            if len(df) < 200:
                print(f"Not enough data for {symbol}")
                continue
            df = add_indicators(df)
            trend = analyze_trend(df)
            # Format message if any condition is met
            if any(k in trend for k in ['start', 'continue', 'warn', 'end']):
                vals = trend['values']
                msg = (
                    f"<b>Kraken 12H Trend Alert ({dt})</b>\n"
                    f"<b>Symbol:</b> <code>{symbol}</code>\n"
                )
                if 'start' in trend:
                    msg += f"Start: <b>{trend['start']}</b>\n"
                if 'continue' in trend:
                    msg += f"Continue: <b>{trend['continue']}</b>\n"
                if 'warn' in trend:
                    msg += f"⚠️ <b>Warning:</b> {trend['warn']}\n"
                if 'end' in trend:
                    msg += f"End: <b>{trend['end']}</b>\n"
                msg += (
                    f"\n<code>cp={vals['cp']:.2f}, EMA8={vals['EMA8']:.2f}, EMA13={vals['EMA13']:.2f}, "
                    f"EMA21={vals['EMA21']:.2f}, EMA50={vals['EMA50']:.2f}, EMA200={vals['EMA200']:.2f}, "
                    f"MA50={vals['MA50']:.2f}, MA200={vals['MA200']:.2f}</code>"
                )
                messages.append(msg)
        except Exception as e:
            print(f"Error processing {symbol}: {e}")

    if messages:
        for msg in messages:
            send_telegram_message(msg)
    else:
        print("No trend signals for any coin.")

if __name__ == "__main__":
    main()
