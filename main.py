import os
import ccxt
import pandas as 
import requests
from datetime import datetime

# --- CONFIGURATION ---

COINS = [
    "XRP/USDT",
    "XMR/USDT",
    "GMX/USDT",
    "LUNA/USDT",
    "TRX/USDT",
    "EIGEN/USDT",
    "APE/USDT",
    "WAVES/USDT",
    "PLUME/USDT",
    "SUSHI/USDT",
    "DOGE/USDT",
    "VIRTUAL/USDT",
    "CAKE/USDT",
    "GRASS/USDT",
    "AAVE/USDT",
    "SUI/USDT",
    "ARB/USDT",
    "XLM/USDT",
    "MNT/USDT",
    "LTC/USDT",
    "NEAR/USDT",
    # Add more symbols here
]

EXCHANGE_ID = 'kucoin'
INTERVAL = '12h'      # Use 6-hour candles (change to '4h' if you want 4-hour candles)
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
    # Use last two closes for trend analysis
    cp1 = df['close'].iloc[-1]   # Most recent close
    cp2 = df['close'].iloc[-2]   # Previous close

    A1 = df['EMA8'].iloc[-1]
    B1 = df['EMA13'].iloc[-1]
    C1 = df['EMA21'].iloc[-1]
    D1 = df['EMA50'].iloc[-1]
    E1 = df['EMA200'].iloc[-1]
    MA50_1 = df['MA50'].iloc[-1]
    MA200_1 = df['MA200'].iloc[-1]

    A2 = df['EMA8'].iloc[-2]
    B2 = df['EMA13'].iloc[-2]
    C2 = df['EMA21'].iloc[-2]
    D2 = df['EMA50'].iloc[-2]
    E2 = df['EMA200'].iloc[-2]
    MA50_2 = df['MA50'].iloc[-2]
    MA200_2 = df['MA200'].iloc[-2]

    # --- Start Conditions: both last closes must meet the trend condition ---
    if (E1 > cp1 > A1 > B1 > C1 > D1 > MA50_1) and (cp1 < MA200_1) and \
       (E2 > cp2 > A2 > B2 > C2 > D2 > MA50_2) and (cp2 < MA200_2):
        results['start'] = 'uptrend'
    elif (E1 < cp1 < A1 < B1 < C1 < D1 < MA50_1) and (cp1 > MA200_1) and \
         (E2 < cp2 < A2 < B2 < C2 < D2 < MA50_2) and (cp2 > MA200_2):
        results['start'] = 'downtrend'

    results['values'] = {
        'cp1': cp1, 'cp2': cp2, 'EMA8': A1, 'EMA13': B1, 'EMA21': C1,
        'EMA50': D1, 'EMA200': E1, 'MA50': MA50_1, 'MA200': MA200_1
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
            if 'start' in trend:
                vals = trend['values']
                msg = (
                    f"<b>Kucoin {INTERVAL.upper()} Trend Alert ({dt})</b>\n"
                    f"<b>Symbol:</b> <code>{symbol}</code>\n"
                    f"Start: <b>{trend['start']}</b>\n"
                    f"\n<code>cp1={vals['cp1']:.5f}, cp2={vals['cp2']:.5f}, EMA8={vals['EMA8']:.5f}, EMA13={vals['EMA13']:.5f}, "
                    f"EMA21={vals['EMA21']:.5f}, EMA50={vals['EMA50']:.5f}, EMA200={vals['EMA200']:.5f}, "
                    f"MA50={vals['MA50']:.5f}, MA200={vals['MA200']:.5f}</code>"
                )
                messages.append(msg)
        except Exception as e:
            print(f"Error processing {symbol}: {e}")

    if messages:
        for msg in messages:
            send_telegram_message(msg)
    else:
        # Send "No trend signals for any coin" if there are no signals
        send_telegram_message("No trend signals for any coin.")

if __name__ == "__main__":
    main()
