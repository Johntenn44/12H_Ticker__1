import subprocess
import sys
import os
import atexit
import requests
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time

import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score

# --- Global Configurations ---
webserver_path = os.path.join(os.path.dirname(__file__), 'webserver.py')
webserver_process = subprocess.Popen([sys.executable, webserver_path])

def cleanup():
    print("Terminating webserver subprocess...")
    webserver_process.terminate()

atexit.register(cleanup)

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
TELEGRAM_CHANNEL_ID = os.environ.get("TELEGRAM_CHANNEL_ID")

exchange = None
exchange_lock = threading.Lock()

def initialize_exchange():
    global exchange
    if exchange is None:
        try:
            exchange = ccxt.kucoin()
            exchange.load_markets()
            print("CCXT exchange initialized and markets loaded.")
        except Exception as e:
            print(f"Error initializing CCXT exchange: {e}")
            sys.exit(1)

initialize_exchange()

CRYPTO_SYMBOLS = [
    "XRP/USDT", "XMR/USDT", "GMX/USDT", "LUNA/USDT", "TRX/USDT", "EIGEN/USDT",
    "APE/USDT", "WAVES/USDT", "PLUME/USDT", "SUSHI/USDT", "DOGE/USDT", "VIRTUAL/USDT",
    "CAKE/USDT", "GRASS/USDT", "AAVE/USDT", "SUI/USDT", "ARB/USDT", "XLM/USDT",
    "MNT/USDT", "LTC/USDT", "NEAR/USDT"
]

# --- Telegram Functions ---
def send_telegram_message(message, chat_id_override=None):
    if not TELEGRAM_BOT_TOKEN:
        print("TELEGRAM_BOT_TOKEN not set! Cannot send messages.")
        return

    MAX_MESSAGE_LENGTH = 4096

    target_chat_ids = []
    if chat_id_override:
        target_chat_ids = [chat_id_override]
    else:
        if TELEGRAM_CHAT_ID:
            target_chat_ids.append(TELEGRAM_CHAT_ID)
        if TELEGRAM_CHANNEL_ID:
            target_chat_ids.append(TELEGRAM_CHANNEL_ID)

    if not target_chat_ids:
        print("No TELEGRAM_CHAT_ID or TELEGRAM_CHANNEL_ID set, message not sent!")
        return

    messages_to_send = []
    if len(message) > MAX_MESSAGE_LENGTH:
        parts = []
        current_part = ""
        for line in message.split('\n'):
            if len(current_part) + len(line) + 1 <= MAX_MESSAGE_LENGTH:
                current_part += line + '\n'
            else:
                parts.append(current_part)
                current_part = line + '\n'
        if current_part:
            parts.append(current_part)
        messages_to_send = parts
        print(f"Message too long ({len(message)} chars), split into {len(messages_to_send)} parts.")
    else:
        messages_to_send = [message]

    sent_any = False
    for msg_part in messages_to_send:
        for chat_id in target_chat_ids:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            payload = {
                'chat_id': chat_id,
                'text': msg_part,
                'parse_mode': 'HTML',
                'disable_web_page_preview': True
            }
            try:
                response = requests.post(url, data=payload, timeout=15)
                if not response.ok:
                    print(f"Failed to send message part to {chat_id}: {response.text}")
                else:
                    sent_any = True
            except Exception as e:
                print(f"Error sending Telegram message part to {chat_id}: {e}")
    if not sent_any and not messages_to_send:
        print("No messages were generated or sent.")

# --- Indicator Calculations ---
# (Keep your existing indicator functions here, unchanged)

# --- ML and Backtest Globals ---
ml_training_X = deque(maxlen=1000)
ml_training_y = deque(maxlen=1000)
ml_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
ml_model_trained = False
ml_model_lock = threading.Lock()

ml_retrain_count = 0
ml_retrain_performance = deque(maxlen=10)  # Store last 10 retrain accuracies

# --- Indicator Functions Mapping ---
# (Keep your INDICATOR_FUNCTIONS dictionary unchanged)

def indicator_signals_to_features(indicator_signals):
    features = []
    for name in INDICATOR_FUNCTIONS.keys():
        val = indicator_signals.get(name)
        if val == "up":
            features.append(1)
        elif val == "down":
            features.append(-1)
        else:
            features.append(0)
    return np.array(features)

def cross_validate_and_train(X, y):
    global ml_model_trained, ml_retrain_count, ml_retrain_performance
    tscv = TimeSeriesSplit(n_splits=5)
    accuracies = []
    for train_index, val_index in tscv.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        acc = accuracy_score(y_val, preds)
        accuracies.append(acc)
    avg_acc = np.mean(accuracies)
    # Train final model on all data
    ml_model.fit(X, y)
    ml_model_trained = True
    ml_retrain_count += 1
    ml_retrain_performance.append(avg_acc)
    print(f"ML model retrained #{ml_retrain_count} with CV accuracy: {avg_acc:.2%}")

def update_ml_model():
    with ml_model_lock:
        if len(ml_training_X) > len(INDICATOR_FUNCTIONS) * 5:
            try:
                X = np.array(list(ml_training_X))
                y = np.array(list(ml_training_y))
                cross_validate_and_train(X, y)
            except Exception as e:
                print(f"Error training ML model: {e}")
        else:
            print("Not enough data to train ML model.")
            global ml_model_trained
            ml_model_trained = False

def add_ml_training_sample(indicator_signals, price_move):
    features = []
    for name in INDICATOR_FUNCTIONS.keys():
        val = indicator_signals.get(name)
        if val == "up":
            features.append(1)
        elif val == "down":
            features.append(-1)
        else:
            features.append(0)
    ml_training_X.append(features)
    ml_training_y.append(1 if price_move > 0 else 0)

def update_performance_and_ml_from_backtest(df):
    start_idx = max(50, 0)
    if len(df) < start_idx + 2:
        return
    for idx in range(start_idx, len(df) - 1):
        regime = get_market_regime(df.iloc[:idx+1])
        if regime == "unknown":
            continue
        current_indicator_signals = {name: func(df, idx) for name, func in INDICATOR_FUNCTIONS.items()}
        price_now = df['close'].iloc[idx]
        price_next = df['close'].iloc[idx+1]
        price_move = price_next - price_now
        for name, signal in current_indicator_signals.items():
            if signal == "up":
                was_correct = price_next > price_now
                update_indicator_performance_regime(name, regime, was_correct)
            elif signal == "down":
                was_correct = price_next < price_now
                update_indicator_performance_regime(name, regime, was_correct)
        add_ml_training_sample(current_indicator_signals, price_move)
    update_ml_model()

# --- Fetch OHLCV and other functions ---
# (Keep your existing fetch_latest_ohlcv, calculate_indicators, get_market_regime, update_indicator_performance_regime, get_indicator_weight_regime unchanged)

def format_retrain_summary_message(timestamp_str):
    if ml_retrain_count == 0:
        return "<i>No ML retraining has occurred yet.</i>"
    avg_acc = np.mean(ml_retrain_performance) if ml_retrain_performance else 0.0
    return (
        f"<b>🤖 ML Retrain Summary @ {timestamp_str} (4h TF)</b>\n\n"
        f"• Total Retrains: <b>{ml_retrain_count}</b>\n"
        f"• Average CV Accuracy: <b>{avg_acc*100:.2f}%</b>\n\n"
        f"<i>Model adapts as new data arrives. Monitor retrain frequency and accuracy.</i>"
    )

def seconds_until_next_4h_utc():
    now = datetime.utcnow()
    next_hour = (now.hour // 4 + 1) * 4
    if next_hour >= 24:
        next_day = now.date() + timedelta(days=1)
        next_run = datetime.combine(next_day, datetime.min.time()) + timedelta(hours=0)
    else:
        next_run = datetime.combine(now.date(), datetime.min.time()) + timedelta(hours=next_hour)
    delta = next_run - now
    return max(delta.total_seconds(), 0)

def main():
    try:
        while True:
            start_time = datetime.utcnow()
            print(f"\nStarting backtest and retrain summary at {start_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")

            MAX_WORKERS = min(16, os.cpu_count() * 2 + 1)
            print(f"Using {MAX_WORKERS} threads for parallel processing.")

            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_to_symbol = {executor.submit(fetch_latest_ohlcv, symbol, '4h', 750): symbol for symbol in CRYPTO_SYMBOLS}
                for future in as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    try:
                        df = future.result()
                        if df is not None and len(df) >= 100:
                            df = calculate_indicators(df.copy())
                            backtest_data_start_idx = max(0, len(df) - 750)
                            backtest_df_subset = df.iloc[backtest_data_start_idx:]
                            if len(backtest_df_subset) > 50:
                                update_performance_and_ml_from_backtest(backtest_df_subset)
                            else:
                                print(f"Skipping backtest for {symbol}, insufficient data.")
                        else:
                            print(f"Insufficient data for {symbol}, skipping backtest.")
                    except Exception as exc:
                        print(f"{symbol} backtest exception: {exc}")

            summary_message = format_retrain_summary_message(start_time.strftime('%Y-%m-%d %H:%M UTC'))
            send_telegram_message(summary_message)
            print("Retrain summary sent to Telegram.")

            sleep_seconds = seconds_until_next_4h_utc()
            print(f"Sleeping {sleep_seconds:.0f} seconds until next 4-hour UTC boundary.")
            time.sleep(sleep_seconds)

    except KeyboardInterrupt:
        print("Interrupted by user. Exiting.")
        cleanup()

if __name__ == "__main__":
    main()
