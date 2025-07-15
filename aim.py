import subprocess
import sys
import os
import atexit
import asyncio
import aiohttp
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict, deque
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
import math

# ────────────────────  Web-server Sub-process  ────────────────────────────────
webserver_path = os.path.join(os.path.dirname(__file__), 'webserver.py')
webserver_process = subprocess.Popen([sys.executable, webserver_path])


def cleanup() -> None:
    """Terminate child processes gracefully on exit."""
    print("Terminating webserver subprocess …")
    webserver_process.terminate()


atexit.register(cleanup)

# ────────────────────  Telegram configuration  ───────────────────────────────
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
TELEGRAM_CHANNEL_ID = os.environ.get("TELEGRAM_CHANNEL_ID")
TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
TELEGRAM_MAX_LEN = 4096


async def _telegram_post(session: aiohttp.ClientSession, text: str,
                         chat_id: str) -> None:
    try:
        async with session.post(
            TELEGRAM_API,
            data={
                "chat_id": chat_id,
                "text": text,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            },
            timeout=15,
        ) as resp:
            if resp.status != 200:
                print(f"Telegram error {chat_id}: {await resp.text()}")
    except Exception as e:
        print(f"Telegram exception: {e}")


async def send_telegram_message(message: str,
                                chat_id_override: str | None = None) -> None:
    """Chunk long messages and send with asyncio + aiohttp."""
    if not TELEGRAM_BOT_TOKEN:
        print("TELEGRAM_BOT_TOKEN not set! Cannot send messages.")
        return

    targets = (
        [chat_id_override]
        if chat_id_override
        else list(filter(None, [TELEGRAM_CHAT_ID, TELEGRAM_CHANNEL_ID]))
    )

    # chunk message
    parts, buff = [], ""
    for line in (message.split("\n") if len(message) > TELEGRAM_MAX_LEN else
                 [message]):
        if len(buff) + len(line) + 1 <= TELEGRAM_MAX_LEN:
            buff += line + "\n"
        else:
            parts.append(buff)
            buff = line + "\n"
    if buff:
        parts.append(buff)

    async with aiohttp.ClientSession() as session:
        await asyncio.gather(*[
            _telegram_post(session, part, chat_id)
            for part in parts
            for chat_id in targets
        ])


# ────────────────────  Exchange initialisation  ──────────────────────────────
exchange: ccxt.kucoin | None = None
exchange_lock = threading.Lock()


def initialize_exchange() -> None:
    global exchange
    if exchange is not None:
        return
    try:
        exchange = ccxt.kucoin()
        exchange.load_markets()
        print("CCXT exchange initialised.")
    except Exception as e:
        print(f"Error initialising CCXT: {e}")
        sys.exit(1)


initialize_exchange()

# ────────────────────  Symbols universe (trimmed)  ───────────────────────────
CRYPTO_SYMBOLS = [
    "XRP/USDT", "XMR/USDT", "GMX/USDT", "LUNA/USDT", "TRX/USDT", "EIGEN/USDT",
    "APE/USDT", "WAVES/USDT", "PLUME/USDT", "SUSHI/USDT", "DOGE/USDT",
    "VIRTUAL/USDT", "CAKE/USDT", "GRASS/USDT", "AAVE/USDT", "SUI/USDT",
    "ARB/USDT", "XLM/USDT", "MNT/USDT", "LTC/USDT", "NEAR/USDT",
]

# ────────────────────  Technical-indicator helpers  ───────────────────────────
def calculate_rsi(series: pd.Series, period: int = 13) -> pd.Series:
    delta = series.diff()
    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calculate_stoch_rsi(
    df: pd.DataFrame, rsi_len: int = 13, stoch_len: int = 8, smooth_k: int = 5,
    smooth_d: int = 3
) -> tuple[pd.Series, pd.Series]:
    rsi = calculate_rsi(df["close"], rsi_len)
    min_rsi = rsi.rolling(window=stoch_len).min()
    max_rsi = rsi.rolling(window=stoch_len).max()
    pct = (rsi - min_rsi) / (max_rsi - min_rsi) * 100
    pct = pct.ffill()
    k = pct.rolling(window=smooth_k).mean()
    d = k.rolling(window=smooth_d).mean()
    return k, d


def calculate_multi_wr(df: pd.DataFrame,
                       lengths: list[int] = [3, 13, 144, 8, 233, 55]
                       ) -> dict[int, pd.Series]:
    wr = {}
    for L in lengths:
        hh = df["high"].rolling(L).max()
        ll = df["low"].rolling(L).min()
        wr[L] = ((hh - df["close"]) / (hh - ll) * -100).ffill()
    return wr


def calculate_kdj(df: pd.DataFrame, length: int = 5, ma1: int = 8,
                  ma2: int = 8) -> tuple[pd.Series, pd.Series, pd.Series]:
    low_min = df["low"].rolling(length, min_periods=1).min()
    high_max = df["high"].rolling(length, min_periods=1).max()
    rsv = ((df["close"] - low_min) / (high_max - low_min) * 100).ffill()
    k = rsv.ewm(span=ma1, adjust=False).mean()
    d = k.ewm(span=ma2, adjust=False).mean()
    j = 3 * k - 2 * d
    return k, d, j


def calculate_macd(close: pd.Series, fast: int = 12, slow: int = 26,
                   signal: int = 9) -> tuple[pd.Series, pd.Series]:
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_f - ema_s
    sig_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, sig_line


def calculate_bollinger_bands(
    close: pd.Series, window: int = 20, num_std: int = 2
) -> tuple[pd.Series, pd.Series]:
    sma = close.rolling(window).mean()
    std = close.rolling(window).std()
    return sma + num_std * std, sma - num_std * std


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["close"] = df["close"].astype(float)
    df["rsi5"] = calculate_rsi(df["close"], 5)
    df["rsi13"] = calculate_rsi(df["close"], 13)
    df["rsi21"] = calculate_rsi(df["close"], 21)

    k, d = calculate_stoch_rsi(df)
    df["stochrsi_k"], df["stochrsi_d"] = k, d

    wr = calculate_multi_wr(df)
    for L, series in wr.items():
        df[f"wr_{L}"] = series

    kdj_k, kdj_d, kdj_j = calculate_kdj(df)
    df["kdj_k"], df["kdj_d"], df["kdj_j"] = kdj_k, kdj_d, kdj_j

    macd_line, macd_signal = calculate_macd(df["close"])
    df["macd_line"], df["macd_signal"] = macd_line, macd_signal

    ub, lb = calculate_bollinger_bands(df["close"])
    df["bb_upper"], df["bb_lower"] = ub, lb

    return df


# ────────────────────  Signal-analysis helpers  ───────────────────────────────
def analyze_trend(series1: pd.Series, series2: pd.Series,
                  idx: int) -> str | None:
    if idx < 1 or pd.isna(series1.iloc[idx]) or pd.isna(series2.iloc[idx]):
        return None
    if series1.iloc[idx] > series2.iloc[idx]:
        return "up"
    if series1.iloc[idx] < series2.iloc[idx]:
        return "down"
    return None


def analyze_wr_positions(wr_dict: dict[int, pd.Series], idx: int) -> str | None:
    try:
        w8, w3, w144, w233, w55 = (wr_dict[x].iloc[idx]
                                   for x in (8, 3, 144, 233, 55))
    except Exception:
        return None

    if w8 > w233 and w3 > w233 and w8 > w144 and w3 > w144:
        return "up"
    if w8 < w55 and w3 < w55:
        return "down"
    return None


INDICATOR_FUNCTIONS: dict[str, callable] = {
    "Stoch RSI":
        lambda df, idx: analyze_trend(df["stochrsi_k"], df["stochrsi_d"], idx),
    "Williams %R":
        lambda df, idx: analyze_wr_positions(
            {L: df[f"wr_{L}"] for L in [3, 13, 144, 8, 233, 55]}, idx),
    "RSI":
        lambda df, idx: "up"
        if df["rsi5"].iloc[idx] > df["rsi13"].iloc[idx] > df[
            "rsi21"].iloc[idx] else
        "down"
        if df["rsi5"].iloc[idx] < df["rsi13"].iloc[idx] < df[
            "rsi21"].iloc[idx] else None,
    "KDJ":
        lambda df, idx: analyze_trend(df["kdj_j"], df["kdj_d"], idx),
    "MACD":
        lambda df, idx: analyze_trend(df["macd_line"], df["macd_signal"], idx),
    "Bollinger":
        lambda df, idx: "up"
        if df["close"].iloc[idx] < df["bb_lower"].iloc[idx] else "down"
        if df["close"].iloc[idx] > df["bb_upper"].iloc[idx] else None,
}

# ────────────────────  ML components  ─────────────────────────────────────────
indicator_performance_regime: defaultdict[str, dict[str, deque]] = defaultdict(
    lambda: {"high_vol": deque(maxlen=200), "low_vol": deque(maxlen=200)}
)

ml_X: deque[list[int]] = deque(maxlen=2000)
ml_y: deque[int] = deque(maxlen=2000)

base_ml_model = LogisticRegression(solver="liblinear", random_state=42)
calibrated_ml_model: CalibratedClassifierCV | None = None
ml_trained: bool = False
ml_lock = threading.Lock()


def get_market_regime(df: pd.DataFrame, window: int = 50,
                      threshold: float = 0.02) -> str:
    if len(df) < window:
        return "unknown"
    vol = df["close"].pct_change().rolling(window).std().iloc[-1]
    return "high_vol" if vol > threshold else "low_vol"


def update_indicator_performance(name: str, regime: str, correct: bool) -> None:
    if regime == "unknown":
        return
    indicator_performance_regime[name][regime].append(1 if correct else 0)


def get_indicator_weight(name: str, regime: str) -> float:
    if regime == "unknown":
        return 0.5
    perf = indicator_performance_regime[name][regime]
    return sum(perf) / len(perf) if perf else 0.5


def add_ml_sample(signals: dict[str, str | None], price_move: float) -> None:
    feats = [(1 if s == "up" else -1 if s == "down" else 0)
             for s in signals.values()]
    ml_X.append(feats)
    ml_y.append(1 if price_move > 0 else 0)


def train_ml_model() -> None:
    """Train once per scan cycle (called in main, not per-symbol)."""
    global ml_trained, calibrated_ml_model
    with ml_lock:
        if len(ml_X) > len(INDICATOR_FUNCTIONS) * 5:
            try:
                base_ml_model.fit(np.array(ml_X), np.array(ml_y))
                calibrated_ml_model = CalibratedClassifierCV(
                    base_ml_model, cv="prefit", method="sigmoid")
                calibrated_ml_model.fit(np.array(ml_X), np.array(ml_y))
                ml_trained = True
                print("ML model trained & calibrated.")
            except Exception as e:
                print(f"ML training error: {e}")
                ml_trained = False
                calibrated_ml_model = None
        else:
            ml_trained = False
            calibrated_ml_model = None


def ml_predict(signals: dict[str, str | None]
               ) -> tuple[str | None, float]:
    if not ml_trained or calibrated_ml_model is None:
        return None, 0.0
    feats = np.array([(1 if s == "up" else -1 if s == "down" else 0)
                      for s in signals.values()]).reshape(1, -1)
    try:
        proba = calibrated_ml_model.predict_proba(feats)[0]
        max_proba = max(proba)
        if max_proba < 0.55:
            return None, 0.0
        return ("buy", proba[1] * 100) if proba[1] > proba[0] else (
            "sell", proba[0] * 100)
    except Exception as e:
        print(f"ML predict error: {e}")
        return None, 0.0


def adaptive_signal(
    df: pd.DataFrame
) -> tuple[str | None, float, dict[str, str | None], str]:
    regime = get_market_regime(df)
    signals = {n: fn(df, -1) for n, fn in INDICATOR_FUNCTIONS.items()}
    ml_sig, ml_conf = ml_predict(signals)
    if ml_sig:
        return ml_sig, ml_conf, signals, regime

    up_w = down_w = total = 0.0
    for n, s in signals.items():
        w = get_indicator_weight(n, regime)
        if s == "up":
            up_w += w
            total += w
        if s == "down":
            down_w += w
            total += w

    if total > 0.1:
        if up_w > down_w * 1.2:
            return "buy", (up_w / total) * 100, signals, regime
        if down_w > up_w * 1.2:
            return "sell", (down_w / total) * 100, signals, regime

    if all(s == "up" for s in signals.values() if s is not None):
        avg_conf = np.mean(
            [get_indicator_weight(n, regime) for n in INDICATOR_FUNCTIONS]) * 100
        return "buy", avg_conf, signals, regime
    if all(s == "down" for s in signals.values() if s is not None):
        avg_conf = np.mean(
            [get_indicator_weight(n, regime) for n in INDICATOR_FUNCTIONS]) * 100
        return "sell", avg_conf, signals, regime

    return None, 0.0, signals, regime


def backtest_update(df: pd.DataFrame) -> None:
    """Generate samples for ML & indicator stats (no training here)."""
    start = max(50, 0)
    if len(df) < start + 2:
        return

    for i in range(start, len(df) - 1):
        regime = get_market_regime(df[: i + 1])
        if regime == "unknown":
            continue
        sigs = {n: fn(df, i) for n, fn in INDICATOR_FUNCTIONS.items()}
        p0, p1 = df["close"].iloc[i], df["close"].iloc[i + 1]

        for n, s in sigs.items():
            if s == "up":
                update_indicator_performance(n, regime, p1 > p0)
            if s == "down":
                update_indicator_performance(n, regime, p1 < p0)

        add_ml_sample(sigs, p1 - p0)
    # NOTE: training moved to main loop – once per cycle


# ────────────────────  Risk / position sizing  ───────────────────────────────
MAX_UNLEVERAGED_NOTIONAL = 1.0  # USD
LEVERAGE = 15                  # 15×


def calculate_position_size_dollars(price: float) -> tuple[float, int]:
    """Return (notional USD, units) for ≤$1 un-levered exposure."""
    units = math.floor(MAX_UNLEVERAGED_NOTIONAL / price)
    return (units * price, units) if units else (0.0, 0)


# ────────────────────  Exchange I/O helpers  ─────────────────────────────────
def fetch_latest_ohlcv(symbol: str,
                       timeframe: str = "4h",
                       limit: int = 750) -> pd.DataFrame | None:
    """Thread-safe CCXT OHLCV fetch."""
    try:
        with exchange_lock:
            if symbol not in exchange.symbols:
                print(f"{symbol} not on exchange.")
                return None
            data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

        df = pd.DataFrame(
            data, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        df = df.apply(pd.to_numeric, errors="coerce")
        df.dropna(subset=["close"], inplace=True)
        return df
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None


# ────────────────────  Single-symbol pipeline  ───────────────────────────────
def process_symbol(symbol: str) -> dict | None:
    df = fetch_latest_ohlcv(symbol)
    if df is None or len(df) < 100:
        return None

    df = calculate_indicators(df)
    sub = df.iloc[-750:]
    if len(sub) > 50:
        # populate ML sample queues
        backtest_update(sub)

    sig, conf, states, regime = adaptive_signal(df)

    if sig in ("buy", "sell") and conf >= 55.0:
        price = df["close"].iloc[-1]
        weights = [get_indicator_weight(n, regime) for n in INDICATOR_FUNCTIONS]
        avg_perf = sum(weights) / len(weights) if weights else 0.5
        ttp = determine_trailing_take_profit(regime, avg_perf)
        stop_loss_pct = (ttp / 100) * 0.5

        notional_usd, units = calculate_position_size_dollars(price)
        if units == 0:
            return None

        return {
            "symbol": symbol,
            "signal": sig,
            "confidence": conf,
            "indicator_states": states,
            "regime": regime,
            "current_price": price,
            "ttp_percent": ttp,
            "stop_loss_pct": stop_loss_pct * 100,
            "position_size_units": units,
            "position_size_usd": notional_usd,
            "leverage": LEVERAGE,
        }
    return None


# ────────────────────  Trailing-take-profit utility  ─────────────────────────
def determine_trailing_take_profit(regime: str, accuracy: float) -> float:
    base = 3.0 if regime == "high_vol" else 1.0
    return max(0.5, min(base * (1.5 - accuracy), 4.0))


# ────────────────────  Message formatting helpers  ───────────────────────────
def format_single_coin_detail(
    symbol: str,
    signal: str,
    confidence: float,
    indicator_states: dict[str, str | None],
    regime: str,
    price_str: str,
    ttp_percent: float | None = None,
    stop_loss_pct: float | None = None,
    position_size_units: int | None = None,
    position_size_usd: float | None = None,
    leverage: int | None = None,
) -> str:
    emo = "🟢 BUY" if signal == "buy" else "🔴 SELL"
    reg_emo = ("📈 High Volatility" if regime == "high_vol" else
               "📉 Low Volatility" if regime == "low_vol" else "❓ Unknown")
    commentary = {
        "buy": ["Momentum building! 🚀", "Breakout ahead! 📈",
                "Bulls taking charge! 🐂"],
        "sell": ["Caution downtrend! ⚠️", "Bears in control. 🐻",
                 "Possible pullback 🔻"],
    }
    comment = np.random.choice(commentary[signal])
    details = "\n".join(
        f"  • {n}: "
        f"{'✅ Up' if s == 'up' else '❌ Down' if s == 'down' else '⚪ Neutral'}"
        for n, s in indicator_states.items()
    )
    ttp_line = (f"  <b>Trailing Take Profit:</b> <code>{ttp_percent:.2f}%</code>\n"
                if ttp_percent else "")
    sl_line = (f"  <b>Stop Loss Distance:</b> <code>{stop_loss_pct:.2f}%</code>\n"
               if stop_loss_pct else "")
    pos_line = ""
    if (position_size_units is not None and position_size_usd is not None
            and leverage is not None):
        pos_line = (
            f"  <b>Position Size:</b> <code>{position_size_units} units "
            f"≈ ${position_size_usd:.2f}</code>\n"
            f"  <b>Leverage:</b> <code>{leverage}×</code>\n"
        )
    display_conf = min(confidence, 99.9)
    return (
        f"<b>{symbol} | {emo} ({display_conf:.1f}%)</b>\n"
        f"  <b>Price:</b> <code>{price_str}</code>\n"
        f"  <b>Market:</b> {reg_emo}\n"
        f"{ttp_line}{sl_line}{pos_line}"
        f"  <i>{comment}</i>\n"
        f"  <u>Indicators:</u>\n{details}\n"
    )


def format_summary_message(coin_results: list[dict],
                           timestamp_str: str) -> str:
    if not coin_results:
        return (
            f"<b>📊 Market Scan @ {timestamp_str} (4h TF)</b>\n\n"
            f"<i>No strong BUY/SELL signals detected.</i>"
        )

    sorted_results = sorted(coin_results,
                            key=lambda x: x["confidence"],
                            reverse=True)
    hour = datetime.utcnow().hour
    greeting = ("🌅 Good morning" if 5 <= hour < 12 else
                "🌞 Good afternoon" if 12 <= hour < 18 else "🌙 Good evening")
    header = (
        f"{greeting}, trader!\n"
        f"<b>📊 Crypto Signals Scan @ {timestamp_str} (4h TF)</b>\n"
        f"<i>Top {min(5, len(sorted_results))} opportunities:</i>\n"
    )
    perf = ("<b>🧠 System Overview:</b>\n" +
            "".join(
                f"  • {n}: High Vol "
                f"{get_indicator_weight(n, 'high_vol'):.1%}, Low Vol "
                f"{get_indicator_weight(n, 'low_vol'):.1%}\n"
                for n in INDICATOR_FUNCTIONS
            ) +
            f"  • ML Model: {'✅ Ready' if ml_trained else '⏳ Training…'}\n\n")

    details = ("<b>⭐ Top Signals:</b>\n" +
               "".join(
                   format_single_coin_detail(
                       res["symbol"],
                       res["signal"],
                       res["confidence"],
                       res["indicator_states"],
                       res["regime"],
                       f"{res['current_price']:.4f}" if
                       res["current_price"] < 10 else
                       f"{res['current_price']:.2f}",
                       ttp_percent=res.get("ttp_percent"),
                       stop_loss_pct=res.get("stop_loss_pct"),
                       position_size_units=res.get("position_size_units"),
                       position_size_usd=res.get("position_size_usd"),
                       leverage=res.get("leverage"),
                   ) for res in sorted_results[:5]))

    closing = np.random.choice([
        "<i>Trade smart, manage risk!</i>",
        "<i>Markets are dynamic. Stay sharp!</i>",
        "<i>Not financial advice.</i>",
    ])
    return header + perf + details + closing


# ────────────────────  Scheduler utilities  ──────────────────────────────────
def seconds_until_next_4h_utc() -> float:
    now = datetime.utcnow()
    next_hour = ((now.hour // 4) + 1) * 4 % 24
    next_run = datetime.combine(now.date(), datetime.min.time()) + timedelta(
        hours=next_hour)
    if next_run <= now:
        next_run += timedelta(days=1)
    return (next_run - now).total_seconds()


# ────────────────────  Main runtime loop  ─────────────────────────────────────
def main() -> None:
    try:
        while True:
            cycle_start = datetime.utcnow()
            timestamp = cycle_start.strftime("%Y-%m-%d %H:%M UTC")
            print(f"\nStarting scan @ {timestamp}")

            coin_results: list[dict] = []
            # Use ×2 threads per core (I/O bound)
            workers = min(32, (os.cpu_count() or 1) * 2)

            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(process_symbol, sym): sym
                    for sym in CRYPTO_SYMBOLS
                }
                for fut in as_completed(futures):
                    res = fut.result()
                    if res:
                        coin_results.append(res)

            # Train ML model ONCE per scan cycle
            train_ml_model()

            print(f"Scan complete — {len(coin_results)} actionable signals.")
            msg = format_summary_message(coin_results, timestamp)
            asyncio.run(send_telegram_message(msg))

            sleep_seconds = seconds_until_next_4h_utc()
            print(f"Sleeping {sleep_seconds:.0f}s until next 4h UTC boundary.")
            time.sleep(sleep_seconds)
    except KeyboardInterrupt:
        print("Interrupted. Exiting.")
        cleanup()


if __name__ == "__main__":
    main()
