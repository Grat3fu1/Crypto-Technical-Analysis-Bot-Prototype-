from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import ccxt
import pandas as pd
import matplotlib.pyplot as plt
import requests


SYMBOL = "BTC/USDT"
TIMEFRAME_DEFAULT = "1m"

MA_PERIODS: List[int] = [5, 7, 14]  # adjust if you want (e.g., [20, 50, 200])


def _load_api_keys() -> Tuple[str, str]:
    """Load Binance API keys from env vars, with optional fallback to a local key.py.

    IMPORTANT: Do NOT commit key.py to GitHub.
    Prefer env vars:
      - BINANCE_API_KEY
      - BINANCE_API_SECRET
    """
    api_key = os.getenv("BINANCE_API_KEY", "").strip()
    api_secret = os.getenv("BINANCE_API_SECRET", "").strip()

    if api_key and api_secret:
        return api_key, api_secret

    # Fallback for local development only
    try:
        import key as key_module  # type: ignore

        api_key = api_key or getattr(key_module, "API_key", "")
        api_secret = api_secret or getattr(key_module, "secret_key", "")
    except Exception:
        pass

    return str(api_key).strip(), str(api_secret).strip()


API_KEY, API_SECRET = _load_api_keys()

_exchange_config = {
    "timeout": 30000,  # 30 seconds
    "enableRateLimit": True,
}
# Only attach credentials if they exist (keeps public endpoints working cleanly)
if API_KEY and API_SECRET:
    _exchange_config.update({"apiKey": API_KEY, "secret": API_SECRET})

exchange = ccxt.binance(_exchange_config)


def test_connection() -> None:
    """Basic connectivity check to Binance."""
    try:
        response = requests.get("https://api.binance.com", timeout=10)
        print("Connection successful:", response.status_code)
    except Exception as exc:
        print("Connection failed:", exc)


# Load markets once up-front
exchange.load_markets()


# ============================== DATA HELPERS ==============================
def get_market_info(symbol: str = SYMBOL) -> None:
    ticker = exchange.fetch_ticker(symbol)
    info = {
        "symbol": ticker.get("symbol"),
        "high": ticker.get("high"),
        "low": ticker.get("low"),
        "bid": ticker.get("bid"),
        "ask": ticker.get("ask"),
        "last": ticker.get("last"),
        "change": ticker.get("change"),
        "percentage": ticker.get("percentage"),
        "basevolume": ticker.get("baseVolume"),
        "quotevolume": ticker.get("quoteVolume"),
    }

    while True:
        cmd = input(
            "Enter command (symbol/high/low/bid/ask/last/change/percentage/basevolume/quotevolume) or 'q': "
        ).strip().lower()

        if cmd == "q":
            break
        if cmd in info:
            print(f"{cmd}: {info[cmd]}")
        else:
            print("Invalid command.")


def get_ohlc(symbol: str = SYMBOL, timeframe: str = TIMEFRAME_DEFAULT, limit: int = 250) -> List[List[float]]:
    """Fetch OHLCV data."""
    return exchange.fetch_ohlcv(symbol, timeframe, limit=limit)


def get_order_book(symbol: str = SYMBOL) -> None:
    order_book = exchange.fetch_order_book(symbol)
    best_bid = order_book["bids"][0][0] if order_book.get("bids") else None
    best_ask = order_book["asks"][0][0] if order_book.get("asks") else None
    print(f"{symbol} Bid: {best_bid}\n{symbol} Ask: {best_ask}")


def get_balance() -> None:
    """Requires API keys."""
    balance = exchange.fetch_balance()
    for coin, amount in balance.get("total", {}).items():
        if amount and amount > 0:
            print(f"{coin}: {amount}")


def add_moving_averages(df: pd.DataFrame, periods: List[int] = MA_PERIODS) -> pd.DataFrame:
    """Add SMA/EMA columns to the DataFrame."""
    for period in periods:
        df[f"SMA_{period}"] = df["close"].rolling(window=period).mean()
        df[f"EMA_{period}"] = df["close"].ewm(span=period, adjust=False).mean()
    return df


def build_dataframe(timeframe: str = TIMEFRAME_DEFAULT, symbol: str = SYMBOL) -> pd.DataFrame:
    """Fetch OHLCV and return a DataFrame with time columns + moving averages."""
    ohlc = get_ohlc(symbol=symbol, timeframe=timeframe)
    df = pd.DataFrame(ohlc, columns=["timestamp", "open", "high", "low", "close", "volume"])

    # Convert timestamp to datetime and extract date/time components (Asia/Jakarta)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert("Asia/Jakarta")
    df["date"] = df["datetime"].dt.date
    df["time"] = df["datetime"].dt.time

    df = add_moving_averages(df, periods=MA_PERIODS)

    display_cols = [
        "date",
        "time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        *[f"SMA_{p}" for p in MA_PERIODS],
        *[f"EMA_{p}" for p in MA_PERIODS],
    ]
    print(df[display_cols])
    print("Most recent data is at the bottom of the table")
    print("-" * 110)
    return df


# --- Backwards-compatible aliases (keeps old function names working) ---
window = MA_PERIODS  # original variable name
add_moving_average = add_moving_averages  # original function name
DataFrame = build_dataframe  # original function name


# ============================== SIGNAL CHECKS ==============================
def ema_distance_check(
    df: pd.DataFrame,
    ema_period: int = 50,
    upper_threshold: float = 0.992,
    lower_threshold: float = 1.008,
    monitor: bool = False,
    counter: int = 0,
) -> Optional[pd.DataFrame]:
    """Check if close is meaningfully above/below EMA_{ema_period}.

    Returns a tiny DataFrame if an alert is triggered, otherwise None.
    """
    ema_col = f"EMA_{ema_period}"
    if ema_col not in df.columns:
        df = add_moving_averages(df.copy(), periods=[ema_period])

    alerts: List[Tuple[int, object, object, float, float, str]] = []
    date_val = df["date"].iloc[-1] if "date" in df.columns else None
    time_val = df["time"].iloc[-1] if "time" in df.columns else None
    close_price = float(df["close"].iloc[-1])
    ema_val = float(df[ema_col].iloc[-1])

    idx = counter if monitor else int(df.index[-1])

    if close_price * upper_threshold >= ema_val:
        alerts.append((idx, date_val, time_val, close_price, ema_val, "Above"))
        print(f"Price is above EMA_{ema_period}: close={close_price}, ema={ema_val}")
    elif close_price * lower_threshold <= ema_val:
        alerts.append((idx, date_val, time_val, close_price, ema_val, "Below"))
        print(f"Price is below EMA_{ema_period}: close={close_price}, ema={ema_val}")

    if not alerts:
        return None

    out = pd.DataFrame(alerts, columns=["index", "date", "time", "close_price", ema_col.lower(), "position"])
    out.set_index("index", inplace=True)
    print(out)
    return out


def close_twice_sma(monitor: bool = False, counter: int = 0) -> Tuple[Optional[pd.DataFrame], int]:
    """Alert when close >= 2x previous SMA_5 (uses the last 25 candles)."""
    if not monitor:
        return None, counter

    data = exchange.fetch_ohlcv(symbol=SYMBOL, timeframe=TIMEFRAME_DEFAULT, limit=25)
    df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = add_moving_averages(df, periods=MA_PERIODS)

    close_val = float(df["close"].iloc[-1])
    prev_sma_5 = float(df["SMA_5"].iloc[-2])

    timestamp = pd.to_datetime(df["timestamp"].iloc[-1], unit="ms", utc=True).tz_convert("Asia/Jakarta")
    date_val = timestamp.date()
    time_val = timestamp.time()

    if close_val >= 2 * prev_sma_5:
        alerts = [(counter, date_val, time_val, "Close >= 2x previous SMA(5)", close_val, prev_sma_5)]
        out = pd.DataFrame(alerts, columns=["index", "date", "time", "status", "close", "previous_sma_5"])
        out.set_index("index", inplace=True)
        print(f"ALERT: Close ({close_val}) >= 2x Previous SMA(5) ({prev_sma_5})")
        print(out)
        return out, counter

    print(f"No alert: Close ({close_val:.2f}) < 2x Previous SMA(5) ({prev_sma_5:.2f})")
    return None, counter


def close_sma_difference(
    df: pd.DataFrame, monitor: bool = False, counter: int = 0, sma_period: int = MA_PERIODS[0]
) -> Tuple[pd.DataFrame, int]:
    """Check if close is above/below SMA_{sma_period}."""
    df = df.copy()
    df = add_moving_averages(df, periods=[sma_period])

    dif_list: List[Tuple[int, object, object, str, float]] = []
    rows_to_check = 1 if monitor else 5

    for i in range(1, rows_to_check + 1):
        dif = float(df["close"].iloc[-i] - df[f"SMA_{sma_period}"].iloc[-i])
        timestamp = pd.to_datetime(df["timestamp"].iloc[-i], unit="ms")
        date_val = timestamp.date()
        time_val = timestamp.time()
        idx = counter if monitor else int(df.index[-i])

        if dif < 0:
            dif_list.append((idx, date_val, time_val, "Below", abs(dif)))
        elif dif > 0:
            dif_list.append((idx, date_val, time_val, "Above", dif))
        else:
            dif_list.append((idx, date_val, time_val, "Equal", 0.0))

    out = pd.DataFrame(dif_list, columns=["Index", "Date", "Time", "Status", "Difference"])
    out.set_index("Index", inplace=True)
    return out, counter


def check_ma_cross(df: pd.DataFrame, fast: int = MA_PERIODS[0], slow: int = MA_PERIODS[1]) -> None:
    """Detect a simple SMA crossover between two periods."""
    df = add_moving_averages(df.copy(), periods=[fast, slow])

    for i in range(1, len(df)):
        prev, curr = i - 1, i
        prev_fast = df[f"SMA_{fast}"].iloc[prev]
        prev_slow = df[f"SMA_{slow}"].iloc[prev]
        curr_fast = df[f"SMA_{fast}"].iloc[curr]
        curr_slow = df[f"SMA_{slow}"].iloc[curr]

        if prev_fast < prev_slow and curr_fast > curr_slow:
            print(f"Golden Cross SMA_{fast} over SMA_{slow}")
        elif prev_fast > prev_slow and curr_fast < curr_slow:
            print(f"Death Cross SMA_{fast} under SMA_{slow}")


def plot_graph(df: pd.DataFrame, periods: List[int] = MA_PERIODS) -> None:
    cols = ["close", *[f"SMA_{p}" for p in periods]]
    df[cols].plot()
    plt.title(f"{SYMBOL} Price with Moving Averages")
    plt.show()


# ============================== MONITORING ==============================
def monitoring(periods: List[int] = MA_PERIODS) -> None:
    """Monitor latest candle and print close/SMA/bid/ask every new candle."""
    init_data = exchange.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME_DEFAULT, limit=1)
    last_candle_time = init_data[-1][0]
    counter = 0

    print("Monitoring started... (Ctrl+C to stop)")
    while True:
        try:
            candles = exchange.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME_DEFAULT, limit=25)
            current_candle_time = candles[-1][0]

            if current_candle_time != last_candle_time:
                last_candle_time = current_candle_time
                counter += 1

                ticker = exchange.fetch_ticker(SYMBOL)
                curr_bid = ticker.get("bid")
                curr_ask = ticker.get("ask")

                df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
                df = add_moving_averages(df, periods=periods)

                dif_df, _ = close_sma_difference(df.iloc[[-1]], monitor=True, counter=counter, sma_period=periods[0])
                print(dif_df)

                close_val = df["close"].iloc[-1]
                sma_val = df[f"SMA_{periods[0]}"].iloc[-1]
                print(
                    f"Close: {close_val}, SMA({periods[0]}): {sma_val}, Bid: {curr_bid}, Ask: {curr_ask}"
                )

            time.sleep(10)

        except KeyboardInterrupt:
            print("Monitoring stopped by user.")
            break
        except Exception as exc:
            print(f"An error occurred: {exc}")
            time.sleep(30)


def main() -> None:
    print("Crypto bot prototype — still in development.")
    commands = {
        "info": lambda: get_market_info(SYMBOL),
        "ohlc": lambda: print(get_ohlc(SYMBOL, TIMEFRAME_DEFAULT)),
        "balance": get_balance,
        "orderbook": lambda: get_order_book(SYMBOL),
        "order book": lambda: get_order_book(SYMBOL),
        "dataframe": lambda: build_dataframe(TIMEFRAME_DEFAULT, SYMBOL),
        "data frame": lambda: build_dataframe(TIMEFRAME_DEFAULT, SYMBOL),
        "monitoring": monitoring,
        "testconn": test_connection,
    }

    while True:
        cmd = input("Enter command (info/ohlc/balance/orderbook/dataframe/plot/monitoring/testconn/q): ").strip().lower()

        if cmd == "q":
            print("Exiting...")
            break

        if cmd == "plot":
            df = build_dataframe(TIMEFRAME_DEFAULT, SYMBOL)
            plot_graph(df)
            continue

        fn = commands.get(cmd)
        if fn is None:
            print("Invalid command. Please try again.")
            continue

        fn()


if __name__ == "__main__":
    main()
