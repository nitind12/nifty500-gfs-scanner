"""
NIFTY 500 - Marubozu Reversal Scanner

Finds two reversal-style candle setups on the latest CLOSED candle:
1) GREEN Marubozu after a preceding fall.
2) RED Marubozu after a preceding rise / local high.

The same script is used by Daily and Weekly workflows through TIMEFRAME:
    TIMEFRAME=DAILY   -> daily candles
    TIMEFRAME=WEEKLY  -> weekly candles

Definitions used here (kept deliberately objective for unattended scanning):
- Marubozu: candle body >= 90% of the full High-Low range and each wick <= 10%
  of the range. Green means Close > Open; Red means Close < Open.
- "After a fall": the previous 5 completed candles have a net decline in
  closing price of at least 3%, and the latest candle is the green Marubozu.
- "After a high": the previous 5 completed candles have a net rise in
  closing price of at least 3%, and the latest candle is the red Marubozu.
- Volume confirmation: the Marubozu candle's volume must be > 10,000 shares.
- The latest candle is always treated as a CLOSED candle because the scanner
  runs after the Daily/Weekly candle close.

Output is created only when at least one setup is found. A *_latest.csv is
also created only when there are results; the common cleanup/email system then
ensures no-data scanners are not mailed.
"""

import os
import time
from datetime import datetime

import pandas as pd
import yfinance as yf

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "output")
TIMEFRAME = os.environ.get("TIMEFRAME", "DAILY").upper()

INTERVAL = "1d" if TIMEFRAME == "DAILY" else "1wk"
PERIOD = "2y" if TIMEFRAME == "DAILY" else "5y"
PREFIX = "nifty500_marubozu_daily" if TIMEFRAME == "DAILY" else "nifty500_marubozu_weekly"

FALL_LOOKBACK = 5
MIN_TREND_MOVE_PCT = 3.0
MIN_BODY_RATIO = 0.90
MAX_WICK_RATIO = 0.10
MIN_CANDLE_VOLUME = 10000
SLEEP_BETWEEN_CALLS = 0.20

NSE_NIFTY500_CSV_URL = "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv"
LOCAL_FALLBACK_LIST = os.environ.get("LOCAL_FALLBACK_LIST", "ind_nifty500list.csv")


def load_symbols():
    import io
    import requests

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "text/csv,application/csv,*/*",
    }
    try:
        session = requests.Session()
        session.get("https://www.nseindia.com", headers=headers, timeout=10)
        r = session.get(NSE_NIFTY500_CSV_URL, headers=headers, timeout=15)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        col = "Symbol" if "Symbol" in df.columns else df.columns[2]
        symbols = df[col].dropna().astype(str).str.strip().tolist()
        if len(symbols) >= 400:
            print(f"[OK] NSE Nifty 500 list: {len(symbols)} symbols")
            return symbols
    except Exception as exc:
        print(f"[WARNING] NSE live list failed: {exc}")

    if os.path.exists(LOCAL_FALLBACK_LIST):
        df = pd.read_csv(LOCAL_FALLBACK_LIST)
        col = "Symbol" if "Symbol" in df.columns else df.columns[2]
        symbols = df[col].dropna().astype(str).str.strip().tolist()
        print(f"[OK] Local fallback list: {len(symbols)} symbols")
        return symbols

    raise RuntimeError("NIFTY 500 constituent list unavailable")


def fetch_data(symbol):
    try:
        df = yf.download(
            symbol + ".NS",
            period=PERIOD,
            interval=INTERVAL,
            progress=False,
            auto_adjust=False,
        )
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        if len(df) < FALL_LOOKBACK + 2:
            return None
        return df
    except Exception:
        return None


def marubozu_type(row):
    high = float(row["High"])
    low = float(row["Low"])
    opn = float(row["Open"])
    close = float(row["Close"])
    candle_range = high - low
    if candle_range <= 0:
        return None

    body = abs(close - opn)
    upper_wick = high - max(opn, close)
    lower_wick = min(opn, close) - low

    body_ratio = body / candle_range
    upper_ratio = upper_wick / candle_range
    lower_ratio = lower_wick / candle_range

    if body_ratio < MIN_BODY_RATIO:
        return None
    if upper_ratio > MAX_WICK_RATIO or lower_ratio > MAX_WICK_RATIO:
        return None

    if close > opn:
        return "GREEN"
    if close < opn:
        return "RED"
    return None


def scan_symbol(symbol):
    df = fetch_data(symbol)
    if df is None:
        return None

    latest = df.iloc[-1]
    previous = df.iloc[-(FALL_LOOKBACK + 1):-1]
    if len(previous) < FALL_LOOKBACK:
        return None

    # Volume must be greater than 10,000 shares on the Marubozu candle itself.
    candle_volume = float(latest["Volume"])
    if candle_volume <= MIN_CANDLE_VOLUME:
        return None

    pattern = marubozu_type(latest)
    if pattern is None:
        return None

    start_close = float(previous["Close"].iloc[0])
    end_close = float(previous["Close"].iloc[-1])
    trend_pct = (end_close / start_close - 1.0) * 100.0

    if pattern == "GREEN" and trend_pct <= -MIN_TREND_MOVE_PCT:
        setup = "GREEN MARUBOZU AFTER FALL"
    elif pattern == "RED" and trend_pct >= MIN_TREND_MOVE_PCT:
        setup = "RED MARUBOZU AFTER HIGH"
    else:
        return None

    high = float(latest["High"])
    low = float(latest["Low"])
    candle_range = high - low
    body_ratio = abs(float(latest["Close"]) - float(latest["Open"])) / candle_range
    upper_wick_ratio = (high - max(float(latest["Open"]), float(latest["Close"]))) / candle_range
    lower_wick_ratio = (min(float(latest["Open"]), float(latest["Close"])) - low) / candle_range

    return {
        "Symbol": symbol,
        "Date": df.index[-1].strftime("%Y-%m-%d"),
        "Setup": setup,
        "Pattern": "Green Marubozu" if pattern == "GREEN" else "Red Marubozu",
        "Open": round(float(latest["Open"]), 2),
        "High": round(high, 2),
        "Low": round(low, 2),
        "Close": round(float(latest["Close"]), 2),
        "Trend5BarsPct": round(trend_pct, 2),
        "BodyPctRange": round(body_ratio * 100, 2),
        "UpperWickPctRange": round(upper_wick_ratio * 100, 2),
        "LowerWickPctRange": round(lower_wick_ratio * 100, 2),
        "Volume": int(candle_volume),
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 90)
    print(f" NIFTY 500 MARUBOZU REVERSAL SCANNER - {TIMEFRAME}")
    print("=" * 90)
    print(f"Criteria: Green Marubozu after >= {MIN_TREND_MOVE_PCT}% fall OR "
          f"Red Marubozu after >= {MIN_TREND_MOVE_PCT}% rise/high")
    print(f"Marubozu: body >= {MIN_BODY_RATIO*100:.0f}% of range; each wick <= {MAX_WICK_RATIO*100:.0f}%")
    print(f"Volume: Marubozu candle volume > {MIN_CANDLE_VOLUME:,} shares")

    symbols = load_symbols()
    results = []

    for i, symbol in enumerate(symbols, 1):
        print(f"Scanning [{i}/{len(symbols)}]: {symbol:<15}", end="\r")
        result = scan_symbol(symbol)
        if result:
            results.append(result)
        time.sleep(SLEEP_BETWEEN_CALLS)

    print(" " * 80, end="\r")

    if not results:
        print("No eligible Marubozu reversal setup found.")
        return

    df = pd.DataFrame(results).sort_values(["Setup", "Trend5BarsPct"], ascending=[True, True])
    date_str = datetime.now().strftime("%Y%m%d")
    dated_path = os.path.join(OUTPUT_DIR, f"{PREFIX}_{date_str}.csv")
    latest_path = os.path.join(OUTPUT_DIR, f"{PREFIX}_latest.csv")
    df.to_csv(dated_path, index=False)
    df.to_csv(latest_path, index=False)

    print(f"Found {len(df)} eligible setup(s).")
    print(df.to_string(index=False))
    print(f"Saved: {dated_path}")
    print(f"Saved: {latest_path}")


if __name__ == "__main__":
    main()
