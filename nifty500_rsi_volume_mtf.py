"""
NIFTY 500 Monthly + Weekly RSI + Daily Volume Scanner
-------------------------------------------------------
Scans all NIFTY 500 stocks and returns stocks where:
  1. Monthly RSI(14) > 60
  2. Weekly RSI(14) > 60
  3. Latest Daily volume > 100,000 shares

The scanner uses daily OHLCV data from yfinance. Weekly and monthly RSI are
calculated from Friday-week and month-end closing prices respectively.

Outputs:
  nifty500_rsi_volume_mtf_YYYYMMDD.csv
  nifty500_rsi_volume_mtf_latest.csv

Run:
  python nifty500_rsi_volume_mtf.py
"""

import datetime as dt
import os
import sys
import time

import pandas as pd
import yfinance as yf

# ---------------- CONFIG ----------------
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "output")
NIFTY500_LIST_URL = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
LOCAL_FALLBACK_LIST = "ind_nifty500list.csv"

RSI_PERIOD = 14
MONTHLY_RSI_MIN = 60
WEEKLY_RSI_MIN = 60
DAILY_VOLUME_MIN = 100000
REQUEST_PAUSE_SEC = 0.15
# ----------------------------------------


def clean_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df


def get_nifty500_symbols():
    """Fetch the current NIFTY 500 list, with repository CSV fallback."""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        df = pd.read_csv(
            NIFTY500_LIST_URL,
            storage_options={"User-Agent": headers["User-Agent"]},
        )
        symbols = [str(s).strip() + ".NS" for s in df["Symbol"].dropna()]
        print(f"Fetched {len(symbols)} symbols from NSE archives.")
        return symbols
    except Exception as exc:
        print(f"Could not fetch live NIFTY 500 list: {exc}")
        if os.path.exists(LOCAL_FALLBACK_LIST):
            df = pd.read_csv(LOCAL_FALLBACK_LIST)
            symbols = [str(s).strip() + ".NS" for s in df["Symbol"].dropna()]
            print(f"Loaded {len(symbols)} symbols from local fallback.")
            return symbols
        print(f"Fallback file not found: {LOCAL_FALLBACK_LIST}")
        sys.exit(1)


def compute_rsi(series, period=RSI_PERIOD):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def check_stock(symbol):
    try:
        daily = yf.download(
            symbol,
            period="3y",
            interval="1d",
            progress=False,
            auto_adjust=False,
        )
        if daily.empty or len(daily) < 250:
            return None

        daily = clean_columns(daily).dropna(subset=["Close", "Volume"])
        close = daily["Close"].astype(float)
        volume = daily["Volume"].astype(float)

        weekly_close = close.resample("W-FRI").last().dropna()
        monthly_close = close.resample("ME").last().dropna()

        if len(weekly_close) < RSI_PERIOD + 1 or len(monthly_close) < RSI_PERIOD + 1:
            return None

        weekly_rsi = compute_rsi(weekly_close).iloc[-1]
        monthly_rsi = compute_rsi(monthly_close).iloc[-1]
        latest_volume = volume.iloc[-1]
        latest_close = close.iloc[-1]

        if not (
            monthly_rsi > MONTHLY_RSI_MIN
            and weekly_rsi > WEEKLY_RSI_MIN
            and latest_volume > DAILY_VOLUME_MIN
        ):
            return None

        return {
            "Symbol": symbol.replace(".NS", ""),
            "Monthly_RSI": round(float(monthly_rsi), 2),
            "Weekly_RSI": round(float(weekly_rsi), 2),
            "Daily_Volume": int(latest_volume),
            "Last_Close": round(float(latest_close), 2),
            "Scan_Date": dt.date.today().isoformat(),
        }

    except Exception as exc:
        print(f"  [skip] {symbol}: {exc}")
        return None


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    symbols = get_nifty500_symbols()
    results = []

    total = len(symbols)
    for i, symbol in enumerate(symbols, 1):
        print(f"[{i}/{total}] Checking {symbol}...")
        result = check_stock(symbol)
        if result:
            print(f"  MATCH: {symbol}")
            results.append(result)
        time.sleep(REQUEST_PAUSE_SEC)

    columns = [
        "Symbol", "Monthly_RSI", "Weekly_RSI", "Daily_Volume",
        "Last_Close", "Scan_Date"
    ]
    out = pd.DataFrame(results, columns=columns)
    if not out.empty:
        out = out.sort_values(
            ["Monthly_RSI", "Weekly_RSI"], ascending=False
        ).reset_index(drop=True)

    date_str = dt.date.today().strftime("%Y%m%d")
    dated_path = os.path.join(OUTPUT_DIR, f"nifty500_rsi_volume_mtf_{date_str}.csv")
    latest_path = os.path.join(OUTPUT_DIR, "nifty500_rsi_volume_mtf_latest.csv")
    out.to_csv(dated_path, index=False)
    out.to_csv(latest_path, index=False)

    print(f"\nDone. {len(out)} stocks matched out of {total} scanned.")
    print(f"Saved: {dated_path}")
    print(f"Saved: {latest_path}")


if __name__ == "__main__":
    main()
