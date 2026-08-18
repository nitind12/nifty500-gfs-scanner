"""
NIFTY 500 Daily + 4H RSI + Hourly Volume Scanner
--------------------------------------------------
Scans all NIFTY 500 stocks and returns stocks where:
  1. Daily RSI(14) > 60
  2. 4-hour RSI(14) > 60
  3. Latest hourly volume > 10,000 shares

Data source: yfinance.

Implementation note:
- Daily RSI is calculated from daily closing prices.
- 4H candles are constructed from 1-hour OHLCV data and aligned to NSE
  market-session boundaries starting at 09:15 IST. The current incomplete
  4H candle is excluded, so RSI is based only on a completed 4H candle.
- Hourly volume condition uses the latest available 1H candle volume.

Outputs:
  nifty500_rsi_volume_intraday_YYYYMMDD.csv
  nifty500_rsi_volume_intraday_latest.csv

Run:
  python nifty500_rsi_volume_intraday.py
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
DAILY_RSI_MIN = 60
FOUR_HOUR_RSI_MIN = 60
HOURLY_VOLUME_MIN = 10000
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


def build_4h(hourly):
    """Build completed 4H OHLCV candles aligned to NSE 09:15 session start."""
    hourly = hourly.copy()
    hourly = hourly[~hourly.index.duplicated(keep="last")]

    # 1h timestamps are aligned around 09:15, 10:15, ... on NSE data.
    # Offset 1h15m makes 4h buckets begin at 09:15 and 13:15.
    four_h = hourly.resample(
        "4h", origin="start_day", offset="1h15min"
    ).agg(
        {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }
    ).dropna(subset=["Open", "High", "Low", "Close"])

    if len(four_h) < 2:
        return four_h

    # A normal NSE session does not contain a full second 4H candle.
    # Exclude the most recent bucket unless it contains a full 4 hours
    # of hourly observations. This prevents RSI from using an incomplete bar.
    latest_bucket = four_h.index[-1]
    latest_hours = hourly.loc[hourly.index >= latest_bucket]
    if len(latest_hours) < 4:
        four_h = four_h.iloc[:-1]

    return four_h


def check_stock(symbol):
    try:
        # Daily history for reliable Daily RSI(14).
        daily = yf.download(
            symbol,
            period="1y",
            interval="1d",
            progress=False,
            auto_adjust=False,
        )
        if daily.empty or len(daily) < 80:
            return None
        daily = clean_columns(daily).dropna(subset=["Close"])
        daily_close = daily["Close"].astype(float)
        daily_rsi = compute_rsi(daily_close).iloc[-1]

        # 1H data is used for both 4H RSI and the hourly-volume filter.
        hourly = yf.download(
            symbol,
            period="60d",
            interval="1h",
            progress=False,
            auto_adjust=False,
        )
        if hourly.empty or len(hourly) < 100:
            return None
        hourly = clean_columns(hourly).dropna(
            subset=["Open", "High", "Low", "Close", "Volume"]
        )
        hourly["Volume"] = hourly["Volume"].astype(float)

        four_h = build_4h(hourly)
        if len(four_h) < RSI_PERIOD + 1:
            return None

        four_h_rsi = compute_rsi(four_h["Close"].astype(float)).iloc[-1]
        latest_hourly_volume = hourly["Volume"].iloc[-1]
        latest_close = daily_close.iloc[-1]

        if not (
            daily_rsi > DAILY_RSI_MIN
            and four_h_rsi > FOUR_HOUR_RSI_MIN
            and latest_hourly_volume > HOURLY_VOLUME_MIN
        ):
            return None

        return {
            "Symbol": symbol.replace(".NS", ""),
            "Daily_RSI": round(float(daily_rsi), 2),
            "4H_RSI": round(float(four_h_rsi), 2),
            "Hourly_Volume": int(latest_hourly_volume),
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
        "Symbol", "Daily_RSI", "4H_RSI", "Hourly_Volume",
        "Last_Close", "Scan_Date"
    ]
    out = pd.DataFrame(results, columns=columns)
    if not out.empty:
        out = out.sort_values(
            ["Daily_RSI", "4H_RSI"], ascending=False
        ).reset_index(drop=True)

    date_str = dt.date.today().strftime("%Y%m%d")
    dated_path = os.path.join(
        OUTPUT_DIR, f"nifty500_rsi_volume_intraday_{date_str}.csv"
    )
    latest_path = os.path.join(
        OUTPUT_DIR, "nifty500_rsi_volume_intraday_latest.csv"
    )
    out.to_csv(dated_path, index=False)
    out.to_csv(latest_path, index=False)

    print(f"\nDone. {len(out)} stocks matched out of {total} scanned.")
    print(f"Saved: {dated_path}")
    print(f"Saved: {latest_path}")


if __name__ == "__main__":
    main()
