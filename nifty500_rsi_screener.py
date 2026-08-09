"""
NIFTY 500 Multi-Timeframe RSI Screener
----------------------------------------
Screens NIFTY 500 for stocks where:
  - Monthly RSI(14) > 60
  - Weekly  RSI(14) > 60
  - Daily   RSI(14) recently dipped into the 35-45 "support" zone and is
            now bouncing (current RSI > the recent low, still below 60)

Output: CSV file with today's date, saved to OUTPUT_DIR.

Run manually:
    python nifty500_rsi_screener.py

Automated via Windows Task Scheduler at 3:50 PM IST on weekdays
(see setup_task_scheduler.bat in the same folder).
"""

import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
import os
import sys
import time

# ---------------- CONFIG ----------------
OUTPUT_DIR = r"D:\dhan\scanner\gfs"   # change if needed
NIFTY500_LIST_URL = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
LOCAL_FALLBACK_LIST = r"D:\dhan\ind_nifty500list.csv"  # keep a manual backup copy here

DAILY_SUPPORT_LOW = 35
DAILY_SUPPORT_HIGH = 45
DAILY_LOOKBACK_BARS = 3   # how many recent daily bars to check for the RSI-40 touch
RSI_PERIOD = 14

REQUEST_PAUSE_SEC = 0.3    # small delay between tickers to avoid throttling
# -----------------------------------------


def get_nifty500_symbols():
    """Fetch the NIFTY 500 constituent list from NSE archives, with local fallback."""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        df = pd.read_csv(NIFTY500_LIST_URL, storage_options={"User-Agent": headers["User-Agent"]})
        symbols = [s.strip() + ".NS" for s in df["Symbol"].tolist()]
        print(f"Fetched {len(symbols)} symbols from NSE archives.")
        return symbols
    except Exception as e:
        print(f"Could not fetch live NIFTY 500 list ({e}). Trying local fallback...")
        if os.path.exists(LOCAL_FALLBACK_LIST):
            df = pd.read_csv(LOCAL_FALLBACK_LIST)
            symbols = [s.strip() + ".NS" for s in df["Symbol"].tolist()]
            print(f"Loaded {len(symbols)} symbols from local fallback.")
            return symbols
        else:
            print("No fallback list found. Please download ind_nifty500list.csv from "
                  "NSE (https://www.niftyindices.com/indices/equity/broad-based-indices/nifty-500) "
                  f"and save it to {LOCAL_FALLBACK_LIST}")
            sys.exit(1)


def compute_rsi(series, period=RSI_PERIOD):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def check_stock(symbol):
    """Returns dict of result if stock passes all 3 conditions, else None."""
    try:
        # Daily data - need enough history for monthly RSI(14) => ~15+ months minimum
        daily = yf.download(symbol, period="3y", interval="1d", progress=False, auto_adjust=True)
        if daily.empty or len(daily) < 250:
            return None

        if isinstance(daily.columns, pd.MultiIndex):
            daily.columns = daily.columns.get_level_values(0)

        close_daily = daily["Close"].dropna()

        # Weekly resample (Friday close)
        weekly = close_daily.resample("W-FRI").last().dropna()
        # Monthly resample (month end close)
        monthly = close_daily.resample("ME").last().dropna()

        if len(weekly) < RSI_PERIOD + 1 or len(monthly) < RSI_PERIOD + 1:
            return None

        rsi_daily = compute_rsi(close_daily)
        rsi_weekly = compute_rsi(weekly)
        rsi_monthly = compute_rsi(monthly)

        if rsi_daily.dropna().empty or rsi_weekly.dropna().empty or rsi_monthly.dropna().empty:
            return None

        latest_daily_rsi = rsi_daily.iloc[-1]
        latest_weekly_rsi = rsi_weekly.iloc[-1]
        latest_monthly_rsi = rsi_monthly.iloc[-1]

        # Condition 1 & 2
        if not (latest_monthly_rsi > 60 and latest_weekly_rsi > 60):
            return None

        # Condition 3: daily RSI took support near 40 recently and is bouncing
        recent_daily_rsi = rsi_daily.tail(DAILY_LOOKBACK_BARS)
        touched_support = recent_daily_rsi.between(DAILY_SUPPORT_LOW, DAILY_SUPPORT_HIGH).any()
        recent_min = recent_daily_rsi.min()
        is_bouncing = latest_daily_rsi > recent_min and latest_daily_rsi < 60

        if not (touched_support and is_bouncing):
            return None

        return {
            "Symbol": symbol.replace(".NS", ""),
            "Monthly_RSI": round(latest_monthly_rsi, 2),
            "Weekly_RSI": round(latest_weekly_rsi, 2),
            "Daily_RSI": round(latest_daily_rsi, 2),
            "Daily_RSI_Recent_Low": round(recent_min, 2),
            "Last_Close": round(close_daily.iloc[-1], 2),
            "Scan_Date": dt.date.today().isoformat(),
        }

    except Exception as e:
        print(f"  [skip] {symbol}: {e}")
        return None


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    symbols = get_nifty500_symbols()

    results = []
    total = len(symbols)
    for i, sym in enumerate(symbols, 1):
        print(f"[{i}/{total}] Checking {sym}...")
        res = check_stock(sym)
        if res:
            print(f"  ✅ MATCH: {sym}")
            results.append(res)
        time.sleep(REQUEST_PAUSE_SEC)

    out_df = pd.DataFrame(results)
    out_df = out_df.sort_values("Weekly_RSI", ascending=False) if not out_df.empty else out_df

    date_str = dt.date.today().strftime("%Y%m%d")
    dated_path = os.path.join(OUTPUT_DIR, f"nifty500_rsi_scan_{date_str}.csv")
    latest_path = os.path.join(OUTPUT_DIR, "nifty500_rsi_scan_latest.csv")

    out_df.to_csv(dated_path, index=False)
    out_df.to_csv(latest_path, index=False)

    print(f"\nDone. {len(out_df)} stocks matched out of {total} scanned.")
    print(f"Saved: {dated_path}")
    print(f"Saved: {latest_path}")


if __name__ == "__main__":
    main()
