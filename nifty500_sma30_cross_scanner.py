"""
NIFTY 500 SMA30 Cross-Above Scanner
----------------------------------------
Screens NIFTY 500 for stocks where:
  - Daily close crossed ABOVE the 30-day SMA within the last N bars
  - Close is still above SMA30 today
  - SMA30 is rising (trend confirmation)
  - (Optional) Volume >= MIN_VOL_RATIO x 20-day average volume

Output: CSV file with today's date, saved to OUTPUT_DIR.

Run manually:
    python nifty500_sma30_cross_scanner.py
"""

import pandas as pd
import yfinance as yf
import datetime as dt
import os
import sys
import time

# ---------------- CONFIG ----------------
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", r"D:\\dhan\\scanner\\gfs")
NIFTY500_LIST_URL = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
LOCAL_FALLBACK_LIST = os.environ.get("LOCAL_FALLBACK_LIST", r"D:\\dhan\\ind_nifty500list.csv")
REPO_FALLBACK_LIST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ind_nifty500list.csv")

SMA_PERIOD = 30
CROSS_LOOKBACK_BARS = int(os.environ.get("CROSS_LOOKBACK_BARS", 3))   # cross must be within last N bars
SLOPE_BARS = int(os.environ.get("SLOPE_BARS", 5))                     # SMA30 must be higher than N bars ago
MIN_VOL_RATIO = float(os.environ.get("MIN_VOL_RATIO", 0))             # 0 = volume filter off
REQUEST_PAUSE_SEC = 0.3
# -----------------------------------------


def get_nifty500_symbols():
    """Fetch the NIFTY 500 constituent list from NSE archives, with local fallbacks."""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        df = pd.read_csv(NIFTY500_LIST_URL, storage_options={"User-Agent": headers["User-Agent"]})
        symbols = [s.strip() + ".NS" for s in df["Symbol"].tolist()]
        print(f"Fetched {len(symbols)} symbols from NSE archives.")
        return symbols
    except Exception as e:
        print(f"Could not fetch live NIFTY 500 list ({e}). Trying local fallback...")

    for path in (LOCAL_FALLBACK_LIST, REPO_FALLBACK_LIST):
        if os.path.exists(path):
            df = pd.read_csv(path)
            symbols = [s.strip() + ".NS" for s in df["Symbol"].tolist()]
            print(f"Loaded {len(symbols)} symbols from {path}.")
            return symbols

    print("No fallback list found. Please download ind_nifty500list.csv from NSE.")
    sys.exit(1)


def check_stock(symbol):
    """Returns dict of result if stock passes all conditions, else None."""
    try:
        df = yf.download(symbol, period="1y", interval="1d", progress=False, auto_adjust=True)
        if df.empty or len(df) < SMA_PERIOD + CROSS_LOOKBACK_BARS + SLOPE_BARS + 2:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.dropna(subset=["Close"]).copy()
        df["SMA30"] = df["Close"].rolling(SMA_PERIOD).mean()
        df["VolAvg20"] = df["Volume"].rolling(20).mean()
        df["Above"] = df["Close"] > df["SMA30"]
        df["CrossUp"] = df["Above"] & ~df["Above"].shift(1, fill_value=False)

        last = df.iloc[-1]
        if not last["Above"]:
            return None

        recent = df.tail(CROSS_LOOKBACK_BARS)
        if not recent["CrossUp"].any():
            return None

        sma_now = last["SMA30"]
        sma_prev = df["SMA30"].iloc[-1 - SLOPE_BARS]
        if not sma_now > sma_prev:
            return None

        vol_ratio = last["Volume"] / last["VolAvg20"] if last["VolAvg20"] else 0
        if vol_ratio < MIN_VOL_RATIO:
            return None

        cross_date = recent[recent["CrossUp"]].index[-1]
        return {
            "Symbol": symbol.replace(".NS", ""),
            "Last_Close": round(float(last["Close"]), 2),
            "SMA30": round(float(sma_now), 2),
            "Pct_Above_SMA": round(float(last["Close"] / sma_now - 1) * 100, 2),
            "SMA30_Slope_Pct": round(float(sma_now / sma_prev - 1) * 100, 2),
            "Vol_Ratio": round(float(vol_ratio), 2),
            "Cross_Date": cross_date.strftime("%Y-%m-%d"),
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
            print(f"  MATCH: {sym}")
            results.append(res)
        time.sleep(REQUEST_PAUSE_SEC)

    out_df = pd.DataFrame(results)
    if not out_df.empty:
        out_df = out_df.sort_values(["Cross_Date", "Vol_Ratio"], ascending=[False, False])

    date_str = dt.date.today().strftime("%Y%m%d")
    dated_path = os.path.join(OUTPUT_DIR, f"nifty500_sma30_cross_{date_str}.csv")
    latest_path = os.path.join(OUTPUT_DIR, "nifty500_sma30_cross_latest.csv")
    out_df.to_csv(dated_path, index=False)
    out_df.to_csv(latest_path, index=False)

    print(f"\nDone. {len(out_df)} stocks matched out of {total} scanned.")
    print(f"Saved: {dated_path}")
    print(f"Saved: {latest_path}")


if __name__ == "__main__":
    main()
