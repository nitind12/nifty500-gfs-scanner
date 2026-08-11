"""
Breakout Scanner - Pharma Sector (Weekly TF, added to daily combined workflow)
----------------------------------------------------------------------------------
Same breakout logic as the main scanner, scoped to Nifty Pharma constituents,
analyzed on Weekly candles. Runs as one more step inside your existing daily
workflow - does NOT send its own email; it just writes a CSV that the final
"send_combined_email.py" step picks up along with everything else.

Locked-in settings:
  - Universe: Nifty Pharma (~20 stocks)
  - Timeframe: Weekly
  - Lookback: 104 weeks (~2 years)

Filters applied (unchanged from the main breakout scanner):
  - Price near/at N-period high (breakout zone)
  - Volume >= 1.5x of 10-week average volume
  - RSI < 70 (avoid overbought chase, per MIB rule)
  - Close in upper half of the candle's range (strong breakout candle)
"""

import io
import os
import sys
import requests
import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt

# ----------------------------- FIXED CONFIG -----------------------------

INDEX_NAME = "Nifty Pharma"
NSE_INDEX_URL = "https://archives.nseindia.com/content/indices/ind_niftypharmalist.csv"
# Small, fairly stable list - used if both live NSE fetch AND local CSV fallback fail.
# Worth re-checking against niftyindices.com occasionally since index reconstitution
# happens periodically (semi-annual).
HARDCODED_PHARMA_FALLBACK = [
    "SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "TORNTPHARMA", "LUPIN",
    "AUROPHARMA", "ALKEM", "ZYDUSLIFE", "MANKIND", "IPCALAB", "GLENMARK",
    "LAURUSLABS", "BIOCON", "ABBOTINDIA", "PFIZER", "GLAXO", "NATCOPHARM",
    "AJANTPHARM", "GRANULES",
]
# Optional: commit a fresher ind_niftypharmalist.csv to the repo root to override the
# hardcoded list above without touching this script.
LOCAL_FALLBACK_LIST = os.environ.get("LOCAL_FALLBACK_LIST_PHARMA", "ind_niftypharmalist.csv")

INTERVAL = "1wk"
TF_LABEL = "Weekly"
LOOKBACK_BARS = 104
LOOKBACK_LABEL = "104 Weeks (~2Y)"

VOLUME_MULTIPLIER = 1.5
VOLUME_AVG_PERIOD = 10
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
NEAR_HIGH_PCT = 2.0
RISK_REWARD_RATIO = 2.0
ATR_PERIOD = 14
SL_ATR_MULTIPLIER = 1.5

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", r"D:\dhan\scanner\gfs")

NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Accept": "text/csv,application/csv,*/*",
}

# --------------------------------------------------------------------


def fetch_index_constituents():
    """Try live NSE fetch, then local CSV, then the hardcoded list as last resort."""
    session = requests.Session()
    session.headers.update(NSE_HEADERS)
    try:
        session.get("https://www.nseindia.com", timeout=10)
        resp = session.get(NSE_INDEX_URL, timeout=15)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        col = "Symbol" if "Symbol" in df.columns else df.columns[2]
        fetched = df[col].astype(str).str.strip().tolist()
        print(f"  [OK] Fetched {len(fetched)} symbols live from NSE.")
        return sorted(f"{s}.NS" for s in fetched if s and s.upper() != "SYMBOL")
    except Exception as e:
        print(f"  [!] Live NSE fetch failed ({e}). Trying local fallback: {LOCAL_FALLBACK_LIST}")
        if os.path.exists(LOCAL_FALLBACK_LIST):
            df = pd.read_csv(LOCAL_FALLBACK_LIST)
            col = "Symbol" if "Symbol" in df.columns else df.columns[2]
            fetched = df[col].astype(str).str.strip().tolist()
            print(f"  [OK] Loaded {len(fetched)} symbols from local fallback.")
            return sorted(f"{s}.NS" for s in fetched if s and s.upper() != "SYMBOL")
        else:
            print(f"  [!] No local fallback CSV found. Using hardcoded pharma list "
                  f"({len(HARDCODED_PHARMA_FALLBACK)} stocks) - verify this periodically "
                  f"against niftyindices.com as index constituents change over time.")
            return sorted(f"{s}.NS" for s in HARDCODED_PHARMA_FALLBACK)


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def recent_swing_low(df: pd.DataFrame, lookback: int = 10) -> float:
    return df["Low"].iloc[-lookback - 1:-1].min()


def analyze_stock(ticker: str) -> dict | None:
    try:
        years = max(3, (LOOKBACK_BARS // 52) + 1)
        period = f"{years}y"
        df = yf.download(ticker, period=period, interval=INTERVAL, progress=False, auto_adjust=True)
        if df.empty or len(df) < max(30, LOOKBACK_BARS):
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df["RSI"] = calculate_rsi(df["Close"], RSI_PERIOD)
        df["ATR"] = calculate_atr(df, ATR_PERIOD)
        df["VolAvg"] = df["Volume"].rolling(VOLUME_AVG_PERIOD).mean()

        latest = df.iloc[-1]
        period_high = df["High"].iloc[-LOOKBACK_BARS:].max()

        close = float(latest["Close"])
        high = float(latest["High"])
        low = float(latest["Low"])
        volume = float(latest["Volume"])
        vol_avg = float(latest["VolAvg"])
        rsi = float(latest["RSI"])
        atr = float(latest["ATR"])

        if pd.isna(vol_avg) or pd.isna(rsi) or pd.isna(atr):
            return None

        pct_from_high = ((period_high - close) / period_high) * 100
        if pct_from_high > NEAR_HIGH_PCT:
            return None

        vol_ratio = volume / vol_avg if vol_avg > 0 else 0
        if vol_ratio < VOLUME_MULTIPLIER:
            return None

        if rsi >= RSI_OVERBOUGHT:
            return None

        day_range = high - low
        close_position = (close - low) / day_range if day_range > 0 else 0
        if close_position < 0.5:
            return None

        swing_low = recent_swing_low(df, lookback=10)
        entry_low = round(close * 0.998, 2)
        entry_high = round(close * 1.005, 2)
        sl = round(min(swing_low, close - atr * SL_ATR_MULTIPLIER), 2)
        risk_per_share = close - sl
        target = round(close + RISK_REWARD_RATIO * risk_per_share, 2)

        return {
            "Ticker": ticker.replace(".NS", ""),
            "Close": round(close, 2),
            "Period High": round(period_high, 2),
            "% From High": round(pct_from_high, 2),
            "Vol Ratio": round(vol_ratio, 2),
            "RSI": round(rsi, 1),
            "Entry Zone": f"{entry_low} - {entry_high}",
            "SL": sl,
            "Target (2R)": target,
            "Risk/Share": round(risk_per_share, 2),
        }
    except Exception as e:
        print(f"  [skip] {ticker}: {e}")
        return None


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n{'='*70}\nBREAKOUT SCANNER (Automated) - {INDEX_NAME} | {TF_LABEL} | {LOOKBACK_LABEL}\n{'='*70}")

    tickers = fetch_index_constituents()
    if not tickers:
        print("No tickers fetched - exiting.")
        sys.exit(1)

    results = []
    total = len(tickers)
    for i, ticker in enumerate(tickers, 1):
        print(f"[{i}/{total}] Scanning {ticker}...")
        res = analyze_stock(ticker)
        if res:
            print(f"  MATCH: {ticker}")
            results.append(res)

    out_df = pd.DataFrame(results)

    date_str = dt.date.today().strftime("%Y%m%d")
    dated_path = os.path.join(OUTPUT_DIR, f"pharma_breakout_scan_weekly_{date_str}.csv")
    latest_path = os.path.join(OUTPUT_DIR, "pharma_breakout_scan_weekly_latest.csv")

    out_df.to_csv(dated_path, index=False)
    out_df.to_csv(latest_path, index=False)

    print(f"\nDone. {len(out_df)} matches out of {total} scanned.")
    print(f"Saved: {dated_path}")
    print(f"Saved: {latest_path}")


if __name__ == "__main__":
    main()
