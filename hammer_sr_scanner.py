"""
hammer_sr_scanner.py
=====================
Scans a watchlist for:
  1. HAMMER candle forming AT SUPPORT   -> potential bullish reversal
  2. INVERTED HAMMER candle forming AT RESISTANCE -> potential bearish reversal

Checked on BOTH:
  - Daily timeframe
  - Hourly timeframe

Candle-quality filter ("long enough to trust"):
  A hammer/inverted-hammer is only flagged if its dominant wick is
  meaningfully long relative to (a) its own body and (b) the recent
  average candle range (ATR). Tiny dojis with a technically-correct
  wick ratio but no real size are rejected.

Data source : yfinance (swap in your Dhan/broker feed if you prefer)
Author      : generated for Nitin Deepak's MIB scanner suite
Usage       : python hammer_sr_scanner.py
Designed to slot into the same GitHub Actions cron pattern as
mib_market_scanner.py / mib_scanner_yfinance.py (see bottom of file).
"""

import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# CONFIG — tune these to match how strict you want the scan to be
# --------------------------------------------------------------------------

# Same conventions as the rest of the repo (mib_market_scanner.py,
# nifty500_rsi_screener.py, etc.) so this plugs straight into the daily
# GitHub Actions workflow.
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "output")
NIFTY500_LIST_URL = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
LOCAL_FALLBACK_LIST = os.environ.get("LOCAL_FALLBACK_LIST", "ind_nifty500list.csv")
REQUEST_PAUSE_SEC = 0.2  # small pause between symbol fetches, be gentle on yfinance


def get_watchlist():
    """
    Fetch the NIFTY 500 constituent list from NSE archives (same source used
    by the other scanners in this repo), falling back to the local CSV
    (ind_nifty500list.csv) when the live fetch fails.
    """
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
        print("No fallback list found - falling back to a small default watchlist.")
        return [
            "RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "TCS.NS",
            "SBIN.NS", "AXISBANK.NS", "KOTAKBANK.NS", "TATASTEEL.NS", "ITC.NS",
        ]


WATCHLIST = None  # resolved lazily in run_scan() via get_watchlist()

TIMEFRAMES = {
    # label : (yfinance interval, yfinance period, lookback bars used to
    #          build support/resistance zones)
    "1D": {"interval": "1d", "period": "2y", "sr_lookback": 60},
    "1H": {"interval": "60m", "period": "60d", "sr_lookback": 80},
}

# --- Support / Resistance detection -----------------------------------
SR_SWING_ORDER = 3        # bars on each side to confirm a swing high/low pivot
SR_CLUSTER_PCT = 0.005    # merge pivot levels within 0.5% of each other
SR_PROXIMITY_PCT = 0.006  # candle must be within 0.6% of a level to "be at" it

# --- Hammer / Inverted-Hammer shape rules --------------------------------
MIN_WICK_TO_BODY_RATIO = 2.5   # dominant wick must be >= 2.5x the body
MAX_OPP_WICK_TO_RANGE = 0.25   # opposite (small) wick <= 25% of total range
MAX_BODY_TO_RANGE = 0.35       # body itself <= 35% of total range
MIN_WICK_TO_ATR_RATIO = 0.60   # "long enough to trust": dominant wick must be
                                # at least 60% of the recent average candle range (ATR)
ATR_PERIOD = 14

# candles scanned for a fresh signal, from the most recent bar backwards
RECENT_BARS_TO_CHECK = 3


# --------------------------------------------------------------------------
# DATA STRUCTURES
# --------------------------------------------------------------------------

@dataclass
class Signal:
    symbol: str
    timeframe: str
    pattern: str            # "Hammer @ Support" / "Inverted Hammer @ Resistance"
    candle_time: pd.Timestamp
    close: float
    level: float
    level_type: str          # "Support" / "Resistance"
    wick_to_body: float
    wick_to_atr: float
    quality: str = field(default="")  # STRONG / OK, set after scoring

    def as_row(self):
        return {
            "Symbol": self.symbol,
            "Timeframe": self.timeframe,
            "Pattern": self.pattern,
            "CandleTime": self.candle_time,
            "Close": round(self.close, 2),
            "Level": round(self.level, 2),
            "LevelType": self.level_type,
            "WickToBody": round(self.wick_to_body, 2),
            "WickToATR": round(self.wick_to_atr, 2),
            "Quality": self.quality,
        }


# --------------------------------------------------------------------------
# CORE HELPERS
# --------------------------------------------------------------------------

def fetch_ohlc(symbol: str, interval: str, period: str) -> pd.DataFrame:
    df = yf.download(symbol, interval=interval, period=period,
                      progress=False, auto_adjust=False)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.title)
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    return df


def compute_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def find_swing_levels(df: pd.DataFrame, lookback: int, order: int = SR_SWING_ORDER):
    """
    Returns (support_levels, resistance_levels) as sorted lists of price
    levels, built from confirmed swing lows/highs over the lookback window
    and clustered together when they sit within SR_CLUSTER_PCT of each other.
    """
    window = df.tail(lookback)
    lows, highs = [], []

    values_low = window["Low"].values
    values_high = window["High"].values
    n = len(window)

    for i in range(order, n - order):
        seg_low = values_low[i - order: i + order + 1]
        seg_high = values_high[i - order: i + order + 1]
        if values_low[i] == seg_low.min():
            lows.append(values_low[i])
        if values_high[i] == seg_high.max():
            highs.append(values_high[i])

    def cluster(levels):
        levels = sorted(levels)
        clustered = []
        for lvl in levels:
            if clustered and abs(lvl - clustered[-1]) / clustered[-1] <= SR_CLUSTER_PCT:
                clustered[-1] = (clustered[-1] + lvl) / 2  # merge
            else:
                clustered.append(lvl)
        return clustered

    return cluster(lows), cluster(highs)


def near_level(price: float, levels, pct: float = SR_PROXIMITY_PCT):
    """Return the nearest level if price is within pct of it, else None."""
    for lvl in levels:
        if abs(price - lvl) / lvl <= pct:
            return lvl
    return None


def classify_candle(o, h, l, c):
    """
    Returns dict with body, upper_wick, lower_wick, total_range for one candle.
    """
    body = abs(c - o)
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    total_range = h - l
    return dict(body=body, upper_wick=upper_wick, lower_wick=lower_wick,
                range=total_range)


def is_trustworthy_hammer(o, h, l, c, atr):
    """
    Classic hammer: long lower wick, small upper wick, small body near the
    top of the range. Rejects weak/undersized candles using ATR.
    """
    m = classify_candle(o, h, l, c)
    if m["range"] <= 0 or m["body"] <= 0:
        return False, 0, 0

    body_ok = m["body"] <= MAX_BODY_TO_RANGE * m["range"]
    lower_wick_dominant = m["lower_wick"] >= MIN_WICK_TO_BODY_RATIO * m["body"]
    upper_wick_small = m["upper_wick"] <= MAX_OPP_WICK_TO_RANGE * m["range"]
    long_enough_vs_atr = (atr and atr > 0 and
                           m["lower_wick"] >= MIN_WICK_TO_ATR_RATIO * atr)

    wick_to_body = m["lower_wick"] / m["body"] if m["body"] else np.inf
    wick_to_atr = m["lower_wick"] / atr if atr else 0

    passed = body_ok and lower_wick_dominant and upper_wick_small and long_enough_vs_atr
    return passed, wick_to_body, wick_to_atr


def is_trustworthy_inverted_hammer(o, h, l, c, atr):
    """
    Inverted hammer: long upper wick, small lower wick, small body near the
    bottom of the range. Rejects weak/undersized candles using ATR.
    """
    m = classify_candle(o, h, l, c)
    if m["range"] <= 0 or m["body"] <= 0:
        return False, 0, 0

    body_ok = m["body"] <= MAX_BODY_TO_RANGE * m["range"]
    upper_wick_dominant = m["upper_wick"] >= MIN_WICK_TO_BODY_RATIO * m["body"]
    lower_wick_small = m["lower_wick"] <= MAX_OPP_WICK_TO_RANGE * m["range"]
    long_enough_vs_atr = (atr and atr > 0 and
                           m["upper_wick"] >= MIN_WICK_TO_ATR_RATIO * atr)

    wick_to_body = m["upper_wick"] / m["body"] if m["body"] else np.inf
    wick_to_atr = m["upper_wick"] / atr if atr else 0

    passed = body_ok and upper_wick_dominant and lower_wick_small and long_enough_vs_atr
    return passed, wick_to_body, wick_to_atr


def score_quality(wick_to_body: float, wick_to_atr: float) -> str:
    """Simple STRONG / OK tag so you can eyeball priority at a glance."""
    if wick_to_body >= 4 and wick_to_atr >= 1.0:
        return "STRONG"
    return "OK"


# --------------------------------------------------------------------------
# SCAN LOGIC
# --------------------------------------------------------------------------

def scan_symbol_timeframe(symbol: str, tf_label: str, tf_cfg: dict) -> list:
    signals = []
    df = fetch_ohlc(symbol, tf_cfg["interval"], tf_cfg["period"])
    if df.empty or len(df) < tf_cfg["sr_lookback"] + ATR_PERIOD:
        return signals

    df["ATR"] = compute_atr(df)
    supports, resistances = find_swing_levels(df, tf_cfg["sr_lookback"])

    recent = df.tail(RECENT_BARS_TO_CHECK)

    for ts, row in recent.iterrows():
        o, h, l, c = row["Open"], row["High"], row["Low"], row["Close"]
        atr = row["ATR"]
        if pd.isna(atr):
            continue

        # -- Hammer at support --------------------------------------------
        support_hit = near_level(l, supports) or near_level(c, supports)
        if support_hit:
            ok, w2b, w2atr = is_trustworthy_hammer(o, h, l, c, atr)
            if ok:
                sig = Signal(symbol, tf_label, "Hammer @ Support", ts, c,
                              support_hit, "Support", w2b, w2atr)
                sig.quality = score_quality(w2b, w2atr)
                signals.append(sig)

        # -- Inverted hammer at resistance ---------------------------------
        resistance_hit = near_level(h, resistances) or near_level(c, resistances)
        if resistance_hit:
            ok, w2b, w2atr = is_trustworthy_inverted_hammer(o, h, l, c, atr)
            if ok:
                sig = Signal(symbol, tf_label, "Inverted Hammer @ Resistance", ts, c,
                              resistance_hit, "Resistance", w2b, w2atr)
                sig.quality = score_quality(w2b, w2atr)
                signals.append(sig)

    return signals


def run_scan(watchlist=None) -> pd.DataFrame:
    watchlist = watchlist or get_watchlist()
    all_signals = []

    for symbol in watchlist:
        for tf_label, tf_cfg in TIMEFRAMES.items():
            try:
                sigs = scan_symbol_timeframe(symbol, tf_label, tf_cfg)
                all_signals.extend(sigs)
            except Exception as e:
                print(f"[WARN] {symbol} {tf_label}: {e}")
        time.sleep(REQUEST_PAUSE_SEC)

    if not all_signals:
        return pd.DataFrame(columns=[
            "Symbol", "Timeframe", "Pattern", "CandleTime", "Close",
            "Level", "LevelType", "WickToBody", "WickToATR", "Quality"
        ])

    result = pd.DataFrame([s.as_row() for s in all_signals])
    result = result.sort_values(
        ["Quality", "Timeframe", "Symbol"], ascending=[False, True, True]
    ).reset_index(drop=True)
    return result


# --------------------------------------------------------------------------
# ENTRY POINT
# --------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Running Hammer/Inverted-Hammer S/R scan @ {datetime.now()}")
    results = run_scan()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    dated_path = os.path.join(OUTPUT_DIR, f"hammer_sr_signals_{date_str}.csv")
    latest_path = os.path.join(OUTPUT_DIR, "hammer_sr_signals_latest.csv")

    if results.empty:
        print("No qualifying hammer/inverted-hammer signals at S/R right now.")
    else:
        print(results.to_string(index=False))

    # Always write both files (even if empty) so send_combined_email.py's
    # has_real_data() check can correctly skip it when there's nothing to send.
    results.to_csv(dated_path, index=False)
    results.to_csv(latest_path, index=False)
    print(f"\nSaved: {dated_path}\nSaved: {latest_path}")

# --------------------------------------------------------------------------
# GitHub Actions integration
# --------------------------------------------------------------------------
# Add this step to .github/workflows/daily_all_scans.yml, before the
# "Send combined email" step, so its *_latest.csv gets picked up
# automatically by send_combined_email.py:
#
#   - name: Run Hammer/Inverted-Hammer @ S-R scanner (Nifty 500, Daily+Hourly)
#     continue-on-error: true
#     env:
#       OUTPUT_DIR: output
#       LOCAL_FALLBACK_LIST: ind_nifty500list.csv
#     run: python hammer_sr_scanner.py
