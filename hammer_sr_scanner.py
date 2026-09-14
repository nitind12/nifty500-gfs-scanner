"""
hammer_sr_scanner.py
=====================
Scans a watchlist for a LIQUIDITY-SWEEP + REVERSAL-CANDLE combo:

  1. LIQUIDITY SWEEP OF SUPPORT + HAMMER
     -> price pierces below a known support level (stop-hunt / liquidity
        grab) but the SAME candle closes back above it, and the candle
        itself is shaped like a trustworthy hammer.
  2. LIQUIDITY SWEEP OF RESISTANCE + INVERTED HAMMER
     -> price pierces above a known resistance level but closes back
        below it, shaped like a trustworthy inverted hammer.

A hammer sitting "near" a level with no actual sweep is NOT enough on its
own anymore — it only fires when the level was genuinely swept (liquidity
taken) and then reclaimed inside that same candle. This mirrors the
Liquidity Sweep Rule already used in the MIB framework (Case 2: single
large/decisive candle sweep).

Checked on BOTH:
  - Daily timeframe
  - Hourly timeframe

Candle-quality filter ("long enough to trust"):
  A hammer/inverted-hammer is only flagged if its dominant wick is
  meaningfully long relative to (a) its own body and (b) the recent
  average candle range (ATR). Tiny dojis with a technically-correct
  wick ratio but no real size are rejected.

De-duplication:
  Only ONE signal per (Symbol, Timeframe, Pattern) is kept — the most
  recent qualifying candle — so you no longer get repeated/redundant
  rows for the same stock+timeframe (e.g. GMDCLTD.NS showing up 2-3
  times because several recent bars all sat near the same level).

Data source : yfinance (swap in your Dhan/broker feed if you prefer)
Author      : generated for Nitin Deepak's MIB scanner suite
Usage       : python hammer_sr_scanner.py
Designed to slot into the same GitHub Actions cron pattern as
mib_market_scanner.py / mib_scanner_yfinance.py.
"""

import os
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


TIMEFRAMES = {
    # label : (yfinance interval, yfinance period, lookback bars used to
    #          build support/resistance zones)
    "1D": {"interval": "1d", "period": "2y", "sr_lookback": 60},
    "1H": {"interval": "60m", "period": "60d", "sr_lookback": 80},
}

# --- Support / Resistance detection -----------------------------------
SR_SWING_ORDER = 3        # bars on each side to confirm a swing high/low pivot
SR_CLUSTER_PCT = 0.005    # merge pivot levels within 0.5% of each other

# --- Liquidity sweep rules -------------------------------------------------
# The wick must pierce THROUGH the level (not just sit near it), and the
# body must close back on the "right" side of it within the same candle.
# MAX_SWEEP_DEPTH_ATR caps how far beyond the level price is allowed to have
# gone — too deep a pierce means it's a genuine breakdown/breakout, not a
# stop-hunt sweep of that specific level.
MAX_SWEEP_DEPTH_ATR = 1.5
MIN_SWEEP_DEPTH_ATR = 0.05   # trivially small piercing (rounding noise) rejected

# --- Hammer / Inverted-Hammer shape rules --------------------------------
MIN_WICK_TO_BODY_RATIO = 2.5   # dominant wick must be >= 2.5x the body
MAX_OPP_WICK_TO_RANGE = 0.25   # opposite (small) wick <= 25% of total range
MAX_BODY_TO_RANGE = 0.35       # body itself <= 35% of total range
MIN_WICK_TO_ATR_RATIO = 0.60   # "long enough to trust": dominant wick must be
                                # at least 60% of the recent average candle range (ATR)
ATR_PERIOD = 14

# candles scanned for a fresh signal, from the most recent bar backwards
# (final output is still de-duplicated down to 1 per Symbol+Timeframe+Pattern)
RECENT_BARS_TO_CHECK = 3


# --------------------------------------------------------------------------
# DATA STRUCTURES
# --------------------------------------------------------------------------

@dataclass
class Signal:
    symbol: str
    timeframe: str
    pattern: str            # "Liquidity Sweep + Hammer @ Support" / "... Inverted Hammer @ Resistance"
    candle_time: pd.Timestamp
    close: float
    level: float
    level_type: str          # "Support" / "Resistance"
    wick_to_body: float
    wick_to_atr: float
    sweep_depth_atr: float
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
            "SweepDepthATR": round(self.sweep_depth_atr, 2),
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


def find_liquidity_sweep_level(extreme: float, close: float, levels: list,
                                direction: str, atr: float):
    """
    direction = 'support'    -> candle's LOW must pierce below a support
                                 level, then CLOSE back above it.
    direction = 'resistance' -> candle's HIGH must pierce above a
                                 resistance level, then CLOSE back below it.

    Returns (level, sweep_depth) for the cleanest valid sweep (smallest
    pierce depth, i.e. the level that was JUST swept, not blown through),
    or (None, None) if no level qualifies.
    """
    if not atr or atr <= 0:
        return None, None

    candidates = []
    for lvl in levels:
        if direction == "support":
            depth = lvl - extreme          # positive if low pierced below lvl
            reclaimed = close > lvl
        else:
            depth = extreme - lvl          # positive if high pierced above lvl
            reclaimed = close < lvl

        if reclaimed and depth > 0:
            depth_atr = depth / atr
            if MIN_SWEEP_DEPTH_ATR <= depth_atr <= MAX_SWEEP_DEPTH_ATR:
                candidates.append((lvl, depth))

    if not candidates:
        return None, None

    candidates.sort(key=lambda x: x[1])  # smallest (cleanest) pierce first
    return candidates[0]


def classify_candle(o, h, l, c):
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


def score_quality(wick_to_body: float, wick_to_atr: float, sweep_depth_atr: float) -> str:
    """STRONG needs a decisive candle AND a real (not trivial) sweep."""
    if wick_to_body >= 4 and wick_to_atr >= 1.0 and sweep_depth_atr >= 0.15:
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

        # -- Liquidity sweep of support + Hammer ---------------------------
        support_level, sweep_depth = find_liquidity_sweep_level(l, c, supports, "support", atr)
        if support_level is not None:
            ok, w2b, w2atr = is_trustworthy_hammer(o, h, l, c, atr)
            if ok:
                sweep_depth_atr = sweep_depth / atr
                sig = Signal(symbol, tf_label, "Liquidity Sweep + Hammer @ Support", ts, c,
                              support_level, "Support", w2b, w2atr, sweep_depth_atr)
                sig.quality = score_quality(w2b, w2atr, sweep_depth_atr)
                signals.append(sig)

        # -- Liquidity sweep of resistance + Inverted hammer ----------------
        resistance_level, sweep_depth = find_liquidity_sweep_level(h, c, resistances, "resistance", atr)
        if resistance_level is not None:
            ok, w2b, w2atr = is_trustworthy_inverted_hammer(o, h, l, c, atr)
            if ok:
                sweep_depth_atr = sweep_depth / atr
                sig = Signal(symbol, tf_label, "Liquidity Sweep + Inverted Hammer @ Resistance", ts, c,
                              resistance_level, "Resistance", w2b, w2atr, sweep_depth_atr)
                sig.quality = score_quality(w2b, w2atr, sweep_depth_atr)
                signals.append(sig)

    return signals


def dedupe_signals(signals: list) -> list:
    """
    Keep only ONE signal per (Symbol, Timeframe, Pattern) - the most recent
    candle. This is what removes redundant repeated rows for the same
    stock+timeframe (e.g. several recent bars all qualifying near the same
    level).
    """
    best = {}
    for sig in signals:
        key = (sig.symbol, sig.timeframe, sig.pattern)
        if key not in best or sig.candle_time > best[key].candle_time:
            best[key] = sig
    return list(best.values())


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

    all_signals = dedupe_signals(all_signals)

    if not all_signals:
        return pd.DataFrame(columns=[
            "Symbol", "Timeframe", "Pattern", "CandleTime", "Close",
            "Level", "LevelType", "WickToBody", "WickToATR", "SweepDepthATR", "Quality"
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
    print(f"Running Liquidity-Sweep + Hammer/Inverted-Hammer S/R scan @ {datetime.now()}")
    results = run_scan()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    dated_path = os.path.join(OUTPUT_DIR, f"hammer_sr_signals_{date_str}.csv")
    latest_path = os.path.join(OUTPUT_DIR, "hammer_sr_signals_latest.csv")

    if results.empty:
        print("No qualifying liquidity-sweep + hammer/inverted-hammer signals right now.")
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
# Already wired into:
#   - .github/workflows/daily_all_scans.yml   (runs as part of the combined
#     3:50 PM IST daily run)
#   - .github/workflows/hammer_sr_scan.yml     (standalone run, 4x/day
#     between 9:16 AM and 3:00 PM IST, weekdays)
