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
own — it only fires when the level was genuinely swept (liquidity taken)
and then reclaimed inside that same candle. This mirrors the Liquidity
Sweep Rule already used in the MIB framework (Case 2: single large/
decisive candle sweep).

LEVELS come from two sources, merged together:
  - AUTO   : confirmed swing highs/lows detected in the price data
  - MANUAL : levels you've marked yourself by eyeballing the chart
             (see manual_sr_levels.csv) - e.g. a level that was tested
             multiple times across months, or a consolidation zone edge
             that pure swing-pivot detection might under-weight.
Each signal's Level/LevelSource columns tell you which kind matched.

Checked on BOTH:
  - Daily timeframe
  - Hourly timeframe

Candle-quality filter ("long enough to trust"):
  A hammer/inverted-hammer is only flagged if its dominant wick is
  meaningfully long relative to (a) its own body and (b) the recent
  average candle range (ATR). Tiny dojis with a technically-correct
  wick ratio but no real size are rejected.

De-duplication:
  Only ONE signal per (Symbol, Timeframe, Pattern) is kept - the most
  recent qualifying candle - so you don't get repeated/redundant rows
  for the same stock+timeframe.

Data source : yfinance (swap in your Dhan/broker feed if you prefer)
Author      : generated for Nitin Deepak's MIB scanner suite
Usage       : python hammer_sr_scanner.py
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
# CONFIG
# --------------------------------------------------------------------------

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "output")
NIFTY500_LIST_URL = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
LOCAL_FALLBACK_LIST = os.environ.get("LOCAL_FALLBACK_LIST", "ind_nifty500list.csv")
REQUEST_PAUSE_SEC = 0.2  # small pause between symbol fetches, be gentle on yfinance

# Manual S/R levels you've marked yourself on the chart. CSV columns:
#   Symbol,LevelType,Price,Note
# LevelType is "Support", "Resistance", or "R/S" (adds it to BOTH lists -
# use this for a zone that has acted as both, like a consolidation range
# edge). Note is optional, for your own reference only.
MANUAL_LEVELS_FILE = os.environ.get("MANUAL_LEVELS_FILE", "manual_sr_levels.csv")


def get_watchlist():
    """
    Fetch the NIFTY 500 constituent list from NSE archives, falling back to
    the local CSV (ind_nifty500list.csv) when the live fetch fails.
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


def load_manual_levels() -> dict:
    """
    Loads manual_sr_levels.csv (Symbol,LevelType,Price[,Note]) into:
        { "PCBL.NS": {"support": [226.5, 320.0], "resistance": [305.0, 335.0, 374.0]}, ... }
    Symbols in the CSV without a ".NS" suffix get it appended automatically.
    Missing file -> empty dict, no error (manual levels are optional).
    """
    levels = {}
    if not os.path.exists(MANUAL_LEVELS_FILE):
        return levels

    try:
        df = pd.read_csv(MANUAL_LEVELS_FILE)
        df.columns = [c.strip() for c in df.columns]
        for _, row in df.iterrows():
            symbol = str(row["Symbol"]).strip()
            if not symbol.upper().endswith(".NS"):
                symbol += ".NS"
            level_type = str(row["LevelType"]).strip().lower()
            price = float(row["Price"])

            entry = levels.setdefault(symbol, {"support": [], "resistance": []})
            if level_type in ("support", "r/s", "rs", "support/resistance"):
                entry["support"].append(price)
            if level_type in ("resistance", "r/s", "rs", "support/resistance"):
                entry["resistance"].append(price)

        print(f"Loaded manual levels for {len(levels)} symbol(s) from {MANUAL_LEVELS_FILE}.")
    except Exception as e:
        print(f"[WARN] Could not parse {MANUAL_LEVELS_FILE}: {e}")

    return levels


TIMEFRAMES = {
    "1D": {"interval": "1d", "period": "2y", "sr_lookback": 60},
    "1H": {"interval": "60m", "period": "60d", "sr_lookback": 80},
}

# --- Support / Resistance detection -----------------------------------
SR_SWING_ORDER = 3        # bars on each side to confirm a swing high/low pivot
SR_CLUSTER_PCT = 0.005    # merge pivot levels within 0.5% of each other

# --- Liquidity sweep rules -------------------------------------------------
MAX_SWEEP_DEPTH_ATR = 1.5
MIN_SWEEP_DEPTH_ATR = 0.05

# --- Hammer / Inverted-Hammer shape rules --------------------------------
MIN_WICK_TO_BODY_RATIO = 2.5
MAX_OPP_WICK_TO_RANGE = 0.25
MAX_BODY_TO_RANGE = 0.35
MIN_WICK_TO_ATR_RATIO = 0.60
ATR_PERIOD = 14

RECENT_BARS_TO_CHECK = 3


# --------------------------------------------------------------------------
# DATA STRUCTURES
# --------------------------------------------------------------------------

@dataclass
class Signal:
    symbol: str
    timeframe: str
    pattern: str
    candle_time: pd.Timestamp
    close: float
    level: float
    level_type: str
    level_source: str        # "Auto" (swing pivot) or "Manual" (your marked level)
    wick_to_body: float
    wick_to_atr: float
    sweep_depth_atr: float
    quality: str = field(default="")

    def as_row(self):
        return {
            "Symbol": self.symbol,
            "Timeframe": self.timeframe,
            "Pattern": self.pattern,
            "CandleTime": self.candle_time,
            "Close": round(self.close, 2),
            "Level": round(self.level, 2),
            "LevelType": self.level_type,
            "LevelSource": self.level_source,
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


def find_raw_swing_levels(df: pd.DataFrame, lookback: int, order: int = SR_SWING_ORDER):
    """Returns (raw_support_prices, raw_resistance_prices) - unclustered."""
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

    return lows, highs


def cluster_levels(levels_with_source: list, pct: float = SR_CLUSTER_PCT) -> list:
    """
    levels_with_source: list of (price, source) tuples, source in
    {'Auto', 'Manual'}. Merges levels within pct of each other. If a
    cluster contains any Manual level, the merged level is tagged Manual
    (your eyeballed level takes priority over a nearby auto pivot).
    Returns sorted list of dicts: [{'price':..., 'source':...}, ...]
    """
    items = sorted(levels_with_source, key=lambda x: x[0])
    clustered = []
    for price, source in items:
        if clustered and abs(price - clustered[-1]["price"]) / clustered[-1]["price"] <= pct:
            merged_price = (clustered[-1]["price"] + price) / 2
            merged_source = "Manual" if (clustered[-1]["source"] == "Manual" or source == "Manual") else "Auto"
            clustered[-1] = {"price": merged_price, "source": merged_source}
        else:
            clustered.append({"price": price, "source": source})
    return clustered


def build_levels(df: pd.DataFrame, lookback: int, manual_for_symbol: dict) -> tuple:
    """Combines auto swing levels + manual levels into final clustered lists."""
    raw_supports, raw_resistances = find_raw_swing_levels(df, lookback)
    manual_supports = manual_for_symbol.get("support", [])
    manual_resistances = manual_for_symbol.get("resistance", [])

    support_candidates = [(p, "Auto") for p in raw_supports] + [(p, "Manual") for p in manual_supports]
    resistance_candidates = [(p, "Auto") for p in raw_resistances] + [(p, "Manual") for p in manual_resistances]

    supports = cluster_levels(support_candidates)
    resistances = cluster_levels(resistance_candidates)
    return supports, resistances


def find_liquidity_sweep_level(extreme: float, close: float, levels: list,
                                direction: str, atr: float):
    """
    levels: list of {'price':..., 'source':...} dicts (from build_levels).
    direction = 'support'    -> candle's LOW must pierce below a support
                                 level, then CLOSE back above it.
    direction = 'resistance' -> candle's HIGH must pierce above a
                                 resistance level, then CLOSE back below it.
    Returns (level_price, sweep_depth, level_source) for the cleanest valid
    sweep, or (None, None, None).
    """
    if not atr or atr <= 0:
        return None, None, None

    candidates = []
    for lvl in levels:
        price, source = lvl["price"], lvl["source"]
        if direction == "support":
            depth = price - extreme
            reclaimed = close > price
        else:
            depth = extreme - price
            reclaimed = close < price

        if reclaimed and depth > 0:
            depth_atr = depth / atr
            if MIN_SWEEP_DEPTH_ATR <= depth_atr <= MAX_SWEEP_DEPTH_ATR:
                candidates.append((price, depth, source))

    if not candidates:
        return None, None, None

    candidates.sort(key=lambda x: x[1])
    return candidates[0]


def classify_candle(o, h, l, c):
    body = abs(c - o)
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    total_range = h - l
    return dict(body=body, upper_wick=upper_wick, lower_wick=lower_wick, range=total_range)


def is_trustworthy_hammer(o, h, l, c, atr):
    m = classify_candle(o, h, l, c)
    if m["range"] <= 0 or m["body"] <= 0:
        return False, 0, 0
    body_ok = m["body"] <= MAX_BODY_TO_RANGE * m["range"]
    lower_wick_dominant = m["lower_wick"] >= MIN_WICK_TO_BODY_RATIO * m["body"]
    upper_wick_small = m["upper_wick"] <= MAX_OPP_WICK_TO_RANGE * m["range"]
    long_enough_vs_atr = atr and atr > 0 and m["lower_wick"] >= MIN_WICK_TO_ATR_RATIO * atr
    wick_to_body = m["lower_wick"] / m["body"] if m["body"] else np.inf
    wick_to_atr = m["lower_wick"] / atr if atr else 0
    passed = body_ok and lower_wick_dominant and upper_wick_small and long_enough_vs_atr
    return passed, wick_to_body, wick_to_atr


def is_trustworthy_inverted_hammer(o, h, l, c, atr):
    m = classify_candle(o, h, l, c)
    if m["range"] <= 0 or m["body"] <= 0:
        return False, 0, 0
    body_ok = m["body"] <= MAX_BODY_TO_RANGE * m["range"]
    upper_wick_dominant = m["upper_wick"] >= MIN_WICK_TO_BODY_RATIO * m["body"]
    lower_wick_small = m["lower_wick"] <= MAX_OPP_WICK_TO_RANGE * m["range"]
    long_enough_vs_atr = atr and atr > 0 and m["upper_wick"] >= MIN_WICK_TO_ATR_RATIO * atr
    wick_to_body = m["upper_wick"] / m["body"] if m["body"] else np.inf
    wick_to_atr = m["upper_wick"] / atr if atr else 0
    passed = body_ok and upper_wick_dominant and lower_wick_small and long_enough_vs_atr
    return passed, wick_to_body, wick_to_atr


def score_quality(wick_to_body: float, wick_to_atr: float, sweep_depth_atr: float) -> str:
    if wick_to_body >= 4 and wick_to_atr >= 1.0 and sweep_depth_atr >= 0.15:
        return "STRONG"
    return "OK"


# --------------------------------------------------------------------------
# SCAN LOGIC
# --------------------------------------------------------------------------

def scan_symbol_timeframe(symbol: str, tf_label: str, tf_cfg: dict, manual_levels: dict) -> list:
    signals = []
    df = fetch_ohlc(symbol, tf_cfg["interval"], tf_cfg["period"])
    if df.empty or len(df) < tf_cfg["sr_lookback"] + ATR_PERIOD:
        return signals

    df["ATR"] = compute_atr(df)
    manual_for_symbol = manual_levels.get(symbol, {})
    supports, resistances = build_levels(df, tf_cfg["sr_lookback"], manual_for_symbol)

    recent = df.tail(RECENT_BARS_TO_CHECK)

    for ts, row in recent.iterrows():
        o, h, l, c = row["Open"], row["High"], row["Low"], row["Close"]
        atr = row["ATR"]
        if pd.isna(atr):
            continue

        support_level, sweep_depth, level_source = find_liquidity_sweep_level(l, c, supports, "support", atr)
        if support_level is not None:
            ok, w2b, w2atr = is_trustworthy_hammer(o, h, l, c, atr)
            if ok:
                sweep_depth_atr = sweep_depth / atr
                sig = Signal(symbol, tf_label, "Liquidity Sweep + Hammer @ Support", ts, c,
                              support_level, "Support", level_source, w2b, w2atr, sweep_depth_atr)
                sig.quality = score_quality(w2b, w2atr, sweep_depth_atr)
                signals.append(sig)

        resistance_level, sweep_depth, level_source = find_liquidity_sweep_level(h, c, resistances, "resistance", atr)
        if resistance_level is not None:
            ok, w2b, w2atr = is_trustworthy_inverted_hammer(o, h, l, c, atr)
            if ok:
                sweep_depth_atr = sweep_depth / atr
                sig = Signal(symbol, tf_label, "Liquidity Sweep + Inverted Hammer @ Resistance", ts, c,
                              resistance_level, "Resistance", level_source, w2b, w2atr, sweep_depth_atr)
                sig.quality = score_quality(w2b, w2atr, sweep_depth_atr)
                signals.append(sig)

    return signals


def dedupe_signals(signals: list) -> list:
    best = {}
    for sig in signals:
        key = (sig.symbol, sig.timeframe, sig.pattern)
        if key not in best or sig.candle_time > best[key].candle_time:
            best[key] = sig
    return list(best.values())


def run_scan(watchlist=None) -> pd.DataFrame:
    watchlist = watchlist or get_watchlist()
    manual_levels = load_manual_levels()
    all_signals = []

    for symbol in watchlist:
        for tf_label, tf_cfg in TIMEFRAMES.items():
            try:
                sigs = scan_symbol_timeframe(symbol, tf_label, tf_cfg, manual_levels)
                all_signals.extend(sigs)
            except Exception as e:
                print(f"[WARN] {symbol} {tf_label}: {e}")
        time.sleep(REQUEST_PAUSE_SEC)

    all_signals = dedupe_signals(all_signals)

    if not all_signals:
        return pd.DataFrame(columns=[
            "Symbol", "Timeframe", "Pattern", "CandleTime", "Close",
            "Level", "LevelType", "LevelSource", "WickToBody", "WickToATR",
            "SweepDepthATR", "Quality"
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

    results.to_csv(dated_path, index=False)
    results.to_csv(latest_path, index=False)
    print(f"\nSaved: {dated_path}\nSaved: {latest_path}")

# --------------------------------------------------------------------------
# GitHub Actions integration
# --------------------------------------------------------------------------
# Already wired into:
#   - .github/workflows/daily_all_scans.yml   (combined 3:50 PM IST run)
#   - .github/workflows/hammer_sr_scan.yml     (standalone, 4x/day
#     between 9:16 AM and 3:00 PM IST, weekdays)
# Manual levels: edit manual_sr_levels.csv at the repo root - no code
# change needed, just add/edit rows.
