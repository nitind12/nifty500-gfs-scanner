"""
MIB Market Scanner - yfinance based
-------------------------------------
Scans a broad universe of NSE stocks (Nifty 50 by default) to automatically
find stocks whose 1H chart currently matches your MIB (Mother Inside Bar)
strategy - both BULLISH continuation (long) and BEARISH breakdown (short)
setups - so you don't have to manually check every stock's chart one by one.

WHAT IT DOES
1. Loops through every stock in UNIVERSE (edit this list any time)
2. For each stock: finds the real Mother Bar (tracing the full inside-bar
   chain backwards, not just the first/smallest qualifying pair) on 1H
3. Checks FRESH breakout direction only (i.e. the break happened on the
   latest closed candle, not several candles ago), RSI(14), MACD(12,26,9),
   and Daily bias (EMA20)
4. LONG setup   = fresh breakout UP + Daily bias BULLISH + RSI < 70 + MACD hist > 0
5. SHORT setup  = fresh breakout DOWN + Daily bias BEARISH + RSI > 30 + MACD hist < 0
   (RSI > 30 avoids chasing an already-oversold move)
6. Adds ATR(14)-based stop-loss distance and position sizing for every
   actionable setup (LONG or SHORT), sized to a fixed % of your capital
7. Ranks all actionable setups by strength so the best candidates float up
8. Prints two ranked reports (LONG table + SHORT table) + saves a CSV

FIX LOG (14 Aug 2026)
- find_mother_bar(): old version stopped at the FIRST (usually tiny, most
  recent) qualifying mother+inside-bar pair. It now walks backwards and
  extends the mother as far back as possible, as long as every bar in
  between stays fully inside that candle's High/Low range - so it finds
  the real, larger Mother candle, matching how you read it visually on
  the chart.
- classify_breakout(): old version only checked "is latest close beyond
  mother high/low" which stayed BREAKOUT_UP/DOWN forever after an old
  break, even if price was just drifting sideways many candles later. It
  now requires the PREVIOUS closed candle to still have been inside the
  mother's range, so only a FRESH breakout (happening on the newest
  candle) is flagged.

USAGE
  pip install yfinance pandas --break-system-packages
  python mib_market_scanner.py

TIP: yfinance rate-limits if you hit it too fast with too many symbols.
The script sleeps briefly between requests - if you scan the full F&O
list (180+ stocks) it may take a few minutes. That's normal.
"""

import time
import os
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

# Works both locally (set OUTPUT_DIR to your D:\dhan\... path via env var)
# and on GitHub Actions (defaults to a relative "output" folder in the
# repo, which the workflow can then pick up for the combined email step).
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "output")

# ---------------------------------------------------------------------------
# UNIVERSE - the stocks to scan. Edit / extend freely.
# Nifty 50 given by default. Add more NSE F&O names as you like (.NS suffix).
# ---------------------------------------------------------------------------
NIFTY50 = [
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK",
    "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV", "BPCL", "BHARTIARTL",
    "BRITANNIA", "CIPLA", "COALINDIA", "DIVISLAB", "DRREDDY",
    "EICHERMOT", "GRASIM", "HCLTECH", "HDFCBANK", "HDFCLIFE",
    "HEROMOTOCO", "HINDALCO", "HINDUNILVR", "ICICIBANK", "ITC",
    "INDUSINDBK", "INFY", "JSWSTEEL", "KOTAKBANK", "LT",
    "M&M", "MARUTI", "NTPC", "NESTLEIND", "ONGC",
    "POWERGRID", "RELIANCE", "SBILIFE", "SBIN", "SUNPHARMA",
    "TCS", "TATACONSUM", "TMPV", "TMCV", "TATASTEEL", "TECHM",
    "TITAN", "UPL", "ULTRACEMCO", "WIPRO",
]
NIFTY50 = sorted(set(NIFTY50))  # dedupe

# Add your own extras here (e.g. PAYTM which isn't in Nifty 50 yet)
# Note: TATAMOTORS demerged into TMPV (Passenger Vehicles) + TMCV (Commercial
# Vehicles) in Nov 2025, so both are scanned above instead of the old single symbol.
EXTRA_WATCHLIST = ["PAYTM", "ADANIPOWER", "MAZDOCK", "IRFC"]

UNIVERSE = [f"{sym}.NS" for sym in (NIFTY50 + EXTRA_WATCHLIST)]

RSI_PERIOD = 14
RSI_HARD_FILTER = 70          # no fresh LONG entry above this (your rule)
RSI_BEARISH_FLOOR = 30        # no fresh SHORT entry below this (avoid chasing oversold)
MOTHER_LOOKBACK = 15
MIN_INSIDE_BARS = 1
SLEEP_BETWEEN_CALLS = 0.6  # seconds, be polite to yfinance

# Only consider stocks whose latest close falls in this price band
PRICE_MIN = 200
PRICE_MAX = 3000

# --- ATR-based risk sizing config (edit these to match your account) ---
ATR_PERIOD = 14
ATR_SL_MULTIPLIER = 1.5       # SL distance = ATR x this (1.0 tight, 1.5 balanced, 2.0 wide/news-driven)
ACCOUNT_CAPITAL = 80000       # your trading capital in Rs - EDIT THIS
MAX_LOSS_PER_TRADE_RS = 2500  # fixed Rs amount you're willing to risk per trade
                               # (this is ~3.1% of capital - higher than the
                               # usual 1% because a smaller amount was giving
                               # SL distances too tight to survive normal noise)

# NSE F&O lot sizes - add symbols as you trade them (changes periodically,
# always double check on NSE/Sensibull before trusting this for a live trade)
LOT_SIZES = {
    "BAJFINANCE": 725,
}


# ---------------------------------------------------------------------------
# INDICATORS
# ---------------------------------------------------------------------------
def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_macd(close: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder's ATR (RMA-smoothed) - matches TradingView's default 'ATR 14 RMA'."""
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, min_periods=period).mean()
    return atr


def risk_sizing(entry_price: float, atr_val: float, setup_type: str, symbol_clean: str):
    """Returns (suggested_sl, sl_distance, max_loss_rs, suggested_qty,
    lot_size, one_lot_risk_rs, suggested_lots). Works for LONG or SHORT.
    IMPORTANT: sl_distance is based on the UNDERLYING's ATR (spot movement),
    not the option premium. Premium moves less than spot (see Delta), so
    one_lot_risk_rs here is a rough upper-bound reference for the underlying
    move - not the actual premium risk. Confirm real premium SL against the
    option's Delta before sizing a live trade."""
    sl_distance = round(atr_val * ATR_SL_MULTIPLIER, 2)
    if setup_type == "LONG":
        suggested_sl = round(entry_price - sl_distance, 2)
    else:
        suggested_sl = round(entry_price + sl_distance, 2)
    max_loss_rs = MAX_LOSS_PER_TRADE_RS
    suggested_qty = int(max_loss_rs // sl_distance) if sl_distance > 0 else 0

    lot_size = LOT_SIZES.get(symbol_clean)
    one_lot_risk_rs = None
    suggested_lots = None
    if lot_size:
        one_lot_risk_rs = round(sl_distance * lot_size, 2)
        suggested_lots = int(max_loss_rs // one_lot_risk_rs) if one_lot_risk_rs > 0 else 0

    return suggested_sl, sl_distance, max_loss_rs, suggested_qty, lot_size, one_lot_risk_rs, suggested_lots


# ---------------------------------------------------------------------------
# MIB DETECTION (FIXED - traces the real mother candle, checks fresh breakout only)
# ---------------------------------------------------------------------------
def find_mother_bar(df: pd.DataFrame, lookback: int, min_inside: int):
    """
    Correct MIB logic:
    - Look at the window of recent CLOSED bars (excluding the very latest bar,
      which is checked separately for a fresh breakout).
    - Anchor at the second-to-last closed bar and walk BACKWARDS: bar M can
      be the mother only if EVERY bar between M and the anchor (inclusive)
      sits fully inside M's High/Low range.
    - Keep extending backwards while containment holds. The furthest-back
      bar for which containment still holds is the TRUE mother - the real,
      larger candle that contains the whole unbroken inside-bar chain.
    - This fixes the old bug where the loop stopped at the FIRST (usually
      tiny, most recent) qualifying mother+inside-bar pair instead of
      tracing the chain back to the actual mother candle you'd circle on
      the chart.
    """
    recent = df.iloc[-(lookback + 5):-1]  # closed bars, excludes current forming bar
    n = len(recent)
    if n < min_inside + 1:
        return None

    anchor_idx = n - 2  # bar right before the newest closed bar
    if anchor_idx < 0:
        return None

    mother_idx = anchor_idx
    i = anchor_idx - 1
    while i >= 0:
        candidate = recent.iloc[i]
        c_high, c_low = candidate["High"], candidate["Low"]
        chain = recent.iloc[i + 1: anchor_idx + 1]
        if (chain["High"] <= c_high).all() and (chain["Low"] >= c_low).all():
            mother_idx = i
            i -= 1
        else:
            break

    inside_count = anchor_idx - mother_idx
    if inside_count < min_inside:
        return None

    mother = recent.iloc[mother_idx]
    return {
        "mother_high": mother["High"],
        "mother_low": mother["Low"],
        "inside_bars": inside_count,
    }


def classify_breakout(df: pd.DataFrame, mother: dict):
    """
    FRESH breakout only:
    - the PREVIOUS closed bar must still have been INSIDE the mother's range
    - the CURRENT (latest) bar's close must break outside that range
    This avoids flagging BREAKOUT_UP/DOWN on an old break that happened
    several candles ago, with price just drifting since - which was the
    bug causing false-positive "actionable" setups.
    """
    latest = df.iloc[-1]
    prev = df.iloc[-2]

    prev_was_inside = (
        prev["High"] <= mother["mother_high"] and prev["Low"] >= mother["mother_low"]
    )

    if not prev_was_inside:
        return "NOT_FRESH", latest

    if latest["Close"] > mother["mother_high"]:
        direction = "BREAKOUT_UP"
    elif latest["Close"] < mother["mother_low"]:
        direction = "BREAKOUT_DOWN"
    else:
        direction = "INSIDE_BOX"
    return direction, latest


def daily_bias(symbol: str):
    daily = yf.download(symbol, period="3mo", interval="1d", progress=False, auto_adjust=False)
    if daily.empty:
        return "UNKNOWN"
    daily = _clean_columns(daily)
    daily["EMA20"] = daily["Close"].ewm(span=20, adjust=False).mean()
    last_close = float(daily["Close"].iloc[-1])
    last_ema = float(daily["EMA20"].iloc[-1])
    return "BULLISH" if last_close > last_ema else "BEARISH"


# ---------------------------------------------------------------------------
# SCAN ONE STOCK
# ---------------------------------------------------------------------------
def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance sometimes returns MultiIndex columns even for a single
    ticker. Flatten so df['Close'] etc. always returns a 1-D Series."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df


def scan_stock(symbol: str):
    df = yf.download(symbol, period="10d", interval="1h", progress=False, auto_adjust=False)
    if df.empty or len(df) < MOTHER_LOOKBACK + 5:
        return {"symbol": symbol.replace(".NS", ""), "status": "NO_DATA"}

    df = _clean_columns(df)
    df = df.copy()

    last_close_price = float(df["Close"].iloc[-1])
    if not (PRICE_MIN <= last_close_price <= PRICE_MAX):
        return {
            "symbol": symbol.replace(".NS", ""),
            "status": "OUT_OF_PRICE_RANGE",
            "last_close": round(last_close_price, 2),
        }

    df["RSI"] = compute_rsi(df["Close"], RSI_PERIOD)
    df["MACD"], df["Signal"], df["Hist"] = compute_macd(df["Close"])
    df["ATR"] = compute_atr(df, ATR_PERIOD)

    mother = find_mother_bar(df, MOTHER_LOOKBACK, MIN_INSIDE_BARS)
    if mother is None:
        return {"symbol": symbol.replace(".NS", ""), "status": "NO_MIB_FOUND"}

    direction, latest = classify_breakout(df, mother)
    bias = daily_bias(symbol)

    rsi_val = round(float(latest["RSI"]), 2) if not pd.isna(latest["RSI"]) else None
    hist_val = round(float(latest["Hist"]), 2) if not pd.isna(latest["Hist"]) else None
    atr_val = round(float(latest["ATR"]), 2) if not pd.isna(latest["ATR"]) else None
    entry_ref = round(float(latest["Close"]), 2)

    # --- Bullish continuation (long) ---
    long_actionable = (
        direction == "BREAKOUT_UP"
        and bias == "BULLISH"
        and rsi_val is not None and rsi_val < RSI_HARD_FILTER
        and hist_val is not None and hist_val > 0
    )

    # --- Bearish continuation / breakdown (short) ---
    short_actionable = (
        direction == "BREAKOUT_DOWN"
        and bias == "BEARISH"
        and rsi_val is not None and rsi_val > RSI_BEARISH_FLOOR
        and hist_val is not None and hist_val < 0
    )

    setup_type = "LONG" if long_actionable else ("SHORT" if short_actionable else "-")

    # explain WHY it isn't actionable, so a bullish/bearish trend stock that
    # got filtered out shows you exactly which condition failed
    reason = "Actionable"
    if setup_type == "-":
        if direction == "NOT_FRESH":
            reason = "Break already happened earlier - not a fresh breakout this candle"
        elif direction == "INSIDE_BOX":
            reason = "Still inside the mother bar box - no breakout yet"
        elif direction == "BREAKOUT_UP" and bias != "BULLISH":
            reason = "Fresh breakout upward but Daily bias not bullish"
        elif direction == "BREAKOUT_UP" and rsi_val is not None and rsi_val >= RSI_HARD_FILTER:
            reason = f"Fresh breakout upward but RSI {rsi_val} >= {RSI_HARD_FILTER} (overbought)"
        elif direction == "BREAKOUT_UP" and hist_val is not None and hist_val <= 0:
            reason = "Fresh breakout upward but MACD histogram not positive"
        elif direction == "BREAKOUT_DOWN" and bias != "BEARISH":
            reason = "Fresh breakout downward but Daily bias not bearish"
        elif direction == "BREAKOUT_DOWN" and rsi_val is not None and rsi_val <= RSI_BEARISH_FLOOR:
            reason = f"Fresh breakout downward but RSI {rsi_val} <= {RSI_BEARISH_FLOOR} (oversold)"
        elif direction == "BREAKOUT_DOWN" and hist_val is not None and hist_val >= 0:
            reason = "Fresh breakout downward but MACD histogram not negative"
        else:
            reason = "Conditions not aligned"

    # strength score for ranking (works for either direction)
    score = None
    if rsi_val is not None and hist_val is not None:
        if long_actionable:
            score = round((RSI_HARD_FILTER - rsi_val) * 0.5 + hist_val, 2)
        elif short_actionable:
            score = round((rsi_val - RSI_BEARISH_FLOOR) * 0.5 + abs(hist_val), 2)

    result = {
        "symbol": symbol.replace(".NS", ""),
        "status": "OK",
        "mother_high": round(float(mother["mother_high"]), 2),
        "mother_low": round(float(mother["mother_low"]), 2),
        "last_close": entry_ref,
        "direction": direction,
        "rsi": rsi_val,
        "macd_hist": hist_val,
        "atr": atr_val,
        "daily_bias": bias,
        "ACTIONABLE": "YES" if setup_type != "-" else "no",
        "setup_type": setup_type,
        "reason": reason,
        "score": score,
    }

    if setup_type != "-" and atr_val is not None:
        sl, sl_dist, max_loss, qty, lot_size, lot_risk, sug_lots = risk_sizing(
            entry_ref, atr_val, setup_type, symbol.replace(".NS", "")
        )
        result.update({
            "suggested_sl": sl,
            "sl_distance": sl_dist,
            "max_loss_rs": max_loss,
            "suggested_qty": qty,
            "lot_size": lot_size,
            "one_lot_risk_rs": lot_risk,
            "suggested_lots": sug_lots,
        })

    return result


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_colwidth", 60)
    pd.set_option("display.float_format", lambda x: f"{x:.2f}")

    print(f"\nMIB Market Scanner run at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Scanning {len(UNIVERSE)} stocks... this may take a few minutes.\n")

    rows = []
    for idx, sym in enumerate(UNIVERSE, 1):
        try:
            result = scan_stock(sym)
        except Exception as e:
            result = {"symbol": sym.replace(".NS", ""), "status": f"ERROR: {e}"}
        rows.append(result)
        print(f"  [{idx}/{len(UNIVERSE)}] {sym} -> {result.get('status')}")
        time.sleep(SLEEP_BETWEEN_CALLS)

    out = pd.DataFrame(rows)
    csv_path = os.path.join(OUTPUT_DIR, f"mib_market_scan_{datetime.now().strftime('%Y%m%d')}.csv")
    out.to_csv(csv_path, index=False)

    if "setup_type" not in out.columns:
        out["setup_type"] = "-"
    if "ACTIONABLE" not in out.columns:
        out["ACTIONABLE"] = "no"

    # --- Full summary: every stock that had valid data, with a clear
    # ACTIONABLE yes/no column, so you can see WHY a bullish-trend stock
    # didn't qualify (e.g. still inside the box, or RSI/MACD not aligned yet)
    summary = out[out["status"] == "OK"].copy()
    summary_cols = ["symbol", "direction", "daily_bias", "rsi", "macd_hist", "ACTIONABLE", "setup_type", "reason"]
    summary_cols = [c for c in summary_cols if c in summary.columns]

    print("\n" + "=" * 70)
    print("FULL SCAN SUMMARY - every stock checked, ACTIONABLE column shows the verdict")
    print("=" * 70)
    if not summary.empty:
        print(summary[summary_cols].sort_values("symbol").to_string(index=False))
        n_actionable = (summary["ACTIONABLE"] == "YES").sum()
        print(f"\n{n_actionable} of {len(summary)} scanned stocks are ACTIONABLE right now.")
    else:
        print("No stocks returned valid data this run.")

    longs = out[out["setup_type"] == "LONG"].sort_values("score", ascending=False)
    shorts = out[out["setup_type"] == "SHORT"].sort_values("score", ascending=False)

    base_cols = ["symbol", "mother_high", "mother_low", "last_close", "rsi", "macd_hist", "atr", "daily_bias", "score"]
    risk_cols = ["suggested_sl", "sl_distance", "max_loss_rs", "suggested_qty", "lot_size", "one_lot_risk_rs", "suggested_lots"]
    cols = [c for c in base_cols + risk_cols if c in out.columns]

    print("\n" + "=" * 70)
    print("BULLISH CONTINUATION SETUPS - LONG (ranked by strength, best first)")
    print("=" * 70)
    if not longs.empty:
        print(longs[cols].to_string(index=False))
    else:
        print("None right now. Patience.")

    print("\n" + "=" * 70)
    print("BEARISH BREAKDOWN SETUPS - SHORT (ranked by strength, best first)")
    print("=" * 70)
    if not shorts.empty:
        print(shorts[cols].to_string(index=False))
    else:
        print("None right now. Patience.")

    if not longs.empty or not shorts.empty:
        print(f"\n(Risk sizing assumes capital=Rs{ACCOUNT_CAPITAL:,}, max risk=Rs{MAX_LOSS_PER_TRADE_RS:,}/trade, "
              f"SL = {ATR_SL_MULTIPLIER}x ATR. Edit these constants at the top of the script.)")
        print("(sl_distance/suggested_qty are based on the UNDERLYING's move, not option premium.")
        print(" suggested_lots uses LOT_SIZES dict - add your symbols there. Still only a reference:")
        print(" premium moves less than spot due to Delta, so confirm real premium risk before entry.)")

    print(f"\nFull results saved to: {csv_path}")


if __name__ == "__main__":
    main()
