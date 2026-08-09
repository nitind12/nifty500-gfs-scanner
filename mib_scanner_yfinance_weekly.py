"""
MIB (Mother Inside Bar) Scanner - yfinance based
--------------------------------------------------
Nitin's multi-stock scanner for spotting Mother Inside Bar setups across a
watchlist - both BULLISH continuation (long) and BEARISH breakdown (short) -
using free yfinance data.

WHAT IT DOES
1. Downloads 1H candles (and Daily candles for bias) for each symbol
2. Finds the most recent "Mother Bar" -> looks back for a large-range bar
   followed by one or more smaller bars contained within its range
   (classic inside-bar consolidation)
3. Checks whether price has broken out of that mother-bar range, up or down
4. Adds RSI(14) and MACD(12,26,9) confirmation
5. Adds a Daily bias filter (price vs 20 EMA on Daily)
6. LONG setup  = breakout UP + Daily bias BULLISH + RSI < 70 + MACD hist > 0
7. SHORT setup = breakout DOWN + Daily bias BEARISH + RSI > 30 + MACD hist < 0
8. Adds ATR(14)-based stop-loss distance and position sizing for every
   actionable setup (LONG or SHORT), sized to a fixed Rs amount per trade
9. Prints a full summary (with a reason for every non-actionable stock)
   plus separate LONG and SHORT actionable tables

USAGE
  pip install yfinance pandas --break-system-packages
  python mib_scanner_yfinance.py

You can edit the WATCHLIST below to add/remove stocks any time.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime

# Cloud/local output folder - defaults to your Windows path, overridden by
# GitHub Actions via OUTPUT_DIR env var (see workflow yml).
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", r"D:\dhan\scanner\gfs")

# ---------------------------------------------------------------------------
# CONFIG - edit this list to scan new stocks. NSE symbols need ".NS" suffix.
# ---------------------------------------------------------------------------
WATCHLIST = [
    "PAYTM.NS",
    "ICICIBANK.NS",
    "HDFCBANK.NS",
    "SBIN.NS",
    "INFY.NS",
    "ADANIPOWER.NS",
    "BAJAJ_AUTO.NS",
    "RELIANCE.NS"
]

RSI_PERIOD = 14
RSI_HARD_FILTER = 70          # no fresh LONG entry above this (your rule)
RSI_BEARISH_FLOOR = 30        # no fresh SHORT entry below this (avoid chasing oversold)
MOTHER_LOOKBACK = 15          # how many recent 1H bars to scan for a mother bar
MIN_INSIDE_BARS = 1           # min number of bars contained inside mother bar

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
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(close: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


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


def risk_sizing(entry_price: float, atr_val: float, direction: str, symbol_clean: str):
    """Returns (suggested_sl, risk_per_unit, max_loss_rs, suggested_qty,
    lot_size, one_lot_risk_rs, suggested_lots).
    IMPORTANT: sl_distance is based on the UNDERLYING's ATR (spot movement),
    not the option premium. Premium moves less than spot (see Delta), so
    one_lot_risk_rs here is a rough upper-bound reference for the underlying
    move - not the actual premium risk. Use it as a sanity check, and confirm
    real premium SL against the option's Delta before sizing a live trade."""
    sl_distance = round(atr_val * ATR_SL_MULTIPLIER, 2)
    if direction == "BREAKOUT_UP":
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
# MIB DETECTION
# ---------------------------------------------------------------------------
def find_mother_bar(df: pd.DataFrame, lookback: int, min_inside: int):
    """
    Scans the last `lookback` bars (excluding the very latest bar) for a
    mother bar: a candle whose range fully contains at least `min_inside`
    subsequent candles (their high/low stay inside mother's high/low).
    Returns dict with mother bar info, or None if not found.
    Picks the MOST RECENT valid mother bar (closest to current price).
    """
    recent = df.iloc[-(lookback + 5):-1]  # exclude latest forming/last bar
    n = len(recent)

    best = None
    for i in range(n - min_inside - 1, -1, -1):  # scan backwards, prefer recent
        mother = recent.iloc[i]
        m_high, m_low = mother["High"], mother["Low"]

        inside_count = 0
        j = i + 1
        while j < n:
            bar = recent.iloc[j]
            if bar["High"] <= m_high and bar["Low"] >= m_low:
                inside_count += 1
                j += 1
            else:
                break

        if inside_count >= min_inside:
            best = {
                "mother_idx": recent.index[i],
                "mother_high": m_high,
                "mother_low": m_low,
                "inside_bars": inside_count,
                "last_inside_idx": recent.index[j - 1] if j > i + 1 else recent.index[i],
            }
            break  # most recent qualifying mother bar found

    return best


def classify_breakout(df: pd.DataFrame, mother: dict):
    """
    Compares latest closed bar to mother bar range to see if price has
    broken out (up/down/none) and whether it's a clean body-close break.
    """
    latest = df.iloc[-1]
    m_high, m_low = mother["mother_high"], mother["mother_low"]

    if latest["Close"] > m_high:
        direction = "BREAKOUT_UP"
    elif latest["Close"] < m_low:
        direction = "BREAKOUT_DOWN"
    else:
        direction = "INSIDE_BOX"

    return direction, latest


# ---------------------------------------------------------------------------
# DAILY BIAS
# ---------------------------------------------------------------------------
def daily_bias(symbol: str):
    daily = yf.download(symbol, period="3y", interval="1wk", progress=False, auto_adjust=False)
    if daily.empty:
        return "UNKNOWN"
    daily = _clean_columns(daily)
    daily["EMA20"] = daily["Close"].ewm(span=20, adjust=False).mean()
    last_close = float(daily["Close"].iloc[-1])
    last_ema = float(daily["EMA20"].iloc[-1])
    return "BULLISH" if last_close > last_ema else "BEARISH"


# ---------------------------------------------------------------------------
# MAIN SCAN
# ---------------------------------------------------------------------------
def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance sometimes returns MultiIndex columns even for a single
    ticker (e.g. ('Close', 'PAYTM.NS')). Flatten to plain column names
    so df['Close'] etc. always returns a 1-D Series, not a DataFrame."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df


def scan_stock(symbol: str):
    df = yf.download(symbol, period="6mo", interval="1d", progress=False, auto_adjust=False)
    if df.empty or len(df) < MOTHER_LOOKBACK + 5:
        return {"symbol": symbol, "status": "NO_DATA"}

    df = _clean_columns(df)
    df = df.copy()

    last_close_price = float(df["Close"].iloc[-1])
    if not (PRICE_MIN <= last_close_price <= PRICE_MAX):
        return {
            "symbol": symbol,
            "status": "OUT_OF_PRICE_RANGE",
            "last_close": round(last_close_price, 2),
        }

    df["RSI"] = compute_rsi(df["Close"], RSI_PERIOD)
    df["MACD"], df["Signal"], df["Hist"] = compute_macd(df["Close"])
    df["ATR"] = compute_atr(df, ATR_PERIOD)

    mother = find_mother_bar(df, MOTHER_LOOKBACK, MIN_INSIDE_BARS)
    if mother is None:
        return {"symbol": symbol, "status": "NO_MIB_FOUND"}

    direction, latest = classify_breakout(df, mother)
    bias = daily_bias(symbol)

    rsi_val = round(float(latest["RSI"]), 2) if not pd.isna(latest["RSI"]) else None
    hist_val = round(float(latest["Hist"]), 2) if not pd.isna(latest["Hist"]) else None
    atr_val = round(float(latest["ATR"]), 2) if not pd.isna(latest["ATR"]) else None
    entry_ref = round(float(latest["Close"]), 2)

    long_actionable = (
        direction == "BREAKOUT_UP"
        and bias == "BULLISH"
        and rsi_val is not None
        and rsi_val < RSI_HARD_FILTER
        and hist_val is not None
        and hist_val > 0
    )

    short_actionable = (
        direction == "BREAKOUT_DOWN"
        and bias == "BEARISH"
        and rsi_val is not None
        and rsi_val > RSI_BEARISH_FLOOR
        and hist_val is not None
        and hist_val < 0
    )

    setup_type = "LONG" if long_actionable else ("SHORT" if short_actionable else "-")
    actionable = setup_type != "-"

    # explain WHY it isn't actionable, so a bullish/bearish-looking stock
    # that got filtered out shows you exactly which condition failed
    reason = "Actionable"
    if not actionable:
        if direction == "INSIDE_BOX":
            reason = "Still inside the mother bar box - no breakout yet"
        elif direction == "BREAKOUT_UP" and bias != "BULLISH":
            reason = "Broke box upward but Daily bias not bullish"
        elif direction == "BREAKOUT_UP" and rsi_val is not None and rsi_val >= RSI_HARD_FILTER:
            reason = f"Broke box upward but RSI {rsi_val} >= {RSI_HARD_FILTER} (overbought)"
        elif direction == "BREAKOUT_UP" and hist_val is not None and hist_val <= 0:
            reason = "Broke box upward but MACD histogram not positive"
        elif direction == "BREAKOUT_DOWN" and bias != "BEARISH":
            reason = "Broke box downward but Daily bias not bearish"
        elif direction == "BREAKOUT_DOWN" and rsi_val is not None and rsi_val <= RSI_BEARISH_FLOOR:
            reason = f"Broke box downward but RSI {rsi_val} <= {RSI_BEARISH_FLOOR} (oversold)"
        elif direction == "BREAKOUT_DOWN" and hist_val is not None and hist_val >= 0:
            reason = "Broke box downward but MACD histogram not negative"
        else:
            reason = "Conditions not aligned"

    result = {
        "symbol": symbol,
        "status": "OK",
        "mother_high": round(float(mother["mother_high"]), 2),
        "mother_low": round(float(mother["mother_low"]), 2),
        "inside_bars": mother["inside_bars"],
        "direction": direction,
        "last_close": entry_ref,
        "rsi": rsi_val,
        "macd_hist": hist_val,
        "atr": atr_val,
        "daily_bias": bias,
        "ACTIONABLE": "YES" if actionable else "no",
        "setup_type": setup_type,
        "reason": reason,
    }

    if actionable and atr_val is not None:
        sl, sl_dist, max_loss, qty, lot_size, lot_risk, sug_lots = risk_sizing(
            entry_ref, atr_val, direction, symbol.replace(".NS", "")
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


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\nMIB Scanner run at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    rows = []
    for sym in WATCHLIST:
        try:
            result = scan_stock(sym)
        except Exception as e:
            result = {"symbol": sym, "status": f"ERROR: {e}"}
        rows.append(result)

    out = pd.DataFrame(rows)
    csv_path = os.path.join(OUTPUT_DIR, f"mib_watchlist_scan_weekly_{datetime.now().strftime('%Y%m%d')}.csv")
    out.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_colwidth", 60)
    pd.set_option("display.float_format", lambda x: f"{x:.2f}")

    if "setup_type" not in out.columns:
        out["setup_type"] = "-"
    if "ACTIONABLE" not in out.columns:
        out["ACTIONABLE"] = "no"

    # --- Full summary: every stock with valid data, with a clear ACTIONABLE
    # yes/no column and a reason when it isn't, so you can see WHY a
    # bullish or bearish looking stock didn't qualify
    summary = out[out["status"] == "OK"].copy()
    summary_cols = ["symbol", "direction", "daily_bias", "rsi", "macd_hist", "ACTIONABLE", "setup_type", "reason"]
    summary_cols = [c for c in summary_cols if c in summary.columns]

    print("=" * 70)
    print("FULL SCAN SUMMARY - every stock checked, ACTIONABLE column shows the verdict")
    print("=" * 70)
    if not summary.empty:
        print(summary[summary_cols].sort_values("symbol").to_string(index=False))
        n_actionable = (summary["ACTIONABLE"] == "YES").sum()
        print(f"\n{n_actionable} of {len(summary)} scanned stocks are ACTIONABLE right now.")
    else:
        print("No stocks returned valid data this run.")

    risk_cols = ["symbol", "mother_high", "mother_low", "last_close", "rsi", "macd_hist",
                 "atr", "daily_bias", "suggested_sl", "sl_distance", "max_loss_rs",
                 "suggested_qty", "lot_size", "one_lot_risk_rs", "suggested_lots"]

    longs = out[out["setup_type"] == "LONG"]
    shorts = out[out["setup_type"] == "SHORT"]
    cols = [c for c in risk_cols if c in out.columns]

    print("\n" + "=" * 70)
    print("BULLISH CONTINUATION SETUPS - LONG")
    print("=" * 70)
    if not longs.empty:
        print(longs[cols].to_string(index=False))
    else:
        print("None right now. Patience.")

    print("\n" + "=" * 70)
    print("BEARISH BREAKDOWN SETUPS - SHORT")
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


if __name__ == "__main__":
    main()
