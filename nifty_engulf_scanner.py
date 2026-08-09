"""
================================================================================
NIFTY 50 / NIFTY 500 - BULLISH/BEARISH ENGULFING @ SUPPORT/RESISTANCE SCANNER
================================================================================
Har baar run karne par yeh script:
  0. Sabse pehle poochega - Nifty50 scan karna hai ya Nifty500 (ya aap
     INDEX_CHOICE variable set karke prompt skip bhi kar sakte ho)
  1. Chuni gayi index list ke saare stocks fetch karta hai (yfinance se)
  2. Pivot-based Support/Resistance levels detect karta hai (aapke MIB
     framework ke tarah - swing high/low based zones)
  3. Latest closed candle par Bullish AUR Bearish, dono Engulfing patterns
     check karta hai:
       - Bullish Engulfing -> Support ke paas -> LONG bias signal
       - Bearish Engulfing -> Resistance ke paas -> SHORT bias signal
     (Off-location engulfing bhi report hota hai, e.g. Bullish @ Resistance
     - yeh weak/warning signal maana jaata hai, isliye kam score milta hai)
  4. Confluence ke liye RSI, Volume confirmation, aur zone type bhi dikhata
     hai - taaki aap khud judge kar sako ki trade lena hai ya nahi.

USAGE:
    python nifty_engulf_scanner.py
    -> Script chalte hi poochega: "1. Nifty50   2. Nifty500" - number daal do.
    -> Agar prompt skip karna ho (e.g. Task Scheduler / cron se run kar rahe ho),
       to niche INDEX_CHOICE variable ko "NIFTY50" ya "NIFTY500" set kar do -
       tab script poochega nahi, seedha wahi list use karegi.

Requirements:
    pip install yfinance pandas numpy requests --break-system-packages

Notes:
  - Data source: Yahoo Finance (yfinance) - free, no API key needed.
  - Nifty500 list NSE ki official CSV se live fetch hoti hai (internet chahiye
    us waqt). Agar fetch fail ho jaye (internet issue / NSE block), script
    automatically apne saved fallback list pe switch ho jaati hai.
  - Timeframe: Daily (1D) candles by default. Aap TIMEFRAME variable badal
    kar '1h' bhi try kar sakte ho (yfinance intraday data sirf last 60 din
    ke liye deta hai).
  - Yeh scanner sirf SIGNAL deta hai. Entry/SL/Target apne MIB rules ke
    hisaab se khud decide karo (candle-close confirmation, OI check, etc.)
================================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import os
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", r"D:\dhan\scanner\gfs")

# --------------------------------------------------------------------------
# CONFIG - yaha se settings adjust karo
# --------------------------------------------------------------------------
# INDEX_CHOICE: "NIFTY50", "NIFTY500", ya None (None = script run hote hi
# terminal pe poochegi ki kaunsi list scan karni hai)
# Hardcoded to NIFTY500 for unattended/cloud runs (skips the prompt).
INDEX_CHOICE = os.environ.get("INDEX_CHOICE", "NIFTY500")

# TF_CHOICE: neeche TIMEFRAME_OPTIONS dict me se koi bhi key (e.g. "1", "2"...),
# ya us option ka naam bhi chalega (e.g. "1h", "daily"). None = terminal pe
# poochegi ki kaunsa timeframe scan karna hai.
# Hardcoded to "7" (Daily) for unattended/cloud runs.
TF_CHOICE = os.environ.get("TF_CHOICE", "7")

# Har option: yfinance interval + kitna purana data (period) + agar yfinance
# me direct interval available nahi hai (jaise 4h/2h), to 'resample' me base
# interval bataya gaya hai jise baad me us TF me resample kiya jaata hai.
TIMEFRAME_OPTIONS = {
    "1": {"label": "5 Minutes",  "interval": "5m",  "period": "60d",  "resample": None},
    "2": {"label": "15 Minutes", "interval": "15m", "period": "60d",  "resample": None},
    "3": {"label": "30 Minutes", "interval": "30m", "period": "60d",  "resample": None},
    "4": {"label": "1 Hour",     "interval": "1h",  "period": "730d", "resample": None},
    "5": {"label": "2 Hour",     "interval": "1h",  "period": "730d", "resample": "2h"},
    "6": {"label": "4 Hour",     "interval": "1h",  "period": "730d", "resample": "4h"},
    "7": {"label": "Daily",      "interval": "1d",  "period": "2y",   "resample": None},
    "8": {"label": "Weekly",     "interval": "1wk", "period": "5y",   "resample": None},
}

# In dono ko resolve_timeframe() run time pe overwrite kar deta hai based on
# choice - default values yaha sirf fallback ke liye hain.
TIMEFRAME = "1h"
LOOKBACK_PERIOD = "730d"
RESAMPLE_RULE = None

PIVOT_LEFT_BARS = 5       # pivot detect karne ke liye left side bars
PIVOT_RIGHT_BARS = 5      # pivot detect karne ke liye right side bars
SR_TOLERANCE_ATR_MULT = 0.5   # S/R zone ke "paas" maana jaye - ATR ka kitna multiple
MIN_VOLUME_RATIO = 1.0    # engulfing candle ka volume, avg volume se kam se kam kitna guna ho (1.0 = average se zyada)
RSI_PERIOD = 14
MAX_RSI_FOR_LONG = 65      # bahut zyada overbought signals avoid karne ke liye (Bullish setups)
MIN_RSI_FOR_SHORT = 35     # bahut zyada oversold signals avoid karne ke liye (Bearish setups)

# --------------------------------------------------------------------------
# NIFTY 50 STOCK LIST (NSE symbols, yfinance format me .NS suffix)
# --------------------------------------------------------------------------
NIFTY50_SYMBOLS = [
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK",
    "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV", "BEL", "BHARTIARTL",
    "CIPLA", "COALINDIA", "DRREDDY", "EICHERMOT", "ETERNAL",
    "GRASIM", "HCLTECH", "HDFCBANK", "HDFCLIFE", "HEROMOTOCO",
    "HINDALCO", "HINDUNILVR", "ICICIBANK", "ITC", "INDUSINDBK",
    "INFY", "JSWSTEEL", "JIOFIN", "KOTAKBANK", "LT",
    "M&M", "MARUTI", "NTPC", "NESTLEIND", "ONGC",
    "POWERGRID", "RELIANCE", "SBILIFE", "SHRIRAMFIN", "SBIN",
    "SUNPHARMA", "TCS", "TATACONSUM", "TMPV", "TMCV", "TATASTEEL",
    "TECHM", "TITAN", "TRENT", "ULTRACEMCO", "WIPRO",
]


# --------------------------------------------------------------------------
# NIFTY 500 - live fetch from NSE (fallback list agar fetch fail ho jaye)
# --------------------------------------------------------------------------
NSE_NIFTY500_CSV_URL = "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv"

# Fallback: agar NSE se live list fetch na ho paye (internet/NSE block etc.)
# Reuses the same ind_nifty500list.csv already committed in the repo (shared
# with the RSI and breakout scanners), instead of a stale hardcoded list.
LOCAL_NIFTY500_FALLBACK_FILE = os.environ.get("LOCAL_FALLBACK_LIST", "ind_nifty500list.csv")


def load_local_nifty500_fallback():
    if os.path.exists(LOCAL_NIFTY500_FALLBACK_FILE):
        df = pd.read_csv(LOCAL_NIFTY500_FALLBACK_FILE)
        col = "Symbol" if "Symbol" in df.columns else df.columns[2]
        symbols = df[col].dropna().astype(str).str.strip().tolist()
        print(f"  [OK] Local fallback file se {len(symbols)} symbols mile.\n")
        return symbols
    print(f"  [WARNING] Local fallback file bhi nahi mili ({LOCAL_NIFTY500_FALLBACK_FILE}). "
          f"Nifty50 list use ho rahi hai as last resort.\n")
    return NIFTY50_SYMBOLS


def fetch_nifty500_symbols():
    """
    NSE ki official website se live Nifty500 constituent list fetch karta hai.
    Return: list of NSE symbols (bina .NS suffix ke), ya fallback list agar fail ho.
    """
    import requests
    import io

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "text/csv,application/csv,*/*",
    }
    try:
        session = requests.Session()
        # NSE ko pehle homepage hit karna padta hai cookies ke liye, warna CSV request block ho jaata hai
        session.get("https://www.nseindia.com", headers=headers, timeout=10)
        resp = session.get(NSE_NIFTY500_CSV_URL, headers=headers, timeout=15)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        symbol_col = "Symbol" if "Symbol" in df.columns else df.columns[2]
        symbols = df[symbol_col].dropna().astype(str).str.strip().tolist()
        if len(symbols) >= 400:  # sanity check - poori list aayi ya nahi
            print(f"  [OK] Nifty500 list NSE se live fetch hui - {len(symbols)} symbols mile.\n")
            return symbols
        raise ValueError("Fetched list bahut chhoti hai, kuch gadbad lagti hai.")
    except Exception as e:
        print(f"  [WARNING] Nifty500 live fetch fail ho gaya ({e}).")
        return load_local_nifty500_fallback()


def resolve_symbol_list():
    """
    INDEX_CHOICE config ke hisaab se, ya terminal prompt se poochkar,
    decide karta hai ki Nifty50 scan karni hai ya Nifty500. Symbol list
    return karta hai.
    """
    choice = INDEX_CHOICE

    if choice is None:
        print("Kaunsi index scan karni hai?")
        print("  1. Nifty 50")
        print("  2. Nifty 500")
        while True:
            ans = input("Apna choice daalo (1 ya 2): ").strip()
            if ans in ("1", "nifty50", "NIFTY50", "50"):
                choice = "NIFTY50"
                break
            elif ans in ("2", "nifty500", "NIFTY500", "500"):
                choice = "NIFTY500"
                break
            else:
                print("  Galat input - sirf 1 ya 2 daalo.\n")

    choice = choice.upper().replace(" ", "")
    if choice in ("NIFTY500", "500"):
        print("\nNifty500 list fetch ho rahi hai NSE se, thoda wait karo...")
        return fetch_nifty500_symbols(), "NIFTY 500"

    return NIFTY50_SYMBOLS, "NIFTY 50"


def resolve_timeframe():
    """
    TF_CHOICE config ke hisaab se, ya terminal prompt se poochkar, decide
    karta hai ki kaunsa timeframe scan karna hai. Returns:
        (interval, period, resample_rule, label)
    """
    choice = TF_CHOICE

    if choice is None:
        print("Kaunsa timeframe scan karna hai?")
        for key in sorted(TIMEFRAME_OPTIONS.keys()):
            opt = TIMEFRAME_OPTIONS[key]
            print(f"  {key}. {opt['label']}")
        while True:
            ans = input(f"Apna choice daalo (1-{len(TIMEFRAME_OPTIONS)}): ").strip()
            if ans in TIMEFRAME_OPTIONS:
                choice = ans
                break
            # naam se bhi match try karo (e.g. "1h", "daily", "weekly")
            matched = None
            for key, opt in TIMEFRAME_OPTIONS.items():
                if ans.lower().replace(" ", "") in opt["label"].lower().replace(" ", ""):
                    matched = key
                    break
            if matched:
                choice = matched
                break
            print(f"  Galat input - 1 se {len(TIMEFRAME_OPTIONS)} tak koi number daalo.\n")
    else:
        choice = str(choice)
        if choice not in TIMEFRAME_OPTIONS:
            # naam se match try karo agar config me seedha "1h", "daily" etc. likha ho
            matched = None
            for key, opt in TIMEFRAME_OPTIONS.items():
                if choice.lower().replace(" ", "") in opt["label"].lower().replace(" ", ""):
                    matched = key
                    break
            choice = matched if matched else "4"  # fallback: 1 Hour

    opt = TIMEFRAME_OPTIONS[choice]
    return opt["interval"], opt["period"], opt["resample"], opt["label"]


def fetch_data(symbol, period, interval, resample_rule=None):
    """
    Yahoo Finance se OHLCV data fetch karta hai.
    Agar resample_rule diya gaya ho (e.g. '2h', '4h') - jo yfinance directly
    support nahi karta - to base interval (usually '1h') fetch karke us TF
    me resample kar diya jaata hai (Open=first, High=max, Low=min,
    Close=last, Volume=sum).
    """
    ticker = symbol + ".NS"
    try:
        df = yf.download(ticker, period=period, interval=interval,
                          progress=False, auto_adjust=True)
        if df is None or df.empty:
            return None
        # yfinance kabhi kabhi MultiIndex columns deta hai - flatten kar dete hain
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()

        if resample_rule:
            df = df.resample(resample_rule, label="left").agg({
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            }).dropna()

        if len(df) < (PIVOT_LEFT_BARS + PIVOT_RIGHT_BARS + 20):
            return None
        return df
    except Exception:
        return None


def calc_atr(df, period=14):
    """ATR (Wilder's RMA) - aapke MIB scanner me jis tarah use hota hai."""
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    return atr


def calc_rsi(df, period=14):
    """Standard RSI calculation."""
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def find_pivot_levels(df, left_bars, right_bars):
    """
    Pivot High / Pivot Low detect karta hai (swing based S/R).
    Ek bar pivot low hoga agar uske left aur right ke 'n' bars se woh
    sabse neeche ho. Same logic pivot high ke liye ulta.
    """
    highs = df["High"].values
    lows = df["Low"].values
    n = len(df)

    pivot_highs = []  # (index, price)
    pivot_lows = []

    for i in range(left_bars, n - right_bars):
        window_high = highs[i - left_bars: i + right_bars + 1]
        window_low = lows[i - left_bars: i + right_bars + 1]

        if highs[i] == window_high.max():
            pivot_highs.append((i, highs[i]))
        if lows[i] == window_low.min():
            pivot_lows.append((i, lows[i]))

    return pivot_highs, pivot_lows


def get_nearby_sr_level(current_price, pivot_highs, pivot_lows, tolerance):
    """
    Current price ke tolerance range ke andar koi resistance ya support
    level hai kya - check karta hai. Sabse paas wala level return karta hai.
    """
    candidates = []

    for _, price in pivot_highs:
        if abs(current_price - price) <= tolerance:
            candidates.append(("Resistance", price, abs(current_price - price)))

    for _, price in pivot_lows:
        if abs(current_price - price) <= tolerance:
            candidates.append(("Support", price, abs(current_price - price)))

    if not candidates:
        return None, None

    candidates.sort(key=lambda x: x[2])
    zone_type, level, _ = candidates[0]
    return zone_type, level


def is_bullish_engulfing(df, i):
    """
    Bullish Engulfing check:
      - Pichli candle bearish (red) honi chahiye
      - Current candle bullish (green) honi chahiye
      - Current candle ka body, pichli candle ke body ko poori tarah
        engulf/cover karta ho (open <= prev_close, close >= prev_open)
    """
    if i < 1:
        return False

    prev_open, prev_close = df["Open"].iloc[i - 1], df["Close"].iloc[i - 1]
    curr_open, curr_close = df["Open"].iloc[i], df["Close"].iloc[i]

    prev_bearish = prev_close < prev_open
    curr_bullish = curr_close > curr_open
    engulfs = (curr_open <= prev_close) and (curr_close >= prev_open)

    return prev_bearish and curr_bullish and engulfs


def is_bearish_engulfing(df, i):
    """
    Bearish Engulfing check:
      - Pichli candle bullish (green) honi chahiye
      - Current candle bearish (red) honi chahiye
      - Current candle ka body, pichli candle ke body ko poori tarah
        engulf karta ho (open >= prev_close, close <= prev_open)
    """
    if i < 1:
        return False

    prev_open, prev_close = df["Open"].iloc[i - 1], df["Close"].iloc[i - 1]
    curr_open, curr_close = df["Open"].iloc[i], df["Close"].iloc[i]

    prev_bullish = prev_close > prev_open
    curr_bearish = curr_close < curr_open
    engulfs = (curr_open >= prev_close) and (curr_close <= prev_open)

    return prev_bullish and curr_bearish and engulfs


def scan_stock(symbol):
    """Ek stock ko fully scan karta hai aur result dictionary return karta hai (ya None)."""
    df = fetch_data(symbol, LOOKBACK_PERIOD, TIMEFRAME, RESAMPLE_RULE)
    if df is None:
        return None

    df["ATR"] = calc_atr(df, 14)
    df["RSI"] = calc_rsi(df, RSI_PERIOD)
    df["AvgVolume20"] = df["Volume"].rolling(20).mean()

    pivot_highs, pivot_lows = find_pivot_levels(df, PIVOT_LEFT_BARS, PIVOT_RIGHT_BARS)

    # Sirf latest CLOSED candle check karte hain (last row)
    last_idx = len(df) - 1
    if last_idx < PIVOT_LEFT_BARS + PIVOT_RIGHT_BARS:
        return None

    bullish = is_bullish_engulfing(df, last_idx)
    bearish = is_bearish_engulfing(df, last_idx)

    if not bullish and not bearish:
        return None

    pattern = "Bullish Engulfing" if bullish else "Bearish Engulfing"
    bias = "LONG" if bullish else "SHORT"

    current_price = df["Close"].iloc[last_idx]
    atr_val = df["ATR"].iloc[last_idx]
    tolerance = atr_val * SR_TOLERANCE_ATR_MULT

    # Pivot list se aakhri candle exclude karte hain (khud se compare na ho)
    ph_filtered = [(idx, p) for idx, p in pivot_highs if idx < last_idx]
    pl_filtered = [(idx, p) for idx, p in pivot_lows if idx < last_idx]

    zone_type, level = get_nearby_sr_level(current_price, ph_filtered, pl_filtered, tolerance)
    if zone_type is None:
        return None  # pattern bana lekin S/R ke paas nahi - skip

    curr_volume = df["Volume"].iloc[last_idx]
    avg_volume = df["AvgVolume20"].iloc[last_idx]
    volume_ratio = curr_volume / avg_volume if avg_volume > 0 else 0
    rsi_val = df["RSI"].iloc[last_idx]

    volume_confirmed = volume_ratio >= MIN_VOLUME_RATIO

    # "Right location" = Bullish@Support ya Bearish@Resistance (textbook confluence)
    right_location = (bullish and zone_type == "Support") or (bearish and zone_type == "Resistance")

    if bullish:
        rsi_ok = rsi_val <= MAX_RSI_FOR_LONG
    else:
        rsi_ok = rsi_val >= MIN_RSI_FOR_SHORT

    # Simple confidence scoring (out of 6)
    score = 0
    if right_location:
        score += 2      # pattern sahi zone (support/resistance) ke saath match kar raha hai
    if volume_confirmed:
        score += 2
    if rsi_ok:
        score += 1
    if abs(current_price - level) <= (atr_val * 0.25):
        score += 1      # bahut hi close hai level ke - extra weight

    return {
        "Symbol": symbol,
        "Date": df.index[last_idx].strftime("%Y-%m-%d"),
        "Pattern": pattern,
        "Bias": bias,
        "Close": round(current_price, 2),
        "Zone": zone_type,
        "Level": round(level, 2),
        "Distance": round(abs(current_price - level), 2),
        "RightLocation": "Yes" if right_location else "No (check)",
        "RSI": round(rsi_val, 1),
        "VolumeRatio": round(volume_ratio, 2),
        "VolumeConfirmed": "Yes" if volume_confirmed else "No",
        "Score(/6)": score,
    }


def main():
    global TIMEFRAME, LOOKBACK_PERIOD, RESAMPLE_RULE

    print("=" * 90)
    print(" BULLISH / BEARISH ENGULFING @ SUPPORT / RESISTANCE SCANNER")
    print("=" * 90)

    symbols, index_label = resolve_symbol_list()
    print()
    TIMEFRAME, LOOKBACK_PERIOD, RESAMPLE_RULE, tf_label = resolve_timeframe()

    print("=" * 90)
    print(f" Index     : {index_label}  ({len(symbols)} stocks)")
    print(f" Run Time  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Timeframe : {tf_label}"
          + (f"  (base interval: {TIMEFRAME}, resampled)" if RESAMPLE_RULE else f"  (interval: {TIMEFRAME})")
          + f"   |   Lookback: {LOOKBACK_PERIOD}")
    print("=" * 90)

    results = []
    total = len(symbols)

    for i, symbol in enumerate(symbols, 1):
        print(f"  Scanning [{i}/{total}]: {symbol:<15}", end="\r")
        res = scan_stock(symbol)
        if res:
            results.append(res)

    print(" " * 60, end="\r")  # clear the progress line

    if not results:
        print("\nAaj koi bhi Nifty50 stock me Support/Resistance ke paas")
        print("Bullish ya Bearish Engulfing pattern nahi mila. Fir se try karo")
        print("agle candle-close ke baad.\n")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        empty_path = os.path.join(OUTPUT_DIR, f"engulfing_scan_{datetime.now().strftime('%Y%m%d')}.csv")
        with open(empty_path, "w") as f:
            f.write("No bullish or bearish engulfing setups found today.\n")
        print(f"Saved: {empty_path}")
        return

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by=["Bias", "Score(/6)"], ascending=[True, False]).reset_index(drop=True)

    bullish_df = results_df[results_df["Bias"] == "LONG"].reset_index(drop=True)
    bearish_df = results_df[results_df["Bias"] == "SHORT"].reset_index(drop=True)

    long_count = len(bullish_df)
    short_count = len(bearish_df)

    print(f"\n{len(results_df)} SETUP(S) MILE  ->  LONG: {long_count}   |   SHORT: {short_count}\n")

    print("-" * 90)
    print(f" BULLISH ENGULFING SETUPS  (LONG bias)  -  {long_count} mila/mile")
    print("-" * 90)
    if long_count > 0:
        print(bullish_df.to_string(index=False))
    else:
        print(" Koi Bullish Engulfing setup nahi mila is baar.")

    print("\n" + "-" * 90)
    print(f" BEARISH ENGULFING SETUPS  (SHORT bias)  -  {short_count} mila/mile")
    print("-" * 90)
    if short_count > 0:
        print(bearish_df.to_string(index=False))
    else:
        print(" Koi Bearish Engulfing setup nahi mila is baar.")

    # CSV me bhi Bullish aur Bearish ko alag-alag rows/section me save karte hain,
    # taaki journal me dekhte waqt dono clearly separate dikhein (blank row +
    # ek label row beech me daal rahe hain taaki Excel me bhi visually clear rahe)
    script_dir = OUTPUT_DIR
    os.makedirs(script_dir, exist_ok=True)
    out_path = os.path.join(script_dir, f"engulfing_scan_{datetime.now().strftime('%Y%m%d')}.csv")

    with open(out_path, "w", newline="") as f:
        f.write(f"BULLISH ENGULFING SETUPS (LONG bias) - {long_count} found\n")
        if long_count > 0:
            bullish_df.to_csv(f, index=False)
        else:
            f.write("No bullish setups found\n")
        f.write("\n")
        f.write(f"BEARISH ENGULFING SETUPS (SHORT bias) - {short_count} found\n")
        if short_count > 0:
            bearish_df.to_csv(f, index=False)
        else:
            f.write("No bearish setups found\n")

    print(f"\nResult CSV save ho gayi (Bullish/Bearish alag section me): {out_path}")

    print("\n" + "=" * 90)
    print(" NOTE: Yeh sirf pattern-detection signal hai. Entry lene se pehle:")
    print("   1. Candle CLOSE confirm karo (agar intraday timeframe use kar rahe ho)")
    print("   2. OI / Option chain se confluence check karo (jahan available ho)")
    print("   3. 'RightLocation = No' wale setups ko extra caution se dekho -")
    print("      matlab pattern bana hai lekin textbook zone match nahi (e.g.")
    print("      Bullish Engulfing @ Resistance) - yeh weaker signal hota hai")
    print("   4. Apne MIB rules (RSI filter, breakeven @1:1, trailing SL) follow karo")
    print("   5. Score(/6) sirf reference ke liye hai - apna judgement zaroor lagao")
    print("=" * 90)


if __name__ == "__main__":
    main()
