#!/usr/bin/env python3
"""
NIFTY hourly scanner for Supertrend + EMA(20)

Signal rules (evaluated on the latest completed hourly candle, IST):
- Bullish  = Close > Supertrend AND Close > EMA20
- Bearish  = Close < Supertrend AND Close < EMA20
Alert triggers only on a NEW flip into bull/bear.

Outputs a JSON summary and (optionally) sends a Telegram alert with:
- Side (BULLISH/BEARISH)
- Close / Supertrend / EMA20
- Suggested weekly option legs (ATM sell + OTM hedge)
- Next NIFTY weekly expiry date (IST), correctly handling:
    • Weekly expiry: every TUESDAY (default)
    • Monthly expiry: LAST TUESDAY of the month
"""
from __future__ import annotations

import os, sys, json, math, traceback, calendar
from datetime import datetime, timedelta, timezone, time as dtime, date
from zoneinfo import ZoneInfo

import pandas as pd
import numpy as np
import requests
import yfinance as yf

from indicators import ema, supertrend

# ---------- Config ----------
TICKER            = os.getenv("NIFTY_TICKER", "^NSEI")   # Yahoo NIFTY50 index
INTERVAL          = "60m"
LOOKBACK_DAYS     = int(os.getenv("LOOKBACK_DAYS", "60"))

EMA_PERIOD        = int(os.getenv("EMA_PERIOD", "20"))
ST_ATR_PERIOD     = int(os.getenv("ST_ATR_PERIOD", "7"))
ST_MULTIPLIER     = float(os.getenv("ST_MULTIPLIER", "3.5"))

OTM_GAP_POINTS    = int(os.getenv("OTM_GAP_POINTS", "100"))   # hedge gap suggestion
ATM_STEP_POINTS   = int(os.getenv("ATM_STEP_POINTS", "50"))    # NIFTY=50, BANKNIFTY=100

# New: expiry weekday (default Tuesday). Accepts "TUE"/"Tuesday"/"1" etc.
EXPIRY_WEEKDAY    = os.getenv("EXPIRY_WEEKDAY", "TUE")
LOG_LEVEL         = os.getenv("LOG_LEVEL", "INFO")

# Telegram (optional)
TG_TOKEN          = os.getenv("TELEGRAM_BOT_TOKEN", "")
TG_CHAT_ID        = os.getenv("TELEGRAM_CHAT_ID", "")
SEND_TELEGRAM     = bool(TG_TOKEN and TG_CHAT_ID)

IST = ZoneInfo("Asia/Kolkata")

def log(msg: str):
    prefix = f"[{datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S %Z')}] " if LOG_LEVEL.upper() == "DEBUG" else ""
    print(prefix + str(msg), flush=True)

# ---------- Helpers ----------
def _weekday_to_int(w: str | int) -> int:
    """Map weekday inputs to Python weekday ints (Mon=0..Sun=6)."""
    if isinstance(w, int):
        return max(0, min(6, w))
    w = str(w).strip().lower()
    mapping = {
        "mon": 0, "monday": 0, "0": 0,
        "tue": 1, "tuesday": 1, "1": 1,
        "wed": 2, "wednesday": 2, "2": 2,
        "thu": 3, "thursday": 3, "3": 3,
        "fri": 4, "friday": 4, "4": 4,
        "sat": 5, "saturday": 5, "5": 5,
        "sun": 6, "sunday": 6, "6": 6,
    }
    return mapping.get(w, 1)  # default Tuesday

EXPIRY_WD_INT = _weekday_to_int(EXPIRY_WEEKDAY)  # default 1 (Tuesday)

def round_to_step(x: float, step: int) -> int:
    return int(round(float(x) / float(step)) * step)

def last_weekday_of_month(y: int, m: int, weekday_int: int) -> date:
    """Return the date of the last given weekday in (y, m). weekday_int: Mon=0..Sun=6."""
    last_dom = calendar.monthrange(y, m)[1]
    d = date(y, m, last_dom)
    # step backward to the requested weekday
    offset = (d.weekday() - weekday_int) % 7
    return d - timedelta(days=offset)

def is_monthly_expiry(d: date, weekday_int: int) -> bool:
    """True if d is the last 'weekday_int' of its month."""
    return d == last_weekday_of_month(d.year, d.month, weekday_int)

def next_weekly_expiry_ist(now_ist: datetime) -> tuple[str, str]:
    """
    Return (expiry_date_str, expiry_type_str) where expiry_type_str ∈ {"Weekly", "Monthly"}.
    - Weekly expiry = EXPIRY_WD_INT (default Tuesday)
    - Monthly expiry = last EXPIRY_WD_INT of the month
    If it's expiry day and AFTER 15:30, roll to next week.
    """
    dow = now_ist.weekday()  # Mon=0..Sun=6
    days_ahead = (EXPIRY_WD_INT - dow) % 7
    candidate = (now_ist + timedelta(days=days_ahead)).date()

    # If today is expiry weekday and after market close, push to next week
    if dow == EXPIRY_WD_INT and now_ist.time() > dtime(15, 30):
        candidate = candidate + timedelta(days=7)

    exp_type = "Monthly" if is_monthly_expiry(candidate, EXPIRY_WD_INT) else "Weekly"
    return str(candidate), exp_type

def suggested_option_legs(side: str, spot: float, now_ist: datetime) -> dict:
    """
    Bullish -> Sell ATM PE, Buy OTM PE
    Bearish -> Sell ATM CE, Buy OTM CE
    Includes proper Tuesday-based expiry + label Weekly/Monthly.
    """
    atm = round_to_step(spot, ATM_STEP_POINTS)
    if side == "BULL":
        legs = {
            "sell": f"{atm} PE (ATM)",
            "buy":  f"{max(ATM_STEP_POINTS, atm - OTM_GAP_POINTS)} PE (hedge ~{OTM_GAP_POINTS})"
        }
    else:
        legs = {
            "sell": f"{atm} CE (ATM)",
            "buy":  f"{atm + OTM_GAP_POINTS} CE (hedge ~{OTM_GAP_POINTS})"
        }
    expiry_str, exp_type = next_weekly_expiry_ist(now_ist)
    legs["expiry"] = expiry_str
    legs["expiry_type"] = exp_type
    return legs

# ---------- Data & Indicators ----------
def _force_float_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    return df

def _flatten_if_multiindex(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize to single-level columns: Open, High, Low, Close, Adj Close, Volume."""
    if not isinstance(df.columns, pd.MultiIndex):
        return df

    # try selecting by ticker on any level
    try:
        for lvl in range(df.columns.nlevels):
            if TICKER in df.columns.get_level_values(lvl):
                return df.xs(TICKER, axis=1, level=lvl, drop_level=True)
    except Exception:
        pass

    # fallback: select first key on last level
    try:
        key = df.columns.get_level_values(-1)[0]
        return df.xs(key, axis=1, level=-1, drop_level=True)
    except Exception:
        # ultimate flatten
        df.columns = ["|".join([str(x) for x in tup if x is not None]) for tup in df.columns.to_list()]
        rename_map = {}
        for col in df.columns:
            low = col.lower()
            if "open" in low and "Open" not in rename_map.values():   rename_map[col] = "Open"
            elif "high" in low and "High" not in rename_map.values(): rename_map[col] = "High"
            elif "low" in low and "Low" not in rename_map.values():   rename_map[col] = "Low"
            elif "adj" in low and "close" in low and "Adj Close" not in rename_map.values(): rename_map[col] = "Adj Close"
            elif "close" in low and "Close" not in rename_map.values(): rename_map[col] = "Close"
            elif "volume" in low and "Volume" not in rename_map.values(): rename_map[col] = "Volume"
        return df.rename(columns=rename_map)

def fetch_hourly() -> pd.DataFrame:
    # Prefer Ticker.history; fallback to download
    try:
        tk = yf.Ticker(TICKER)
        df = tk.history(period=f"{LOOKBACK_DAYS}d", interval=INTERVAL, auto_adjust=False)
    except Exception:
        df = None

    if df is None or df.empty:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=LOOKBACK_DAYS)
        df = yf.download(
            TICKER, start=start, end=end, interval=INTERVAL,
            auto_adjust=False, progress=False, group_by="column"
        )

    if df is None or df.empty:
        raise RuntimeError("No data returned from yfinance. Try later or change ticker.")

    # tz-awareness
    if df.index.tz is None:
        df.index = df.index.tz_localize(timezone.utc)
    df = df.tz_convert(IST)

    # normalize columns
    df = _flatten_if_multiindex(df)
    needed = ["Open", "High", "Low", "Close"]
    for n in needed:
        if n not in df.columns:
            raise RuntimeError(f"Missing column '{n}' after fetch; columns={list(df.columns)}")

    df = _force_float_cols(df, ["Open", "High", "Low", "Close", "Adj Close", "Volume"])
    return df

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["EMA20"] = ema(out["Close"], EMA_PERIOD).astype("float64")
    st = supertrend(out, period=ST_ATR_PERIOD, multiplier=ST_MULTIPLIER)
    out["ST"] = pd.to_numeric(st["ST"], errors="coerce").astype("float64")
    out["ST_DIR"] = pd.to_numeric(st["ST_DIR"], errors="coerce").astype("int64")
    return out

def latest_hourly_close_row_index(df: pd.DataFrame) -> int | None:
    """
    Accept a bar that looks like today's hourly close (~:15 past the hour).
    """
    idx_last = len(df) - 1
    ts_last: datetime = df.index[idx_last]
    now_ist = datetime.now(IST)

    if ts_last.date() != now_ist.date():
        log(f"Last bar {ts_last} not from today; skipping.")
        return None
    # hourly close window tolerance
    if not (10 <= int(ts_last.minute) <= 20):
        log(f"Last bar minute {ts_last.minute} not in 10..20; skipping (ts={ts_last}).")
        return None
    if (now_ist - ts_last) > timedelta(hours=6):
        log(f"Last bar {ts_last} stale (>6h); skipping.")
        return None
    return idx_last

def _get_scalar(x) -> float:
    """Convert to plain float; raise on NaN or non-scalar."""
    try:
        val = float(x)
        if math.isnan(val):
            raise ValueError("NaN")
        return val
    except Exception:
        raise ValueError("not a scalar")

# ---------- Signal ----------
def evaluate_signal(df: pd.DataFrame) -> dict:
    idx_last = latest_hourly_close_row_index(df)
    if idx_last is None:
        return {"status": "NO_ACTION"}

    try:
        c  = _get_scalar(df["Close"].iat[idx_last])
        st = _get_scalar(df["ST"].iat[idx_last])
        e  = _get_scalar(df["EMA20"].iat[idx_last])
    except ValueError:
        return {"status": "NO_ACTION"}

    pc = pst = pe = np.nan
    if idx_last >= 1:
        try:
            pc  = _get_scalar(df["Close"].iat[idx_last-1])
            pst = _get_scalar(df["ST"].iat[idx_last-1])
            pe  = _get_scalar(df["EMA20"].iat[idx_last-1])
        except ValueError:
            pc = pst = pe = np.nan

    # Current regime
    bull = (c > st) and (c > e)
    bear = (c < st) and (c < e)

    # Prior regime
    bull_prev = (not np.isnan(pc)) and (not np.isnan(pst)) and (not np.isnan(pe)) and (pc > pst) and (pc > pe)
    bear_prev = (not np.isnan(pc)) and (not np.isnan(pst)) and (not np.isnan(pe)) and (pc < pst) and (pc < pe)

    crossed_to_bull = bull and (not bull_prev)
    crossed_to_bear = bear and (not bear_prev)

    ts_last: datetime = df.index[idx_last]
    now_ist = datetime.now(IST)

    if crossed_to_bull:
        legs = suggested_option_legs("BULL", c, now_ist)
        return {
            "status": "SIGNAL",
            "side": "BULLISH",
            "timestamp": ts_last.isoformat(),
            "close": c, "ema20": e, "supertrend": st,
            "option_legs": legs
        }
    if crossed_to_bear:
        legs = suggested_option_legs("BEAR", c, now_ist)
        return {
            "status": "SIGNAL",
            "side": "BEARISH",
            "timestamp": ts_last.isoformat(),
            "close": c, "ema20": e, "supertrend": st,
            "option_legs": legs
        }

    return {
        "status": "NO_SIGNAL",
        "timestamp": ts_last.isoformat(),
        "close": c, "ema20": e, "supertrend": st
    }

# ---------- Alerts ----------
def send_telegram(msg: str) -> None:
    if not SEND_TELEGRAM:
        log("Telegram disabled (missing TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID).")
        return
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    try:
        r = requests.post(url, json={"chat_id": TG_CHAT_ID, "text": msg})
        if r.status_code != 200:
            log(f"Telegram error: {r.status_code} {r.text}")
        else:
            log("Telegram sent.")
    except Exception as e:
        log(f"Telegram send failed: {e}")

def format_signal_text(sig: dict) -> str:
    if sig.get("status") != "SIGNAL":
        return ""
    side = sig["side"]
    ts_ist = datetime.fromisoformat(sig["timestamp"]).astimezone(IST).strftime("%Y-%m-%d %H:%M IST")
    spot = sig["close"]; st = sig["supertrend"]; ema20 = sig["ema20"]; legs = sig["option_legs"]
    return (
        f"📈 NIFTY Hourly Signal ({ts_ist})\n"
        f"Side: {side}\n"
        f"Close: {spot:.2f} | ST: {st:.2f} | EMA20: {ema20:.2f}\n"
        f"Suggested ({legs['expiry_type']}):\n"
        f"Expiry: {legs['expiry']}\n"
        f" • SELL {legs['sell']}\n"
        f" • BUY  {legs['buy']}\n"
        f"Exit if Supertrend flips or by Tue (expiry) midday."
    )

# ---------- Main ----------
def main():
    try:
        log("Fetching data...")
        df = fetch_hourly()
        log(f"Fetched: shape={df.shape}, cols={list(df.columns)}")
        log(f"Last bar: {df.index[-1]}")
        df = compute_indicators(df)
        df = _force_float_cols(df, ["EMA20", "ST"])  # sanity

        sig = evaluate_signal(df)

        print(json.dumps(sig, indent=2))  # Always print JSON

        if sig.get("status") == "SIGNAL":
            msg = format_signal_text(sig)
            log(msg)
            send_telegram(msg)
        else:
            log(f"No new signal. Status: {sig['status']}")
    except Exception as e:
        log("ERROR: " + repr(e))
        traceback.print_exc()
        raise

if __name__ == "__main__":
    try:
        main()
    except Exception:
        sys.exit(1)
